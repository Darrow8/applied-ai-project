#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ---------
# Utilities
# ---------

def is_router_tensor(name: str) -> bool:
    """
    Heuristics for router/gate tensors in MoE layers.
    Adjust if your model uses different names.
    """
    # Common patterns: ".router", ".gate", ".block_sparse_moe.router"
    return (
        ".router" in name
        or name.endswith(".gate.weight")
        or name.endswith(".gate.bias")
        or ".block_sparse_moe.router" in name
    )

def contains_expert_k(name: str, k: int) -> bool:
    # Mixtral: ...mlp.experts.{k}.(w1|w2|w3|up|down|gate).(weight|bias)
    return f".experts.{k}." in name

def slice_out_expert_dim(t: torch.Tensor, k: int, num_experts: int) -> torch.Tensor:
    """
    Remove expert k from a router linear tensor, which could be shaped:
    - [num_experts, hidden]  (weight with out_features first)
    - [hidden, num_experts]  (weight with out_features last)
    - [num_experts]          (bias)
    """
    if t.ndim == 2:
        if t.shape[0] == num_experts:
            return torch.cat([t[:k], t[k+1:]], dim=0)
        if t.shape[1] == num_experts:
            return torch.cat([t[:, :k], t[:, k+1:]], dim=1)
        # Not a router weight we want to slice.
        return t
    if t.ndim == 1 and t.shape[0] == num_experts:
        return torch.cat([t[:k], t[k+1:]], dim=0)
    return t

def find_num_experts_from_config(cfg) -> int:
    # Mixtral exposes `num_local_experts`
    if hasattr(cfg, "num_local_experts"):
        return int(cfg.num_local_experts)
    # Fallback: a few other MoE configs use `num_experts`
    if hasattr(cfg, "num_experts"):
        return int(cfg.num_experts)
    raise ValueError("Could not find num_experts/num_local_experts in config.")

def set_num_experts_in_config(cfg, new_val: int):
    if hasattr(cfg, "num_local_experts"):
        cfg.num_local_experts = int(new_val)
    elif hasattr(cfg, "num_experts"):
        cfg.num_experts = int(new_val)
    else:
        raise ValueError("Could not set num_experts on config.")

# ---------------------------
# A) Soft masking (no saving)
# ---------------------------

def attach_soft_mask_hooks(model: torch.nn.Module, expert_idx: int, num_experts: int):
    """
    For each router Linear(hidden -> num_experts), mask expert_idx to -inf
    before top-k routing. This does not change weights on disk.
    """
    def mask_router_logits(mod, inputs):
        hidden = inputs[0]
        logits = torch.nn.functional.linear(hidden, mod.weight, mod.bias)
        if logits.size(-1) != num_experts:
            return (logits,)  # skip unexpected shapes
        logits[..., expert_idx] = float("-inf")
        return (logits,)

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == num_experts:
            # Heuristic to limit hooks to router/gate projections
            if "router" in name or "gate" in name:
                mod.register_forward_pre_hook(mask_router_logits)

# ------------------------------
# B) Hard prune (save & publish)
# ------------------------------

def hard_prune_expert(
    model_id: str,
    expert_idx: int,
    output_dir: str,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> Tuple[str, int, int]:
    """
    Load model from HF, remove expert weights & shrink routers, update config,
    and save to output_dir as a new HF-format checkpoint.
    Returns (output_dir, old_num_experts, new_num_experts).
    """
    torch_dtype = dict(
        float16=torch.float16,
        bfloat16=torch.bfloat16,
        float32=torch.float32,
    )[dtype]

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    num_experts = find_num_experts_from_config(cfg)

    if not (0 <= expert_idx < num_experts):
        raise ValueError(f"expert_idx {expert_idx} out of range [0, {num_experts-1}]")

    # Load on CPU to avoid huge VRAM usage during surgery
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=False,   # we want a full state_dict()
        trust_remote_code=trust_remote_code,
    )

    sd = model.state_dict()
    new_sd = {}

    for k, v in sd.items():
        # 1) Drop expert tensors (the whole expert block)
        if contains_expert_k(k, expert_idx):
            continue

        # 2) Shrink router tensors by slicing out the expert dimension
        if is_router_tensor(k):
            new_sd[k] = slice_out_expert_dim(v, expert_idx, num_experts)
            continue

        # 3) Pass-through
        new_sd[k] = v

    # 4) Update config: num_experts -> num_experts - 1
    set_num_experts_in_config(cfg, num_experts - 1)

    # Rebuild a fresh model with new config so shapes line up,
    # then load the edited state_dict.
    pruned_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    missing, unexpected = pruned_model.load_state_dict(new_sd, strict=False)

    # Sanity checks (there may be some buffers listed as missing on certain arches)
    unexpected_real = [m for m in unexpected if "lm_head.weight" not in m]
    if unexpected_real:
        print("Warning: unexpected keys after pruning:\n", unexpected_real)
    if missing:
        # Many MoE impls register aux-loss scalars/buffers; warn but continue.
        print("Note: missing keys after pruning (often safe for buffers):\n", missing)

    # Save pruned model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pruned_model.save_pretrained(output_dir, safe_serialization=True)
    # Copy tokenizer & special tokens
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
        tok.save_pretrained(output_dir)
    except Exception as e:
        print("Tokenizer copy failed (continuing):", e)

    return output_dir, num_experts, num_experts - 1

# -----------
# CLI / Main
# -----------

def main():
    ap = argparse.ArgumentParser(description="Mask or prune an expert from a Hugging Face MoE model.")
    ap.add_argument("--model_id", required=True, help="e.g. mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--expert", type=int, required=True, help="0-based index of expert to remove/disable")
    ap.add_argument("--mode", choices=["soft", "hard"], default="soft",
                    help="'soft' masks at inference; 'hard' edits weights & saves a new model")
    ap.add_argument("--out_dir", default="pruned_model",
                    help="Where to save the pruned model (only used for --mode hard)")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--push_to_hub", action="store_true",
                    help="After hard prune, push to your HF repo specified by --repo_id")
    ap.add_argument("--repo_id", default=None, help="Your target repo (e.g., username/model-pruned)")
    ap.add_argument("--private", action="store_true", help="Create private repo when pushing")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass through to Transformers loaders")

    args = ap.parse_args()

    if args.mode == "soft":
        # Demo harness: run a quick generation with the masked expert
        print(f"[soft] Loading {args.model_id} and masking expert {args.expert}…")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if args.dtype == "float16" else getattr(torch, args.dtype),
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)

        cfg = model.config
        num_experts = find_num_experts_from_config(cfg)
        attach_soft_mask_hooks(model, args.expert, num_experts)

        prompt = "Write a haiku about mixture-of-experts models."
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40)
        print(tok.decode(out[0], skip_special_tokens=True))
        print(f"\n[soft] Expert {args.expert} was masked during routing (no weights changed).")

    else:
        print(f"[hard] Pruning expert {args.expert} from {args.model_id} …")
        out_dir, old_n, new_n = hard_prune_expert(
            model_id=args.model_id,
            expert_idx=args.expert,
            output_dir=args.out_dir,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"[hard] Saved to {out_dir}. Experts: {old_n} -> {new_n}")

        if args.push_to_hub:
            if not args.repo_id:
                raise SystemExit("--push_to_hub requires --repo_id")
            # Re-load via save_pretrained dir and push
            print(f"[hard] Pushing to Hub: {args.repo_id}")
            pruned_model = AutoModelForCausalLM.from_pretrained(out_dir, device_map="cpu")
            pruned_model.push_to_hub(args.repo_id, private=args.private)
            try:
                tok = AutoTokenizer.from_pretrained(out_dir, use_fast=True)
                tok.push_to_hub(args.repo_id, private=args.private)
            except Exception as e:
                print("Tokenizer push skipped/failed:", e)
            print("[hard] Push complete. Make sure your HF token is set via HUGGING_FACE_HUB_TOKEN or HF_HOME login.")

if __name__ == "__main__":
    main()
