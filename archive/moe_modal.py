#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Tuple

import modal
from modal import App, Image, method, enter

# Define the Modal app
app = App("moe-pruning")

# Create a custom image with required dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",  # Some models need this
        "protobuf",       # Some models need this
    )
)

# Create a volume for model caching
volume = modal.Volume.from_name("moe-models", create_if_missing=True)

# ---------
# Utilities (same as original)
# ---------

def is_router_tensor(name: str) -> bool:
    """
    Heuristics for router/gate tensors in MoE layers.
    Adjust if your model uses different names.
    """
    return (
        ".router" in name
        or name.endswith(".gate.weight")
        or name.endswith(".gate.bias")
        or ".block_sparse_moe.router" in name
    )

def contains_expert_k(name: str, k: int) -> bool:
    return f".experts.{k}." in name

def slice_out_expert_dim(t, k: int, num_experts: int):
    """
    Remove expert k from a router linear tensor.
    """
    import torch
    
    if t.ndim == 2:
        if t.shape[0] == num_experts:
            return torch.cat([t[:k], t[k+1:]], dim=0)
        if t.shape[1] == num_experts:
            return torch.cat([t[:, :k], t[:, k+1:]], dim=1)
        return t
    if t.ndim == 1 and t.shape[0] == num_experts:
        return torch.cat([t[:k], t[k+1:]], dim=0)
    return t

def find_num_experts_from_config(cfg) -> int:
    if hasattr(cfg, "num_local_experts"):
        return int(cfg.num_local_experts)
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

# ------------------------------
# Modal Class for MoE Operations
# ------------------------------

@app.cls(
    image=image,
    gpu="A10G",  # or "T4" for smaller models, "A100" for larger
    volumes={"/cache": volume},
    timeout=3600,  # 1 hour timeout
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
class MoEPruner:
    
    @enter()
    def setup(self):
        """Initialize environment and set cache directories."""
        # Set HuggingFace cache to our persistent volume
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
        
        # Create cache directories if they don't exist
        Path("/cache/huggingface").mkdir(parents=True, exist_ok=True)
    
    @method()
    def soft_mask_inference(
        self,
        model_id: str,
        expert_idx: int,
        prompt: str = "Write a haiku about mixture-of-experts models.",
        max_new_tokens: int = 40,
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> str:
        """
        Run inference with a soft-masked expert (no weights changed).
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        
        print(f"Loading {model_id} with soft masking of expert {expert_idx}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            use_fast=True, 
            trust_remote_code=trust_remote_code
        )
        
        # Get number of experts
        cfg = model.config
        num_experts = find_num_experts_from_config(cfg)
        
        # Attach soft mask hooks
        self._attach_soft_mask_hooks(model, expert_idx, num_experts)
        
        # Run inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Generated text: {result}\n\nExpert {expert_idx} was masked during routing."
    
    def _attach_soft_mask_hooks(self, model, expert_idx: int, num_experts: int):
        """Attach hooks to mask expert during inference."""
        import torch
        
        def mask_router_logits(mod, inputs):
            hidden = inputs[0]
            logits = torch.nn.functional.linear(hidden, mod.weight, mod.bias)
            if logits.size(-1) != num_experts:
                return (logits,)
            logits[..., expert_idx] = float("-inf")
            return (logits,)
        
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == num_experts:
                if "router" in name or "gate" in name:
                    mod.register_forward_pre_hook(mask_router_logits)
    
    @method()
    def hard_prune_expert(
        self,
        model_id: str,
        expert_idx: int,
        output_repo: str = None,  # HuggingFace repo to push to
        dtype: str = "float16",
        trust_remote_code: bool = False,
        private: bool = False,
    ) -> dict:
        """
        Load model, remove expert weights & shrink routers, update config,
        and optionally push to HuggingFace Hub.
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfApi, create_repo
        
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        
        print(f"Loading config for {model_id}...")
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        num_experts = find_num_experts_from_config(cfg)
        
        if not (0 <= expert_idx < num_experts):
            raise ValueError(f"expert_idx {expert_idx} out of range [0, {num_experts-1}]")
        
        print(f"Loading model {model_id} for pruning...")
        # Load on CPU to avoid huge VRAM usage during surgery
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=False,
            trust_remote_code=trust_remote_code,
        )
        
        print(f"Pruning expert {expert_idx}...")
        sd = model.state_dict()
        new_sd = {}
        
        for k, v in sd.items():
            # 1) Drop expert tensors
            if contains_expert_k(k, expert_idx):
                continue
            
            # 2) Shrink router tensors
            if is_router_tensor(k):
                new_sd[k] = slice_out_expert_dim(v, expert_idx, num_experts)
                continue
            
            # 3) Pass-through
            new_sd[k] = v
        
        # Update config
        set_num_experts_in_config(cfg, num_experts - 1)
        
        # Rebuild model with new config
        print("Building pruned model...")
        pruned_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
        missing, unexpected = pruned_model.load_state_dict(new_sd, strict=False)
        
        # Sanity checks
        unexpected_real = [m for m in unexpected if "lm_head.weight" not in m]
        if unexpected_real:
            print(f"Warning: unexpected keys: {unexpected_real}")
        if missing:
            print(f"Note: missing keys (often safe): {missing}")
        
        # Save locally first
        local_dir = "/tmp/pruned_model"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        pruned_model.save_pretrained(local_dir, safe_serialization=True)
        
        # Copy tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                use_fast=True, 
                trust_remote_code=trust_remote_code
            )
            tokenizer.save_pretrained(local_dir)
        except Exception as e:
            print(f"Tokenizer copy failed: {e}")
        
        result = {
            "status": "success",
            "original_experts": num_experts,
            "new_experts": num_experts - 1,
            "removed_expert": expert_idx,
        }
        
        # Push to Hub if requested
        if output_repo:
            print(f"Pushing to HuggingFace Hub: {output_repo}")
            try:
                # Get HF token from environment or use default
                token = os.environ.get("HF_TOKEN", True)
                
                # Create repo if it doesn't exist
                api = HfApi(token=token)
                try:
                    create_repo(output_repo, private=private, token=token)
                except Exception:
                    pass  # Repo might already exist
                
                # Push model and tokenizer
                pruned_model.push_to_hub(output_repo, private=private, token=token)
                tokenizer.push_to_hub(output_repo, private=private, token=token)
                
                result["hub_repo"] = output_repo
                print(f"Successfully pushed to {output_repo}")
            except Exception as e:
                result["hub_error"] = str(e)
                print(f"Failed to push to hub: {e}")
        
        return result

# ------------------------------
# CLI Functions for Modal
# ------------------------------

@app.local_entrypoint()
def main(
    model_id: str,
    expert: int,
    mode: str = "soft",
    output_repo: str = None,
    dtype: str = "float16",
    trust_remote_code: bool = False,
    private: bool = False,
    prompt: str = None,
):
    """
    Modal CLI entrypoint for MoE expert pruning.
    
    Examples:
        # Soft masking (inference only):
        modal run moe_modal.py --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 --expert 3
        
        # Hard pruning with Hub upload:
        modal run moe_modal.py --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 --expert 3 --mode hard --output-repo myusername/mixtral-pruned
        
        # With HF token:
        HF_TOKEN=hf_xxx modal run moe_modal.py --model-id ... --mode hard --output-repo ...
    """
    pruner = MoEPruner()
    
    if mode == "soft":
        # Run soft masking inference
        if prompt is None:
            prompt = "Write a haiku about mixture-of-experts models."
        
        result = pruner.soft_mask_inference.remote(
            model_id=model_id,
            expert_idx=expert,
            prompt=prompt,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        print(result)
    
    elif mode == "hard":
        # Run hard pruning
        result = pruner.hard_prune_expert.remote(
            model_id=model_id,
            expert_idx=expert,
            output_repo=output_repo,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            private=private,
        )
        
        print(f"\n=== Pruning Results ===")
        print(f"Status: {result['status']}")
        print(f"Experts: {result['original_experts']} -> {result['new_experts']}")
        print(f"Removed expert: {result['removed_expert']}")
        
        if "hub_repo" in result:
            print(f"Pushed to: {result['hub_repo']}")
        elif "hub_error" in result:
            print(f"Hub push failed: {result['hub_error']}")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

# Optional: Function for programmatic use
@app.function(image=image, gpu="A10G", volumes={"/cache": volume})
def prune_expert_function(
    model_id: str,
    expert_idx: int,
    output_repo: str = None,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict:
    """
    Standalone function for pruning an expert.
    Can be called from other Modal apps or scripts.
    """
    pruner = MoEPruner()
    return pruner.hard_prune_expert(
        model_id=model_id,
        expert_idx=expert_idx,
        output_repo=output_repo,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )