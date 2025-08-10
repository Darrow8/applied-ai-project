# Run MMLU on Qwen/Qwen3-4B-Instruct-2507 using Modal.com
#
# Usage (from your terminal):
#   pip install modal
#   modal token new
#   # create your secret once (HF token from https://huggingface.co/settings/tokens)
#   # modal secret create huggingface-token HF_TOKEN=hf_xxx
#   python benchmark.py
#
# Optional flags:
#   python benchmark.py --model Qwen/Qwen3-4B-Instruct-2507 --k-shot 5 --limit -1 --gpu A100
#
# Notes:
# - This script expects a Modal secret named "huggingface-token" that exposes env var HF_TOKEN.
# - By default it requests an A100 GPU. You can try "H100" if your org has access.
# - Torch GPU wheels are installed via NVIDIA index URL in the image build step.

import argparse
import modal

app = modal.App("qwen-mmlu-benchmark")

# ---------- Image / Environment ----------
# Build a container image with all deps. We install CUDA-enabled torch wheels.
image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .run_commands(
        # CUDA-enabled torch (adjust CUDA version if needed)
        "pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
        # Core Python deps
        "pip install --upgrade transformers>=4.42.0 datasets accelerate tqdm huggingface_hub",
    )
)

# Hugging Face token secret (must be created in your Modal account)
HF_SECRET = modal.Secret.from_name("huggingface-token")


@app.function(
    image=image,
    secrets=[HF_SECRET],
    gpu="A100",            # or "H100" if available
    timeout=60 * 60,       # 1 hour
)
def run_mmlu_remote(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    k_shot: int = 3,
    max_new_tokens: int = 2,
    temperature: float = 0.0,
    limit_per_subject: int = -1,  # -1 = full test split
    seed: int = 123,
):
    import os
    import re
    import random
    from collections import defaultdict

    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    # ---------- Auth ----------
    # Use your Modal secret "huggingface-token" -> env var HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found. Create a Modal secret named 'huggingface-token' with HF_TOKEN=...")

    login(token=hf_token)

    # ---------- Repro ----------
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- Helpers ----------
    CHOICES = ["A", "B", "C", "D"]
    SYSTEM_INSTRUCT = (
        "You are a careful expert test-taker. "
        "Answer with only a single capital letter (A, B, C, or D). No explanation."
    )

    def choice_letter(index: int) -> str:
        return CHOICES[int(index)]

    def format_example(example, include_answer: bool) -> str:
        q = example["question"].strip()
        A, B, C, D = [c.strip() for c in example["choices"]]
        if include_answer:
            ans = choice_letter(example["answer"])
            return f"Question: {q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {ans}"
        return f"Question: {q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"

    def build_prompt(subject, dev_examples, q_example, k):
        parts = [f"Subject: {subject}"]
        for ex in dev_examples[:k]:
            parts.append(format_example(ex, include_answer=True))
        parts.append(format_example(q_example, include_answer=False))
        return "\n\n".join(parts)

    def extract_choice(text: str) -> str:
        # First standalone A/B/C/D wins
        m = re.search(r"\b([ABCD])\b", text.strip())
        if m:
            return m.group(1)
        m = re.search(r"Answer\s*:\s*([ABCD])", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return ""

    # ---------- Load model ----------
    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer bf16 on NVIDIA; fallback to fp32 otherwise
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # ---------- Load MMLU ----------
    print("Loading MMLU (cais/mmlu)…")
    ds = load_dataset("cais/mmlu", "all")
    subjects = sorted(ds["test"].unique("subject"))

    per_subject_correct = defaultdict(int)
    per_subject_total = defaultdict(int)

    # ---------- Evaluate ----------
    for subject in subjects:
        dev_split = ds["dev"].filter(lambda x: x["subject"] == subject)
        test_split = ds["test"].filter(lambda x: x["subject"] == subject)

        k = min(k_shot, len(dev_split))
        n_total = len(test_split)
        n_eval = n_total if limit_per_subject < 0 else min(limit_per_subject, n_total)

        print(f"\nSubject: {subject} | k-shot={k} | evaluating {n_eval}/{n_total}")

        dev_list = [dev_split[i] for i in range(len(dev_split))]

        for i in tqdm(range(n_eval), leave=False):
            q_ex = test_split[i]
            prompt = build_prompt(subject, dev_list, q_ex, k)

            # Use chat template (Instruct model)
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_choice(gen_text)
            gold = choice_letter(q_ex["answer"])

            per_subject_total[subject] += 1
            if pred == gold:
                per_subject_correct[subject] += 1

        corr = per_subject_correct[subject]
        tot = per_subject_total[subject]
        acc = 100.0 * corr / tot if tot else 0.0
        print(f"Subject accuracy — {subject}: {corr}/{tot} = {acc:.2f}%")

    # ---------- Summary ----------
    total_corr = sum(per_subject_correct.values())
    total_cnt = sum(per_subject_total.values())
    micro = 100.0 * total_corr / total_cnt if total_cnt else 0.0
    macro = (
        sum((per_subject_correct[s] / per_subject_total[s]) for s in subjects if per_subject_total[s] > 0)
        / sum(1 for s in subjects if per_subject_total[s] > 0)
        * 100.0
    )

    print("\n================= RESULTS =================")
    for s in subjects:
        tot = per_subject_total[s]
        if tot > 0:
            acc = 100.0 * per_subject_correct[s] / tot
            print(f"{s:35s}: {acc:6.2f}%  ({per_subject_correct[s]}/{tot})")
    print("------------------------------------------")
    print(f"Micro Avg (overall): {micro:.2f}%  ({total_corr}/{total_cnt})")
    print(f"Macro Avg (mean over subjects): {macro:.2f}%")
    print("==========================================")


# ---------- Local entrypoint ----------
@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    k_shot: int = 5,
    limit: int = -1,              # -1 = full test split
    max_new_tokens: int = 2,
    temperature: float = 0.0,
    gpu: str = "A100"             # "A100" or "H100"
):
    """
    You can pass args when running the script, e.g.:
      python benchmark.py --model Qwen/Qwen3-4B-Instruct-2507 --k-shot 5 --limit -1 --gpu A100
    """
    # Parse flags given to the Python script
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("--k-shot", type=int, default=k_shot)
    parser.add_argument("--limit", type=int, default=limit)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--gpu", type=str, default=gpu)
    args, _ = parser.parse_known_args()

    # Modal resources (e.g., GPU) are configured on the function decorator.
    # Per-call overrides like `.options(gpu=...)` are not supported.
    if args.gpu and args.gpu.upper() != "A100":
        print(f"Warning: requested GPU '{args.gpu}' is ignored. Edit the @app.function(gpu=...) decorator to change it.")

    run_mmlu_remote.remote(
        model_name=args.model,
        k_shot=args.k_shot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit_per_subject=args.limit,
    )
