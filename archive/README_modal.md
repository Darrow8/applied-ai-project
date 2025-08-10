# MoE Expert Pruning on Modal

This Modal app provides GPU-accelerated pruning of experts from Mixture-of-Experts (MoE) models like Mixtral.

## Setup

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal setup
```

3. (Optional) Set your HuggingFace token for pushing to Hub:
```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

### Soft Masking (Inference Only)

Test inference with an expert masked (no permanent changes):

```bash
modal run moe_modal.py \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --expert 3 \
  --prompt "Explain quantum computing"
```

### Hard Pruning (Permanent Removal)

Remove an expert and save the pruned model:

```bash
# Prune locally (results saved in Modal's temporary storage)
modal run moe_modal.py \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --expert 3 \
  --mode hard

# Prune and push to HuggingFace Hub
modal run moe_modal.py \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --expert 3 \
  --mode hard \
  --output-repo yourusername/mixtral-7-experts \
  --private  # Optional: make repo private
```

### Programmatic Usage

You can also use this as a Modal function in other Python scripts:

```python
import modal

app = modal.App.lookup("moe-pruning")
prune_fn = modal.Function.lookup("moe-pruning", "prune_expert_function")

result = prune_fn.remote(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    expert_idx=3,
    output_repo="myusername/pruned-model"
)
print(result)
```

## Parameters

- `--model-id`: HuggingFace model ID (required)
- `--expert`: 0-based index of expert to remove (required)
- `--mode`: "soft" for inference masking or "hard" for permanent pruning (default: "soft")
- `--output-repo`: HuggingFace repo to push pruned model to (for hard mode)
- `--dtype`: Model precision - "float16", "bfloat16", or "float32" (default: "float16")
- `--trust-remote-code`: Allow execution of remote code from model repo
- `--private`: Create private repo when pushing to Hub
- `--prompt`: Custom prompt for soft masking inference

## GPU Configuration

The app uses:
- A10G GPU by default (good balance of cost/performance)
- Persistent volume for model caching
- 1-hour timeout for large models
- 5-minute warm container timeout

You can modify the GPU type in the `@app.cls` decorator:
- `"T4"`: Cheaper, good for smaller models
- `"A10G"`: Default, good for most models
- `"A100"`: More expensive, for very large models

## Model Caching

Models are cached in a persistent Modal volume (`moe-models`) to avoid re-downloading on subsequent runs.

## Supported Models

Tested with:
- Mixtral-8x7B models
- Any HuggingFace MoE model with standard expert naming conventions

The code looks for expert tensors with patterns like:
- `.experts.{k}.`
- Router/gate tensors with `.router` or `.gate` in the name