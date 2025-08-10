# GPT-OSS 20B on Modal

This repository contains Modal-optimized code for running OpenAI's GPT-OSS 20B model.

## Features

- **21B total parameters** with 3.6B active parameters (MoE architecture)
- **Native MXFP4 quantization** support for MoE layers
- **Multiple quantization options** (8-bit, 4-bit) for memory efficiency
- **GPU-accelerated** inference on Modal's infrastructure
- **Persistent model caching** to avoid re-downloads

## Quick Start

### 1. Setup Modal

```bash
pip install modal
modal setup
```

### 2. Run Inference

Basic inference:
```bash
modal run gpt_oss_modal.py --action inference --prompt "Explain quantum computing"
```

With 8-bit quantization (uses less memory):
```bash
modal run gpt_oss_modal.py --action inference --prompt "Write a poem" --use-8bit
```

### 3. Analyze Model Structure

```bash
modal run gpt_oss_modal.py --action analyze
```

## Usage Examples

### Simple Python Wrapper

```bash
python run_gpt_oss.py "What is the meaning of life?" --tokens 200
```

### Batch Testing

```bash
python test_prompts.py
```

### Custom Integration

```python
import modal

app = modal.App.lookup("gpt-oss-moe")
inference_fn = modal.Function.lookup("gpt-oss-moe", "run_gpt_oss_inference")

result = inference_fn.remote(
    prompt="Explain AI safety",
    max_tokens=150,
    quantize=True  # Use 8-bit quantization
)
print(result)
```

## Model Specifications

- **Model ID**: `openai/gpt-oss-20b`
- **Architecture**: Mixture of Experts (MoE)
- **Total Parameters**: 21B
- **Active Parameters**: 3.6B per forward pass
- **License**: Apache 2.0
- **GPU Requirements**: 
  - Full precision: H100 or A100-40GB recommended
  - 8-bit quantization: A10G or better
  - 4-bit quantization: T4 possible

## Performance Notes

1. **First Run**: Initial run will download the model (~40GB), which is cached for future use
2. **Inference Speed**: Varies based on GPU and quantization:
   - H100: ~50-100 tokens/sec
   - A100: ~30-60 tokens/sec
   - A10G (8-bit): ~20-40 tokens/sec
3. **Memory Usage**:
   - Full precision (bf16): ~40GB VRAM
   - 8-bit quantization: ~20GB VRAM
   - 4-bit quantization: ~10GB VRAM

## Advanced Features

### Expert Pruning (Experimental)

Remove specific experts from the MoE layers:

```bash
# Analyze first to understand structure
modal run gpt_oss_modal.py --action analyze

# Prune expert 2
modal run gpt_oss_modal.py --action prune --expert 2

# Prune and push to HuggingFace Hub
HF_TOKEN=hf_xxx modal run gpt_oss_modal.py \
  --action prune \
  --expert 2 \
  --output-repo yourusername/gpt-oss-pruned
```

### Configurable Reasoning Levels

GPT-OSS supports different reasoning levels (low, medium, high) which can be configured in prompts for different use cases.

## Troubleshooting

### MXFP4 Warning
If you see "MXFP4 quantization requires triton >= 3.4.0", the model will automatically fall back to bf16 precision. This is normal and doesn't affect functionality.

### Out of Memory
If you encounter OOM errors:
1. Use `--use-8bit` flag for 8-bit quantization
2. Use `--use-4bit` flag for 4-bit quantization (experimental)
3. Reduce `--max-tokens` parameter
4. Upgrade to a larger GPU (H100 recommended)

### Slow First Run
The first run downloads the model which takes time. Subsequent runs use the cached model and are much faster.

## Cost Optimization

- Use 8-bit quantization for most tasks (minimal quality loss)
- Set appropriate `max_tokens` limits
- Use A10G GPUs for cost-effective inference
- Leverage Modal's autoscaling for batch processing

## License

The GPT-OSS model is licensed under Apache 2.0. See the [model card](https://huggingface.co/openai/gpt-oss-20b) for details.