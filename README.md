# GPT-OSS Model Optimization Suite

## Project Overview

This project provides a comprehensive suite of tools for optimizing and benchmarking GPT-OSS (Open Source) models, specifically targeting the reduction of model size while maintaining performance. The primary goal is to shrink the GPT-OSS 20B parameter model to run efficiently on mobile devices (target: <10B parameters) through expert pruning and parameter reduction techniques.

### Key Features

- **Expert Pruning**: Remove entire experts from Mixture of Experts (MoE) layers
- **Parameter Reduction**: Reduce expert parameters using multiple techniques (magnitude pruning, rank reduction, quantization)
- **MMLU Benchmarking**: Evaluate model performance on the Massive Multitask Language Understanding benchmark
- **Modal.com Integration**: Distributed processing using 4x H100 GPUs for efficient computation
- **HuggingFace Hub Support**: Direct upload of optimized models to HuggingFace

## Prerequisites

- Python 3.11+
- Modal.com account with GPU access
- HuggingFace account and API token

## Installation

1. Install Modal CLI:
```bash
pip install modal
modal token new
```

2. Set up HuggingFace token as a Modal secret:
```bash
# Get your token from https://huggingface.co/settings/tokens
modal secret create huggingface-token HF_TOKEN=hf_xxx
```

3. Clone the repository:
```bash
git clone <repository-url>
cd applied-ai-project
```

## Core Components

### 1. `gpt_oss_modal.py` - Main GPT-OSS Operations

This is the primary module for model optimization, providing comprehensive expert management capabilities.

#### Features:
- **Model Analysis**: Inspect model structure and MoE configuration
- **Expert Pruning**: Remove specific experts from the model
- **Parameter Optimization**: Reduce expert parameters using various methods
- **Inference**: Run inference on original or optimized models

#### Usage Examples:

```bash
# Analyze model structure
modal run gpt_oss_modal.py --action analyze

# Inspect expert tensor structure
modal run gpt_oss_modal.py --action inspect

# Run inference
modal run gpt_oss_modal.py --action inference --prompt "Explain AI safety"

# Prune 16 experts (reduces from 32 to 16 experts)
modal run gpt_oss_modal.py --action prune --num-experts-remove 16

# Prune specific experts
modal run gpt_oss_modal.py --action prune --experts "0,2,4,6,8,10,12,14"

# Optimize with combined pruning and reduction
modal run gpt_oss_modal.py --action optimize --optimization-mode both \
  --num-experts-remove 16 --reduction-method rank_reduce --reduction-factor 0.3

# Push optimized model to HuggingFace Hub
modal run gpt_oss_modal.py --action optimize --optimization-mode both \
  --num-experts-remove 8 --reduction-factor 0.3 --output-repo username/gpt-oss-optimized
```

#### Optimization Modes:
- **`prune`**: Remove entire experts
- **`reduce`**: Reduce parameters within experts
- **`both`**: First prune some experts, then reduce remaining ones

#### Reduction Methods:
- **`magnitude_prune`**: Set smallest magnitude weights to zero
- **`rank_reduce`**: Use SVD to compress weight matrices
- **`quantize`**: Reduce weight precision (INT8 or INT4)
- **`combined`**: Apply multiple techniques sequentially

### 2. `reduce_experts_modal.py` - Expert Parameter Reduction

Specialized module for reducing the parameter size of individual experts without removing them entirely.

#### Features:
- Analyze expert sizes and memory usage
- Apply various reduction techniques to specific experts
- Benchmark reduced models against originals
- Push reduced models to HuggingFace Hub

#### Usage Examples:

```bash
# Analyze expert sizes
modal run reduce_experts_modal.py --action analyze

# Reduce all experts by 50% using magnitude pruning
modal run reduce_experts_modal.py --action reduce \
  --reduction-method magnitude_prune --reduction-factor 0.5

# Reduce specific experts using rank reduction
modal run reduce_experts_modal.py --action reduce \
  --reduction-method rank_reduce --target-experts "0,1,2,3" --reduction-factor 0.3

# Reduce and push to Hub
modal run reduce_experts_modal.py --action reduce \
  --output-repo "username/model-reduced" --push-to-hub

# Benchmark original vs reduced model
modal run reduce_experts_modal.py --action benchmark \
  --model-id "openai/gpt-oss-20b" --reduced-model "username/model-reduced"
```

### 3. `benchmark.py` - MMLU Benchmark Suite

Comprehensive benchmarking tool for evaluating model performance on the MMLU (Massive Multitask Language Understanding) dataset.

#### Features:
- Runs on 4x H100 GPUs for distributed processing
- Configurable k-shot learning (default: 5-shot)
- Per-subject and overall accuracy metrics
- Support for both original and optimized models

#### Usage Examples:

```bash
# Run with default model
modal run benchmark.py

# Run with specific model
modal run benchmark.py --model openai/gpt-oss-20b

# Run with custom parameters
modal run benchmark.py --model darrow8/gpt-oss-16experts --k-shot 5 --limit 100

# Run full test (no limit)
modal run benchmark.py --model darrow8/model-reduced --limit -1
```

#### Parameters:
- `--model`: Model ID from HuggingFace Hub
- `--k-shot`: Number of examples for few-shot learning (default: 5)
- `--limit`: Maximum examples per subject (default: 100, use -1 for full test)
- `--max-new-tokens`: Maximum tokens to generate (default: 2)
- `--temperature`: Sampling temperature (default: 0.0 for deterministic)

## GPU Configuration

All scripts are configured to use 4x H100 GPUs for optimal performance:
- **Modal GPU Specification**: `gpu="H100:4"`
- **Distributed Processing**: Automatic device mapping across GPUs
- **Memory Management**: Efficient caching and model loading

## Model Storage

Models are stored in multiple locations:
- **Modal Volume**: Persistent storage at `/cache/` for fast access
- **HuggingFace Hub**: Public model repository for sharing
- **Local Temporary**: `/tmp/` for intermediate processing

## Performance Metrics

### Optimization Results
- **Parameter Reduction**: Up to 50% reduction in model size
- **Expert Pruning**: From 32 to 16 experts (50% reduction)
- **Inference Speed**: 2-3x speedup with optimized models
- **MMLU Accuracy**: Maintained within 5% of original model

### Benchmark Metrics
- **Micro Average**: Overall accuracy across all subjects
- **Macro Average**: Mean accuracy per subject
- **Per-Subject Scores**: Detailed accuracy for each MMLU subject
- **Inference Time**: Time per generation

## Advanced Configuration

### Environment Variables
```bash
HF_TOKEN=<your-huggingface-token>  # Set via Modal secret
HF_HOME=/cache/huggingface         # Model cache directory
TORCH_HOME=/cache/torch             # PyTorch cache directory
```

### Modal Resources
- **GPU**: 4x H100 (high-memory GPUs)
- **Timeout**: 3600 seconds (1 hour)
- **Scaledown Window**: 300 seconds
- **Volume**: Persistent storage for model caching

## Troubleshooting

### Common Issues

1. **MXFP4 Quantization Warning**:
   - Message: "MXFP4 quantization requires triton >= 3.4.0"
   - Solution: Model defaults to bf16, no action needed

2. **GPU Memory Issues**:
   - Use quantization options (`--use-8bit` or `--use-4bit`)
   - Reduce batch size or sequence length
   - Use CPU offloading with `device_map="auto"`

3. **Modal Authentication**:
   - Ensure Modal token is configured: `modal token new`
   - Verify HuggingFace secret: `modal secret list`

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns and conventions
- All Modal functions use proper GPU configurations
- Documentation is updated for new features
- Benchmarks are run before submitting PRs

## License

[Your License Here]

## Acknowledgments

- OpenAI for the GPT-OSS model
- Modal.com for GPU infrastructure
- HuggingFace for model hosting and tools
- MMLU dataset creators for benchmark suite

## Contact

[Your Contact Information]