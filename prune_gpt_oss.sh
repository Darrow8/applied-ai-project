#!/bin/bash

# Script to prune GPT-OSS from 32 to 16 experts and push to HuggingFace
# Usage: ./prune_gpt_oss.sh <your-hf-username>

set -e  # Exit on error

# Check if HF username is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <your-hf-username>"
    echo "Example: $0 john-doe"
    exit 1
fi

HF_USERNAME=$1
REPO_NAME="gpt-oss-16experts"
FULL_REPO="${HF_USERNAME}/${REPO_NAME}"

echo "=========================================="
echo "GPT-OSS Expert Pruning Script"
echo "=========================================="
echo "Target repo: ${FULL_REPO}"
echo ""

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo "Please run: export HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "✅ HuggingFace token detected"
echo ""

# Step 1: Analyze original model
echo "Step 1: Analyzing original model structure..."
echo "------------------------------------------"
modal run gpt_oss_modal.py --action analyze

echo ""
echo "Step 2: Pruning 16 experts (keeping 16 out of 32)..."
echo "-----------------------------------------------------"
echo "This will:"
echo "  - Remove 16 evenly-spaced experts"
echo "  - Reduce model from 32 to 16 experts"
echo "  - Save to HuggingFace: ${FULL_REPO}"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Step 2: Run pruning
modal run gpt_oss_modal.py \
    --action prune \
    --num-experts-remove 16 \
    --output-repo "${FULL_REPO}" \
    --use-8bit

echo ""
echo "=========================================="
echo "✅ Pruning Complete!"
echo "=========================================="
echo "Model saved to: https://huggingface.co/${FULL_REPO}"
echo ""
echo "Next steps:"
echo "1. Visit your HuggingFace repo to verify the upload"
echo "2. Test inference with: modal run gpt_oss_modal.py --action inference --model-id ${FULL_REPO}"
echo "3. Update the model card with pruning details"