#!/usr/bin/env python3
"""
Debug script for GPT-OSS pruning issues.
Runs locally to understand tensor shapes before Modal execution.
"""

def analyze_error(error_msg):
    """Analyze common pruning errors and suggest fixes."""
    
    if "size mismatch" in error_msg:
        print("\n❌ Size Mismatch Error Detected")
        print("This typically means:")
        print("1. The model has a different expert count than expected")
        print("2. The tensor slicing logic needs adjustment")
        print("\nSuggested fixes:")
        print("- Run: modal run gpt_oss_modal.py --action inspect")
        print("- Check the actual tensor shapes")
        print("- Verify the number of experts (appears to be 32)")
        
    elif "out of range" in error_msg:
        print("\n❌ Expert Index Out of Range")
        print("The model has fewer experts than the index you specified")
        print("GPT-OSS has 32 experts (0-31)")
        
    elif "CUDA out of memory" in error_msg or "OOM" in error_msg:
        print("\n❌ Out of Memory Error")
        print("Suggestions:")
        print("- Use --use-8bit flag for quantization")
        print("- Switch to a larger GPU (H100)")
        print("- Reduce batch size or sequence length")
        
    else:
        print("\n❓ Unknown Error")
        print("Please share the full error message for debugging")

def check_requirements():
    """Check if required packages are installed."""
    required = ["modal", "torch", "transformers"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages installed")
    return True

def suggest_commands():
    """Suggest useful debugging commands."""
    print("\n📋 Useful Commands for Debugging:")
    print("\n1. Inspect model structure:")
    print("   modal run gpt_oss_modal.py --action inspect")
    
    print("\n2. Test inference (verify model loads):")
    print("   modal run gpt_oss_modal.py --action inference --prompt 'test' --use-8bit")
    
    print("\n3. Analyze model config:")
    print("   modal run gpt_oss_modal.py --action analyze")
    
    print("\n4. Try pruning with different expert:")
    print("   modal run gpt_oss_modal.py --action prune --expert 0")
    
    print("\n5. Monitor Modal logs:")
    print("   modal app logs gpt-oss-moe")

if __name__ == "__main__":
    import sys
    
    print("🔍 GPT-OSS Pruning Debugger")
    print("="*50)
    
    if not check_requirements():
        sys.exit(1)
    
    if len(sys.argv) > 1:
        # If error message passed as argument
        error_msg = " ".join(sys.argv[1:])
        analyze_error(error_msg)
    else:
        suggest_commands()
    
    print("\n💡 Tip: The GPT-OSS model has 32 experts in MoE layers")
    print("   Expert indices should be 0-31")
    print("   Each expert handles ~3.6B/32 ≈ 112M active parameters")