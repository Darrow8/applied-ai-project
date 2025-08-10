#!/usr/bin/env python3
"""
Verify that expert pruning worked correctly.
"""

import subprocess
import json

def verify_pruned_model(original_experts=32, removed_expert=2):
    """Verify the pruned model has correct number of experts."""
    
    expected_experts = original_experts - 1
    
    print(f"🔍 Verification Check")
    print(f"Original experts: {original_experts}")
    print(f"Removed expert: #{removed_expert}")
    print(f"Expected remaining: {expected_experts}")
    print("="*50)
    
    # Commands to verify
    checks = [
        {
            "name": "Model Structure",
            "cmd": ["modal", "run", "gpt_oss_modal.py", "--action", "analyze"],
            "look_for": f"num_experts.*{expected_experts}"
        },
        {
            "name": "Tensor Shapes",
            "cmd": ["modal", "run", "gpt_oss_modal.py", "--action", "inspect"],
            "look_for": f"\\[{expected_experts},"
        }
    ]
    
    results = []
    
    for check in checks:
        print(f"\n✓ Checking: {check['name']}")
        try:
            result = subprocess.run(
                check['cmd'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if check['look_for'] in result.stdout:
                print(f"  ✅ Found expected pattern: {check['look_for']}")
                results.append(True)
            else:
                print(f"  ❌ Pattern not found: {check['look_for']}")
                results.append(False)
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Check timed out")
            results.append(False)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    if all(results):
        print("✅ VERIFICATION PASSED")
        print(f"The model now has {expected_experts} experts")
    else:
        print("❌ VERIFICATION FAILED")
        print("Some checks did not pass. Review the output above.")
    
    return all(results)

def test_pruned_inference(prompt="Test inference after pruning"):
    """Test if the pruned model can still run inference."""
    
    print(f"\n🧪 Testing Inference on Pruned Model")
    print(f"Prompt: {prompt}")
    print("="*50)
    
    cmd = [
        "modal", "run", "gpt_oss_modal.py",
        "--action", "inference",
        "--prompt", prompt,
        "--max-tokens", "20",
        "--use-8bit"  # Use quantization to save memory
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode == 0:
            print("✅ Inference successful!")
            # Extract generated text if present
            if "Generated Text" in result.stdout:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if "Generated Text" in line:
                        print(f"Output: {lines[i+1] if i+1 < len(lines) else 'N/A'}")
            return True
        else:
            print("❌ Inference failed")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Inference timed out")
        return False
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("🔧 GPT-OSS Pruning Verification Tool")
    print("="*50)
    
    # Get parameters from command line or use defaults
    original = 32  # GPT-OSS has 32 experts
    removed = 2    # Default to expert 2
    
    if len(sys.argv) > 1:
        removed = int(sys.argv[1])
    
    # Run verification
    structure_ok = verify_pruned_model(original, removed)
    
    # Test inference if structure looks good
    if structure_ok:
        inference_ok = test_pruned_inference()
        
        if inference_ok:
            print("\n🎉 SUCCESS: Pruning completed and verified!")
        else:
            print("\n⚠️  WARNING: Structure OK but inference issues")
    else:
        print("\n❌ Pruning verification failed")
        print("Run: python debug_pruning.py")
        sys.exit(1)