#!/usr/bin/env python3
"""
Test various prompts with GPT-OSS on Modal.
"""

import subprocess
import json
import time
from typing import List, Dict

# Test prompts for different capabilities
TEST_PROMPTS = [
    {
        "name": "Coding",
        "prompt": "Write a Python function to calculate fibonacci numbers",
        "tokens": 150
    },
    {
        "name": "Reasoning",
        "prompt": "If I have 3 apples and buy 5 more, then give away 2, how many do I have?",
        "tokens": 50
    },
    {
        "name": "Creative",
        "prompt": "Write a haiku about artificial intelligence",
        "tokens": 30
    },
    {
        "name": "Technical",
        "prompt": "Explain how transformers work in machine learning",
        "tokens": 200
    },
    {
        "name": "Analysis",
        "prompt": "What are the pros and cons of using microservices architecture?",
        "tokens": 150
    }
]

def run_test(prompt_info: Dict, use_quantization: bool = False) -> None:
    """Run a single test prompt."""
    print(f"\n{'='*60}")
    print(f"Test: {prompt_info['name']}")
    print(f"Prompt: {prompt_info['prompt']}")
    print(f"Max tokens: {prompt_info['tokens']}")
    print(f"Quantization: {'8-bit' if use_quantization else 'None'}")
    print('='*60)
    
    cmd = [
        "modal", "run", "gpt_oss_modal.py",
        "--action", "inference",
        "--prompt", prompt_info['prompt'],
        "--max-tokens", str(prompt_info['tokens'])
    ]
    
    if use_quantization:
        cmd.append("--use-8bit")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(f"\nTime taken: {elapsed:.2f} seconds")
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    return elapsed

def main():
    print("GPT-OSS 20B Test Suite")
    print("="*60)
    
    # Test without quantization
    print("\nRunning tests with full precision...")
    for prompt_info in TEST_PROMPTS[:2]:  # Just run first 2 to save time
        run_test(prompt_info, use_quantization=False)
        time.sleep(2)  # Small delay between tests
    
    # Test with 8-bit quantization
    print("\n\nRunning tests with 8-bit quantization...")
    for prompt_info in TEST_PROMPTS[:2]:  # Just run first 2 to save time
        run_test(prompt_info, use_quantization=True)
        time.sleep(2)

if __name__ == "__main__":
    main()