#!/usr/bin/env python3
"""
Simple CLI wrapper for running GPT-OSS inference on Modal.
Usage:
    python run_gpt_oss.py "Your prompt here"
    python run_gpt_oss.py "Your prompt" --tokens 200 --8bit
"""

import subprocess
import sys
import argparse

def run_inference(prompt, max_tokens=100, use_8bit=False, use_4bit=False):
    """Run GPT-OSS inference via Modal."""
    
    cmd = [
        "modal", "run", "gpt_oss_modal.py",
        "--action", "inference",
        "--prompt", prompt,
        "--max-tokens", str(max_tokens)
    ]
    
    if use_8bit:
        cmd.append("--use-8bit")
    if use_4bit:
        cmd.append("--use-4bit")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run GPT-OSS inference on Modal")
    parser.add_argument("prompt", help="The prompt to send to the model")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    return run_inference(
        args.prompt,
        max_tokens=args.tokens,
        use_8bit=vars(args)["8bit"],
        use_4bit=vars(args)["4bit"]
    )

if __name__ == "__main__":
    sys.exit(main())