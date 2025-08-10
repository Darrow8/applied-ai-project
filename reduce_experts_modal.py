#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import modal
from modal import App, Image, method, enter

# Define the Modal app
app = App("gpt-oss-expert-reduction")

# Create a custom image with required dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "bitsandbytes",
        "numpy",
        "scipy",
        "scikit-learn",
    )
)

# Create a volume for model caching
volume = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)

# ---------
# Expert Reduction Utilities
# ---------

def get_expert_tensors(state_dict: Dict, expert_idx: int) -> Dict[str, Any]:
    """Extract all tensors belonging to a specific expert."""
    expert_tensors = {}
    for name, tensor in state_dict.items():
        if f".experts.{expert_idx}." in name or f"expert_{expert_idx}" in name:
            expert_tensors[name] = tensor
    return expert_tensors

def magnitude_prune_tensor(tensor, sparsity: float = 0.5):
    """
    Apply magnitude-based pruning to a tensor.
    Sets the smallest magnitude weights to zero.
    """
    import torch
    
    if len(tensor.shape) < 2:
        return tensor
    
    # Calculate threshold for pruning
    flat_tensor = tensor.abs().flatten()
    k = int(len(flat_tensor) * sparsity)
    if k > 0:
        threshold = torch.kthvalue(flat_tensor, k)[0].item()
        mask = tensor.abs() > threshold
        pruned_tensor = tensor * mask
        return pruned_tensor
    return tensor

def reduce_expert_rank(tensor, rank_reduction: float = 0.25):
    """
    Reduce the rank of weight matrices using SVD.
    This compresses the weight matrix while preserving important information.
    """
    import torch
    
    if len(tensor.shape) != 2:
        return tensor
    
    # Perform SVD
    U, S, V = torch.svd(tensor)
    
    # Determine new rank
    original_rank = min(tensor.shape)
    new_rank = max(1, int(original_rank * (1 - rank_reduction)))
    
    # Truncate to lower rank
    U_reduced = U[:, :new_rank]
    S_reduced = S[:new_rank]
    V_reduced = V[:, :new_rank]
    
    # Reconstruct with lower rank
    reconstructed = torch.mm(torch.mm(U_reduced, torch.diag(S_reduced)), V_reduced.t())
    
    return reconstructed

def quantize_expert_weights(tensor, bits: int = 8):
    """
    Quantize weights to reduce precision.
    """
    import torch
    
    if bits == 8:
        # Simulate INT8 quantization
        scale = tensor.abs().max() / 127.0
        quantized = torch.round(tensor / scale).clamp(-128, 127)
        dequantized = quantized * scale
        return dequantized
    elif bits == 4:
        # Simulate INT4 quantization
        scale = tensor.abs().max() / 7.0
        quantized = torch.round(tensor / scale).clamp(-8, 7)
        dequantized = quantized * scale
        return dequantized
    else:
        return tensor

# ------------------------------
# Modal Class for Expert Reduction
# ------------------------------

@app.cls(
    image=image,
    gpu="H200",
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=300,
)
class ExpertReducer:
    
    @enter()
    def setup(self):
        """Initialize environment and set cache directories."""
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
        os.environ["TORCH_HOME"] = "/cache/torch"
        
        # Create cache directories
        for dir_path in ["/cache/huggingface", "/cache/torch"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("Environment setup complete")
    
    @method()
    def analyze_expert_sizes(self, model_id: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
        """
        Analyze the parameter count and memory usage of each expert.
        """
        import torch
        from transformers import AutoConfig
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        
        print(f"Analyzing expert sizes in {model_id}...")
        
        # Download model
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir="/cache/huggingface",
            local_dir=f"/cache/models/{model_id}",
            local_dir_use_symlinks=False
        )
        
        # Find safetensors files
        safetensor_files = list(Path(local_path).glob("*.safetensors"))
        if not safetensor_files:
            return {"error": "No safetensors files found"}
        
        expert_stats = {}
        total_params = 0
        
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    tensor_shape = f.get_tensor(key).shape
                    param_count = np.prod(tensor_shape)
                    
                    # Check if this is an expert tensor
                    for i in range(64):  # Assuming max 64 experts
                        if f".experts.{i}." in key or f"expert_{i}" in key:
                            if i not in expert_stats:
                                expert_stats[i] = {
                                    "param_count": 0,
                                    "tensors": [],
                                    "memory_mb": 0
                                }
                            expert_stats[i]["param_count"] += param_count
                            expert_stats[i]["tensors"].append({
                                "name": key,
                                "shape": str(tensor_shape),
                                "params": param_count
                            })
                            # Assuming float32 (4 bytes per param)
                            expert_stats[i]["memory_mb"] += (param_count * 4) / (1024 * 1024)
                    
                    total_params += param_count
        
        return {
            "total_params": total_params,
            "num_experts": len(expert_stats),
            "expert_stats": expert_stats,
            "avg_params_per_expert": sum(e["param_count"] for e in expert_stats.values()) / len(expert_stats) if expert_stats else 0
        }
    
    @method()
    def reduce_experts(
        self,
        model_id: str = "openai/gpt-oss-20b",
        reduction_method: str = "magnitude_prune",  # magnitude_prune, rank_reduce, quantize, combined
        reduction_factor: float = 0.5,  # How much to reduce (0.5 = 50% reduction)
        target_experts: Optional[List[int]] = None,  # None means all experts
        output_repo: Optional[str] = None,
        push_to_hub: bool = False,
        save_to_volume: bool = True,  # Save to Modal volume
        return_download_url: bool = False,  # Generate a download URL
    ) -> Dict[str, Any]:
        """
        Reduce the parameter size of experts in the model.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from huggingface_hub import HfApi, create_repo
        from safetensors.torch import save_file, load_file
        
        print(f"Loading model {model_id}...")
        
        # Download model
        local_path = Path(f"/cache/models/{model_id}")
        if not local_path.exists():
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                cache_dir="/cache/huggingface",
                local_dir=str(local_path),
                local_dir_use_symlinks=False
            )
        
        # Load configuration
        config = AutoConfig.from_pretrained(str(local_path))
        
        # Find and process safetensors files
        safetensor_files = list(local_path.glob("*.safetensors"))
        
        modified_tensors = {}
        reduction_stats = {
            "original_params": 0,
            "reduced_params": 0,
            "experts_modified": [],
            "reduction_method": reduction_method,
            "reduction_factor": reduction_factor
        }
        
        for file_path in safetensor_files:
            print(f"Processing {file_path.name}...")
            state_dict = load_file(str(file_path))
            
            for name, tensor in state_dict.items():
                # Check if this tensor belongs to an expert
                expert_idx = None
                for i in range(64):  # Assuming max 64 experts
                    if f".experts.{i}." in name or f"expert_{i}" in name:
                        expert_idx = i
                        break
                
                # If it's an expert tensor and we should modify it
                if expert_idx is not None and (target_experts is None or expert_idx in target_experts):
                    original_size = tensor.numel()
                    reduction_stats["original_params"] += original_size
                    
                    # Apply reduction based on method
                    if reduction_method == "magnitude_prune":
                        modified_tensor = magnitude_prune_tensor(tensor, sparsity=reduction_factor)
                    elif reduction_method == "rank_reduce":
                        modified_tensor = reduce_expert_rank(tensor, rank_reduction=reduction_factor)
                    elif reduction_method == "quantize":
                        bits = 4 if reduction_factor > 0.5 else 8
                        modified_tensor = quantize_expert_weights(tensor, bits=bits)
                    elif reduction_method == "combined":
                        # Apply multiple techniques
                        modified_tensor = magnitude_prune_tensor(tensor, sparsity=reduction_factor/2)
                        if len(modified_tensor.shape) == 2:
                            modified_tensor = reduce_expert_rank(modified_tensor, rank_reduction=reduction_factor/2)
                    else:
                        modified_tensor = tensor
                    
                    modified_tensors[name] = modified_tensor
                    
                    # Count non-zero parameters after reduction
                    if reduction_method == "magnitude_prune":
                        reduced_size = (modified_tensor != 0).sum().item()
                    else:
                        reduced_size = modified_tensor.numel()
                    
                    reduction_stats["reduced_params"] += reduced_size
                    
                    if expert_idx not in reduction_stats["experts_modified"]:
                        reduction_stats["experts_modified"].append(expert_idx)
                else:
                    # Keep tensor unchanged
                    modified_tensors[name] = tensor
        
        # Calculate reduction percentage
        if reduction_stats["original_params"] > 0:
            reduction_stats["reduction_percentage"] = (
                (reduction_stats["original_params"] - reduction_stats["reduced_params"]) 
                / reduction_stats["original_params"] * 100
            )
        else:
            reduction_stats["reduction_percentage"] = 0
        
        # Save modified model
        if output_repo and push_to_hub:
            print(f"Saving reduced model to {output_repo}...")
            
            # Create output directory
            output_dir = Path(f"/cache/reduced_models/{output_repo.replace('/', '_')}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save modified tensors
            save_file(modified_tensors, output_dir / "model.safetensors")
            
            # Copy config and tokenizer files
            for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                src = local_path / file
                if src.exists():
                    import shutil
                    shutil.copy(src, output_dir / file)
            
            # Update config to reflect reduction
            config_path = output_dir / "config.json"
            if config_path.exists():
                import json
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                config_data["model_reduction"] = {
                    "method": reduction_method,
                    "factor": reduction_factor,
                    "experts_modified": reduction_stats["experts_modified"],
                    "reduction_percentage": reduction_stats["reduction_percentage"]
                }
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
            
            # Push to Hub
            api = HfApi()
            try:
                create_repo(repo_id=output_repo, exist_ok=True, token=os.getenv("HF_TOKEN"))
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=output_repo,
                    token=os.getenv("HF_TOKEN")
                )
                reduction_stats["output_repo"] = output_repo
                print(f"Successfully pushed to {output_repo}")
            except Exception as e:
                reduction_stats["push_error"] = str(e)
                print(f"Error pushing to hub: {e}")
        
        return reduction_stats
    
    @method()
    def benchmark_reduced_model(
        self,
        original_model: str = "openai/gpt-oss-20b",
        reduced_model: str = None,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare performance between original and reduced models.
        """
        import torch
        import time
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if test_prompts is None:
            test_prompts = [
                "The future of artificial intelligence is",
                "Climate change is one of the most pressing issues because",
                "In quantum computing, the main challenge is",
            ]
        
        results = {"original": {}, "reduced": {}}
        
        # Test original model
        print(f"Testing original model: {original_model}")
        tokenizer = AutoTokenizer.from_pretrained(original_model)
        
        model = AutoModelForCausalLM.from_pretrained(
            original_model,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/cache/huggingface"
        )
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            inference_time = time.time() - start_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results["original"][prompt] = {
                "generated": generated_text,
                "inference_time": inference_time
            }
        
        del model
        torch.cuda.empty_cache()
        
        # Test reduced model if provided
        if reduced_model:
            print(f"Testing reduced model: {reduced_model}")
            model = AutoModelForCausalLM.from_pretrained(
                reduced_model,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir="/cache/huggingface"
            )
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                inference_time = time.time() - start_time
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results["reduced"][prompt] = {
                    "generated": generated_text,
                    "inference_time": inference_time
                }
            
            # Calculate speedup
            avg_original = sum(r["inference_time"] for r in results["original"].values()) / len(test_prompts)
            avg_reduced = sum(r["inference_time"] for r in results["reduced"].values()) / len(test_prompts)
            results["speedup"] = avg_original / avg_reduced
        
        return results

# ------------------------------
# Main Function for CLI
# ------------------------------

@app.local_entrypoint()
def main(
    action: str = "analyze",  # analyze, reduce, benchmark
    model_id: str = "openai/gpt-oss-20b",
    reduction_method: str = "magnitude_prune",  # magnitude_prune, rank_reduce, quantize, combined
    reduction_factor: float = 0.5,
    target_experts: Optional[str] = None,  # Comma-separated list
    output_repo: Optional[str] = None,
    push_to_hub: bool = False,
):
    """
    CLI for reducing GPT-OSS expert parameters.
    
    Examples:
        # Analyze expert sizes
        modal run reduce_experts_modal.py --action analyze
        
        # Reduce all experts by 50% using magnitude pruning
        modal run reduce_experts_modal.py --action reduce --reduction-method magnitude_prune --reduction-factor 0.5
        
        # Reduce specific experts using rank reduction
        modal run reduce_experts_modal.py --action reduce --reduction-method rank_reduce --target-experts "0,1,2,3" --reduction-factor 0.3
        
        # Reduce and push to Hub
        modal run reduce_experts_modal.py --action reduce --output-repo "username/model-reduced" --push-to-hub
        
        # Benchmark original vs reduced
        modal run reduce_experts_modal.py --action benchmark --model-id "openai/gpt-oss-20b" --reduced-model "username/model-reduced"
    """
    
    reducer = ExpertReducer()
    
    if action == "analyze":
        result = reducer.analyze_expert_sizes.remote(model_id=model_id)
        
        print("\n=== Expert Size Analysis ===")
        print(f"Total parameters: {result.get('total_params', 0):,}")
        print(f"Number of experts: {result.get('num_experts', 0)}")
        print(f"Average params per expert: {result.get('avg_params_per_expert', 0):,.0f}")
        
        if "expert_stats" in result:
            print("\n=== Per-Expert Statistics ===")
            for idx, stats in sorted(result["expert_stats"].items())[:5]:  # Show first 5
                print(f"Expert {idx}:")
                print(f"  Parameters: {stats['param_count']:,}")
                print(f"  Memory: {stats['memory_mb']:.2f} MB")
                print(f"  Tensors: {len(stats['tensors'])}")
    
    elif action == "reduce":
        # Parse target experts
        experts_list = None
        if target_experts:
            experts_list = [int(x.strip()) for x in target_experts.split(",")]
        
        result = reducer.reduce_experts.remote(
            model_id=model_id,
            reduction_method=reduction_method,
            reduction_factor=reduction_factor,
            target_experts=experts_list,
            output_repo=output_repo,
            push_to_hub=push_to_hub,
        )
        
        print("\n=== Reduction Results ===")
        print(f"Method: {result['reduction_method']}")
        print(f"Factor: {result['reduction_factor']}")
        print(f"Original parameters: {result['original_params']:,}")
        print(f"Reduced parameters: {result['reduced_params']:,}")
        print(f"Reduction: {result['reduction_percentage']:.2f}%")
        print(f"Experts modified: {len(result['experts_modified'])}")
        
        if "output_repo" in result:
            print(f"Output repository: {result['output_repo']}")
    
    elif action == "benchmark":
        # For benchmark, model_id is original, output_repo is reduced model
        result = reducer.benchmark_reduced_model.remote(
            original_model=model_id,
            reduced_model=output_repo,
        )
        
        print("\n=== Benchmark Results ===")
        for model_type in ["original", "reduced"]:
            if model_type in result:
                print(f"\n{model_type.upper()} Model:")
                for prompt, data in result[model_type].items():
                    print(f"\nPrompt: {prompt[:50]}...")
                    print(f"Time: {data['inference_time']:.2f}s")
        
        if "speedup" in result:
            print(f"\nSpeedup: {result['speedup']:.2f}x")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: analyze, reduce, benchmark")

if __name__ == "__main__":
    main()