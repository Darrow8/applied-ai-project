#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

import modal
from modal import App, Image, method, enter, Secret

# Define the Modal app
app = App("gpt-oss-moe")

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
        "bitsandbytes",  # For quantization support
        "numpy",
        "scipy",  # For SVD in rank reduction
    )
)

# Create a volume for model caching
volume = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)

# ---------
# GPT-OSS Specific Utilities
# ---------

def is_moe_router_tensor(name: str) -> bool:
    """
    Identify router/gate tensors in GPT-OSS MoE layers.
    GPT-OSS uses MXFP4 quantization for MoE layers.
    """
    # Common patterns for GPT-OSS and similar architectures
    patterns = [
        ".router",
        ".gate",
        ".moe_gate",
        ".expert_gate",
        "moe.gate",
        "moe.router",
        ".block_sparse_moe.router",
    ]
    return any(pattern in name.lower() for pattern in patterns)

def contains_expert_k(name: str, k: int) -> bool:
    """Check if tensor name contains expert k."""
    patterns = [
        f".experts.{k}.",
        f".expert_{k}.",
        f"expert.{k}.",
        f"moe.experts.{k}.",
    ]
    return any(pattern in name for pattern in patterns)

def is_gpt_oss_expert_tensor(name: str) -> bool:
    """
    Check if this is a GPT-OSS expert tensor that needs dimension reduction.
    GPT-OSS stores experts as dimensions in tensors rather than separate layers.
    """
    # GPT-OSS patterns where experts are stored as first dimension
    # Include all MLP expert tensors (weights and biases)
    patterns = [
        "mlp.experts.",  # Catch all expert tensors
    ]
    return any(pattern in name for pattern in patterns)

def get_num_experts_from_config(cfg) -> int:
    """Extract number of experts from GPT-OSS config."""
    # GPT-OSS specific: it has 32 experts
    # The model uses a different structure where experts are embedded in the weight tensors
    
    # Try various common config attributes
    expert_attrs = [
        "num_experts",
        "num_local_experts", 
        "n_experts",
        "moe_num_experts",
        "num_moe_experts",
        "n_expert",  # GPT-OSS might use this
    ]
    
    for attr in expert_attrs:
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    
    # Default for GPT-OSS based on observed tensor shapes
    print("Warning: Could not find expert count in config, defaulting to 32 for GPT-OSS")
    return 32

def slice_expert_from_router(tensor, expert_idx: int, num_experts: int):
    """Remove expert dimension from router tensor."""
    import torch
    
    if tensor.ndim == 2:
        # Weight matrix [hidden, num_experts] or [num_experts, hidden]
        if tensor.shape[0] == num_experts:
            return torch.cat([tensor[:expert_idx], tensor[expert_idx+1:]], dim=0)
        elif tensor.shape[1] == num_experts:
            return torch.cat([tensor[:, :expert_idx], tensor[:, expert_idx+1:]], dim=1)
    elif tensor.ndim == 1 and tensor.shape[0] == num_experts:
        # Bias vector [num_experts]
        return torch.cat([tensor[:expert_idx], tensor[expert_idx+1:]], dim=0)
    
    return tensor

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
# Modal Class for GPT-OSS Operations
# ------------------------------

@app.cls(
    image=image,
    gpu="H200",  # GPT-OSS 20B needs more memory
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=300,
    secrets=[Secret.from_name("huggingface-token")],  # Uses your existing HF_TOKEN secret
)
class GPTOSSProcessor:
    
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
    def inspect_expert_structure(self, model_id: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
        """
        Deeply inspect the expert structure of the model.
        """
        import torch
        from transformers import AutoModelForCausalLM
        
        print(f"Loading model to inspect expert structure...")
        
        # Load just the state dict to inspect
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        state_dict = model.state_dict()
        
        expert_tensors = {}
        router_tensors = {}
        
        for name, tensor in state_dict.items():
            # Check for expert-related tensors
            if "expert" in name.lower() or "moe" in name.lower():
                shape_str = str(list(tensor.shape))
                if "expert" in name.lower():
                    expert_tensors[name] = shape_str
                if "router" in name.lower() or "gate" in name.lower():
                    router_tensors[name] = shape_str
        
        # Find tensors with dimension 32 (likely number of experts)
        tensors_with_32 = {}
        for name, tensor in state_dict.items():
            if 32 in tensor.shape:
                tensors_with_32[name] = str(list(tensor.shape))
        
        del model
        del state_dict
        torch.cuda.empty_cache()
        
        # Limit tensors_with_32 to first 10 items
        tensors_with_32_limited = dict(list(tensors_with_32.items())[:10])
        
        return {
            "expert_tensors": expert_tensors,
            "router_tensors": router_tensors,
            "tensors_with_32_dim": tensors_with_32_limited,
            "total_tensors_with_32": len(tensors_with_32)
        }
    
    @method()
    def analyze_model(self, model_id: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
        """
        Analyze the GPT-OSS model structure to understand MoE configuration.
        """
        from transformers import AutoConfig
        import torch
        
        print(f"Analyzing model: {model_id}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Extract model info
        info = {
            "model_id": model_id,
            "model_type": getattr(config, "model_type", "unknown"),
            "total_params": getattr(config, "n_params", None),
            "active_params": getattr(config, "n_active_params", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", None)),
            "config_attributes": list(config.__dict__.keys()),
        }
        
        # Try to find MoE configuration
        try:
            num_experts = get_num_experts_from_config(config)
            info["num_experts"] = num_experts
            info["has_moe"] = True
        except ValueError as e:
            info["has_moe"] = False
            info["moe_error"] = str(e)
        
        # Check for MoE-related attributes
        moe_attrs = {}
        for attr in dir(config):
            if "moe" in attr.lower() or "expert" in attr.lower():
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        moe_attrs[attr] = str(value)
                except:
                    pass
        
        if moe_attrs:
            info["moe_config"] = moe_attrs
        
        return info
    
    @method()
    def run_inference(
        self,
        model_id: str = "openai/gpt-oss-20b",
        prompt: str = "Explain the benefits of mixture of experts models in AI:",
        max_new_tokens: int = 100,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> str:
        """
        Run inference with GPT-OSS model.
        Uses quantization options to fit in memory.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {model_id}")
        print(f"Quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        # Add quantization if requested
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded, running inference...")
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    @method()
    def optimize_experts(
        self,
        model_id: str = "openai/gpt-oss-20b",
        optimization_mode: str = "prune",  # "prune", "reduce", or "both"
        experts_to_remove: List[int] = None,  # For pruning
        num_experts_to_remove: int = None,  # For pruning
        reduction_method: str = "magnitude_prune",  # For reduction: magnitude_prune, rank_reduce, quantize, combined
        reduction_factor: float = 0.5,  # For reduction: how much to reduce (0.5 = 50%)
        target_experts: List[int] = None,  # For reduction: which experts to reduce (None = all)
        output_repo: Optional[str] = None,
        load_in_8bit: bool = True,
    ) -> Dict[str, Any]:
        """
        Combined method to optimize experts by pruning and/or reducing their parameters.
        
        Optimization modes:
        - "prune": Remove entire experts
        - "reduce": Reduce parameters within experts
        - "both": First prune some experts, then reduce remaining ones
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfApi
        from safetensors.torch import load_file, save_file
        
        print(f"Starting expert optimization for {model_id}")
        print(f"Optimization mode: {optimization_mode}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Try to get number of experts
        try:
            num_experts = get_num_experts_from_config(config)
        except ValueError:
            return {
                "status": "error",
                "message": "Could not determine MoE configuration for this model"
            }
        
        # Load model
        print(f"Loading model with {num_experts} experts...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "cpu",
            "torch_dtype": torch.float16 if load_in_8bit else torch.bfloat16,
        }
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        state_dict = model.state_dict()
        new_state_dict = {}
        
        optimization_stats = {
            "original_experts": num_experts,
            "optimization_mode": optimization_mode,
            "removed_experts": [],
            "reduced_experts": [],
            "removed_params": 0,
            "modified_params": 0,
            "original_param_count": 0,
            "final_param_count": 0,
        }
        
        # Step 1: Prune experts if requested
        if optimization_mode in ["prune", "both"]:
            # Determine which experts to remove
            if experts_to_remove is None and num_experts_to_remove is not None:
                experts_to_remove = list(range(0, num_experts, num_experts // num_experts_to_remove))[:num_experts_to_remove]
            elif experts_to_remove is None and optimization_mode == "prune":
                return {
                    "status": "error",
                    "message": "Must specify experts_to_remove or num_experts_to_remove for pruning"
                }
            
            if experts_to_remove:
                print(f"Pruning {len(experts_to_remove)} experts: {experts_to_remove}")
                optimization_stats["removed_experts"] = experts_to_remove
        else:
            experts_to_remove = []
        
        # Step 2: Process all tensors
        for name, param in state_dict.items():
            original_size = param.numel()
            optimization_stats["original_param_count"] += original_size
            
            # Handle GPT-OSS expert tensors (stored as dimensions)
            if is_gpt_oss_expert_tensor(name):
                if param.shape[0] == num_experts:
                    # First, remove pruned experts
                    if experts_to_remove:
                        indices_to_keep = [i for i in range(num_experts) if i not in experts_to_remove]
                        param = param[indices_to_keep]
                        optimization_stats["modified_params"] += 1
                    
                    # Then, reduce remaining experts if requested
                    if optimization_mode in ["reduce", "both"]:
                        # Determine which experts to reduce
                        experts_to_reduce = target_experts if target_experts else list(range(param.shape[0]))
                        
                        for expert_idx in experts_to_reduce:
                            if expert_idx < param.shape[0]:
                                # Apply reduction to this expert's slice
                                expert_tensor = param[expert_idx]
                                
                                if reduction_method == "magnitude_prune":
                                    reduced_tensor = magnitude_prune_tensor(expert_tensor, sparsity=reduction_factor)
                                elif reduction_method == "rank_reduce" and len(expert_tensor.shape) == 2:
                                    reduced_tensor = reduce_expert_rank(expert_tensor, rank_reduction=reduction_factor)
                                elif reduction_method == "quantize":
                                    bits = 4 if reduction_factor > 0.5 else 8
                                    reduced_tensor = quantize_expert_weights(expert_tensor, bits=bits)
                                elif reduction_method == "combined":
                                    reduced_tensor = magnitude_prune_tensor(expert_tensor, sparsity=reduction_factor/2)
                                    if len(reduced_tensor.shape) == 2:
                                        reduced_tensor = reduce_expert_rank(reduced_tensor, rank_reduction=reduction_factor/2)
                                else:
                                    reduced_tensor = expert_tensor
                                
                                param[expert_idx] = reduced_tensor
                                
                                if expert_idx not in optimization_stats["reduced_experts"]:
                                    optimization_stats["reduced_experts"].append(expert_idx)
                    
                    new_state_dict[name] = param
                else:
                    new_state_dict[name] = param
            
            # Handle router/gate tensors
            elif is_moe_router_tensor(name) and experts_to_remove:
                try:
                    indices_to_keep = [i for i in range(num_experts) if i not in experts_to_remove]
                    
                    if param.ndim == 2:
                        if param.shape[0] == num_experts:
                            new_param = param[indices_to_keep]
                        elif param.shape[1] == num_experts:
                            new_param = param[:, indices_to_keep]
                        else:
                            new_param = param
                    elif param.ndim == 1 and param.shape[0] == num_experts:
                        new_param = param[indices_to_keep]
                    else:
                        new_param = param
                    
                    new_state_dict[name] = new_param
                    if new_param.shape != param.shape:
                        optimization_stats["modified_params"] += 1
                except Exception as e:
                    print(f"Warning: Could not modify router {name}: {e}")
                    new_state_dict[name] = param
            else:
                # Pass through unchanged
                new_state_dict[name] = param
            
            # Count final parameters
            if name in new_state_dict:
                optimization_stats["final_param_count"] += new_state_dict[name].numel()
        
        # Update config if experts were pruned
        if experts_to_remove:
            new_num_experts = num_experts - len(experts_to_remove)
            if hasattr(config, "num_experts"):
                config.num_experts = new_num_experts
            elif hasattr(config, "num_local_experts"):
                config.num_local_experts = new_num_experts
            optimization_stats["new_experts"] = new_num_experts
        
        # Calculate reduction percentage
        if optimization_stats["original_param_count"] > 0:
            optimization_stats["reduction_percentage"] = (
                (optimization_stats["original_param_count"] - optimization_stats["final_param_count"]) 
                / optimization_stats["original_param_count"] * 100
            )
        
        print(f"Optimization complete:")
        print(f"  - Removed {len(optimization_stats['removed_experts'])} experts")
        print(f"  - Reduced {len(optimization_stats['reduced_experts'])} experts")
        print(f"  - Parameter reduction: {optimization_stats.get('reduction_percentage', 0):.2f}%")
        
        # Save model
        output_dir = "/tmp/optimized_gpt_oss"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Saving optimized model...")
        optimized_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        optimized_model.load_state_dict(new_state_dict, strict=False)
        optimized_model.save_pretrained(output_dir, safe_serialization=True)
        
        # Save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")
        
        # Save to Modal volume
        volume_path = f"/cache/optimized_models/{output_repo.replace('/', '_') if output_repo else 'optimized_model'}"
        Path(volume_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving to Modal volume: {volume_path}")
        
        optimized_model.save_pretrained(volume_path, safe_serialization=True)
        if 'tokenizer' in locals():
            tokenizer.save_pretrained(volume_path)
        
        optimization_stats["volume_path"] = volume_path
        optimization_stats["status"] = "success"
        
        # Push to Hub if requested
        if output_repo:
            print(f"Pushing to HuggingFace Hub: {output_repo}")
            try:
                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("No HF_TOKEN found")
                    optimization_stats["hub_error"] = "No HF_TOKEN found"
                else:
                    from huggingface_hub import login, create_repo
                    
                    try:
                        login(token=token)
                        print("Successfully authenticated with HuggingFace")
                    except Exception as e:
                        print(f"Failed to authenticate: {e}")
                        optimization_stats["hub_error"] = f"Authentication failed: {e}"
                        return optimization_stats
                    
                    api = HfApi(token=token)
                    
                    try:
                        api.create_repo(repo_id=output_repo, exist_ok=True, token=token)
                        print(f"Repository {output_repo} is ready")
                    except Exception as e:
                        print(f"Warning: Could not create repo: {e}")
                    
                    optimized_model.push_to_hub(output_repo, use_auth_token=token)
                    if 'tokenizer' in locals():
                        tokenizer.push_to_hub(output_repo, use_auth_token=token)
                    
                    optimization_stats["hub_repo"] = output_repo
            except Exception as e:
                optimization_stats["hub_error"] = str(e)
        
        return optimization_stats
    
    @method()
    def prune_experts(
        self,
        model_id: str = "openai/gpt-oss-20b",
        experts_to_remove: List[int] = None,
        num_experts_to_remove: int = None,
        output_repo: Optional[str] = None,
        load_in_8bit: bool = True,
    ) -> Dict[str, Any]:
        """
        Prune multiple experts from GPT-OSS MoE layers.
        Can specify either a list of expert indices or number of experts to remove.
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import create_repo, HfApi
        
        print(f"Starting expert pruning for {model_id}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Try to get number of experts
        try:
            num_experts = get_num_experts_from_config(config)
        except ValueError:
            return {
                "status": "error",
                "message": "Could not determine MoE configuration for this model"
            }
        
        # Determine which experts to remove
        if experts_to_remove is None:
            if num_experts_to_remove is not None:
                # Remove evenly spaced experts
                experts_to_remove = list(range(0, num_experts, num_experts // num_experts_to_remove))[:num_experts_to_remove]
            else:
                return {
                    "status": "error",
                    "message": "Must specify either experts_to_remove or num_experts_to_remove"
                }
        
        # Validate expert indices
        for idx in experts_to_remove:
            if not (0 <= idx < num_experts):
                return {
                    "status": "error",
                    "message": f"Expert index {idx} out of range [0, {num_experts-1}]"
                }
        
        print(f"Removing {len(experts_to_remove)} experts: {experts_to_remove}")
        print(f"Keeping {num_experts - len(experts_to_remove)} experts")
        
        print(f"Loading model with {num_experts} experts...")
        
        # Load model (with quantization to save memory during pruning)
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "cpu",  # Load on CPU for surgery
            "torch_dtype": torch.bfloat16,
        }
        
        if load_in_8bit:
            # Note: Can't modify 8bit models easily, so we load normally for pruning
            model_kwargs["torch_dtype"] = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        print("Analyzing model structure...")
        
        # Get state dict
        state_dict = model.state_dict()
        new_state_dict = {}
        
        removed_params = []
        modified_params = []
        
        # Debug: Check what expert tensors we have
        expert_tensors_found = []
        for name in state_dict.keys():
            if "mlp.experts" in name:
                expert_tensors_found.append((name, state_dict[name].shape))
        
        if expert_tensors_found:
            print(f"Found {len(expert_tensors_found)} expert tensors:")
            for name, shape in expert_tensors_found[:5]:  # Show first 5
                print(f"  {name}: {shape}")
        
        for name, param in state_dict.items():
            # Handle GPT-OSS expert tensors (stored as dimensions)
            if is_gpt_oss_expert_tensor(name):
                # GPT-OSS stores all experts in first dimension [num_experts, ...]
                # This includes both weights [32, H, W] and biases [32, H]
                if param.shape[0] == num_experts:
                    # Keep only experts not in removal list
                    indices_to_keep = [i for i in range(num_experts) if i not in experts_to_remove]
                    new_param = param[indices_to_keep]
                    new_state_dict[name] = new_param
                    modified_params.append(name)
                    if len(modified_params) <= 5:  # Only print first few to avoid spam
                        print(f"Modified {name}: {param.shape} -> {new_param.shape}")
                else:
                    # Not an expert dimension tensor, pass through
                    new_state_dict[name] = param
                continue
            
            # Skip individual expert tensors (if any)
            skip_tensor = False
            for expert_idx in experts_to_remove:
                if contains_expert_k(name, expert_idx):
                    removed_params.append(name)
                    skip_tensor = True
                    break
            if skip_tensor:
                continue
            
            # Modify router/gate tensors
            if is_moe_router_tensor(name):
                try:
                    # For router tensors, we need to remove multiple expert dimensions
                    indices_to_keep = [i for i in range(num_experts) if i not in experts_to_remove]
                    
                    if param.ndim == 2:
                        if param.shape[0] == num_experts:
                            new_param = param[indices_to_keep]
                        elif param.shape[1] == num_experts:
                            new_param = param[:, indices_to_keep]
                        else:
                            new_param = param
                    elif param.ndim == 1 and param.shape[0] == num_experts:
                        new_param = param[indices_to_keep]
                    else:
                        new_param = param
                    
                    if new_param.shape != param.shape:
                        new_state_dict[name] = new_param
                        modified_params.append(name)
                    else:
                        new_state_dict[name] = param
                except Exception as e:
                    print(f"Warning: Could not modify {name}: {e}")
                    new_state_dict[name] = param
            else:
                # Pass through unchanged
                new_state_dict[name] = param
        
        print(f"Removed {len(removed_params)} expert parameters")
        print(f"Modified {len(modified_params)} router parameters")
        
        # Update config
        new_num_experts = num_experts - len(experts_to_remove)
        if hasattr(config, "num_experts"):
            config.num_experts = new_num_experts
        elif hasattr(config, "num_local_experts"):
            config.num_local_experts = new_num_experts
        print(f"Updated config: {num_experts} -> {new_num_experts} experts")
        
        # Save locally
        output_dir = "/tmp/pruned_gpt_oss"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Saving pruned model...")
        
        # Create new model with updated config and load pruned weights
        pruned_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        pruned_model.load_state_dict(new_state_dict, strict=False)
        
        # Save model
        pruned_model.save_pretrained(output_dir, safe_serialization=True)
        
        # Save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")
        
        result = {
            "status": "success",
            "original_experts": num_experts,
            "new_experts": new_num_experts,
            "removed_experts": experts_to_remove,
            "num_removed": len(experts_to_remove),
            "removed_params": len(removed_params),
            "modified_params": len(modified_params),
        }
        
        # Save to Modal volume
        volume_path = f"/cache/pruned_models/{output_repo.replace('/', '_') if output_repo else 'pruned_model'}"
        Path(volume_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving to Modal volume: {volume_path}")
        
        pruned_model.save_pretrained(volume_path, safe_serialization=True)
        if 'tokenizer' in locals():
            tokenizer.save_pretrained(volume_path)
        
        result["volume_path"] = volume_path
        
        # Push to Hub if requested
        if output_repo:
            print(f"Pushing to HuggingFace Hub: {output_repo}")
            try:
                # Get token from Modal secret (HF_TOKEN)
                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("No HF_TOKEN found. Ensure your Modal secret 'huggingface-token' is properly configured")
                    result["hub_error"] = "No HF_TOKEN found"
                else:
                    # Initialize HuggingFace API with token
                    from huggingface_hub import login
                    
                    # Login first to validate token
                    try:
                        login(token=token)
                        print("Successfully authenticated with HuggingFace")
                    except Exception as e:
                        print(f"Failed to authenticate: {e}")
                        result["hub_error"] = f"Authentication failed: {e}"
                        return result
                    
                    api = HfApi(token=token)
                    
                    # Create repo if it doesn't exist
                    try:
                        api.create_repo(repo_id=output_repo, exist_ok=True, token=token)
                        print(f"Repository {output_repo} is ready")
                    except Exception as e:
                        print(f"Warning: Could not create repo: {e}")
                    
                    # Push model
                    pruned_model.push_to_hub(output_repo, use_auth_token=token)
                    if 'tokenizer' in locals():
                        tokenizer.push_to_hub(output_repo, use_auth_token=token)
                    
                    result["hub_repo"] = output_repo
            except Exception as e:
                result["hub_error"] = str(e)
        
        return result

# ------------------------------
# CLI Interface
# ------------------------------

@app.local_entrypoint()
def main(
    action: str = "analyze",
    model_id: str = "openai/gpt-oss-20b",
    num_experts_remove: int = None,
    experts: str = None,  # Comma-separated list
    output_repo: str = None,
    prompt: str = None,
    max_tokens: int = 100,
    use_8bit: bool = False,
    use_4bit: bool = False,
    # New parameters for optimization
    optimization_mode: str = "prune",  # prune, reduce, both
    reduction_method: str = "magnitude_prune",  # magnitude_prune, rank_reduce, quantize, combined
    reduction_factor: float = 0.5,
    target_experts: str = None,  # Comma-separated list for reduction
):
    """
    Modal CLI for GPT-OSS model operations.
    
    Examples:
        # Analyze model structure
        modal run gpt_oss_modal.py --action analyze
        
        # Inspect expert tensor structure (debug)
        modal run gpt_oss_modal.py --action inspect
        
        # Run inference
        modal run gpt_oss_modal.py --action inference --prompt "Explain AI safety"
        
        # PRUNING EXAMPLES (remove entire experts):
        # Prune 16 experts (evenly spaced) - reduces from 32 to 16 experts
        modal run gpt_oss_modal.py --action prune --num-experts-remove 16
        
        # Prune specific experts
        modal run gpt_oss_modal.py --action prune --experts "0,2,4,6,8,10,12,14"
        
        # OPTIMIZATION EXAMPLES (combined pruning and reduction):
        # Just reduce all experts by 50% using magnitude pruning
        modal run gpt_oss_modal.py --action optimize --optimization-mode reduce --reduction-method magnitude_prune --reduction-factor 0.5
        
        # Prune half the experts AND reduce remaining ones
        modal run gpt_oss_modal.py --action optimize --optimization-mode both --num-experts-remove 16 --reduction-method rank_reduce --reduction-factor 0.3
        
        # Reduce specific experts only
        modal run gpt_oss_modal.py --action optimize --optimization-mode reduce --target-experts "0,1,2,3" --reduction-method quantize --reduction-factor 0.5
        
        # Push optimized model to HuggingFace Hub
        modal run gpt_oss_modal.py --action optimize --optimization-mode both --num-experts-remove 8 --reduction-factor 0.3 --output-repo username/gpt-oss-optimized
    """
    processor = GPTOSSProcessor()
    
    if action == "analyze":
        print("Analyzing GPT-OSS model structure...")
        result = processor.analyze_model.remote(model_id)
        
        print("\n=== Model Analysis ===")
        for key, value in result.items():
            if key != "config_attributes":
                print(f"{key}: {value}")
        
        if "config_attributes" in result:
            print(f"\nConfig has {len(result['config_attributes'])} attributes")
    
    elif action == "inspect":
        print("Inspecting GPT-OSS expert structure...")
        result = processor.inspect_expert_structure.remote(model_id)
        
        print("\n=== Expert Structure Analysis ===")
        print(f"\nExpert tensors found: {len(result.get('expert_tensors', {}))}")
        for name, shape in list(result.get('expert_tensors', {}).items())[:5]:
            print(f"  {name}: {shape}")
        
        print(f"\nRouter/Gate tensors found: {len(result.get('router_tensors', {}))}")
        for name, shape in list(result.get('router_tensors', {}).items())[:5]:
            print(f"  {name}: {shape}")
        
        print(f"\nTensors with dimension 32: {result.get('total_tensors_with_32', 0)}")
        for name, shape in list(result.get('tensors_with_32_dim', {}).items())[:5]:
            print(f"  {name}: {shape}")
            
    elif action == "inference":
        if prompt is None:
            prompt = "Explain the benefits of mixture of experts models in AI:"
        
        print(f"Running inference with GPT-OSS...")
        print(f"Prompt: {prompt}")
        print(f"Quantization: 8bit={use_8bit}, 4bit={use_4bit}")
        
        result = processor.run_inference.remote(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_tokens,
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
        )
        
        print("\n=== Generated Text ===")
        print(result)
        
    elif action == "prune":
        # Parse experts to remove
        if experts:
            # Comma-separated list of expert indices
            experts_to_remove = [int(x.strip()) for x in experts.split(",")]
            print(f"Pruning specific experts: {experts_to_remove}")
        elif num_experts_remove:
            # Number of experts to remove (evenly spaced)
            experts_to_remove = None
            print(f"Pruning {num_experts_remove} experts (evenly spaced)")
        else:
            print("Error: Must specify either --experts or --num-experts-remove")
            return
        
        result = processor.prune_experts.remote(
            model_id=model_id,
            experts_to_remove=experts_to_remove,
            num_experts_to_remove=num_experts_remove,
            output_repo=output_repo,
            load_in_8bit=use_8bit,
        )
        
        print("\n=== Pruning Results ===")
        for key, value in result.items():
            print(f"{key}: {value}")
    
    elif action == "optimize":
        # Parse expert lists
        experts_to_remove_list = None
        target_experts_list = None
        
        if experts:
            experts_to_remove_list = [int(x.strip()) for x in experts.split(",")]
            print(f"Experts to remove: {experts_to_remove_list}")
        
        if target_experts:
            target_experts_list = [int(x.strip()) for x in target_experts.split(",")]
            print(f"Target experts for reduction: {target_experts_list}")
        
        print(f"Running expert optimization...")
        print(f"  Mode: {optimization_mode}")
        print(f"  Reduction method: {reduction_method}")
        print(f"  Reduction factor: {reduction_factor}")
        
        result = processor.optimize_experts.remote(
            model_id=model_id,
            optimization_mode=optimization_mode,
            experts_to_remove=experts_to_remove_list,
            num_experts_to_remove=num_experts_remove,
            reduction_method=reduction_method,
            reduction_factor=reduction_factor,
            target_experts=target_experts_list,
            output_repo=output_repo,
            load_in_8bit=use_8bit,
        )
        
        print("\n=== Optimization Results ===")
        for key, value in result.items():
            if key not in ["volume_path", "hub_error"]:
                print(f"{key}: {value}")
        
        if "volume_path" in result:
            print(f"\nModel saved to: {result['volume_path']}")
        
        if "hub_repo" in result:
            print(f"Pushed to Hub: {result['hub_repo']}")
        elif "hub_error" in result:
            print(f"Hub push error: {result['hub_error']}")
            
    else:
        print(f"Unknown action: {action}")
        print("Available actions: analyze, inspect, inference, prune, optimize")

# Optional: Standalone function for programmatic use
@app.function(image=image, gpu="H200", volumes={"/cache": volume})
def run_gpt_oss_inference(
    prompt: str,
    max_tokens: int = 100,
    model_id: str = "openai/gpt-oss-20b",
    quantize: bool = True,
) -> str:
    """
    Standalone function for running GPT-OSS inference.
    """
    processor = GPTOSSProcessor()
    return processor.run_inference(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_tokens,
        load_in_8bit=quantize,
    )