#!/usr/bin/env python3
"""
Model Parameter Size Analyzer

This script provides utilities to analyze and report the parameter size of various model types,
including PyTorch models, Hugging Face models, and safetensors files.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np


def count_parameters_torch(model) -> Dict[str, Any]:
    """
    Count parameters in a PyTorch model.
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Dictionary with parameter statistics
    """
    import torch
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    param_details = []
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
        
        param_details.append({
            'name': name,
            'shape': list(param.shape),
            'count': param_count,
            'trainable': param.requires_grad,
            'dtype': str(param.dtype)
        })
    
    # Calculate memory usage (assuming float32 by default)
    memory_bytes = {}
    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
        dtype_params = sum(p.numel() for p in model.parameters() if p.dtype == dtype)
        if dtype_params > 0:
            bytes_per_param = torch.tensor([], dtype=dtype).element_size()
            memory_bytes[str(dtype)] = dtype_params * bytes_per_param
    
    total_memory_mb = sum(memory_bytes.values()) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'memory_mb': total_memory_mb,
        'memory_breakdown': memory_bytes,
        'param_details': param_details
    }


def count_parameters_safetensors(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Count parameters in a safetensors file.
    
    Args:
        file_path: Path to safetensors file
    
    Returns:
        Dictionary with parameter statistics
    """
    from safetensors import safe_open
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    total_params = 0
    param_details = []
    layer_stats = {}
    
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            param_count = np.prod(tensor.shape)
            total_params += param_count
            
            # Track layer statistics
            layer_name = key.split('.')[0] if '.' in key else 'root'
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {'count': 0, 'params': 0}
            layer_stats[layer_name]['count'] += 1
            layer_stats[layer_name]['params'] += param_count
            
            param_details.append({
                'name': key,
                'shape': list(tensor.shape),
                'count': param_count,
                'dtype': str(tensor.dtype)
            })
    
    # Estimate memory (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'memory_mb': memory_mb,
        'num_tensors': len(param_details),
        'layer_stats': layer_stats,
        'param_details': param_details,
        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
    }


def count_parameters_hf_model(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Count parameters in a Hugging Face model.
    
    Args:
        model_id: Hugging Face model ID or local path
        cache_dir: Optional cache directory
    
    Returns:
        Dictionary with parameter statistics
    """
    from transformers import AutoModel, AutoConfig
    from huggingface_hub import model_info, list_repo_files
    import torch
    
    stats = {
        'model_id': model_id,
        'config': {},
        'parameter_count': {},
        'file_info': {}
    }
    
    # Try to get model info from Hub
    try:
        info = model_info(model_id)
        stats['hub_info'] = {
            'downloads': info.downloads,
            'likes': info.likes,
            'tags': info.tags,
            'pipeline_tag': info.pipeline_tag
        }
        
        # Get file sizes
        files = list_repo_files(model_id)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        pytorch_files = [f for f in files if f.endswith('.bin')]
        
        stats['file_info'] = {
            'safetensor_files': len(safetensor_files),
            'pytorch_files': len(pytorch_files),
            'total_files': len(files)
        }
    except Exception as e:
        stats['hub_info'] = {'error': str(e)}
    
    # Load config
    try:
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        stats['config'] = {
            'model_type': config.model_type,
            'hidden_size': getattr(config, 'hidden_size', None),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
            'num_attention_heads': getattr(config, 'num_attention_heads', None),
            'vocab_size': getattr(config, 'vocab_size', None),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', None)
        }
        
        # For MoE models
        if hasattr(config, 'num_experts'):
            stats['config']['num_experts'] = config.num_experts
        if hasattr(config, 'num_experts_per_tok'):
            stats['config']['num_experts_per_tok'] = config.num_experts_per_tok
            
    except Exception as e:
        stats['config'] = {'error': str(e)}
    
    # Try to load model and count parameters
    try:
        # Load with minimal memory usage
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        param_stats = count_parameters_torch(model)
        stats['parameter_count'] = param_stats
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        stats['parameter_count'] = {'error': str(e), 'note': 'Could not load full model'}
    
    return stats


def analyze_model_directory(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze all model files in a directory.
    
    Args:
        directory: Path to directory containing model files
    
    Returns:
        Dictionary with comprehensive statistics
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    stats = {
        'directory': str(directory),
        'total_params': 0,
        'total_memory_mb': 0,
        'files': []
    }
    
    # Check for safetensors files
    safetensor_files = list(directory.glob("*.safetensors"))
    for file_path in safetensor_files:
        file_stats = count_parameters_safetensors(file_path)
        stats['files'].append({
            'name': file_path.name,
            'type': 'safetensors',
            'params': file_stats['total_params'],
            'memory_mb': file_stats['memory_mb']
        })
        stats['total_params'] += file_stats['total_params']
        stats['total_memory_mb'] += file_stats['memory_mb']
    
    # Check for PyTorch files
    pytorch_files = list(directory.glob("*.bin")) + list(directory.glob("*.pt")) + list(directory.glob("*.pth"))
    for file_path in pytorch_files:
        try:
            import torch
            state_dict = torch.load(file_path, map_location='cpu')
            file_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            file_memory = (file_params * 4) / (1024 * 1024)  # Assuming float32
            
            stats['files'].append({
                'name': file_path.name,
                'type': 'pytorch',
                'params': file_params,
                'memory_mb': file_memory
            })
            stats['total_params'] += file_params
            stats['total_memory_mb'] += file_memory
        except Exception as e:
            stats['files'].append({
                'name': file_path.name,
                'type': 'pytorch',
                'error': str(e)
            })
    
    # Check for config file
    config_file = directory / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            stats['config'] = json.load(f)
    
    return stats


def format_param_count(count: int) -> str:
    """
    Format parameter count in human-readable format.
    
    Args:
        count: Number of parameters
    
    Returns:
        Formatted string
    """
    if count >= 1e12:
        return f"{count/1e12:.2f}T"
    elif count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)


def print_parameter_summary(stats: Dict[str, Any], detailed: bool = False):
    """
    Print a formatted summary of parameter statistics.
    
    Args:
        stats: Statistics dictionary from any count_parameters function
        detailed: Whether to print detailed layer information
    """
    print("\n" + "="*60)
    print("MODEL PARAMETER ANALYSIS")
    print("="*60)
    
    if 'model_id' in stats:
        print(f"Model: {stats['model_id']}")
    elif 'directory' in stats:
        print(f"Directory: {stats['directory']}")
    
    # Basic statistics
    if 'total_params' in stats:
        print(f"\nTotal Parameters: {format_param_count(stats['total_params'])} ({stats['total_params']:,})")
    
    if 'trainable_params' in stats:
        print(f"Trainable Parameters: {format_param_count(stats['trainable_params'])} ({stats['trainable_params']:,})")
        print(f"Non-trainable Parameters: {format_param_count(stats['non_trainable_params'])} ({stats['non_trainable_params']:,})")
    
    if 'memory_mb' in stats:
        print(f"Estimated Memory: {stats['memory_mb']:.2f} MB ({stats['memory_mb']/1024:.2f} GB)")
    elif 'total_memory_mb' in stats:
        print(f"Total Memory: {stats['total_memory_mb']:.2f} MB ({stats['total_memory_mb']/1024:.2f} GB)")
    
    # Config information
    if 'config' in stats and isinstance(stats['config'], dict):
        print("\nModel Configuration:")
        for key, value in stats['config'].items():
            if value is not None and key != 'error':
                print(f"  {key}: {value}")
    
    # Layer statistics
    if 'layer_stats' in stats and detailed:
        print("\nLayer Statistics:")
        for layer_name, layer_info in sorted(stats['layer_stats'].items(), 
                                            key=lambda x: x[1]['params'], 
                                            reverse=True)[:10]:
            print(f"  {layer_name}: {format_param_count(layer_info['params'])} params in {layer_info['count']} tensors")
    
    # File information
    if 'files' in stats and stats['files']:
        print("\nFile Breakdown:")
        for file_info in stats['files']:
            if 'params' in file_info:
                print(f"  {file_info['name']}: {format_param_count(file_info['params'])} params ({file_info.get('memory_mb', 0):.2f} MB)")
            elif 'error' in file_info:
                print(f"  {file_info['name']}: Error - {file_info['error']}")
    
    print("="*60 + "\n")


def compare_models(models: List[Union[str, Path]], cache_dir: Optional[str] = None):
    """
    Compare parameter counts across multiple models.
    
    Args:
        models: List of model IDs or paths
        cache_dir: Optional cache directory
    """
    results = []
    
    for model in models:
        print(f"\nAnalyzing: {model}")
        try:
            if Path(model).exists():
                if Path(model).is_file():
                    stats = count_parameters_safetensors(model)
                else:
                    stats = analyze_model_directory(model)
            else:
                stats = count_parameters_hf_model(model, cache_dir)
            
            stats['name'] = str(model)
            results.append(stats)
        except Exception as e:
            print(f"Error analyzing {model}: {e}")
            results.append({'name': str(model), 'error': str(e)})
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'Parameters':<15} {'Memory (GB)':<15} {'Status':<10}")
    print("-"*80)
    
    for result in results:
        name = result['name'][:29]
        if 'error' in result:
            print(f"{name:<30} {'N/A':<15} {'N/A':<15} {'Error':<10}")
        else:
            params = result.get('total_params', result.get('parameter_count', {}).get('total_params', 0))
            memory = result.get('memory_mb', result.get('total_memory_mb', 0)) / 1024
            print(f"{name:<30} {format_param_count(params):<15} {memory:.2f} GB{'':<11} {'OK':<10}")
    
    print("="*80 + "\n")


def main():
    """
    Main function for CLI usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model parameter sizes")
    parser.add_argument("target", help="Model ID, file path, or directory to analyze")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed layer statistics")
    parser.add_argument("--compare", "-c", nargs="+", help="Compare multiple models")
    parser.add_argument("--cache-dir", help="Cache directory for Hugging Face models")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        models = [args.target] + args.compare
        compare_models(models, args.cache_dir)
    else:
        # Analyze single model
        target = Path(args.target) if Path(args.target).exists() else args.target
        
        try:
            if isinstance(target, Path):
                if target.is_file() and target.suffix == '.safetensors':
                    stats = count_parameters_safetensors(target)
                elif target.is_dir():
                    stats = analyze_model_directory(target)
                else:
                    raise ValueError(f"Unsupported file type: {target}")
            else:
                stats = count_parameters_hf_model(target, args.cache_dir)
            
            print_parameter_summary(stats, detailed=args.detailed)
            
            if args.output:
                # Save to JSON
                output_path = Path(args.output)
                # Remove non-serializable objects
                clean_stats = json.loads(json.dumps(stats, default=str))
                with open(output_path, 'w') as f:
                    json.dump(clean_stats, f, indent=2)
                print(f"Results saved to: {output_path}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()