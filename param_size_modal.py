#!/usr/bin/env python3
"""
Model Parameter Size Analyzer for Hugging Face Models on Modal

This script analyzes parameter sizes of Hugging Face models using Modal for GPU/compute resources.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np

import modal
from modal import App, Image, method, enter

# Define the Modal app
app = App("model-parameter-analyzer")

# Create custom image with required dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface-hub",
        "numpy",
    )
)

# Create volume for model caching
volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",  # Use A10G for better cost-efficiency for analysis
    volumes={"/cache": volume},
    timeout=1800,
    scaledown_window=60,
)
class ModelParameterAnalyzer:
    """Modal class for analyzing model parameters."""
    
    @enter()
    def setup(self):
        """Initialize environment and set cache directories."""
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
        os.environ["TORCH_HOME"] = "/cache/torch"
        
        # Create cache directories
        for dir_path in ["/cache/huggingface", "/cache/torch", "/cache/models"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("Environment setup complete")
    
    @method()
    def analyze_hf_model(self, model_id: str, load_in_memory: bool = False) -> Dict[str, Any]:
        """
        Analyze a Hugging Face model's parameter size without loading the full model.
        
        Args:
            model_id: Hugging Face model ID
            load_in_memory: Whether to load the model in memory for exact counting
        
        Returns:
            Dictionary with comprehensive parameter statistics
        """
        from transformers import AutoConfig
        from huggingface_hub import model_info, snapshot_download, list_repo_files
        from safetensors import safe_open
        import torch
        
        print(f"Analyzing model: {model_id}")
        
        stats = {
            'model_id': model_id,
            'config': {},
            'parameter_analysis': {},
            'file_analysis': {},
            'memory_estimates': {},
            'architecture_details': {}
        }
        
        # Get model info from Hub
        try:
            info = model_info(model_id)
            stats['hub_info'] = {
                'downloads': getattr(info, 'downloads', None),
                'likes': getattr(info, 'likes', None),
                'tags': info.tags if hasattr(info, 'tags') else [],
                'pipeline_tag': getattr(info, 'pipeline_tag', None),
                'model_size': getattr(info, 'model_size', None),
            }
            
            # Get repository files
            files = list_repo_files(model_id)
            safetensor_files = [f for f in files if f.endswith('.safetensors')]
            pytorch_files = [f for f in files if f.endswith('.bin')]
            
            stats['file_analysis'] = {
                'safetensor_files': safetensor_files,
                'pytorch_files': pytorch_files,
                'num_safetensor_files': len(safetensor_files),
                'num_pytorch_files': len(pytorch_files),
                'total_files': len(files)
            }
        except Exception as e:
            stats['hub_info'] = {'error': str(e)}
        
        # Load and analyze config
        try:
            config = AutoConfig.from_pretrained(model_id, cache_dir="/cache/huggingface")
            
            # Extract key configuration parameters
            stats['config'] = {
                'model_type': config.model_type,
                'architectures': getattr(config, 'architectures', None),
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
                'intermediate_size': getattr(config, 'intermediate_size', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
                'hidden_act': getattr(config, 'hidden_act', None),
                'rope_theta': getattr(config, 'rope_theta', None),
                'tie_word_embeddings': getattr(config, 'tie_word_embeddings', None),
            }
            
            # MoE specific parameters
            if hasattr(config, 'num_experts') or hasattr(config, 'num_local_experts'):
                stats['config']['moe_config'] = {
                    'num_experts': getattr(config, 'num_experts', getattr(config, 'num_local_experts', None)),
                    'num_experts_per_tok': getattr(config, 'num_experts_per_tok', None),
                    'expert_interval': getattr(config, 'expert_interval', None),
                    'router_type': getattr(config, 'router_type', None),
                }
            
            # Calculate theoretical parameter count
            stats['parameter_analysis']['theoretical'] = self._calculate_theoretical_params(config)
            
        except Exception as e:
            stats['config'] = {'error': str(e)}
        
        # Download and analyze model files
        try:
            print("Downloading model files for analysis...")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir="/cache/huggingface",
                local_dir=f"/cache/models/{model_id.replace('/', '_')}",
                local_dir_use_symlinks=False
            )
            
            # Analyze safetensors files
            total_params = 0
            layer_breakdown = {}
            dtype_breakdown = {}
            expert_analysis = {}
            
            safetensor_paths = list(Path(local_path).glob("*.safetensors"))
            
            for file_path in safetensor_paths:
                print(f"Analyzing {file_path.name}...")
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        param_count = np.prod(tensor.shape)
                        total_params += param_count
                        
                        # Track dtype distribution
                        dtype_str = str(tensor.dtype)
                        if dtype_str not in dtype_breakdown:
                            dtype_breakdown[dtype_str] = 0
                        dtype_breakdown[dtype_str] += param_count
                        
                        # Layer analysis
                        layer_type = self._categorize_layer(key)
                        if layer_type not in layer_breakdown:
                            layer_breakdown[layer_type] = {
                                'count': 0,
                                'params': 0,
                                'tensors': []
                            }
                        layer_breakdown[layer_type]['count'] += 1
                        layer_breakdown[layer_type]['params'] += param_count
                        
                        # Expert analysis for MoE models
                        expert_idx = self._extract_expert_index(key)
                        if expert_idx is not None:
                            if expert_idx not in expert_analysis:
                                expert_analysis[expert_idx] = {
                                    'params': 0,
                                    'tensors': 0
                                }
                            expert_analysis[expert_idx]['params'] += param_count
                            expert_analysis[expert_idx]['tensors'] += 1
            
            stats['parameter_analysis']['actual'] = {
                'total_params': total_params,
                'layer_breakdown': layer_breakdown,
                'dtype_breakdown': dtype_breakdown
            }
            
            if expert_analysis:
                stats['parameter_analysis']['expert_analysis'] = {
                    'num_experts': len(expert_analysis),
                    'avg_params_per_expert': sum(e['params'] for e in expert_analysis.values()) / len(expert_analysis),
                    'expert_details': expert_analysis
                }
            
            # Memory estimates
            stats['memory_estimates'] = self._calculate_memory_estimates(
                total_params, dtype_breakdown
            )
            
            # Architecture analysis
            stats['architecture_details'] = self._analyze_architecture(
                layer_breakdown, config if 'config' in locals() else None
            )
            
        except Exception as e:
            stats['parameter_analysis']['error'] = str(e)
        
        # Optionally load model in memory for exact verification
        if load_in_memory:
            try:
                print("Loading model in memory for verification...")
                from transformers import AutoModelForCausalLM
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir="/cache/huggingface"
                )
                
                # Count parameters directly
                total_params_verified = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                stats['parameter_analysis']['verified'] = {
                    'total_params': total_params_verified,
                    'trainable_params': trainable_params,
                    'frozen_params': total_params_verified - trainable_params
                }
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                stats['parameter_analysis']['verification_error'] = str(e)
        
        return stats
    
    def _categorize_layer(self, tensor_name: str) -> str:
        """Categorize a tensor based on its name."""
        if 'embed' in tensor_name.lower():
            return 'embeddings'
        elif 'ln' in tensor_name or 'layernorm' in tensor_name.lower():
            return 'layer_norm'
        elif 'attn' in tensor_name or 'attention' in tensor_name:
            return 'attention'
        elif 'mlp' in tensor_name or 'ffn' in tensor_name or 'feed_forward' in tensor_name:
            return 'mlp'
        elif 'expert' in tensor_name:
            return 'moe_experts'
        elif 'router' in tensor_name or 'gate' in tensor_name:
            return 'moe_routing'
        elif 'lm_head' in tensor_name or 'output' in tensor_name:
            return 'output_layer'
        else:
            return 'other'
    
    def _extract_expert_index(self, tensor_name: str) -> Optional[int]:
        """Extract expert index from tensor name."""
        import re
        
        # Pattern for .experts.{idx}. or expert_{idx}
        patterns = [
            r'\.experts\.(\d+)\.',
            r'expert_(\d+)',
            r'expert(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, tensor_name)
            if match:
                return int(match.group(1))
        
        return None
    
    def _calculate_theoretical_params(self, config) -> Dict[str, Any]:
        """Calculate theoretical parameter count from config."""
        params = {}
        
        if hasattr(config, 'hidden_size') and hasattr(config, 'vocab_size'):
            # Embeddings
            params['embeddings'] = config.vocab_size * config.hidden_size
            
            if hasattr(config, 'num_hidden_layers'):
                # Transformer layers
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                
                # Attention
                if hasattr(config, 'num_attention_heads'):
                    # QKV projections + output projection
                    params['attention'] = num_layers * (4 * hidden_size * hidden_size)
                
                # MLP
                if hasattr(config, 'intermediate_size'):
                    intermediate_size = config.intermediate_size
                    params['mlp'] = num_layers * (
                        hidden_size * intermediate_size +  # up projection
                        intermediate_size * hidden_size    # down projection
                    )
                    
                    # Gate projection for gated architectures
                    if hasattr(config, 'hidden_act') and 'glu' in str(config.hidden_act).lower():
                        params['mlp'] += num_layers * hidden_size * intermediate_size
                
                # Layer norms
                params['layer_norm'] = num_layers * 2 * hidden_size
            
            # Output layer
            if not getattr(config, 'tie_word_embeddings', False):
                params['output'] = config.vocab_size * config.hidden_size
            
            params['total_theoretical'] = sum(params.values())
        
        return params
    
    def _calculate_memory_estimates(self, total_params: int, dtype_breakdown: Dict) -> Dict[str, Any]:
        """Calculate memory usage estimates."""
        memory = {}
        
        # Bytes per parameter for different dtypes
        dtype_bytes = {
            'torch.float32': 4,
            'torch.float16': 2,
            'torch.bfloat16': 2,
            'torch.int8': 1,
            'torch.uint8': 1,
        }
        
        # Calculate actual memory from dtype breakdown
        total_bytes = 0
        for dtype_str, param_count in dtype_breakdown.items():
            bytes_per_param = dtype_bytes.get(dtype_str, 4)  # Default to float32
            total_bytes += param_count * bytes_per_param
        
        memory['actual_mb'] = total_bytes / (1024 * 1024)
        memory['actual_gb'] = total_bytes / (1024 * 1024 * 1024)
        
        # Estimates for different precision levels
        memory['fp32_gb'] = (total_params * 4) / (1024 ** 3)
        memory['fp16_gb'] = (total_params * 2) / (1024 ** 3)
        memory['int8_gb'] = total_params / (1024 ** 3)
        memory['int4_gb'] = (total_params * 0.5) / (1024 ** 3)
        
        # Inference memory estimates (including activations, KV cache)
        memory['inference_estimate_fp16_gb'] = memory['fp16_gb'] * 1.2  # 20% overhead
        memory['inference_estimate_int8_gb'] = memory['int8_gb'] * 1.1   # 10% overhead
        
        return memory
    
    def _analyze_architecture(self, layer_breakdown: Dict, config: Optional[Any]) -> Dict[str, Any]:
        """Analyze model architecture details."""
        analysis = {}
        
        # Calculate parameter distribution percentages
        total_params = sum(layer['params'] for layer in layer_breakdown.values())
        
        if total_params > 0:
            analysis['parameter_distribution'] = {
                layer_type: {
                    'percentage': (layer_info['params'] / total_params) * 100,
                    'params': layer_info['params'],
                    'tensors': layer_info['count']
                }
                for layer_type, layer_info in layer_breakdown.items()
            }
        
        # Architecture insights
        if config:
            analysis['architecture_type'] = config.model_type if hasattr(config, 'model_type') else 'unknown'
            
            # Check for specific architectures
            if 'moe_experts' in layer_breakdown:
                analysis['is_moe'] = True
                analysis['moe_overhead'] = layer_breakdown['moe_experts']['params'] / total_params * 100
            else:
                analysis['is_moe'] = False
            
            # Attention mechanism
            if hasattr(config, 'num_key_value_heads') and hasattr(config, 'num_attention_heads'):
                if config.num_key_value_heads < config.num_attention_heads:
                    analysis['attention_type'] = 'grouped_query_attention'
                    analysis['gqa_ratio'] = config.num_attention_heads / config.num_key_value_heads
                else:
                    analysis['attention_type'] = 'multi_head_attention'
        
        return analysis
    
    @method()
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare parameter sizes across multiple models.
        
        Args:
            model_ids: List of Hugging Face model IDs
        
        Returns:
            Comparison dictionary
        """
        results = []
        
        for model_id in model_ids:
            print(f"\n{'='*60}")
            print(f"Analyzing: {model_id}")
            print('='*60)
            
            try:
                stats = self.analyze_hf_model(model_id, load_in_memory=False)
                results.append(stats)
            except Exception as e:
                results.append({
                    'model_id': model_id,
                    'error': str(e)
                })
        
        # Create comparison summary
        comparison = {
            'models': [],
            'summary': {}
        }
        
        for result in results:
            model_summary = {
                'model_id': result['model_id'],
                'status': 'error' if 'error' in result else 'success'
            }
            
            if 'error' not in result:
                actual_params = result.get('parameter_analysis', {}).get('actual', {})
                memory = result.get('memory_estimates', {})
                
                model_summary.update({
                    'total_params': actual_params.get('total_params', 0),
                    'memory_gb': memory.get('actual_gb', 0),
                    'fp16_memory_gb': memory.get('fp16_gb', 0),
                    'model_type': result.get('config', {}).get('model_type', 'unknown'),
                    'num_layers': result.get('config', {}).get('num_hidden_layers', 0),
                    'hidden_size': result.get('config', {}).get('hidden_size', 0),
                    'is_moe': result.get('architecture_details', {}).get('is_moe', False)
                })
            
            comparison['models'].append(model_summary)
        
        # Calculate summary statistics
        valid_models = [m for m in comparison['models'] if m['status'] == 'success']
        if valid_models:
            comparison['summary'] = {
                'total_models': len(model_ids),
                'successful': len(valid_models),
                'failed': len(model_ids) - len(valid_models),
                'largest_model': max(valid_models, key=lambda x: x.get('total_params', 0))['model_id'],
                'smallest_model': min(valid_models, key=lambda x: x.get('total_params', 0))['model_id'],
                'avg_params': sum(m.get('total_params', 0) for m in valid_models) / len(valid_models),
                'total_params_all': sum(m.get('total_params', 0) for m in valid_models)
            }
        
        return comparison
    
    @method()
    def export_analysis(self, model_id: str, output_format: str = "json") -> str:
        """
        Export analysis results in various formats.
        
        Args:
            model_id: Hugging Face model ID
            output_format: Format for export (json, markdown, csv)
        
        Returns:
            Formatted string with analysis results
        """
        stats = self.analyze_hf_model(model_id)
        
        if output_format == "json":
            return json.dumps(stats, indent=2, default=str)
        
        elif output_format == "markdown":
            md = f"# Model Parameter Analysis: {model_id}\n\n"
            
            # Basic info
            md += "## Configuration\n"
            config = stats.get('config', {})
            md += f"- Model Type: {config.get('model_type', 'N/A')}\n"
            md += f"- Hidden Size: {config.get('hidden_size', 'N/A')}\n"
            md += f"- Layers: {config.get('num_hidden_layers', 'N/A')}\n"
            md += f"- Vocab Size: {config.get('vocab_size', 'N/A')}\n\n"
            
            # Parameters
            md += "## Parameter Count\n"
            actual = stats.get('parameter_analysis', {}).get('actual', {})
            md += f"- Total Parameters: {self._format_number(actual.get('total_params', 0))}\n\n"
            
            # Memory
            md += "## Memory Requirements\n"
            memory = stats.get('memory_estimates', {})
            md += f"- FP32: {memory.get('fp32_gb', 0):.2f} GB\n"
            md += f"- FP16: {memory.get('fp16_gb', 0):.2f} GB\n"
            md += f"- INT8: {memory.get('int8_gb', 0):.2f} GB\n\n"
            
            # Layer breakdown
            if 'layer_breakdown' in actual:
                md += "## Layer Distribution\n"
                for layer_type, info in actual['layer_breakdown'].items():
                    percentage = (info['params'] / actual['total_params']) * 100 if actual['total_params'] > 0 else 0
                    md += f"- {layer_type}: {self._format_number(info['params'])} ({percentage:.1f}%)\n"
            
            return md
        
        elif output_format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Metric', 'Value'])
            
            # Write key metrics
            writer.writerow(['Model ID', model_id])
            writer.writerow(['Total Parameters', stats.get('parameter_analysis', {}).get('actual', {}).get('total_params', 0)])
            writer.writerow(['Memory (GB)', stats.get('memory_estimates', {}).get('actual_gb', 0)])
            writer.writerow(['Model Type', stats.get('config', {}).get('model_type', 'N/A')])
            writer.writerow(['Layers', stats.get('config', {}).get('num_hidden_layers', 0)])
            writer.writerow(['Hidden Size', stats.get('config', {}).get('hidden_size', 0)])
            
            return output.getvalue()
        
        else:
            return f"Unsupported format: {output_format}"
    
    def _format_number(self, num: int) -> str:
        """Format large numbers in readable format."""
        if num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)


@app.local_entrypoint()
def main(
    model_id: str = "meta-llama/Llama-2-7b-hf",
    action: str = "analyze",  # analyze, compare, export
    compare_with: Optional[str] = None,  # Comma-separated list for comparison
    output_format: str = "json",  # json, markdown, csv
    load_in_memory: bool = False,
    output_file: Optional[str] = None
):
    """
    CLI for analyzing model parameter sizes.
    
    Examples:
        # Analyze a single model
        modal run param_size_modal.py --model-id meta-llama/Llama-2-7b-hf
        
        # Compare multiple models
        modal run param_size_modal.py --model-id meta-llama/Llama-2-7b-hf --action compare --compare-with "meta-llama/Llama-2-13b-hf,mistralai/Mistral-7B-v0.1"
        
        # Export analysis to markdown
        modal run param_size_modal.py --model-id meta-llama/Llama-2-7b-hf --action export --output-format markdown
        
        # Analyze with in-memory verification
        modal run param_size_modal.py --model-id meta-llama/Llama-2-7b-hf --load-in-memory
    """
    
    analyzer = ModelParameterAnalyzer()
    
    if action == "analyze":
        result = analyzer.analyze_hf_model.remote(model_id, load_in_memory=load_in_memory)
        
        # Print summary
        print("\n" + "="*60)
        print(f"MODEL PARAMETER ANALYSIS: {model_id}")
        print("="*60)
        
        # Configuration
        config = result.get('config', {})
        print("\nConfiguration:")
        print(f"  Model Type: {config.get('model_type', 'N/A')}")
        print(f"  Architecture: {config.get('architectures', ['N/A'])[0] if config.get('architectures') else 'N/A'}")
        print(f"  Hidden Size: {config.get('hidden_size', 'N/A')}")
        print(f"  Layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"  Attention Heads: {config.get('num_attention_heads', 'N/A')}")
        print(f"  Vocab Size: {config.get('vocab_size', 'N/A'):,}")
        
        # MoE config if present
        if 'moe_config' in config:
            print("\nMoE Configuration:")
            moe = config['moe_config']
            print(f"  Number of Experts: {moe.get('num_experts', 'N/A')}")
            print(f"  Experts per Token: {moe.get('num_experts_per_tok', 'N/A')}")
        
        # Parameter count
        actual = result.get('parameter_analysis', {}).get('actual', {})
        if actual:
            print(f"\nTotal Parameters: {actual.get('total_params', 0):,}")
            
            # Expert analysis for MoE
            if 'expert_analysis' in result.get('parameter_analysis', {}):
                expert = result['parameter_analysis']['expert_analysis']
                print(f"\nExpert Analysis:")
                print(f"  Number of Experts: {expert['num_experts']}")
                print(f"  Avg Params per Expert: {expert['avg_params_per_expert']:,.0f}")
        
        # Memory estimates
        memory = result.get('memory_estimates', {})
        if memory:
            print(f"\nMemory Requirements:")
            print(f"  Actual: {memory.get('actual_gb', 0):.2f} GB")
            print(f"  FP32: {memory.get('fp32_gb', 0):.2f} GB")
            print(f"  FP16: {memory.get('fp16_gb', 0):.2f} GB")
            print(f"  INT8: {memory.get('int8_gb', 0):.2f} GB")
            print(f"  INT4: {memory.get('int4_gb', 0):.2f} GB")
        
        # Architecture details
        arch = result.get('architecture_details', {})
        if 'parameter_distribution' in arch:
            print(f"\nParameter Distribution:")
            for layer_type, info in sorted(arch['parameter_distribution'].items(), 
                                          key=lambda x: x[1]['percentage'], 
                                          reverse=True):
                print(f"  {layer_type}: {info['percentage']:.1f}% ({info['params']:,} params)")
        
        # Save output if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nFull analysis saved to: {output_file}")
    
    elif action == "compare":
        if not compare_with:
            print("Error: --compare-with required for comparison")
            return
        
        model_ids = [model_id] + [m.strip() for m in compare_with.split(",")]
        result = analyzer.compare_models.remote(model_ids)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<40} {'Parameters':<15} {'Memory (GB)':<12} {'Type':<15}")
        print("-"*80)
        
        for model in result['models']:
            name = model['model_id'][:39]
            if model['status'] == 'success':
                params = analyzer._format_number(model.get('total_params', 0))
                memory = model.get('fp16_memory_gb', 0)
                model_type = model.get('model_type', 'unknown')
                print(f"{name:<40} {params:<15} {memory:<12.2f} {model_type:<15}")
            else:
                print(f"{name:<40} {'ERROR':<15} {'N/A':<12} {'N/A':<15}")
        
        print("="*80)
        
        if 'summary' in result:
            print("\nSummary:")
            print(f"  Total Models: {result['summary']['total_models']}")
            print(f"  Successful: {result['summary']['successful']}")
            print(f"  Failed: {result['summary']['failed']}")
            if result['summary']['successful'] > 0:
                print(f"  Largest: {result['summary']['largest_model']}")
                print(f"  Smallest: {result['summary']['smallest_model']}")
                print(f"  Average Parameters: {analyzer._format_number(int(result['summary']['avg_params']))}")
    
    elif action == "export":
        result = analyzer.export_analysis.remote(model_id, output_format)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"Analysis exported to: {output_file}")
        else:
            print(result)
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: analyze, compare, export")


if __name__ == "__main__":
    main()