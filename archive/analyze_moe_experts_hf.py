#!/usr/bin/env python3
"""
Analyze MoE expert importance using HuggingFace Transformers directly.
This version loads the model from HuggingFace and analyzes expert routing patterns.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExpertActivation:
    """Track individual expert activation."""
    layer_idx: int
    expert_idx: int
    routing_weight: float
    token_position: int
    token: str


@dataclass 
class ExpertProfile:
    """Profile for each expert across all layers."""
    expert_id: int
    layer_id: int
    total_activations: int = 0
    total_weight: float = 0.0
    avg_weight: float = 0.0
    token_specialization: Dict[str, int] = field(default_factory=dict)
    position_preference: List[int] = field(default_factory=list)
    
    def update(self, weight: float, token: str, position: int):
        self.total_activations += 1
        self.total_weight += weight
        self.avg_weight = self.total_weight / self.total_activations
        self.token_specialization[token] = self.token_specialization.get(token, 0) + 1
        self.position_preference.append(position)


class GPTOSSExpertAnalyzer:
    """Analyze expert importance in gpt-oss MoE model."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", device: str = None):
        """
        Initialize the analyzer with HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu/mps)
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        self.expert_profiles = {}
        self.activation_history = []
        
        print(f"Initializing analyzer for {model_name} on {self.device}")
        
    def _get_device(self) -> str:
        """Determine best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, use_auth_token: Optional[str] = None):
        """
        Load the model from HuggingFace.
        
        Args:
            use_auth_token: HuggingFace auth token if needed
        """
        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=use_auth_token,
            trust_remote_code=True
        )
        
        print(f"Loading model from {self.model_name}...")
        print("Note: This is a large model (20B parameters) and may take time to download...")
        
        # Load with appropriate precision based on device
        if self.device == "cuda":
            dtype = torch.float16  # Use fp16 for GPU
        else:
            dtype = torch.float32  # Use fp32 for CPU/MPS
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                use_auth_token=use_auth_token,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
            
            # Initialize expert profiles based on model architecture
            self._initialize_expert_profiles()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load with reduced precision...")
            
            # Fallback to 8-bit quantization if available
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    use_auth_token=use_auth_token,
                    trust_remote_code=True
                )
                self.model.eval()
                print("Model loaded with 8-bit quantization")
                self._initialize_expert_profiles()
            except:
                raise RuntimeError("Failed to load model. Please check your setup and available memory.")
    
    def _initialize_expert_profiles(self):
        """Initialize expert profiles based on model architecture."""
        # Inspect model architecture to determine MoE configuration
        num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 24
        
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx] if hasattr(self.model, 'model') else None
            
            # Check if this layer has MoE
            if layer and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                num_experts = len(layer.mlp.experts)
            else:
                num_experts = 16  # Default assumption for gpt-oss
            
            for expert_idx in range(num_experts):
                key = (layer_idx, expert_idx)
                self.expert_profiles[key] = ExpertProfile(
                    expert_id=expert_idx,
                    layer_id=layer_idx
                )
        
        print(f"Initialized {len(self.expert_profiles)} expert profiles")
    
    def _register_hooks(self):
        """Register forward hooks to capture expert routing."""
        self.routing_data = []
        
        def routing_hook(module, input, output, layer_idx):
            """Capture routing decisions from MoE layers."""
            if hasattr(output, 'router_probs') or hasattr(module, 'gate'):
                # Extract routing information based on model architecture
                if hasattr(output, 'router_probs'):
                    router_probs = output.router_probs
                elif hasattr(module, 'gate'):
                    # Calculate routing probabilities from gate
                    gate_output = module.gate(input[0])
                    router_probs = torch.softmax(gate_output, dim=-1)
                else:
                    return
                
                self.routing_data.append({
                    'layer': layer_idx,
                    'router_probs': router_probs.detach().cpu()
                })
        
        # Register hooks on MoE layers
        handles = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and (hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'router')):
                handle = layer.mlp.register_forward_hook(
                    lambda m, i, o, idx=idx: routing_hook(m, i, o, idx)
                )
                handles.append(handle)
        
        return handles
    
    def analyze_text(self, text: str, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Analyze expert activations for given text.
        
        Args:
            text: Input text to analyze
            max_tokens: Maximum tokens to generate
            
        Returns:
            Analysis results including expert activation patterns
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Clear previous routing data
        self.routing_data = []
        
        # Register hooks
        hooks = self._register_hooks()
        
        try:
            with torch.no_grad():
                # Generate with model to trigger expert routing
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Process routing data
                for routing_info in self.routing_data:
                    layer_idx = routing_info['layer']
                    router_probs = routing_info['router_probs']
                    
                    # Get top-k experts for each token
                    k = min(4, router_probs.shape[-1])  # Top-4 experts
                    top_k_values, top_k_indices = torch.topk(router_probs, k, dim=-1)
                    
                    # Update expert profiles
                    for batch_idx in range(top_k_values.shape[0]):
                        for seq_idx in range(min(top_k_values.shape[1], len(input_tokens))):
                            for expert_rank in range(k):
                                expert_idx = top_k_indices[batch_idx, seq_idx, expert_rank].item()
                                weight = top_k_values[batch_idx, seq_idx, expert_rank].item()
                                
                                if weight > 0.01:  # Threshold for significant activation
                                    key = (layer_idx, expert_idx)
                                    if key in self.expert_profiles:
                                        token = input_tokens[seq_idx] if seq_idx < len(input_tokens) else "<generated>"
                                        self.expert_profiles[key].update(
                                            weight=weight,
                                            token=token,
                                            position=seq_idx
                                        )
                                        
                                        # Record activation
                                        self.activation_history.append(
                                            ExpertActivation(
                                                layer_idx=layer_idx,
                                                expert_idx=expert_idx,
                                                routing_weight=weight,
                                                token_position=seq_idx,
                                                token=token
                                            )
                                        )
        
        finally:
            # Remove hooks
            for handle in hooks:
                handle.remove()
        
        # Compile analysis results
        results = self._compile_analysis_results()
        results['input_text'] = text
        results['num_tokens_analyzed'] = len(input_tokens)
        
        return results
    
    def _compile_analysis_results(self) -> Dict[str, Any]:
        """Compile comprehensive analysis results."""
        results = {
            'expert_importance': {},
            'layer_statistics': {},
            'token_specialization': {},
            'activation_patterns': {}
        }
        
        # Calculate expert importance scores
        for key, profile in self.expert_profiles.items():
            layer_idx, expert_idx = key
            
            if profile.total_activations > 0:
                importance_score = (
                    profile.avg_weight * np.log1p(profile.total_activations)
                )
                
                results['expert_importance'][f"L{layer_idx}_E{expert_idx}"] = {
                    'importance_score': float(importance_score),
                    'activation_count': profile.total_activations,
                    'average_weight': float(profile.avg_weight),
                    'total_weight': float(profile.total_weight)
                }
                
                # Layer statistics
                if layer_idx not in results['layer_statistics']:
                    results['layer_statistics'][layer_idx] = {
                        'total_activations': 0,
                        'active_experts': 0,
                        'avg_weight_per_expert': {}
                    }
                
                results['layer_statistics'][layer_idx]['total_activations'] += profile.total_activations
                results['layer_statistics'][layer_idx]['active_experts'] += 1
                results['layer_statistics'][layer_idx]['avg_weight_per_expert'][expert_idx] = float(profile.avg_weight)
                
                # Token specialization
                if len(profile.token_specialization) > 0:
                    top_tokens = sorted(
                        profile.token_specialization.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    results['token_specialization'][f"L{layer_idx}_E{expert_idx}"] = top_tokens
        
        # Sort experts by importance
        results['expert_importance'] = dict(
            sorted(
                results['expert_importance'].items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )
        )
        
        return results
    
    def visualize_expert_importance(self, save_path: str = "expert_importance.png"):
        """
        Create visualization of expert importance across layers.
        
        Args:
            save_path: Path to save the visualization
        """
        # Prepare data for visualization
        importance_matrix = np.zeros((24, 16))  # Assuming 24 layers, 16 experts
        
        for key, profile in self.expert_profiles.items():
            layer_idx, expert_idx = key
            if profile.total_activations > 0:
                importance = profile.avg_weight * np.log1p(profile.total_activations)
                if layer_idx < 24 and expert_idx < 16:
                    importance_matrix[layer_idx, expert_idx] = importance
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            importance_matrix,
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance Score'},
            xticklabels=[f'E{i}' for i in range(16)],
            yticklabels=[f'L{i}' for i in range(24)]
        )
        plt.title('Expert Importance Heatmap Across Layers')
        plt.xlabel('Expert Index')
        plt.ylabel('Layer Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Visualization saved to {save_path}")
    
    def generate_report(self) -> str:
        """Generate comprehensive text report of expert importance."""
        results = self._compile_analysis_results()
        
        report = []
        report.append("=" * 80)
        report.append("GPT-OSS MoE EXPERT IMPORTANCE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Top important experts
        report.append("TOP 10 MOST IMPORTANT EXPERTS:")
        report.append("-" * 40)
        
        for i, (expert_key, scores) in enumerate(list(results['expert_importance'].items())[:10], 1):
            report.append(
                f"{i:2d}. {expert_key:10s} | "
                f"Score: {scores['importance_score']:8.4f} | "
                f"Activations: {scores['activation_count']:5d} | "
                f"Avg Weight: {scores['average_weight']:.4f}"
            )
        
        report.append("")
        
        # Layer statistics
        report.append("LAYER ACTIVATION STATISTICS:")
        report.append("-" * 40)
        
        for layer_idx in sorted(results['layer_statistics'].keys()):
            stats = results['layer_statistics'][layer_idx]
            report.append(
                f"Layer {layer_idx:2d}: "
                f"Total Acts: {stats['total_activations']:6d} | "
                f"Active Experts: {stats['active_experts']:3d}"
            )
        
        report.append("")
        
        # Token specialization insights
        report.append("EXPERT TOKEN SPECIALIZATION (Top 5):")
        report.append("-" * 40)
        
        for expert_key, top_tokens in list(results['token_specialization'].items())[:10]:
            if top_tokens:
                tokens_str = ", ".join([f"{token}({count})" for token, count in top_tokens[:3]])
                report.append(f"{expert_key}: {tokens_str}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function."""
    # Test texts covering different domains
    test_texts = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The quantum mechanical wave function describes the probability amplitude",
        "In logical reasoning, modus ponens states that if P implies Q and P is true",
        "The French Revolution began in 1789 and transformed European politics",
        "Once upon a time in a digital realm, an AI learned to dream"
    ]
    
    # Initialize analyzer
    print("Initializing GPT-OSS Expert Analyzer...")
    analyzer = GPTOSSExpertAnalyzer(model_name="openai/gpt-oss-20b")
    
    # Check if HuggingFace token is needed
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print("Using HuggingFace token from environment")
    
    try:
        # Load model
        print("\nLoading model (this may take several minutes)...")
        analyzer.load_model(use_auth_token=hf_token)
        
        # Analyze test texts
        print("\nAnalyzing test texts...")
        all_results = []
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n[{i}/{len(test_texts)}] Analyzing: {text[:50]}...")
            results = analyzer.analyze_text(text, max_tokens=30)
            all_results.append(results)
            
            # Print brief summary
            num_important = len([
                k for k, v in results['expert_importance'].items()
                if v['importance_score'] > 0.1
            ])
            print(f"  → Found {num_important} highly active experts")
        
        # Generate report
        print("\nGenerating analysis report...")
        report = analyzer.generate_report()
        print(report)
        
        # Save results
        with open("expert_analysis_results.json", "w") as f:
            json.dump({
                'test_texts': test_texts,
                'analysis_results': all_results,
                'summary': results
            }, f, indent=2)
        
        print("\nResults saved to expert_analysis_results.json")
        
        # Create visualization
        print("Creating visualization...")
        analyzer.visualize_expert_importance()
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("\nNote: This script requires:")
        print("1. Sufficient memory to load the 20B parameter model")
        print("2. HuggingFace access token if the model is gated")
        print("3. PyTorch with CUDA support for GPU acceleration (recommended)")
        print("\nTo use with HuggingFace token:")
        print("export HF_TOKEN=your_token_here")


if __name__ == "__main__":
    main()