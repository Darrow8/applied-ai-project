#!/usr/bin/env python3
"""
Analyze the importance of different experts in the MoE architecture of gpt-oss 20B model.
This script evaluates expert importance through multiple metrics including:
- Activation frequency across different input types
- Average contribution magnitude to outputs
- Token routing patterns
- Expert specialization analysis
"""

import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
import aiohttp
import modal

# Import the base configuration from the main script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from gpt-oss.py (handle hyphen in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("gpt_oss", "gpt-oss.py")
gpt_oss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt_oss_module)

vllm_image = gpt_oss_module.vllm_image
MODEL_NAME = gpt_oss_module.MODEL_NAME
MODEL_REVISION = gpt_oss_module.MODEL_REVISION
hf_cache_vol = gpt_oss_module.hf_cache_vol
vllm_cache_vol = gpt_oss_module.vllm_cache_vol
FAST_BOOT = gpt_oss_module.FAST_BOOT
MAX_INPUTS = gpt_oss_module.MAX_INPUTS
CUDA_GRAPH_CAPTURE_SIZES = gpt_oss_module.CUDA_GRAPH_CAPTURE_SIZES
N_GPU = gpt_oss_module.N_GPU
MINUTES = gpt_oss_module.MINUTES
VLLM_PORT = gpt_oss_module.VLLM_PORT


@dataclass
class ExpertMetrics:
    """Metrics for individual expert evaluation."""
    expert_id: int
    activation_count: int = 0
    total_contribution: float = 0.0
    avg_contribution: float = 0.0
    token_types: Dict[str, int] = field(default_factory=dict)
    input_categories: Dict[str, int] = field(default_factory=dict)
    layer_activations: Dict[int, int] = field(default_factory=dict)
    routing_weights: List[float] = field(default_factory=list)
    
    def update(self, contribution: float, token_type: str = None, 
               input_category: str = None, layer: int = None, routing_weight: float = None):
        """Update expert metrics with new activation data."""
        self.activation_count += 1
        self.total_contribution += contribution
        self.avg_contribution = self.total_contribution / self.activation_count
        
        if token_type:
            self.token_types[token_type] = self.token_types.get(token_type, 0) + 1
        if input_category:
            self.input_categories[input_category] = self.input_categories.get(input_category, 0) + 1
        if layer is not None:
            self.layer_activations[layer] = self.layer_activations.get(layer, 0) + 1
        if routing_weight is not None:
            self.routing_weights.append(routing_weight)


class MoEExpertAnalyzer:
    """Analyzer for MoE expert importance evaluation."""
    
    def __init__(self, num_experts: int = 16, num_layers: int = 24):
        """
        Initialize the analyzer.
        
        Args:
            num_experts: Number of experts in the MoE architecture
            num_layers: Number of MoE layers in the model
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.expert_metrics = {
            i: ExpertMetrics(expert_id=i) for i in range(num_experts)
        }
        self.global_stats = {
            'total_tokens_processed': 0,
            'total_forward_passes': 0,
            'category_distribution': defaultdict(int)
        }
        
    def categorize_input(self, text: str) -> str:
        """Categorize input text for analysis."""
        categories = {
            'code': ['def ', 'class ', 'import ', 'function', 'var ', 'const ', '{}', '[]', '()'],
            'math': ['equation', 'calculate', 'solve', 'derivative', 'integral', 'matrix', 'vector'],
            'reasoning': ['therefore', 'because', 'if', 'then', 'logic', 'conclude', 'premise'],
            'factual': ['what', 'when', 'where', 'who', 'which', 'define', 'explain'],
            'creative': ['imagine', 'story', 'poem', 'create', 'design', 'innovate'],
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'general'
    
    def analyze_token_type(self, token: str) -> str:
        """Determine the type of token for analysis."""
        if token.isdigit():
            return 'numeric'
        elif token.isalpha():
            return 'alphabetic'
        elif any(c in token for c in '+-*/=<>!&|'):
            return 'operator'
        elif any(c in token for c in '()[]{}'):
            return 'bracket'
        elif any(c in token for c in '.,;:'):
            return 'punctuation'
        else:
            return 'mixed'
    
    async def track_expert_activations(self, model_output: Dict[str, Any], 
                                      input_text: str) -> None:
        """
        Track expert activations from model output.
        
        Args:
            model_output: Output from the model including expert routing information
            input_text: Original input text for categorization
        """
        input_category = self.categorize_input(input_text)
        self.global_stats['category_distribution'][input_category] += 1
        
        # Parse expert routing information if available
        if 'expert_routing' in model_output:
            routing_info = model_output['expert_routing']
            
            for layer_idx, layer_routing in enumerate(routing_info.get('layers', [])):
                selected_experts = layer_routing.get('selected_experts', [])
                routing_weights = layer_routing.get('routing_weights', [])
                
                for expert_idx, weight in zip(selected_experts, routing_weights):
                    if expert_idx < self.num_experts:
                        self.expert_metrics[expert_idx].update(
                            contribution=weight,
                            input_category=input_category,
                            layer=layer_idx,
                            routing_weight=weight
                        )
        
        self.global_stats['total_forward_passes'] += 1
    
    def calculate_importance_scores(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate comprehensive importance scores for each expert.
        
        Returns:
            Dictionary mapping expert IDs to their importance metrics
        """
        importance_scores = {}
        
        total_activations = sum(
            m.activation_count for m in self.expert_metrics.values()
        )
        
        for expert_id, metrics in self.expert_metrics.items():
            if metrics.activation_count == 0:
                importance_scores[expert_id] = {
                    'activation_frequency': 0.0,
                    'contribution_magnitude': 0.0,
                    'specialization_score': 0.0,
                    'routing_consistency': 0.0,
                    'overall_importance': 0.0
                }
                continue
            
            # Activation frequency score
            activation_freq = metrics.activation_count / max(total_activations, 1)
            
            # Average contribution magnitude
            contribution_mag = metrics.avg_contribution
            
            # Specialization score (entropy-based)
            category_dist = np.array(list(metrics.input_categories.values()))
            if len(category_dist) > 0:
                category_probs = category_dist / category_dist.sum()
                entropy = -np.sum(category_probs * np.log2(category_probs + 1e-10))
                max_entropy = np.log2(len(metrics.input_categories))
                specialization = 1 - (entropy / max(max_entropy, 1))
            else:
                specialization = 0.0
            
            # Routing consistency (std deviation of routing weights)
            if len(metrics.routing_weights) > 1:
                routing_std = np.std(metrics.routing_weights)
                routing_consistency = 1 / (1 + routing_std)  # Lower std = higher consistency
            else:
                routing_consistency = 0.0
            
            # Overall importance score (weighted combination)
            overall_importance = (
                0.3 * activation_freq +
                0.3 * contribution_mag +
                0.2 * specialization +
                0.2 * routing_consistency
            )
            
            importance_scores[expert_id] = {
                'activation_frequency': activation_freq,
                'contribution_magnitude': contribution_mag,
                'specialization_score': specialization,
                'routing_consistency': routing_consistency,
                'overall_importance': overall_importance
            }
        
        return importance_scores
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        importance_scores = self.calculate_importance_scores()
        
        # Sort experts by overall importance
        sorted_experts = sorted(
            importance_scores.items(),
            key=lambda x: x[1]['overall_importance'],
            reverse=True
        )
        
        report = []
        report.append("=" * 80)
        report.append("MoE EXPERT IMPORTANCE ANALYSIS REPORT")
        report.append(f"Model: {MODEL_NAME}")
        report.append(f"Analysis Date: {datetime.now(timezone.utc)}")
        report.append("=" * 80)
        report.append("")
        
        # Global statistics
        report.append("GLOBAL STATISTICS:")
        report.append(f"  Total forward passes: {self.global_stats['total_forward_passes']}")
        report.append(f"  Total tokens processed: {self.global_stats['total_tokens_processed']}")
        report.append("")
        
        # Category distribution
        report.append("INPUT CATEGORY DISTRIBUTION:")
        for category, count in self.global_stats['category_distribution'].items():
            percentage = (count / max(self.global_stats['total_forward_passes'], 1)) * 100
            report.append(f"  {category:12s}: {count:5d} ({percentage:.1f}%)")
        report.append("")
        
        # Expert rankings
        report.append("EXPERT IMPORTANCE RANKINGS:")
        report.append("-" * 80)
        report.append(f"{'Rank':<6} {'Expert':<8} {'Overall':<10} {'Activation':<12} "
                     f"{'Contribution':<14} {'Specialization':<16} {'Consistency':<12}")
        report.append("-" * 80)
        
        for rank, (expert_id, scores) in enumerate(sorted_experts, 1):
            metrics = self.expert_metrics[expert_id]
            report.append(
                f"{rank:<6} {expert_id:<8} "
                f"{scores['overall_importance']:<10.4f} "
                f"{scores['activation_frequency']:<12.4f} "
                f"{scores['contribution_magnitude']:<14.4f} "
                f"{scores['specialization_score']:<16.4f} "
                f"{scores['routing_consistency']:<12.4f}"
            )
        
        report.append("")
        
        # Expert specializations
        report.append("EXPERT SPECIALIZATIONS:")
        report.append("-" * 80)
        
        for expert_id in range(min(5, self.num_experts)):  # Top 5 experts
            metrics = self.expert_metrics[expert_id]
            if metrics.activation_count > 0:
                report.append(f"\nExpert {expert_id}:")
                report.append(f"  Total activations: {metrics.activation_count}")
                
                if metrics.input_categories:
                    report.append("  Specialized in:")
                    sorted_categories = sorted(
                        metrics.input_categories.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for category, count in sorted_categories[:3]:
                        percentage = (count / metrics.activation_count) * 100
                        report.append(f"    - {category}: {percentage:.1f}%")
                
                if metrics.layer_activations:
                    most_active_layer = max(
                        metrics.layer_activations.items(),
                        key=lambda x: x[1]
                    )
                    report.append(f"  Most active in layer: {most_active_layer[0]}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


# Test prompts for different categories
TEST_PROMPTS = [
    # Code-related
    ("Write a Python function to calculate fibonacci numbers", "code"),
    ("Explain how a binary search tree works with code examples", "code"),
    
    # Math-related
    ("Solve the equation x^2 + 5x + 6 = 0", "math"),
    ("Calculate the derivative of f(x) = 3x^3 + 2x^2 - 5x + 1", "math"),
    
    # Reasoning
    ("If all birds can fly and penguins are birds, can penguins fly? Explain the logical fallacy", "reasoning"),
    ("What are the implications of Gödel's incompleteness theorems?", "reasoning"),
    
    # Factual
    ("What is the capital of France?", "factual"),
    ("Explain photosynthesis in simple terms", "factual"),
    
    # Creative
    ("Write a haiku about artificial intelligence", "creative"),
    ("Create a short story about a robot learning to paint", "creative"),
]


app = modal.App("moe-expert-analyzer")


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
async def analyze_experts_with_hooks():
    """
    Run expert analysis with model hooks to capture routing information.
    This requires modifying the vLLM server to expose expert routing data.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model for expert analysis...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    analyzer = MoEExpertAnalyzer(num_experts=16, num_layers=24)
    
    # Hook to capture expert routing decisions
    routing_info = {}
    
    def capture_routing_hook(module, input, output, layer_idx):
        """Hook to capture MoE routing decisions."""
        if hasattr(output, 'router_logits'):
            routing_info[f'layer_{layer_idx}'] = {
                'router_logits': output.router_logits.detach().cpu().numpy().tolist(),
                'selected_experts': output.selected_experts.detach().cpu().numpy().tolist(),
                'routing_weights': output.routing_weights.detach().cpu().numpy().tolist()
            }
        return output
    
    # Register hooks on MoE layers
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            layer.mlp.register_forward_hook(
                lambda m, inp, out, idx=i: capture_routing_hook(m, inp, out, idx)
            )
    
    print("Running analysis on test prompts...")
    
    for prompt_text, expected_category in TEST_PROMPTS:
        print(f"\nAnalyzing: {prompt_text[:50]}...")
        
        # Tokenize and generate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
        
        # Process routing information
        if routing_info:
            mock_output = {
                'expert_routing': {
                    'layers': [
                        routing_info.get(f'layer_{i}', {})
                        for i in range(len(model.model.layers))
                    ]
                }
            }
            
            await analyzer.track_expert_activations(mock_output, prompt_text)
        
        routing_info.clear()
    
    # Generate and save report
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Save detailed metrics
    with open("/tmp/expert_analysis.json", "w") as f:
        analysis_data = {
            'importance_scores': analyzer.calculate_importance_scores(),
            'expert_metrics': {
                i: {
                    'activation_count': m.activation_count,
                    'avg_contribution': m.avg_contribution,
                    'input_categories': m.input_categories,
                    'layer_activations': m.layer_activations
                }
                for i, m in analyzer.expert_metrics.items()
            },
            'global_stats': dict(analyzer.global_stats)
        }
        json.dump(analysis_data, f, indent=2)
    
    print("\nDetailed analysis saved to /tmp/expert_analysis.json")
    
    return report


@app.local_entrypoint()
async def main():
    """Run the expert importance analysis."""
    print("Starting MoE Expert Importance Analysis...")
    print("=" * 80)
    
    # Run lightweight analysis using inference API
    analyzer = MoEExpertAnalyzer(num_experts=16, num_layers=24)
    
    # Simulate analysis with multiple test inputs
    print("Running simulated analysis with test prompts...")
    
    for prompt_text, category in TEST_PROMPTS:
        print(f"Processing: {prompt_text[:50]}...")
        
        # Simulate expert activations (in real scenario, these would come from model)
        mock_output = simulate_expert_routing(prompt_text, category)
        await analyzer.track_expert_activations(mock_output, prompt_text)
    
    # Generate report
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Save the report
    with open("expert_importance_report.txt", "w") as f:
        f.write(report)
    
    print("\nReport saved to expert_importance_report.txt")
    
    # For full analysis with actual model hooks, uncomment:
    # print("\nRunning full analysis with model hooks...")
    # full_report = await analyze_experts_with_hooks.remote()


def simulate_expert_routing(text: str, category: str) -> Dict[str, Any]:
    """
    Simulate expert routing for demonstration purposes.
    In production, this would come from actual model inference.
    """
    np.random.seed(hash(text) % 2**32)
    
    # Simulate different routing patterns based on category
    routing_patterns = {
        'code': [0, 1, 4, 7, 8, 12],  # Experts specialized in code
        'math': [2, 3, 5, 9, 11, 14],  # Math specialists
        'reasoning': [1, 3, 6, 10, 13, 15],  # Reasoning specialists
        'factual': [0, 2, 4, 8, 11, 13],  # Factual knowledge
        'creative': [5, 6, 7, 9, 10, 12],  # Creative tasks
        'general': list(range(16))  # All experts
    }
    
    preferred_experts = routing_patterns.get(category, routing_patterns['general'])
    
    layers = []
    for layer_idx in range(24):  # 24 layers
        # Select 2-4 experts per layer (top-k routing)
        k = np.random.randint(2, 5)
        
        # Bias selection towards preferred experts
        all_experts = list(range(16))
        weights = [3.0 if e in preferred_experts else 1.0 for e in all_experts]
        weights = np.array(weights) / sum(weights)
        
        selected = np.random.choice(all_experts, size=k, replace=False, p=weights)
        routing_weights = np.random.dirichlet(np.ones(k)) # Normalize weights
        
        layers.append({
            'selected_experts': selected.tolist(),
            'routing_weights': routing_weights.tolist()
        })
    
    return {'expert_routing': {'layers': layers}}


if __name__ == "__main__":
    asyncio.run(main())