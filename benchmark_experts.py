#!/usr/bin/env python3
"""
Benchmark and evaluate MoE expert importance through performance analysis.
This script evaluates the contribution of different experts by:
1. Running inference with all experts
2. Selectively disabling experts
3. Measuring performance degradation
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


@dataclass
class BenchmarkResult:
    """Results from expert benchmark test."""
    disabled_experts: List[Tuple[int, int]]  # (layer, expert) pairs
    perplexity: float
    inference_time: float
    accuracy: float
    token_throughput: float
    memory_usage: float


class ExpertImportanceBenchmark:
    """Benchmark expert importance through ablation studies."""
    
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize benchmark with model.
        
        Args:
            model: The loaded MoE model
            tokenizer: Model tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.baseline_results = None
        self.ablation_results = []
        
    def prepare_evaluation_data(self) -> List[Dict]:
        """Prepare diverse evaluation dataset."""
        eval_samples = [
            # Code understanding
            {
                "input": "def quicksort(arr): if len(arr) <= 1: return arr",
                "target": "pivot = arr[len(arr) // 2]",
                "category": "code"
            },
            # Mathematical reasoning
            {
                "input": "If x + 2y = 10 and x - y = 1, then",
                "target": "x = 4 and y = 3",
                "category": "math"
            },
            # Factual knowledge
            {
                "input": "The capital of Japan is",
                "target": "Tokyo",
                "category": "factual"
            },
            # Logical reasoning
            {
                "input": "All mammals are warm-blooded. Whales are mammals. Therefore,",
                "target": "whales are warm-blooded",
                "category": "reasoning"
            },
            # Creative writing
            {
                "input": "The old lighthouse stood alone on the cliff,",
                "target": "its beacon cutting through the foggy night",
                "category": "creative"
            }
        ]
        
        return eval_samples * 10  # Repeat for more robust evaluation
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def measure_inference_speed(self, text: str, num_tokens: int = 50) -> Tuple[float, float]:
        """
        Measure inference speed.
        
        Returns:
            Tuple of (total_time, tokens_per_second)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
        # Actual measurement
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False
            )
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        tokens_per_second = num_tokens / total_time
        
        return total_time, tokens_per_second
    
    def measure_memory_usage(self) -> float:
        """Measure current GPU memory usage in MB."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def disable_experts(self, experts_to_disable: List[Tuple[int, int]]):
        """
        Disable specific experts by zeroing their weights.
        
        Args:
            experts_to_disable: List of (layer_idx, expert_idx) tuples
        """
        for layer_idx, expert_idx in experts_to_disable:
            if hasattr(self.model, 'model'):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                    # Zero out expert weights
                    with torch.no_grad():
                        for param in layer.mlp.experts[expert_idx].parameters():
                            param.mul_(0)
    
    def restore_experts(self, original_state: Dict):
        """Restore experts to original state."""
        self.model.load_state_dict(original_state)
    
    def run_baseline_benchmark(self) -> BenchmarkResult:
        """Run benchmark with all experts enabled."""
        print("Running baseline benchmark with all experts...")
        
        eval_data = self.prepare_evaluation_data()
        total_perplexity = 0
        total_time = 0
        correct_predictions = 0
        
        for sample in tqdm(eval_data, desc="Baseline evaluation"):
            # Perplexity
            perplexity = self.calculate_perplexity(sample["input"] + " " + sample["target"])
            total_perplexity += perplexity
            
            # Inference speed
            inf_time, throughput = self.measure_inference_speed(sample["input"], num_tokens=20)
            total_time += inf_time
            
            # Simple accuracy check
            inputs = self.tokenizer(sample["input"], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Check if key words from target appear in generation
                target_words = sample["target"].lower().split()
                generated_words = generated.lower().split()
                match_score = sum(1 for word in target_words if word in generated_words) / len(target_words)
                if match_score > 0.3:
                    correct_predictions += 1
        
        avg_perplexity = total_perplexity / len(eval_data)
        accuracy = correct_predictions / len(eval_data)
        avg_throughput = len(eval_data) * 20 / total_time  # tokens per second
        memory = self.measure_memory_usage()
        
        self.baseline_results = BenchmarkResult(
            disabled_experts=[],
            perplexity=avg_perplexity,
            inference_time=total_time,
            accuracy=accuracy,
            token_throughput=avg_throughput,
            memory_usage=memory
        )
        
        print(f"Baseline - Perplexity: {avg_perplexity:.2f}, Accuracy: {accuracy:.2%}, "
              f"Throughput: {avg_throughput:.1f} tok/s")
        
        return self.baseline_results
    
    def run_ablation_study(self, num_experts_to_test: int = 5) -> List[BenchmarkResult]:
        """
        Run ablation study by disabling experts.
        
        Args:
            num_experts_to_test: Number of expert combinations to test
        """
        print(f"\nRunning ablation study on {num_experts_to_test} expert combinations...")
        
        # Save original model state
        original_state = self.model.state_dict()
        
        # Identify experts to test (simplified - test individual experts)
        experts_to_test = [
            [(0, 0)],  # Disable first expert in first layer
            [(0, 1)],  # Disable second expert in first layer
            [(12, 0)],  # Disable first expert in middle layer
            [(23, 0)],  # Disable first expert in last layer
            [(0, 0), (0, 1)],  # Disable multiple experts
        ][:num_experts_to_test]
        
        results = []
        
        for experts_to_disable in experts_to_test:
            print(f"\nTesting with experts disabled: {experts_to_disable}")
            
            # Disable experts
            self.disable_experts(experts_to_disable)
            
            # Run evaluation
            eval_data = self.prepare_evaluation_data()[:20]  # Smaller subset for ablation
            total_perplexity = 0
            total_time = 0
            correct_predictions = 0
            
            for sample in eval_data:
                perplexity = self.calculate_perplexity(sample["input"] + " " + sample["target"])
                total_perplexity += perplexity
                
                inf_time, _ = self.measure_inference_speed(sample["input"], num_tokens=10)
                total_time += inf_time
                
                # Quick accuracy check
                inputs = self.tokenizer(sample["input"], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                    generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    if any(word in generated.lower() for word in sample["target"].lower().split()[:2]):
                        correct_predictions += 1
            
            result = BenchmarkResult(
                disabled_experts=experts_to_disable,
                perplexity=total_perplexity / len(eval_data),
                inference_time=total_time,
                accuracy=correct_predictions / len(eval_data),
                token_throughput=len(eval_data) * 10 / total_time,
                memory_usage=self.measure_memory_usage()
            )
            
            results.append(result)
            self.ablation_results.append(result)
            
            print(f"  Perplexity: {result.perplexity:.2f} "
                  f"(+{((result.perplexity/self.baseline_results.perplexity - 1) * 100):.1f}%), "
                  f"Accuracy: {result.accuracy:.2%}")
            
            # Restore model
            self.restore_experts(original_state)
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return results
    
    def calculate_expert_importance(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate importance score for each tested expert.
        
        Returns:
            Dictionary mapping (layer, expert) to importance score
        """
        if not self.baseline_results or not self.ablation_results:
            raise ValueError("Must run baseline and ablation studies first")
        
        importance_scores = {}
        
        for result in self.ablation_results:
            for layer_idx, expert_idx in result.disabled_experts:
                # Calculate degradation when this expert is disabled
                perplexity_degradation = (result.perplexity / self.baseline_results.perplexity - 1)
                accuracy_degradation = (self.baseline_results.accuracy - result.accuracy)
                speed_degradation = (self.baseline_results.token_throughput / result.token_throughput - 1)
                
                # Combined importance score
                importance = (
                    0.4 * perplexity_degradation +
                    0.4 * accuracy_degradation +
                    0.2 * speed_degradation
                )
                
                importance_scores[(layer_idx, expert_idx)] = max(0, importance)
        
        return importance_scores
    
    def visualize_results(self):
        """Create visualization of benchmark results."""
        if not self.ablation_results:
            print("No ablation results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Perplexity comparison
        ax = axes[0, 0]
        labels = ["Baseline"] + [str(r.disabled_experts) for r in self.ablation_results]
        perplexities = [self.baseline_results.perplexity] + [r.perplexity for r in self.ablation_results]
        ax.bar(range(len(labels)), perplexities)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity Impact of Disabling Experts")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Accuracy comparison
        ax = axes[0, 1]
        accuracies = [self.baseline_results.accuracy] + [r.accuracy for r in self.ablation_results]
        ax.bar(range(len(labels)), accuracies)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Impact of Disabling Experts")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Throughput comparison
        ax = axes[1, 0]
        throughputs = [self.baseline_results.token_throughput] + [r.token_throughput for r in self.ablation_results]
        ax.bar(range(len(labels)), throughputs)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Tokens/Second")
        ax.set_title("Throughput Impact of Disabling Experts")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Expert importance scores
        ax = axes[1, 1]
        importance_scores = self.calculate_expert_importance()
        if importance_scores:
            experts = list(importance_scores.keys())
            scores = list(importance_scores.values())
            ax.bar(range(len(experts)), scores)
            ax.set_xlabel("Expert (Layer, Index)")
            ax.set_ylabel("Importance Score")
            ax.set_title("Expert Importance Scores")
            ax.set_xticklabels([f"L{l},E{e}" for l, e in experts], rotation=45)
        
        plt.tight_layout()
        plt.savefig("expert_benchmark_results.png", dpi=150)
        plt.close()
        
        print("Visualization saved to expert_benchmark_results.png")
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.baseline_results:
            return "No benchmark results available. Run baseline benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("MoE EXPERT IMPORTANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Baseline results
        report.append("BASELINE PERFORMANCE (All Experts Enabled):")
        report.append("-" * 40)
        report.append(f"  Perplexity: {self.baseline_results.perplexity:.2f}")
        report.append(f"  Accuracy: {self.baseline_results.accuracy:.2%}")
        report.append(f"  Throughput: {self.baseline_results.token_throughput:.1f} tokens/sec")
        report.append(f"  Memory Usage: {self.baseline_results.memory_usage:.1f} MB")
        report.append("")
        
        if self.ablation_results:
            # Ablation results
            report.append("ABLATION STUDY RESULTS:")
            report.append("-" * 40)
            
            for result in self.ablation_results:
                expert_str = ", ".join([f"L{l}E{e}" for l, e in result.disabled_experts])
                perp_change = ((result.perplexity / self.baseline_results.perplexity - 1) * 100)
                acc_change = ((result.accuracy - self.baseline_results.accuracy) * 100)
                
                report.append(f"\nDisabled: {expert_str}")
                report.append(f"  Perplexity: {result.perplexity:.2f} ({perp_change:+.1f}%)")
                report.append(f"  Accuracy: {result.accuracy:.2%} ({acc_change:+.1f}%)")
                report.append(f"  Throughput: {result.token_throughput:.1f} tokens/sec")
            
            report.append("")
            
            # Expert importance ranking
            importance_scores = self.calculate_expert_importance()
            if importance_scores:
                report.append("EXPERT IMPORTANCE RANKING:")
                report.append("-" * 40)
                
                sorted_experts = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, ((layer, expert), score) in enumerate(sorted_experts[:10], 1):
                    report.append(f"{rank:2d}. Layer {layer:2d}, Expert {expert:2d}: {score:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def simplified_benchmark():
    """Run a simplified benchmark without loading the full model."""
    print("Running simplified MoE expert importance simulation...")
    
    # Simulated benchmark results
    class MockModel:
        def __init__(self):
            self.num_layers = 24
            self.num_experts = 16
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.randint(0, 50000, (1, len(text.split())))}
        
        def decode(self, ids, **kwargs):
            return "simulated output text"
    
    # Create mock benchmark
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    
    print("\nSimulating expert importance analysis...")
    
    # Simulate importance scores for different experts
    np.random.seed(42)
    importance_data = {}
    
    for layer in range(24):
        for expert in range(16):
            # Simulate that some experts are more important
            if layer < 8:  # Early layers
                base_importance = 0.3 if expert < 4 else 0.1
            elif layer < 16:  # Middle layers
                base_importance = 0.5 if expert in [2, 5, 8, 11] else 0.2
            else:  # Late layers
                base_importance = 0.4 if expert % 3 == 0 else 0.15
            
            importance = base_importance + np.random.normal(0, 0.05)
            importance_data[(layer, expert)] = max(0, importance)
    
    # Generate report
    sorted_experts = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 60)
    print("SIMULATED EXPERT IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("\nTOP 20 MOST IMPORTANT EXPERTS:")
    print("-" * 40)
    print(f"{'Rank':<6} {'Layer':<8} {'Expert':<8} {'Importance':<12}")
    print("-" * 40)
    
    for rank, ((layer, expert), score) in enumerate(sorted_experts[:20], 1):
        print(f"{rank:<6} {layer:<8} {expert:<8} {score:<12.4f}")
    
    print("\nLAYER-WISE IMPORTANCE SUMMARY:")
    print("-" * 40)
    
    layer_importance = {}
    for (layer, expert), score in importance_data.items():
        if layer not in layer_importance:
            layer_importance[layer] = []
        layer_importance[layer].append(score)
    
    for layer in sorted(layer_importance.keys()):
        scores = layer_importance[layer]
        avg_importance = np.mean(scores)
        max_importance = np.max(scores)
        print(f"Layer {layer:2d}: Avg={avg_importance:.3f}, Max={max_importance:.3f}")
    
    print("\nEXPERT SPECIALIZATION PATTERNS:")
    print("-" * 40)
    print("Early layers (0-7):   Focus on basic token processing")
    print("Middle layers (8-15): Complex reasoning and associations")
    print("Late layers (16-23):  Output generation and refinement")
    
    print("\n" + "=" * 60)
    
    # Save results
    with open("expert_importance_simulation.json", "w") as f:
        json.dump({
            "importance_scores": {f"L{l}_E{e}": float(s) for (l, e), s in sorted_experts[:50]},
            "layer_summary": {
                str(l): {"avg": float(np.mean(scores)), "max": float(np.max(scores))}
                for l, scores in [(k, layer_importance[k]) for k in sorted(layer_importance.keys())]
            }
        }, f, indent=2)
    
    print("\nResults saved to expert_importance_simulation.json")


if __name__ == "__main__":
    # Run simplified benchmark (doesn't require loading the full model)
    simplified_benchmark()