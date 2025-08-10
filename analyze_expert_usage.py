#!/usr/bin/env python3
"""
Analyze which experts are used in GPT-OSS for different tasks.
This script tracks expert activation patterns across various prompts and task types.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter

import modal
from modal import App, Image, method, enter

# Define the Modal app
app = App("gpt-oss-expert-usage-analyzer")

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
        "matplotlib",
        "seaborn",
        "pandas",
    )
)

# Create volume for model caching
volume = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100",  # Use A100 for faster processing
    volumes={"/cache": volume},
    timeout=3600,
    scaledown_window=60,
)
class ExpertUsageAnalyzer:
    """Analyze expert activation patterns in GPT-OSS models."""
    
    @enter()
    def setup(self):
        """Initialize environment and set cache directories."""
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
        os.environ["TORCH_HOME"] = "/cache/torch"
        
        # Create cache directories
        for dir_path in ["/cache/huggingface", "/cache/torch", "/cache/results"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("Environment setup complete")
    
    @method()
    def analyze_expert_routing(
        self,
        model_id: str = "openai/gpt-oss-20b",
        task_prompts: Optional[Dict[str, List[str]]] = None,
        max_length: int = 100,
        temperature: float = 0.0,  # Use 0 for deterministic routing
        track_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze which experts are activated for different tasks.
        
        Args:
            model_id: Hugging Face model ID
            task_prompts: Dictionary mapping task categories to prompts
            max_length: Maximum generation length
            temperature: Sampling temperature (0 for deterministic)
            track_layers: Whether to track per-layer expert usage
        
        Returns:
            Dictionary with expert usage statistics
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Default task prompts covering different domains
        if task_prompts is None:
            task_prompts = {
                "coding": [
                    "def fibonacci(n):",
                    "class BinaryTree:",
                    "import numpy as np\n# Calculate matrix multiplication",
                    "// Write a function to reverse a linked list in C++",
                    "SELECT * FROM users WHERE",
                ],
                "math": [
                    "Solve for x: 2x^2 + 5x - 3 = 0",
                    "The derivative of sin(x) * cos(x) is",
                    "Calculate the integral of 1/x from 1 to e:",
                    "If a triangle has sides 3, 4, and 5, its area is",
                    "The limit as x approaches 0 of sin(x)/x equals",
                ],
                "science": [
                    "The process of photosynthesis converts",
                    "Newton's second law states that F =",
                    "DNA replication occurs during the",
                    "The speed of light in vacuum is approximately",
                    "Chemical formula for sulfuric acid is",
                ],
                "creative": [
                    "Once upon a time in a distant galaxy,",
                    "The old mansion stood silent, its windows",
                    "She picked up the mysterious letter and read:",
                    "The sunset painted the sky in shades of",
                    "In the year 2150, humanity discovered",
                ],
                "factual": [
                    "The capital of France is",
                    "World War II ended in the year",
                    "The largest planet in our solar system is",
                    "The author of '1984' is",
                    "The chemical symbol for gold is",
                ],
                "reasoning": [
                    "If all roses are flowers and some flowers fade quickly, then",
                    "John is taller than Mary. Mary is taller than Sue. Therefore,",
                    "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
                    "A store offers 20% off. If an item costs $50 after discount, the original price was",
                    "Three friends split a bill equally. If each pays $15, the total bill was",
                ],
            }
        
        print(f"Loading model: {model_id}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/cache/huggingface")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/cache/huggingface",
        )
        
        # Hook to track expert routing
        expert_activations = defaultdict(lambda: defaultdict(list))
        layer_expert_counts = defaultdict(lambda: defaultdict(Counter))
        
        def hook_fn(name, task_type, prompt_idx):
            def fn(module, input, output):
                if hasattr(output, 'router_logits') or hasattr(module, 'gate'):
                    # Extract routing decisions
                    if hasattr(output, 'router_logits'):
                        router_logits = output.router_logits
                    else:
                        # For different MoE implementations
                        router_logits = module.gate(input[0])
                    
                    # Get top-k experts
                    topk_values, topk_indices = torch.topk(router_logits, k=min(2, router_logits.size(-1)))
                    
                    # Store expert indices
                    experts_used = topk_indices.cpu().numpy().flatten().tolist()
                    expert_activations[task_type][f"{prompt_idx}_{name}"].extend(experts_used)
                    
                    # Count per layer if tracking
                    if track_layers:
                        layer_name = name.split('.')[2] if '.' in name else name  # Extract layer number
                        for expert_idx in experts_used:
                            layer_expert_counts[task_type][layer_name][expert_idx] += 1
            
            return fn
        
        # Register hooks for MoE layers
        hooks = []
        for name, module in model.named_modules():
            if 'experts' in name.lower() or 'moe' in name.lower() or 'gate' in name.lower():
                for task_type in task_prompts:
                    for prompt_idx in range(len(task_prompts[task_type])):
                        hook = module.register_forward_hook(hook_fn(name, task_type, prompt_idx))
                        hooks.append(hook)
        
        # Analyze each task category
        results = {
            'task_analysis': {},
            'expert_specialization': {},
            'cross_task_similarity': {},
            'layer_analysis': {} if track_layers else None,
        }
        
        all_expert_usage = defaultdict(Counter)
        
        for task_type, prompts in task_prompts.items():
            print(f"\nAnalyzing {task_type} tasks...")
            task_expert_usage = Counter()
            task_outputs = []
            
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                task_outputs.append(generated_text)
                
                # Aggregate expert usage for this prompt
                for key, experts in expert_activations[task_type].items():
                    if key.startswith(f"{prompts.index(prompt)}_"):
                        task_expert_usage.update(experts)
                        all_expert_usage[task_type].update(experts)
            
            # Calculate statistics for this task
            total_activations = sum(task_expert_usage.values())
            expert_distribution = {
                int(expert): count / total_activations 
                for expert, count in task_expert_usage.most_common()
            } if total_activations > 0 else {}
            
            results['task_analysis'][task_type] = {
                'total_prompts': len(prompts),
                'unique_experts_used': len(task_expert_usage),
                'top_5_experts': dict(task_expert_usage.most_common(5)),
                'expert_distribution': expert_distribution,
                'sample_outputs': task_outputs[:3],  # Store first 3 outputs as samples
            }
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze expert specialization
        expert_task_affinity = defaultdict(dict)
        all_experts = set()
        for task_type, expert_counts in all_expert_usage.items():
            all_experts.update(expert_counts.keys())
        
        for expert in all_experts:
            total_uses = sum(all_expert_usage[task].get(expert, 0) for task in task_prompts.keys())
            if total_uses > 0:
                for task in task_prompts.keys():
                    task_uses = all_expert_usage[task].get(expert, 0)
                    expert_task_affinity[expert][task] = task_uses / total_uses
        
        # Identify specialized experts (high affinity for specific tasks)
        specialized_experts = {}
        for expert, affinities in expert_task_affinity.items():
            if affinities:
                max_affinity = max(affinities.values())
                if max_affinity > 0.5:  # Expert used >50% for one task type
                    specialized_task = max(affinities, key=affinities.get)
                    specialized_experts[expert] = {
                        'primary_task': specialized_task,
                        'affinity': max_affinity,
                        'task_distribution': affinities
                    }
        
        results['expert_specialization'] = {
            'specialized_experts': specialized_experts,
            'generalist_experts': [
                expert for expert in expert_task_affinity 
                if expert not in specialized_experts and len(expert_task_affinity[expert]) > 1
            ],
            'total_experts_activated': len(all_experts),
        }
        
        # Calculate cross-task similarity based on expert usage overlap
        similarity_matrix = {}
        for task1 in task_prompts.keys():
            similarity_matrix[task1] = {}
            experts1 = set(all_expert_usage[task1].keys())
            
            for task2 in task_prompts.keys():
                experts2 = set(all_expert_usage[task2].keys())
                
                if experts1 and experts2:
                    # Jaccard similarity
                    intersection = len(experts1 & experts2)
                    union = len(experts1 | experts2)
                    similarity = intersection / union if union > 0 else 0
                else:
                    similarity = 0
                
                similarity_matrix[task1][task2] = similarity
        
        results['cross_task_similarity'] = similarity_matrix
        
        # Layer-wise analysis if requested
        if track_layers:
            layer_stats = {}
            for task_type, layer_data in layer_expert_counts.items():
                layer_stats[task_type] = {}
                for layer, expert_counts in layer_data.items():
                    layer_stats[task_type][layer] = {
                        'unique_experts': len(expert_counts),
                        'total_activations': sum(expert_counts.values()),
                        'top_experts': dict(expert_counts.most_common(3))
                    }
            results['layer_analysis'] = layer_stats
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    @method()
    def visualize_expert_usage(
        self,
        analysis_results: Dict[str, Any],
        output_path: str = "/cache/results/expert_analysis.png"
    ) -> str:
        """
        Create visualizations of expert usage patterns.
        
        Args:
            analysis_results: Results from analyze_expert_routing
            output_path: Path to save visualization
        
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Expert usage by task
        task_data = []
        for task, info in analysis_results['task_analysis'].items():
            for expert, count in info['top_5_experts'].items():
                task_data.append({'Task': task, 'Expert': f"E{expert}", 'Count': count})
        
        if task_data:
            df_task = pd.DataFrame(task_data)
            pivot_task = df_task.pivot(index='Expert', columns='Task', values='Count').fillna(0)
            sns.heatmap(pivot_task, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0])
            axes[0, 0].set_title('Expert Usage by Task (Top 5 Experts per Task)')
        
        # 2. Task similarity matrix
        similarity_data = analysis_results['cross_task_similarity']
        if similarity_data:
            df_sim = pd.DataFrame(similarity_data)
            sns.heatmap(df_sim, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1], vmin=0, vmax=1)
            axes[0, 1].set_title('Task Similarity (Based on Expert Overlap)')
        
        # 3. Expert specialization
        spec_experts = analysis_results['expert_specialization']['specialized_experts']
        if spec_experts:
            tasks = list(set(e['primary_task'] for e in spec_experts.values()))
            task_counts = Counter(e['primary_task'] for e in spec_experts.values())
            
            axes[1, 0].bar(task_counts.keys(), task_counts.values())
            axes[1, 0].set_xlabel('Task Type')
            axes[1, 0].set_ylabel('Number of Specialized Experts')
            axes[1, 0].set_title('Expert Specialization Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Overall expert activation frequency
        all_expert_counts = Counter()
        for task_info in analysis_results['task_analysis'].values():
            all_expert_counts.update(task_info['top_5_experts'])
        
        if all_expert_counts:
            top_10_experts = dict(all_expert_counts.most_common(10))
            axes[1, 1].bar([f"E{e}" for e in top_10_experts.keys()], top_10_experts.values())
            axes[1, 1].set_xlabel('Expert ID')
            axes[1, 1].set_ylabel('Total Activations')
            axes[1, 1].set_title('Top 10 Most Active Experts Overall')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
        return output_path
    
    @method()
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """
        Generate a detailed report of expert usage analysis.
        
        Args:
            analysis_results: Results from analyze_expert_routing
            format: Output format (markdown or json)
        
        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(analysis_results, indent=2, default=str)
        
        # Generate markdown report
        report = "# GPT-OSS Expert Usage Analysis Report\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        total_experts = analysis_results['expert_specialization']['total_experts_activated']
        specialized_count = len(analysis_results['expert_specialization']['specialized_experts'])
        generalist_count = len(analysis_results['expert_specialization']['generalist_experts'])
        
        report += f"- **Total Unique Experts Activated**: {total_experts}\n"
        report += f"- **Specialized Experts**: {specialized_count} ({specialized_count/total_experts*100:.1f}%)\n"
        report += f"- **Generalist Experts**: {generalist_count} ({generalist_count/total_experts*100:.1f}%)\n\n"
        
        # Task Analysis
        report += "## Task-Specific Analysis\n\n"
        for task, info in analysis_results['task_analysis'].items():
            report += f"### {task.capitalize()}\n"
            report += f"- Unique experts used: {info['unique_experts_used']}\n"
            report += f"- Top 5 experts: {', '.join(f'E{e}({c})' for e, c in info['top_5_experts'].items())}\n"
            
            if info['sample_outputs']:
                report += f"- Sample output: `{info['sample_outputs'][0][:100]}...`\n"
            report += "\n"
        
        # Expert Specialization
        report += "## Expert Specialization Patterns\n\n"
        
        if analysis_results['expert_specialization']['specialized_experts']:
            report += "### Highly Specialized Experts\n"
            report += "| Expert | Primary Task | Affinity | Task Distribution |\n"
            report += "|--------|--------------|----------|-------------------|\n"
            
            for expert_id, spec_info in list(analysis_results['expert_specialization']['specialized_experts'].items())[:10]:
                task_dist = ', '.join(f"{t}:{v:.2f}" for t, v in spec_info['task_distribution'].items())
                report += f"| E{expert_id} | {spec_info['primary_task']} | {spec_info['affinity']:.2f} | {task_dist} |\n"
            report += "\n"
        
        # Task Similarity
        report += "## Cross-Task Similarity\n\n"
        report += "Similarity scores based on expert usage overlap (0=no overlap, 1=identical):\n\n"
        
        sim_matrix = analysis_results['cross_task_similarity']
        if sim_matrix:
            tasks = list(sim_matrix.keys())
            report += "| Task |" + " | ".join(tasks) + " |\n"
            report += "|------|" + "---|" * len(tasks) + "\n"
            
            for task1 in tasks:
                row = f"| {task1} |"
                for task2 in tasks:
                    score = sim_matrix[task1][task2]
                    row += f" {score:.2f} |"
                report += row + "\n"
            report += "\n"
        
        # Layer Analysis (if available)
        if analysis_results.get('layer_analysis'):
            report += "## Layer-wise Expert Activation\n\n"
            report += "Summary of expert activation patterns across model layers:\n\n"
            
            for task, layer_data in list(analysis_results['layer_analysis'].items())[:2]:
                report += f"### {task.capitalize()}\n"
                for layer, stats in list(layer_data.items())[:3]:
                    report += f"- Layer {layer}: {stats['unique_experts']} unique experts, "
                    report += f"{stats['total_activations']} total activations\n"
                report += "\n"
        
        # Insights
        report += "## Key Insights\n\n"
        
        # Find most similar tasks
        max_sim = 0
        similar_tasks = None
        for t1 in sim_matrix:
            for t2 in sim_matrix[t1]:
                if t1 != t2 and sim_matrix[t1][t2] > max_sim:
                    max_sim = sim_matrix[t1][t2]
                    similar_tasks = (t1, t2)
        
        if similar_tasks:
            report += f"1. **Most Similar Tasks**: {similar_tasks[0]} and {similar_tasks[1]} "
            report += f"(similarity: {max_sim:.2f})\n"
        
        # Task with most unique experts
        task_unique = max(analysis_results['task_analysis'].items(), 
                         key=lambda x: x[1]['unique_experts_used'])
        report += f"2. **Most Diverse Task**: {task_unique[0]} uses {task_unique[1]['unique_experts_used']} unique experts\n"
        
        # Expert usage concentration
        all_expert_counts = Counter()
        for task_info in analysis_results['task_analysis'].values():
            all_expert_counts.update(task_info['top_5_experts'])
        
        if all_expert_counts:
            top_10_usage = sum(count for _, count in all_expert_counts.most_common(10))
            total_usage = sum(all_expert_counts.values())
            concentration = top_10_usage / total_usage * 100
            report += f"3. **Expert Concentration**: Top 10 experts handle {concentration:.1f}% of all activations\n"
        
        return report


@app.local_entrypoint()
def main(
    model_id: str = "openai/gpt-oss-20b",
    tasks: Optional[str] = None,  # Comma-separated list of task types
    custom_prompts: Optional[str] = None,  # JSON file with custom prompts
    max_length: int = 100,
    temperature: float = 0.0,
    output_format: str = "markdown",  # markdown or json
    save_visualization: bool = True,
    output_file: Optional[str] = None,
):
    """
    Analyze expert usage in GPT-OSS for different tasks.
    
    Examples:
        # Basic analysis with default tasks
        modal run analyze_expert_usage.py
        
        # Analyze specific task categories
        modal run analyze_expert_usage.py --tasks "coding,math,science"
        
        # Use custom prompts from JSON file
        modal run analyze_expert_usage.py --custom-prompts my_prompts.json
        
        # Generate JSON report
        modal run analyze_expert_usage.py --output-format json --output-file results.json
        
        # Analyze with temperature for stochastic routing
        modal run analyze_expert_usage.py --temperature 0.7
    """
    
    analyzer = ExpertUsageAnalyzer()
    
    # Prepare task prompts
    task_prompts = None
    
    if custom_prompts:
        # Load custom prompts from JSON file
        with open(custom_prompts, 'r') as f:
            task_prompts = json.load(f)
    elif tasks:
        # Filter to specific tasks
        all_tasks = {
            "coding": [
                "def fibonacci(n):",
                "class BinaryTree:",
                "import numpy as np\n# Calculate matrix multiplication",
            ],
            "math": [
                "Solve for x: 2x^2 + 5x - 3 = 0",
                "The derivative of sin(x) * cos(x) is",
                "Calculate the integral of 1/x from 1 to e:",
            ],
            "science": [
                "The process of photosynthesis converts",
                "Newton's second law states that F =",
                "DNA replication occurs during the",
            ],
            "creative": [
                "Once upon a time in a distant galaxy,",
                "The old mansion stood silent, its windows",
                "She picked up the mysterious letter and read:",
            ],
            "factual": [
                "The capital of France is",
                "World War II ended in the year",
                "The largest planet in our solar system is",
            ],
            "reasoning": [
                "If all roses are flowers and some flowers fade quickly, then",
                "John is taller than Mary. Mary is taller than Sue. Therefore,",
                "A store offers 20% off. If an item costs $50 after discount, the original price was",
            ],
        }
        
        task_list = [t.strip() for t in tasks.split(',')]
        task_prompts = {t: all_tasks[t] for t in task_list if t in all_tasks}
    
    # Run analysis
    print(f"\nAnalyzing expert usage in {model_id}")
    print(f"Temperature: {temperature} (0=deterministic routing)")
    print(f"Max generation length: {max_length}")
    
    results = analyzer.analyze_expert_routing.remote(
        model_id=model_id,
        task_prompts=task_prompts,
        max_length=max_length,
        temperature=temperature,
        track_layers=True
    )
    
    # Generate visualization
    if save_visualization:
        viz_path = analyzer.visualize_expert_usage.remote(results)
        print(f"\nVisualization saved: {viz_path}")
    
    # Generate report
    report = analyzer.generate_report.remote(results, format=output_format)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    else:
        print("\n" + "="*60)
        print("EXPERT USAGE ANALYSIS RESULTS")
        print("="*60)
        
        if output_format == "markdown":
            # Print summary for console
            total_experts = results['expert_specialization']['total_experts_activated']
            specialized = len(results['expert_specialization']['specialized_experts'])
            
            print(f"\nTotal Experts Activated: {total_experts}")
            print(f"Specialized Experts: {specialized}")
            print(f"Generalist Experts: {len(results['expert_specialization']['generalist_experts'])}")
            
            print("\nTask Analysis Summary:")
            for task, info in results['task_analysis'].items():
                print(f"  {task}: {info['unique_experts_used']} unique experts")
                top_expert = list(info['top_5_experts'].keys())[0] if info['top_5_experts'] else 'None'
                print(f"    Top expert: E{top_expert}")
            
            print("\nExpert Specialization Examples:")
            for expert_id, spec in list(results['expert_specialization']['specialized_experts'].items())[:5]:
                print(f"  Expert {expert_id}: {spec['primary_task']} specialist ({spec['affinity']:.2%})")
            
            print("\n" + "="*60)
            print("\nFull report available in markdown format")
            print("Run with --output-file to save the complete analysis")
        else:
            print(report)


if __name__ == "__main__":
    main()