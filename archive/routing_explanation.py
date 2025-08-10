#!/usr/bin/env python3
"""
Detailed explanation and implementation of MoE routing calculation.
This demonstrates how routing decisions are made in Mixture of Experts models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class MoERouter(nn.Module):
    """
    Router/Gate network that determines expert selection in MoE.
    This is the core component that calculates routing decisions.
    """
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        """
        Initialize the router.
        
        Args:
            hidden_dim: Dimension of input hidden states
            num_experts: Total number of experts available
            top_k: Number of experts to select per token
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Linear layer that produces routing scores
        # This is the "gate" that learns which expert should handle which token
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Optional: noise for load balancing during training
        self.noise_std = 0.1
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate routing weights and expert selection.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            routing_weights: Weights for selected experts [batch, seq_len, top_k]
            selected_experts: Indices of selected experts [batch, seq_len, top_k]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Step 1: Calculate router logits (scores for each expert)
        # This is where the router network evaluates each token against each expert
        router_logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]
        
        # Step 2: Add noise during training for load balancing (optional)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Step 3: Convert to probabilities using softmax
        # This ensures scores are normalized and sum to 1
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq_len, num_experts]
        
        # Step 4: Select top-k experts with highest probabilities
        # This is the key routing decision - which experts to activate
        routing_weights, selected_experts = torch.topk(
            router_probs, 
            k=min(self.top_k, self.num_experts), 
            dim=-1
        )
        
        # Step 5: Renormalize weights so they sum to 1
        # This ensures the expert outputs are properly weighted
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts, router_logits


class ExpertRoutingAnalyzer:
    """
    Analyzer for understanding routing decisions in MoE models.
    """
    
    def __init__(self):
        self.routing_history = []
        self.expert_load = {}
        
    def analyze_routing_decision(
        self, 
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        tokens: List[str] = None
    ) -> Dict:
        """
        Analyze a single routing decision.
        
        Args:
            router_logits: Raw scores from gate network [batch, seq_len, num_experts]
            routing_weights: Final routing weights [batch, seq_len, top_k]
            selected_experts: Selected expert indices [batch, seq_len, top_k]
            tokens: Optional token strings for analysis
            
        Returns:
            Analysis dictionary with routing insights
        """
        batch_size, seq_len, num_experts = router_logits.shape
        _, _, top_k = routing_weights.shape
        
        analysis = {
            'routing_entropy': [],
            'expert_usage': {},
            'token_expert_mapping': [],
            'load_balance_score': 0.0,
            'routing_confidence': []
        }
        
        # Calculate routing entropy (measure of routing certainty)
        router_probs = F.softmax(router_logits, dim=-1)
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
        analysis['routing_entropy'] = entropy.mean().item()
        
        # Analyze expert usage distribution
        expert_counts = torch.zeros(num_experts)
        for expert_idx in selected_experts.flatten():
            expert_counts[expert_idx] += 1
        
        for i in range(num_experts):
            analysis['expert_usage'][i] = int(expert_counts[i].item())
        
        # Calculate load balance score (how evenly distributed the load is)
        # Perfect balance = 1.0, poor balance = close to 0
        expected_load = selected_experts.numel() / num_experts
        load_variance = torch.var(expert_counts)
        analysis['load_balance_score'] = 1.0 / (1.0 + load_variance.item() / (expected_load + 1e-6))
        
        # Analyze routing confidence (difference between top-1 and top-2 weights)
        if top_k >= 2:
            confidence = routing_weights[:, :, 0] - routing_weights[:, :, 1]
            analysis['routing_confidence'] = confidence.mean().item()
        
        # Token-expert mapping if tokens provided
        if tokens is not None:
            for seq_idx in range(min(seq_len, len(tokens))):
                token_routing = {
                    'token': tokens[seq_idx],
                    'experts': selected_experts[0, seq_idx].tolist(),
                    'weights': routing_weights[0, seq_idx].tolist()
                }
                analysis['token_expert_mapping'].append(token_routing)
        
        return analysis
    
    def calculate_expert_importance_metrics(
        self,
        routing_history: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Calculate importance metrics for each expert based on routing history.
        
        Args:
            routing_history: List of routing analysis results
            
        Returns:
            Dictionary mapping expert ID to importance metrics
        """
        expert_metrics = {}
        total_activations = 0
        
        # Aggregate statistics across all routing decisions
        for routing in routing_history:
            for expert_id, count in routing['expert_usage'].items():
                if expert_id not in expert_metrics:
                    expert_metrics[expert_id] = {
                        'activation_count': 0,
                        'activation_frequency': 0.0,
                        'avg_routing_weight': [],
                        'associated_tokens': []
                    }
                
                expert_metrics[expert_id]['activation_count'] += count
                total_activations += count
                
                # Track token associations
                for mapping in routing.get('token_expert_mapping', []):
                    if expert_id in mapping['experts']:
                        idx = mapping['experts'].index(expert_id)
                        weight = mapping['weights'][idx]
                        expert_metrics[expert_id]['avg_routing_weight'].append(weight)
                        expert_metrics[expert_id]['associated_tokens'].append(mapping['token'])
        
        # Calculate final metrics
        for expert_id, metrics in expert_metrics.items():
            # Activation frequency (how often this expert is selected)
            metrics['activation_frequency'] = metrics['activation_count'] / max(total_activations, 1)
            
            # Average routing weight when selected
            if metrics['avg_routing_weight']:
                metrics['avg_routing_weight'] = np.mean(metrics['avg_routing_weight'])
            else:
                metrics['avg_routing_weight'] = 0.0
            
            # Calculate importance score combining frequency and weight
            metrics['importance_score'] = (
                metrics['activation_frequency'] * 0.5 +
                metrics['avg_routing_weight'] * 0.5
            )
            
            # Token specialization (entropy of token distribution)
            if metrics['associated_tokens']:
                token_counts = {}
                for token in metrics['associated_tokens']:
                    token_counts[token] = token_counts.get(token, 0) + 1
                
                # Calculate entropy
                total = sum(token_counts.values())
                probs = [count/total for count in token_counts.values()]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                metrics['specialization_score'] = 1.0 / (1.0 + entropy)  # Lower entropy = higher specialization
            else:
                metrics['specialization_score'] = 0.0
        
        return expert_metrics


def demonstrate_routing_calculation():
    """
    Demonstrate how routing is calculated in practice.
    """
    print("=" * 80)
    print("MoE ROUTING CALCULATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Setup
    batch_size = 1
    seq_len = 10
    hidden_dim = 768
    num_experts = 8
    top_k = 2
    
    # Create sample input
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Initialize router
    router = MoERouter(hidden_dim, num_experts, top_k)
    
    # Calculate routing
    routing_weights, selected_experts, router_logits = router(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Number of experts: {num_experts}")
    print(f"Experts selected per token (top-k): {top_k}")
    print()
    
    # Analyze routing
    analyzer = ExpertRoutingAnalyzer()
    analysis = analyzer.analyze_routing_decision(
        router_logits, 
        routing_weights, 
        selected_experts,
        tokens
    )
    
    print("ROUTING ANALYSIS:")
    print("-" * 40)
    print(f"Routing entropy: {analysis['routing_entropy']:.4f}")
    print(f"Load balance score: {analysis['load_balance_score']:.4f}")
    print(f"Routing confidence: {analysis['routing_confidence']:.4f}")
    print()
    
    print("EXPERT USAGE DISTRIBUTION:")
    print("-" * 40)
    for expert_id, count in sorted(analysis['expert_usage'].items()):
        bar = "█" * int(count * 2)
        print(f"Expert {expert_id}: {count:2d} activations {bar}")
    print()
    
    print("TOKEN-EXPERT ROUTING:")
    print("-" * 40)
    print(f"{'Token':<10} {'Expert 1':>10} {'Weight 1':>10} {'Expert 2':>10} {'Weight 2':>10}")
    print("-" * 40)
    
    for mapping in analysis['token_expert_mapping']:
        token = mapping['token']
        experts = mapping['experts']
        weights = mapping['weights']
        print(f"{token:<10} {experts[0]:>10d} {weights[0]:>10.4f} {experts[1]:>10d} {weights[1]:>10.4f}")
    
    print()
    print("KEY INSIGHTS:")
    print("-" * 40)
    print("1. Each token is routed to top-k experts based on gate network scores")
    print("2. Routing weights determine how much each expert contributes")
    print("3. Load balancing ensures no single expert is overloaded")
    print("4. Routing entropy measures routing certainty (lower = more certain)")
    print("5. Expert specialization emerges from learned routing patterns")
    
    # Visualize routing matrix
    visualize_routing_matrix(router_logits[0], selected_experts[0], tokens)


def visualize_routing_matrix(router_logits, selected_experts, tokens):
    """
    Visualize the routing decision matrix.
    """
    router_probs = F.softmax(router_logits, dim=-1).detach().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(router_probs.T, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Routing Probability')
    
    # Mark selected experts
    for token_idx in range(len(tokens)):
        for expert_idx in selected_experts[token_idx]:
            plt.scatter(token_idx, expert_idx, c='blue', s=100, marker='*')
    
    plt.xlabel('Token Position')
    plt.ylabel('Expert Index')
    plt.title('MoE Routing Matrix (★ = Selected Experts)')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(router_probs.shape[1]), [f'E{i}' for i in range(router_probs.shape[1])])
    plt.tight_layout()
    plt.savefig('routing_matrix.png', dpi=150)
    plt.close()
    
    print(f"\nRouting matrix visualization saved to routing_matrix.png")


def explain_routing_in_gpt_oss():
    """
    Explain specific routing mechanisms in GPT-OSS model.
    """
    print("\n" + "=" * 80)
    print("GPT-OSS SPECIFIC ROUTING MECHANISMS")
    print("=" * 80)
    print()
    
    explanation = """
    GPT-OSS uses a sophisticated MoE routing system:
    
    1. ARCHITECTURE:
       - 20B model has 16 experts per MoE layer
       - Uses top-2 routing (2 experts per token)
       - MoE layers alternate with standard transformer layers
    
    2. ROUTING CALCULATION:
       a) Gate Network: Linear layer produces logits for each expert
          logits = W_gate @ hidden_state + bias
       
       b) Softmax: Convert logits to probabilities
          probs = softmax(logits / temperature)
       
       c) Top-k Selection: Choose k experts with highest probabilities
          top_k_probs, top_k_indices = topk(probs, k=2)
       
       d) Renormalization: Ensure selected weights sum to 1
          final_weights = top_k_probs / sum(top_k_probs)
    
    3. LOAD BALANCING:
       - Auxiliary loss encourages even expert usage
       - Prevents mode collapse where few experts handle everything
       - Loss = coefficient * variance(expert_usage_counts)
    
    4. EXPERT SPECIALIZATION:
       - Early layers: Basic token/syntax processing
       - Middle layers: Semantic understanding, reasoning
       - Late layers: Output generation, coherence
    
    5. ROUTING PATTERNS:
       - Code tokens → specialized code experts
       - Math expressions → mathematical reasoning experts
       - Natural language → general language experts
    
    6. EFFICIENCY:
       - Only 2 experts active per token (12.5% of available experts)
       - Reduces computation while maintaining model capacity
       - Enables 20B parameters with ~3B active parameters
    """
    
    print(explanation)
    
    # Show example routing decision
    print("\nEXAMPLE ROUTING DECISION:")
    print("-" * 40)
    print("Input token: 'function'")
    print("Router scores: [0.05, 0.35, 0.08, 0.12, 0.15, 0.10, 0.08, 0.07]")
    print("Top-2 experts: Expert 1 (35%), Expert 4 (15%)")
    print("Final weights: [0.70, 0.30] (after renormalization)")
    print("Output: 0.70 * Expert1(token) + 0.30 * Expert4(token)")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_routing_calculation()
    explain_routing_in_gpt_oss()
    
    print("\n" + "=" * 80)
    print("ROUTING CALCULATION COMPLETE")
    print("=" * 80)