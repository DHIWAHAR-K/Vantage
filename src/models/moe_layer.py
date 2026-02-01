"""
Mixture of Experts Layer Implementation
Combines sparse router with expert networks for efficient computation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Dict

from .router import SparseRouter
from .expert import ExpertParallel


class MoELayer(nn.Module):
    """
    Complete Mixture of Experts layer with sparse routing.
    
    Architecture:
        Input -> LayerNorm -> Router (top-K selection) -> Experts (parallel) -> Output
    
    Only activates top-K experts per token for efficiency while maintaining
    model capacity through the total number of experts.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension for experts
        num_experts: Total number of expert networks
        num_experts_per_token: Number of experts to activate per token
        expert_capacity: Maximum capacity per expert
        dropout: Dropout probability
        router_aux_loss_coef: Coefficient for load balancing loss
        router_z_loss_coef: Coefficient for router z-loss
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity: float = 1.25,
        dropout: float = 0.1,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        
        # Pre-norm for stability
        self.norm = nn.LayerNorm(hidden_size)
        
        # Sparse router
        self.router = SparseRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_capacity=expert_capacity,
        )
        
        # Expert networks
        self.experts = ExpertParallel(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        
    def __call__(
        self,
        hidden_states: mx.array,
        training: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            training: Whether in training mode
            
        Returns:
            output: Processed tensor (batch_size, seq_len, hidden_size)
            aux_losses: Dictionary of auxiliary losses for training
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Pre-normalization
        hidden_states = self.norm(hidden_states)
        
        # Route tokens to experts
        expert_indices, expert_weights, router_losses = self.router(
            hidden_states,
            training=training,
        )
        
        # Execute selected experts
        expert_output = self.experts(
            hidden_states,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
        )
        
        # Residual connection
        output = residual + expert_output
        
        # Combine and scale auxiliary losses
        aux_losses = {}
        if training and router_losses:
            # Load balancing loss
            load_balance_loss = (
                router_losses.get("router_importance_loss", 0.0) +
                router_losses.get("router_load_loss", 0.0)
            )
            aux_losses["load_balance_loss"] = load_balance_loss * self.router_aux_loss_coef
            
            # Router z-loss for stability
            aux_losses["router_z_loss"] = (
                router_losses.get("router_z_loss", 0.0) * self.router_z_loss_coef
            )
            
            # Total auxiliary loss
            aux_losses["moe_aux_loss"] = (
                aux_losses["load_balance_loss"] + aux_losses["router_z_loss"]
            )
        
        return output, aux_losses
    
    def get_expert_statistics(
        self,
        hidden_states: mx.array,
    ) -> Dict[str, mx.array]:
        """
        Get statistics about expert utilization.
        Useful for analysis and debugging.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary with expert utilization statistics
        """
        # Get routing decisions
        logits = self.router._compute_gate_logits(hidden_states, training=False)
        expert_indices, expert_weights, _ = self.router._top_k_routing(logits)
        
        batch_size, seq_len, top_k = expert_indices.shape
        total_tokens = batch_size * seq_len
        
        # Count tokens per expert
        tokens_per_expert = mx.zeros(self.num_experts)
        for i in range(self.num_experts):
            mask = (expert_indices == i).astype(mx.float32)
            tokens_per_expert[i] = mask.sum()
        
        # Average weights per expert
        routing_probs = mx.softmax(logits, axis=-1)
        avg_weight_per_expert = routing_probs.mean(axis=(0, 1))
        
        return {
            "tokens_per_expert": tokens_per_expert,
            "tokens_per_expert_pct": tokens_per_expert / (total_tokens * top_k) * 100,
            "avg_weight_per_expert": avg_weight_per_expert,
            "expert_utilization": (tokens_per_expert > 0).sum() / self.num_experts * 100,
        }


class SparseMoEConfig:
    """Complete configuration for MoE layer"""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_experts: int = 16,
        num_experts_per_token: int = 2,
        expert_capacity: float = 1.25,
        dropout: float = 0.1,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity = expert_capacity
        self.dropout = dropout
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "expert_capacity": self.expert_capacity,
            "dropout": self.dropout,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_z_loss_coef": self.router_z_loss_coef,
        }
    
    def compute_active_parameters(self) -> int:
        """
        Compute number of active parameters per forward pass.
        
        Returns:
            Number of active parameters
        """
        # Router parameters (always active)
        router_params = self.hidden_size * self.num_experts
        
        # Expert parameters (only top-K active)
        params_per_expert = (
            self.hidden_size * self.intermediate_size * 3  # gate, up, down projections
        )
        active_expert_params = params_per_expert * self.num_experts_per_token
        
        total_active = router_params + active_expert_params
        
        return total_active
    
    def compute_total_parameters(self) -> int:
        """
        Compute total number of parameters in MoE layer.
        
        Returns:
            Total number of parameters
        """
        # Router parameters
        router_params = self.hidden_size * self.num_experts
        
        # All expert parameters
        params_per_expert = (
            self.hidden_size * self.intermediate_size * 3  # gate, up, down projections
        )
        total_expert_params = params_per_expert * self.num_experts
        
        # LayerNorm parameters
        norm_params = self.hidden_size * 2  # weight and bias
        
        total = router_params + total_expert_params + norm_params
        
        return total
    
    def compute_efficiency_ratio(self) -> float:
        """
        Compute ratio of active to total parameters.
        
        Returns:
            Efficiency ratio (lower is more efficient)
        """
        active = self.compute_active_parameters()
        total = self.compute_total_parameters()
        return active / total
