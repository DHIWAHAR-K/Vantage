"""
Sparse Router Implementation for MoE
Implements top-K gating with load balancing
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional
import math


class SparseRouter(nn.Module):
    """
    Sparse router with top-K expert selection and load balancing.
    
    The router learns to route tokens to the most appropriate experts
    while maintaining balanced expert utilization.
    
    Args:
        hidden_size: Input dimension
        num_experts: Total number of experts
        num_experts_per_token: Number of experts to activate per token (top-K)
        expert_capacity: Maximum capacity per expert (for load balancing)
        jitter_noise: Standard deviation of noise for exploration during training
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity: float = 1.25,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity = expert_capacity
        self.jitter_noise = jitter_noise
        
        # Gating network: simple linear layer
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def _compute_gate_logits(self, hidden_states: mx.array, training: bool = True) -> mx.array:
        """
        Compute gating logits with optional noise for exploration.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            training: Whether in training mode
            
        Returns:
            Gate logits (batch_size, seq_len, num_experts)
        """
        # Project to expert dimension
        logits = self.gate(hidden_states)
        
        if training and self.jitter_noise > 0:
            # Add Gumbel noise for exploration
            # Gumbel(0,1) = -log(-log(Uniform(0,1)))
            noise = mx.random.uniform(shape=logits.shape)
            noise = -mx.log(-mx.log(noise + 1e-10) + 1e-10)
            logits = logits + self.jitter_noise * noise
        
        return logits
    
    def _top_k_routing(
        self,
        logits: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Perform top-K routing with softmax normalization.
        
        Args:
            logits: Gate logits (batch_size, seq_len, num_experts)
            
        Returns:
            expert_indices: Selected expert indices (batch_size, seq_len, top_k)
            expert_weights: Normalized weights (batch_size, seq_len, top_k)
            router_logits: Full logits for computing auxiliary losses
        """
        # Get top-K experts
        top_k_logits, top_k_indices = mx.top_k(logits, k=self.num_experts_per_token)
        
        # Softmax over selected experts for normalization
        top_k_weights = mx.softmax(top_k_logits, axis=-1)
        
        return top_k_indices, top_k_weights, logits
    
    def _compute_load_balancing_loss(
        self,
        logits: mx.array,
        expert_indices: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute load balancing auxiliary losses.
        
        Two losses:
        1. Importance loss: Encourages uniform expert importance
        2. Load loss: Encourages uniform token distribution
        
        Args:
            logits: Full gate logits (batch_size, seq_len, num_experts)
            expert_indices: Selected experts (batch_size, seq_len, top_k)
            
        Returns:
            importance_loss: Variance of expert probabilities
            load_loss: Variance of expert token counts
        """
        # Compute routing probabilities
        routing_probs = mx.softmax(logits, axis=-1)  # (batch, seq, num_experts)
        
        # 1. Importance loss: variance of average routing probability per expert
        # We want P(expert) ≈ 1/num_experts for all experts
        avg_prob_per_expert = routing_probs.mean(axis=(0, 1))  # (num_experts,)
        target_prob = 1.0 / self.num_experts
        importance_loss = mx.var(avg_prob_per_expert) * self.num_experts
        
        # 2. Load loss: variance of token count per expert
        # Count how many tokens are routed to each expert
        batch_size, seq_len, top_k = expert_indices.shape
        total_tokens = batch_size * seq_len
        
        # Create one-hot encoding of expert assignments
        expert_mask = mx.zeros((batch_size, seq_len, top_k, self.num_experts))
        for i in range(self.num_experts):
            expert_mask[:, :, :, i] = (expert_indices == i).astype(mx.float32)
        
        # Count tokens per expert
        tokens_per_expert = expert_mask.sum(axis=(0, 1, 2))  # (num_experts,)
        
        # Normalize and compute variance
        tokens_per_expert = tokens_per_expert / (total_tokens * top_k)
        target_load = 1.0 / self.num_experts
        load_loss = mx.var(tokens_per_expert) * self.num_experts
        
        return importance_loss, load_loss
    
    def _compute_router_z_loss(self, logits: mx.array) -> mx.array:
        """
        Router z-loss: encourages smaller logits for better stability.
        
        Z-loss = log(sum(exp(logits)))^2
        
        Args:
            logits: Gate logits (batch_size, seq_len, num_experts)
            
        Returns:
            z_loss: Scalar loss value
        """
        # Log-sum-exp for numerical stability
        log_z = mx.logsumexp(logits, axis=-1)  # (batch, seq)
        z_loss = (log_z ** 2).mean()
        
        return z_loss
    
    def __call__(
        self,
        hidden_states: mx.array,
        training: bool = True,
    ) -> Tuple[mx.array, mx.array, dict]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            training: Whether in training mode
            
        Returns:
            expert_indices: Selected expert indices (batch_size, seq_len, top_k)
            expert_weights: Normalized weights (batch_size, seq_len, top_k)
            aux_losses: Dictionary of auxiliary losses for training
        """
        # Compute gate logits
        logits = self._compute_gate_logits(hidden_states, training)
        
        # Top-K routing
        expert_indices, expert_weights, full_logits = self._top_k_routing(logits)
        
        # Compute auxiliary losses for load balancing
        aux_losses = {}
        if training:
            importance_loss, load_loss = self._compute_load_balancing_loss(
                full_logits, expert_indices
            )
            z_loss = self._compute_router_z_loss(full_logits)
            
            aux_losses = {
                "router_importance_loss": importance_loss,
                "router_load_loss": load_loss,
                "router_z_loss": z_loss,
            }
        
        return expert_indices, expert_weights, aux_losses


class RouterConfig:
    """Configuration for sparse router"""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_experts: int = 16,
        num_experts_per_token: int = 2,
        expert_capacity: float = 1.25,
        jitter_noise: float = 0.01,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity = expert_capacity
        self.jitter_noise = jitter_noise
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
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "expert_capacity": self.expert_capacity,
            "jitter_noise": self.jitter_noise,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_z_loss_coef": self.router_z_loss_coef,
        }
