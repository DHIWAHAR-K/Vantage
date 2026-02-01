"""
Expert Network Implementation for MoE
Each expert is a specialized feedforward network with SwiGLU activation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class ExpertNetwork(nn.Module):
    """
    Single expert feedforward network with SwiGLU activation.
    
    Architecture:
        x -> LayerNorm -> Linear(gate) & Linear(up) -> SwiGLU -> Linear(down) -> Dropout
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (typically 4x hidden_size)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # SwiGLU requires two projections for gating
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with SwiGLU activation.
        
        SwiGLU(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # SwiGLU: element-wise product of gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SiLU (Swish) activation: x * sigmoid(x)
        hidden = nn.silu(gate) * up
        
        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output


class ExpertParallel(nn.Module):
    """
    Parallel execution of multiple experts for efficiency.
    
    Instead of running experts sequentially, we batch them together
    using MLX's efficient tensor operations.
    
    Args:
        num_experts: Number of expert networks
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension for each expert
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Create all experts
        self.experts = [
            ExpertNetwork(hidden_size, intermediate_size, dropout)
            for _ in range(num_experts)
        ]
        
    def __call__(
        self,
        x: mx.array,
        expert_indices: mx.array,
        expert_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Execute selected experts in parallel.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            expert_indices: Expert indices to activate (batch_size, seq_len, top_k)
            expert_weights: Weights for combining experts (batch_size, seq_len, top_k)
            
        Returns:
            Weighted combination of expert outputs
        """
        batch_size, seq_len, _ = x.shape
        
        if expert_weights is None:
            # Equal weighting if not provided
            top_k = expert_indices.shape[-1]
            expert_weights = mx.ones((batch_size, seq_len, top_k)) / top_k
        
        # Flatten batch and sequence dimensions for easier processing
        x_flat = x.reshape(-1, self.hidden_size)  # (batch * seq, hidden)
        indices_flat = expert_indices.reshape(-1, expert_indices.shape[-1])  # (batch * seq, top_k)
        weights_flat = expert_weights.reshape(-1, expert_weights.shape[-1])  # (batch * seq, top_k)
        
        # Initialize output
        output = mx.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = indices_flat == expert_idx  # (batch * seq, top_k)
            
            # Get weights for this expert
            expert_token_weights = mx.where(
                expert_mask,
                weights_flat,
                mx.zeros_like(weights_flat)
            )  # (batch * seq, top_k)
            
            # Sum weights across top_k dimension
            expert_token_weights = expert_token_weights.sum(axis=-1, keepdims=True)  # (batch * seq, 1)
            
            # Only process if this expert is used
            if expert_token_weights.sum() > 0:
                # Execute expert
                expert_output = self.experts[expert_idx](x_flat)
                
                # Weighted contribution
                output = output + expert_output * expert_token_weights
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        
        return output


class ExpertConfig:
    """Configuration for expert networks"""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_experts: int = 16,
        dropout: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.dropout = dropout
    
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
            "dropout": self.dropout,
        }
