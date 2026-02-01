"""
Schema Encoder for understanding database structure
Encodes table names, column names, types, and relationships
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Optional, Tuple


class SchemaEncoder(nn.Module):
    """
    Encodes database schema information with cross-attention support.
    
    Processes:
    - Table names and descriptions
    - Column names, types, and constraints
    - Foreign key relationships
    
    Args:
        hidden_size: Dimension of hidden states
        num_layers: Number of encoder layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        
        # Schema embedding layers
        self.layers = [
            SchemaEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        
        self.final_norm = nn.LayerNorm(hidden_size)
        
    def __call__(
        self,
        schema_embeddings: mx.array,
        schema_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Encode schema information.
        
        Args:
            schema_embeddings: Schema token embeddings (batch, schema_len, hidden)
            schema_mask: Attention mask for schema tokens (batch, schema_len)
            
        Returns:
            Encoded schema (batch, schema_len, hidden)
        """
        hidden_states = schema_embeddings
        
        # Process through encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, schema_mask)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states


class SchemaEncoderLayer(nn.Module):
    """
    Single encoder layer for schema processing.
    
    Architecture:
        Input -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> Output
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiHeadAttention(
            dims=hidden_size,
            num_heads=num_attention_heads,
            bias=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through encoder layer.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, hidden)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        attn_output = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            mask=attention_mask,
        )
        hidden_states = residual + self.attn_dropout(attn_output)
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending to schema while generating SQL.
    
    Allows the decoder to attend to encoded schema information.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiHeadAttention(
            dims=hidden_size,
            num_heads=num_attention_heads,
            bias=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(
        self,
        hidden_states: mx.array,
        schema_states: mx.array,
        schema_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Cross-attend to schema.
        
        Args:
            hidden_states: Decoder hidden states (batch, seq_len, hidden)
            schema_states: Encoded schema (batch, schema_len, hidden)
            schema_mask: Schema attention mask (batch, schema_len)
            
        Returns:
            Output with schema context (batch, seq_len, hidden)
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Cross-attention: Q from decoder, K,V from schema
        attn_output = self.cross_attn(
            hidden_states,  # Query
            schema_states,  # Key
            schema_states,  # Value
            mask=schema_mask,
        )
        
        output = residual + self.dropout(attn_output)
        
        return output


class SchemaConfig:
    """Configuration for schema encoder"""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        dropout: float = 0.1,
        cross_attention_interval: int = 4,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.cross_attention_interval = cross_attention_interval
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "cross_attention_interval": self.cross_attention_interval,
        }
