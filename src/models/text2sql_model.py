"""
Vantage Text-to-SQL Model
Decoder-only transformer with Mixture of Experts and schema understanding
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Tuple, List
import yaml
from pathlib import Path

from .moe_layer import MoELayer
from .schema_encoder import SchemaEncoder, CrossAttentionLayer
from .router import SparseRouter


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for better position encoding.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq
        
    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """
        Generate RoPE sin/cos embeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            cos_embed: Cosine embeddings (seq_len, dim)
            sin_embed: Sine embeddings (seq_len, dim)
        """
        # Position indices
        positions = mx.arange(seq_len).astype(mx.float32)
        
        # Compute frequencies for each position
        freqs = mx.outer(positions, self.inv_freq)  # (seq_len, dim/2)
        
        # Duplicate for full dimension
        freqs = mx.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)
        
        # Compute cos and sin
        cos_embed = mx.cos(freqs)
        sin_embed = mx.sin(freqs)
        
        return cos_embed, sin_embed


def apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (seq_len, head_dim)
        sin: Sine embeddings (seq_len, head_dim)
        
    Returns:
        q_rot: Rotated queries
        k_rot: Rotated keys
    """
    # Rotate half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
    
    # Apply rotation
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot


class VantageTransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and MoE feedforward.
    
    Optionally includes cross-attention to schema.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_size: int,
        dropout: float = 0.1,
        has_cross_attention: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.has_cross_attention = has_cross_attention
        
        # Self-attention with GQA (Grouped Query Attention)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiHeadAttention(
            dims=hidden_size,
            num_heads=num_attention_heads,
            bias=False,
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Cross-attention to schema (optional)
        if has_cross_attention:
            self.cross_attn = CrossAttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
        
        # MoE feedforward
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dropout=dropout,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
        )
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        schema_states: Optional[mx.array] = None,
        schema_mask: Optional[mx.array] = None,
        cos_embed: Optional[mx.array] = None,
        sin_embed: Optional[mx.array] = None,
        training: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden)
            attention_mask: Causal attention mask
            schema_states: Encoded schema (batch, schema_len, hidden)
            schema_mask: Schema attention mask
            cos_embed: RoPE cosine embeddings
            sin_embed: RoPE sine embeddings
            training: Whether in training mode
            
        Returns:
            output: Processed tensor
            aux_losses: Auxiliary losses from MoE
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        
        # TODO: Apply RoPE to Q and K before attention
        attn_output = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            mask=attention_mask,
        )
        hidden_states = residual + self.attn_dropout(attn_output)
        
        # Cross-attention to schema (if applicable)
        if self.has_cross_attention and schema_states is not None:
            hidden_states = self.cross_attn(
                hidden_states,
                schema_states,
                schema_mask,
            )
        
        # MoE feedforward
        hidden_states, aux_losses = self.moe(hidden_states, training=training)
        
        return hidden_states, aux_losses


class VantageModel(nn.Module):
    """
    Complete Vantage text-to-SQL model with MoE and schema understanding.
    
    Architecture:
        - Token embeddings
        - RoPE position encoding
        - N transformer blocks with MoE
        - Schema cross-attention every K blocks
        - LM head for SQL generation
    """
    
    def __init__(self, config: 'VantageConfig'):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # RoPE position embeddings
        self.rope = RoPEEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        
        # Schema encoder
        self.schema_encoder = SchemaEncoder(
            hidden_size=config.hidden_size,
            num_layers=config.schema_encoder_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout,
        )
        
        # Transformer blocks with MoE
        self.layers = []
        for layer_idx in range(config.num_layers):
            # Add cross-attention every K layers
            has_cross_attention = (
                (layer_idx + 1) % config.cross_attention_interval == 0
            )
            
            block = VantageTransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                intermediate_size=config.intermediate_size,
                dropout=config.hidden_dropout,
                has_cross_attention=has_cross_attention,
                router_aux_loss_coef=config.router_aux_loss_coef,
                router_z_loss_coef=config.router_z_loss_coef,
            )
            self.layers.append(block)
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # LM head for token prediction
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def __call__(
        self,
        input_ids: mx.array,
        schema_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        schema_mask: Optional[mx.array] = None,
        training: bool = True,
    ) -> Dict[str, mx.array]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            schema_ids: Schema token IDs (batch, schema_len)
            attention_mask: Causal attention mask
            schema_mask: Schema attention mask
            training: Whether in training mode
            
        Returns:
            Dictionary with logits and losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # RoPE embeddings
        cos_embed, sin_embed = self.rope(seq_len)
        
        # Encode schema if provided
        schema_states = None
        if schema_ids is not None:
            schema_embeddings = self.embed_tokens(schema_ids)
            schema_states = self.schema_encoder(schema_embeddings, schema_mask)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = mx.tril(mx.ones((seq_len, seq_len)))
            attention_mask = mx.expand_dims(attention_mask, axis=0)  # (1, seq_len, seq_len)
        
        # Collect auxiliary losses
        total_aux_loss = 0.0
        aux_losses_list = []
        
        # Process through transformer blocks
        for layer in self.layers:
            hidden_states, aux_losses = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                schema_states=schema_states,
                schema_mask=schema_mask,
                cos_embed=cos_embed,
                sin_embed=sin_embed,
                training=training,
            )
            
            if training and aux_losses:
                total_aux_loss = total_aux_loss + aux_losses.get("moe_aux_loss", 0.0)
                aux_losses_list.append(aux_losses)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {
            "logits": logits,
        }
        
        if training and total_aux_loss > 0:
            output["aux_loss"] = total_aux_loss / len(self.layers)
            output["aux_losses_detail"] = aux_losses_list
        
        return output
    
    @classmethod
    def from_config(cls, config_path: str):
        """Load model from configuration file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = VantageConfig.from_dict(config_dict)
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load pretrained model from checkpoint"""
        model_path = Path(model_path)
        
        # Load configuration
        config_path = model_path / "config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = VantageConfig.from_dict(config_dict['model'])
        model = cls(config)
        
        # Load weights
        weights_path = model_path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model to checkpoint"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {"model": self.config.to_dict()}
        with open(save_path / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f)
        
        # Save weights
        weights = dict(self.parameters())
        mx.savez(str(save_path / "weights.npz"), **weights)


class VantageConfig:
    """Complete configuration for Vantage model"""
    
    def __init__(
        self,
        # Model architecture
        name: str = "vantage-medium-8b",
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 16,
        intermediate_size: int = 16384,
        vocab_size: int = 32000,
        max_position_embeddings: int = 4096,
        
        # MoE config
        num_experts: int = 32,
        num_experts_per_token: int = 2,
        expert_capacity: float = 1.25,
        
        # RoPE config
        rope_theta: float = 10000.0,
        
        # Regularization
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.1,
        expert_dropout: float = 0.1,
        
        # Load balancing
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        
        # Schema encoder
        schema_encoder_layers: int = 6,
        cross_attention_interval: int = 4,
    ):
        self.name = name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity = expert_capacity
        
        self.rope_theta = rope_theta
        
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.expert_dropout = expert_dropout
        
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        
        self.schema_encoder_layers = schema_encoder_layers
        self.cross_attention_interval = cross_attention_interval
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        # Handle nested schema_encoder config
        if "schema_encoder" in config_dict:
            schema_config = config_dict.pop("schema_encoder")
            config_dict["schema_encoder_layers"] = schema_config.get("num_layers", 6)
            config_dict["cross_attention_interval"] = schema_config.get("cross_attention_interval", 4)
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "expert_capacity": self.expert_capacity,
            "rope_theta": self.rope_theta,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "expert_dropout": self.expert_dropout,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_z_loss_coef": self.router_z_loss_coef,
            "schema_encoder_layers": self.schema_encoder_layers,
            "cross_attention_interval": self.cross_attention_interval,
        }
    
    @classmethod
    def small(cls):
        """Small model configuration (2B parameters)"""
        return cls(
            name="vantage-small-2b",
            hidden_size=2048,
            num_layers=24,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=8192,
            num_experts=16,
            schema_encoder_layers=4,
        )
    
    @classmethod
    def medium(cls):
        """Medium model configuration (8B parameters)"""
        return cls(
            name="vantage-medium-8b",
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=64,
            num_key_value_heads=16,
            intermediate_size=16384,
            num_experts=32,
            schema_encoder_layers=6,
        )
    
    @classmethod
    def large(cls):
        """Large model configuration (24B parameters)"""
        return cls(
            name="vantage-large-24b",
            hidden_size=6144,
            num_layers=40,
            num_attention_heads=96,
            num_key_value_heads=24,
            intermediate_size=24576,
            num_experts=64,
            max_position_embeddings=8192,
            schema_encoder_layers=8,
        )
