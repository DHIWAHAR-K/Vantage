"""
Load T5-small from HuggingFace and convert to MLX format.

T5 (Text-to-Text Transfer Transformer) is perfect for text-to-SQL:
- Encoder processes the question + schema
- Decoder generates the SQL query

This module handles converting the PyTorch T5 model to MLX format.
"""

from pathlib import Path
from typing import Dict, Tuple
import json

import mlx.core as mx
import mlx.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class T5Config:
    """
    Configuration for T5-small model.
    
    These are the standard T5-small settings from HuggingFace.
    """
    def __init__(self):
        self.vocab_size = 32128
        self.d_model = 512          # Embedding dimension
        self.d_kv = 64              # Key/value dimension per head
        self.d_ff = 2048            # Feed-forward hidden dimension
        self.num_layers = 6         # Encoder and decoder layers
        self.num_heads = 8          # Attention heads
        self.dropout_rate = 0.1
        self.layer_norm_epsilon = 1e-6
        self.decoder_start_token_id = 0
        self.pad_token_id = 0


def convert_pytorch_to_mlx(pytorch_state_dict: Dict) -> Dict:
    """
    Convert PyTorch T5 weights to MLX format.
    
    PyTorch uses different tensor formats than MLX, so we need to convert.
    
    Args:
        pytorch_state_dict: T5 model weights from HuggingFace
        
    Returns:
        Dictionary of MLX arrays
    """
    mlx_weights = {}
    
    print("Converting PyTorch weights to MLX format...")
    
    for name, param in pytorch_state_dict.items():
        # Convert to numpy first, then to MLX
        numpy_param = param.detach().cpu().numpy()
        mlx_param = mx.array(numpy_param)
        
        # Store with same name
        mlx_weights[name] = mlx_param
    
    print(f"Converted {len(mlx_weights)} weight tensors")
    
    return mlx_weights


def load_t5_small(
    cache_dir: str = "models/t5_small"
) -> Tuple[Dict, T5Tokenizer, T5Config]:
    """
    Load T5-small from HuggingFace and convert to MLX.
    
    This downloads the pre-trained T5-small model (if not cached) and
    converts it to MLX format for efficient training on M3 Max.
    
    Args:
        cache_dir: Directory to cache the model
        
    Returns:
        Tuple of (mlx_weights, tokenizer, config)
        
    Example:
        >>> weights, tokenizer, config = load_t5_small()
        >>> print(f"Loaded T5 with {config.vocab_size:,} vocab")
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    mlx_weights_file = cache_path / "mlx_weights.safetensors"
    
    # Check if already converted
    if mlx_weights_file.exists():
        print(f"Loading cached MLX weights from {mlx_weights_file}...")
        mlx_weights = load_mlx_weights(str(mlx_weights_file))
        tokenizer = T5Tokenizer.from_pretrained(str(cache_path))
        config = T5Config()
        return mlx_weights, tokenizer, config
    
    # Download from HuggingFace
    print("Downloading T5-small from HuggingFace...")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    print(f"T5-small loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Convert to MLX
    mlx_weights = convert_pytorch_to_mlx(model.state_dict())
    
    # Save for future use
    print(f"Caching MLX weights to {cache_path}...")
    save_mlx_weights(mlx_weights, str(mlx_weights_file))
    tokenizer.save_pretrained(str(cache_path))
    
    # Create config
    config = T5Config()
    
    return mlx_weights, tokenizer, config


def save_mlx_weights(weights: Dict, path: str):
    """
    Save MLX weights to disk.
    
    Args:
        weights: Dictionary of MLX arrays
        path: File path to save to
    """
    # Convert to saveable format
    saveable = {}
    for name, array in weights.items():
        saveable[name] = array
    
    # Save using MLX's built-in save
    mx.savez(path, **saveable)
    print(f"Weights saved to {path}")


def load_mlx_weights(path: str) -> Dict:
    """
    Load MLX weights from disk.
    
    Args:
        path: File path to load from
        
    Returns:
        Dictionary of MLX arrays
    """
    weights = dict(mx.load(path))
    print(f"Loaded {len(weights)} weight tensors")
    return weights


class SimpleT5ForTextToSQL(nn.Module):
    """
    Simplified T5 model for text-to-SQL using MLX.
    
    This is a lightweight wrapper around T5's encoder-decoder architecture,
    optimized for the text-to-SQL task.
    
    The model takes:
        Input: "translate English to SQL: How many users? Schema: users(id, name)"
        Output: "SELECT COUNT(*) FROM users"
    """
    
    def __init__(self, weights: Dict, config: T5Config):
        super().__init__()
        self.config = config
        self.weights = weights
        
        # We'll use the pre-trained weights directly for now
        # In a full implementation, we'd build the T5 architecture in MLX
        print("T5 model initialized with pre-trained weights")
    
    def forward(self, input_ids: mx.array, decoder_input_ids: mx.array) -> mx.array:
        """
        Forward pass through T5.
        
        Args:
            input_ids: Encoder input [batch, seq_len]
            decoder_input_ids: Decoder input [batch, target_len]
            
        Returns:
            Logits [batch, target_len, vocab_size]
        """
        # This is a simplified version
        # Full T5 implementation would encode, cross-attend, and decode
        raise NotImplementedError("Use mlx-lm's T5 implementation or wait for full port")


if __name__ == "__main__":
    # Test loading T5
    print("Testing T5 loading...")
    
    weights, tokenizer, config = load_t5_small()
    
    print(f"\nT5-small loaded successfully!")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Model dimension: {config.d_model}")
    print(f"Layers: {config.num_layers}")
    print(f"Attention heads: {config.num_heads}")
    
    # Test tokenizer
    test_text = "translate English to SQL: How many users are there? Schema: users(id, name, email)"
    tokens = tokenizer.encode(test_text)
    print(f"\nTest encoding: '{test_text[:50]}...'")
    print(f"Tokens: {tokens[:10]}... ({len(tokens)} total)")
    
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded[:50]}...'")
