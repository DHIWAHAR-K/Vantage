"""
Convert T5-small from HuggingFace to MLX format.

This is a one-time setup script to download and convert T5-small.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_t5_small


def main():
    """Convert T5-small and save in MLX format."""
    print("="*60)
    print("CONVERTING T5-SMALL TO MLX FORMAT")
    print("="*60)
    
    print("\nThis will:")
    print("1. Download T5-small from HuggingFace (~250MB)")
    print("2. Convert PyTorch weights to MLX format")
    print("3. Cache for future use")
    
    print("\nStarting conversion...")
    
    try:
        weights, tokenizer, config = load_t5_small()
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        
        print(f"\nModel: T5-small")
        print(f"Vocabulary: {config.vocab_size:,} tokens")
        print(f"Model dimension: {config.d_model}")
        print(f"Layers: {config.num_layers}")
        print(f"Attention heads: {config.num_heads}")
        print(f"\nCached at: models/t5_small/")
        
        # Test tokenizer
        print("\nTesting tokenizer...")
        test_text = "translate English to SQL: Show all users"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"Input: '{test_text}'")
        print(f"Tokens: {len(tokens)}")
        print(f"Decoded: '{decoded}'")
        
        print("\nReady to fine-tune!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("- transformers installed: pip install transformers")
        print("- torch installed: pip install torch")
        print("- Internet connection (to download from HuggingFace)")
        sys.exit(1)


if __name__ == "__main__":
    main()
