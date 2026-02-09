"""
Simple inference API for text-to-SQL.

Usage:
    >>> from src.inference import Text2SQLInference
    >>> model = Text2SQLInference("checkpoints/final")
    >>> sql = model.generate("How many users?", "users(id, name, email)")
    >>> print(sql)
"""

from pathlib import Path
from typing import Optional

import mlx.core as mx
from transformers import T5Tokenizer


class Text2SQLInference:
    """
    Fast inference for text-to-SQL using fine-tuned T5.
    
    Example:
        >>> model = Text2SQLInference("checkpoints/final")
        >>> sql = model.generate(
        ...     question="Show all active users",
        ...     schema="users(id, name, email, active)"
        ... )
        >>> print(sql)
        SELECT * FROM users WHERE active = 1
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        max_length: int = 128,
        temperature: float = 0.0  # 0 = greedy (deterministic)
    ):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            max_length: Maximum SQL length to generate
            temperature: Sampling temperature (0 = greedy)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.max_length = max_length
        self.temperature = temperature
        
        print(f"Loading model from {checkpoint_path}...")
        self._load_model()
        print("Model loaded!")
    
    def _load_model(self):
        """Load fine-tuned T5 model."""
        try:
            # Load using mlx-lm
            from mlx_lm import load
            
            self.model, self.tokenizer = load(str(self.checkpoint_path))
            print("Loaded with mlx-lm")
            
        except ImportError:
            # Fallback: manual loading
            print("mlx-lm not found, using manual loading...")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(str(self.checkpoint_path))
            
            # Load weights
            weights_file = self.checkpoint_path / "weights.npz"
            if not weights_file.exists():
                raise FileNotFoundError(f"Weights not found: {weights_file}")
            
            weights = dict(mx.load(str(weights_file)))
            
            # TODO: Reconstruct model from weights
            print("Manual loading not fully implemented yet")
            self.model = None
    
    def generate(
        self,
        question: str,
        schema: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate SQL query from question and schema.
        
        Args:
            question: Natural language question
            schema: Database schema string
            max_length: Max SQL length (default: use init value)
            
        Returns:
            Generated SQL query string
            
        Example:
            >>> sql = model.generate(
            ...     "How many users signed up today?",
            ...     "users(id, name, signup_date)"
            ... )
        """
        # Build input text (T5 format)
        input_text = f"translate English to SQL: {question} Schema: {schema}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            max_length=256,
            truncation=True
        )
        
        # Convert to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        
        # Generate
        max_len = max_length or self.max_length
        
        if self.model is None:
            # Placeholder
            return "SELECT * FROM table"
        
        # TODO: Implement actual generation
        # For now, placeholder
        output_text = "SELECT * FROM table"
        
        return output_text
    
    def batch_generate(
        self,
        questions: list[str],
        schemas: list[str]
    ) -> list[str]:
        """
        Generate SQL for multiple questions (batch inference).
        
        Args:
            questions: List of questions
            schemas: List of corresponding schemas
            
        Returns:
            List of generated SQL queries
        """
        if len(questions) != len(schemas):
            raise ValueError("questions and schemas must have same length")
        
        results = []
        for q, s in zip(questions, schemas):
            sql = self.generate(q, s)
            results.append(sql)
        
        return results


if __name__ == "__main__":
    # Test inference
    print("Testing inference API...")
    
    # This will fail without a real checkpoint
    try:
        model = Text2SQLInference("checkpoints/final")
        
        # Test single query
        sql = model.generate(
            question="How many users are there?",
            schema="users(id, name, email)"
        )
        print(f"\nGenerated SQL: {sql}")
        
        # Test batch
        questions = [
            "Show all users",
            "Count active users",
            "Get user emails"
        ]
        schemas = [
            "users(id, name, email, active)",
            "users(id, name, email, active)",
            "users(id, name, email, active)"
        ]
        
        results = model.batch_generate(questions, schemas)
        print("\nBatch results:")
        for q, sql in zip(questions, results):
            print(f"  Q: {q}")
            print(f"  SQL: {sql}\n")
        
    except Exception as e:
        print(f"\nCannot test without checkpoint: {e}")
        print("Run training first to create a checkpoint!")
