"""
Efficient data loader for pre-processed SynSQL data.

Loads train/val data from processed JSON files created by prepare_data.py.
Supports curriculum learning by filtering on difficulty.
"""

import json
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from dataclasses import dataclass
import random

import mlx.core as mx


@dataclass
class Text2SQLExample:
    """A single text-to-SQL training example."""
    question: str
    schema: str
    sql: str
    db_id: str
    difficulty: str = "Medium"


class ProcessedDataLoader:
    """
    Load pre-processed SynSQL data for training.
    
    Much faster than streaming from raw data.json since data is
    already processed and split.
    
    Example:
        >>> loader = ProcessedDataLoader(
        ...     data_dir="data/processed",
        ...     tokenizer=tokenizer,
        ...     split="train"
        ... )
        >>> for batch in loader.get_batches(batch_size=32):
        ...     # Train on batch
        ...     pass
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        split: str = "train",
        max_input_length: int = 256,
        max_output_length: int = 128,
        difficulty_filter: Optional[str] = None
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to processed data directory
            tokenizer: T5 tokenizer
            split: "train" or "val"
            max_input_length: Max input sequence length
            max_output_length: Max output sequence length
            difficulty_filter: Filter by difficulty ("Simple", "Medium", "Complex")
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.difficulty_filter = difficulty_filter
        
        # Load data
        data_file = self.data_dir / f"{split}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Processed data not found: {data_file}\n"
                f"Run: python scripts/prepare_data.py"
            )
        
        print(f"Loading {split} data from {data_file}...")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data):,} examples")
        
        # Apply difficulty filter if specified
        if difficulty_filter:
            original_count = len(self.data)
            self.data = [ex for ex in self.data if ex['difficulty'] == difficulty_filter]
            print(f"Filtered to {len(self.data):,} {difficulty_filter} examples (from {original_count:,})")
        
        # Show difficulty distribution
        self._show_stats()
    
    def _show_stats(self):
        """Show dataset statistics."""
        difficulty_counts = {}
        for ex in self.data:
            diff = ex['difficulty']
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        print(f"\nDifficulty distribution ({self.split}):")
        for diff, count in sorted(difficulty_counts.items()):
            print(f"  {diff}: {count:,} ({count/len(self.data)*100:.1f}%)")
    
    def get_examples(self, shuffle: bool = True) -> List[Dict]:
        """
        Get all examples.
        
        Args:
            shuffle: Whether to shuffle
            
        Returns:
            List of examples
        """
        examples = self.data.copy()
        if shuffle:
            random.shuffle(examples)
        return examples
    
    def get_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_epochs: int = 1
    ) -> Iterator[Dict[str, mx.array]]:
        """
        Get batches of tokenized examples.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle each epoch
            num_epochs: Number of epochs to iterate
            
        Yields:
            Dictionary with input_ids, attention_mask, labels
        """
        for epoch in range(num_epochs):
            # Get examples
            examples = self.get_examples(shuffle=shuffle)
            
            # Batch
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i:i + batch_size]
                
                # Skip incomplete batch at end
                if len(batch_examples) < batch_size:
                    continue
                
                yield self._prepare_batch(batch_examples)
    
    def _prepare_batch(self, examples: List[Dict]) -> Dict[str, mx.array]:
        """
        Tokenize and prepare a batch.
        
        Args:
            examples: List of example dicts
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Build input texts (T5 format)
        input_texts = []
        for ex in examples:
            input_text = f"translate English to SQL: {ex['question']} Schema: {ex['schema']}"
            input_texts.append(input_text)
        
        # Build output texts
        output_texts = [ex['sql'] for ex in examples]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        
        # Tokenize outputs (labels)
        labels = self.tokenizer(
            output_texts,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        
        # Convert to MLX arrays
        return {
            "input_ids": mx.array(inputs["input_ids"]),
            "attention_mask": mx.array(inputs["attention_mask"]),
            "labels": mx.array(labels["input_ids"])
        }
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)


if __name__ == "__main__":
    # Test data loader
    from transformers import T5Tokenizer
    
    print("Testing processed data loader...")
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    try:
        # Test train loader
        train_loader = ProcessedDataLoader(
            data_dir="data/processed",
            tokenizer=tokenizer,
            split="train"
        )
        
        print(f"\nTrain dataset size: {len(train_loader):,}")
        
        # Test batching
        print("\nTesting batch generation...")
        batch_count = 0
        for batch in train_loader.get_batches(batch_size=32):
            batch_count += 1
            if batch_count == 1:
                print(f"Batch shape:")
                print(f"  input_ids: {batch['input_ids'].shape}")
                print(f"  attention_mask: {batch['attention_mask'].shape}")
                print(f"  labels: {batch['labels'].shape}")
            if batch_count >= 3:
                break
        
        print(f"\n✅ Successfully loaded and batched {batch_count} batches")
        
        # Test curriculum filtering
        print("\n\nTesting curriculum filtering (Simple SQL)...")
        simple_loader = ProcessedDataLoader(
            data_dir="data/processed",
            tokenizer=tokenizer,
            split="train",
            difficulty_filter="Simple"
        )
        
        print(f"Simple SQL dataset size: {len(simple_loader):,}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nRun data preparation first:")
        print("  python scripts/prepare_data.py")
