"""
Preprocess text-to-SQL datasets
"""

import argparse
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import load_datasets
from src.data.preprocessing import Tokenizer, SQLPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess text-to-SQL datasets")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input data directory (from download_datasets.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of preprocessing workers"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config["data"]["data_dir"] = args.data_dir
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessors
    print("\nInitializing preprocessors...")
    tokenizer = Tokenizer()
    sql_preprocessor = SQLPreprocessor()
    
    # Process each split
    for split in ["train", "validation"]:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print('='*50)
        
        # Load datasets
        print("Loading datasets...")
        examples = load_datasets(config["data"], split=split)
        print(f"Loaded {len(examples)} examples")
        
        # Preprocess
        print("Preprocessing...")
        processed_examples = []
        
        for example in tqdm(examples):
            # Normalize SQL
            normalized_sql = sql_preprocessor.normalize(example.sql)
            
            # Tokenize
            encoded = tokenizer.encode_example(
                question=example.question,
                schema=example.schema,
                sql=normalized_sql,
                max_length=config["training"]["max_seq_length"],
            )
            
            processed_example = {
                "input_ids": encoded["input_ids"],
                "labels": encoded.get("labels", []),
                "question": example.question,
                "sql": normalized_sql,
                "db_id": example.db_id,
                "source": example.source,
            }
            
            processed_examples.append(processed_example)
        
        # Save
        output_file = output_dir / f"{split}.pkl"
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_examples, f)
        
        print(f"✓ Saved {len(processed_examples)} examples")
    
    # Save tokenizer
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"\n✓ Tokenizer saved to {tokenizer_dir}")
    
    # Save metadata
    metadata = {
        "config": config,
        "num_train_examples": len(processed_examples),
        "max_seq_length": config["training"]["max_seq_length"],
        "vocab_size": tokenizer.vocab_size,
    }
    
    metadata_file = output_dir / "metadata.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)
    
    print(f"\n{'='*50}")
    print("PREPROCESSING COMPLETE")
    print('='*50)
    print(f"\nProcessed data saved to: {output_dir}")
    print("\nNext step: Run train.py to start training")


if __name__ == "__main__":
    main()
