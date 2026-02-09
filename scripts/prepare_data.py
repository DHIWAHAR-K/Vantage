"""
Prepare SynSQL data for T5 fine-tuning.

This script:
1. Loads SynSQL data (2.5M examples)
2. Integrates schemas from tables.json
3. Creates train/validation split
4. Saves processed data for training

Run once before training.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_synsql_data(synsql_dir: str) -> tuple[List[Dict], Dict[str, str]]:
    """
    Load SynSQL data and schemas.
    
    Args:
        synsql_dir: Path to synsql directory
        
    Returns:
        Tuple of (examples_list, schemas_dict)
    """
    data_file = Path(synsql_dir) / "data.json"
    tables_file = Path(synsql_dir) / "tables.json"
    
    print("Loading SynSQL data...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} examples")
    
    print("\nLoading database schemas...")
    with open(tables_file, 'r') as f:
        tables_data = json.load(f)
    
    # Build schema lookup
    schemas = {}
    for table_info in tables_data:
        db_id = table_info['db_id']
        table_names = table_info['table_names']
        column_names = table_info['column_names']
        
        # Build schema string
        schema_parts = []
        for table_idx, table_name in enumerate(table_names):
            cols = [col_name for t_idx, col_name in column_names 
                   if t_idx == table_idx and col_name != '*']
            if cols:
                # Limit to 8 columns to keep input short
                schema_parts.append(f"{table_name}({', '.join(cols[:8])})")
        
        schemas[db_id] = " | ".join(schema_parts) if schema_parts else "no_schema"
    
    print(f"Loaded schemas for {len(schemas):,} databases")
    
    return data, schemas


def create_train_val_split(
    data: List[Dict],
    schemas: Dict[str, str],
    val_ratio: float = 0.05,
    subset_size: int = None
) -> tuple[List[Dict], List[Dict]]:
    """
    Create train/validation split with difficulty balance.
    
    Args:
        data: List of SynSQL examples
        schemas: Schema lookup dict
        val_ratio: Validation set ratio (default 5%)
        subset_size: If set, use only this many examples
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    print("\nCreating train/validation split...")
    
    # Apply subset limit if specified
    if subset_size and subset_size < len(data):
        print(f"Using subset of {subset_size:,} examples (from {len(data):,})")
        data = data[:subset_size]
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Integrate schemas
    print("Integrating schemas...")
    
    train_examples = []
    for ex in train_data:
        db_id = ex.get('db_id', 'unknown')
        train_examples.append({
            'question': ex.get('question', ''),
            'schema': schemas.get(db_id, 'no_schema'),
            'sql': ex.get('sql', ''),
            'db_id': db_id,
            'difficulty': ex.get('sql_complexity', 'Medium')
        })
    
    val_examples = []
    for ex in val_data:
        db_id = ex.get('db_id', 'unknown')
        val_examples.append({
            'question': ex.get('question', ''),
            'schema': schemas.get(db_id, 'no_schema'),
            'sql': ex.get('sql', ''),
            'db_id': db_id,
            'difficulty': ex.get('sql_complexity', 'Medium')
        })
    
    print(f"\nSplit complete:")
    print(f"  Training: {len(train_examples):,} examples")
    print(f"  Validation: {len(val_examples):,} examples")
    
    # Show difficulty distribution
    train_diff = {}
    for ex in train_examples:
        diff = ex['difficulty']
        train_diff[diff] = train_diff.get(diff, 0) + 1
    
    print(f"\nTraining set difficulty distribution:")
    for diff, count in sorted(train_diff.items()):
        print(f"  {diff}: {count:,} ({count/len(train_examples)*100:.1f}%)")
    
    return train_examples, val_examples


def save_processed_data(
    train_examples: List[Dict],
    val_examples: List[Dict],
    output_dir: str
):
    """
    Save processed data to JSON files.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed data to {output_path}...")
    
    # Save train
    train_file = output_path / "train.json"
    with open(train_file, 'w') as f:
        json.dump(train_examples, f, indent=2)
    print(f"  ✅ Saved {len(train_examples):,} training examples")
    
    # Save validation
    val_file = output_path / "val.json"
    with open(val_file, 'w') as f:
        json.dump(val_examples, f, indent=2)
    print(f"  ✅ Saved {len(val_examples):,} validation examples")
    
    # Save summary stats
    stats = {
        'total_examples': len(train_examples) + len(val_examples),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'val_ratio': len(val_examples) / (len(train_examples) + len(val_examples)),
        'difficulty_distribution': {}
    }
    
    for ex in train_examples:
        diff = ex['difficulty']
        if diff not in stats['difficulty_distribution']:
            stats['difficulty_distribution'][diff] = {'train': 0, 'val': 0}
        stats['difficulty_distribution'][diff]['train'] += 1
    
    for ex in val_examples:
        diff = ex['difficulty']
        if diff not in stats['difficulty_distribution']:
            stats['difficulty_distribution'][diff] = {'train': 0, 'val': 0}
        stats['difficulty_distribution'][diff]['val'] += 1
    
    stats_file = output_path / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ Saved statistics")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)


def main():
    """Main data preparation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SynSQL data for training")
    parser.add_argument(
        "--synsql-dir",
        type=str,
        default="data/synsql",
        help="Path to SynSQL directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000000,
        help="Use subset of examples (default: 5M)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation set ratio (default: 0.05 = 5%%)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNSQL DATA PREPARATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  SynSQL directory: {args.synsql_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Subset size: {args.subset_size:,} examples")
    print(f"  Validation ratio: {args.val_ratio*100:.1f}%")
    
    # Load data
    data, schemas = load_synsql_data(args.synsql_dir)
    
    # Create split
    train_examples, val_examples = create_train_val_split(
        data,
        schemas,
        val_ratio=args.val_ratio,
        subset_size=args.subset_size
    )
    
    # Save
    save_processed_data(train_examples, val_examples, args.output_dir)
    
    print(f"\nProcessed data saved to: {args.output_dir}")
    print("\n✅ Ready for training!")
    print(f"\nNext step: python scripts/finetune.py")


if __name__ == "__main__":
    main()
