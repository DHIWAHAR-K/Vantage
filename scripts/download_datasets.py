"""
Download text-to-SQL datasets
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import json


def download_spider(output_dir: Path):
    """Download Spider dataset"""
    print("Downloading Spider...")
    
    try:
        train_data = load_dataset("spider", split="train")
        val_data = load_dataset("spider", split="validation")
        
        spider_dir = output_dir / "spider"
        spider_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(spider_dir / "train.json", 'w') as f:
            json.dump([dict(ex) for ex in train_data], f)
        
        with open(spider_dir / "dev.json", 'w') as f:
            json.dump([dict(ex) for ex in val_data], f)
        
        print(f"✓ Spider downloaded to {spider_dir}")
    except Exception as e:
        print(f"✗ Spider download failed: {e}")


def download_wikisql(output_dir: Path):
    """Download WikiSQL dataset"""
    print("Downloading WikiSQL...")
    
    try:
        # WikiSQL is large, so we just verify it's accessible
        train_data = load_dataset("wikisql", split="train", streaming=True)
        
        wikisql_dir = output_dir / "wikisql"
        wikisql_dir.mkdir(parents=True, exist_ok=True)
        
        # Create marker file
        with open(wikisql_dir / "README.txt", 'w') as f:
            f.write("WikiSQL will be streamed from HuggingFace during training.\n")
        
        print(f"✓ WikiSQL configured for streaming")
    except Exception as e:
        print(f"✗ WikiSQL setup failed: {e}")


def download_gretel(output_dir: Path):
    """Download Gretel synthetic dataset"""
    print("Downloading Gretel Synthetic...")
    
    try:
        gretel_dir = output_dir / "gretel"
        gretel_dir.mkdir(parents=True, exist_ok=True)
        
        # Large dataset, stream during training
        with open(gretel_dir / "README.txt", 'w') as f:
            f.write("Gretel Synthetic will be streamed from HuggingFace during training.\n")
            f.write("Dataset: gretelai/synthetic_text_to_sql\n")
        
        print(f"✓ Gretel configured for streaming")
    except Exception as e:
        print(f"✗ Gretel setup failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download text-to-SQL datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["spider", "wikisql", "gretel"],
        choices=["spider", "bird", "wikisql", "gretel", "all"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading datasets to {output_dir}")
    print("=" * 50)
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["spider", "wikisql", "gretel"]
    
    for dataset in datasets:
        if dataset == "spider":
            download_spider(output_dir)
        elif dataset == "wikisql":
            download_wikisql(output_dir)
        elif dataset == "gretel":
            download_gretel(output_dir)
        else:
            print(f"Unknown dataset: {dataset}")
    
    print("=" * 50)
    print("Dataset download complete!")
    print(f"\nDatasets saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run preprocess_data.py to prepare data for training")
    print("2. Run train.py to start training")


if __name__ == "__main__":
    main()
