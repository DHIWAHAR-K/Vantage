"""
Export Vantage model to HuggingFace format
"""

import argparse
from pathlib import Path
import sys
import json
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.text2sql_model import VantageModel
from src.data.preprocessing import Tokenizer


def create_model_card(model_config: dict, output_dir: Path):
    """Create HuggingFace model card"""
    
    model_card = f"""---
language: en
license: mit
tags:
- text-to-sql
- sql
- database
- mixture-of-experts
- mlx
datasets:
- spider
- wikisql
metrics:
- exact_match
- execution_accuracy
---

# {model_config['name']}

## Model Description

{model_config['name']} is a Mixture of Experts (MoE) model for converting natural language questions into SQL queries.

**Model Details:**
- Parameters: {model_config.get('total_params', 'N/A')}
- Active Parameters: ~{model_config.get('active_params', 'N/A')} per forward pass
- Experts: {model_config.get('num_experts', 'N/A')}
- Hidden Size: {model_config.get('hidden_size', 'N/A')}
- Layers: {model_config.get('num_layers', 'N/A')}

## Intended Use

This model is designed for:
- Converting natural language questions to SQL queries
- Database query assistance
- SQL learning and education
- Building natural language database interfaces

## How to Use

```python
from vantage import VantageAPI

# Load model
api = VantageAPI.from_pretrained("{model_config['name']}")

# Define schema
schema = \"\"\"
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL
);
\"\"\"

# Generate SQL
question = "What is the average salary by department?"
sql = api.generate(question, schema)
print(sql)
```

## Training Data

Trained on:
- Spider: Complex multi-table queries
- BIRD-SQL: Cross-domain generalization
- WikiSQL: Simple single-table queries
- Gretel Synthetic: Augmentation data

## Performance

See full evaluation results in the model repository.

## Limitations

- English language only
- May hallucinate non-existent schema elements
- Complex nested queries may be challenging
- Always validate generated SQL before execution

## Citation

```bibtex
@software{{vantage2024,
  title={{Vantage: Text-to-SQL with Mixture of Experts}},
  author={{DHIWAHAR-K}},
  year={{2024}},
  url={{https://github.com/DHIWAHAR-K/Vantage}}
}}
```

## Contact

- Author: DHIWAHAR-K
- Email: adhithyak99@gmail.com
- GitHub: https://github.com/DHIWAHAR-K/Vantage
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(model_card)


def main():
    parser = argparse.ArgumentParser(description="Export model to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID (username/model-name)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("EXPORTING MODEL TO HUGGINGFACE FORMAT")
    print("="*50)
    print(f"\nModel: {model_path}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = VantageModel.from_pretrained(str(model_path))
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained(str(model_path))
    
    # Save in HuggingFace format
    print("\nSaving model...")
    model.save_pretrained(str(output_dir))
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_dir))
    
    # Create model card
    print("Creating model card...")
    model_config = {
        "name": model.config.name,
        "num_experts": model.config.num_experts,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_layers,
    }
    create_model_card(model_config, output_dir)
    
    # Copy additional files
    print("Copying documentation...")
    docs_to_copy = [
        "ARCHITECTURE.md",
        "TRAINING.md",
        "EVALUATION.md",
        "API.md",
    ]
    
    docs_dir = Path(__file__).parent.parent / "docs"
    for doc in docs_to_copy:
        src = docs_dir / doc
        if src.exists():
            shutil.copy(src, output_dir / doc)
    
    print("\n" + "="*50)
    print("EXPORT COMPLETE")
    print("="*50)
    print(f"\nModel exported to: {output_dir}")
    
    # Push to hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            print("\nError: --repo_id required for pushing to hub")
            return
        
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=args.repo_id,
                repo_type="model",
            )
            
            print(f"✓ Model pushed to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"✗ Push failed: {e}")
            print("You may need to run: huggingface-cli login")
    
    print("\nNext steps:")
    print("1. Test the exported model locally")
    print("2. Push to HuggingFace Hub with --push_to_hub")
    print("3. Create a demo with scripts/demo.py")


if __name__ == "__main__":
    main()
