"""
Main training script for Vantage models
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.text2sql_model import VantageModel, VantageConfig
from src.data.preprocessing import Tokenizer
from src.data.dataset_loader import load_datasets
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Vantage text-to-SQL model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.wandb_project:
        config["training"]["wandb_project"] = args.wandb_project
        config["training"]["use_wandb"] = True
    if args.wandb_run_name:
        config["training"]["wandb_run_name"] = args.wandb_run_name
    
    # Set seed
    import mlx.core as mx
    mx.random.seed(args.seed)
    
    # Create model
    print("\nInitializing model...")
    model_config = VantageConfig.from_dict(config["model"])
    model = VantageModel(model_config)
    
    # Count parameters
    total_params = sum(p.size for p in model.parameters().values())
    print(f"Model: {model_config.name}")
    print(f"Total parameters: {total_params:,}")
    
    # Create tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = Tokenizer()
    
    # Load datasets
    print(f"\nLoading datasets from {args.data_dir}...")
    train_dataset = load_datasets(
        config["data"],
        split="train"
    )
    
    eval_dataset = load_datasets(
        config["data"],
        split="validation"
    )
    
    if args.debug:
        print("DEBUG MODE: Using small dataset")
        train_dataset = train_dataset[:100]
        eval_dataset = eval_dataset[:20]
        config["training"]["num_train_steps"] = 1000
        config["training"]["eval_steps"] = 100
        config["training"]["logging_steps"] = 10
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config["training"],
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50 + "\n")
    
    trainer.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"Best checkpoint: {trainer.checkpoint_manager.get_best_checkpoint()}")


if __name__ == "__main__":
    main()
