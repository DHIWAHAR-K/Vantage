"""
Fine-tune T5-small on SynSQL dataset.

Main training script with curriculum learning and thermal management.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetune_t5 import T5Finetuner, FinetuneConfig


def load_config(config_path: str) -> FinetuneConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML
        
    Returns:
        FinetuneConfig instance
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract training config
    training = config_dict.get('training', {})
    curriculum = training.get('curriculum', {})
    
    # Create config
    config = FinetuneConfig(
        base_model=config_dict.get('model', {}).get('base_model', 't5-small'),
        batch_size=training.get('batch_size', 32),
        gradient_accumulation=training.get('gradient_accumulation', 8),
        learning_rate=training.get('learning_rate', 5e-5),
        weight_decay=training.get('weight_decay', 0.01),
        warmup_steps=training.get('warmup_steps', 5000),
        max_steps=training.get('max_steps', 100000),
        max_input_length=training.get('max_seq_length', 256),
        max_output_length=training.get('max_output_length', 128),
        subset_size=training.get('synsql_subset_size', 5000000),
        shuffle_buffer=training.get('shuffle_buffer', 10000),
        phase1_steps=curriculum.get('phase1_steps', 40000),
        phase2_steps=curriculum.get('phase2_steps', 35000),
        phase3_steps=curriculum.get('phase3_steps', 25000),
        eval_every=training.get('eval_every', 10000),
        save_every=training.get('save_every', 10000),
        cooling_break_every=training.get('cooling_break_every', 20000),
        cooling_break_duration=training.get('cooling_break_duration', 120),
        checkpoint_dir=training.get('checkpoint_dir', 'checkpoints'),
        log_dir=training.get('log_dir', 'logs')
    )
    
    return config


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune T5-small on SynSQL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/t5_finetune.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create finetuner
    print("Initializing fine-tuner...")
    finetuner = T5Finetuner(config)
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
        print("Checkpoint loading not yet implemented")
    
    # Start training
    print("\nStarting training...")
    try:
        finetuner.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        finetuner._save_checkpoint(name="interrupted")
        print("Checkpoint saved. You can resume later with --resume")
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save checkpoint...")
        try:
            finetuner._save_checkpoint(name="error")
            print("Emergency checkpoint saved")
        except:
            print("Could not save checkpoint")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
