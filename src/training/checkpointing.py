"""
Model checkpointing and state management
"""

import mlx.core as mx
from pathlib import Path
import json
import shutil
from typing import Dict, Optional, Any
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints during training.
    
    Features:
    - Save/load model weights
    - Save/load optimizer state
    - Save/load scheduler state
    - Track best model by metric
    - Keep only K best checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_checkpoint_max: int = 5,
        save_optimizer: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_optimizer = save_optimizer
        
        # Track checkpoints and metrics
        self.checkpoints: list = []  # List of (step, path, metric)
        self.best_metric: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None
    
    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer,
        scheduler,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint to disk.
        
        Args:
            step: Current training step
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Scheduler instance
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{step}"
        if is_best:
            checkpoint_name = "checkpoint-best"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_weights = dict(model.parameters())
        mx.savez(str(checkpoint_path / "model.npz"), **model_weights)
        
        # Save model config
        if hasattr(model, 'config'):
            with open(checkpoint_path / "config.json", 'w') as f:
                json.dump(model.config.to_dict(), f, indent=2)
        
        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            optimizer_state = {
                "step_count": getattr(optimizer, 'step_count', 0),
                "learning_rate": optimizer.learning_rate,
            }
            
            # Save optimizer moments if available
            if hasattr(optimizer, 'm'):
                mx.savez(
                    str(checkpoint_path / "optimizer.npz"),
                    **{f"m_{k}": v for k, v in optimizer.m.items()},
                    **{f"v_{k}": v for k, v in optimizer.v.items()},
                )
            
            with open(checkpoint_path / "optimizer_state.json", 'w') as f:
                json.dump(optimizer_state, f, indent=2)
        
        # Save scheduler state
        if scheduler is not None:
            scheduler_state = scheduler.state_dict()
            with open(checkpoint_path / "scheduler_state.json", 'w') as f:
                json.dump(scheduler_state, f, indent=2)
        
        # Save training metrics
        checkpoint_info = {
            "step": step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(checkpoint_path / "checkpoint_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        # Track checkpoint
        if not is_best:
            metric_value = metrics.get("eval_loss", metrics.get("loss", 0.0))
            self.checkpoints.append((step, checkpoint_path, metric_value))
            
            # Remove old checkpoints if exceeding limit
            if len(self.checkpoints) > self.keep_checkpoint_max:
                self._remove_old_checkpoints()
        else:
            self.best_checkpoint_path = checkpoint_path
        
        print(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model instance to load weights into
            optimizer: Optimizer instance to load state into
            scheduler: Scheduler instance to load state into
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model weights
        model_weights = mx.load(str(checkpoint_path / "model.npz"))
        model.load_weights(list(model_weights.items()))
        
        # Load optimizer state
        if load_optimizer and optimizer is not None:
            optimizer_state_path = checkpoint_path / "optimizer_state.json"
            if optimizer_state_path.exists():
                with open(optimizer_state_path, 'r') as f:
                    optimizer_state = json.load(f)
                
                optimizer.learning_rate = optimizer_state.get("learning_rate", optimizer.learning_rate)
                
                # Load optimizer moments
                optimizer_path = checkpoint_path / "optimizer.npz"
                if optimizer_path.exists():
                    optimizer_arrays = mx.load(str(optimizer_path))
                    
                    # Restore m and v
                    if hasattr(optimizer, 'm'):
                        for key, value in optimizer_arrays.items():
                            if key.startswith("m_"):
                                param_name = key[2:]  # Remove 'm_' prefix
                                optimizer.m[param_name] = value
                            elif key.startswith("v_"):
                                param_name = key[2:]  # Remove 'v_' prefix
                                optimizer.v[param_name] = value
        
        # Load scheduler state
        if scheduler is not None:
            scheduler_state_path = checkpoint_path / "scheduler_state.json"
            if scheduler_state_path.exists():
                with open(scheduler_state_path, 'r') as f:
                    scheduler_state = json.load(f)
                scheduler.load_state_dict(scheduler_state)
        
        # Load checkpoint info
        info_path = checkpoint_path / "checkpoint_info.json"
        checkpoint_info = {}
        if info_path.exists():
            with open(info_path, 'r') as f:
                checkpoint_info = json.load(f)
        
        return checkpoint_info
    
    def _remove_old_checkpoints(self):
        """Remove oldest checkpoints to stay within limit"""
        # Sort by metric (lower is better for loss)
        self.checkpoints.sort(key=lambda x: x[2])
        
        # Remove worst checkpoints
        while len(self.checkpoints) > self.keep_checkpoint_max:
            _, checkpoint_path, _ = self.checkpoints.pop()
            
            # Don't remove if it's the best checkpoint
            if checkpoint_path != self.best_checkpoint_path:
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    print(f"Removed checkpoint: {checkpoint_path}")
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint_path
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint"""
        if not self.checkpoints:
            return None
        
        # Sort by step number
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[0], reverse=True)
        return sorted_checkpoints[0][1]
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        return [(step, str(path), metric) for step, path, metric in self.checkpoints]


def save_model_for_inference(
    model,
    tokenizer,
    save_path: str,
):
    """
    Save model in format optimized for inference.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        save_path: Directory to save model
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_weights = dict(model.parameters())
    mx.savez(str(save_path / "weights.npz"), **model_weights)
    
    # Save config
    if hasattr(model, 'config'):
        with open(save_path / "config.json", 'w') as f:
            json.dump(model.config.to_dict(), f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(save_path))
    
    # Save metadata
    metadata = {
        "model_type": "vantage-text2sql",
        "framework": "mlx",
        "version": "0.1.0",
        "saved_at": datetime.now().isoformat(),
    }
    
    with open(save_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved model for inference to {save_path}")


def load_model_for_inference(
    model_path: str,
    model_class,
):
    """
    Load model for inference.
    
    Args:
        model_path: Path to saved model
        model_class: Model class to instantiate
        
    Returns:
        Loaded model instance
    """
    model_path = Path(model_path)
    
    # Load config
    with open(model_path / "config.json", 'r') as f:
        config_dict = json.load(f)
    
    # Create model
    from ..models.text2sql_model import VantageConfig
    config = VantageConfig.from_dict(config_dict)
    model = model_class(config)
    
    # Load weights
    weights = mx.load(str(model_path / "weights.npz"))
    model.load_weights(list(weights.items()))
    
    return model
