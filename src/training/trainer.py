"""
Main training loop for Vantage model
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from pathlib import Path

from .optimizer import create_optimizer, clip_gradients, GradientAccumulator, create_loss_function
from .scheduler import create_scheduler
from .checkpointing import CheckpointManager


class Trainer:
    """
    Main trainer for Vantage text-to-SQL model.
    
    Handles:
    - Training loop with gradient accumulation
    - Evaluation on validation set
    - Checkpointing
    - Learning rate scheduling
    - Logging (W&B integration optional)
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        config: Dict,
        tokenizer=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer
        
        # Training hyperparameters
        self.batch_size = config.get("batch_size", 16)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.num_train_steps = config.get("num_train_steps", 100000)
        self.eval_steps = config.get("eval_steps", 1000)
        self.logging_steps = config.get("logging_steps", 100)
        self.save_steps = config.get("save_steps", 5000)
        
        # Mixed precision
        self.use_mixed_precision = config.get("mixed_precision") == "bf16"
        
        # Create optimizer
        self.optimizer = create_optimizer(
            config,
            model.parameters(),
        )
        
        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            config,
        )
        
        # Gradient accumulation
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=self.gradient_accumulation_steps
        )
        
        # Loss function
        self.loss_fn = create_loss_function(
            aux_loss_coef=config.get("router_aux_loss_coef", 0.01),
        )
        
        # Checkpointing
        output_dir = config.get("output_dir", "./checkpoints")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=output_dir,
            keep_checkpoint_max=config.get("keep_checkpoint_max", 5),
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_metric = float('inf')
        
        # Logging
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb(config)
    
    def _init_wandb(self, config: Dict):
        """Initialize Weights & Biases logging"""
        try:
            import wandb
            
            wandb.init(
                project=config.get("wandb_project", "vantage-text2sql"),
                name=config.get("wandb_run_name", f"run-{time.strftime('%Y%m%d-%H%M%S')}"),
                config=config,
            )
            self.wandb = wandb
        except ImportError:
            print("W&B not available, skipping")
            self.use_wandb = False
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_train_steps} steps")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
        # Training loop
        train_iterator = self._get_train_iterator()
        
        progress_bar = tqdm(
            total=self.num_train_steps,
            initial=self.global_step,
            desc="Training",
        )
        
        running_loss = 0.0
        running_aux_loss = 0.0
        start_time = time.time()
        
        while self.global_step < self.num_train_steps:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reset iterator
                train_iterator = self._get_train_iterator()
                batch = next(train_iterator)
                self.epoch += 1
            
            # Forward and backward pass
            loss, loss_components = self._training_step(batch)
            
            # Accumulate metrics
            running_loss += loss_components["lm_loss"].item()
            if "aux_loss" in loss_components:
                running_aux_loss += loss_components["aux_loss"].item()
            
            # Update if accumulated enough gradients
            if self.gradient_accumulator.should_update():
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = running_loss / self.logging_steps
                    avg_aux_loss = running_aux_loss / self.logging_steps
                    elapsed_time = time.time() - start_time
                    throughput = self.logging_steps / elapsed_time
                    
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/aux_loss": avg_aux_loss,
                        "train/learning_rate": self.optimizer.learning_rate,
                        "train/throughput": throughput,
                        "train/epoch": self.epoch,
                    }
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{self.optimizer.learning_rate:.2e}",
                    })
                    
                    if self.use_wandb:
                        self.wandb.log(log_dict, step=self.global_step)
                    
                    running_loss = 0.0
                    running_aux_loss = 0.0
                    start_time = time.time()
                
                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    print(f"\nStep {self.global_step} - Eval metrics:")
                    for key, value in eval_metrics.items():
                        print(f"  {key}: {value:.4f}")
                    
                    if self.use_wandb:
                        self.wandb.log(
                            {f"eval/{k}": v for k, v in eval_metrics.items()},
                            step=self.global_step,
                        )
                    
                    # Check if best model
                    eval_loss = eval_metrics.get("loss", float('inf'))
                    is_best = eval_loss < self.best_eval_metric
                    if is_best:
                        self.best_eval_metric = eval_loss
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0 or self.global_step == self.num_train_steps:
                    is_best = eval_metrics.get("loss", float('inf')) < self.best_eval_metric if 'eval_metrics' in locals() else False
                    
                    self.checkpoint_manager.save_checkpoint(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics=eval_metrics if 'eval_metrics' in locals() else {"loss": running_loss / self.logging_steps},
                        is_best=is_best,
                    )
        
        progress_bar.close()
        print("Training complete!")
        
        # Save final model
        self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={"loss": running_loss / self.logging_steps},
            is_best=False,
        )
    
    def _training_step(self, batch: Dict) -> Tuple[mx.array, Dict]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value and loss components
        """
        # Forward pass
        def forward_and_loss(params):
            # Update model parameters
            self.model.update(params)
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                schema_ids=batch.get("schema_ids"),
                attention_mask=batch.get("attention_mask"),
                schema_mask=batch.get("schema_mask"),
                training=True,
            )
            
            # Compute loss
            loss, loss_components = self.loss_fn(outputs, batch["labels"])
            
            return loss, loss_components
        
        # Compute gradients
        params = dict(self.model.parameters())
        (loss, loss_components), gradients = nn.value_and_grad(
            forward_and_loss,
            argnums=0,
        )(params)
        
        # Clip gradients
        gradients, grad_norm = clip_gradients(gradients, self.max_grad_norm)
        
        # Accumulate gradients
        self.gradient_accumulator.accumulate(gradients)
        
        # Update if ready
        if self.gradient_accumulator.should_update():
            accumulated_grads = self.gradient_accumulator.get_accumulated_gradients()
            
            # Optimizer step
            updated_params = self.optimizer.apply_gradients(accumulated_grads, params)
            self.model.update(updated_params)
            
            # Scheduler step
            self.scheduler.step()
            
            # Reset accumulator
            self.gradient_accumulator.reset()
        
        return loss, loss_components
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nRunning evaluation...")
        
        self.model.eval()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        total_samples = 0
        
        eval_iterator = self._get_eval_iterator()
        
        for batch in tqdm(eval_iterator, desc="Evaluating"):
            # Forward pass (no gradients)
            outputs = self.model(
                input_ids=batch["input_ids"],
                schema_ids=batch.get("schema_ids"),
                attention_mask=batch.get("attention_mask"),
                schema_mask=batch.get("schema_mask"),
                training=False,
            )
            
            # Compute loss
            loss, loss_components = self.loss_fn(outputs, batch["labels"])
            
            batch_size = batch["input_ids"].shape[0]
            total_loss += loss.item() * batch_size
            if "aux_loss" in loss_components:
                total_aux_loss += loss_components["aux_loss"].item() * batch_size
            total_samples += batch_size
        
        self.model.train()
        
        metrics = {
            "loss": total_loss / total_samples,
            "aux_loss": total_aux_loss / total_samples if total_aux_loss > 0 else 0.0,
            "perplexity": mx.exp(total_loss / total_samples).item(),
        }
        
        return metrics
    
    def _get_train_iterator(self):
        """Create training data iterator"""
        # Simple iterator over dataset
        # In practice, would use DataLoader with shuffling, etc.
        
        indices = list(range(len(self.train_dataset)))
        
        while True:
            # Shuffle
            import random
            random.shuffle(indices)
            
            # Yield batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_examples = [self.train_dataset[idx] for idx in batch_indices]
                
                # Collate batch
                batch = self._collate_batch(batch_examples)
                
                yield batch
    
    def _get_eval_iterator(self):
        """Create evaluation data iterator"""
        for i in range(0, len(self.eval_dataset), self.batch_size):
            batch_examples = [
                self.eval_dataset[j]
                for j in range(i, min(i + self.batch_size, len(self.eval_dataset)))
            ]
            
            batch = self._collate_batch(batch_examples)
            yield batch
    
    def _collate_batch(self, examples: List[Dict]) -> Dict[str, mx.array]:
        """
        Collate examples into batch.
        
        Args:
            examples: List of examples
            
        Returns:
            Batched tensors
        """
        # Use tokenizer to batch encode
        if self.tokenizer is not None:
            return self.tokenizer.batch_encode(
                examples,
                max_length=self.config.get("max_seq_length", 2048),
            )
        else:
            # Assume examples are already tokenized
            # Pad to max length in batch
            max_len = max(len(ex["input_ids"]) for ex in examples)
            
            batch_input_ids = []
            batch_labels = []
            batch_attention_mask = []
            
            for ex in examples:
                input_ids = ex["input_ids"]
                labels = ex.get("labels", input_ids)
                
                # Pad
                padding_length = max_len - len(input_ids)
                padded_input_ids = input_ids + [0] * padding_length
                padded_labels = labels + [-100] * padding_length
                attention_mask = [1] * len(input_ids) + [0] * padding_length
                
                batch_input_ids.append(padded_input_ids)
                batch_labels.append(padded_labels)
                batch_attention_mask.append(attention_mask)
            
            return {
                "input_ids": mx.array(batch_input_ids),
                "labels": mx.array(batch_labels),
                "attention_mask": mx.array(batch_attention_mask),
            }
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            load_optimizer=True,
        )
        
        self.global_step = checkpoint_info.get("step", 0)
        print(f"Resumed training from step {self.global_step}")
