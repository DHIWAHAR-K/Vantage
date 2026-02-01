"""
Learning rate schedulers for training
"""

import math
from typing import Callable, Optional


class LRScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
    
    def get_lr(self) -> float:
        """Compute current learning rate"""
        raise NotImplementedError
    
    def state_dict(self) -> dict:
        """Get scheduler state"""
        return {
            "step_count": self.step_count,
            "base_lr": self.base_lr,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state"""
        self.step_count = state_dict["step_count"]
        self.base_lr = state_dict["base_lr"]


class LinearWarmupCosineDecay(LRScheduler):
    """
    Linear warmup followed by cosine decay.
    
    Common schedule for transformer training:
    - Linear increase from 0 to base_lr over warmup_steps
    - Cosine decay from base_lr to min_lr over remaining steps
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base LR
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = self.base_lr * min_lr_ratio
    
    def get_lr(self) -> float:
        """Compute learning rate for current step"""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class LinearWarmup(LRScheduler):
    """
    Linear warmup to base learning rate.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
    """
    
    def __init__(self, optimizer, warmup_steps: int):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
    
    def get_lr(self) -> float:
        """Compute learning rate"""
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count / self.warmup_steps)
        else:
            return self.base_lr


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    Args:
        optimizer: Optimizer instance
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self) -> float:
        """Compute learning rate"""
        cosine = 0.5 * (1 + math.cos(math.pi * self.step_count / self.T_max))
        return self.eta_min + (self.base_lr - self.eta_min) * cosine


class PolynomialDecayLR(LRScheduler):
    """
    Polynomial decay learning rate schedule.
    
    Args:
        optimizer: Optimizer instance
        total_steps: Total training steps
        end_lr: Final learning rate
        power: Polynomial power
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        end_lr: float = 0.0,
        power: float = 1.0,
    ):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
    
    def get_lr(self) -> float:
        """Compute learning rate"""
        progress = min(self.step_count / self.total_steps, 1.0)
        decay = (1 - progress) ** self.power
        return self.end_lr + (self.base_lr - self.end_lr) * decay


class InverseSquareRootLR(LRScheduler):
    """
    Inverse square root learning rate schedule.
    
    Used in "Attention is All You Need" paper.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
    """
    
    def __init__(self, optimizer, warmup_steps: int):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.decay_factor = self.base_lr * (warmup_steps ** 0.5)
    
    def get_lr(self) -> float:
        """Compute learning rate"""
        step = max(self.step_count, 1)
        
        if step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)
        else:
            return self.decay_factor * (step ** -0.5)


def create_scheduler(
    optimizer,
    config: dict,
) -> LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
        
    Returns:
        Scheduler instance
    """
    scheduler_type = config.get("lr_scheduler_type", "cosine").lower()
    
    warmup_steps = config.get("warmup_steps", 2000)
    total_steps = config.get("num_train_steps", 100000)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)
    
    if scheduler_type == "cosine":
        scheduler = LinearWarmupCosineDecay(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "linear":
        scheduler = PolynomialDecayLR(
            optimizer=optimizer,
            total_steps=total_steps,
            end_lr=config.get("learning_rate", 3e-4) * min_lr_ratio,
            power=1.0,
        )
    elif scheduler_type == "polynomial":
        scheduler = PolynomialDecayLR(
            optimizer=optimizer,
            total_steps=total_steps,
            end_lr=config.get("learning_rate", 3e-4) * min_lr_ratio,
            power=config.get("poly_power", 2.0),
        )
    elif scheduler_type == "inverse_sqrt":
        scheduler = InverseSquareRootLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
        )
    elif scheduler_type == "constant":
        # Just warmup, then constant
        scheduler = LinearWarmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class WarmupScheduler:
    """
    Simple warmup scheduler that can be composed with other schedulers.
    """
    
    def __init__(
        self,
        base_scheduler: LRScheduler,
        warmup_steps: int,
    ):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            warmup_factor = self.step_count / self.warmup_steps
            self.base_scheduler.optimizer.learning_rate = (
                self.base_scheduler.base_lr * warmup_factor
            )
        else:
            # Regular schedule
            self.base_scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.base_scheduler.optimizer.learning_rate
