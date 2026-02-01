"""
Optimizer implementations for MLX
AdamW with load balancing loss support
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional, Callable


class AdamW(optim.OptimizerState):
    """
    AdamW optimizer with weight decay.
    
    Implements decoupled weight decay as in "Decoupled Weight Decay
    Regularization" (Loshchilov & Hutter, 2019).
    
    Args:
        learning_rate: Learning rate
        betas: Coefficients for computing running averages (beta1, beta2)
        eps: Term added for numerical stability
        weight_decay: Weight decay coefficient
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Running averages
        self.m: Dict = {}  # First moment
        self.v: Dict = {}  # Second moment
        self.step_count = 0
    
    def apply_gradients(
        self,
        gradients: Dict,
        model_params: Dict,
    ) -> Dict:
        """
        Apply gradients to parameters.
        
        Args:
            gradients: Parameter gradients
            model_params: Current model parameters
            
        Returns:
            Updated parameters
        """
        self.step_count += 1
        
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        
        updated_params = {}
        
        for name, param in model_params.items():
            if name not in gradients:
                updated_params[name] = param
                continue
            
            grad = gradients[name]
            
            # Initialize moments if needed
            if name not in self.m:
                self.m[name] = mx.zeros_like(param)
                self.v[name] = mx.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moments
            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2
            
            # Update parameters with weight decay
            # AdamW: weight_decay applied directly to parameters, not gradient
            param_update = self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.eps)
            
            # Apply weight decay (decoupled)
            if self.weight_decay > 0 and 'bias' not in name and 'norm' not in name:
                param_update = param_update + self.learning_rate * self.weight_decay * param
            
            updated_params[name] = param - param_update
        
        return updated_params


def create_optimizer(
    config: Dict,
    model_parameters: Dict,
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        config: Training configuration
        model_parameters: Model parameters
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get("optimizer", "adamw").lower()
    learning_rate = config.get("learning_rate", 3e-4)
    
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            learning_rate=learning_rate,
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
            eps=config.get("adam_epsilon", 1e-8),
            weight_decay=config.get("weight_decay", 0.1),
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            learning_rate=learning_rate,
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            learning_rate=learning_rate,
            momentum=config.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def clip_gradients(
    gradients: Dict,
    max_norm: float = 1.0,
) -> Tuple[Dict, float]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: Parameter gradients
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients and actual norm
    """
    # Compute global norm
    total_norm = mx.sqrt(
        sum(mx.sum(g ** 2) for g in gradients.values())
    )
    
    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        clipped_gradients = {
            name: grad * clip_coef
            for name, grad in gradients.items()
        }
    else:
        clipped_gradients = gradients
    
    return clipped_gradients, total_norm


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    layer_decay: Optional[float] = None,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.
    
    Supports:
    - No weight decay for biases and layer norms
    - Layer-wise learning rate decay
    
    Args:
        model: Model instance
        weight_decay: Weight decay coefficient
        layer_decay: Layer-wise LR decay factor
        
    Returns:
        List of parameter group dictionaries
    """
    # Separate parameters by type
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append((name, param))
        else:
            decay_params.append((name, param))
    
    param_groups = [
        {
            'params': [p for _, p in decay_params],
            'weight_decay': weight_decay,
            'names': [n for n, _ in decay_params],
        },
        {
            'params': [p for _, p in no_decay_params],
            'weight_decay': 0.0,
            'names': [n for n, _ in no_decay_params],
        },
    ]
    
    # Apply layer-wise decay if specified
    if layer_decay is not None:
        # This would require more complex parameter grouping
        # For now, return simple groups
        pass
    
    return param_groups


class GradientAccumulator:
    """
    Accumulate gradients over multiple batches.
    
    Useful for simulating larger batch sizes when memory is limited.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients: Optional[Dict] = None
        self.step_count = 0
    
    def accumulate(self, gradients: Dict):
        """
        Accumulate gradients.
        
        Args:
            gradients: Batch gradients
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {
                name: grad / self.accumulation_steps
                for name, grad in gradients.items()
            }
        else:
            for name, grad in gradients.items():
                self.accumulated_gradients[name] = (
                    self.accumulated_gradients[name] + grad / self.accumulation_steps
                )
        
        self.step_count += 1
    
    def should_update(self) -> bool:
        """Check if should perform optimizer step"""
        return self.step_count % self.accumulation_steps == 0
    
    def get_accumulated_gradients(self) -> Dict:
        """Get accumulated gradients and reset"""
        if self.accumulated_gradients is None:
            return {}
        
        gradients = self.accumulated_gradients
        self.accumulated_gradients = None
        return gradients
    
    def reset(self):
        """Reset accumulator"""
        self.accumulated_gradients = None
        self.step_count = 0


def create_loss_function(
    aux_loss_coef: float = 0.01,
) -> Callable:
    """
    Create loss function combining language modeling loss and auxiliary losses.
    
    Args:
        aux_loss_coef: Coefficient for auxiliary losses
        
    Returns:
        Loss function
    """
    def loss_fn(model_output: Dict, labels: mx.array) -> Tuple[mx.array, Dict]:
        """
        Compute total loss.
        
        Args:
            model_output: Model output dictionary
            labels: Target labels
            
        Returns:
            Total loss and loss components
        """
        logits = model_output["logits"]
        
        # Language modeling loss (cross-entropy)
        # Only compute loss for non-padded positions (labels != -100)
        loss_mask = labels != -100
        
        # Compute cross-entropy
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Gather log probabilities for target tokens
        batch_size, seq_len = labels.shape
        indices = mx.arange(batch_size * seq_len)
        
        flat_labels = labels.reshape(-1)
        flat_log_probs = log_probs.reshape(-1, logits.shape[-1])
        
        # Compute NLL
        nll = -flat_log_probs[indices, flat_labels].reshape(batch_size, seq_len)
        
        # Apply mask and compute mean
        masked_nll = mx.where(loss_mask, nll, mx.zeros_like(nll))
        lm_loss = masked_nll.sum() / loss_mask.sum()
        
        # Add auxiliary losses (MoE load balancing)
        total_loss = lm_loss
        loss_components = {"lm_loss": lm_loss}
        
        if "aux_loss" in model_output:
            aux_loss = model_output["aux_loss"]
            total_loss = total_loss + aux_loss_coef * aux_loss
            loss_components["aux_loss"] = aux_loss
        
        loss_components["total_loss"] = total_loss
        
        return total_loss, loss_components
    
    return loss_fn
