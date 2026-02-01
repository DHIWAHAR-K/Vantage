# Vantage Training Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hardware Requirements](#hardware-requirements)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Running Training](#running-training)
6. [Monitoring](#monitoring)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py --output_dir ./data

# 3. Preprocess data
python scripts/preprocess_data.py \
    --config configs/medium_model.yaml \
    --data_dir ./data \
    --output_dir ./processed_data

# 4. Start training
python scripts/train.py \
    --config configs/medium_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./checkpoints/medium \
    --wandb_project vantage-text2sql
```

## Hardware Requirements

### Minimum Requirements

**For Small Model (2B)**:
- Apple Silicon: M1 Max or later
- RAM: 32GB+ unified memory
- Storage: 100GB+ SSD
- Training time: ~3-4 days

**For Medium Model (8B)**:
- Apple Silicon: M2 Ultra or M3 Max
- RAM: 64GB+ unified memory
- Storage: 200GB+ SSD
- Training time: ~5-7 days

**For Large Model (24B)**:
- Apple Silicon: M3 Max/Ultra
- RAM: 128GB+ unified memory
- Storage: 500GB+ SSD
- Training time: ~10-14 days

### Recommended: M3 Max Configuration

- **CPU/GPU**: M3 Max (16-core CPU, 40-core GPU)
- **Memory**: 128GB unified memory
- **Storage**: 2TB+ SSD (for datasets, checkpoints, logs)
- **Cooling**: Good ventilation for sustained training

### Memory Usage Estimates

| Model | Training (batch=16) | Peak Memory | Gradient Checkpointing |
|-------|-------------------|-------------|----------------------|
| Small | ~45GB | ~50GB | Not needed |
| Medium | ~95GB | ~110GB | Recommended |
| Large | ~165GB | ~180GB | Required |

**Tips for Memory Optimization**:
```yaml
# configs/large_model.yaml
training:
  batch_size: 8  # Reduce if OOM
  gradient_accumulation_steps: 4  # Maintain effective batch=32
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

## Dataset Preparation

### Supported Datasets

1. **Spider**: 10,181 examples, complex multi-table queries
2. **BIRD-SQL**: 12,751 examples, cross-domain reasoning
3. **WikiSQL**: 80,654 examples, simple single-table queries
4. **Gretel Synthetic**: 100,000+ examples, augmentation

### Download Datasets

```bash
python scripts/download_datasets.py \
    --datasets spider bird wikisql gretel \
    --output_dir ./data \
    --cache_dir ./cache
```

### Dataset Structure

```
data/
├── spider/
│   ├── train.json
│   ├── dev.json
│   └── database/
├── bird/
│   ├── train.json
│   └── database/
├── wikisql/
│   └── ... (auto-downloaded via HuggingFace)
└── gretel/
    └── ... (streamed from HuggingFace)
```

### Preprocessing

```bash
python scripts/preprocess_data.py \
    --config configs/medium_model.yaml \
    --data_dir ./data \
    --output_dir ./processed_data \
    --num_workers 8
```

**What preprocessing does**:
1. SQL normalization (keyword case, whitespace)
2. Schema extraction and serialization
3. Tokenization with special tokens
4. Train/validation split
5. Caching for faster loading

### Data Augmentation

Enable augmentation in config:

```yaml
data:
  augmentation_prob: 0.4
  schema_perturbation_prob: 0.15
```

**Augmentation strategies**:
- Question paraphrasing (template-based)
- Schema renaming (table/column variations)
- SQL transformations (equivalent queries)

## Training Configuration

### Model Sizes

**Small (2B)**:
```yaml
# configs/small_model.yaml
model:
  hidden_size: 2048
  num_layers: 24
  num_experts: 16
  num_experts_per_token: 2

training:
  batch_size: 32
  learning_rate: 3e-4
  num_train_steps: 100000
```

**Medium (8B)**:
```yaml
# configs/medium_model.yaml
model:
  hidden_size: 4096
  num_layers: 32
  num_experts: 32
  num_experts_per_token: 2

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 2e-4
  num_train_steps: 200000
  gradient_checkpointing: true
```

**Large (24B)**:
```yaml
# configs/large_model.yaml
model:
  hidden_size: 6144
  num_layers: 40
  num_experts: 64
  num_experts_per_token: 2

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  num_train_steps: 300000
  gradient_checkpointing: true
  activation_checkpointing: true
```

### Dataset Mixing

Control dataset proportions:

```yaml
data:
  spider_weight: 0.4      # 40% Spider
  bird_sql_weight: 0.3    # 30% BIRD-SQL
  wiki_sql_weight: 0.2    # 20% WikiSQL
  gretel_synthetic_weight: 0.1  # 10% Synthetic
```

**Rationale**:
- Spider: Complex queries, high quality
- BIRD-SQL: Cross-domain generalization
- WikiSQL: Coverage, simple queries
- Gretel: Augmentation, diversity

### Optimization Settings

```yaml
training:
  # Optimizer
  optimizer: adamw
  learning_rate: 2e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
  
  # Schedule
  lr_scheduler_type: cosine
  warmup_steps: 2000
  min_lr_ratio: 0.1
  
  # Precision
  mixed_precision: bf16
```

### MoE-Specific Settings

```yaml
model:
  # Expert configuration
  num_experts: 32
  num_experts_per_token: 2
  expert_capacity: 1.25
  
  # Load balancing
  router_aux_loss_coef: 0.01
  router_z_loss_coef: 0.001
```

**Load balancing coefficients**:
- Too high (>0.05): Hurts main task performance
- Too low (<0.001): Experts collapse
- Sweet spot: 0.01

## Running Training

### Basic Training

```bash
python scripts/train.py \
    --config configs/medium_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./checkpoints/medium
```

### Training with W&B Logging

```bash
python scripts/train.py \
    --config configs/medium_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./checkpoints/medium \
    --wandb_project vantage-text2sql \
    --wandb_run_name medium-run-1
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    --config configs/medium_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./checkpoints/medium \
    --resume_from_checkpoint ./checkpoints/medium/checkpoint-50000
```

### Training Script Options

```bash
python scripts/train.py --help

Options:
  --config PATH              Path to config file (required)
  --data_dir PATH           Path to preprocessed data (required)
  --output_dir PATH         Output directory for checkpoints
  --resume_from_checkpoint  Resume from checkpoint
  --wandb_project NAME      W&B project name
  --wandb_run_name NAME     W&B run name
  --seed INT                Random seed (default: 42)
  --debug                   Debug mode (smaller dataset)
```

## Monitoring

### Metrics to Track

**Training Metrics**:
- `train/loss`: Language modeling loss
- `train/aux_loss`: MoE load balancing loss
- `train/learning_rate`: Current LR
- `train/grad_norm`: Gradient norm (monitor for instability)
- `train/throughput`: Steps per second

**Validation Metrics**:
- `eval/loss`: Validation loss
- `eval/perplexity`: Exp(loss)
- `eval/exact_match`: Exact SQL match accuracy
- `eval/valid_sql`: % syntactically valid SQL

**Expert Utilization**:
- `expert/utilization`: % experts used
- `expert/load_balance`: Variance in token distribution
- `expert/tokens_per_expert`: Token distribution

### Weights & Biases Dashboard

Key plots to monitor:

1. **Loss Curves**: Smooth decrease without spikes
2. **Expert Utilization**: Should be 80-100%
3. **Learning Rate**: Warmup then decay
4. **Gradient Norm**: Stable, not exploding

### TensorBoard (Alternative)

```bash
# Log with TensorBoard
tensorboard --logdir ./checkpoints/medium/logs --port 6006
```

### Checkpoint Management

Checkpoints saved every `save_steps` (default: 5000):

```
checkpoints/medium/
├── checkpoint-5000/
├── checkpoint-10000/
├── checkpoint-15000/
├── checkpoint-best/      # Best validation loss
└── checkpoint-latest/    # Most recent
```

Best checkpoint tracked by validation exact match.

## Hyperparameter Tuning

### Learning Rate

**Finding optimal LR**:

```bash
# LR finder (sweep)
for lr in 1e-5 1e-4 2e-4 3e-4 5e-4; do
    python scripts/train.py \
        --config configs/medium_model.yaml \
        --learning_rate $lr \
        --output_dir ./lr_sweep/lr_${lr}
done
```

**Guidelines**:
- Small model: 3e-4 to 5e-4
- Medium model: 1e-4 to 3e-4
- Large model: 5e-5 to 1e-4

Rule of thumb: Larger models need smaller LR.

### Batch Size

Effective batch size = batch_size × gradient_accumulation_steps

**Recommended effective batch sizes**:
- Small: 32-64
- Medium: 32-64
- Large: 32-128

Larger batches → more stable but slower training.

### Warmup Steps

Typical: 1-2% of total training steps

```yaml
training:
  num_train_steps: 200000
  warmup_steps: 2000  # 1% warmup
```

### Weight Decay

Prevent overfitting on training data:

```yaml
training:
  weight_decay: 0.1  # Standard for LLMs
```

Higher weight decay (0.2) if overfitting.

### Load Balancing Coefficients

**If experts collapse** (few experts used):
```yaml
model:
  router_aux_loss_coef: 0.02  # Increase (was 0.01)
```

**If validation loss high but training low**:
```yaml
model:
  router_aux_loss_coef: 0.005  # Decrease
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. Reduce batch size:
   ```yaml
   training:
     batch_size: 8  # Was 16
     gradient_accumulation_steps: 4  # Was 2
   ```

2. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. Use mixed precision:
   ```yaml
   training:
     mixed_precision: bf16
   ```

4. Reduce sequence length:
   ```yaml
   training:
     max_seq_length: 1024  # Was 2048
   ```

### Loss Spikes / NaN Loss

**Causes**:
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions**:
1. Reduce learning rate by 2-5x
2. Lower warmup steps
3. Check gradient clipping:
   ```yaml
   training:
     max_grad_norm: 0.5  # Was 1.0
   ```

4. Add z-loss for router stability:
   ```yaml
   model:
     router_z_loss_coef: 0.001
   ```

### Expert Collapse

**Symptoms**:
- Expert utilization < 50%
- Some experts never activated

**Solutions**:
1. Increase load balancing loss:
   ```yaml
   model:
     router_aux_loss_coef: 0.02
   ```

2. Add expert dropout:
   ```yaml
   model:
     expert_dropout: 0.1
   ```

3. Adjust expert capacity:
   ```yaml
   model:
     expert_capacity: 1.5  # Allow more tokens per expert
   ```

### Slow Training

**Optimizations**:
1. Increase batch size (if memory allows)
2. Reduce preprocessing workers if disk-bound
3. Use faster data format (preprocessed)
4. Check for CPU bottlenecks (use Activity Monitor)

### Overfitting

**Symptoms**:
- Low train loss, high validation loss
- Validation metrics plateau while training improves

**Solutions**:
1. Increase dropout:
   ```yaml
   model:
     hidden_dropout: 0.15  # Was 0.1
     expert_dropout: 0.15
   ```

2. Increase weight decay:
   ```yaml
   training:
     weight_decay: 0.2  # Was 0.1
   ```

3. More data augmentation:
   ```yaml
   data:
     augmentation_prob: 0.5
   ```

4. Early stopping on validation metric

### Underfitting

**Symptoms**:
- Both train and validation loss high
- Slow improvement

**Solutions**:
1. Increase model size
2. Train longer (more steps)
3. Increase learning rate
4. Reduce regularization (dropout, weight decay)

## Training Time Estimates

### M3 Max (40-core GPU, 128GB RAM)

| Model | Steps | Batch Size | Time per Step | Total Time |
|-------|-------|-----------|---------------|-----------|
| Small | 100K | 32 | 2.5s | ~3 days |
| Medium | 200K | 16 | 4.0s | ~9 days |
| Large | 300K | 8 | 8.0s | ~28 days |

**Factors affecting speed**:
- Sequence length (longer = slower)
- Number of experts (more = slower routing)
- Gradient accumulation (more steps per update)

## Best Practices

1. **Start Small**: Train small model first to validate pipeline
2. **Monitor Early**: Check first 1000 steps for issues
3. **Checkpoint Often**: Save every 5000 steps
4. **Validate Regularly**: Evaluate every 1000 steps
5. **Track Metrics**: Use W&B or TensorBoard
6. **Version Control**: Track config changes in git
7. **Document Experiments**: Note what worked/didn't work
8. **Save Best Model**: Track best validation checkpoint

## Example Training Commands

### Quick Test Run

```bash
# Test on small subset
python scripts/train.py \
    --config configs/small_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./debug \
    --debug \
    --num_train_steps 1000
```

### Full Production Run

```bash
# Full training with all bells and whistles
python scripts/train.py \
    --config configs/medium_model.yaml \
    --data_dir ./processed_data \
    --output_dir ./checkpoints/medium-prod \
    --wandb_project vantage-production \
    --wandb_run_name medium-final-v1 \
    --seed 42
```

### Multi-Stage Training

```bash
# Stage 1: Train on large dataset mix
python scripts/train.py --config configs/stage1.yaml

# Stage 2: Fine-tune on high-quality data only
python scripts/train.py \
    --config configs/stage2.yaml \
    --resume_from_checkpoint ./checkpoints/stage1/checkpoint-best
```

## Next Steps

After training:
1. Evaluate on benchmarks (see `EVALUATION.md`)
2. Export to HuggingFace format
3. Deploy for inference
4. Collect user feedback
5. Iterate on model/data
