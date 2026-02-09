# T5-Small Fine-tuning for Text-to-SQL with MLX

A practical, laptop-friendly approach to training text-to-SQL models.

## Overview

Fine-tune T5-small (60M parameters) on SynSQL's massive dataset using MLX optimization for Apple Silicon. This approach is **actually trainable** on a MacBook M3 Max.

## Key Features

- Pre-trained T5-small (60M params) as base
- Fine-tuned on 22.9M SynSQL examples
- MLX optimized for M3 Max (fast + cool)
- Thermal management built-in
- 4-8 hour training time (realistic!)
- Competitive benchmark accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
```bash
python scripts/prepare_data.py --subset-size 5000000
```

### 2. Convert T5 to MLX Format
```bash
python scripts/convert_t5_to_mlx.py
```

### 3. Fine-tune on SynSQL
```bash
python scripts/finetune.py --config configs/t5_finetune.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model
```

### 5. Try the Demo
```bash
streamlit run scripts/demo.py
```

## Training Details

- **Base Model**: T5-small (60M params)
- **Dataset**: SynSQL (5M subset of 22.9M examples)
- **Training Time**: 4-8 hours on M3 Max
- **Batch Size**: 32 (thermal-friendly)
- **Effective Batch**: 256 (with gradient accumulation)
- **Curriculum**: Simple → Medium → Complex SQL

## Expected Performance

| Metric | Target |
|--------|--------|
| Spider Exact Match | 55-65% |
| Spider Execution | 60-70% |
| WikiSQL Execution | 80-85% |
| Inference Speed | <50ms |
| Training Time | 4-8 hours |

## Why T5-Small?

- Pre-trained on massive corpus (understands language)
- Encoder-decoder perfect for text-to-SQL
- 60M params = fast inference
- Proven architecture
- Fine-tuning 10-20x faster than training from scratch

## License

MIT License - See LICENSE file
