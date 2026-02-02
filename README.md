# Vantage: Text-to-SQL with Mixture of Experts

**Production-ready text-to-SQL generation using Mixture of Experts architecture, optimized for Apple Silicon with MLX**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Vantage is a state-of-the-art text-to-SQL model that uses a **Mixture of Experts (MoE)** architecture for efficient and accurate SQL query generation from natural language. Built with MLX for optimal performance on Apple Silicon (M1/M2/M3), Vantage offers three model variants to balance between speed and accuracy.

### Key Features

- **Sparse Mixture of Experts**: Only ~12.5% of parameters active per forward pass
- **Three Model Sizes**: Small (2B), Medium (8B), Large (24B) parameters
- **Optimized for Apple Silicon**: Native MLX implementation for M-series chips
- **Production Ready**: Comprehensive evaluation, benchmarks, and deployment tools
- **Multi-Dataset Training**: Spider, BIRD-SQL, WikiSQL, and synthetic data
- **Schema-Aware**: Cross-attention mechanism for understanding database structures

## Architecture

Vantage uses a decoder-only transformer with Mixture of Experts layers. Each MoE layer contains multiple expert networks, with a learned router selecting the top-K experts for each token. This sparse activation pattern significantly reduces computation while maintaining model capacity.

```
Text Query + Schema → Tokenizer → MoE Transformer Layers → SQL Generation
                                          ↓
                                    Sparse Router
                                    (Top-2 Selection)
                                          ↓
                              [Expert 1] [Expert 2] ... [Expert N]
```

**Key Components:**
- Sparse router with top-K gating (K=2)
- Expert networks with SwiGLU activation
- Load balancing auxiliary loss
- Schema cross-attention every 4 layers
- Rotary position embeddings (RoPE)

## Model Variants

| Model | Parameters | Active Params | Memory | Spider EX | Speed |
|-------|-----------|---------------|---------|-----------|-------|
| Small | 2B | 250M | ~50GB | 70%+ | Fast |
| Medium | 8B | 1B | ~110GB | 78%+ | Balanced |
| Large | 24B | 3B | ~180GB | 82%+ | Maximum Accuracy |

## Training

```bash
# Download datasets
python scripts/download_datasets.py

# Preprocess data
python scripts/preprocess_data.py --config configs/medium_model.yaml

# Train model
python scripts/train.py \
    --config configs/medium_model.yaml \
    --output_dir ./checkpoints/medium \
    --wandb_project vantage-text2sql
```

## Evaluation

```bash
# Evaluate on Spider benchmark
python scripts/evaluate.py \
    --model_path ./checkpoints/medium/best_model \
    --benchmark spider \
    --output_dir ./results
```

**Benchmark Results:**
- Spider: State-of-the-art accuracy on complex multi-table queries
- BIRD-SQL: Strong cross-domain generalization
- WikiSQL: Near-perfect accuracy on simple queries

## Project Structure

```
vantage/
├── src/
│   ├── models/          # MoE architecture implementation
│   ├── data/            # Dataset loaders and preprocessing
│   ├── training/        # Training loop and optimization
│   ├── evaluation/      # Metrics and benchmarking
│   └── inference/       # Generation and API
├── configs/             # Model configurations (small/medium/large)
├── scripts/             # Training, evaluation, and export scripts
├── notebooks/           # Jupyter notebooks for experimentation and demos
└── tests/               # Unit tests
```

## Citation

If you use Vantage in your research, please cite:

```bibtex
@software{vantage2024,
  title={Vantage: Text-to-SQL with Mixture of Experts},
  author={DHIWAHAR-K},
  year={2024},
  url={https://github.com/DHIWAHAR-K/Vantage}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- Trained on Spider, BIRD-SQL, WikiSQL, and Gretel datasets
- Inspired by Mixtral and Switch Transformer architectures

## Contact

- GitHub: [@DHIWAHAR-K](https://github.com/DHIWAHAR-K)
- Email: adhithyak99@gmail.com

---

**Note**: This is a research project. Always validate generated SQL queries before execution in production environments.
