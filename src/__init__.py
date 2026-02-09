"""
Vantage: T5-Small Fine-tuning for Text-to-SQL with MLX

A practical, laptop-friendly approach to training text-to-SQL models.
"""

__version__ = "0.1.0"

from src.model_loader import load_t5_small, T5Config
from src.synsql_loader import SynSQLStreamer, Text2SQLExample
from src.finetune_t5 import T5Finetuner, FinetuneConfig
from src.inference import Text2SQLInference

__all__ = [
    "load_t5_small",
    "T5Config",
    "SynSQLStreamer",
    "Text2SQLExample",
    "T5Finetuner",
    "FinetuneConfig",
    "Text2SQLInference"
]
