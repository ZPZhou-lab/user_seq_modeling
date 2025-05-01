# arguments.py

from dataclasses import dataclass
from enum import Enum


class ModelPath(Enum):
    """
    Model paths for different pretrained models.
    """
    Qwen3_1B: str = '../../models/Qwen3-0.6B-Base/'
    Qwen3_2B: str = '../../models/Qwen3-1.7B-Base/'
    TinyLlama_1B: str = '../../models/TinyLlama-1.1B-3T/'

@dataclass
class TrainingConfig:
    data_dir: str
    model_path: ModelPath
    batch_size: int = 16
    max_seq_len: int = 64
    max_text_len: int = 32
    num_negatives: int = 256
    EVENT_TOEKN: str = '[EVENT]'