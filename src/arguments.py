# arguments.py

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_dir: str
    model_path: str
    batch_size: int = 16
    max_seq_len: int = 64
    max_text_len: int = 32
    num_negatives: int = 256
    EVENT_TOEKN: str = '[EVENT]'