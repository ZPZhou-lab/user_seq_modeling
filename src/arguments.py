# arguments.py
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ModelPath(Enum):
    """
    Model paths for different pretrained models.
    """
    Qwen3_1B: str = '../../models/Qwen3-0.6B-Base/'
    Qwen3_2B: str = '../../models/Qwen3-1.7B-Base/'
    TinyLlama_1B: str = '../../models/TinyLlama-1.1B-3T/'


@dataclass
class TimeEmbeddingConfig:
    use_time_embedding: bool = True
    mode: str = 'absolute'
    time_hiddens: int = 256
    max_diff_day: int = 720
    max_year_ago: int = 10
    mixup_activation: str = 'silu'


@dataclass
class TrainingConfig:
    train_data_dir: str
    valid_data_dir: str
    model_path: ModelPath
    shard_size: int     = 10000
    batch_size: int     = 16
    max_seq_len: int    = 64
    max_text_len: int   = 32
    num_negatives: int  = 256
    EVENT_TOEKN: str = '[EVENT]'
    # training args
    name: str = 'experiment'
    log_dir: str = './logs'
    save_dir: str = './checkpoints'
    learning_rate: float    = 1e-5
    weight_decay: float     = 0.01
    warmup_steps: int       = 100
    top_warmup_steps: int   = -1
    grad_accum_steps: int   = 1
    max_steps: int          = 10000
    log_freq: int           = 100
    eval_steps: int         = 100
    max_evel_iter: int      = 1000
    temprature: float       = 0.05
    nce_threshold: float    = 0.99
    nce_loss_lambda: float  = 0.5

    def get_log_dir(self):
        # generate time-suffix
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{self.log_dir}/{self.name}_{time_suffix}"

    def get_save_dir(self):
        return f"{self.save_dir}/{self.name}"