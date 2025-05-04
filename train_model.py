import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.arguments import TrainingConfig, TimeEmbeddingConfig, ModelPath
from src.user_encoder import UserEncoder
from src.v1 import TextEventSequencePairDataLoader
from src.v1 import EventEncoder
from src.hierarchical import HierarchicalModel
from src.train import (
    LearningRateScheduler,
    TensorboardLogger,
    train_step, valid_context
)

ts_config = TimeEmbeddingConfig(
    use_time_embedding=True,
    mode='absolute',
    time_hiddens=256,
    max_diff_day=720,
    max_year_ago=10,
    mixup_activation='silu'
)

config = TrainingConfig(
    # dataset args
    train_data_dir='./data',
    valid_data_dir='./data',
    model_path=ModelPath.Qwen3_1B,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64,
    # training args
    name='test',
    log_dir='./logs',
    save_dir='./ckpt',
    learning_rate=1e-5,
    top_warmup_steps=-1,
    warmup_steps=100,
    max_steps=10000,
    log_freq=1,
    eval_steps=10,
    temprature=0.05,
    nce_threshold=0.99,
    nce_loss_lambda=0.5
)

def worker_setup(ddp: int, seed: int=42):
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank, ddp_local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        master_process = ddp_rank == 0 # master do logging
    else:
        # use single GPU training
        ddp_rank, ddp_local_rank = 0, 0
        ddp_world_size = 1
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        master_process = True
    
    torch.cuda.set_device(device)
    # fix random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


if __name__ == '__main__':
    ddp = int(os.environ.get('RANK', -1)) != -1
    # setup worker
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = worker_setup(ddp, 42)

    # build data loader
    train_loader = TextEventSequencePairDataLoader(config, rank=0, ts_config=ts_config, split='train')
    valid_loader = TextEventSequencePairDataLoader(config, rank=0, ts_config=ts_config, split='valid')

    # build event encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len,
        use_flat_flash_attention=True
    )
    # build user encoder
    user_encoder = UserEncoder(
        model_path=config.model_path,
        ts_config=ts_config
    )
    model = HierarchicalModel(
        event_encoder=event_encoder,
        user_encoder=user_encoder,
        temperature=config.temprature,
        nce_threshold=config.nce_threshold,
        num_classes=1
    )
    model.to(device)

    # wrap model with DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    else:
        raw_model = model
    
    optimizer = raw_model.build_optimizer(learning_rate=config.learning_rate)
    lr_scheduler = LearningRateScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        top_warmup_steps=config.top_warmup_steps,
        learning_rate=config.learning_rate, 
        lower_pct=0.1
    )
    

    if master_process:
        print("="*30 + " BEGIN TRAININ " + "="*30)
        # create tensorboard logger
        tb_logger = TensorboardLogger(log_dir=config.get_log_dir(), log_freq=config.log_freq)
    else:
        tb_logger = None

    # begin training
    max_steps = 10
    for step in range(max_steps):
        # train step
        train_step(model, train_loader, optimizer, lr_scheduler, step + 1, ddp, 
                   nce_loss_lambda=config.nce_loss_lambda, 
                   grad_accum_steps=1, 
                   device=device, 
                   master_process=master_process, 
                   tb_logger=tb_logger)
        # valid step
        if (step + 1) % config.eval_steps == 0:
            valid_context(model, valid_loader, step + 1, ddp, config.get_save_dir(),
                          nce_loss_lambda=config.nce_loss_lambda,
                          device=device, 
                          master_process=master_process, 
                          tb_logger=tb_logger)
            if master_process:
                tb_logger.flush()

    # destroy process group
    dist.destroy_process_group() if ddp else None