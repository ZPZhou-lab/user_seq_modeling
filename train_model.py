import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.arguments import TrainingConfig, TimeEmbeddingConfig, ModelPath
from src.encoder_user import UserEncoder
from src.encoder_event import EventEncoder
from src.hierarchical import HierarchicalModel
from src.dataset import (
    TextEventSequencePairDataset, 
    build_dataloader
)
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
    shard_size=1000,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64,
    padding=False,
    add_user_token=True,
    # training args
    name='test',
    log_dir='./logs',
    save_dir='./ckpt',
    learning_rate=1e-5,
    top_warmup_steps=100,
    warmup_steps=100,
    grad_accum_steps=5,
    max_steps=200,
    log_freq=50,
    eval_steps=200,
    max_evel_iter=200,
    temperature=0.1,
    nce_threshold=0.99,
    nce_loss_lambda=0.05
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
    # setup worker
    ddp = int(os.environ.get('RANK', -1)) != -1
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = worker_setup(ddp, 42)

    # build data loader
    train_set = TextEventSequencePairDataset(config, ts_config, split='train', rank=ddp_rank)
    valid_set = TextEventSequencePairDataset(config, ts_config, split='valid', rank=ddp_rank)
    train_loader = build_dataloader(train_set, config, rank=ddp_rank, num_workers=4)
    valid_loader = build_dataloader(valid_set, config, rank=ddp_rank, num_workers=4)

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
        config=config,
        event_encoder=event_encoder,
        user_encoder=user_encoder,
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
    lr_scheduler = LearningRateScheduler(config=config, optimizer=optimizer, lower_pct=0.1)
    lr_scheduler.init()
    
    if master_process:
        print("="*30 + " BEGIN TRAININ " + "="*30)
        # create tensorboard logger
        tb_logger = TensorboardLogger(log_dir=config.get_log_dir(), log_freq=config.log_freq)
    else:
        tb_logger = None

    # begin training
    for step in range(config.max_steps):
        # train step
        train_step(config, model, train_loader, optimizer, lr_scheduler, step + 1, ddp, 
                   device=device, 
                   master_process=master_process, 
                   tb_logger=tb_logger)
        # valid step
        if (step + 1) % config.eval_steps == 0:
            valid_context(config, model, valid_loader, optimizer, step + 1, ddp,
                          device=device, 
                          master_process=master_process, 
                          tb_logger=tb_logger)
            if master_process:
                tb_logger.flush()

    # destroy process group
    dist.destroy_process_group() if ddp else None