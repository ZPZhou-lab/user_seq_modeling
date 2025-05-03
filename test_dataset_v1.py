import os
import torch
import torch.distributed as dist
from src.arguments import TrainingConfig, ModelPath
from src.user_encoder import UserEncoder
from src.v1 import TextEventSequencePairDataLoader
from src.v1 import EventEncoder
from src.hierarchical import HierarchicalModel
import time

config = TrainingConfig(
    data_dir='./data',
    model_path=ModelPath.Qwen3_1B,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64
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
    train_loader = TextEventSequencePairDataLoader(config, rank=0)

    # build event encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len,
        use_flat_flash_attention=True
    )
    # build user encoder
    user_encoder = UserEncoder(model_path=config.model_path)
    model = HierarchicalModel(
        event_encoder=event_encoder,
        user_encoder=user_encoder,
        num_classes=1
    )
    model.to(device)
    optimizer = model.build_optimizer()


    s = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in range(5):
            batch = train_loader.next_batch()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # call user encoder
            outputs = model(**batch)
            # print(user_outputs.hidden_states[0][-1])
            print(outputs.nce_loss, outputs.ce_loss)
    
    e = time.time()
    print(f"Time taken for 10 iterations: {e - s:.2f} seconds")

    # destroy process group
    dist.destroy_process_group() if ddp else None