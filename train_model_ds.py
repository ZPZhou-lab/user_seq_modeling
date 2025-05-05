import torch
import torch.distributed as dist
from src.arguments import TrainingConfig, ModelPath
from src.encoder_user import UserEncoder
from src.encoder_event import EventEncoder
from src.hierarchical import HierarchicalModel, HierarchicalModelOutput
from src.dataset import TextEventSequencePairDataLoaderV2 as TextEventSequencePairDataLoader
from src.train import (
    TensorboardLogger,
    eval_performance
)
import deepspeed
import json
import argparse
import time
from src.common import ScalerAccumulator, TensorAccumulator

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='User Sequence Modeling with DeepSpeed')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    
    # DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

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
    warm_up_steps=100,
    max_steps=10000,
    log_freq=1,
    eval_steps=10,
    temprature=0.05,
    nce_threshold=0.99,
    nce_loss_lambda=0.5
)

def worker_setup(local_rank, seed=42):
    # setup distributed training
    device = f'cuda:{local_rank}'
    master_process = local_rank == 0
    
    torch.cuda.set_device(device)
    # fix random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    return local_rank, device, master_process

# training logic for DeepSpeed
def train_step_ds(model, batch, nce_loss_lambda=0.5):
    outputs: HierarchicalModelOutput = model(**batch)
    loss = nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
    
    return loss, outputs

# validation logic for DeepSpeed
def valid_context_ds(
    model,
    valid_loader,
    eval_step,
    save_dir='./ckpt',
    nce_loss_lambda=0.5,
    device='cuda',
    master_process=False,
    tb_logger=None
):
    model.eval()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    eval_tracker = TensorAccumulator()

    valid_loader.reset()
    with torch.no_grad():
        for step in range(valid_loader.total_steps):
            # get batch data
            batch = valid_loader.next_batch()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # call model
            outputs = model(**batch)
            # calculate loss
            loss = nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
            loss_tracker.update(
                nce_loss=outputs.nce_loss, 
                ce_loss=outputs.ce_loss, 
                loss=loss
            )
            # fetch for evaluation
            eval_tracker.update(
                probs=torch.sigmoid(outputs.logits).view(-1),
                labels=batch['labels'].view(-1)
            )
    
    # evaluate performance
    loss_dict = loss_tracker.values
    eval_dict = eval_performance(eval_tracker.values)
    # time cost
    e_time = time.time()
    time_used = (e_time - s_time)

    if master_process:
        metrics = {
            **loss_dict,
            **eval_dict,
            "eval_time_cost": time_used,
        }
        if tb_logger is not None:
            tb_logger.log(metrics, eval_step, prefix="valid")
        # save model checkpoint
        model.save_checkpoint(save_dir, f"ckpt-{eval_step:06d}")


if __name__ == '__main__':
    args = parse_args()
    local_rank, device, master_process = worker_setup(args.local_rank)

    # create data loader
    train_loader = TextEventSequencePairDataLoader(config, rank=local_rank)
    valid_loader = TextEventSequencePairDataLoader(config, rank=local_rank, split='valid')

    # build event encoder and user encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len,
        use_flat_flash_attention=True
    )
    user_encoder = UserEncoder(model_path=config.model_path)
    
    # build model
    model = HierarchicalModel(
        event_encoder=event_encoder,
        user_encoder=user_encoder,
        temperature=config.temprature,
        nce_threshold=config.nce_threshold,
        num_classes=1
    )
    # load DeepSpeed configuration
    ds_config = json.load(open('ds_config.json'))
    # wrap model with DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # create tensorboard logger
    if master_process:
        print("="*30 + " BEGIN TRAINING " + "="*30)
        tb_logger = TensorboardLogger(log_dir=config.get_log_dir(), log_freq=config.log_freq)
    else:
        tb_logger = None

    # begin training
    max_steps = 20
    for step in range(max_steps):
        model_engine.train()
        # get batch data
        batch = train_loader.next_batch()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # call model
        loss, outputs = train_step_ds(model_engine, batch, config.nce_loss_lambda)
        # backward
        model_engine.backward(loss)
        model_engine.step()
        
        # log training metrics
        if master_process and tb_logger.trigger_logger(step + 1):
            metrics = {
                "nce_loss": outputs.nce_loss.item(),
                "ce_loss": outputs.ce_loss.item(),
                "loss": loss.item(),
                "learning_rate": model_engine.get_lr()[0]
            }
            tb_logger.log(metrics, step + 1, prefix="train")
        
        # validation step
        if (step + 1) % config.eval_steps == 0:
            valid_context_ds(
                model_engine, valid_loader, step + 1,
                save_dir=config.get_save_dir(),
                nce_loss_lambda=config.nce_loss_lambda,
                device=device,
                master_process=master_process,
                tb_logger=tb_logger
            )
            if master_process:
                tb_logger.flush()

    # cleanup
    if master_process and tb_logger is not None:
        tb_logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()