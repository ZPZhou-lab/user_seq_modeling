import torch
import torch.distributed as dist
from src.arguments import TrainingConfig, TimeEmbeddingConfig, ModelPath
from src.encoder_user import UserEncoder
from src.encoder_event import EventEncoder
from src.hierarchical import HierarchicalModel, HierarchicalModelOutput
from src.dataset import (
    TextEventSequencePairDataset, 
    sequential_event_collate_fn,
    build_dataloader
)
from src.train import (
    TensorboardLogger,
    LearningRateScheduler,
    eval_performance
)
import deepspeed
from deepspeed import DeepSpeedEngine
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
    shard_size=100,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64,
    # training args
    name='test',
    log_dir='./logs',
    save_dir='./ckpt',
    learning_rate=1e-5,
    top_warmup_steps=20,
    warmup_steps=100,
    grad_accum_steps=4,
    max_steps=120,
    log_freq=20,
    eval_steps=60,
    max_evel_iter=100,
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
# def train_step_ds(config: TrainingConfig, model, batch):
#     outputs: HierarchicalModelOutput = model(**batch)
#     loss = config.nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
    
#     return loss, outputs

def train_step_ds(
    config: TrainingConfig,
    model: DeepSpeedEngine,
    train_loader,
    lr_scheduler: LearningRateScheduler,
    step: int,
    ddp: bool=False,
    device: str='cuda',
    master_process: bool=False,
    tb_logger: TensorboardLogger = None,
):
    # training loop
    model.train()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    model.zero_grad()

    for micro_step in range(config.grad_accum_steps):
        batch = next(train_loader)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # call model
        outputs: HierarchicalModelOutput = model(**batch)
        # calculate loss
        loss = config.nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
        loss = loss / config.grad_accum_steps
        loss_tracker.update(
            nce_loss=outputs.nce_loss, 
            ce_loss=outputs.ce_loss, 
            loss=loss
        )
        model.backward(loss)

    # sync loss_tracker
    loss_tracker.synchronize(ddp)
    loss_dict = loss_tracker.values
    # set learning rate
    lr_scheduler.step()
    model.step()
    e_time = time.time()
    time_used = (e_time - s_time) / config.grad_accum_steps

    if master_process and tb_logger is not None:
        # accumulate the metrics
        tb_logger.accum(
            nce_loss=loss_dict['nce_loss'],
            ce_loss=loss_dict['ce_loss'],
            loss=loss_dict['loss'],
            time_per_iter=time_used
        )
        # Export metrics
        if tb_logger.trigger_logger(step):
            metrics = tb_logger.values
            metrics['learning_rate'] = lr_scheduler.get_lr()
            tb_logger.log(metrics, step, prefix="train")


# validation logic for DeepSpeed
def valid_context_ds(
    config: TrainingConfig,
    model,
    valid_loader,
    eval_step,
    device='cuda',
    master_process=False,
    tb_logger: TensorboardLogger = None
):
    model.eval()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    eval_tracker = TensorAccumulator()

    with torch.no_grad():
        for step in range(min(len(valid_loader), config.max_evel_iter)):
            # get batch data
            batch = next(valid_loader)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # call model
            outputs = model(**batch)
            # calculate loss
            loss = config.nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
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
        model.save_checkpoint(config.get_save_dir(), f"ckpt-{eval_step:06d}")


if __name__ == '__main__':
    args = parse_args()
    local_rank, device, master_process = worker_setup(args.local_rank)

    # build data loader
    train_set = TextEventSequencePairDataset(config, ts_config, split='train', rank=local_rank)
    valid_set = TextEventSequencePairDataset(config, ts_config, split='valid', rank=local_rank)
    train_loader = build_dataloader(train_set, config, collate_fn=sequential_event_collate_fn, rank=local_rank, num_workers=2)
    valid_loader = build_dataloader(valid_set, config, collate_fn=sequential_event_collate_fn, rank=local_rank, num_workers=2)

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

    # load DeepSpeed configuration
    ds_config = json.load(open('ds_config.json'))
    ds_config['train_micro_batch_size_per_gpu'] = config.batch_size
    # wrap model with DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.build_param_groups(weight_decay=config.weight_decay),
        config=ds_config
    )
    lr_scheduler = LearningRateScheduler(optimizer=optimizer, config=config, lower_pct=0.1, use_deepspeed=False)
    lr_scheduler.init()

    # create tensorboard logger
    if master_process:
        print("="*30 + " BEGIN TRAINING " + "="*30)
        tb_logger = TensorboardLogger(log_dir=config.get_log_dir(), log_freq=config.log_freq)
    else:
        tb_logger = None

    # begin training
    for step in range(config.max_steps):
        # train step
        train_step_ds(
            config, model_engine, train_loader, lr_scheduler, step + 1,
            ddp=True,
            device=device,
            master_process=master_process,
            tb_logger=tb_logger
        )
        # validation step
        if (step + 1) % config.eval_steps == 0:
            valid_context_ds(
                config, model_engine, valid_loader, int(step + 1),
                device=device,
                master_process=master_process,
                tb_logger=tb_logger
            )
            if master_process:
                tb_logger.flush()

    # # begin training
    # for micro_step in range(config.max_steps * config.grad_accum_steps):
    #     step = (micro_step + 1) / config.grad_accum_steps
    #     s_time = time.time()
    #     model_engine.train()
    #     # get batch data
    #     batch = next(train_loader)
    #     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
    #     # call model
    #     loss, outputs = train_step_ds(config, model_engine, batch)
    #     # backward
    #     lr_scheduler.step()
    #     model_engine.backward(loss)
    #     model_engine.step()
    #     e_time = time.time()
    #     time_used = (e_time - s_time)

    #     # log training metrics
    #     if master_process and tb_logger is not None:
    #         # accumulate loss
    #         tb_logger.accum(
    #             nce_loss=outputs.nce_loss.item(),
    #             ce_loss=outputs.ce_loss.item(),
    #             loss=loss.item(),
    #             time_per_iter=time_used
    #         )
    #         if tb_logger.trigger_logger(step):
    #             metrics = tb_logger.values
    #             metrics["learning_rate"] = lr_scheduler.get_lr()
    #             tb_logger.log(metrics, int(step), prefix="train")
        
    #     # validation step
    #     if step % config.eval_steps == 0:
    #         valid_context_ds(
    #             config, model_engine, valid_loader, int(step),
    #             device=device,
    #             master_process=master_process,
    #             tb_logger=tb_logger
    #         )
    #         if master_process:
    #             tb_logger.flush()

    # cleanup
    if master_process and tb_logger is not None:
        tb_logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()