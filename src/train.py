import os
import torch
from torch.optim import Optimizer
import time
import math
from .common import ScalerAccumulator, TensorAccumulator
from .dataset import EventSequenceDataLoaderMeta
from .hierarchical import HierarchicalModel, HierarchicalModelOutput
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from typing import Dict

# cos-decay with warm-up learning rate scheduler
class LearningRateScheduler:
    def __init__(self, 
        optimizer: Optimizer,
        warmup_steps: int=100, 
        max_steps: int=10000,
        top_warmup_steps: int=-1,
        learning_rate: float=1e-5, 
        lower_pct: float=0.1
    ):
        self.optimizer          = optimizer
        self.warmup_steps       = warmup_steps
        self.max_steps          = max_steps
        self.top_warmup_steps   = top_warmup_steps
        self.max_lr             = learning_rate
        self.min_lr             = lower_pct * learning_rate
        self.current_step       = 0
        self._local_step        = 0
        self._top_warmup_flag   = True if top_warmup_steps > 0 else False
        
        # save the original parameter groups
        self.orig_param_groups = []
        for group in optimizer.param_groups:
            self.orig_param_groups.append({
                'params': group['params'], 
                'lr': group.get('lr', learning_rate),
                'weight_decay': group.get('weight_decay', 0.0)})
    
    def step(self):
        """update the learning rate for each step"""
        self.current_step += 1
        
        # stap 1: warmup for head classifier
        if self._top_warmup_flag:
            self._local_step += 1
            lr_scale = min(1.0, self._local_step / self.top_warmup_steps)
            current_lr = self.min_lr + (self.max_lr - self.min_lr) * lr_scale
            
            # update the optimizer parameter groups
            for param_group in self.optimizer.param_groups:
                name = param_group['name'].split('.')[0]
                if name == 'classifier':
                    param_group['lr'] = current_lr
                else:
                    param_group['lr'] = 0.0
            # check if the current step is the last step of warmup
            # if so, reset step so as to start the next stage
            if self.current_step == self.top_warmup_steps:
                self._top_warmup_flag = False
                self._local_step = 0
        else:
            self._local_step += 1
            # step 2: cos-decay with warmup for all parameters
            if self._local_step < self.warmup_steps:
                current_lr = self._local_step / self.warmup_steps * self.max_lr
            elif self._local_step > self.max_steps:
                current_lr = self.min_lr
            else:
                coeff = 1.0 + math.cos(math.pi * (self._local_step - self.warmup_steps) / (self.max_steps - self.warmup_steps))
                current_lr = self.min_lr + 0.5 * coeff * (self.max_lr - self.min_lr)
            
            # update the optimizer parameter groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
    
    def get_lr(self) -> Dict[str, float]:
        """Get the current learning rate for each parameter group"""
        group_lr = {}
        for param_group in self.optimizer.param_groups:
            name = param_group['name'].split('.')[0]
            group_lr[name] = param_group['lr']
        return group_lr


class TensorboardLogger:
    def __init__(self, log_dir, log_freq=1):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_freq = log_freq
    
    def log(self, metrics, step, prefix="train"):
        """Log metrics to tensorboard
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int): Current training step
        """
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.writer.add_scalars(f"{prefix}/{key}", value, step)
            else:
                self.writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def trigger_logger(self, step):
        """Trigger the logger to log the metrics"""
        return (step % self.log_freq) == 0


def train_step(
    model: HierarchicalModel,
    train_loader: EventSequenceDataLoaderMeta,
    optimizer,
    lr_scheduler: LearningRateScheduler,
    step: int,
    ddp: bool=False,
    nce_loss_lambda: float=0.5,
    grad_accum_steps: int=1,
    device: str='cuda',
    master_process: bool=False,
    tb_logger: TensorboardLogger = None,
):
    # training loop
    model.train()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    optimizer.zero_grad()

    for micro_step in range(grad_accum_steps):
        batch = train_loader.next_batch()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # set gradient sync
        # model.require_backward_grad_sync = micro_step == grad_accum_steps - 1 \
        #     if ddp else model.require_backward_grad_sync
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # call model
            outputs: HierarchicalModelOutput = model(**batch)
        # calculate loss
        loss = nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
        loss = loss / grad_accum_steps
        loss_tracker.update(
            nce_loss=outputs.nce_loss, 
            ce_loss=outputs.ce_loss, 
            loss=loss
        )
        loss.backward()

    # sync loss_tracker
    loss_tracker.synchronize(ddp)
    loss_dict = loss_tracker.values
    
    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set learning rate
    lr_scheduler.step()

    optimizer.step()
    torch.cuda.synchronize()
    s_end = time.time()
    time_used = (s_end - s_time)

    if master_process and tb_logger.trigger_logger(step):
        # Export metrics
        metrics = {
            **loss_dict,
            "time_per_iter": time_used,
            "grad_norm": norm.item() if isinstance(norm, torch.Tensor) else norm,
            "learning_rate": lr_scheduler.get_lr()
        }
        
        # Log to tensorboard at specified frequency
        if tb_logger is not None and step % tb_logger.log_freq == 0:
            tb_logger.log(metrics, step, prefix="train")


def valid_context(
    model: HierarchicalModel,
    valid_loader: EventSequenceDataLoaderMeta,
    eval_step: int,
    ddp: bool=False,
    save_dir: str='./ckpt',
    nce_loss_lambda: float=0.5,
    device: str='cuda',
    master_process: bool=False,
    tb_logger: TensorboardLogger = None
):
    # validation loop
    model.eval()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    eval_tracker = TensorAccumulator()

    valid_loader.reset()
    with torch.no_grad():
        for step in range(valid_loader.total_steps):
            # get the current batch
            batch = valid_loader.next_batch()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # call model
                outputs: HierarchicalModelOutput = model(**batch)
            # calculate loss
            loss = nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
            loss_tracker.update(
                nce_loss=outputs.nce_loss, 
                ce_loss=outputs.ce_loss, 
                loss=loss
            )

            # fetch probs for evaluation
            eval_tracker.update(
                probs=torch.sigmoid(outputs.logits).view(-1),
                labels=batch['labels'].view(-1)
            )
    
    # sync loss_tracker
    loss_tracker.synchronize(ddp)
    eval_tracker.synchronize(ddp)
    loss_dict = loss_tracker.values
    eval_dict = eval_performance(eval_tracker.values)
    # calculate the time used
    e_time = time.time()
    time_used = (e_time - s_time)

    if master_process:
        # Export metrics
        metrics = {
            **loss_dict,
            **eval_dict,
            "eval_time_cost": time_used,
        }
        
        # Log to tensorboard at specified frequency
        if tb_logger is not None:
            tb_logger.log(metrics, eval_step, prefix="valid")

        # save the checkpoint
        raw_model = model.module if ddp else model
        raw_model.save_pretrained(save_path=os.path.join(save_dir, f"ckpt-{eval_step:06d}"))


def eval_performance(eval_dict):
    """
    Evaluate the performance of the model using ROC AUC score
    
    eval_dict: dict
        Dictionary containing the evaluation results
        - probs: (N, ) numpy array of predicted probabilities
        - labels: (N, ) numpy array of true labels
    """
    # calculate roc auc score
    auc = roc_auc_score(eval_dict['labels'], eval_dict['probs'])
    
    return {
        "auc": auc,
    }