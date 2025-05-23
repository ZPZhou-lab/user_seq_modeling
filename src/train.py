import os
import torch
from torch.optim import Optimizer
import time
import math
from .arguments import TrainingConfig
from .common import ScalerAccumulator, TensorAccumulator
from .hierarchical import HierarchicalModel, HierarchicalModelOutput
from .dataset import EventSequencePairLoaaderWrapper
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from typing import Dict

# cos-decay with warm-up learning rate scheduler
class LearningRateScheduler:
    def __init__(self, 
        config: TrainingConfig,
        optimizer: Optimizer,
        use_deepspeed: bool=False,
        lower_pct: float=0.1
    ):
        self.optimizer          = optimizer
        self.warmup_steps       = config.warmup_steps
        self.max_steps          = config.max_steps
        self.top_warmup_steps   = config.top_warmup_steps
        self.max_lr             = config.learning_rate
        self.min_lr             = lower_pct * config.learning_rate
        self.use_deepspeed      = use_deepspeed
        self.accum_steps        = config.grad_accum_steps if use_deepspeed else 1
        self.current_step       = 0
        self._local_step        = 0
        self._top_warmup_flag   = True if config.top_warmup_steps > 0 else False
        
        # save the original parameter groups
        self.orig_param_groups = []
        for group in optimizer.param_groups:
            self.orig_param_groups.append({
                'params': group['params'], 
                'lr': group.get('lr', config.learning_rate),
                'weight_decay': group.get('weight_decay', config.weight_decay)})
    
    def init(self):
        """initialize the learning rate for each parameter group"""
        # set the learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            name = param_group['name'].split('.')[0]
            if name == 'classifier':
                param_group['lr'] = self.min_lr
            else:
                param_group['lr'] = 0.0

    def step(self):
        """update the learning rate for each step consider accumlation steps"""
        self.current_step += 1
        # if reach accumulation steps, update the learning rate
        if self.current_step % self.accum_steps == 0:
            self._step()
    
    def _step(self):
        """update the learning rate for each step"""        
        # stap 1: warmup for head classifier
        self._local_step += 1
        if self._top_warmup_flag:
            
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
            if self._local_step == self.top_warmup_steps:
                self._top_warmup_flag = False
                self._local_step = 0
        else:
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
        self._values = {'count': 0}
    
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
    
    def accum(self, **kwargs):
        """Accumulate the metrics to be logged"""
        self._values['count'] += 1
        for key, value in kwargs.items():
            if key not in self._values:
                self._values[key] = 0
            self._values[key] += value
    
    @property
    def values(self):
        """Get the accumulated values"""
        if len(self._values) == 1:
            raise ValueError("Logger has no values to log")
        values = {key: value / self._values['count'] for key, value in self._values.items()}
        values.pop('count')
        # reset the values
        self._values = {'count': 0}

        return values
    
    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def trigger_logger(self, step):
        """Trigger the logger to log the metrics"""
        return (step % self.log_freq) == 0


def train_step(
    config: TrainingConfig,
    model: HierarchicalModel,
    train_loader: EventSequencePairLoaaderWrapper,
    optimizer: Optimizer,
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
    optimizer.zero_grad()

    batch_sizes = []
    for micro_step in range(config.grad_accum_steps):
        batch = next(train_loader)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch['num_negatives'] = train_loader.num_negatives
        batch['is_padded'] = config.padding
        batch_sizes.append(batch['attention_mask'].size(0))
        
        # set gradient sync
        if ddp:
            model.require_backward_grad_sync = micro_step == config.grad_accum_steps - 1
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
    e_time = time.time()
    actual_batch_size = sum(batch_sizes)
    time_used = (e_time - s_time) / actual_batch_size

    if master_process and tb_logger is not None:
        # accumulate the metrics
        tb_logger.accum(
            nce_loss=loss_dict['nce_loss'],
            ce_loss=loss_dict['ce_loss'],
            loss=loss_dict['loss'],
            time_per_iter=time_used,
            batch_size=actual_batch_size,
            grad_norm=norm.item() if isinstance(norm, torch.Tensor) else norm
        )
        # Export metrics
        if tb_logger.trigger_logger(step):
            metrics = tb_logger.values
            ce_loss, nce_loss = metrics.pop('ce_loss'), metrics.pop('nce_loss')
            raw_model = model.module if ddp else model
            metrics['temperature'] = raw_model.temperature
            metrics['lr'] = lr_scheduler.get_lr()
            metrics['task'] = {'ce_loss': ce_loss, 'nce_loss': config.nce_loss_lambda * nce_loss}
            tb_logger.log(metrics, step, prefix="train")


def valid_context(
    config: TrainingConfig,
    model: HierarchicalModel,
    valid_loader: EventSequencePairLoaaderWrapper,
    optimizer: Optimizer,
    eval_step: int,
    ddp: bool=False,
    device: str='cuda',
    master_process: bool=False,
    tb_logger: TensorboardLogger = None
):
    # validation loop
    model.eval()
    s_time = time.time()
    loss_tracker = ScalerAccumulator()
    eval_tracker = TensorAccumulator()

    # valid_loader.reset()
    with torch.no_grad():
        for step in range(min(len(valid_loader), config.max_evel_iter)):
            # get the current batch
            batch = next(valid_loader)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['num_negatives'] = valid_loader.num_negatives
            batch['is_padded'] = config.padding
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # call model
                outputs: HierarchicalModelOutput = model(**batch)
            # calculate loss
            loss = config.nce_loss_lambda * outputs.nce_loss + outputs.ce_loss
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

    if master_process and tb_logger is not None:
        # Export metrics
        metrics = {
            **loss_dict,
            **eval_dict,
            "eval_time_cost": time_used,
        }
        tb_logger.log(metrics, eval_step, prefix="valid")

        # save the checkpoint
        raw_model = model.module if ddp else model
        raw_model.save_pretrained(save_path=os.path.join(config.get_save_dir(), f"ckpt-{eval_step:06d}"))
        # save the optimizer state
        torch.save(optimizer.state_dict(), os.path.join(config.get_save_dir(), f"ckpt-{eval_step:06d}", "optimizer.pt"))


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