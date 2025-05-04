# common.py
import os
import torch
import torch.distributed as dist
from torch import nn
from typing import List

def all_gather(data,
               group=None,
               sync_grads=False):
    group = group if group is not None else torch.distributed.group.WORLD
    if torch.distributed.get_world_size() > 1:
        from torch.distributed import nn
        if sync_grads:
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
        with torch.no_grad():
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
    else:
        return data.unsqueeze(0)


def create_device_info():
    # create device
    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return local_rank, device


def build_activation_function(activation: str):
    """
    Build activation function.
    """
    ACTIVATIONS = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'swish': nn.SiLU(),
        'silu': nn.SiLU()
    }
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function: {activation}")
    return ACTIVATIONS[activation]


class ClassificationHead(nn.Module):
    def __init__(self, 
        num_hiddens: int,
        num_classes: int,
        dropout: float=0.0,
        activation: str='relu',
        **kwargs
    ):
        super(ClassificationHead, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        # build module
        self.proj_in = nn.Linear(num_hiddens, num_hiddens)
        self.proj_out = nn.Linear(num_hiddens, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = build_activation_function(activation)
        # init weights
        self._init_std = kwargs.get('init_std', 0.02)
        self._init_bias = kwargs.get('init_bias', 0.0)
        self.apply(self._init_weights)
        
    def forward(self, x):
        return self.proj_out(self.dropout_layer(self.activation(self.proj_in(x))))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, self._init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, self._init_bias)


class TimeStampEmbedding(nn.Module):
    def __init__(self,
        num_hiddens: int, 
        mode: str='absolute', 
        time_hiddens: int=512,
        max_diff_day: int=720,
        max_year_age: int=10,
        mixup_activation: str='silu',
        **kwargs
    ):
        super().__init__()
        # mode control the precision of time 
        # mode `absolute` need tuple with `(year, month, day, hour, minute, second)`
        # mode `relative` need tuple with `(days, hours, minutes and seconds)`
        self.mode = mode
        self.num_hiddens = num_hiddens
        self.time_hiddens = time_hiddens
        self.max_diff_day = max_diff_day
        self.max_year_age = max_year_age
        self.mixup_activation = mixup_activation

        # build the embedding layer
        if self.mode == 'absolute':
            self.embed_time = nn.ModuleList([nn.Embedding(x, time_hiddens) for x in [max_year_age, 13, 32, 24, 60, 60]])
            self.num_time_loc = 6
        else:
            self.embed_time = nn.ModuleList([nn.Embedding(x, time_hiddens) for x in [max_diff_day, 24, 60, 60]])
            self.num_time_loc = 4
        
        # use a MLP to mix the time embeddings into hidden states
        self.time_mixup = nn.Sequential(
            nn.Linear(self.num_time_loc * time_hiddens, num_hiddens),
            build_activation_function(mixup_activation),
            nn.Linear(num_hiddens, num_hiddens)
        )
        # init weights
        self._init_std = kwargs.get('init_std', 0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, self._init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, self._init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    
    def forward(self, timestamps: torch.Tensor):
        """
        forward timestamp to time embedding

        Parameters
        ----------
        timestamps: torch.Tensor
            the timestamps of each event, with shape `(batch, seq_len, num_time_loc)` or `(seq_len, num_time_loc)`.
        """
        # get the shape and check the input
        orig_shape = timestamps.shape
        T = timestamps.size(2) if timestamps.ndim == 3 else timestamps.size(1)
        assert T == self.num_time_loc, f"the input shape {timestamps.shape} is not match the time loc {self.num_time_loc}"
        # reshape to 2D tensor
        if timestamps.ndim == 3:
            timestamps = timestamps.view(-1, T)

        # (N, T) -> (T, N, hiddens) -> (N, T * hiddens)
        embeddings = [
            self.embed_time[i](timestamps[..., i]) for i in range(self.num_time_loc)
        ]
        embeddings = torch.cat(embeddings, dim=-1)
        # (N, T * hiddens) -> (N, hiddens)
        embeddings = self.time_mixup(embeddings)
        return embeddings.view(*orig_shape[:-1], -1)


class Accumulator:
    def __init__(self):
        self._values = {}
        self.count = 0

    def reset(self):
        self._values = {}
        self.count = 0
    
    def update(self, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def synchronize(self, ddp=False):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def __getitem__(self, key):
        if key in self._values:
            return self._values[key]
        else:
            raise KeyError(f"Key {key} not found in accumulator.")


class ScalerAccumulator(Accumulator):
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.values:
                self._values[key] = 0.0
            self._values[key] += value.detach() if hasattr(value, 'detach') else value
        self.count += 1
                
    def synchronize(self, ddp=False):
        if ddp:
            for key in self._values:
                tensor = self._values[key]
                if isinstance(tensor, torch.Tensor):
                    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    @property
    def values(self):
        values = {key: value / max(1, self.count) for key, value in self._values.items()}
        # call item() if value is a tensor
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.item()
        return values


class TensorAccumulator(Accumulator):
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._values:
                self._values[key] = value
            else:
                self._values[key] = torch.cat((self._values[key], value), dim=0)
        self.count += 1
    
    def synchronize(self, ddp=False):
        if ddp:
            for key in self._values:
                value: torch.Tensor = all_gather(self._values[key])
                self._values[key] = torch.cat([v for v in value], dim=0)
    
    @property
    def values(self):
        values = {key: value for key, value in self._values.items()}
        # call item() if value is a tensor
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.cpu().float().numpy()
        return values