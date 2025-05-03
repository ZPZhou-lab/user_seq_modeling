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
        'swish': nn.SiLU()
    }
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function: {activation}")
    return ACTIVATIONS[activation]


class ClassificationHead(nn.Module):
    def __init__(self, 
        num_hiddens: int,
        num_classes: int,
        dropout: float=0.0,
        activation: str='relu'
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
        
    def forward(self, x):
        return self.proj_out(self.dropout_layer(self.activation(self.proj_in(x))))

    def init_weights(self):
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


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