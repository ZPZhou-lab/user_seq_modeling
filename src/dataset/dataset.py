import torch
import math
import random
from transformers import AutoTokenizer
from torch.utils.data import Sampler, DataLoader
from src.arguments import TrainingConfig, TimeEmbeddingConfig
from typing import List
from abc import abstractmethod
from datasets import Dataset
from .utils import (
    _set_world_size_and_rank,
    TableReader
)
import logging
logger = logging.getLogger('Dataset')


class EventSequenceDataLoaderMeta:
    """
    sample text event sequence pair-wise dataset for training
    """
    def __init__(self, 
        config: TrainingConfig, 
        ts_config: TimeEmbeddingConfig,
        split: str = 'train',
        prefix_prompt: str = '',
        world_size: int=None, 
        rank: int = None
    ):
        _set_world_size_and_rank(self, world_size, rank)
        self.config = config
        self.ts_config = ts_config
        self.split          = split
        self.prefix_prompt  = prefix_prompt
        # save config
        self.shard_size     = config.shard_size
        self.batch_size     = config.batch_size
        self.max_seq_len    = config.max_seq_len
        self.max_text_len   = config.max_text_len
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path.value, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_tokens(config.EVENT_TOEKN, special_tokens=True)
        self.tokenizer.padding_side = "left"
        self.num_negatives = math.ceil(config.num_negatives / (self.batch_size * self.world_size))

        # preload metadata
        self._preload_dataset()
        self.curr_shard_idx = 0
        self.curr_shard = self.safe_load()

    def _preload_dataset(self):
        self.data_dir = self.config.train_data_dir if self.split == 'train' else self.config.valid_data_dir
        reader = TableReader(table=self.data_dir)
        
        # create shard indices
        n_total = reader.table_size
        self._shards_loc = []
        for shard_s in range(0, n_total, self.shard_size * self.world_size):
            shard_e = shard_s + self.shard_size * self.world_size
            if shard_e <= n_total:
                start = shard_s + self.rank * self.shard_size
                limit = (self.shard_size // self.batch_size) * self.batch_size
            else:
                # split the last part into `world_size` shards if remain > batch_size * world_size
                remain = n_total - shard_s
                global_batch = self.world_size * self.batch_size
                if remain > global_batch:
                    last_shard_size = (remain // global_batch) * global_batch // self.world_size
                    start = shard_s + self.rank * last_shard_size
                    limit = last_shard_size
                else:
                    # remain can not be split into `world_size` chunks to build a batch
                    break
            self._shards_loc.append((start, limit))
        # the total samples in the dataset
        self._total = sum([limit for _, limit in self._shards_loc]) * self.world_size
    
    def __len__(self):
        return self._total

    def safe_load(self):
        """load the current shard"""
        start, limit = self._shards_loc[self.curr_shard_idx]
        frame = TableReader(table=self.data_dir).to_pandas(start=start, limit=limit)
        shard = Dataset.from_pandas(frame)
        return shard
    
    @property
    def curr_shard_range(self):
        """get the shard range"""
        return self._shards_loc[self.curr_shard_idx]
    
    @property
    def curr_shard_size(self):
        """get the shard size"""
        return self._shards_loc[self.curr_shard_idx][1]
    

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        generate a sample from the dataset
        """
        raise NotImplementedError("Please implement __getitem__ method in the subclass.")


    def _get_local_idx(self, global_idx: int):
        """check if need to load the next shard and return the local index for the current shard"""
        shard_end = self.curr_shard_range[0] + self.curr_shard_range[1]
        # fetch the next shard if the current shard reaches the end
        if global_idx >= shard_end:
            self.curr_shard_idx = (self.curr_shard_idx + 1) % len(self._shards_loc)
            self.curr_shard = self.safe_load()
        elif global_idx < self.curr_shard_range[0]:
            self.curr_shard_idx = 0
            self.curr_shard = self.safe_load()
        
        # get the local index in the current shard
        local_idx = global_idx - self.curr_shard_range[0]
        return local_idx

    def _sampling_event_sequence(self, num_samples: int, idx: int=None):
        """
        sampling events from the current shard
        if `idx` provided, using event from other samples for negative sampling
        """
        event_seq = []
        for i in range(num_samples):
            # generate random index
            rnd_idx = random.randint(0, len(self.curr_shard) - 1)
            while idx is not None and rnd_idx == idx:
                # get the negative sample if idx is provided
                rnd_idx = random.randint(0, len(self.curr_shard) - 1)
            
            # sampling a event
            event_len = len(self.curr_shard[rnd_idx]['events'])
            event = self.curr_shard[rnd_idx]['events'][random.randint(0, event_len - 1)]
            event_seq.append(event)
        
        return event_seq
    
    def _padding_event_sequence(self, event_seq: List[List[str]]):
        """
        padding event sequence into max_seq_len, and create attention mask
        """
        event_seq = event_seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(event_seq)
        mask = [1] * len(event_seq)
        if pad_len > 0:
            pad_event_seq = self._sampling_event_sequence(pad_len)
            # add padding event seq
            event_seq = pad_event_seq + event_seq
            mask = [0] * pad_len + mask

        return event_seq, mask
    
    def reset(self):
        """
        reset the current shard and position
        """
        self.curr_shard_idx = 0
        self.curr_shard = self.safe_load()


class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset, world_size: int=None, rank: int=None):
        self.dataset = dataset
        _set_world_size_and_rank(self, world_size, rank)

    def __iter__(self):
        """
        each node load a shard of data with `shard_size` samples, 
        there are `num_replicas * shard_size` samples in total at the same time.\n
        The sampler samples the sequence of indices from the dataset._shards_idx intervals.
        """
        indices = []
        for start, limit in self.dataset._shards_loc:
            # For each shard assigned to this rank, generate indices within that shard
            for i in range(limit):
                indices.append(start + i)
        return iter(indices)

    def __len__(self):
        shards = [limit for _, limit in self.dataset._shards_loc]
        return sum(shards)


class EventSequencePairLoaaderWrapper:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.num_negatives = dataloader.dataset.num_negatives
        self._iter = iter(dataloader)
    
    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            # reset the iterator if it reaches the end
            self._iter = iter(self.dataloader)
            return next(self._iter)
    
    def __len__(self):
        return len(self.dataloader)


def build_dataloader(
    dataset,
    config: TrainingConfig, 
    collate_fn: callable,
    rank: int=0,
    **kwargs
):
    sampler = SequentialDistributedSampler(dataset, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=kwargs.get('num_workers', 0),
        pin_memory=kwargs.get('pin_memory', True),
        multiprocessing_context=kwargs.get('multiprocessing_context', None),
        persistent_workers=True if kwargs.get('num_workers', 0) > 0 else False,
        drop_last=True
    )
    
    return EventSequencePairLoaaderWrapper(data_loader)