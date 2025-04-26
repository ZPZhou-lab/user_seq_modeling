import os
import torch
from transformers import AutoTokenizer
import torch.distributed as dist
import math
import random
from src.arguments import TrainingConfig
from typing import List
from abc import abstractmethod
import logging
logger = logging.getLogger('Dataset')


class EventSequenceDataLoaderMeta:
    """
    sample text event sequence pair-wise dataset for training
    """
    def __init__(self, 
        config: TrainingConfig, 
        rank: int = 0,
        prefix_prompt: str = '',
    ):
        self.config = config
        self.rank = rank
        self.world_size     = dist.get_world_size() if dist.is_initialized() else 1
        self.prefix_prompt  = prefix_prompt
        self.data_dir       = config.data_dir
        self.model_path     = config.model_path
        self.batch_size     = config.batch_size
        self.max_seq_len    = config.max_seq_len
        self.max_text_len   = config.max_text_len
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        # add special tokens
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_tokens(config.EVENT_TOEKN, special_tokens=True)
        self.tokenizer.padding_side = "left"
        self.num_negatives = math.ceil(config.num_negatives / (self.batch_size * self.world_size))

        # preload metadata
        self._preload_dataset()

    def _preload_dataset(self):
        """preload dataset and create shards info"""
        shards = os.listdir(self.data_dir)
        shards = sorted([shard for shard in shards if shard.endswith('.pkl')])
        assert len(shards) > 0, f"no shards found in {self.data_dir}"
        
        self.shards = [os.path.join(self.data_dir, shard) for shard in shards]
        if self.rank == 0:
            logger.info(f"Found {len(self.shards)} shards in {self.data_dir}")
        
        # create shard info
        self.shard_samples, self.cumulative_samples = [], [0]
        for shard in self.shards:
            samples = torch.load(shard, map_location='cpu', weights_only=True)
            self.shard_samples.append(len(samples))
            self.cumulative_samples.append(self.cumulative_samples[-1] + len(samples))
        self.total_samples = self.cumulative_samples[-1]
        
        # init current shard
        self.current_pos = 0
        self.current_shard_idx = 0
        self.current_shard = torch.load(
            self.shards[self.current_shard_idx], map_location='cpu', weights_only=True)

    def __len__(self):
        return self.total_samples

    def next_batch(self):
        """get the next batch of data"""
        
        end = self.current_pos + self.batch_size * self.world_size
        if end > self.total_samples:
            # reset current pos
            self.current_pos = 0
            return self.next_batch()

        # get the global_idx in current batch
        buf = list(range(self.rank + self.current_pos, end, self.world_size))
        self.current_pos += self.batch_size * self.world_size

        # get the current batch
        samples = self._preprocess_samples(buf)
        return samples


    @abstractmethod
    def _preprocess_samples(self, buf: List[int]):
        """
        preprocess samples for given buf samples indices

        Parameters
        ----------
        buf : List[int]
            list of global indices for the current batch
        """
        raise NotImplementedError("_preprocess_samples() not implemented")


    def _sampling_event_sequence(self, num_samples: int, idx: int=None):
        """
        sampling events from the current shard
        if `idx` provided, using event from other samples for negative sampling
        """
        event_seq = []
        for i in range(num_samples):
            # generate random index
            rnd_idx = random.randint(0, len(self.current_shard) - 1)
            while idx is not None and rnd_idx == idx:
                # get the negative sample if idx is provided
                rnd_idx = random.randint(0, len(self.current_shard) - 1)
            
            # sampling a event
            event_len = len(self.current_shard[rnd_idx]['event'])
            event = self.current_shard[rnd_idx]['event'][random.randint(0, event_len - 1)]
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
        self.current_pos = 0
        self.current_shard_idx = 0
        self.current_shard = torch.load(
            self.shards[self.current_shard_idx], map_location='cpu', weights_only=True)
        
    def get_local_idx(self, global_idx: int):
        """
        get the local index from global index
        """
        shard_idx = next(
            i for i, cum in enumerate(self.cumulative_samples) if global_idx < cum
        ) - 1
        if shard_idx != self.current_shard_idx:
            self.current_shard_idx = shard_idx
            self.current_shard = torch.load(
                self.shards[self.current_shard_idx], map_location='cpu', weights_only=True)
        
        local_idx = global_idx - self.cumulative_samples[shard_idx]
        return local_idx