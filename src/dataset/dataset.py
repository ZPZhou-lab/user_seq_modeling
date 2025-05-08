import os
import torch
from transformers import AutoTokenizer
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader
import math
import random
from src.arguments import TrainingConfig, TimeEmbeddingConfig
from typing import List
from abc import abstractmethod
from datetime import datetime
import logging
logger = logging.getLogger('Dataset')

def format_event(event_list: List[str]):
    action_time, event = event_list
    return action_time, event


def _set_world_size_and_rank(obj, world_size: int, rank: int):
    """
    Set the world size and rank for distributed training.
    """
    if world_size is None:
        obj.world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        obj.world_size = world_size
    
    if rank is None:
        obj.rank = dist.get_rank() if dist.is_initialized() else 0
    else:
        obj.rank = rank


class EventSequenceDataLoaderMeta:
    """
    sample text event sequence pair-wise dataset for training
    """
    def __init__(self, 
        config: TrainingConfig, 
        ts_config: TimeEmbeddingConfig,
        rank: int = 0,
        prefix_prompt: str = '',
        split: str = 'train',
    ):
        self.config = config
        self.ts_config = ts_config
        self.rank = rank
        self.world_size     = dist.get_world_size() if dist.is_initialized() else 1
        self.prefix_prompt  = prefix_prompt
        self.split          = split
        self.data_dir       = config.train_data_dir if split == 'train' else config.valid_data_dir
        self.model_path     = config.model_path
        self.batch_size     = config.batch_size
        self.max_seq_len    = config.max_seq_len
        self.max_text_len   = config.max_text_len
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path.value, trust_remote_code=True)
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
            samples = self.safe_load(shard)
            self.shard_samples.append(len(samples))
            self.cumulative_samples.append(self.cumulative_samples[-1] + len(samples))
        self.total_samples = self.cumulative_samples[-1]
        
        # init current shard
        self.current_pos = 0
        self.current_shard_idx = 0
        self.current_shard = self.safe_load()

    def __len__(self):
        return self.total_samples

    @property
    def total_steps(self):
        """total steps for one epoch iteration"""
        return self.total_samples // (self.batch_size * self.world_size)

    def next_batch(self):
        """get the next batch of data"""
        
        end = self.current_pos + self.batch_size * self.world_size
        if end > self.total_samples:
            # reset current pos
            self.current_pos = 0
            self.current_shard_idx = 0
            self.current_shard = self.safe_load()
            return self.next_batch()

        # get the global_idx in current batch
        buf = list(range(self.rank + self.current_pos, end, self.world_size))
        self.current_pos += self.batch_size * self.world_size

        # get the current batch
        return self._preprocess_samples(buf)


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
        self.current_shard = self.safe_load()
        
    def get_local_idx(self, global_idx: int):
        """
        get the local index from global index
        """
        shard_idx = next(
            i for i, cum in enumerate(self.cumulative_samples) if global_idx < cum
        ) - 1
        if shard_idx != self.current_shard_idx:
            self.current_shard_idx = shard_idx
            self.current_shard = self.safe_load()
        
        local_idx = global_idx - self.cumulative_samples[shard_idx]
        return local_idx
    
    def safe_load(self, path: str=None):
        """
        safe load the dataset shard
        """
        if path is None:
            path = self.shards[self.current_shard_idx]

        max_retries = 5
        for retry in range(max_retries):
            try:
                return torch.load(path, map_location='cpu', weights_only=True)
            except (AssertionError, Exception) as e:
                if retry < max_retries - 1:
                    logger.warning(f"Error loading shard {path}, retrying {retry+1}/{max_retries}: {str(e)}")
                    import time
                    time.sleep(1)  # Add a small delay before retrying
                else:
                    logger.error(f"Failed to load shard after {max_retries} attempts: {str(e)}")
                    # Return empty list as fallback to prevent crash
                    raise e


def get_action_time_diff(
    action_time: str, 
    observe_time: str,
    mode: str='relative',
    max_diff_day: int=720,
    max_year_ago: int=10
):
    """
    locate the `action_time` as diff from `observe_time`.

    Parameters
    ----------
    action_time: str
        the time when the action happened.
    observe_time: str
        the time when do the observation.
    mode: str
        the mode of time diff, one of `relative`, `absolute`, default is `relative`.\n
        - `relative`: locate the time as diff in tuple `(days, hours, minutes and seconds)`.\n
        - `absolute`: locate the time as tuple `(-year, month, day, hour, minute, second)`.\n
    max_diff_day: int
        the max diff day, default is 720 days, time_diff > 720 days will be truncated to 720 days,\
        only used when mode is `relative`
    max_year_age: int
        the max year age, default is 10 years, year > 10 years will be truncated to 10 years,\
        only used when mode is `absolute`
    """
    # calculate the date diff from now
    # get the day, hour, minute, second diff
    obs_time = datetime.strptime(observe_time, '%Y-%m-%d %H:%M:%S')
    act_time = datetime.strptime(action_time, '%Y-%m-%d %H:%M:%S')
    diff = obs_time - act_time
    # Extract total days
    day_diff = diff.days

    if mode == 'relative':
        if day_diff >= max_diff_day:
            return (max_diff_day - 1, 23, 59, 59)
        # Calculate remaining seconds
        remainder = diff.seconds
        # Extract hours, minutes, seconds
        hour_diff = remainder // 3600
        remainder = remainder % 3600
        minute_diff = remainder // 60
        second_diff = remainder % 60
        return (day_diff, hour_diff, minute_diff, second_diff)
    elif mode == 'absolute':
        year = max(obs_time.year - act_time.year, 0)
        year = min(year, max_year_ago)
        return (year, act_time.month, act_time.day, act_time.hour, act_time.minute, act_time.second)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'relative' or 'absolute'.")


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