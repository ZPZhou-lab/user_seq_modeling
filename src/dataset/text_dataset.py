import os
import torch
import random
import math
from typing import List
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from .dataset import (
    _set_world_size_and_rank,
    get_action_time_diff,
    format_event
)
from src.arguments import TrainingConfig, TimeEmbeddingConfig
from datasets import Dataset


def format_event(event_list: List[str]):
    action_time, event = event_list
    return action_time, event


class TableReader:
    def __init__(self, table: str):
        shards = os.listdir(table)
        shards = sorted([shard for shard in shards if shard.endswith('.pkl')])
        assert len(shards) > 0, f"no shards found in {table}"        
        shards = [os.path.join(table, shard) for shard in shards]
        self.reader = pd.concat([torch.load(shard, weights_only=False) for shard in shards], ignore_index=True)

    def to_pandas(self, start: int=0, limit: int=-1):
        if limit == -1:
            limit = len(self.reader)
        return self.reader.iloc[start:(start + limit)].copy()
    
    @property
    def table_size(self):
        return len(self.reader)
    
    def __len__(self):
        return len(self.reader)


class TextEventSequencePairDataset:
    def __init__(self, 
                 config: TrainingConfig,
                 ts_config: TimeEmbeddingConfig,
                 split: str='train',
                 prefix_prompt: str='',
                 world_size: int=None, 
                 rank: int=None):
        """
        Parameters
        ----------
        config: TrainingConfig
            The training configuration.
        ts_config: TimeEmbeddingConfig
            The time embedding configuration.
        split: str
            The dataset split, either 'train' or 'valid'.
        prefix_prompt: str
            The prefix prompt for the event.
        world_size: int
            The number of processes in the distributed training.
        rank: int
            The rank of the current process in the distributed training.
        """
        _set_world_size_and_rank(self, world_size, rank)
        self.config = config
        self.ts_config = ts_config
        self.split = split
        self.prefix_prompt = prefix_prompt
        # save config
        self.shard_size = config.shard_size
        self.batch_size = config.batch_size
        self.max_text_len = config.max_text_len
        self.max_seq_len  = config.max_seq_len
        
        # create tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path.value, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_tokens(config.EVENT_TOEKN, special_tokens=True)
        self.tokenizer.padding_side = "left"
        self.num_negatives = math.ceil(config.num_negatives / (self.batch_size * self.world_size))
        
        # init current shard
        self._preload_dataset()
        self.curr_shard_idx = 0
        self.curr_shard = self.safe_load()
    
    def _preload_dataset(self):
        data_dir = self.config.train_data_dir if self.split == 'train' else self.config.valid_data_dir
        self.reader = TableReader(table=data_dir)
        
        # create shard indices
        n_total = len(self.reader)
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
        frame = self.reader.to_pandas(start=start, limit=limit)
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
    
    def __getitem__(self, global_idx: int):
        """
        generate a sample from the dataset
        """

        # find the shard and local index
        local_idx = self._get_local_idx(global_idx)
        
        observe_time = self.curr_shard[local_idx]['observe_time']
        pos_event_seq = self.curr_shard[local_idx]['event']
        label = self.curr_shard[local_idx]['label']
        
        # padding event_seq into max_seq_len
        pos_event_seq, attention_mask = self._padding_event_sequence(pos_event_seq)

        # save the inputs
        pos_tokens, pos_varlen, pos_position_ids = [], [], []
        neg_tokens, neg_varlen, neg_position_ids = [], [], []
        time_ids = []

        # craete the positive samples
        # event is a list [timestamp: str, event: str]
        for event in pos_event_seq:
            action_time, event = format_event(event)
            # create the event token
            prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
            tokens = self.tokenizer.encode(prompt)[:self.max_text_len]
            pos_tokens.extend(tokens)
            pos_varlen.append(len(tokens))
            pos_position_ids.extend((torch.arange(len(tokens)) + (self.max_text_len - len(tokens))).tolist())
            time_ids.append(get_action_time_diff(
                action_time=action_time, 
                observe_time=observe_time,
                mode=self.ts_config.mode,
                max_diff_day=self.ts_config.max_diff_day,
                max_year_ago=self.ts_config.max_year_ago
            ))

        # create the negative samples
        neg_event_seq = self._sampling_event_sequence(self.num_negatives, local_idx)
        for event in neg_event_seq:
            _, event = format_event(event)
            # create the event token
            prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
            tokens = self.tokenizer.encode(prompt)[:self.max_text_len]
            neg_tokens.extend(tokens)
            neg_varlen.append(len(tokens))
            neg_position_ids.extend((torch.arange(len(tokens)) + (self.max_text_len - len(tokens))).tolist())

        
        return {
            'pos_input_ids':    torch.as_tensor(pos_tokens, dtype=torch.long),
            'pos_varlen':       torch.as_tensor(pos_varlen, dtype=torch.long),
            'pos_position_ids': torch.as_tensor(pos_position_ids, dtype=torch.long),
            'neg_input_ids':    torch.as_tensor(neg_tokens, dtype=torch.long),
            'neg_varlen':       torch.as_tensor(neg_varlen, dtype=torch.long),
            'neg_position_ids': torch.as_tensor(neg_position_ids, dtype=torch.long),
            'attention_mask':   torch.as_tensor(attention_mask, dtype=torch.long),
            'time_ids':         torch.as_tensor(time_ids, dtype=torch.long),
            'labels':           label
        }

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
            event_len = len(self.curr_shard[rnd_idx]['event'])
            event = self.curr_shard[rnd_idx]['event'][random.randint(0, event_len - 1)]
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


def sequential_event_collate_fn(samples):
    """
    collate function for the dataset
    """
    batchs = {}
    # concat input_ids, varlen, position_ids
    for key in ['pos_input_ids', 'neg_input_ids', 'pos_varlen', 'neg_varlen', 'pos_position_ids', 'neg_position_ids']:
        batchs[key] = torch.cat([sample[key] for sample in samples], dim=0)
    # stack attention_mask, time_ids, labels
    batchs['attention_mask'] = torch.stack([sample['attention_mask'] for sample in samples], dim=0)
    batchs['time_ids'] = torch.stack([sample['time_ids'] for sample in samples], dim=0)
    batchs['labels'] = torch.as_tensor([sample['labels'] for sample in samples], dtype=torch.long)
    
    return batchs