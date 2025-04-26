# dataset.py

import os
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from typing import List, Dict, Any
import math
import random

@dataclass
class TrainingConfig:
    data_dir: str
    model_path: str
    batch_size: int = 16
    max_seq_len: int = 64
    max_text_len: int = 32
    num_negatives: int = 256
    EVENT_TOEKN: str = '[EVENT]'


class TextEventSequencePairDataLoader:
    """
    sample text event sequence pair-wise dataset for training
    """
    def __init__(self, config: TrainingConfig, rank: int=0, prefix_prompt: str=''):
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
        # number of negatives
        self.num_negatives = math.ceil(config.num_negatives / (self.batch_size * self.world_size))

        # preload dataset
        self._preload_dataset()
    
    def _preload_dataset(self):
        """preload dataset and create shards info"""
        shards = os.listdir(self.data_dir)
        shards = sorted([shard for shard in shards if shard.endswith('.pkl')])
        self.shards = [os.path.join(self.data_dir, shard) for shard in shards]
        if self.rank == 0:
            print(f"Found {len(self.shards)} shards in {self.data_dir}")
        
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
            self.shards[self.current_shard_idx], 
            map_location='cpu', weights_only=True)
    
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

    def _preprocess_samples(self, buf: List[int]):
        """preprocess samples for training"""
        
        pos_tokens, pos_varlen, pos_position_ids = [], [], []
        neg_tokens, neg_varlen, neg_position_ids = [], [], []
        attention_mask, labels = [], []

        for global_idx in buf:
            # find the current shard
            shard_idx = next(
                i for i, cum in enumerate(self.cumulative_samples) if global_idx < cum
            ) - 1
            if shard_idx != self.current_shard_idx:
                self.current_shard_idx = shard_idx
                self.current_shard = torch.load(
                    self.shards[shard_idx], map_location='cpu', weights_only=True)
            
            # get the local index
            local_idx = global_idx - self.cumulative_samples[shard_idx]
            event_seq = self.current_shard[local_idx]['event']
            label = self.current_shard[local_idx]['label']

            # padding event_seq into max_seq_len
            event_seq, mask = self._padding_event_seq(event_seq)
            attention_mask.append(mask)
            labels.append(label)

            # craete the positive samples
            # event is a list [timestamp: str, event: str]
            for event in event_seq:
                timestamp, event = event
                # create the event token
                prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
                tokens = self.tokenizer.encode(prompt)[-self.max_text_len:]
                pos_tokens.extend(tokens)
                pos_varlen.append(len(tokens))
                pos_position_ids.extend((torch.arange(len(tokens)) + (self.max_text_len - len(tokens))).tolist())

            # create the negative samples
            neg_event_seq = self._sampling_event_sequence(self.num_negatives, local_idx)
            for event in neg_event_seq:
                timestamp, event = event
                # create the event token
                prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
                tokens = self.tokenizer.encode(prompt)[-self.max_text_len:]
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
            'attention_mask':   torch.as_tensor(attention_mask, dtype=torch.long)
        }

        
    def _padding_event_seq(self, event_seq: List[List[str]]):
        """padding event sequence into max_seq_len"""
        # padding event_seq into max_seq_len
        event_seq = event_seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(event_seq)
        mask = [1] * len(event_seq)
        if pad_len > 0:
            pad_event_seq = self._sampling_event_sequence(pad_len)
            # add padding event seq
            event_seq = pad_event_seq + event_seq
            mask = [0] * pad_len + mask

        return event_seq, mask
    
    def _sampling_event_sequence(self, num_samples: int, idx: int=None):
        """
        sampling events from the current shard
        if `idx` provided, using event from other samples
        """
        event_seq = []
        for i in range(num_samples):
            neg_idx = random.randint(0, len(self.current_shard) - 1)
            while idx is not None and neg_idx == idx:
                # get the negative sample
                neg_idx = random.randint(0, len(self.current_shard) - 1)
            
            event_len = len(self.current_shard[neg_idx]['event'])
            neg_event = self.current_shard[neg_idx]['event'][random.randint(0, event_len - 1)]
            event_seq.append(neg_event)
        
        return event_seq