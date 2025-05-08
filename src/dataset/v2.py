# dataset.py
import os
import torch
from typing import List
import pandas as pd
from .dataset import (
    EventSequenceDataLoaderMeta, 
    get_action_time_diff, 
    format_event
)
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


class TextEventSequencePairDataLoader(EventSequenceDataLoaderMeta):
    """
    sample text event sequence pair-wise dataset for training
    """
    def __init__(self, config, ts_config, rank = 0, prefix_prompt = '', split = 'train'):
        self.shard_size = config.shard_size
        super().__init__(config, ts_config, rank, prefix_prompt, split)
        
    def _preload_dataset(self):
        """preload dataset and create transformers dataset"""

        # create shard info
        self.reader = TableReader(self.data_dir)
        self.total_samples = self.reader.table_size
        self.shards = []
        for shard in range(0, self.total_samples, self.shard_size * self.world_size):
            starts = [shard + i * self.shard_size for i in range(self.world_size)]
            limits = [min(self.shard_size, self.total_samples - start) for start in starts]
            start, limit = starts[self.rank], min(limits)
            self.shards.append((start, limit))
        if self.rank == 0:
            print(f"There are {len(self.shards)} shards in each node")
                
        # init current shard
        self.current_pos = 0
        self.current_shard_idx = 0
        # load the dataset
        self.current_shard = self.safe_load()

    
    def safe_load(self, path: str=None):
        start, limit = self.shards[self.current_shard_idx]
        df = self.reader.to_pandas(start=start, limit=limit)
        shard = Dataset.from_pandas(df)
        return shard
    
    def next_batch(self):
        """get the next batch of data"""
        end = self.current_pos + self.batch_size
        if end > self.shards[self.current_shard_idx][1]:
            # reset current pos
            self.current_pos = 0
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.current_shard = self.safe_load()
            return self.next_batch()

        # get the global idx in the current batch
        buf = list(range(self.current_pos, end))
        self.current_pos += self.batch_size

        # get the current batch
        return self._preprocess_samples(buf)

    def get_local_idx(self, global_idx: int):
        return global_idx

    def _preprocess_samples(self, buf: List[int]):
        pos_tokens, pos_varlen, pos_position_ids = [], [], []
        neg_tokens, neg_varlen, neg_position_ids = [], [], []
        attention_mask, labels, time_ids = [], [], []

        for global_idx in buf:
            # find the shard and local index
            local_idx = self.get_local_idx(global_idx)
            
            observe_time = self.current_shard[local_idx]['observe_time']
            pos_event_seq = self.current_shard[local_idx]['event']
            label = self.current_shard[local_idx]['label']
            
            # padding event_seq into max_seq_len
            pos_event_seq, mask = self._padding_event_sequence(pos_event_seq)
            attention_mask.append(mask)
            labels.append(label)

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
            'time_ids':         torch.as_tensor(time_ids, dtype=torch.long).view(self.batch_size, self.max_seq_len, -1),
            'labels':           torch.as_tensor(labels, dtype=torch.int32),
        }