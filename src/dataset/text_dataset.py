from turtle import pos
import torch
from typing import List
from torch.utils.data import Dataset

from src.arguments import TrainingConfig
from .dataset import EventSequenceDataLoaderMeta
from .utils import TableReader, get_action_time_diff
from datasets import Dataset


def format_event(event_list: List[str]):
    action_time, event = event_list
    return action_time, event


class TextEventSequencePairDataset(EventSequenceDataLoaderMeta):    
    # def safe_load(self):
    #     """load the current shard"""
    #     start, limit = self._shards_loc[self.curr_shard_idx]
    #     frame = TableReader(table=self.data_dir).to_pandas(start=start, limit=limit)
    #     shard = Dataset.from_pandas(frame)
    #     return shard

    def __getitem__(self, global_idx: int):
        if self.EVENT_TOKEN is None:
            self.EVENT_TOKEN = self.tokenizer.encode(self.config.EVENT_TOEKN) 
        # find the shard and local index
        local_idx = self._get_local_idx(global_idx)
        
        observe_time = self.curr_shard[local_idx]['observe_time']
        pos_event_seq = self.curr_shard[local_idx]['events']
        label = self.curr_shard[local_idx]['label']
        
        # padding event_seq into max_seq_len
        if self.config.padding:
            pos_event_seq, attention_mask = self._padding_event_sequence(pos_event_seq)
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        else:
            pos_event_seq = pos_event_seq[-self.max_seq_len:]
            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask[-len(pos_event_seq):] = 1
            # copy a fake event at the end as user token
            # this token will be replaced in inference
            if self.config.add_user_token:
                pos_event_seq.append(pos_event_seq[-1])

        # save the inputs
        pos_tokens, pos_varlen, pos_position_ids = [], [], []
        neg_tokens, neg_varlen, neg_position_ids = [], [], []
        time_ids = []

        # craete the positive samples
        # event is a list [timestamp: str, event: str]
        for event in pos_event_seq:
            action_time, event = format_event(event)
            # create the event token
            prompt = self.prefix_prompt + event
            tokens = self.tokenizer.encode(prompt)[:(self.max_text_len - 1)] + self.EVENT_TOKEN
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
            prompt = self.prefix_prompt + event
            tokens = self.tokenizer.encode(prompt)[:(self.max_text_len - 1)] + self.EVENT_TOKEN
            neg_tokens.extend(tokens)
            neg_varlen.append(len(tokens))
            neg_position_ids.extend((torch.arange(len(tokens)) + (self.max_text_len - len(tokens))).tolist())

        # prepare for user sequence
        user_varlen = len(pos_event_seq)
        offset = self.max_seq_len - user_varlen if self.config.add_user_token else (self.max_seq_len + 1 - user_varlen)
        user_position_ids = torch.arange(user_varlen) + offset
        user_token_mask = torch.zeros(user_varlen, dtype=torch.bool)
        user_token_mask[-1] = True

        return {
            'pos_input_ids':    torch.as_tensor(pos_tokens, dtype=torch.long),
            'pos_varlen':       torch.as_tensor(pos_varlen, dtype=torch.long),
            'pos_position_ids': torch.as_tensor(pos_position_ids, dtype=torch.long),
            'neg_input_ids':    torch.as_tensor(neg_tokens, dtype=torch.long),
            'neg_varlen':       torch.as_tensor(neg_varlen, dtype=torch.long),
            'neg_position_ids': torch.as_tensor(neg_position_ids, dtype=torch.long),
            'attention_mask':   torch.as_tensor(attention_mask, dtype=torch.long),
            'user_varlen':      user_varlen,
            'user_position_ids':user_position_ids,
            'user_token_mask':  user_token_mask,
            'time_ids':         torch.as_tensor(time_ids, dtype=torch.long),
            'labels':           label
        }

    def sequential_event_collate_fn(self, samples):
        """
        collate function for the dataset
        """
        pos_input_ids, pos_varlen, pos_position_ids = [], [], []
        neg_input_ids, neg_varlen, neg_position_ids = [], [], []
        attention_mask, labels, time_ids = [], [], []
        user_varlen, user_position_ids, user_token_mask = [], [], []
        
        # add the negative samples
        for sample in samples:    
            neg_input_ids.append(sample['neg_input_ids'])
            neg_varlen.append(sample['neg_varlen'])
            neg_position_ids.append(sample['neg_position_ids'])
    
        # process the positive samples
        for sample in samples:
            # add the positive samples
            pos_input_ids.append(sample['pos_input_ids'])
            pos_varlen.append(sample['pos_varlen'])
            pos_position_ids.append(sample['pos_position_ids'])
            attention_mask.append(sample['attention_mask'])
            time_ids.append(sample['time_ids'])
            labels.append(sample['labels'])

            if not self.config.padding:
                user_varlen.append(sample['user_varlen'])
                user_position_ids.append(sample['user_position_ids'])
                user_token_mask.append(sample['user_token_mask'])
        
        # format batchs
        batchs = {
            'pos_input_ids':    torch.cat(pos_input_ids, dim=0),
            'pos_varlen':       torch.cat(pos_varlen, dim=0),
            'pos_position_ids': torch.cat(pos_position_ids, dim=0),
            'neg_input_ids':    torch.cat(neg_input_ids, dim=0),
            'neg_varlen':       torch.cat(neg_varlen, dim=0),
            'neg_position_ids': torch.cat(neg_position_ids, dim=0),
            'attention_mask':   torch.stack(attention_mask, dim=0),
            'labels':           torch.as_tensor(labels, dtype=torch.long),
            'time_ids':         torch.stack(time_ids, dim=0) if self.config.padding else torch.cat(time_ids, dim=0)
        }
        if not self.config.padding:
            batchs['user_varlen']       = torch.as_tensor(user_varlen, dtype=torch.long)
            batchs['user_position_ids'] = torch.cat(user_position_ids, dim=0)
            batchs['user_token_mask']   = torch.cat(user_token_mask, dim=0)
        
        return batchs