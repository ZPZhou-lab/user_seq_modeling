import torch
from typing import List
from torch.utils.data import Dataset
from .dataset import EventSequenceDataLoaderMeta
from .utils import get_action_time_diff
from datasets import Dataset


def format_event(event_list: List[str]):
    action_time, event = event_list
    return action_time, event


class TextEventSequencePairDataset(EventSequenceDataLoaderMeta):    
    def safe_load(self):
        """load the current shard"""
        start, limit = self._shards_loc[self.curr_shard_idx]
        frame = self.reader.to_pandas(start=start, limit=limit)
        shard = Dataset.from_pandas(frame)
        return shard

    def __getitem__(self, global_idx: int):
        # find the shard and local index
        local_idx = self._get_local_idx(global_idx)
        
        observe_time = self.curr_shard[local_idx]['observe_time']
        pos_event_seq = self.curr_shard[local_idx]['events']
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