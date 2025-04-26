# dataset.py
import torch
from typing import List, Dict, Any, Union
from src.arguments import TrainingConfig
from src.dataset import EventSequenceDataLoaderMeta


class TextEventSequencePairDataLoader(EventSequenceDataLoaderMeta):
    """
    sample text event sequence pair-wise dataset for training
    """
    def _preprocess_samples(self, buf: List[int]):
        time_ids, labels = [], []
        pos_event_len, pos_events_tokens = [], []
        neg_event_len, neg_events_tokens = [], []

        for global_idx in buf:
            # find the shard and local index
            local_idx = self.get_local_idx(global_idx)

            # truncate the event sequence to max_seq_len
            pos_event_seq = self.current_shard[local_idx]['event'][-self.max_seq_len:]
            neg_event_seq = self._sampling_event_sequence(self.num_negatives, local_idx)
            label = self.current_shard[local_idx]['label']


            pos_event_len.append(len(pos_event_seq))
            neg_event_len.append(len(neg_event_seq))
            labels.append(label)
            
            for event in pos_event_seq:
                # tokenize the event
                timestamp, event = event
                prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
                tokens = self.tokenizer.encode(prompt)
                pos_events_tokens.append(tokens[:self.max_text_len])
                time_ids.append(timestamp)

            for event in neg_event_seq:
                # tokenize the event
                timestamp, event = event
                prompt = self.prefix_prompt + event + self.config.EVENT_TOEKN
                tokens = self.tokenizer.encode(prompt)
                neg_events_tokens.append(tokens[:self.max_text_len])

        # pad tokens and create attention mask
        pos_input_ids = self.tokenizer.pad(
            encoded_inputs={'input_ids': pos_events_tokens}, 
            padding='max_length', max_length=self.max_text_len, return_tensors='pt',
        )
        neg_input_ids = self.tokenizer.pad(
            encoded_inputs={'input_ids': neg_events_tokens}, 
            padding='max_length', max_length=self.max_text_len, return_tensors='pt',
        )
        # add position ids
        # pos_input_ids['position_ids'] = torch.stack([
        #     torch.as_tensor([0] * (self.max_text_len - seq_len) + list(range(seq_len)), dtype=torch.long)
        #     for seq_len in pos_input_ids['attention_mask'].sum(dim=1)])
        # neg_input_ids['position_ids'] = torch.stack([
        #     torch.as_tensor([0] * (self.max_text_len - seq_len) + list(range(seq_len)), dtype=torch.long)
        #     for seq_len in neg_input_ids['attention_mask'].sum(dim=1)])
        pos_input_ids['position_ids'] = None
        neg_input_ids['position_ids'] = None

        return {
            'pos_input_ids':    pos_input_ids,
            'pos_event_len':    pos_event_len,
            'neg_input_ids':    neg_input_ids,
            'neg_event_len':    neg_event_len,
            'time_ids':         time_ids,
            'labels':           torch.as_tensor(labels, dtype=torch.int32),
        }