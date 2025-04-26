import sys
if sys.platform == 'darwin':
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class EventEncoder(nn.Module):
    """
    Encodes events sequences into hidden_states
    """
    def __init__(self, 
        model_path: str,
        max_seq_len: int,
        use_flash_attention: bool = False,
        num_add_tokens: int = 2,
    ):
        super(EventEncoder, self).__init__()
        # load tokenizer and llm
        self.EVENT_TOKEN = '[EVENT]'
        self.max_seq_len = max_seq_len
        self.use_flash_attention = use_flash_attention
        self.num_add_tokens = num_add_tokens

        # load pretrained llm
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.use_cache = False
        hf_config.return_dict = True
        if use_flash_attention:
            hf_config._attn_implementation = 'flash_attention_2'
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, 
            config=hf_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # add new embed_tokens to the llm
        self.llm.resize_token_embeddings(len(self.tokenizer) + num_add_tokens, mean_resizing=True)


    def forward(self, 
        input_ids: Dict[str, torch.Tensor], 
        event_len: List[int],
        padding: Optional[bool] = True
    ) -> torch.Tensor:
        """
        encode events into hidden_states, a special token `[EVENT]` is padded to the end of each event
        and the last hidden_state of the sequence(i.e. the embedding of last token `[EVENT]`) is returned.

        Parameters
        ----------
        input_ids: Dict[torch.Tensor]
            The tokenized events with shape (num_events_agg, sql_len) 
            with two keys: 'input_ids' and 'attention_mask'
        event_len: List[int]
            The variable length of the event sequences with shape (num_events_agg, )
        padding: bool
            Whether to pad the hidden states to the max_seq_len
        """       
        # call llm to extract hidden states with shape (num_events_agg, hidden_size)
        hidden_states = self._extract_hidden_states(**input_ids)
        # split hidden_states into chunks (batch, batch_seq_len, hidden_size)
        hidden_states = torch.split(hidden_states, event_len, dim=0)

        # padding hidden states to the max_seq_len
        if padding:
            return self._padding_hidden_states(hidden_states, event_len)
        else:
            return torch.stack(hidden_states, dim=0), None


    def _extract_hidden_states(self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        call llm to extract hidden states
        """
        # get embedding
        outputs = self.llm.base_model(
            input_ids=input_ids.to(self.llm.device),
            attention_mask=attention_mask.to(self.llm.device)
        )
        # extract the last token hidden state
        return outputs.last_hidden_state[:, -1, :] 
    

    def _padding_hidden_states(self, hidden_states: Tuple[torch.Tensor], event_len: List[int]):
        hidden_states = [torch.cat([
            torch.zeros(self.max_seq_len - event_len[i], event_seq.size(-1)).to(event_seq.device, dtype=event_seq.dtype),
            event_seq[-self.max_seq_len:, :]
        ], dim=0) for i, event_seq in enumerate(hidden_states)]
        # stack to (batch, max_len, hidden_size)
        hidden_states = torch.stack(hidden_states, dim=0)

        # create attention mask using events_len
        attention_mask = torch.zeros(size=hidden_states.size()[:-1], dtype=torch.long).to(hidden_states.device)
        for i, seq_len in enumerate(event_len):
            attention_mask[i, -seq_len:] = 1
        
        return hidden_states, attention_mask


class UserEncoder(nn.Module):
    def __init__(self,
        model_path: str,
    ):
        super(UserEncoder, self).__init__()
        # load pretrained llm
        llm = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # remove embedding layer and only keep the encoder
        self.encoder = llm.base_model
    
    def forward(self, 
        event_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        encode user inputs into hidden_states
        """
        # get embedding
        outputs = self.encoder(
            inputs_embeds=event_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False
        )
        # extract the last hidden state with shape (batch, seq_len, hidden_size)
        return outputs.last_hidden_state
    

class GenerativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, neg_samples=10):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.neg_samples = neg_samples
        
    def forward(self, 
        encoder_outputs: torch.Tensor,
        event_embeddings: torch.Tensor
    ):
        """
        Parameters
        ----------
        encoder_outputs: (batch, seq_len, hidden)
            The encoder outputs generated by the UserEncoder, are the next-event embeddings predictions
        event_embeddings: (batch, seq_len, hidden)
            The event embeddings generated by the EventEncoder, are the inputs of UserEncoder
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # postive samples
        positives = event_embeddings[:, 1:, :]  # (batch, seq_len-1, hidden)
        predictions = encoder_outputs[:, :-1, :]  # (batch, seq_len-1, hidden)
        
        # randomly sample negative samples within the batch from the other event sequences
        # with shape (batch, seq_len-1, neg_samples, hidden)
        negatives = self._negative_sampling(event_embeddings)  
        
        # calculate the similarity scores
        pos_scores = F.cosine_similarity(predictions, positives, dim=-1) / self.temperature  # (batch, seq_len-1)
        neg_scores = torch.einsum('bsh,bsnh->bsn', predictions, negatives) / self.temperature  # (batch, seq_len-1, neg_samples)
        
        # combine positive and negative scores (batch, seq_len-1, neg_samples+1)
        logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        
        # calculate the info NCE loss
        labels = torch.zeros(batch_size, seq_len-1, dtype=torch.long, device=encoder_outputs.device)
        loss = F.cross_entropy(logits.view(-1, self.neg_samples + 1), labels.view(-1))
        
        return loss
    
    def _negative_sampling(self, event_embeddings: torch.Tensor):
        """
        sample negative samples from the event embeddings
        """
        device = event_embeddings.device
        batch_size, seq_len, hidden_dim = event_embeddings.shape

        # flatten into (batch * seq_len, hidden_dim)
        event_embeddings = event_embeddings.view(-1, hidden_dim)

        event_indices = torch.arange(batch_size * seq_len, device=device)
        neg_indices = []
        mask_offset = 0
        for b in range(batch_size):
            # create mask for the current batch
            mask = torch.where((event_indices - mask_offset) // seq_len)[0]
            random_indices = torch.randperm(len(mask) * (seq_len - 1)) % len(mask)
            random_indices = random_indices[:self.neg_samples * (seq_len - 1)]
            neg_indices.append(mask[random_indices])
            mask_offset += seq_len
        # stack negative indices and gather negative samples
        neg_indices = torch.concat(neg_indices, dim=0)
        negatives = event_embeddings[neg_indices]
        # reshape negatives to (batch, seq_len-1, neg_samples, hidden_dim)
        negatives = negatives.view(batch_size, seq_len-1, self.neg_samples, hidden_dim)
        
        return negatives