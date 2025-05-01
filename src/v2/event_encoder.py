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
from src.arguments import ModelPath


class EventEncoder(nn.Module):
    """
    Encodes events sequences into hidden_states
    """
    def __init__(self, 
        model_path: ModelPath,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.value, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path.value, trust_remote_code=True)
        hf_config.use_cache = False
        hf_config.return_dict = True
        if use_flash_attention:
            hf_config._attn_implementation = 'flash_attention_2'
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path.value, 
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
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        call llm to extract hidden states
        """
        # get embedding
        outputs = self.llm.base_model(
            input_ids=input_ids.to(self.llm.device),
            attention_mask=attention_mask.to(self.llm.device, dtype=torch.bool),
            # position_ids=position_ids.to(self.llm.device) if position_ids is not None else None,
        )
        # extract the last token hidden state
        return outputs.last_hidden_state[:, -1, :] 
    

    def _padding_hidden_states(self, hidden_states: Tuple[torch.Tensor], event_len: List[int]=None):
        if event_len is not None:
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
        else:
            # hidden_states is already padded
            return hidden_states, None