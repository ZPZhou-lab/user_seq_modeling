import sys
if sys.platform == "darwin":
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from .modeling.modeling_llama import LlamaForCausalLM as CustomLlamaForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union


class EventEncoder(nn.Module):
    """
    Encodes events sequences into hidden_states
    """
    def __init__(self, 
        model_path: str,
        max_seq_len: int,
        use_flash_attention: bool = False
    ):
        super(EventEncoder, self).__init__()
        # load tokenizer and llm
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.use_cache = False
        hf_config.return_dict = True
        hf_config.use_ft_flash_attn = use_flash_attention
        self.llm = CustomLlamaForCausalLM.from_pretrained(
            model_path, 
            config=hf_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # add new embed_tokens to the llm
        self.llm.resize_token_embeddings(len(self.tokenizer) + 2, mean_resizing=True)

    def forward(self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        seq_varlen: torch.Tensor
    ) -> torch.Tensor:
        """
        encode events into hidden_states, a special token `[EVENT]` is padded to the end of each event
        and the last hidden_state of the sequence(i.e. the embedding of last token `[EVENT]`) is returned.

        Parameters
        ----------
        input_ids: torch.Tensor
            The tokenized events with shape (sql_len, )
        position_ids: torch.Tensor
            The position ids of the events with shape (sql_len, )
        seq_varlen: torch.Tensor
            The variable length of the event sequences with shape (batch, )
        """
        # get the event_embeddings
        hidden_states = self.llm.get_input_embeddings()(input_ids.to(self.llm.device))
        seq_len, embed_size = hidden_states.size(0), hidden_states.size(1)
        seq_varlen_cum = torch.cumsum(seq_varlen, dim=0)

        # using flash-attention to get the hidden states
        hidden_states = self.llm.base_model(
            inputs_embeds=hidden_states.unsqueeze(0), 
            cu_input_lens=seq_varlen.to(self.llm.device),
            position_ids=position_ids.unsqueeze(0).to(self.llm.device)
        ).last_hidden_state

        # extract the last token hidden state and reshape it to (batch, seq_len, hidden)
        hidden_states = hidden_states[:, seq_varlen_cum - 1, :].squeeze(0)
        hidden_states = hidden_states.view(-1, self.max_seq_len, embed_size)

        return hidden_states