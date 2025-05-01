import sys
import os
if sys.platform == "darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from transformers import AutoTokenizer, AutoConfig
from .modeling.modeling_llama import LlamaForCausalLM as CustomLlamaForCausalLM
from .modeling.modeling_qwen3 import Qwen3ForCausalLM as CustomQwen3ForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any
from src.arguments import ModelPath


class EventEncoder(nn.Module):
    """
    Encodes events sequences into hidden_states
    """
    def __init__(self, 
        model_path: ModelPath,
        max_seq_len: int,
        use_flat_flash_attention: bool = True,
        num_add_tokens: int = 2
    ):
        super(EventEncoder, self).__init__()
        # create device
        if dist.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f'cuda:{local_rank}')
        else:
            local_rank = 0
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.local_rank = local_rank
        self.device = device
        # load tokenizer and llm
        self.max_seq_len = max_seq_len
        self.num_add_tokens = num_add_tokens
        hf_config = AutoConfig.from_pretrained(model_path.value, trust_remote_code=True)
        hf_config.use_cache = False
        hf_config.return_dict = True
        hf_config.use_ft_flash_attn = use_flat_flash_attention
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.value, trust_remote_code=True)
        self.create_pretrained_model(model_path, hf_config)

    def create_pretrained_model(self, model_path: ModelPath, hf_config: Any):
        """
        Create a pretrained model from the given model path.
        """
        print(f"Loading {model_path.name} Event model...")
        if model_path.name.startswith("Qwen3"):
            self.llm = CustomQwen3ForCausalLM.from_pretrained(
                model_path.value,
                config=hf_config,
                torch_dtype='auto', device_map=self.device
            ).to(self.device)
        elif model_path.name.startswith("TinyLlama"):
            self.llm = CustomLlamaForCausalLM.from_pretrained(
                model_path.value,
                config=hf_config,
                torch_dtype='auto', device_map=self.device
            ).to(self.device)
        # add new embed_tokens to the llm
        self.llm.resize_token_embeddings(len(self.tokenizer) + self.num_add_tokens, mean_resizing=True)

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