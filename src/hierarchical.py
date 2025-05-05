import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from .common import all_gather, create_device_info, ClassificationHead
from .encoder_user import UserEncoder
from .encoder_event import EventEncoder
import os


@dataclass
class HierarchicalModelOutput:
    hidden_states: torch.Tensor
    user_embedding: torch.Tensor = None
    logits: torch.Tensor    = None
    nce_loss: torch.Tensor  = None
    ce_loss: torch.Tensor   = None
    loss: torch.Tensor = None


class HierarchicalModel(nn.Module):
    def __init__(self,
        event_encoder: EventEncoder,
        user_encoder: UserEncoder,
        temperature: float = 0.05,
        nce_threshold: float = 0.99,
        num_classes: int = None,
    ):
        super(HierarchicalModel, self).__init__()
        self.local_rank, self.device = create_device_info()
        
        # load pretrained llm
        self.event_encoder = event_encoder
        self.user_encoder = user_encoder

        # add generative NCE loss
                # add generative NCE loss
        self.nce_loss_func = GenerativeInfoNCELoss(
            temperature=temperature,
            nce_threshold=nce_threshold
        )
        # add classifier head
        if num_classes is not None:
            self.classifier = ClassificationHead(
                num_hiddens=self.user_encoder.llm.config.hidden_size, 
                num_classes=num_classes,
                activation='relu'
            )
            self.classifier.to(self.device)
        else:
            self.classifier = None


    def forward(self,
        pos_input_ids: torch.Tensor,
        pos_position_ids: torch.Tensor,
        pos_varlen: torch.Tensor,
        attention_mask: torch.Tensor,
        time_ids: torch.Tensor=None,
        neg_input_ids: torch.Tensor=None,
        neg_position_ids: torch.Tensor=None,
        neg_varlen: torch.Tensor=None,
        add_user_token: bool = True,
        labels: torch.Tensor=None,
        **kwargs
    ):
        """
        encode user inputs into hidden_states

        Parameters
        ----------
        pos_input_ids: (num_pos, )
            The input ids for the positive event sequence.
        pos_position_ids: (seq_len, )
            The position ids for the positive event sequence.
        pos_varlen: (num_pos, )
            The variable length of the positive event sequence.
        attention_mask: (batch, seq_len)
            The attention mask for the inputs to UserEncoder.
        time_ids: (batch, seq_len, num_time_loc)
            The time ids for the inputs to UserEncoder.
        neg_input_ids: (num_neg, )
            The input ids for the negative event sequence.
        neg_position_ids: (seq_len, )
            The position ids for the negative event sequence.
        neg_varlen: (num_neg, )
            The variable length of the negative event sequence.
        add_user_token: bool
            Whether to add user token to the input embeddings. Default is True.
        labels: (batch, )
            The labels for the classification task. Default is None.
        """
        pos_hidden_states = self.encode_event(input_ids=pos_input_ids,
                                              position_ids=pos_position_ids,
                                              seq_varlen=pos_varlen)
        if neg_input_ids is not None:
            neg_hidden_states = self.encode_event(input_ids=neg_input_ids,
                                                  position_ids=neg_position_ids,
                                                  seq_varlen=neg_varlen)
        else:
            neg_hidden_states = None

        predictions, user_embedding = self.user_encoder(
            event_embeddings=pos_hidden_states,
            attention_mask=attention_mask,
            time_ids=time_ids,
            add_user_token=add_user_token
        )
   
        # add classifier head
        if self.classifier is not None:
            logits = self.classifier(user_embedding)
        else:
            logits = None
        
        if logits is not None and labels is not None:
            if self.classifier.num_classes == 1:
                ce_loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).float())
            else:
                ce_loss = F.cross_entropy(
                    logits.view(-1, self.classifier.num_classes), labels.view(-1).long())
        else:
            ce_loss = 0.0

        # calculate the loss
        if neg_hidden_states is not None:
            nce_loss = self.nce_loss_func(
                predictions=predictions, 
                positives=pos_hidden_states, 
                negatives=neg_hidden_states,
                attention_mask=attention_mask
            )
        else:
            nce_loss = 0.0
        
        # return the output
        return HierarchicalModelOutput(
            hidden_states=predictions,
            user_embedding=user_embedding,
            logits=logits,
            nce_loss=nce_loss,
            ce_loss=ce_loss
        )
    
    def encode_event(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_varlen: torch.Tensor
    ):
        """
        encode event inputs into hidden_states
        """
        return self.event_encoder(
            input_ids=input_ids, position_ids=position_ids, seq_varlen=seq_varlen
        )
    
    def build_optimizer(self, 
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8
    ):
        """
        build optimizer
        """
        param_groups = self.build_param_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(param_groups, lr=learning_rate, eps=adam_epsilon)
    
    def build_param_groups(self, weight_decay: float = 0.01):
        encoder_with_wd, encoder_without_wd = [], []
        classifier_with_wd, classifier_without_wd = [], []

        for name, param in self.named_parameters():
            if 'classifier' in name:
                if param.ndim >= 2:
                    classifier_with_wd.append(param)
                else:
                    classifier_without_wd.append(param)
            else:
                if param.ndim >= 2:
                    encoder_with_wd.append(param)
                else:
                    encoder_without_wd.append(param)

        grouped_parameters = [
            {'name': 'encoder.with_wd',      'params': encoder_with_wd,      'weight_decay': weight_decay},
            {'name': 'encoder.without_wd',   'params': encoder_without_wd,   'weight_decay': 0.0},
            {'name': 'classifier.with_wd',   'params': classifier_with_wd,   'weight_decay': weight_decay},
            {'name': 'classifier.without_wd','params': classifier_without_wd,'weight_decay': 0.0}
        ]
        print(f"Prams of encoder    use weight-decay: {sum([p.numel() for p in encoder_with_wd])}")
        print(f"Prams of encoder    not use weight-decay: {sum([p.numel() for p in encoder_without_wd])}")
        print(f"Prams of classifier use weight-decay: {sum([p.numel() for p in classifier_with_wd])}")
        print(f"Prams of classifier not use weight-decay: {sum([p.numel() for p in classifier_without_wd])}")

        return grouped_parameters


    def save_pretrained(self, save_path: str):
        """
        Save the pretrained model to the given path.
        """
        self.event_encoder.save_pretrained(os.path.join(save_path, "event_encoder"))
        self.user_encoder.save_pretrained(os.path.join(save_path, "user_encoder"))
        if self.classifier is not None:
            torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
        # generate config.json
        config = {
            "model_type": "HierarchicalModel",
            "temperature": self.nce_loss_func.temperature.item(),
            "nce_threshold": self.nce_loss_func.nce_threshold,
            "num_classes": self.classifier.num_classes if self.classifier is not None else None
        }
        torch.save(config, os.path.join(save_path, "config.json"))
    

    def from_pretrained(self, model_path: str):
        """
        Load the pretrained model from the given path.
        """
        self.event_encoder.from_pretrained(os.path.join(model_path, "event_encoder"))
        self.user_encoder.from_pretrained(os.path.join(model_path, "user_encoder"))
        if self.classifier is not None:
            self.classifier.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        # load config.json
        config = torch.load(os.path.join(model_path, "config.json"))
        self.nce_loss_func.temperature.data = torch.tensor(config["temperature"])
        self.nce_loss_func.nce_threshold = config["nce_threshold"]
    

class GenerativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05, nce_threshold=0.99):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.nce_threshold = nce_threshold
        
    def forward(self, 
        predictions: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Parameters
        ----------
        predictions: (batch, seq_len, hidden)
            The outputs generated by the UserEncoder, are the next-event embeddings predictions.
        positives: (batch, seq_len, hidden)
            The positive event embeddings, generated by the EventEncoder, are the inputs of UserEncoder.
        negatives: (batch, num_neg, hidden)
            The sampled negative event embeddings, generated by the EventEncoder.
        attention_mask: (batch, seq_len)
            The attention mask for the inputs to UserEncoder.
        """
        _, _, hiddens = predictions.shape
        
        # postive samples
        predictions = predictions[:, :-1, :]  # (batch, seq_len-1, hidden)
        positives = positives[:, 1:, :]       # (batch, seq_len-1, hidden)
        
        with torch.no_grad():
            self.temperature.clamp_(0, np.log(100.0))
        temperature = self.temperature.exp()

        predictions = predictions / predictions.norm(dim=-1, keepdim=True)
        positives = positives / positives.norm(dim=-1, keepdim=True)
        negatives = negatives / negatives.norm(dim=-1, keepdim=True)

        # calculate the positive and negative scores
        # (batch, seq_len-1, 1)
        pos_scores = F.cosine_similarity(predictions, positives, dim=-1).unsqueeze(-1)

        # gather all negative samples from other devices
        if dist.is_initialized():
            negatives_all = all_gather(negatives, sync_grads=True) # (num_neg, hidden)
            negatives_all = negatives_all.reshape(-1, hiddens).transpose(-1, -2) # (hidden, num_neg)
        else:
            negatives_all = negatives.reshape(-1, hiddens).transpose(-1, -2) # (hidden, num_neg)
        
        neg_scores = torch.matmul(predictions, negatives_all) # (batch, seq_len-1, num_neg)
        # mask scores if the negative is similar to the positive
        mask = torch.matmul(positives, negatives_all) > self.nce_threshold
        neg_scores[mask] = torch.finfo(neg_scores.dtype).min

        # calculate the loss
        logits = torch.cat([pos_scores, neg_scores], dim=-1) # (batch, seq_len-1, num_neg + 1)
        logits = logits[attention_mask[:, :-1].bool()] * temperature # (batch * (seq_len - 1), num_neg + 1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss