import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, T5EncoderModel
from transformers.file_utils import ModelOutput
from peft import LoraConfig, get_peft_model
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, BertModel

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_c_reps: Optional[Tensor] = None
    p_g_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


def initialize_model(vocab_size=30522, hidden_size=768, num_hidden_layers=24, num_attention_heads=12,
                     intermediate_size=3072):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )
    model = BertModel(config)
    return model


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 model_type: str = 'encoder_only',
                 use_lora: bool = False,
                 target_module=None,
                 lora_rank=32,
                 bili=0.5
                 ):
        super().__init__()
        if target_module is None:
            target_module = ['q']
        if model_type == 'byt5':
            self.model = T5EncoderModel.from_pretrained(model_name)
        elif model_type == 'no_pretrain':
            print('initial model')
            self.model = initialize_model(num_hidden_layers=6)
        elif model_type == 'no_pretrain_no_token':
            self.model = initialize_model(vocab_size=28996, num_hidden_layers=6)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        if use_lora:
            model = AutoModel.from_pretrained(model_name, device_map='auto')

            for name, module in model.named_modules():
                print(name)

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=4,
                init_lora_weights=True,
                target_modules=['self_attn'],
                lora_dropout=0,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model_type = model_type
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.bili = bili
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError(
                    "Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def _encode(self, features):

        hidden_states = self.model(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                   return_dict=True).last_hidden_state
        lens = features['attention_mask'].sum(dim=1)
        p_reps = (hidden_states * features['attention_mask'].unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
        return torch.nn.functional.normalize(p_reps, dim=1)

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage_context: Dict[str, Tensor] = None,
                passage_goal: Dict[str, Tensor] = None, input: Dict[str, Tensor] = None):
        if input is None:
            if self.model_type == 'encoder_only':
                q_reps = self.encode(query)
                p_c_reps = self.encode(passage_context)
                p_g_reps = self.encode(passage_goal)
            else:
                q_reps = self._encode(query)
                p_c_reps = self._encode(passage_context)
                p_g_reps = self._encode(passage_goal)

            if self.training:
                if self.negatives_cross_device and self.use_inbatch_neg:
                    q_reps = self._dist_gather_tensor(q_reps)
                    p_c_reps = self._dist_gather_tensor(p_c_reps)
                    p_g_reps = self._dist_gather_tensor(p_g_reps)

                group_size = p_c_reps.size(0) // q_reps.size(0)
                if self.use_inbatch_neg:
                    scores = (self.compute_similarity(q_reps, p_c_reps)*self.bili + self.compute_similarity(q_reps,
                                                                                                  p_g_reps)*(1-self.bili)) / self.temperature  # B B*G

                    scores = scores.view(q_reps.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * group_size
                    loss = self.compute_loss(scores, target)
                else:
                    scores = (self.compute_similarity(q_reps[:, None, :, ],
                                                                          p_c_reps.view(q_reps.size(0), group_size, -1)).squeeze(
                                            1) * self.bili + self.compute_similarity(q_reps[:, None, :, ],
                                                                         p_g_reps.view(q_reps.size(0), group_size, -1)).squeeze(
                                            1) * (1-self.bili)) / self.temperature  # B G

                    scores = scores.view(q_reps.size(0), -1)
                    target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                    loss = self.compute_loss(scores, target)

            else:
                scores = (self.compute_similarity(q_reps, p_c_reps)*self.bili + self.compute_similarity(q_reps,
                                                                                              p_g_reps)*(1-self.bili))
                loss = None
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_c_reps=p_c_reps,
                p_g_reps=p_g_reps
            )
        else:
            with torch.no_grad():
                last_hidden_state = self.model(**input, return_dict=True).last_hidden_state
                return last_hidden_state

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
