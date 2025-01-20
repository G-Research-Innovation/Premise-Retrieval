import logging
import os
import json
import re
import numpy as np
from transformers import AutoTokenizer
from typing import Optional
from tqdm import tqdm
import torch
from transformers.trainer import Trainer

import torch.distributed as dist
from modeling import CrossEncoder

logger = logging.getLogger(__name__)


def process_strings(string_list):
    processed_list = []
    for s in string_list:
        s = "<VAR>" + s
        s = re.sub(r'\n\s+', ' ', s)
        processed_list.append(s)
    result = ''.join(processed_list)
    return result


def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_dict = {}
            data = json.loads(line)
            context = data['premise']['state']['context']
            goal = data['premise']['state']['goal']
            context = process_strings(context)
            goal = re.sub(r'\n\s+', ' ', goal)
            temp_dict['context'] = context
            temp_dict['goal'] = "<GOAL>" + goal
            temp_dict['state'] = temp_dict['context'] + temp_dict['goal']
            corpus[data['id']] = temp_dict
    return corpus


def get_query_pair(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


class CETrainer(Trainer):
    def __init__(self, *args1, model_path='',
                 corpus_path="",
                 eval_path="",rerank_num=20, **kwargs):
        super(CETrainer, self).__init__(*args1, **kwargs)
        self.tokenizer_for_test = AutoTokenizer.from_pretrained(model_path)
        self.corpus = load_corpus(corpus_path=corpus_path)
        self.query_data = get_query_pair(eval_path)
        self.rerank_num = rerank_num

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs, num_items_in_batch=None):
        return model(inputs)['loss']

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            self.model.eval()

            R1 = []
            R5 = []
            R10 = []
            for item in tqdm(range(len(self.query_data)), desc="Processing query data"):
                query_all = self.query_data[item]['query']
                pred_premise = self.query_data[item]['pre_premise']
                ground_truth = self.query_data[item]['ground_truths']
                logits_all = None
                for query_dict in query_all:
                    query_goal = query_dict['goal']
                    query_context = query_dict['context']
                    context = process_strings(query_context)
                    goal = re.sub(r'\n\s+', ' ', query_goal)
                    query = context + '<GOAL>' + goal
                    pairs = []
                    count=0
                    for pre_pre in pred_premise:
                        temp_list = []
                        pre_pre_state = self.corpus[pre_pre]['state']
                        temp_list.append(query)
                        temp_list.append(pre_pre_state)
                        pairs.append(temp_list)
                        count = count +1
                        if count == self.rerank_num:
                            break
                    with torch.no_grad():
                        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
                        inputs = inputs.to('cuda')
                        scores = self.model.hf_model(**inputs, return_dict=True).logits.view(-1, ).float()
                        if logits_all is None:
                            logits_all = scores
                        else:
                            logits_all += scores

                probabilities_all = torch.nn.functional.softmax(logits_all, dim=0)
                sorted_indices = torch.argsort(probabilities_all, descending=True)
                sorted_pre_pre = [(pred_premise[idx], probabilities_all[idx].item()) for idx in sorted_indices]
                rerank_pres = []
                for pre_pre, score in sorted_pre_pre:
                    rerank_pres.append(pre_pre)
                label = set(ground_truth)
                TP1 = rerank_pres[0] in label
                R1.append(float(TP1) / min(1, len(label)))
                TP5 = len(label.intersection(rerank_pres[:5]))
                R5.append(float(TP5) / min(5, len(label)))
                TP10 = len(label.intersection(rerank_pres[:10]))
                R10.append(float(TP10) / min(10, len(label)))
            R1 = 100 * np.mean(R1)
            R5 = 100 * np.mean(R5)
            R10 = 100 * np.mean(R10)
            self.log({f"{metric_key_prefix}_R1": R1})
            self.log({f"{metric_key_prefix}_R5": R5})
            self.log({f"{metric_key_prefix}_R10": R10})
            temp_dict = {}
            temp_dict['eval_Recall@10'] = R10
            return temp_dict
