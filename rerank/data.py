import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from arguments import DataArguments
import json
import re


class data_transform(object):
    def __init__(self, shuffle_prob, remove_prob):
        self.shuffle_prob = shuffle_prob
        self.remove_prob = remove_prob

    def __call__(self, sample):
        augmented_list = sample.copy()
        if random.random() < self.shuffle_prob:
            random.shuffle(augmented_list)
        if len(sample) >= 15 and random.random() < self.remove_prob:
            augmented_list = [elem for elem in augmented_list if random.random() > 0.2]
        return augmented_list


class TrainDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            is_transform,
            transformation,
            model_type: str = 'encoder_only',
            seed=42,
            module_premise_path=""

    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')
            self.dataset = self.dataset.shuffle(seed=seed)
            self.tokenizer = tokenizer
            self.args = args
            self.model_type = model_type
            self.is_transform = is_transform
            self.transformation = transformation
            self.total_len = len(self.dataset)
            self._load_corpus()
            self.module_id_dict = self.get_module_premise(module_premise_path=module_premise_path)
            self.corpus_total_len = len(self.corpus)
            self.all_corpus_ids = range(self.corpus_total_len)
            
    def get_module_premise(self, module_premise_path):

        with open(module_premise_path, "r", encoding="utf-8") as f:
            module_id_dict = json.load(f)
        return module_id_dict

    def _load_corpus(self):
        corpus_path = self.args.corpus_data
        self.corpus = {}
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                temp_dict = {}
                data = json.loads(line)
                context = data['premise']['state']['context']
                goal = data['premise']['state']['goal']
                context = self.process_strings(context)
                goal = re.sub(r'\n\s+', ' ', goal)
                temp_dict['context'] = context
                temp_dict['goal'] = "<GOAL>" + goal
                temp_dict['state'] = temp_dict['context'] + temp_dict['goal']
                temp_dict['module'] = data['def_path']
                self.corpus[data['id']] = temp_dict

    def process_strings(self, string_list):
        processed_list = []
        for s in string_list:
            s = "<VAR>" + s
            s = re.sub(r'\n\s+', ' ', s)
            processed_list.append(s)
        result = ''.join(processed_list)
        return result

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        string_combine = qry_encoding + "[SEP]" + doc_encoding
        item = self.tokenizer(
            string_combine,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query_dict = self.dataset[item]['query']
        if self.is_transform:
            query_context = self.transformation(query_dict['context'])
        else:
            query_context = query_dict['context']
        query_goal = query_dict['goal']
        context = self.process_strings(query_context)
        goal = re.sub(r'\n\s+', ' ', query_goal)
        query = context + '<GOAL>' + goal
        if self.model_type == 'encoder_only':
            query = query
        else:
            query = query + ' <|endoftext|>'

        pos = self.dataset[item]['ground_truths']
        all_neg = self.dataset[item]['neg_premises']
        negs = random.sample(all_neg, (self.args.train_group_size - 1))
        pos_str = self.corpus[pos]['state']
        batch_data = []
        batch_data.append(self.create_one_example(query, pos_str))
        for neg in negs:
            batch_data.append(self.create_one_example(query, self.corpus[neg]['state']))

        return batch_data


@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
