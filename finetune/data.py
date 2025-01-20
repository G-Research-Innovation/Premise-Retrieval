import os.path
import random
from dataclasses import dataclass
import re
import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from arguments import DataArguments
import json


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


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            is_transform,
            transformation,
            model_type: str = 'encoder_only',
            seed=42,
            module_premise_path = "",
            train_data_percentage: float = 1.0
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')
        self.dataset = self.dataset.shuffle(seed=seed)
        if train_data_percentage < 1.0:
            total_examples = len(self.dataset)
            num_samples_to_select = int(total_examples * train_data_percentage)
            selected_indices = random.sample(range(total_examples), num_samples_to_select)
            self.dataset = self.dataset.select(selected_indices)

        self.tokenizer = tokenizer
        self.args = args
        self.model_type = model_type
        self.is_transform = is_transform
        self.transformation = transformation
        self.total_len = len(self.dataset)
        self._load_corpus()
        self.module_id_dict = self.get_module_premise(module_premise_path=module_premise_path)
        self.corpus_total_len = len(self.corpus)
        self.all_corpus_ids = set(range(self.corpus_total_len))

    def process_strings(self, string_list):
        processed_list = []
        for s in string_list:
            s = "<VAR>" + s
            s = re.sub(r'\n\s+', ' ', s)
            processed_list.append(s)
        result = ''.join(processed_list)
        return result

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
                temp_dict['goal'] = "<GOAL>"+goal
                temp_dict['module']=data['def_path']
                self.corpus[data['id']] = temp_dict

    def split_context_goal(self, original):
        temp_dict = {}
        if self.model_type == 'encoder_only':
            temp_dict['context'] = original.split('<GOAL>')[0]
            temp_dict['goal'] = original.split('<GOAL>')[1]
        else:
            temp_dict['context'] = original.split('<GOAL>')[0] + ' <|endoftext|>'
            temp_dict['goal'] = original.split('<GOAL>')[1] + ' <|endoftext|>'
        return temp_dict

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query_dict = self.dataset[item]['state']

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
        passages = []

        pos = self.dataset[item]['premise']
        passages.append(pos)
        '''
        pos_module = self.corpus[pos]['module']
        query_module_id.extend(self.module_id_dict[pos_module])
        '''
        flag = True
        while flag:
            negs = random.sample(self.all_corpus_ids, self.args.train_group_size - 1)
            if not set(negs) & set([pos]):
                flag = False

        passages.extend(negs)
        if self.model_type == 'encoder_only':
            passages_context = [self.corpus[p]['context'] for p in passages]
            passages_goal = [self.corpus[p]['goal'] for p in passages]
        else:
            passages_context = [self.corpus[p]['context'] + ' <|endoftext|>' for p in passages]
            passages_goal = [self.corpus[p]['goal'] + ' <|endoftext|>' for p in passages]

        return query, passages_context, passages_goal


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage_context = [f[1] for f in features]
        passage_goal = [f[2] for f in features]
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage_context[0], list):
            passage_context = sum(passage_context, [])
        if isinstance(passage_goal[0], list):
            passage_goal = sum(passage_goal, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_c_collated = self.tokenizer(
            passage_context,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        d_g_collated = self.tokenizer(
            passage_goal,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage_context": d_c_collated, "passage_goal": d_g_collated}
