import datasets
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from dataclasses import field
from transformers.modeling_utils import *
from transformers import TrainingArguments
import transformers
import toml
import argparse
import wandb
from transformers import set_seed
from torch.utils.data import Dataset
import random
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


class MyDataset(Dataset):
    def __init__(self, data_file, is_train, transform, tokenizer):
        self.data = datasets.load_dataset('json', data_files=data_file, split='train')
        self.is_train = is_train
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def process_strings(self, string_list):
        processed_list = []
        for s in string_list:
            s = "<VAR>" + s
            s = re.sub(r'\n\s+', ' ', s)
            processed_list.append(s)
        result = ''.join(processed_list)
        return result

    def __getitem__(self, item):
        context = self.data[item]['state']['context']
        goal = self.data[item]['state']['goal']

        if self.is_train:
            context = self.transform(context)
        context = self.process_strings(context)
        goal = re.sub(r'\n\s+', ' ', goal)
        combine = context + '<GOAL>' + goal
        input = self.tokenizer(combine, truncation=True, padding='max_length', max_length=512,
                               return_special_tokens_mask=True)
        return input


def prepare_dataset_new(tokenizer, train_file, eval_file, shuffle_prob, remove_prob):
    transform = data_transform(shuffle_prob=shuffle_prob, remove_prob=remove_prob)
    train_dataset = MyDataset(data_file=train_file, is_train=True, transform=transform, tokenizer=tokenizer)
    eval_dataset = MyDataset(data_file=eval_file, is_train=False, transform=transform, tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    return data_module


def prepare_dataset(tokenizer, data_file, test_size=0.2, valid_size=0.1, seed=42):
    dataset = load_dataset('text', data_files=data_file, split='train')
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512,
                         return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    train_test_split = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
    train_valid_split = train_test_split['train'].train_test_split(test_size=valid_size, seed=seed)

    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )
    return data_module


def initialize_model(vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12,
                     intermediate_size=3072, max_position_embeddings=512):

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings

    )
    model = BertForMaskedLM(config)
    return model


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}, )
    train_file_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_file_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})


# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = toml.load(config_path)
    set_seed(42)
    wandb.init(project="Leandojo_Hyper_6num_Pretrain", name='pretrain_' + config['config_name'])
    pretrain_config = config.get('pretrain', {})
    script_args = TrainingArguments(**pretrain_config)
    print("Output Directory:", script_args.output_dir)

    tokenizer_dir = config['tokenizer']['save_dir']
    tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_dir, 'vocab.txt'), do_lower_case=False)
    special_tokens = ["<VAR>", "<GOAL>"]
    tokenizer.add_tokens(special_tokens)
    vocab_size = len(tokenizer)
    data_module = prepare_dataset_new(tokenizer=tokenizer, train_file=config['pretrain']['train_file_path'],
                                      eval_file=config['pretrain']['eval_file_path'],
                                      shuffle_prob=0,
                                      remove_prob=0
                                      )

    model = initialize_model(vocab_size=vocab_size, hidden_size=config['model']['hidden_size'],
                             num_hidden_layers=config['model']['num_hidden_layers'],
                             num_attention_heads=config['model']['num_attention_heads'],
                             intermediate_size=config['model']['intermediate_size'],
                             max_position_embeddings=config['model']['max_position_embeddings'])
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, 'ft'))


if __name__ == '__main__':
    main()
