import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default='encoder_only', metadata={"help": "The type of the model, including encoder-only, decoder-only, encoder-decoder"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    corpus_data: str = field(
        default=None, metadata={"help": "Path to corpus data"}
    )
    eval_data: str = field(
        default=None, metadata={"help": "Path to eval data"}
    )
    used_premise: str = field(
        default=None, metadata={"help": "Path to used premise data"}
    )
    module_premise: str = field(
        default=None, metadata={"help": "Path to module mapping premise data"}
    )
    shuffl_prob: float = field(
        default=0.75,
        metadata={
            "help": "The shuffle ratio of data transform"
        },
    )
    remove_prob: float = field(
        default=0.75,
        metadata={
            "help": "The remove ratio of data transform"
        },
    )
    rerank_num: int = field(
        default=20,
        metadata={
            "help": "The number of rerank premise"
        },
    )
    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
