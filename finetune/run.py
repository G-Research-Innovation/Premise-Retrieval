import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import wandb
from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import TrainDatasetForEmbedding, EmbedCollator, data_transform
from modeling import BiEncoderModel
from trainer import BiTrainer
import argparse
import toml
import transformers
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def main():
    parser_config = argparse.ArgumentParser()
    parser_config.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args_config = parser_config.parse_args()
    config = toml.load(args_config.config_path)
    shuffl_prob = config['dataset']['shuffle_ratio']
    remove_prob = config['dataset']['random_delete_ratio']
    wandb.init(project="Leandojo_6num_finetune", name='finetune_' + config['config_name'])

    args = []
    for key, value in config['finetune'].items():
        args.append(f"--{key}")
        args.append(str(value))

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    print(training_args)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    if model_args.model_type == 'encoder_only' or model_args.model_type == 'no_pretrain':
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            padding_side="right",
            use_fast=True,
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           normlized=training_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           use_inbatch_neg=training_args.use_inbatch_neg,
                           model_type=model_args.model_type,
                           use_lora=model_args.use_lora,
                           target_module=model_args.target_module,
                           lora_rank=model_args.lora_rank,
                           bili = model_args.bili
                           )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model.model,
        )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    transfor = data_transform(shuffle_prob=shuffl_prob, remove_prob=remove_prob)
    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer, model_type=model_args.model_type,
                                             is_transform=True, transformation=transfor, module_premise_path=data_args.module_premise, train_data_percentage=data_args.train_data_percentage)
    import torch.distributed as dist
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1



    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
        ),
        tokenizer=tokenizer,
        use_lora=model_args.use_lora,
        orig_dir=model_args.model_name_or_path,
        model_type=model_args.model_type,
        eval_dataset_train=data_args.eval_dataset,
        used_premise=data_args.used_premise,
        bili = model_args.bili,
        device_num = training_args.device_num
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    print(train_dataset.total_len // (training_args.per_device_train_batch_size * world_size))
    print(train_dataset.total_len)
    print(world_size)

    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
