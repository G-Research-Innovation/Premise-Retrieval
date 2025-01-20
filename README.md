
# Lean State Premise Retrieval

This project focuses on using **state** in the Lean language for premise retrieval.

## Environment Set up

Set up the enviroment by running:

```bash
conda create -n pytorch python=3.10
pip install -r requirements.txt
```

## Dataset

First, download our dataset from [this link](https://huggingface.co/datasets/ruc-ai4math/mathlib_handler_benchmark_410).

Next, preprocess the dataset. You can generate the dataset required for our method by running the following script.

You can skip the preprocess step and directly use the dataset from the download.

```bash
python preprocess/generate_our_from_raw.py --config_path config/random.toml
python preprocess/generate_pretrain_tokenizer_data.py --config_path config/random.toml
```

## Training the Tokenizer

Train the Lean-specific tokenizer by running the following script:

```bash
python pretrain/pretrain_tokenizer.py --config_path config/random.toml
```

## Training the Retrieval Model

### Pretraining

To pretrain the Retrieval model, you can run the following script:

```bash
torchrun --nproc_per_node=NUM_GPUS pretrain/pretrain.py --config_path config/random.toml
```

### Fine-tuning

After pretraining, fine-tune the model using the script below. 
You also need to adjust the `device_num` parameter under the `[finetune]` section in the `config/random.toml` file according to the number of GPUs you are using. 

```bash
torchrun --nproc_per_node=NUM_GPUS finetune/run.py --config_path config/random.toml
```

## Training the Rerank Model

### Data Generation for Rerank Model

To generate the data for training the rerank model, first, you need to modify the `eval_file` under the `[eval]` section in the `config.toml` file to the path where `train_expand_premise.jsonl` is located. Also, change the `save_output_path` to `train_expand_premise_for_rerank_s0_d0.jsonl`. Then, run the following script:

You can skip this step and directly use the dataset we have generated.

```bash
python test_finetune/generate_hard_negative.py --config_path config/random.toml
```

### Pretraining

Pretrain the rerank model with this script:

```bash
torchrun --nproc_per_node=NUM_GPUS pretrain/pretrain.py --config_path config/random_1024.toml
```

### Fine-tuning

Fine-tune the pretrained rerank model by running:

```bash
torchrun --nproc_per_node=NUM_GPUS rerank/run.py --config_path config/random_1024.toml
```

When using distributed training, if you encounter communication issues between different GPUs when saving the model at the end of training. 
You can try to modify some codes in `transformers.trainer.py` file, specifically, replace the following code in `_inner_training_loop` function
```python
if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    # Wait for everyone to get here so we are sure the model has been saved by process 0.
    if is_torch_xla_available():
        xm.rendezvous("load_best_model_at_end")
    elif args.parallel_mode == ParallelMode.DISTRIBUTED:
        dist.barrier()
    elif is_sagemaker_mp_enabled():
        smp.barrier()
```
with this modified version:

```python
if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    # Wait for everyone to get here so we are sure the model has been saved by process 0.
    if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        self._load_best_model()
```
## Testing

You can download the trained models from [this link](https://huggingface.co/ruc-ai4math/Lean_State_Search_Random) and place them in the respective directories.

### Testing the Retrieval Model

To evaluate the Retrieval model, you need to modify the `eval_file` under the `[eval]` section in the `config/random.toml` file to the path where `test_increase.jsonl` is located. Also, change the `save_output_path` to `retrieval_output_test_s0_d0.jsonl`. Then, run the following script:

```bash
python test_finetune/eval_increase_test.py --config_path config/random.toml
```

### Testing the Rerank Model

To evaluate the Rerank model, run:

```bash
python test_finetune/test_rerank.py --config_path config/random.toml
```

