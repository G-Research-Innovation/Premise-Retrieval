
# Lean State Premise Retrieval

This project focuses on using **state** in the Lean language for premise retrieval.

## Environment Set up

Set up the enviroment by running:

```bash
conda create -n pytorch python=3.10
pip install -r requirements.txt
```

## Dataset

First, generate the raw dataset using our script, which can be downloaded from [this link](https://huggingface.co/datasets/ruc-ai4math/mathlib_handler_benchmark_410).

Next, preprocess the dataset. You can generate the dataset required for our method by running the following script:

```bash
python process_data/generate_our_from_raw.py --config_path config/random.toml
python process_data/generate_pretrain_tokenizer_data.py --config_path config/random.toml
```

## Training the Tokenizer

Next, train the Lean-specific tokenizer:

```bash
python pretrain/pretrain_tokenizer.py --config_path config/random.toml
```

## Training the Retrieval Model

### Pretraining

Now, pretrain the Retrieval model:

```bash
torchrun --nproc_per_node=8 pretrain/pretrain.py --config_path config/random.toml
```

### Fine-tuning

After pretraining, fine-tune the model using the script below:

```bash
torchrun --nproc_per_node=8 finetune/run.py --config_path config/random.toml
```

## Training the Rerank Model

### Data Generation for Rerank Model

To generate the data for training the rerank model, first, you need to modify the `eval_file` under the `[eval]` section in the `config.toml` file to the path where `train_expand_premise.jsonl` is located. Also, change the `save_output_path` to `train_expand_premise_for_rerank_s0_d0.jsonl`. Then, run the following script:

```bash
python test_finetune/generate_hard_negative.py --config_path config/random.toml
```

### Pretraining

Pretrain the rerank model with this script:

```bash
torchrun --nproc_per_node=8 pretrain/pretrain.py --config_path config/random_1024.toml
```

### Fine-tuning

Fine-tune the pretrained rerank model by running:

```bash
torchrun --nproc_per_node=8 rerank/run.py --config_path config/random_1024.toml
```

## Testing

You can download the trained models from [this link](https://huggingface.co/ruc-ai4math/Lean_State_Search_Random) and place them in the respective directories.

### Testing the Retrieval Model

To evaluate the Retrieval model, you need to modify the `eval_file` under the `[eval]` section in the `config.toml` file to the path where `test_increase.jsonl` is located. Also, change the `save_output_path` to `retrieval_output_test_s0_d0.jsonl`. Then, run the following script:

```bash
python test_finetune/eval_increase_test.py --config_path config/random.toml
```

### Testing the Rerank Model

To evaluate the Rerank model, run:

```bash
python test_finetune/test_rerank.py --config_path config/random.toml
```

