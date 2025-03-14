from datasets import load_dataset
from huggingface_hub import snapshot_download

repo = 'ruc-ai4math/mathlib_handler_benchmark_410'
snapshot_download(repo_id=repo, repo_type='dataset', local_dir='mathlib_handler_benchmark_410')