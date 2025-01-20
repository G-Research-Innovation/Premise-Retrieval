import logging
import datasets
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from flag_model import FlagModel
import json
import toml
import wandb
import argparse
import re
import os

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )

    corpus_data: str = field(
        default="",
        metadata={'help': 'candidate passages'}
    )
    query_data: str = field(
        default="",
        metadata={'help': 'queries and their positive passages for evaluation'}
    )

    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=256,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=True,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path_context: str = field(
        default="embeddings_context.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

    save_path_goal: str = field(
        default="embeddings_goal.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

    model_type: str = field(
        default="encoder_only",
        metadata={'help': 'Type of Model'}
    )

    eval_file: str = field(
        default="",
        metadata={'help': 'Path of eval file'}
    )

    corpus_file: str = field(
        default="",
        metadata={'help': 'Path of corpus file'}
    )
    save_output_path: str = field(
        default="",
        metadata={'help': 'Path of corpus file'}
    )


def index(model: FlagModel, corpus, batch_size: int = 256, max_length: int = 512,
          save_path: str = None, save_embedding: bool = False,
          load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)

    else:
        corpus_embeddings = model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]

        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings

    corpus_embeddings = corpus_embeddings.astype(np.float32)
    return corpus_embeddings


def search(model: FlagModel, queries: datasets, corpus_embeddings_context, corpus_embeddings_goal, k: int = 1024,
           batch_size: int = 256,
           max_length: int = 512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query = []

    for item in queries:
        context = item['state']['context']
        goal = item['state']['goal']
        context = process_strings(context)
        goal = re.sub(r'\n\s+', ' ', goal)
        combine = context + '<GOAL>' + goal
        query.append(combine)

    query_embeddings_e = model.encode_queries(query, batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings_e)
    print(query_size)
    all_scores = []
    all_indices = []
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embeddings = query_embeddings_e[i: j]
        similarities_context = np.dot(query_embeddings, corpus_embeddings_context.T)
        similarities_goal = np.dot(query_embeddings, corpus_embeddings_goal.T)
        similarities = (similarities_context + similarities_goal) / 2
        indices = np.argsort(similarities, axis=1)[:, ::-1][:, :k]
        scores = np.take_along_axis(similarities, indices, axis=1)

        all_scores.append(scores)
        all_indices.append(indices)

    all_scores1 = np.concatenate(all_scores, axis=0)
    all_indices1 = np.concatenate(all_indices, axis=0)

    return all_scores1, all_indices1


def process_strings(string_list):
    processed_list = []
    for s in string_list:
        s = "<VAR>" + s
        s = re.sub(r'\n\s+', ' ', s)
        processed_list.append(s)
    result = ''.join(processed_list)
    return result


def main():
    parser_config = argparse.ArgumentParser()
    parser_config.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args_config = parser_config.parse_args()
    config = toml.load(args_config.config_path)
    wandb.init(project="Leandojo_Lean_Sate_Search", name=config['config_name'] + 'eval')
    args_list = []
    for key, value in config["eval"].items():
        args_list.append(f"--{key}")
        args_list.append(str(value))

    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses(args_list)[0]

    corpus = datasets.load_dataset('json', data_files=args.corpus_file,
                                   split='train')
    model = FlagModel(
        args.encoder,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
        use_fp16=args.fp16,
        model_type=args.model_type,
        pooling_method='mean',
        normalize_embeddings=True
    )

    corpus_context = []
    corpus_goal = []
    for item in corpus["premise"]:
        combine = process_strings(item['state']["context"])
        corpus_context.append(combine)
        goal = re.sub(r'\n\s+', ' ', item['state']["goal"])
        corpus_goal.append("<GOAL>" + goal)

    corpus_embeddings_context = index(
        model=model,
        corpus=corpus_context,
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        save_path=args.save_path_context,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )

    corpus_embeddings_goal = index(
        model=model,
        corpus=corpus_goal,
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        save_path=args.save_path_goal,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )

    query_all_state = []
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_all_state.append(data)

    group_size = 3000

    output_file = args.save_output_path
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"file {output_file} deleted")
    else:
        print(f"file {output_file} not exist")
    with open(output_file, "w+", encoding="utf-8") as file:
        for i in range(0, len(query_all_state), group_size):
            now_process_data = query_all_state[i:i + group_size]
            retrieval_results = []
            ground_truths = []
            query_state = []
            all_premises = []
            try:
                scores_one, indices_one = search(
                    model=model,
                    queries=now_process_data,
                    corpus_embeddings_context=corpus_embeddings_context,
                    corpus_embeddings_goal=corpus_embeddings_goal,
                    k=args.k,
                    batch_size=args.batch_size,
                    max_length=args.max_query_length
                )
                for indice in indices_one:
                    # filter invalid indices
                    indice = indice[indice != -1].tolist()
                    retrieval_results.append(corpus[indice]["id"])
                for sample in now_process_data:
                    ground_truths.append(sample["premise"])
                    all_premises.append(sample['all_premises'])
                    query_state.append(sample['state'])
            except:
                print("no one state")
            assert len(retrieval_results) == len(ground_truths) == len(query_state) == len(all_premises), "列表长度不一致！"

            for query, ground_truth, pre_premise, all_premise in zip(query_state, ground_truths, retrieval_results,
                                                                     all_premises):
                neg_premise = [item for item in pre_premise if item not in all_premise]

                data = {
                    "query": query,
                    "ground_truths": ground_truth,
                    "pre_premise": pre_premise,
                    "all_premises": all_premise,
                    "neg_premises": neg_premise
                }
                file.write(json.dumps(data, ensure_ascii=False) + "\n")

            print(f"Data written to {i}")

            retrieval_results.clear()
            ground_truths.clear()
            query_state.clear()


    print(f"Data written to {output_file}")




if __name__ == "__main__":
    main()

