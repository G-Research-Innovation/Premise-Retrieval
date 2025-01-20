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
import csv
import os
from sklearn.metrics import roc_auc_score, ndcg_score
from pathlib import Path

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
        default=20,
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
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings

    corpus_embeddings = corpus_embeddings.astype(np.float32)
    return corpus_embeddings


def search_state_combine(model, queries, corpus_embeddings_context, corpus_embeddings_goal, k: int = 10,
                         batch_size: int = 256,
                         max_length: int = 512):
    query_size = len(queries)
    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size), desc='Searching'):
        item = queries[i]['state']
        state_context_all = item
        combined_query_embedding = None

        for state_context in state_context_all:
            context = state_context['context']
            goal = state_context['goal']
            context = process_strings(context)
            goal = re.sub(r'\n\s+', ' ', goal)
            combine = context + '<GOAL>' + goal
            query = [combine]

            query_embedding = model.encode_queries(query, batch_size=batch_size, max_length=max_length)

            if combined_query_embedding is None:
                combined_query_embedding = query_embedding
            else:
                combined_query_embedding += query_embedding

        similarities_context = np.dot(combined_query_embedding, corpus_embeddings_context.T)
        similarities_goal = np.dot(combined_query_embedding, corpus_embeddings_goal.T)
        similarities_total = (similarities_context + similarities_goal) / 2

        indices = np.argsort(similarities_total, axis=1)[:, ::-1][:, :k]
        scores = np.take_along_axis(similarities_total, indices, axis=1)

        all_scores.append(scores)
        all_indices.append(indices)

    all_scores1 = np.concatenate(all_scores, axis=0)
    all_indices1 = np.concatenate(all_indices, axis=0)
    return all_scores1, all_indices1


def search(model: FlagModel, queries: datasets, corpus_embeddings_context, corpus_embeddings_goal, k: int = 10,
           batch_size: int = 256,
           max_length: int = 512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query = []

    for item in queries:
        context = item['state'][0]['context']
        goal = item['state'][0]['goal']
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


def moudle_id(module_id_path):
    with open(module_id_path,
              'r') as file:
        data = json.load(file)

    module_dict = {}
    premise_name_dict = {}

    for module_name, premise_ids in data.items():
        module_dict[module_name] = premise_ids
        for premise_id in premise_ids:
            premise_name_dict[premise_id] = module_name

    return module_dict, premise_name_dict


def find_relevant_premise(ground_truth, module_dict, premise_name_dict):
    all_premise = []
    for id in ground_truth:
        module = premise_name_dict[id]
        premises = module_dict[module]
        all_premise.extend(premises)
    all_premise = list(set(all_premise) - set(ground_truth))
    return all_premise


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluate2(preds,
              preds_scores,
              labels,
              save_name,
              module_id_path,
              cutoffs=[1, 5, 10]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    module_dict, premise_name_dict = moudle_id(module_id_path)
    R1 = []
    R5 = []
    R10 = []
    R15 = []
    R20 = []
    R100 = []
    R1_cut = []
    R5_cut = []
    R10_cut = []
    R15_cut = []
    R20_cut = []
    R100_cut = []
    P1 = []
    P3 = []
    P5 = []
    P10 = []
    P15 = []
    P20 = []
    pred_hard_encodings = []
    pred_hard_encodings_auc = []
    all_scores = []
    MRR = []

    for pred, label, score in zip(preds, labels, preds_scores):
        if len(label) == 0:
            continue
        all_scores.append(np.array(score).reshape(1, -1))
        relevant_truth = find_relevant_premise(label, module_dict, premise_name_dict)
        label = set(label)
        TP1 = pred[0] in label
        R1.append(float(TP1) / len(label))
        TP5 = len(label.intersection(pred[:5]))
        R5.append(float(TP5) / len(label))
        TP10 = len(label.intersection(pred[:10]))
        R10.append(float(TP10) / len(label))
        TP15 = len(label.intersection(pred[:15]))
        R15.append(float(TP15) / len(label))
        TP20 = len(label.intersection(pred[:20]))
        R20.append(float(TP20) / len(label))

        TP100 = len(label.intersection(pred[:100]))
        R100.append(float(TP100) / len(label))
        R1_cut.append(float(TP1) / min(1, len(label)))
        R5_cut.append(float(TP5) / min(5, len(label)))
        R10_cut.append(float(TP10) / min(10, len(label)))
        R15_cut.append(float(TP15) / min(15, len(label)))
        R20_cut.append(float(TP20) / min(20, len(label)))
        R100_cut.append(float(TP10) / min(100, len(label)))
        TP3 = len(label.intersection(pred[:3]))
        P1.append(float(TP1) / 1)
        P3.append(float(TP3) / 3)
        P5.append(float(TP5) / 5)
        P10.append(float(TP10) / 10)
        P15.append(float(TP15) / 15)
        P20.append(float(TP20) / 20)

        pred_hard_encoding_auc = np.isin(pred, list(label)).astype(int).tolist()
        pred_hard_encoding = []
        for pre in pred:
            if pre in list(label):
                pred_hard_encoding.append(1)
            elif pre in relevant_truth:
                pred_hard_encoding.append(0.3)
            else:
                pred_hard_encoding.append(0)
        pred_hard_encodings.append(pred_hard_encoding)
        pred_hard_encodings_auc.append(pred_hard_encoding_auc)

        for j, p in enumerate(pred):
            if p in label:
                MRR.append(1.0 / (j + 1))
                break
        else:
            MRR.append(0.0)
    R1 = 100 * np.mean(R1)
    R5 = 100 * np.mean(R5)
    R10 = 100 * np.mean(R10)
    R15 = 100 * np.mean(R15)
    R20 = 100 * np.mean(R20)
    R1_cut = 100 * np.mean(R1_cut)
    R5_cut = 100 * np.mean(R5_cut)
    R10_cut = 100 * np.mean(R10_cut)
    R15_cut = 100 * np.mean(R15_cut)
    R20_cut = 100 * np.mean(R20_cut)
    P1 = 100 * np.mean(P1)
    P3 = 100 * np.mean(P3)
    P5 = 100 * np.mean(P5)
    P10 = 100 * np.mean(P10)
    P15 = 100 * np.mean(P15)
    P20 = 100 * np.mean(P20)
    preds_scores = np.concatenate(all_scores, axis=0)
    pred_hard_encodings1d = np.asarray(pred_hard_encodings_auc).flatten()
    preds_scores1d = preds_scores.flatten()
    auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)
    cutoffs = [1, 5, 10, 15, 20]
    print(np.shape(preds_scores))
    nDCG_all = {}
    # nDCG
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
        print(f"nDCG@{cutoff}: {nDCG * 100}")
        nDCG_all[f"nDCG@{cutoff}"] = nDCG
    MRR = np.mean(MRR)
    F1_1 = calculate_f1(P1, R1)
    F1_5 = calculate_f1(P5, R5)
    F1_10 = calculate_f1(P10, R10)
    F1_15 = calculate_f1(P15, R15)
    F1_20 = calculate_f1(P20, R20)

    print(f"F1@1: {F1_1:.4f}")
    print(f"F1@5: {F1_5:.4f}")
    print(f"F1@10: {F1_10:.4f}")
    print(f"F1@15: {F1_15:.4f}")
    print(f"F1@20: {F1_20:.4f}")

    print("R1:", R1)
    print("R5:", R5)
    print("R10:", R10)
    print("R15:", R15)
    print("R20:", R20)
    print("R1_CUT:", R1_cut)
    print("R5_CUT:", R5_cut)
    print("R10_CUT:", R10_cut)
    print("R15_CUT:", R15_cut)
    print("R20_CUT:", R20_cut)
    print("P1:", P1)
    print("P3:", P3)
    print("P5:", P5)
    print("P10:", P10)
    print("P15:", P15)
    print("P20:", P20)

    print("MRR:", MRR)
    print('AUC:', auc)

    output_dir = 'Result_save_retrieval/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    row = [
        nDCG_all['nDCG@1'], nDCG_all['nDCG@5'], nDCG_all['nDCG@10'], nDCG_all['nDCG@15'], nDCG_all['nDCG@20'],
        F1_1, F1_5, F1_10, F1_15, F1_20,
        R1, R5, R10, R15, R20, R1_cut, R5_cut, R10_cut, R15_cut, R20_cut,
        P1, P3, P5, P10, P15, P20,
        MRR, auc
    ]

    header = [
        'nDCG@1', 'nDCG@5', 'nDCG@10', 'nDCG@15', 'nDCG@20',
        'F1@1', 'F1@5', 'F1@10', 'F1@15', 'F1@20',
        'R1', 'R5', 'R10', 'R15', 'R20', 'R1_CUT', 'R5_CUT', 'R10_CUT', 'R15_CUT', 'R20_CUT',
        'P1', 'P3', 'P5', 'P10', 'P15', 'P20',
        'MRR', 'AUC'
    ]

    output_file = os.path.join(output_dir, save_name + '.csv')

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row)

    print(f"Metrics saved to {output_file}")

    metrics = {
        'nDCG@1': nDCG_all['nDCG@1'],
        'nDCG@5': nDCG_all['nDCG@5'],
        'nDCG@10': nDCG_all['nDCG@10'],
        'nDCG@15': nDCG_all['nDCG@15'],
        'nDCG@20': nDCG_all['nDCG@20'],
        'F1@1': F1_1,
        'F1@5': F1_5,
        'F1@10': F1_10,
        'F1@15': F1_15,
        'F1@20': F1_20,
        'R1': R1,
        'R5': R5,
        'R10': R10,
        'R15': R15,
        'R20': R20,
        'R1_CUT': R1_cut,
        'R5_CUT': R5_cut,
        'R10_CUT': R10_cut,
        'R15_CUT': R15_cut,
        'R20_CUT': R20_cut,
        'P1': P1,
        'P3': P3,
        'P5': P5,
        'P10': P10,
        'P15': P15,
        'P20': P20,
        'MRR': MRR,
        'AUC': auc
    }

    output_file_json = os.path.join(output_dir, save_name + '.json')

    with open(output_file_json, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    return metrics

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
    module_id_path = config['finetune']['module_premise']
    wandb.init(project="Leandojo_Lean_Sate_Search", name=config['config_name'] + 'eval')
    args_list = []
    for key, value in config["eval"].items():
        args_list.append(f"--{key}")
        args_list.append(str(value))

    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses(args_list)[0]
    config_name = args_config.config_path
    config_name = Path(config_name).name
    config_name = config_name.split('.')[0]
    data_name = Path(args.eval_file).name
    data_name = data_name.split('.')[0]
    save_name = config_name + "_" + data_name

    corpus = datasets.load_dataset('json', data_files=args.corpus_file,
                                   split='train')
    print(args.encoder)
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
    c_equal = 0
    c_exclude = 0
    query_one_state = []
    query_multi_state = []
    query_all_state = []
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_all_state.append(data)
            if len(data['state']) == 1:
                c_equal += 1
                query_one_state.append(data)
            else:
                c_exclude += 1
                query_multi_state.append(data)
    retrieval_results = []
    ground_truths = []
    query_state = []
    scores_for_queries = []
    try:
        scores_one, indices_one = search(
            model=model,
            queries=query_one_state,
            corpus_embeddings_context=corpus_embeddings_context,
            corpus_embeddings_goal=corpus_embeddings_goal,
            k=args.k,
            batch_size=args.batch_size,
            max_length=args.max_query_length
        )
        # Iterate over the indices and process them
        for i, indice in enumerate(indices_one):
            # Filter invalid indices (those that are -1)
            valid_indices = indice[indice != -1].tolist()
            valid_scores = scores_one[i][indice != -1].tolist()  # Apply the same filtering to scores
            # Append the results to retrieval_results, scores_for_queries
            retrieval_results.append(corpus[valid_indices]["id"])
            scores_for_queries.append(valid_scores)  # Store the valid scores for each query

        for sample in query_one_state:
            ground_truths.append(sample["premise"])
            query_state.append(sample['state'])
    except:
        print("no one state")

    try:
        scores_multi, indices_multi = search_state_combine(
            model=model,
            queries=query_multi_state,
            corpus_embeddings_context=corpus_embeddings_context,
            corpus_embeddings_goal=corpus_embeddings_goal,
            k=args.k,
            batch_size=args.batch_size,
            max_length=args.max_query_length
        )

        # Iterate over the indices and process them
        for i, indice in enumerate(indices_multi):
            # Filter invalid indices (those that are -1)
            valid_indices = indice[indice != -1].tolist()
            valid_scores = scores_multi[i][indice != -1].tolist()  # Apply the same filtering to scores
            # Append the results to retrieval_results, scores_for_queries
            retrieval_results.append(corpus[valid_indices]["id"])
            scores_for_queries.append(valid_scores)  # Store the valid scores for each query

        for sample in query_multi_state:
            ground_truths.append(sample["premise"])
            query_state.append(sample['state'])
    except:
        print("no multi state")

    assert len(retrieval_results) == len(ground_truths) == len(query_state) == len(scores_for_queries), "列表长度不一致！"

    metrics = evaluate2(retrieval_results, scores_for_queries, ground_truths, save_name, module_id_path=module_id_path)
    print(metrics)

    assert len(retrieval_results) == len(ground_truths) == len(query_state) == len(scores_for_queries), "列表长度不一致！"

    output_file = args.save_output_path

    with open(output_file, "w", encoding="utf-8") as file:
        for query, ground_truth, pre_premise in zip(query_state, ground_truths, retrieval_results):
            data = {
                "query": query,
                "ground_truths": ground_truth,
                "pre_premise": pre_premise
            }
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()

