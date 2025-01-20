import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import re
import numpy as np
from tqdm import tqdm
import argparse
import toml
from sklearn.metrics import roc_auc_score, ndcg_score
from pathlib import Path
import csv
def moudle_id(module_id_path):
    with open(module_id_path, 'r') as file:
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

def process_strings(string_list):
    processed_list = []
    for s in string_list:
        s = "<VAR>" + s
        s = re.sub(r'\n\s+', ' ', s)
        processed_list.append(s)
    result = ''.join(processed_list)
    return result


def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_dict = {}
            data = json.loads(line)
            context = data['premise']['state']['context']
            goal = data['premise']['state']['goal']
            context = process_strings(context)
            goal = re.sub(r'\n\s+', ' ', goal)
            temp_dict['context'] = context
            temp_dict['goal'] = "<GOAL>" + goal
            temp_dict['state'] = temp_dict['context'] + temp_dict['goal']
            corpus[data['id']] = temp_dict
    return corpus


def get_query_pair(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def main():
    parser_config = argparse.ArgumentParser()
    parser_config.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args_config = parser_config.parse_args()
    config = toml.load(args_config.config_path)

    model_path = config['rerank']['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda')
    model.eval()
    print(model.config.num_labels)
    corpus = load_corpus(
        corpus_path=config['eval']['corpus_file'])

    data_path = config['rerank']['test_path']
    query_data = get_query_pair(data_path)
    R1 = []
    R5 = []
    R10 = []
    R15 = []
    R20 = []
    R1_cut = []
    R5_cut = []
    R10_cut = []
    R15_cut = []
    R20_cut = []
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
    count = 0
    rerank_num = config['rerank']['rerank_num']
    module_id_path = config['finetune']['module_premise']
    module_dict, premise_name_dict = moudle_id(module_id_path)
    for item in tqdm(range(len(query_data)), desc="Processing Items"):
        query_all = query_data[item]['query']
        logits_all = None
        if len(query_all) != 1:
            count = count + 1
        pred_premise = query_data[item]['pre_premise']
        ground_truth = query_data[item]['ground_truths']
        relevant_truth = find_relevant_premise(ground_truth, module_dict, premise_name_dict)
        if len(ground_truth) == 0:
            continue
        for query_dict in query_all:
            query_goal = query_dict['goal']
            query_context = query_dict['context']
            context = process_strings(query_context)
            goal = re.sub(r'\n\s+', ' ', query_goal)
            query = context + '<GOAL>' + goal
            pairs = []
            count = 0
            for pre_pre in pred_premise:
                temp_list = []
                pre_pre_state = corpus[pre_pre]['state']
                temp_list.append(query)
                temp_list.append(pre_pre_state)
                pairs.append(temp_list)
                count = count + 1
                if count == rerank_num:
                    break

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
                inputs = inputs.to('cuda')
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
                if logits_all is None:
                    logits_all = scores
                else:
                    logits_all += scores
        probabilities_all = torch.nn.functional.softmax(logits_all, dim=0)
        sorted_indices = torch.argsort(probabilities_all, descending=True)
        sorted_pre_pre = [(pred_premise[idx], probabilities_all[idx].item()) for idx in sorted_indices]

        rerank_pres = []
        score_pres = []
        for pre_pre, score in sorted_pre_pre:
            rerank_pres.append(pre_pre)
            score_pres.append(score)
        all_scores.append(np.array(score_pres).reshape(1, -1))
        label = set(ground_truth)
        TP1 = rerank_pres[0] in label
        R1.append(float(TP1) / len(label))
        TP5 = len(label.intersection(rerank_pres[:5]))
        R5.append(float(TP5) / len(label))
        TP10 = len(label.intersection(rerank_pres[:10]))
        R10.append(float(TP10) / len(label))
        TP15 = len(label.intersection(rerank_pres[:15]))
        R15.append(float(TP15) / len(label))
        TP20 = len(label.intersection(rerank_pres[:20]))
        R20.append(float(TP20) / len(label))

        R1_cut.append(float(TP1) / min(1, len(label)))
        R5_cut.append(float(TP5) / min(5, len(label)))
        R10_cut.append(float(TP10) / min(10, len(label)))
        R15_cut.append(float(TP15) / min(15, len(label)))
        R20_cut.append(float(TP20) / min(20, len(label)))
        TP3 = len(label.intersection(rerank_pres[:3]))
        P1.append(float(TP1) / 1)
        P3.append(float(TP3) / 3)
        P5.append(float(TP5) / 5)
        P10.append(float(TP10) / 10)
        P15.append(float(TP15) / 15)
        P20.append(float(TP20) / 20)

        pred_hard_encoding_auc = np.isin(rerank_pres, list(label)).astype(int).tolist()
        pred_hard_encoding = []
        for pre in rerank_pres:
            if pre in list(label):
                pred_hard_encoding.append(1)
            elif pre in relevant_truth:
                pred_hard_encoding.append(0.3)
            else:
                pred_hard_encoding.append(0)
        pred_hard_encodings.append(pred_hard_encoding)
        pred_hard_encodings_auc.append(pred_hard_encoding_auc)
        for j, p in enumerate(rerank_pres):
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
        print(f"nDCG@{cutoff}: {nDCG*100}")
        nDCG_all[f"nDCG@{cutoff}"]=nDCG
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
    print(count)

    config_name = args_config.config_path
    config_name = Path(config_name).name
    config_name = config_name.split('.')[0]
    data_name = Path(data_path).name
    data_name = data_name.split('.')[0]
    save_name = config_name+"_"+data_name
    output_dir = 'Result_save/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    row = [
        nDCG_all['nDCG@1'], nDCG_all['nDCG@5'], nDCG_all['nDCG@10'],nDCG_all['nDCG@15'], nDCG_all['nDCG@20'],
        F1_1, F1_5, F1_10,F1_15,F1_20,
        R1, R5, R10,R15, R20,  R1_cut, R5_cut, R10_cut,R15_cut, R20_cut,
        P1, P3, P5, P10,P15, P20,
        MRR, auc
    ]

    header = [
        'nDCG@1', 'nDCG@5', 'nDCG@10','nDCG@15', 'nDCG@20',
        'F1@1', 'F1@5', 'F1@10', 'F1@15', 'F1@20',
        'R1', 'R5', 'R10', 'R15', 'R20',  'R1_CUT', 'R5_CUT', 'R10_CUT','R15_CUT', 'R20_CUT',
        'P1', 'P3', 'P5', 'P10','P15', 'P20',
        'MRR', 'AUC'
    ]

    output_file = os.path.join(output_dir, save_name+'.csv')

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

    print(f"Metrics saved to {output_file_json}")
if __name__ == '__main__':
    main()
