import json
import re
import random
import argparse
import toml

def process_strings(string_list):
    processed_list = []

    for s in string_list:
        s = "<VAR>" + s
        s = re.sub(r'\n\s+', ' ', s)
        processed_list.append(s)
    result = ''.join(processed_list)
    return result


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_data(json_data, train_ratio=0.95, seed=42):
    random.seed(seed)
    random.shuffle(json_data)
    train_size = int(train_ratio * len(json_data))
    train_data = json_data[:train_size]
    eval_data = json_data[train_size:]
    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args = parser.parse_args()
    config = toml.load(args.config_path)
    file_path = config['dataset']['train_dataset_path']
    corpus_path = config['finetune']['used_premise']
    json_data = load_jsonl(file_path)
    corpus_data = load_jsonl(corpus_path)
    json_data.extend(corpus_data)
    all_final_data = []
    output_file = config['tokenizer']['corpus_path']
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(json_data)):
            now_data = json_data[i]
            tem_dict = {}

            if 'state' in now_data.keys():
                context = now_data['state']['context']
                goal = now_data['state']['goal']
                tem_dict['context'] = context
                tem_dict['goal'] = goal
            else:
                context = now_data['premise']['state']['context']
                goal = now_data['premise']['state']['goal']
                tem_dict['context'] = context
                tem_dict['goal'] = goal
            context = process_strings(context)
            goal = re.sub(r'\n\s+', ' ', goal)
            combine = context + '<GOAL>' + goal
            f.write(combine + '\n')

            temp_dict = {}
            temp_dict['state'] = tem_dict
            all_final_data.append(temp_dict)


    print("The total data length：",len(json_data))

    train_data, eval_data = split_data(all_final_data)

    train_file_path = config['pretrain']['train_file_path']
    eval_file_path = config['pretrain']['eval_file_path']

    save_jsonl(train_data, train_file_path)
    save_jsonl(eval_data, eval_file_path)



if __name__ == '__main__':
    main()
