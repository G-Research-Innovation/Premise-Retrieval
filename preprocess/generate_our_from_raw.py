import os
import argparse
import json
from dataclasses import dataclass, field
from typing import Generator
import re
import toml


@dataclass(eq=True, unsafe_hash=True)
class Pos:
    """Position in source files.

    We use 1-index to keep it consistent with code editors such as Visual Studio Code.
    """

    line_nb: int
    """Line number
    """

    column_nb: int
    """Column number
    """

    @classmethod
    def from_str(cls, s: str) -> "Pos":
        """Construct a :class:`Pos` object from its string representation, e.g., :code:`"(323, 1109)"`."""
        assert s.startswith("(") and s.endswith(
            ")"
        ), f"Invalid string representation of a position: {s}"
        line, column = s[1:-1].split(",")
        line_nb = int(line)
        column_nb = int(column)
        return cls(line_nb, column_nb)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.line_nb
        yield self.column_nb

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __lt__(self, other):
        return self.line_nb < other.line_nb or (
                self.line_nb == other.line_nb and self.column_nb < other.column_nb
        )

    def __le__(self, other):
        return self < other or self == other


@dataclass(unsafe_hash=True)
class Premise_corpus:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False, compare=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """
    code: str = field(compare=False)

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
                isinstance(self.start, Pos)
                and isinstance(self.end, Pos)
        )
        assert (
                self.start <= self.end
        )
        assert isinstance(self.code, str) and self.code != ""


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False, compare=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    # code: str = field(compare=False)
    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
                isinstance(self.start, Pos)
                and isinstance(self.end, Pos)
        )
        assert (
                self.start <= self.end
        )
        # assert isinstance(self.code, str) and self.code != ""


def get_code_without_doc_string(code):
    pattern = r"/--(.*?)-/\n"
    cleaned_code = re.sub(pattern, "", code, flags=re.DOTALL)
    return cleaned_code


def _load_corpus_update(corpus_path, output_path_statement, output_path_module_id):
    corpus = []
    count = 0
    moudel_id = {}
    path_premise = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            def_path = data['path']
            premise_num = []
            for premise in data['premises']:
                if premise['is_thm'] == False:
                    continue
                temp_dict = {}
                premise_state = {}
                premise_state['context'] = [item[1:-1] for item in premise['args']]
                premise_state['goal'] = premise['goal']
                premise['state'] = premise_state
                premise['code'] = get_code_without_doc_string(premise['code'])
                temp_dict['id'] = count
                temp_dict['premise'] = premise
                temp_dict['def_path'] = def_path
                now_premise = Premise(def_path, premise['full_name'],
                                      Pos(premise['start'][0], premise['start'][1]),
                                      Pos(premise['end'][0], premise['end'][1]))
                p = now_premise
                full_name = p.full_name
                if full_name is None:
                    continue
                if "user__.n" in full_name or premise['code'] == "":
                    # Ignore ill-formed premises (often due to errors in ASTs).
                    continue
                if full_name.startswith("[") and full_name.endswith("]"):
                    # Ignore mutual definitions.
                    continue
                if count == 0:
                    print(premise_state)
                    print(now_premise)
                if now_premise in moudel_id:
                    raise ValueError
                    # continue
                moudel_id[now_premise] = count
                premise_num.append(count)
                count += 1
                corpus.append(temp_dict)

            if def_path in path_premise:
                raise ValueError
            path_premise[def_path] = premise_num

    with open(output_path_statement, 'w', encoding='utf-8') as out_f:
        for item in corpus:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(output_path_module_id, 'w', encoding='utf-8') as f:
        json.dump(path_premise, f, ensure_ascii=False, indent=4)

    return moudel_id


def make_hashable(data):
    if isinstance(data, dict):
        return frozenset((key, make_hashable(value)) for key, value in data.items())
    elif isinstance(data, list):
        return tuple(make_hashable(item) for item in data)
    else:
        return data


def process_state(state):
    processed_state = re.sub(r"^case.*(?:\n|$)", "", state, flags=re.MULTILINE)
    split_states = processed_state.split("\n\n")
    proofstates = []
    for s in split_states:
        s = s.strip()
        if s == "no goals":
            proofstates.append({"context": [], "goal": "no goals"})
            continue
        if "⊢" not in s:
            continue
        try:
            context_str, goal = s.split("⊢")
            context_str = re.sub(r"\n\s+", " ", context_str).strip()
            goal = re.sub(r"\n\s+", " ", goal).strip()
            context = list(filter(lambda v: ":" in v, context_str.split("\n")))
            proofstates.append({"context": context, "goal": goal})
        except:
            print(state, s)
    return proofstates


def _get_dataset_path(path, save_path, modeul_id):
    data_list = []
    not_in = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            traced_tactics = data['tactics']
            def_path = data['file_path']
            for i in range(len(traced_tactics)):
                now_tactic = traced_tactics[i]
                annotated_tactic = now_tactic['premises']
                state_before = now_tactic['state_before']
                state_after = now_tactic['state_after']
                all_state_before = process_state(state_before)
                all_state_after = process_state(state_after)
                premises_list = []
                for j in range(len(annotated_tactic)):
                    now_premise = annotated_tactic[j]
                    premise = Premise(now_premise['def_path'], now_premise['full_name'],
                                      Pos(now_premise['def_pos'][0], now_premise['def_pos'][1]),
                                      Pos(now_premise['def_end_pos'][0], now_premise['def_end_pos'][1]))
                    if premise not in modeul_id:
                        not_in.add(premise)
                        continue
                    premises_list.append(modeul_id[premise])
                if len(premises_list) == 0:
                    continue
                premises_list = list(set(premises_list))
                premises_list = sorted(premises_list)
                for k in range(len(all_state_before)):
                    now_state = all_state_before[k]
                    if now_state['goal'] != "no goals":
                        temp_dict = {}
                        temp_dict["state"] = now_state
                        temp_dict['premise'] = premises_list
                        temp_dict['module'] = [def_path]
                        temp_dict['state_str'] = state_before
                        data_list.append(temp_dict)

                for k in range(len(all_state_after)):
                    now_state = all_state_after[k]
                    if now_state['goal'] != "no goals":
                        temp_dict = {}
                        temp_dict["state"] = now_state
                        temp_dict['premise'] = premises_list
                        temp_dict['module'] = [def_path]
                        temp_dict['state_str'] = state_after
                        data_list.append(temp_dict)

    state_hashes = {}

    for temp_dict in data_list:
        state = temp_dict["state"]
        premise = temp_dict["premise"]
        module = temp_dict["module"]

        state_hash = hash(make_hashable(state))

        if state_hash in state_hashes:
            matched_dict = state_hashes[state_hash]
            matched_dict["premise"].extend(premise)
            matched_dict["module"].extend(module)
            matched_dict["premise"] = list(set(matched_dict["premise"]))
            matched_dict["module"] = list(set(matched_dict["module"]))
        else:
            state_hashes[state_hash] = {
                "state": state,
                "premise": premise,
                "module": module
            }

    merged_data_list = list(state_hashes.values())

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in merged_data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"Data written to {save_path}")
    print(len(not_in))


def _get_dataset_path_test(path, save_path, is_train, modeul_id):
    data_list = []
    not_in = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            traced_tactics = data['tactics']
            for i in range(len(traced_tactics)):
                now_tactic = traced_tactics[i]
                annotated_tactic = now_tactic['premises']
                state_str = now_tactic['state_before']
                state_before = process_state(now_tactic['state_before'])
                premises_list = []
                for j in range(len(annotated_tactic)):
                    now_premise = annotated_tactic[j]
                    premise = Premise(now_premise['def_path'], now_premise['full_name'],
                                      Pos(now_premise['def_pos'][0], now_premise['def_pos'][1]),
                                      Pos(now_premise['def_end_pos'][0], now_premise['def_end_pos'][1]))
                    if premise not in modeul_id:
                        not_in.add(premise)
                        continue
                    premises_list.append(modeul_id[premise])
                if len(premises_list) == 0:
                    continue
                premises_list = list(set(premises_list))
                premises_list = sorted(premises_list)
                if state_before[0]['goal'] != "no goals":

                    temp_dict = {}
                    temp_dict['state'] = state_before
                    temp_dict['premise'] = premises_list
                    temp_dict['state_str'] = state_str
                    data_list.append(temp_dict)
                else:
                    if len(state_before) != 0:
                        raise ValueError
    if is_train:
        state_hashes = {}

        for temp_dict in data_list:
            state = temp_dict["state"]
            premise = temp_dict["premise"]
            state_str = temp_dict['state_str']
            state_hash = hash(make_hashable(state))

            # Flag to indicate if we've found a matching state in merged_data_list
            if state_hash in state_hashes:
                matched_dict = state_hashes[state_hash]
                matched_dict["premise"].extend(premise)
            else:
                state_hashes[state_hash] = {
                    "state": state,
                    "premise": premise,
                    'state_str': state_str
                }

        merged_data_list = list(state_hashes.values())
    else:
        merged_data_list = data_list

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in merged_data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(len(merged_data_list))
    print(f"Data written to {save_path}")


def expand_data(input_file, output_file):
    expanded_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            for premise_id in record["premise"]:
                new_record = record.copy()
                new_record["premise"] = premise_id
                new_record['all_premises'] = record['premise']
                expanded_data.append(new_record)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in expanded_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Data written to {output_file}")
    print(len(expanded_data))


def main():
    parser_config = argparse.ArgumentParser()
    parser_config.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args_config = parser_config.parse_args()
    config = toml.load(args_config.config_path)
    all_path_ori = config['dataset']['parent_path']
    all_path_save = config['dataset']['parent_save_path_our']
    if not os.path.exists(all_path_save):
        os.makedirs(all_path_save)

    train_path = all_path_ori + "train.jsonl"
    test_path = all_path_ori + "test.jsonl"
    val_path = all_path_ori + "valid.jsonl"
    train_path_save = all_path_save + 'train.jsonl'
    test_path_save = all_path_save + 'test.jsonl'
    val_path_save = all_path_save + 'val.jsonl'
    test_path_save_increase = all_path_save + 'test_increase.jsonl'
    val_path_save_increase = all_path_save + 'val_increase.jsonl'
    train_expand_path = all_path_save + "train_expand_premise.jsonl"

    modeul_id = _load_corpus_update(
        config['dataset']['all_data_path'] + "corpus.jsonl",
        config['dataset']['all_data_path'] + 'statement.jsonl',
        config['dataset']['all_data_path'] + 'module_id_mapping.json',
    )

    _get_dataset_path(train_path, train_path_save, modeul_id)
    _get_dataset_path(test_path, test_path_save, modeul_id)
    _get_dataset_path(val_path, val_path_save, modeul_id)

    expand_data(train_path_save, train_expand_path)

    _get_dataset_path_test(test_path, test_path_save_increase, is_train=False, modeul_id=modeul_id)
    _get_dataset_path_test(val_path, val_path_save_increase, is_train=False, modeul_id=modeul_id)


if __name__ == "__main__":
    main()
