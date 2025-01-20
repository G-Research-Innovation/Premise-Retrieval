from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from pathlib import Path
from transformers import BertTokenizer, RobertaTokenizer
import argparse
import toml
import os


def WordPiece_Tokenizer(corpus_path, save_dir):
    file_path = corpus_path
    assert Path(file_path).exists(), f"file {file_path} not exist！"

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "<VAR>", "<GOAL>"]
    tokenizer.train(
        files=[file_path],
        vocab_size=30520,
        min_frequency=2,
        limit_alphabet=1000,
        special_tokens=special_tokens,
        wordpieces_prefix="##"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer.save_model(save_dir)

    tokenizer = BertTokenizer.from_pretrained(os.path.join(save_dir, 'vocab.txt'), do_lower_case=False)
    special_tokens = ["<VAR>", "<GOAL>"]
    tokenizer.add_tokens(special_tokens)
    sample_text = "This is a<VAR>sample sentence."
    tokenized_output = tokenizer.encode(sample_text)

    print("Token IDs:", tokenized_output)
    print("Decoded text:", tokenizer.decode(tokenized_output))


def BPE_tokenizer(corpus_path, save_dir):
    file_path = corpus_path
    assert Path(file_path).exists(), f"file {file_path} not exist！"
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "</s>", "<pad>", "<VAR>", "<GOAL>"]
    tokenizer.train(
        files=[file_path],
        vocab_size=52000,
        special_tokens=special_tokens
    )
    tokenizer.save_model(save_dir)
    tokenizer = RobertaTokenizer.from_pretrained(save_dir)
    special_tokens = ["<VAR>", "<GOAL>"]
    tokenizer.add_tokens(special_tokens)
    sample_text = "This is a<VAR>sample sentence."
    tokenized_output = tokenizer.encode(sample_text)
    print("Token IDs:", tokenized_output)
    print("Decoded text:", tokenizer.decode(tokenized_output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", required=True, type=str, help="Path to the config file"
    )
    args = parser.parse_args()
    config = toml.load(args.config_path)
    corpus_path = config['tokenizer']['corpus_path']
    tokenizer_type = config['tokenizer']['tokenizer_type']
    save_dir = config['tokenizer']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if tokenizer_type == 'WordPiece':
        WordPiece_Tokenizer(corpus_path, save_dir)
    elif tokenizer_type == 'BPE':
        BPE_tokenizer(corpus_path, save_dir)


if __name__ == '__main__':
    main()
