import re
import json
from pathlib import Path

import jieba
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm

import config
from tokenizer import MyTokenizer


outdir = Path(config.PROCESSED_DATA_DIR)

if not outdir.exists():
    outdir.mkdir(parents=True)


# Load and extract sentences.
ds = load_dataset("Jax-dan/HundredCV-Chat", cache_dir=config.RAW_DATA_DIR)
sentences = []
for dialog in tqdm(ds["train"]["dialog"], desc="Load dialog to list"):
    sentences.extend(dialog)

# Trim prefix.
PREFIX_PATTERN = re.compile(r"user\d：") # Note: here is chinese colon "：".
sentences = [PREFIX_PATTERN.sub("", sentence) for sentence in sentences]

train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

MyTokenizer.build_vocab(train_sentences, config.VOCAB_FILE)
tokenizer = MyTokenizer.from_vocab(config.VOCAB_FILE)

train_encoded_list = [tokenizer.encode(sentence) for sentence in train_sentences]
test_encoded_list = [tokenizer.encode(sentence) for sentence in test_sentences]

def make_input_and_target(encoded_list, seq_len=config.SEQ_LEN):

    out = []
    for encoded in encoded_list:
        if len(encoded) <= seq_len:  # i is the index of target
            i = len(encoded)-1
            out.append({
                "input": encoded[:i] + [0] * (seq_len-i),
                "target": encoded[i]
            })
        else:
            for i in range(seq_len, len(encoded)):
                out.append({
                    "input": encoded[(i-seq_len):i],
                    "target": encoded[i]
                })
    return out

train_data = make_input_and_target(train_encoded_list)
test_data = make_input_and_target(test_encoded_list)

def write_jsonl(list_of_dict, file):
    with open(file, "w") as f:
        for x in list_of_dict:
            f.write(json.dumps(x) + "\n")

write_jsonl(train_data, outdir / "train_data.jsonl")
write_jsonl(test_data, outdir / "test_data.jsonl")
