from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from tokenizer import PadTokenizer

outdir = Path(config.PROCESSED_DATA_DIR)

if not outdir.exists():
    outdir.mkdir(parents=True)

raw_data = pd.read_csv("data/raw/online_shopping_10_cats.zip")

# Remove empty strings.
raw_data = raw_data.loc[raw_data["review"].str.len() > 1, ]

# Remove spaces.
raw_data["review"] = raw_data["review"].str.replace(" ", "")

train_df, test_df = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=raw_data["label"])  # pyright: ignore
train_df: pd.DataFrame
test_df: pd.DataFrame

PadTokenizer.build_vocab(train_df["review"].to_list(), config.VOCAB_FILE)

tokenizer = PadTokenizer.from_vocab(config.VOCAB_FILE)

train_input = [tokenizer.encode(x) for x in tqdm(train_df["review"].to_list())]

# Get token length distribution.
print("Training input token length 95% quantile:", pd.Series([len(x) for x in train_input]).quantile(0.95))
# 116.0

print("Use", config.SEQ_LEN, "as input sequence length.")

train_df["target"] = train_df["label"]
train_df["input"] = [tokenizer.encode_to_seq_len(x, config.SEQ_LEN) for x in tqdm(train_df["review"].to_list(), desc="Encode train input")]
test_df["target"] = test_df["label"]
test_df["input"] = [tokenizer.encode_to_seq_len(x, config.SEQ_LEN) for x in tqdm(test_df["review"].to_list(), desc="Encode test input")]

train_df[["input", "target"]].to_json(config.TRAIN_FILE, orient="records", lines=True)
test_df[["input", "target"]].to_json(config.TEST_FILE, orient="records", lines=True)
