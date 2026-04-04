from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split
import test

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer

outdir = Path(config.PROCESSED_DATA_PATH)

if not outdir.exists():
    outdir.mkdir(parents=True)


with ZipFile("data/raw/AIchallenger2017.zip", 'r') as z:
    with z.open("AIchallenger2017/cmn.txt") as f:
        data = pd.read_table(f, sep="\t", header=None, usecols=[0, 1], names=["en", "cn"], encoding="utf-8") # type: ignore


train_df, test_df = train_test_split(data, test_size=0.2) # type: ignore
train_df: pd.DataFrame
test_df: pd.DataFrame

ChineseTokenizer.build_vocab(train_df["cn"], config.PROCESSED_DATA_PATH + "cn_vocab.txt")
EnglishTokenizer.build_vocab(train_df["en"], config.PROCESSED_DATA_PATH + "en_vocab.txt")

cn_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_PATH + "cn_vocab.txt")
en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_PATH + "en_vocab.txt")

train_data = pd.DataFrame({
    "input": [cn_tokenizer.encode(text, add_sos_eos=False) for text in train_df["cn"]],
    "target": [en_tokenizer.encode(text, add_sos_eos=True) for text in train_df["en"]]
})

test_data = pd.DataFrame({
    "input": [cn_tokenizer.encode(text, add_sos_eos=False) for text in test_df["cn"]],
    "target": [en_tokenizer.encode(text, add_sos_eos=True) for text in test_df["en"]]
})

train_data.to_json(config.PROCESSED_DATA_PATH + "train_data.jsonl", orient="records", lines=True)
test_data.to_json(config.PROCESSED_DATA_PATH + "test_data.jsonl", orient="records", lines=True)
