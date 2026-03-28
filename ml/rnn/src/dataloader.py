import torch
import pandas as pd

from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data):
        self.input = torch.tensor(data["input"].tolist())
        self.target = torch.tensor(data["target"].tolist())

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {"input": self.input[idx,], "target": self.target[idx]}


def get_test_dataset():
    
    data = pd.read_json("data/processed/test_data.jsonl", lines=True)

    dataset = MyDataset(data)

    return dataset


def get_train_dataset():

    data = pd.read_json("data/processed/train_data.jsonl", lines=True)

    dataset = MyDataset(data)

    return dataset


def get_volcabulary():
    
    with open("data/processed/volcabulary.txt", "r") as f:
        volcabulary = {word.strip(): index for index, word in enumerate(f.readlines())}

    return volcabulary

