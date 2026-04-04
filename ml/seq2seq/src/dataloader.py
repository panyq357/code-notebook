import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):

    def __init__(self, data):
        self.input = torch.tensor(data["input"].tolist(), dtype=torch.long)
        self.target = torch.tensor(data["target"].tolist(), dtype=torch.long)

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


def get_cn_volcabulary():
    
    with open("data/processed/cn_vocab.txt", "r") as f:
        volcabulary = {word.strip(): index for index, word in enumerate(f.readlines())}

    return volcabulary

def get_en_volcabulary():
    
    with open("data/processed/en_vocab.txt", "r") as f:
        volcabulary = {word.strip(): index for index, word in enumerate(f.readlines())}

    return volcabulary


def collate_fn(batch):

    input = pad_sequence([item["input"] for item in batch], batch_first=True, padding_value=0)
    target = pad_sequence([item["target"] for item in batch], batch_first=True, padding_value=0)

    return {"input": input, "target": target}


def get_train_dataloader():
    
    train_dataset = get_train_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    return train_dataloader

def get_test_dataloader():
    
    test_dataset = get_test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    return test_dataloader
