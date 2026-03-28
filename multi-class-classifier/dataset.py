# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import HEADS, TEXT_COLUMN

class MultiHeadDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_length=256):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_columns = [name for name, _ in HEADS]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row[TEXT_COLUMN],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = []
        for col in self.label_columns:
            labels.append(int(row[col]))
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }