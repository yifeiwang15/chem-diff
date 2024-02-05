import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
import random

logging.basicConfig(level=logging.INFO)

# BAD: this should not be global
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def train_valid_split(df_smiles, ratio=0.8):
    indices = df_smiles.index.tolist()
    random.shuffle(indices)

    split_index = int(len(indices) * ratio)
    
    list1_indices = indices[:split_index]
    list2_indices = indices[split_index:]
    
    df_train = df_smiles.iloc[list1_indices]
    df_val = df_smiles.iloc[list2_indices]
    return df_train, df_val
    
    
def get_dataloader(smiles, tokenizer, batch_size=20, max_length=64, shuffle=False):
    dataset = SMILESDataset(smiles, tokenizer, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    # return dataloader
    
    while True:
        for batch in dataloader:
            yield batch


class SMILESDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_length=64):
        self.smiles = smiles # a list of smiles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]

        encoding = self.tokenizer.encode_plus(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
