import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
import random

from src.utils.mytokenizers import regexTokenizer

logging.basicConfig(level=logging.INFO)

# BAD: this should not be global
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

VALID_CONDITION_NAMES = ['qed', 'logp', 'molwt', 'source', 'HBA', 'HBD', 'SAS', 'TPSA',
       'NumRotBonds', 'scaffold_smiles']

def train_valid_split(df_smiles, ratio=0.8):
    indices = df_smiles.index.tolist()
    random.shuffle(indices)

    split_index = int(len(indices) * ratio)
    
    list1_indices = indices[:split_index]
    list2_indices = indices[split_index:]
    
    df_train = df_smiles.iloc[list1_indices]
    df_val = df_smiles.iloc[list2_indices]
    return df_train, df_val
    
    
def get_dataloader(smiles, tokenizer, batch_size=20, max_length=64, shuffle=False,
                   condition_names = None, corrupt_prob = 0.4):

    """
    :param smiles: input dataframe, where "SMILES" column indicates the smiles string
    :param tokenizer: the tokenizer used to encode the smiles string
    :param batch_size:
    :param max_length:
    :param shuffle:
    :param condition_names: None, or list of condition names
    :return: a generator that yields batches
    """

    if condition_names is not None:
        ## check if condition name is valid and if the input df has such column

        for cond_name in condition_names:
            assert cond_name in VALID_CONDITION_NAMES, "Invalid condition name: {}".format(cond_name)
            assert cond_name in smiles.columns, "Input file does not contain condition: {}".format(cond_name)

    dataset = SMILESDataset(smiles, tokenizer, max_length=max_length,
                            condition_names=condition_names, corrupt_prob=corrupt_prob)

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
    def __init__(self, smiles, tokenizer, max_length=64,
                 condition_names=None, corrupt_prob=0.4):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition_names = condition_names
        self.regexTok = regexTokenizer()
        self.corrupt_prob = corrupt_prob

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles['SMILES'].iloc[idx]
        if random.random() < 0.4:
            _, corrupted_smile = self.regexTok.corrupt_one(smile)  # Ensure this returns the corrupted SMILES string
        else:
            corrupted_smile = smile

        encoding = self.tokenizer.encode_plus(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        corrupted_encoding = self.tokenizer.encode_plus(
            corrupted_smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        dic = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'corrupt_ids': corrupted_encoding['input_ids'].flatten(),
        }

        dic_cond = {}

        if self.condition_names is None:
            return dic
        else:
            for cond in self.condition_names:
                dic_cond[cond] = self.smiles[cond].iloc[idx]
            return dic, dic_cond
