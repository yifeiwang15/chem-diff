import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
import random
from rdkit import Chem
import csv

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
    
    
def get_dataloader(smiles, tokenizer, dataset, batch_size=20, max_length=64, shuffle=False,
                   condition_names = None, augment_prob=0.):

    """
    :param smiles: input dataframe, where "SMILES" column indicates the smiles string
    :param tokenizer: the tokenizer used to encode the smiles string
    :param batch_size:
    :param max_length:
    :param shuffle:
    :param condition_names: None, or list of condition names
    :return: a generator that yields batches
    """

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
                 condition_names=None, augment_prob = 0.):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition_names = condition_names
        self.augment_prob = augment_prob
        self.randomized_smiles = []

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles['SMILES'].iloc[idx]

        #randomly select equivalent smile hopefully to increase novelty
        if random.random() < self.augment_prob:
            smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile), doRandom=True)
            self.randomized_smiles.append(smile)

        encoding = self.tokenizer.encode_plus(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        dic = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        dic_cond = {}

        if self.condition_names is None:
            return dic
        else:
            for cond in self.condition_names:
                dic_cond[cond] = self.smiles[cond].iloc[idx]
            return dic, dic_cond

    def write_randomized_smiles_to_csv(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])  # Writing header
            for smile in self.randomized_smiles:
                writer.writerow([smile])

