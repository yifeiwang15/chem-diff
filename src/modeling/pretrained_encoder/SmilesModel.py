import torch
from torch import nn
import numpy as np
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaConfig

# Modified from https://github.com/Qihoo360/CReSS/blob/master/model/model_smiles.py

class SmilesEncoder(nn.Module):
    def __init__(self,
                 roberta_tokenizer_path='src/modeling/pretrained_encoder/PretrainWeights/tokenizer-smiles-roberta-1e',
                 device=torch.device('cpu'),
                 smiles_maxlen=100,
                 vocab_size=181,
                 max_position_embeddings=505,
                 num_attention_heads=12,
                 num_hidden_layers=6,
                 type_vocab_size=1,
                 feature_dim=768,
                 **kwargs
                 ):
        super(SmilesEncoder, self).__init__(**kwargs)
        self.smiles_maxlen = smiles_maxlen
        self.feature_dim = feature_dim
        self.smiles_tokenizer = RobertaTokenizer.from_pretrained(
                roberta_tokenizer_path, max_len=self.smiles_maxlen)
        self.device = device
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            hidden_size=self.feature_dim
        )

        self.model = RobertaModel(config=self.config).to(self.device)
        self.dense = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, input):

        ## preprocess smiles tokenization in dataset collate_fn
        smiles_ids = []
        smiles_mask = []
        for smiles in input:
            if isinstance(smiles, float):
                if np.isnan(smiles):
                    smiles = ''
                else:
                    raise ValueError("Invalid smiles value" + str(smiles))
            elif smiles is None or smiles == '' or len(smiles) == 0:
                smiles = ''
            encode_dict = self.smiles_tokenizer.encode_plus(
                text=smiles,
                max_length=self.smiles_maxlen,
                padding='max_length',
                truncation=True)
            smiles_ids.append(encode_dict['input_ids'])
            smiles_mask.append(encode_dict['attention_mask'])

        smiles_ids = torch.tensor(smiles_ids).to(self.device)
        smiles_mask = torch.tensor(smiles_mask).to(self.device)
        with torch.no_grad():
            outs = self.model(smiles_ids, smiles_mask)['last_hidden_state']
            # outs = self.dense(outs)
            # outs = outs / outs.norm(dim=-1, keepdim=True)
        return outs, smiles_mask
