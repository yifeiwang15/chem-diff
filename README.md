# Chem-diff
This is the implementation of **Diff-MolGen: controllable molecule generation by diffusion language model.**

Some codes are modified from [minimal-text-diffusion](https://github.com/madaan/minimal-text-diffusion).

Illustration: ![DDPM for chem diff](https://github.com/yifeiwang15/chem-diff/blob/main/ddpm.png)

A DDPM model gradually denoising a hidden continous embedding and then rounding it into a molecule in SMILES format.

## Updates and TODO list
* (02/16/2024) We successfully settled the unconditional generation with good performance in terms of novelty, 
diversity on 3 datasets. The validity is around 0.8, trying to improve it to around 0.9.
Should run more experiments for deciding the hyperparameters 
(model size, diffusion step, traing epochs, max_seq).
* (02/16/2024) Implemented classifier-based guidance module 
(see `./src/controllable/property_optimizer.py` and `./src/controllable/controllable_smiles_generation.py`) for improving qed.
However, this method doesn't work very well and incurs big complexity in generation.
  (Maybe this method inherently not fit molecule generation, maybe not) We will further explore and finetune it. 
* (TODO) To implement classifier-free guidance using strategy of ControlNet and GLIGEN.

## Installation
```linux
conda create -n chem-diff python=3.9
conda activate chem-diff 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
conda install gcc_linux-64
pip3 install mpi4py
pip install blobfile boto3 botocore datasets ftfy huggingface_hub numpy pandas regex requests sacremoses sentencepiece six spacy tokenizers tqdm transformers wandb
```

If your machine only has one GPU, you may encounter running issues of mpi4py package. Here are some alternative ways 
to install mpi4py for single gpu usage. First run `conda install openmpi`, then run `conda install -c conda-forge mpi4py openmpi`.

If you still meet the error like `no module named mpi4py`, try `pip3 install mpi4py` and `conda install gcc_linux-64`.

If you still meet the error like `ImportError: libmpi.so.40: cannot open shared object file: No such file or directory`, try
`install -c conda-forge openmpi=4.1.4=ha1ae619_100` as suggested in (https://github.com/theislab/cellrank/issues/864)

### Quick start after setting up the environment and downloading the data
```linux
bash scripts/run_train.sh moses2 0 False False False 5000
```




## Dataset preparaion.
We tested 3 datasets `gbd13, moses2, guacamol2`. For each dataset, please create a folder under `./data`,
download and place the dataset file in the folder. For example, set the moses2 data in directory `./data/moses2/moses2.csv`.

**Resources for downloading:**
* Download `gdb13.smi` from https://gdb.unibe.ch/downloads/
* The processed Guacamol and MOSES datasets in csv format can be downloaded from this link:
(Processed data provided from [MolGPT](https://github.com/devalab/molgpt).)
https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing

## Unconditional molecule generation
### Training
See the running script `submit_train.sh`. Checkpoints of model weights will be saved in `./ckpts` with prefix `ema_0.9999` xxx.
### Generation
See the running script `submit_generate.sh`. Specify the directory of the saved checkpoints. 
This return a file of generated molecules like
```python
[CLS]O=C(CC1CCC2(CCCCC1)N(C(=O)N1CCOCC2)C1CCC1)C1CC1[SEP][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
[CLS]O=C(CCCc1cccs1)N1CCC(C2CCCO2)CC1[SEP][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
[CLS]Cc1cc(C(=O)N2CCC3(CC2)NC(=O)c2nc(C)c(C)c23)ccc1C[PAD][PAD][SEP][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
[CLS]COC(=O)CCNC(=O)c1ccc2ccccc2cc2c1[SEP][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
```
### Evaluation
The evaluation of generated molecules includes 3 general metrics (validity, novelty, diversity) 
and some molecular property related metrics (logp, qedm, sas). 
All evaluation functions are saved in `utils/metric.py`.

Here is a sample of the summary of generated molecules
```python
-------------------------------------------------------------------------------------
The validity for generated molecules is 0.63
-------------------------------------------------------------------------------------
The novelty for generated molecules is 0.99
-------------------------------------------------------------------------------------
The diversity for generated molecules is 1.00
-------------------------------------------------------------------------------------
                                               Smiles     logp       qed   ses
0     O=C(CC1CCC2(CCCCC1)N(C(=O)N1CCOCC2)C1CCC1)C1CC1      NaN       NaN  None
1                    O=C(CCCc1cccs1)N1CCC(C2CCCO2)CC1  3.48840  0.833171  None
2    Cc1cc(C(=O)N2CCC3(CC2)NC(=O)c2nc(C)c(C)c23)ccc1C      NaN       NaN  None
3                    COC(=O)CCNC(=O)c1ccc2ccccc2cc2c1      NaN       NaN  None
4               Cc1ccc(-c2nc3c(sc4ncnc23)cc(=O)n1)cc1      NaN       NaN  None
..                                                ...      ...       ...   ...
995   Cc1c(C(=O)N2CC(C(=O)c3ccc(F)cc3)CC2)cnn1C1CCCC1  3.79072  0.770703  None
996   Cc1nc(CCNC(=O)N2CCc3nc(C)c(=O)n(C)c(C#N)c32)cs1      NaN       NaN  None
997   CC(=O)Nc1cc(C(=O)NCC(C(=O)OC(C)(C)C)C2CC2)ccc1C  3.05112  0.763798  None
998   CC(=O)c1ccc(C(=O)Nc2ccc(-c3cccc(C#N)c3)cc2O)cc1  4.38578  0.535859  None
999             Cc1cc(C)n(CC(=O)N2CCOCC2C2CCCCC2C)c1C  3.46706  0.849303  None

[1000 rows x 4 columns]
```


## Conditional generation
* classifer-based guidance for optimizing properties like qed. 
See (see `./src/controllable/property_optimizer.py` and `./src/controllable/controllable_smiles_generation.py`)
