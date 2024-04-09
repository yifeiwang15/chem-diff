import os
import sys
from typing import List, Union

from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit import RDLogger
import pandas as pd
RDLogger.DisableLog('rdApp.*')
import argparse


class Evaluation_metric:
    def __init__(self, file_path, train_set_path, out_dir):
        # 构造器方法，初始化实例变量
        self.file_path = file_path
        self.train_set_path = train_set_path
        self.gen = None
        self.train_data = None
        print(out_dir)
        self.out_dir = out_dir
        self.randomized_data = None  # Container for the randomized data

    def clean_smiles(self):
        self.train_data = pd.read_csv(self.train_set_path)
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        cleaned_smiles = []
        for line in lines:
            # 移除 '[CLS]', '[SEP]' 和所有 '[PAD]' 符号
            clean_line = line.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            cleaned_smiles.append(clean_line)

        

        self.gen = cleaned_smiles   

    def load_randomized_smiles(self, randomized_path):
        # Load randomized SMILES and append to the train data for novelty calculation
        self.randomized_data = pd.read_csv(randomized_path)

       
    def calc_novelty(self):
        """
        Calculates the novelty score of the generated molecule list.
        the novelty score is the ratio between the amount of molecules that aren't in the train set
        and the size of the entire generate molecule list.

        |gen_molecules - train_set_molecules|
        ---------------------------
            |gen_molecules|

        Args:
            train_path:
                The data used to train the model with or a list of the loaded molecdules

            generated_molecules:
                List of molecuels that were generated by the model, molecules are in SMILES form.

        Returns:
            The novelty score of the generated set.
            novlty score ranges between 0 and 1.
        """
        if self.randomized_data is not None:
            # Combine train data with randomized data if available
            combined_data = pd.concat([self.train_data, self.randomized_data], ignore_index=True)
        else:
            combined_data = self.train_data

        train_set = set(combined_data['SMILES']) 
        generated_molecules = set(self.gen)
        new_molecules = generated_molecules - train_set
        new_molecules = len(new_molecules)

        novelty_score = new_molecules / len(generated_molecules)
        return novelty_score

    def calc_diversity(self):
        """
        Calculates the diversity of the generate molecule list.
        The diversity is the number of unique molecules in the generated list.

        |unqiue(gen_molecules)|
        ------------------
            |gen_molecules|

        Args:
            gen_molecules:
                List of molecules that werte generated by the model, molecules are in SMILES form.

        Returns:
            The diversity score of the generated set.
            diversity score ranges between 0 and 1.
        """ 
        return len(set(self.gen)) / len(self.gen)

    def calc_logp(self, smile) -> float:
        """
        Calculates the logP for a given molecule.

        Args:
            mol:
                An rdkit molecule object.

        Returns:
            the molecule log p which is his the log ratio between
            water solubility and octanol solubility.
        """
        mol = Chem.MolFromSmiles(smile)
        if mol:  # 确保分子被正确解析
            log_p = Crippen.MolLogP(mol)
            return log_p
        else:
            return None
    def calc_qed(self, smile) -> float:
        """
        Calculates the quantitative estimation of drug-likeness of a given molecule.

        Args:
            mol:
                An rdkit molecule object.

        Returns:
            the molecule qed value which estimate how much this molecule
            resemebles a drug.
        """
        mol = Chem.MolFromSmiles(smile)
        if mol:  # 确保分子被正确解析
            qed = QED.qed(mol)
            return qed
        else:
            return None
    def calc_sas(self, smile) -> float:
        """
        Calculates the Synthetic Accessiblity Score (SAS) of a drug-like molecule
        based on the molecular compelxity and fragment contribution.
        
        code taken from: https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py

        Args:
            A rdkit molecule object.

        Returns:
            The SAS score of a given molecule.

        Raises

        """
        mol = Chem.MolFromSmiles(smile)
        try:
            sascore = sascorer.calculateScore(mol)
            return sascore
        except Exception:
            return None
    def calc_valid_molecules(self) -> float:
        valid_molecules = [mol for mol in self.gen if Chem.MolFromSmiles(mol) is not None]
        return len(valid_molecules) / len(self.gen)

    def get_all_metrics(self):
        with open(self.out_dir, 'w') as file:
            print("-------------------------------------------------------------------------------------", file=file)
            validity = self.calc_valid_molecules()
            print("The validity for generated molecules is {:.4f}".format(validity), file=file)
            print("-------------------------------------------------------------------------------------", file=file)
            novlty = self.calc_novelty()
            print("The nolvelty for generated molecules is {:.4f}".format(novlty), file=file)
            print("-------------------------------------------------------------------------------------", file=file)
            diversity = self.calc_diversity()
            print("The diversity for generated molecules is {:.2f}".format(diversity), file=file)
            print("-------------------------------------------------------------------------------------", file=file)
            
            cleaned_smiles_pd = pd.DataFrame(self.gen, columns=['Smiles'])

            
            cleaned_smiles_pd['logp'] = cleaned_smiles_pd['Smiles'].apply(lambda x: self.calc_logp(x))
            cleaned_smiles_pd['qed'] = cleaned_smiles_pd['Smiles'].apply(lambda x: self.calc_qed(x))
            cleaned_smiles_pd['ses'] = cleaned_smiles_pd['Smiles'].apply(lambda x: self.calc_sas(x))
            print(cleaned_smiles_pd, file=file)
            cleaned_smiles_pd.to_csv(self.out_dir + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', type=str, required = True, help="name of the trained dataset")
    parser.add_argument('--gen_path', type=str, required = True, help="name of the generated dataset")
    parser.add_argument('--out_dir', type=str, required = True, help="name of the our dir")
    parser.add_argument('--randomized_path', type=str, required=False, help="path to the randomized smiles file")
    
    args = parser.parse_args()

    eval_metric = Evaluation_metric(args.gen_path, args.train_set_path, args.out_dir)
    eval_metric.clean_smiles()
    eval_metric.load_randomized_smiles(args.randomized_path)  # Load and add the randomized smiles
    
    eval_metric.get_all_metrics()
    print("finished")

 

