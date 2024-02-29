import sys
from pathlib import Path
preprocessing_modules_path = '/home/annaborisova/projects/cyclpept-ML-models/preprocessing_modules'
sys.path.append(str(preprocessing_modules_path))
from preprocessing import preprocess_peptide_data_advanced, cut_df
from splitter import *
from descriptors import *
from rdkit.Chem import Descriptors


def cut_df(df):
    columns = ['SMILES']
    return df[columns]


if __name__ == '__main__':
    df = cut_df(preprocess_peptide_data_advanced())
    chameleon = get_chameleonicity_like_descriptor(df)
    chameleon.to_csv('chameleonicity.csv')