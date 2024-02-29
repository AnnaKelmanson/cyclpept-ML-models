import sys
from pathlib import Path

# Get the absolute path of the preprocessing_modules directory
preprocessing_modules_path = '/home/annaborisova/projects/cyclpept-ML-models/preprocessing_modules'

# Add this path to sys.path to make it available for imports
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
    df_morgan=generate_morgan_count(df)
    df_morgan_moldesc = add_molecular_descriptors(df_morgan)
    df_morgan_moldesc_3d = generate_3d_descriptors(df_morgan_moldesc)
    df_morgan_moldesc_3d.to_csv('descriptors.csv')