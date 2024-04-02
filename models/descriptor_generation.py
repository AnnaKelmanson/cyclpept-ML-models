import sys
from pathlib import Path

# Get the absolute path of the preprocessing_modules directory
preprocessing_modules_path = '/home/annaborisova/projects/cyclpept-ML-models/preprocessing_modules'

# Add this path to sys.path to make it available for imports
sys.path.append(str(preprocessing_modules_path))
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from preprocessing import preprocess_peptide_data_advanced, cut_df
from splitter import *
from descriptors import *
from rdkit.Chem import Descriptors
from pathlib import Path
models_dir = Path(__file__).parent
root_dir = models_dir.parent
model_path = root_dir / 'fitted_models' / 'random_forest_regressor_model.joblib'
data_path = root_dir / 'data' / 'rf.csv'

def cut_df(df):
    columns = ['SMILES', 'PAMPA']
    return df[columns]


def calculate_hbd(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcNumHBD(mol)

def generate_descriptors(df):
    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['num_hbd'] = df['SMILES'].apply(calculate_hbd)
    df['TPSA'] = df['Mol'].apply(Descriptors.TPSA)
    df['LogP'] = df['Mol'].apply(Descriptors.MolLogP)
    #df_morgan=generate_morgan_count(df)
    #df_morgan_moldesc = add_molecular_descriptors(df_morgan)
    #df_morgan_moldesc_3d = generate_3d_descriptors(df_morgan_moldesc)
    return df

if __name__ == '__main__':
    df = cut_df(preprocess_peptide_data_advanced())
    df = generate_descriptors(df)
    df.to_csv(str(root_dir / 'data' / 'rf.csv'))