import sys
from pathlib import Path
import joblib
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
from sklearn.preprocessing import LabelEncoder

model_path = '/home/annaborisova/projects/cyclpept-ML-models/drafts/random_forest_regressor_model.joblib'
def calculate_hbd(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcNumHBD(mol)

def generate_descriptors(df):
    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['num_hbd'] = df['SMILES'].apply(calculate_hbd)
    df['TPSA'] = df['Mol'].apply(Descriptors.TPSA)
    df['LogP'] = df['Mol'].apply(Descriptors.MolLogP)
    df_morgan=generate_morgan_count(df)
    # Additional descriptor generation can be added here
    return df_morgan

def encode_columns(df):
    """
    Encodes object type columns in the DataFrame except for 'SMILES' column.
    
    :param df: DataFrame with columns to encode
    :return: Tuple of (encoded DataFrame, encoders dictionary)
    """
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col != 'SMILES':  # Skip the 'SMILES' column
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le  # Store the encoder
    return df_encoded

def make_predictions(smiles_list):

    loaded_clf = joblib.load(model_path)
    
    df = pd.DataFrame(smiles_list, columns=['SMILES'])
    

    df = generate_descriptors(df)
    df = encode_columns(df)
    # Select the columns that the model was trained on
    # Assuming the model was trained on 'num_hbd', 'TPSA', and 'LogP' features
    features = df.drop(columns=['SMILES'])
    
    # Make predictions using the loaded model
    predictions = loaded_clf.predict(features)
    
    return predictions

# Example usage:
if __name__ == '__main__':
    # Example list of SMILES strings
    smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'C1CCCCC1']
    
    # Make predictions
    predicted_values = make_predictions(smiles_list)
    
    # Print the predicted values
    print(predicted_values)
