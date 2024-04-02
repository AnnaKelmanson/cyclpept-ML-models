import sys
from pathlib import Path

# Get the path of the current file (e.g., script or module)
current_file_path = Path(__file__)

# Assuming your file structure keeps 'preprocessing_modules' at the same level as 'models'
# Get the root directory of your project (go up two levels from the current file)
root_dir = current_file_path.parent.parent

# Construct the path to the 'preprocessing_modules' directory
preprocessing_modules_path = root_dir / 'preprocessing_modules'

# Add this path to sys.path to make it available for imports
sys.path.append(str(preprocessing_modules_path))

from descriptor_generation import generate_descriptors
from splitter import split_dataset_by_scaffold_and_similarity
from descriptors import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def split_dataframe_X_Y(df):
    df = df.set_index('SMILES')
    df = df.drop('Scaffold', axis = 1)
    df = df.drop('Mean_Tanimoto_Similarity', axis = 1)
    pampa_df = df[['PAMPA']]
    rest_df = df.drop(columns=['PAMPA'])
    return rest_df, pampa_df

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
    return df_encoded, encoders

def decode_columns(df_encoded, encoders):
    """
    Decodes columns in the DataFrame using the provided encoders.
    
    :param df_encoded: DataFrame with encoded columns
    :param encoders: Dictionary of column name to LabelEncoder mappings
    :return: DataFrame with decoded columns
    """
    df_decoded = df_encoded.copy()
    for col, le in encoders.items():
        df_decoded[col] = le.inverse_transform(df_decoded[col])
    return df_decoded

def deleting_irrelevant_columns(df):
    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass
    result_df = df.drop('Mol', axis=1)
    df_labeled = result_df.copy()
    return df_labeled

def encode(df_labeled):
    df_encoded, encoders = encode_columns(df_labeled)
    return df_encoded

def split(df_encoded):
    train, test, val = split_dataset_by_scaffold_and_similarity(df_encoded)
    return train, test, val

def generation_fitting_sets(csv):
    df = pd.read_csv(csv) 
    df = deleting_irrelevant_columns(df)
    df = encode(df)
    train, test, val = split(df)
    X_train, y_train = split_dataframe_X_Y(train)
    X_test, y_test = split_dataframe_X_Y(test)
    X_val, y_val = split_dataframe_X_Y(val)
    return X_train, y_train, X_test, y_test, X_val, y_val

def generate_input_for_model(smiles_list):
    df = pd.DataFrame(smiles_list, columns=['SMILES'])
    df = generate_descriptors(df)
    df = deleting_irrelevant_columns(df)
    df = encode(df)
    df = df.set_index('SMILES')
    #df.columns = df.columns.astype(str)
    return df