import pandas as pd
from rdkit import Chem
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter, ButinaSplitter
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MurckoScaffoldSmilesFromSmiles
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from rdkit.Chem import Descriptors
import numpy as np
from scipy.spatial.distance import cdist
from rdkit.Chem import rdMolDescriptors
from concurrent.futures import ProcessPoolExecutor


def split_dataset_by_scaffold_and_similarity(df):
    '''Based on scaffold and stratification: it takes the least similar scaffold and put aside in validation set, 
        the rest split with taking into account distribution of scaffolds, the single instances goes to train'''
    
    df['Scaffold'] = df['SMILES'].apply(MurckoScaffoldSmilesFromSmiles)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=1024) for smi in df["SMILES"]]

    mean_tan_sim = []

    for i, fp in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:i] + fingerprints[i+1:])
        mean_similarity = sum(similarities) / len(similarities)
        mean_tan_sim.append(mean_similarity)

    df['Mean_Tanimoto_Similarity'] = mean_tan_sim

    df_sorted = df.sort_values(by='Mean_Tanimoto_Similarity')
    validation_set = df_sorted.iloc[:int(len(df) * 0.1)]

    df_remaining = df_sorted.iloc[int(len(df) * 0.1):]

    scaffold_counts = df_remaining['Scaffold'].value_counts()
    single_instance_scaffolds = scaffold_counts[scaffold_counts == 1].index
    multi_instance_df = df_remaining[~df_remaining['Scaffold'].isin(single_instance_scaffolds)]
    single_instance_df = df_remaining[df_remaining['Scaffold'].isin(single_instance_scaffolds)]

    train_size = 0.8 / (0.8 + 0.1) 

    train_set, test_set = train_test_split(multi_instance_df, 
                                       test_size=1-train_size, 
                                       stratify=multi_instance_df['Scaffold'],
                                       random_state=42)


    train_set = pd.concat([train_set, single_instance_df])
    return train_set, test_set, validation_set

def random_splitter(df):

    train_df, temp_test_df = train_test_split(df, train_size=0.8, random_state=42, shuffle=True)

    val_df, test_df = train_test_split(temp_test_df, test_size=0.5, random_state=42, shuffle=True)

    return train_df, test_df, val_df


def kennard_stone_percentage_split(df, smiles_col, percentage):
    """
    Splits the dataset into two DataFrames using the Kennard-Stone algorithm,
    automatically identifying descriptor columns and based on a percentage.
    
    Parameters:
    - df: DataFrame containing the dataset with descriptors and a SMILES column.
    - smiles_col: The name of the column containing the SMILES strings.
    - percentage: The percentage of samples to select.
    
    Returns:
    - selected_df: DataFrame with the selected samples.
    - remaining_df: DataFrame with the remaining samples.
    """
    
    # Identify descriptor columns (numeric columns excluding the SMILES column)
    descriptor_cols = df.select_dtypes(include=[np.number]).columns.drop(smiles_col, errors='ignore')
    
    # Extract the descriptor matrix
    x_variables = df[descriptor_cols].values
    original_x = x_variables.copy()
    
    # Calculate the number of samples to select based on the percentage
    num_samples = int(len(df) * percentage)
    
    # Calculate the Euclidean distance to the mean
    distance_to_average = np.sum((x_variables - np.mean(x_variables, axis=0))**2, axis=1)
    max_distance_index = np.argmax(distance_to_average)
    selected_indices = [max_distance_index]
    
    for _ in range(1, num_samples):
        remaining_indices = list(set(range(len(x_variables))) - set(selected_indices))
        selected_descriptors = original_x[selected_indices, :]
        remaining_descriptors = original_x[remaining_indices, :]
        
        distances = cdist(remaining_descriptors, selected_descriptors, 'euclidean')
        min_distances = np.min(distances, axis=1)
        max_min_distance_index = np.argmax(min_distances)
        
        selected_indices.append(remaining_indices[max_min_distance_index])
    
    # Create DataFrames for selected and remaining samples
    selected_df = df.iloc[selected_indices].reset_index(drop=True)
    remaining_indices = list(set(range(len(df))) - set(selected_indices))
    remaining_df = df.iloc[remaining_indices].reset_index(drop=True)
    
    return selected_df, remaining_df

def check_leakage(train_set, test_set, validation_set):
    train_indices = set(train_set.index)
    test_indices = set(test_set.index)
    validation_indices = set(validation_set.index)

    train_test_overlap = train_indices.intersection(test_indices)
    train_validation_overlap = train_indices.intersection(validation_indices)
    test_validation_overlap = test_indices.intersection(validation_indices)

    # Print the results
    print("Overlap between train and test sets:", train_test_overlap)
    print("Overlap between train and validation sets:", train_validation_overlap)
    print("Overlap between test and validation sets:", test_validation_overlap)

    assert len(train_test_overlap) == 0, "There is an overlap between train and test sets."
    assert len(train_validation_overlap) == 0, "There is an overlap between train and validation sets."
    assert len(test_validation_overlap) == 0, "There is an overlap between test and validation sets."

    print("All sets are unique with no overlap.")