import pandas as pd
from rdkit import Chem
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter, ButinaSplitter
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MurckoScaffoldSmilesFromSmiles
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from rdkit.Chem import Descriptors
import numpy as np
from scipy.spatial.distance import cdist
from rdkit.Chem import rdMolDescriptors
from concurrent.futures import ProcessPoolExecutor

def preprocess_peptide_data_basic():
    '''if there are duplicates, it takes the assay from the latest paper'''
    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA.csv')
    unique_smiles_df = df.sort_values('Year', ascending=False).drop_duplicates(subset='SMILES', keep='first')
    missing_detection_limits_df = unique_smiles_df[pd.isna(unique_smiles_df['Detection_Limit_1']) & pd.isna(unique_smiles_df['Detection_Limit_2'])]
    suspicious_rows_df = missing_detection_limits_df[missing_detection_limits_df["PAMPA"] == -10.0]
    non_missing_detection_limits_df = unique_smiles_df[~(pd.isna(unique_smiles_df['Detection_Limit_1']) & pd.isna(unique_smiles_df['Detection_Limit_2']))]
    all_suspicious_df = pd.concat([non_missing_detection_limits_df, suspicious_rows_df], ignore_index=True)
    clean_df = missing_detection_limits_df[missing_detection_limits_df["PAMPA"] != -10.0].reset_index(drop=True)
    # all_suspicious_df.to_csv('unclear_values_PAMPA_v1.csv', index=False)
    # clean_df.to_csv('filtered_PAMPA_v1.csv', index=False)
    merged_df = df.merge(unique_smiles_df, how='outer', indicator=True)
    duplicates_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    # duplicates_df.to_csv('duplicates_v1.csv', index=False)
    
    return clean_df

def preprocess_peptide_data_advanced():
    '''if there are duplicates, it checks first std, if it is more than 1 it deletes the assay, if below it takes mean'''
    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA.csv')
    
    smiles_counts = df['SMILES'].value_counts()
    unique_smiles = smiles_counts[smiles_counts == 1].index
    unique_smiles_df = df[df['SMILES'].isin(unique_smiles)]
    non_unique_smiles_df = df[~df['SMILES'].isin(unique_smiles)]
    
    stats_df = non_unique_smiles_df.groupby('SMILES')['PAMPA'].agg(['std', 'mean']).reset_index()
    valid_smiles = stats_df[stats_df['std'] <= 1]['SMILES']

    filtered_non_unique_smiles_df = non_unique_smiles_df[non_unique_smiles_df['SMILES'].isin(valid_smiles)]
    
    mean_PAMPA_df = stats_df[['SMILES', 'mean']]
    merged_df = filtered_non_unique_smiles_df.merge(mean_PAMPA_df, on='SMILES', how='left', suffixes=('', '_mean'))
    merged_df['PAMPA'] = merged_df['mean'].fillna(merged_df['PAMPA']).drop(columns=['mean'])
    
    final_df = merged_df.sort_values('Year', ascending=False).drop_duplicates(subset='SMILES', keep='first')
    
    combined_df = pd.concat([unique_smiles_df, final_df], ignore_index=True)
    clean_combined_df = combined_df[pd.isna(combined_df['Detection_Limit_1']) & pd.isna(combined_df['Detection_Limit_2'])]
    
    clean_combined_df = clean_combined_df[clean_combined_df["PAMPA"] != -10.0].reset_index(drop=True)
    
    missing_ids = ~df['CycPeptMPDB_ID'].isin(clean_combined_df['CycPeptMPDB_ID'])
    missing_rows_df = df[missing_ids]
    # missing_rows_df.to_csv('deleted_v2.csv', index=False)
    # clean_combined_df.to_csv('filtered_PAMPA_v2.csv', index=False)
    
    return clean_combined_df


def cut_df(df):
    columns = ['CycPeptMPDB_ID',
    'Source',
    'Year',
    'Original_Name_in_Source_Literature',
    'Structurally_Unique_ID',
    'Same_Peptides_ID',
    'Same_Peptides_Source',
    'Same_Peptides_Permeability',
    'Same_Peptides_Assay',
    'SMILES',
    'HELM',
    'HELM_URL',
    'Sequence',
    'Sequence_LogP',
    'Sequence_TPSA',
    'Monomer_Length',
    'Monomer_Length_in_Main_Chain',
    'Molecule_Shape',
    'Permeability',
    'PAMPA',
    'Caco2',
    'MDCK',
    'RRCK',
    'Detection_Limit_1',
    'Detection_Limit_2']

    return df[columns]

def check_validity(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        print('invalid SMILES')
        return False
    else:
        try:
            Chem.SanitizeMol(m)
            #print('valid SMILES and chemistry')
            return True
        except:
            print('invalid chemistry')
            return False
        
# for mol in df['SMILES']:
#     check_validity(mol)
        
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


def generate_morgan_count(df, smiles_col='SMILES', radius=2):
    """
    Generates Morgan count descriptors for a dataframe containing SMILES structures.

    Parameters:
    - df: DataFrame containing SMILES strings.
    - smiles_col: Name of the column containing SMILES strings.
    - radius: The radius of the Morgan fingerprints.

    Returns:
    - DataFrame with Morgan count descriptors added.
    """

    morgan_descriptors = []

    for index, row in df.iterrows():

        mol = Chem.MolFromSmiles(row[smiles_col])
        
        fp = AllChem.GetMorganFingerprint(mol, radius=radius, useCounts=True)
        
        fp_dict = fp.GetNonzeroElements()
        
        morgan_descriptors.append(fp_dict)
    
    fp_df = pd.DataFrame(morgan_descriptors)
    fp_df = fp_df.fillna(0).astype(int)
    result_df = pd.concat([df, fp_df], axis=1)
    
    return result_df

def generate_3d_descriptors(df, smiles_col):
    """
    Function to generate 3D descriptors for a DataFrame of SMILES strings.
    
    Parameters:
    - df: DataFrame containing the SMILES strings.
    - smiles_col: The name of the column containing the SMILES strings.
    
    Returns:
    - DataFrame with the original data and new columns for WHIM descriptors.
    """
    
    # Define a helper function to process each SMILES string
    def process_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)  
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)  
            AllChem.MMFFOptimizeMolecule(mol)  
            whim_descriptors = CalcWHIM(mol)  
            return whim_descriptors
        except:
            return [None] * 114  
    
    whim_results = df[smiles_col].apply(process_smiles)
    

    whim_df = pd.DataFrame(whim_results.tolist(), columns=[f'WHIM_{i}' for i in range(114)])
    result_df = pd.concat([df, whim_df], axis=1)
    
    return result_df

def add_molecular_descriptors(df, smiles_col):
    """
    Function to add molecular descriptors for a DataFrame of SMILES strings using RDKit.
    
    Parameters:
    - df: DataFrame containing the SMILES strings.
    - smiles_col: The name of the column containing the SMILES strings.
    
    Returns:
    - DataFrame with the original data and new columns for molecular descriptors.
    """

    def calculate_descriptors(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:  
                return [None] * len(Descriptors._descList)  
            return Descriptors.CalcMolDescriptors(mol)
        except:
            return [None] * len(Descriptors._descList)  
    
    descriptor_results = df[smiles_col].apply(calculate_descriptors)
    
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    
    descriptors_df = pd.DataFrame(descriptor_results.tolist(), columns=descriptor_names)
    result_df = pd.concat([df, descriptors_df], axis=1)
    
    return result_df

def parallelize_dataframe_computation(df, function, n_workers=4):
    smiles_list = df['SMILES'].to_list()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(function, smiles_list))
    return results

def IMHB_var(smiles):
    def find_intramolecular_hbonds(mol, confId=-1, eligibleAtoms=[7,8], distTol=2.5):
        """
        mol: RDKit molecule object
        confId: Conformer ID to use for distance calculation
        eligibleAtoms: List of atomic numbers for eligible H-bond donors or acceptors (N and O by default)
        distTol: Maximum accepted distance (in Ångströms) for an H-bond
        """
        res = []
        conf = mol.GetConformer(confId)
        for i in range(mol.GetNumAtoms()):
            atomi = mol.GetAtomWithIdx(i)
            if atomi.GetAtomicNum() == 1:  
                for j in range(mol.GetNumAtoms()):
                    atomj = mol.GetAtomWithIdx(j)
                    if atomj.GetAtomicNum() in eligibleAtoms:
                        d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                        if d <= distTol:
                            res.append((i, j, d))
        return res

    def calc_rotatable_bonds(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  
            return rdMolDescriptors.CalcNumRotatableBonds(mol)
        else:
            return None

    def number_of_conformers(num_of_rotatable_bonds):
        if num_of_rotatable_bonds<=7:
            number_of_conformers=50
        elif num_of_rotatable_bonds>=8 and num_of_rotatable_bonds<=12:
            number_of_conformers=200
        else:
            number_of_conformers=300
        return number_of_conformers

    def cluster_conformers(mol, threshold=1.5):
        # print("Performing energy minimization and sorting conformers by energy...")
        conformers = mol.GetConformers()
        energies = []
        for conf in conformers:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
            ff.Minimize()
            energy = ff.CalcEnergy()
            energies.append((conf.GetId(), energy))
        energies.sort(key=lambda x: x[1])

        # print(f"Lowest energy conformer ID: {energies[0][0]}, Energy: {energies[0][1]}")
        
        keep = [energies[0][0]]
        
        # print("Clustering conformers based on RMSD threshold...")
        for index, (conf_id, energy) in enumerate(energies[1:]):
            # print(f"Analyzing conformer {conf_id} with energy {energy}...")
            unique = True
            for kept_id in keep:
                # print(f"  Comparing against conformer {kept_id}...")
                rmsd = AllChem.AlignMol(mol, mol, prbCid=conf_id, refCid=kept_id)
                # print(f"  RMSD between conformer {conf_id} and {kept_id}: {rmsd}")
                if rmsd < threshold:
                    unique = False
                    # print(f"  Conformer {conf_id} is similar to conformer {kept_id} (RMSD: {rmsd}). Discarding.")
                    break
            if unique:
                keep.append(conf_id)
                # print(f"  Conformer {conf_id} is unique and kept (Energy: {energy}).")
        
        # print(f"Total unique conformers after clustering: {len(keep)}")
        return keep



    # print("Starting hydrogen bond analysis...")
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_rotatable_bonds = calc_rotatable_bonds(smiles)
    num_conformers = number_of_conformers(num_rotatable_bonds)
    # print(f"Generating {num_conformers} conformers...")
    _ = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, pruneRmsThresh=1, randomSeed=1)
    
    # print("Clustering conformers based on RMSD...")
    keep = cluster_conformers(mol)
    # print(f"{len(keep)} unique conformers identified after clustering.")
    
    hbond_lengths = []  
    for confId in keep:
        hbonds = find_intramolecular_hbonds(mol, confId=confId)
        hbond_lengths.append(len(hbonds))  
    
    return max(hbond_lengths)-min(hbond_lengths) 

def get_chameleonicity_like_descriptor(df):
    results = parallelize_dataframe_computation(df, IMHB_var)
    df['IMHB_var'] = results
    return df