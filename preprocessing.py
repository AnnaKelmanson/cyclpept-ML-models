import pandas as pd
from rdkit import Chem
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter, ButinaSplitter
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MurckoScaffoldSmilesFromSmiles
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
from sklearn.model_selection import train_test_split

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