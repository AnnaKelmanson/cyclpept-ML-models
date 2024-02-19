import pandas as pd

def preprocess_peptide_data_v1():
    # Load dataset
    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA.csv')
    
    # Remove duplicates based on SMILES column, keeping the latest entry
    unique_smiles_df = df.sort_values('Year', ascending=False).drop_duplicates(subset='SMILES', keep='first')
    
    # Find and separate rows with missing detection limits
    missing_detection_limits_df = unique_smiles_df[pd.isna(unique_smiles_df['Detection_Limit_1']) & pd.isna(unique_smiles_df['Detection_Limit_2'])]
    
    # Identify suspicious rows based on PAMPA values
    suspicious_rows_df = missing_detection_limits_df[missing_detection_limits_df["PAMPA"] == -10.0]
    
    # Combine suspicious rows from both criteria
    non_missing_detection_limits_df = unique_smiles_df[~(pd.isna(unique_smiles_df['Detection_Limit_1']) & pd.isna(unique_smiles_df['Detection_Limit_2']))]
    all_suspicious_df = pd.concat([non_missing_detection_limits_df, suspicious_rows_df], ignore_index=True)
    
    # Filter out the suspicious rows from the main dataset
    clean_df = missing_detection_limits_df[missing_detection_limits_df["PAMPA"] != -10.0].reset_index(drop=True)
    
    # Save outputs
    all_suspicious_df.to_csv('unclear_values_PAMPA_v1.csv', index=False)
    clean_df.to_csv('filtered_PAMPA_v1.csv', index=False)
    
    # Identify and save duplicate entries not in the final dataset
    merged_df = df.merge(unique_smiles_df, how='outer', indicator=True)
    duplicates_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    duplicates_df.to_csv('duplicates_v1.csv', index=False)
    
    return clean_df

def preprocess_peptide_data_v2():
    # Load dataset
    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA.csv')
    
    # Identify unique and non-unique SMILES entries
    smiles_counts = df['SMILES'].value_counts()
    unique_smiles = smiles_counts[smiles_counts == 1].index
    unique_smiles_df = df[df['SMILES'].isin(unique_smiles)]
    non_unique_smiles_df = df[~df['SMILES'].isin(unique_smiles)]
    
    # Calculate standard deviation and mean PAMPA for non-unique SMILES
    stats_df = non_unique_smiles_df.groupby('SMILES')['PAMPA'].agg(['std', 'mean']).reset_index()
    valid_smiles = stats_df[stats_df['std'] <= 1]['SMILES']
    
    # Filter entries based on PAMPA variation
    filtered_non_unique_smiles_df = non_unique_smiles_df[non_unique_smiles_df['SMILES'].isin(valid_smiles)]
    
    # Merge with mean PAMPA values where applicable
    mean_PAMPA_df = stats_df[['SMILES', 'mean']]
    merged_df = filtered_non_unique_smiles_df.merge(mean_PAMPA_df, on='SMILES', how='left', suffixes=('', '_mean'))
    merged_df['PAMPA'] = merged_df['mean'].fillna(merged_df['PAMPA']).drop(columns=['mean'])
    
    # Remove duplicates, keep the latest entry
    final_df = merged_df.sort_values('Year', ascending=False).drop_duplicates(subset='SMILES', keep='first')
    
    # Combine unique and processed non-unique SMILES entries
    combined_df = pd.concat([unique_smiles_df, final_df], ignore_index=True)
    clean_combined_df = combined_df[pd.isna(combined_df['Detection_Limit_1']) & pd.isna(combined_df['Detection_Limit_2'])]
    
    # Remove suspicious PAMPA values
    clean_combined_df = clean_combined_df[clean_combined_df["PAMPA"] != -10.0].reset_index(drop=True)
    
    # Identify and save missing IDs
    missing_ids = ~df['CycPeptMPDB_ID'].isin(clean_combined_df['CycPeptMPDB_ID'])
    missing_rows_df = df[missing_ids]
    missing_rows_df.to_csv('deleted_v2.csv', index=False)
    clean_combined_df.to_csv('filtered_PAMPA_v2.csv', index=False)
    
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