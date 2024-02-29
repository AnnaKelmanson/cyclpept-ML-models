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
from pandarallel import pandarallel
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import mapply


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

def generate_3d_descriptors(df, smiles_col='SMILES'):
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

def add_molecular_descriptors(df, smiles_col='SMILES'):
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

# def parallelize_dataframe_computation(df, function, n_workers=24):
#     smiles_list = df['SMILES'].to_list()
#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         results = list(executor.map(function, smiles_list))
#     return results

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
    pandarallel.initialize(progress_bar=True)
    df['IMHB_var'] = df['SMILES'].parallel_apply(IMHB_var)
    return df



