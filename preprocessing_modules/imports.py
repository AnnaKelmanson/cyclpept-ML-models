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