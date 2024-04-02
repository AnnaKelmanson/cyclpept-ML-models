import joblib
import pandas as pd
from .preparation_for_training import generate_input_for_model
from pathlib import Path

models_dir = Path(__file__).parent
root_dir = models_dir.parent
model_path = root_dir / 'fitted_models' / 'random_forest_regressor_model.joblib'
data_path = root_dir / 'data' / 'rf.csv'


def make_predictions(smiles_list, model_path=model_path):
    loaded_clf = joblib.load(model_path)
    df = generate_input_for_model(smiles_list)
    predictions = loaded_clf.predict(df)
    
    return predictions

def return_pandas_df(smiles_list, predictions):
    result_df = pd.DataFrame({
    'SMILES': smiles_list,
    'Predicted_Value': predictions})
    return result_df

if __name__=='__main__':
    smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'C1CCCCC1']
    predictions = make_predictions(smiles_list)
    #df = return_pandas_df(smiles_list, predictions)
    print(predictions)