from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from .preparation_for_training import generation_fitting_sets

from pathlib import Path

# Assuming this file is directly inside the 'models' directory
models_dir = Path(__file__).parent
root_dir = models_dir.parent
model_path = root_dir / 'fitted_models' / 'random_forest_regressor_model.joblib'
data_path = root_dir / 'data' / 'rf.csv'


def evaluate_model(clf, X_train, y_train, X_val, y_val):
    """
    Evaluates the RandomForestRegressor model on training and validation datasets.
    
    :param clf: The trained RandomForestRegressor model
    :param X_train: Training features
    :param y_train: Training labels
    :param X_val: Validation features
    :param y_val: Validation labels
    """
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    
    # Training set performance
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    
    # Validation set performance
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    
    # Cross-validation
    scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')
    cv_mse = -np.mean(scores)
    
    print(f"Training MSE: {mse_train}, R^2: {r2_train}")
    print(f"Validation MSE: {mse_val}, R^2: {r2_val}")
    print(f"Cross-Validation MSE: {cv_mse}")

def fit_and_save_model(model_path):
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    csv = data_path
    X_train, y_train, X_test, y_test, X_val, y_val = generation_fitting_sets(csv)
    clf.fit(X_train, y_train.values.ravel())
    joblib.dump(clf, model_path)

# Later on, in another script or after restarting the kernel, load the model
#loaded_clf = joblib.load('random_forest_regressor_model.joblib')

if __name__ == '__main__':
    fit_and_save_model(model_path)