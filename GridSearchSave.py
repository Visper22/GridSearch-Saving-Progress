import numpy as np
import pandas as pd
import os
import time
import sys
from typing import Tuple, Union 
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


def evaluate_fold(
    estimator, 
    train_index: NDArray[np.int_], 
    val_index: NDArray[np.int_], 
    X: np.ndarray, 
    y: np.ndarray, 
    early_stop: bool=False
    ) -> Tuple[float, float, float]:
    
    X_train_fold, X_val_fold = X[train_index], X[val_index] 
    y_train_fold, y_val_fold = y[train_index], y[val_index] 

    # If early_stopping
    if early_stop:
        # Split train set into train and evaluation set
        X_train_fold, X_eval, y_train_fold, y_eval = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)
        # Fit model on train fold and use eval_set for early stopping
        estimator.fit(X_train_fold, y_train_fold, eval_set=[(X_eval, y_eval)], verbose=False)
        n_est = len(next(iter(estimator.evals_result().get('validation_0').values())))

    else:
        estimator.fit(X_train_fold, y_train_fold, verbose=False)
        n_est = None
        
    # Predict on train set
    y_pred_train = estimator.predict(X_train_fold)
    train_score = roc_auc_score(y_train_fold, y_pred_train)

    # Predict on test set
    y_pred_val = estimator.predict(X_val_fold)
    val_score = roc_auc_score(y_val_fold, y_pred_val)

    return (n_est, train_score, val_score) if early_stop else (train_score, val_score)

def cross_validation(
    estimator, 
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int=5, 
    early_stop: bool=False
    ) -> Tuple[Union[int, None], float, float]:
    
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    return Parallel(n_jobs=-1)(delayed(evaluate_fold)(estimator, train_index, val_index, X, y, early_stop) for train_index, val_index in kf.split(X, y))



def grid_search(
    estimator, 
    params_grid:dict, 
    X: np.ndarray, 
    y: np.ndarray, 
    path: str, 
    early_stop: bool=False, 
    restart: bool=False
    ) -> None:
    
    if restart:
        if os.path.isfile(path):
            results = pd.read_parquet(path)
            if len(params_grid) <= len(results):
                sys.exit('Search completed')
            index_aux = len(results) - 1
        else:
            sys.exit('No file found')
    else:
        results = pd.DataFrame()
        
    params_grid = ParameterGrid(params_grid)
    
    for i, ps in enumerate(params_grid):
        if restart and i <= index_aux:
            continue
        
        print(f"Evaluating parameter set {i + 1}/{len(params_grid)}: {ps}")
    
        estimator.set_params(**ps)
        result_dict = ps.copy()

        results_list = cross_validation(estimator, X, y, n_splits, early_stop=early_stop)
        
        if early_stop:
            n_estimator, train_scores, val_scores = zip(*results_list)
    
            for split, est in enumerate(n_estimator):
                result_dict[f'estimator_split{split}'] = est
        
            # Compute average score across all folds
            result_dict.update({
                'mean_n_estimator': np.mean(n_estimator),
                'mean_train_score': np.mean(train_scores),
                'mean_val_score': np.mean(val_scores)
            })
        else:
            train_scores, val_scores = zip(*results_list)
        
            # Compute average score across all folds
            result_dict.update({
                'mean_train_score': np.mean(train_scores),
                'mean_val_score': np.mean(val_scores)
            })
        

        results_line = pd.DataFrame(result_dict, index=[0])
        results = pd.concat([results, results_line], ignore_index=True)
        # results.to_parquet(path, compression='gzip')

        if (i + 1) % 5 == 0 or (i + 1) == len(params_grid):
            results.to_parquet(path, compression='gzip')
            
        print(f"Average Train Score: {result_dict['mean_train_score']}")
        print(f"Average Validation Score: {result_dict['mean_val_score']}")


# Create a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Define base parameters
base_params = {
    'nthread': -1,
    'enable_categorical': True,
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'binary:logistic',
    'device': 'cuda',
    'n_estimators': 500,
    'eval_metric': 'auc',
    'early_stopping_rounds': 5,
    'verbosity': 0,
    'seed': 42
}

# Define hyperparameter grid
params_grid = {
    'min_child_weight': [0, 2],
    'gamma': [0, 0.25],
    'reg_lambda': [10, 20],
    'max_depth': [3, 6, 9],
    'eta': [0.01, 0.05]
}

# Configure settings
n_splits = 5
path = 'search_results.gzip'
clf = XGBClassifier(**base_params)

start_time = time.time()  
grid_search(clf, params_grid, X, y, path=path, early_stop=True, restart=False)
print("--- %s seconds ---" % (time.time() - start_time))
