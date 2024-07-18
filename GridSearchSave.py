# Basic Import
import numpy as np
import pandas as pd
import os
import copy
# Modelling
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score


# Create a synthetic dataset
X, y = make_classification(n_samples=100000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Configure cross-validation and early stopping
n_splits = 5
early_stopping_rounds = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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
params_grid = ParameterGrid({
    'min_child_weight':[0, 2, 4],
    'gamma':[0, 0.25, 1],
    'reg_lambda':[10, 20, 100],
    'max_depth': [3, 6, 9],
    'eta': [0.01, 0.05, 0.1]
})

# Define restart = False when starting from the beggining
restart = True
path = 'search_results.gzip'
index = 0

# Perform cross-validation with early stopping
clf = XGBClassifier()

if restart == True:
    if os.path.isfile(path) == True:
        results = pd.read_parquet(path)

        if len(params_grid) > len(results):
            index_aux = len(results) - 1
        else:
            print('Search completed')         
    else:
        print('No file found')
else:
    results = pd.DataFrame()

for i, ps in enumerate(params_grid):
    if 'index_aux' in locals():
        if i <= index_aux:
            index = index + 1
            continue
    print(i)
    print(ps)
    
    val = copy.deepcopy(base_params)
    val.update(ps)
    for param in ps:
        results.loc[index, param] = ps[param]
        train_scores = []
        val_scores = []
        n_estimator = []

    for train_index, val_index in kf.split(X, y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Split train set into train and evaluation set
        X_train_fold, X_eval, y_train_fold, y_eval = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)

        # Prepare the model
        model = clf.set_params(**val)

        # Fit model on train fold and use eval_set for early stopping
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_eval, y_eval)], verbose=False)
        n_est = len(model.evals_result().get('validation_0').get('auc'))
        n_estimator.append(n_est)
        #train_scores.append(score)

        # Predict on train set
        y_pred_train = model.predict(X_train_fold)
        train_score = roc_auc_score(y_train_fold, y_pred_train)
        train_scores.append(train_score)
        # Predict on test set
        y_pred_val = model.predict(X_val_fold)
        val_score = roc_auc_score(y_val_fold, y_pred_val)
        val_scores.append(val_score)

    # Compute average score across all folds
    
    mean_train_score = np.mean(train_scores)
    mean_val_score = np.mean(val_scores)
    mean_n_estimator = np.mean(n_estimator)

    split=0
    for est in range(len(n_estimator)):
        results.loc[index, 'estimator_split'+str(split)] = n_estimator[est]
        split = split + 1
    results.loc[index, 'mean_n_estimator'] = mean_n_estimator
    results.loc[index, 'mean_train_score'] = mean_train_score
    results.loc[index, 'mean_val_score'] = mean_val_score
    results.to_parquet(path, compression='gzip')
    index = index + 1
    print(f"Average Train Score: {mean_train_score}")
    print(f"Average Validation Score: {mean_val_score}")