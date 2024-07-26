# Basic import
import numpy as np
import pandas as pd
import os
import time
import random
import sys
from typing import Tuple, Union, List, Dict
from numpy.typing import NDArray
# ML import
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import get_scorer
# Parallel processing import
from joblib import Parallel, delayed


class CrossValidator:
    def __init__(self, estimator, scoring: List[str]=['roc_auc'], n_splits: int=5, early_stop: bool=False):
        self.estimator = estimator
        self.scoring = scoring
        self.n_splits = n_splits
        self.early_stop = early_stop

    def scorer(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        scores = {}
        for metric in self.scoring:
            score_func = get_scorer(metric)._score_func
            scores[metric] = score_func(y_true, y_pred)
        return scores
        
    def evaluate_fold(self, train_index: NDArray[np.int_], val_index: NDArray[np.int_], X: np.ndarray, y: np.ndarray
        ) -> Union[Tuple[int, Dict[str, float], dict[str, float]], Tuple[dict[str, float], Dict[str, float]]]:
        
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        if self.early_stop:
            X_train_fold, X_eval, y_train_fold, y_eval = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)
            self.estimator.fit(X_train_fold, y_train_fold, eval_set=[(X_eval, y_eval)], verbose=False)
            n_est = len(next(iter(self.estimator.evals_result().get('validation_0').values())))
        else:
            self.estimator.fit(X_train_fold, y_train_fold, verbose=False)

        y_pred_train = self.estimator.predict(X_train_fold)
        train_score = self.scorer(y_train_fold, y_pred_train)
        y_pred_val = self.estimator.predict(X_val_fold)
        val_score = self.scorer(y_val_fold, y_pred_val)

        return (n_est, train_score, val_score) if self.early_stop else (train_score, val_score)

    def cross_validation(self, X: np.ndarray, y: np.ndarray
                         ) -> Union[Tuple[Dict[str, float], Dict[str, float]], Tuple[int, Dict[str, float], Dict[str, float]]]:
        kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=42)
        results = Parallel(n_jobs=-1)(
            delayed(self.evaluate_fold)(train_index, val_index, X, y) for train_index, val_index in kf.split(X, y))
        return results


class Search:
    def __init__(self, estimator, param_grid: dict, path: str, scoring: List[str]=['roc_auc'], 
                 n_splits: int=5, early_stop: bool=False):
        
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.scoring = scoring
        self.n_splits = n_splits
        self.path = path
        self.early_stop = early_stop   

        if os.path.isfile(self.path):
            self.results = pd.read_parquet(self.path)
            if len(self.param_grid) <= len(self.results):
                sys.exit('Search completed')
            self.restart = True
            self.index_aux = len(self.results) - 1
        else:
            self.restart = False
            self.results = pd.DataFrame()

    def save_results(self, result_dict: dict, i: int, total: int):
        results_line = pd.DataFrame(result_dict, index=[0])
        self.results = pd.concat([self.results, results_line], ignore_index=True)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            self.results.to_parquet(self.path, compression='gzip')

    def grid_search(self, X: np.ndarray, y: np.ndarray):
        cv = CrossValidator(self.estimator, n_splits=self.n_splits, scoring=self.scoring, early_stop=self.early_stop)
        for i, ps in enumerate(self.params_grid):
            if self.restart and i <= self.index_aux:
                continue

            print(f"Evaluating parameter set {i + 1}/{len(self.param_grid)}: {ps}")
            self.estimator.set_params(**ps)
            result_dict = ps.copy()

            results_list = cv.cross_validation(X, y)
            
            if self.early_stop:
                n_estimators, train_scores_list, val_scores_list = zip(*results_list)
                for split, est in enumerate(n_estimators):
                    result_dict[f'estimator_split{split}'] = est

                for metric in self.scoring:
                    train_scores = [train_score[metric] for train_score in train_scores_list]
                    val_scores = [val_score[metric] for val_score in val_scores_list]
                    result_dict.update({
                        f'mean_train_score_{metric}': np.mean(train_scores),
                        f'mean_val_score_{metric}': np.mean(val_scores)
                    })

            else:
                train_scores_list, val_scores_list = zip(*results_list)
                for metric in self.scoring:
                    train_scores = [train_score[metric] for train_score in train_scores_list]
                    val_scores = [val_score[metric] for val_score in val_scores_list]
                    result_dict.update({
                        f'mean_train_score_{metric}': np.mean(train_scores),
                        f'mean_val_score_{metric}': np.mean(val_scores)
                    })

            self.save_results(result_dict, i, len(params_grid))

            print(f"Average Train Scores: {', '.join(f'{metric}: {result_dict[f"mean_train_score_{metric}"]}' for metric in self.scoring)}")
            print(f"Average Validation Scores: {', '.join(f'{metric}: {result_dict[f"mean_val_score_{metric}"]}' for metric in self.scoring)}")

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

# Initialize search object
search = Search(clf, params_grid, path, scoring=['roc_auc', 'accuracy'], n_splits= 5, early_stop=True)
# Choose the search strategy
start_time = time.time()
search.grid_search(X, y)
print("--- %s seconds ---" % (time.time() - start_time))
