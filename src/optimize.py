# ============================================================
# Optuna HPO — LightGBM / CatBoost / XGBoost
# ============================================================
import sys
import os
import json
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from train import prepare_data
from utils import accuracy, seed_everything
from config import SEED

seed_everything(SEED)

N_SPLITS   = 5
CV_SEED    = 42
N_TRIALS   = 60   # per model
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def _cv_score(model_type, params, X_train, y, cat_cols):
    """Run 1-seed 5-fold CV, return mean OOF accuracy."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X_train, y):
        X_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr  = y.iloc[tr_idx].reset_index(drop=True)
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        # dummy X_test = X_val (we only need val score)
        if model_type == 'lgb':
            from models.lgb_model import train_lgb_fold
            r = train_lgb_fold(X_tr, y_tr, X_val, y_val, X_val, params, cat_cols, CV_SEED)
        elif model_type == 'cat':
            from models.cat_model import train_cat_fold
            r = train_cat_fold(X_tr, y_tr, X_val, y_val, X_val, params, cat_cols, CV_SEED)
        elif model_type == 'xgb':
            from models.xgb_model import train_xgb_fold
            r = train_xgb_fold(X_tr, y_tr, X_val, y_val, X_val, params, CV_SEED)
        scores.append(r['val_score'])
    return float(np.mean(scores))


# ── LGB ──────────────────────────────────────────────────────

def _lgb_objective(trial, X_train, y, cat_cols):
    params = {
        'objective':        'binary',
        'metric':           'binary_error',
        'boosting_type':    'gbdt',
        'verbose':          -1,
        'random_state':     CV_SEED,
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 31, 127),
        'max_depth':        trial.suggest_int('max_depth', 4, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples':trial.suggest_int('min_child_samples', 10, 50),
        'lambda_l1':        trial.suggest_float('lambda_l1', 1e-4, 5.0, log=True),
        'lambda_l2':        trial.suggest_float('lambda_l2', 1e-4, 5.0, log=True),
        # num_boost_round=5000 + early_stopping handled by train_lgb_fold
    }
    return _cv_score('lgb', params, X_train, y, cat_cols)


# ── CAT ──────────────────────────────────────────────────────

def _cat_objective(trial, X_train, y, cat_cols):
    params = {
        'loss_function':       'Logloss',
        'eval_metric':         'Accuracy',
        'verbose':             0,
        'random_seed':         CV_SEED,
        'iterations':          trial.suggest_int('iterations', 1000, 5000, step=500),
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth':               trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength':     trial.suggest_float('random_strength', 0.5, 3.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count':        trial.suggest_int('border_count', 32, 128),
    }
    return _cv_score('cat', params, X_train, y, cat_cols)


# ── XGB ──────────────────────────────────────────────────────

def _xgb_objective(trial, X_train, y, cat_cols):
    params = {
        'objective':        'binary:logistic',
        'eval_metric':      'error',
        'verbosity':        0,
        'random_state':     CV_SEED,
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth':        trial.suggest_int('max_depth', 4, 8),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        # num_boost_round=5000 + early_stopping handled by train_xgb_fold
    }
    return _cv_score('xgb', params, X_train, y, cat_cols)


# ── 主流程 ────────────────────────────────────────────────────

def run_hpo(models=None, n_trials=N_TRIALS):
    if models is None:
        models = ['lgb', 'cat']

    print('=' * 60)
    print('HPO with Optuna')
    print('=' * 60)

    X_train, y, X_test, test_ids, train_ids, cat_cols, _ = prepare_data()

    objectives = {
        'lgb': _lgb_objective,
        'cat': _cat_objective,
        'xgb': _xgb_objective,
    }

    # Load existing params so we can merge (allows resuming)
    out_path = os.path.join(OUTPUT_DIR, 'hpo_best_params.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_params_all = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            best_params_all = json.load(f)

    for model_type in models:
        print(f'\n[{model_type.upper()}] Starting {n_trials} trials...')
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        obj_fn = objectives[model_type]
        study.optimize(
            lambda trial, m=model_type, fn=obj_fn: fn(trial, X_train, y, cat_cols),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best = study.best_trial
        print(f'  Best OOF acc: {best.value:.5f}')
        print(f'  Best params:')
        for k, v in best.params.items():
            print(f'    {k}: {v}')
        best_params_all[model_type] = best.params

        # Save after each model so progress is not lost on interruption
        with open(out_path, 'w') as f:
            json.dump(best_params_all, f, indent=2)
        print(f'  [saved] {out_path}')

    return best_params_all


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='lgb,cat', help='comma-separated')
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    args = parser.parse_args()
    models = args.models.split(',')
    run_hpo(models=models, n_trials=args.trials)
