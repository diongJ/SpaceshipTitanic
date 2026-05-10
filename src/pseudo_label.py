# ============================================================
# Pseudo-labeling — 用高置信测试预测扩充训练集，重训并集成
# 流程：
#   1. 从当前 ensemble 预测中取置信样本（prob>HIGH 或 <LOW）
#   2. 原始训练集 + 伪标签 → 扩充训练集
#   3. 重训 LGB / CAT（可加载 HPO 参数）
#   4. 测试集仍为全部 4277 样本（格式不变）
#   5. OOF 只报告原始训练集部分
# ============================================================
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import SEED, OUTPUT_DIR, TRAIN_CSV, TEST_CSV, FEAT_GROUPS, CAT_COLS
from utils import seed_everything, accuracy, save_oof, save_preds

seed_everything(SEED)

HIGH_TH   = 0.95
LOW_TH    = 0.05
PRED_FILE = os.path.join(OUTPUT_DIR, 'preds', 'ensemble_lr_stack.csv')


def load_pseudo_labels(pred_file=PRED_FILE, high=HIGH_TH, low=LOW_TH):
    df = pd.read_csv(pred_file)
    mask = (df['Transported_Prob'] >= high) | (df['Transported_Prob'] <= low)
    df_conf = df[mask].copy()
    df_conf['PseudoLabel'] = (df_conf['Transported_Prob'] >= high).astype(int)
    n_true  = df_conf['PseudoLabel'].sum()
    n_false = (df_conf['PseudoLabel'] == 0).sum()
    print(f'[PL] 高置信样本: {len(df_conf)}/{len(df)}  True={n_true}  False={n_false}')
    return df_conf[['PassengerId', 'PseudoLabel']]


def build_augmented_data(hpo_params_file=None):
    """
    返回:
      X_aug, y_aug   — 扩充训练集（原始 + 伪标签）
      X_test, test_ids — 完整测试集 4277 行
      train_ids      — 原始训练集 PassengerId
      n_orig         — 原始训练集行数
      cat_cols       — 类别列列表
    """
    from data import load_raw
    from preprocess import preprocess
    from features import build_features

    train_raw, test_raw = load_raw()
    y_orig = train_raw['Transported'].astype(int)
    n_orig = len(train_raw)

    train_c, test_c = preprocess(train_raw, test_raw)
    train_f, test_f = build_features(train_c, test_c)

    te_cols = set(FEAT_GROUPS.get('target_enc', []))
    feature_cols = [c for g in FEAT_GROUPS for c in FEAT_GROUPS[g]
                    if c in train_f.columns and c not in te_cols]

    cat_cols_present = [c for c in CAT_COLS if c in train_f.columns]

    X_train = train_f[feature_cols].copy()
    X_test  = test_f[feature_cols].copy()
    train_ids = train_f['PassengerId']
    test_ids  = test_f['PassengerId']

    # 伪标签
    pl = load_pseudo_labels()
    pl_map = dict(zip(pl['PassengerId'], pl['PseudoLabel']))
    test_f_pl = test_f[test_f['PassengerId'].isin(pl_map)].copy()

    X_pl  = test_f_pl[feature_cols].copy()
    y_pl  = test_f_pl['PassengerId'].map(pl_map).values

    # concat 前不做 category 转换，concat 后统一转，避免不同 categories 集合导致 str 回退
    X_aug = pd.concat([X_train, X_pl], ignore_index=True)
    y_aug = np.concatenate([y_orig.values, y_pl])

    # category 转换在 concat 后统一做，避免不同 category 集合拼接后退化为 str
    for col in cat_cols_present:
        for df in [X_aug, X_test]:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            df[col] = df[col].astype('category')

    print(f'[PL] 扩充训练集: {len(X_aug)} 行 (原始={n_orig}, 伪标签={len(X_pl)})')
    print(f'[PL] 测试集: {len(X_test)} 行  特征: {len(feature_cols)}')
    return X_aug, y_aug, X_test, test_ids, train_ids, cat_cols_present, n_orig


def run_pseudo_label_cv(models=None, seeds=None, n_splits=5,
                        hpo_params_file=None):
    if models is None:
        models = ['lgb', 'cat']
    if seeds is None:
        seeds = [42, 2024]

    print('=' * 60)
    print('Pseudo-label Training')
    print('=' * 60)

    hpo_params = {}
    if hpo_params_file and os.path.exists(hpo_params_file):
        with open(hpo_params_file) as f:
            hpo_params = json.load(f)
        print(f'[HPO] 加载参数: {list(hpo_params.keys())}')

    from sklearn.model_selection import StratifiedKFold
    from train import _train_one_fold, _get_model_params

    X_aug, y_aug, X_test, test_ids, train_ids, cat_cols, n_orig = \
        build_augmented_data(hpo_params_file)

    y_series = pd.Series(y_aug)

    for model_type in models:
        print(f'\n[{model_type.upper()}] seeds={seeds} folds={n_splits}')

        oof_sum   = np.zeros(n_orig)
        test_list = []
        fold_scores = []

        for seed in seeds:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_aug, y_series), 1):
                X_tr  = X_aug.iloc[tr_idx].reset_index(drop=True)
                y_tr  = y_series.iloc[tr_idx].reset_index(drop=True)
                X_val = X_aug.iloc[val_idx].reset_index(drop=True)
                y_val = y_series.iloc[val_idx].reset_index(drop=True)

                params, cat_override = _get_model_params(model_type, seed, fold_idx)

                # Apply HPO params
                if model_type in hpo_params:
                    hp = hpo_params[model_type]
                    keys = {
                        'lgb': ['learning_rate','num_leaves','max_depth','feature_fraction',
                                'bagging_fraction','bagging_freq','min_child_samples',
                                'lambda_l1','lambda_l2'],
                        'cat': ['iterations','learning_rate','depth','l2_leaf_reg',
                                'random_strength','bagging_temperature','border_count'],
                        'xgb': ['learning_rate','max_depth','subsample','colsample_bytree',
                                'min_child_weight','reg_alpha','reg_lambda'],
                    }
                    for k in keys.get(model_type, []):
                        if k in hp:
                            params[k] = hp[k]

                cat = cat_override if cat_override is not None else cat_cols
                result = _train_one_fold(model_type, X_tr, y_tr, X_val, y_val,
                                         X_test, params, cat, seed)

                # OOF: 只统计 val_idx 中属于原始训练集的行（index < n_orig）
                orig_mask = val_idx < n_orig
                orig_pos  = val_idx[orig_mask]           # positions in X_aug
                pred_pos  = np.where(orig_mask)[0]       # positions in X_val / y_val_pred
                if len(orig_pos) > 0:
                    oof_sum[orig_pos] += result['y_val_pred'][pred_pos]

                test_list.append(result['y_test_pred'])
                fold_scores.append({'seed': seed, 'fold': fold_idx,
                                    'acc': round(result['val_score'], 5)})
                print(f'  [seed={seed} fold={fold_idx}] acc={result["val_score"]:.5f}')

        oof_preds = oof_sum / len(seeds)
        test_avg  = np.mean(test_list, axis=0)

        y_orig_arr = y_aug[:n_orig]
        oof_acc = accuracy(y_orig_arr, oof_preds >= 0.5)
        cv_mean = np.mean([s['acc'] for s in fold_scores])
        print(f'  -> OOF acc (原始训练集)={oof_acc:.5f}  CV mean={cv_mean:.5f}')

        exp_name = f'{model_type}_pl_{n_splits}fold_{len(seeds)}seed'
        save_oof(pd.DataFrame({
            'PassengerId': train_ids.values,
            'Transported_Prob': oof_preds
        }), exp_name)
        save_preds(test_ids.values, test_avg, exp_name)

    print(f'\n{"=" * 60}\nPseudo-label training complete.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='lgb,cat')
    parser.add_argument('--seeds',  default='42,2024')
    parser.add_argument('--hpo',    default=None)
    args = parser.parse_args()
    models = args.models.split(',')
    seeds  = [int(s) for s in args.seeds.split(',')]
    hpo_file = args.hpo or os.path.join(OUTPUT_DIR, 'hpo_best_params.json')
    run_pseudo_label_cv(models=models, seeds=seeds, hpo_params_file=hpo_file)
