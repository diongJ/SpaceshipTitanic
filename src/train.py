# ============================================================
# 阶段 4 — 训练主流程：多种子 CV、OOF 保存、测试集预测
# 支持 StratifiedKFold / GroupKFold，模型通过 config 开关
# ============================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import SEED, FEAT_GROUPS, CAT_COLS, LGB_PARAMS, XGB_PARAMS, CAT_PARAMS, ET_PARAMS, HGB_PARAMS
from data import load_raw
from preprocess import preprocess
from features import build_features
from utils import seed_everything, accuracy, save_oof, save_preds


def _compute_fold_label_features(tr_keys, y_tr_arr, val_keys, test_keys, global_mean):
    """
    Generic fold-wise label feature computation (works for GroupId or CabinKey).
    Keys that are None indicate 'no group/cabin' — these rows receive global_mean.
    Returns (tr_feats, val_feats, te_feats), each a tuple of (transported, not_transported, ratio).
    """
    grp_t, grp_nt = {}, {}
    for i, key in enumerate(tr_keys):
        if key is None:
            continue
        grp_t[key] = grp_t.get(key, 0) + int(y_tr_arr[i])
        grp_nt[key] = grp_nt.get(key, 0) + int(1 - y_tr_arr[i])

    def encode(keys, loo_y=None):
        t  = np.array([grp_t.get(k, 0) if k is not None else 0.0 for k in keys], dtype=float)
        nt = np.array([grp_nt.get(k, 0) if k is not None else 0.0 for k in keys], dtype=float)
        if loo_y is not None:
            for i, k in enumerate(keys):
                if k is not None:
                    t[i]  -= loo_y[i]
                    nt[i] -= (1 - loo_y[i])
        tot  = t + nt
        valid = np.array([k is not None for k in keys])
        ratio = np.where(valid & (tot > 0), t / tot, global_mean)
        return t, nt, ratio

    return encode(tr_keys, loo_y=y_tr_arr), encode(val_keys), encode(test_keys)


def _label_feature_setup(train_f, test_f, y, tr_keys, te_keys):
    """
    Shared helper: compute label-based transport features from train keys (LOO) and test keys.
    tr_keys / te_keys: array of keys (None = no group/cabin info).
    Returns (train_f, test_f) with 3 new columns: *TransportedCount, *NotTransportedCount, *TransportRatio.
    Prefix is inferred from the key type; caller renames as needed.
    """
    y_arr = np.array(y)
    global_mean = float(y_arr.mean())
    (tr_t, tr_nt, tr_r), _, (te_t, te_nt, te_r) = _compute_fold_label_features(
        tr_keys, y_arr, tr_keys, te_keys, global_mean)
    # For training LOO, recompute properly
    grp_t, grp_nt = {}, {}
    for i, k in enumerate(tr_keys):
        if k is None:
            continue
        grp_t[k] = grp_t.get(k, 0) + int(y_arr[i])
        grp_nt[k] = grp_nt.get(k, 0) + int(1 - y_arr[i])
    loo_t = np.array([grp_t.get(k, 0) - y_arr[i] if k is not None else 0.0
                      for i, k in enumerate(tr_keys)], dtype=float)
    loo_nt = np.array([grp_nt.get(k, 0) - (1-y_arr[i]) if k is not None else 0.0
                       for i, k in enumerate(tr_keys)], dtype=float)
    loo_tot = loo_t + loo_nt
    valid_tr = np.array([k is not None for k in tr_keys])
    loo_r = np.where(valid_tr & (loo_tot > 0), loo_t / loo_tot, global_mean)
    return loo_t, loo_nt, loo_r, te_t, te_nt, te_r


def _add_group_label_features(train_f, test_f, y):
    tr_keys = train_f['GroupId'].values
    te_keys = test_f['GroupId'].values
    loo_t, loo_nt, loo_r, te_t, te_nt, te_r = _label_feature_setup(
        train_f, test_f, y, tr_keys, te_keys)
    train_f = train_f.copy()
    test_f  = test_f.copy()
    train_f['Group_TrainTransportedCount']    = loo_t
    train_f['Group_TrainNotTransportedCount'] = loo_nt
    train_f['Group_TrainTransportRatio']      = loo_r
    test_f['Group_TrainTransportedCount']     = te_t
    test_f['Group_TrainNotTransportedCount']  = te_nt
    test_f['Group_TrainTransportRatio']       = te_r
    return train_f, test_f


def _make_cabin_keys(df):
    """Build cabin key string (Deck_CabinNum_Side), or None for rows with missing cabin."""
    d = df['Deck'].astype(str)
    n = df['CabinNum'].astype(str)
    s = df['Side'].astype(str)
    keys = (d + '_' + n + '_' + s).values
    no_cabin = np.array(['nan' in k.lower() for k in keys])
    result = keys.astype(object)
    result[no_cabin] = None
    return result


def _add_cabin_label_features(train_f, test_f, y):
    tr_keys = _make_cabin_keys(train_f)
    te_keys = _make_cabin_keys(test_f)
    loo_t, loo_nt, loo_r, te_t, te_nt, te_r = _label_feature_setup(
        train_f, test_f, y, tr_keys, te_keys)
    train_f = train_f.copy()
    test_f  = test_f.copy()
    train_f['Cabin_TrainTransportedCount']    = loo_t
    train_f['Cabin_TrainNotTransportedCount'] = loo_nt
    train_f['Cabin_TrainTransportRatio']      = loo_r
    test_f['Cabin_TrainTransportedCount']     = te_t
    test_f['Cabin_TrainNotTransportedCount']  = te_nt
    test_f['Cabin_TrainTransportRatio']       = te_r
    return train_f, test_f


def get_feature_names(groups=None):
    """从 FEAT_GROUPS 组装特征列名。传入 groups 列表只取部分组，默认全部。"""
    if groups is None:
        groups = list(FEAT_GROUPS.keys())
    return [col for g in groups for col in FEAT_GROUPS[g]]


def prepare_data(feature_groups=None):
    """加载 → 预处理 → 特征工程 → 返回 (X_train, y, X_test, test_ids, train_ids)"""
    print('=' * 60)
    print('Phase 4: Training Pipeline')
    print('=' * 60)

    seed_everything(SEED)
    train_raw, test_raw = load_raw()
    y = train_raw['Transported'].astype(int)

    train_c, test_c = preprocess(train_raw, test_raw)
    train_f, test_f = build_features(train_c, test_c)
    train_f, test_f = _add_group_label_features(train_f, test_f, y)

    feature_cols = get_feature_names(feature_groups)
    # 排除目标编码列（由 ensemble.py meta-learner 使用，不在 base model 中）
    te_cols = set(FEAT_GROUPS.get('target_enc', []))
    # 只保留实际存在的列
    feature_cols = [c for c in feature_cols if c in train_f.columns and c not in te_cols]
    print(f'  feature count: {len(feature_cols)}')

    # PassengerId 用于保存 OOF / 提交
    train_ids = train_f['PassengerId']
    test_ids = test_f['PassengerId']

    X_train = train_f[feature_cols].copy()
    X_test = test_f[feature_cols].copy()

    # 类别列转换为 category dtype（LGB/XGB/CAT 原生支持）
    cat_cols_present = [c for c in CAT_COLS if c in X_train.columns]
    for col in cat_cols_present:
        # bool → int 先（XGBoost 无法处理 bool categories）
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # GroupId 用于 GroupKFold
    group_ids = train_f['GroupId'].values

    print(f'  categorical columns: {len(cat_cols_present)}')
    return X_train, y, X_test, test_ids, train_ids, cat_cols_present, group_ids


def run_cv(models=None, seeds=None, n_splits=5, cv_scheme='stratified', feature_groups=None):
    """
    主训练入口。

    Parameters
    ----------
    models : list[str] | None
        要训练的模型列表，可选 'lgb', 'xgb', 'cat'。默认 ['lgb']。
    seeds : list[int] | None
        CV 随机种子列表。默认 [42, 2024]。
    n_splits : int
        折数，默认 5。
    cv_scheme : str
        'stratified' 或 'group'。
    feature_groups : list[str] | None
        使用的特征组列表，默认全部（FEAT_GROUPS 所有 key）。
    """
    if models is None:
        models = ['lgb']
    if seeds is None:
        seeds = [42, 2024]

    X_train, y, X_test, test_ids, train_ids, cat_cols, all_groups = prepare_data(feature_groups)

    groups = all_groups if cv_scheme == 'group' else None
    global_mean = float(y.mean())

    # Prepare fold-wise recomputation for group + cabin label features
    # Group label features already pre-computed in prepare_data() using full-training LOO.
    # No fold-wise recomputation needed — X_test uses full training labels for max signal.

    for model_type in models:
        print(f'\n{"─" * 50}')
        print(f'[{model_type.upper()}] seeds={seeds} folds={n_splits} scheme={cv_scheme}')
        print(f'{"─" * 50}')

        oof_preds_sum = np.zeros(len(X_train))
        test_preds_list = []
        fold_scores = []
        fold_num = 0

        for seed in seeds:
            if cv_scheme == 'group':
                cv = GroupKFold(n_splits=n_splits)
                split_iter = cv.split(X_train, y, groups=groups)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                split_iter = cv.split(X_train, y)

            for fold_idx, (tr_idx, val_idx) in enumerate(split_iter, 1):
                fold_num += 1

                X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
                y_tr = y.iloc[tr_idx].reset_index(drop=True)
                X_val = X_train.iloc[val_idx].reset_index(drop=True)
                y_val = y.iloc[val_idx].reset_index(drop=True)

                X_test_fold = X_test

                # 分发到各模型训练函数
                params, cat_cols_override = _get_model_params(model_type, seed, fold_idx)
                cat = cat_cols_override if cat_cols_override is not None else cat_cols

                result = _train_one_fold(model_type, X_tr, y_tr, X_val, y_val, X_test_fold, params, cat, seed)

                # 累加各 seed 的 OOF，最后除以 seed 数量取平均
                oof_preds_sum[val_idx] += result['y_val_pred']
                test_preds_list.append(result['y_test_pred'])
                fold_scores.append({
                    'seed': seed, 'fold': fold_idx, 'acc': round(result['val_score'], 5)
                })
                print(f'  [seed={seed} fold={fold_idx}] acc={result["val_score"]:.5f}')

        oof_preds = oof_preds_sum / len(seeds)

        # 平均测试集预测
        test_avg = np.mean(test_preds_list, axis=0)

        # 汇总
        oof_acc = accuracy(y, oof_preds >= 0.5)
        cv_mean = np.mean([s['acc'] for s in fold_scores])
        cv_std = np.std([s['acc'] for s in fold_scores])
        print(f'  -> OOF acc={oof_acc:.5f}  CV mean={cv_mean:.5f}+-{cv_std:.5f}')

        # 保存
        exp_name = f'{model_type}_{cv_scheme}_{n_splits}fold_{len(seeds)}seed'
        save_oof(pd.DataFrame({
            'PassengerId': train_ids.values,
            'Transported_Prob': oof_preds
        }), exp_name)
        save_preds(test_ids.values, test_avg, exp_name)

    print(f'\n{"=" * 60}')
    print('Phase 4 complete.')
    print(f'{"=" * 60}')


def _get_model_params(model_type, seed, fold_idx):
    """返回 (params_dict, cat_cols_override_or_None)"""
    if model_type == 'lgb':
        p = LGB_PARAMS.copy()
        p['random_state'] = seed
        return p, None  # cat_cols from prepare_data
    elif model_type == 'xgb':
        p = XGB_PARAMS.copy()
        p['random_state'] = seed
        return p, None
    elif model_type == 'cat':
        p = CAT_PARAMS.copy()
        p['random_seed'] = seed
        return p, None
    elif model_type == 'et':
        p = ET_PARAMS.copy()
        p['random_state'] = seed * 10 + fold_idx
        return p, None
    elif model_type == 'hgb':
        p = HGB_PARAMS.copy()
        p['random_state'] = seed * 10 + fold_idx
        return p, None
    else:
        raise ValueError(f'Unknown model_type: {model_type}')


def _train_one_fold(model_type, X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed):
    if model_type == 'lgb':
        from models.lgb_model import train_lgb_fold
        return train_lgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed)
    elif model_type == 'xgb':
        from models.xgb_model import train_xgb_fold
        return train_xgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, seed)
    elif model_type == 'cat':
        from models.cat_model import train_cat_fold
        return train_cat_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed)
    elif model_type == 'et':
        from models.et_model import train_et_fold
        return train_et_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed)
    elif model_type == 'hgb':
        from models.hgb_model import train_hgb_fold
        return train_hgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')


# ── CLI 入口 ────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # 默认：LGB baseline with StratifiedKFold
    _models = ['lgb']
    _cv = 'stratified'
    _seeds = [42, 2024]

    # 简单参数解析
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--models':
            _models = args[i + 1].split(',')
            i += 2
        elif args[i] == '--cv':
            _cv = args[i + 1]
            i += 2
        elif args[i] == '--seeds':
            _seeds = [int(s) for s in args[i + 1].split(',')]
            i += 2
        else:
            i += 1

    print(f'Models: {_models} | CV: {_cv} | Seeds: {_seeds}')
    run_cv(models=_models, seeds=_seeds, n_splits=5, cv_scheme=_cv)
