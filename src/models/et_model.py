# ============================================================
# 阶段 4 — Extra Trees 单 fold 训练
# ============================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from utils import accuracy


def _cat_to_codes(X):
    """将 category 列转为 int codes，非 category 列保持不变。"""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes
    return X


def train_et_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed):
    X_tr = _cat_to_codes(X_tr)
    X_val = _cat_to_codes(X_val)
    X_test = _cat_to_codes(X_test)

    model = ExtraTreesClassifier(**params)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    val_score = accuracy(y_val, y_val_pred >= 0.5)

    return {
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'model': model,
        'val_score': val_score,
    }
