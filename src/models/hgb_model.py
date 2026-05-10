# ============================================================
# 阶段 4 — HistGradientBoosting 单 fold 训练
# ============================================================
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from utils import accuracy


def train_hgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols, seed):
    # 计算 category 列在 X_tr 列中的位置索引
    cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]

    model = HistGradientBoostingClassifier(**params)
    model.set_params(categorical_features=cat_indices)
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
