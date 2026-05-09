# ============================================================
# 阶段 5 — 集成：加权 Blend + LR Stacking + 阈值优化
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import SEED, OUTPUT_DIR
from utils import accuracy, save_preds, save_submission, seed_everything

seed_everything(SEED)

# 默认实验名（与 train.py 保持一致）
DEFAULT_EXPS = {
    'lgb': 'lgb_stratified_5fold_2seed',
    'xgb': 'xgb_stratified_5fold_2seed',
    'cat': 'cat_stratified_5fold_2seed',
}


# ── 数据加载 ────────────────────────────────────────────────

def load_labels():
    """从原始训练集读取真实标签，按 PassengerId 排列。"""
    from config import TRAIN_CSV
    df = pd.read_csv(TRAIN_CSV)[['PassengerId', 'Transported']]
    df['Transported'] = df['Transported'].astype(int)
    return df.set_index('PassengerId')['Transported']


def load_oof_preds(exps=None):
    """
    返回 (oof_dict, pred_dict)
      oof_dict  : {model_name: pd.Series indexed by PassengerId}
      pred_dict : {model_name: pd.Series indexed by PassengerId}
    """
    if exps is None:
        exps = DEFAULT_EXPS
    oof_dict, pred_dict = {}, {}
    for name, exp in exps.items():
        oof_path  = os.path.join(OUTPUT_DIR, 'oof',   f'{exp}.csv')
        pred_path = os.path.join(OUTPUT_DIR, 'preds', f'{exp}.csv')
        oof_dict[name]  = pd.read_csv(oof_path ).set_index('PassengerId')['Transported_Prob']
        pred_dict[name] = pd.read_csv(pred_path).set_index('PassengerId')['Transported_Prob']
        print(f'  [load] {name}: OOF={len(oof_dict[name])}, PRED={len(pred_dict[name])}')
    return oof_dict, pred_dict


def align_matrices(oof_dict, pred_dict, y):
    """
    将 OOF/Pred 对齐到 y 的索引顺序，返回 numpy 矩阵。
    返回: (oof_mat, pred_mat, model_names)
    """
    model_names = list(oof_dict.keys())
    train_ids = y.index
    test_ids  = list(pred_dict[model_names[0]].index)

    oof_mat  = np.column_stack([oof_dict[m][train_ids].values for m in model_names])
    pred_mat = np.column_stack([pred_dict[m][test_ids].values  for m in model_names])
    return oof_mat, pred_mat, model_names, test_ids


# ── 加权 Blend ──────────────────────────────────────────────

def search_weights(oof_mat, y_arr, model_names):
    """
    用 scipy.optimize.minimize (Nelder-Mead) 搜索最优混合权重。
    约束：权重非负且归一化。
    返回: (best_weights, best_oof_acc_at_0.5)
    """
    n = oof_mat.shape[1]

    def neg_acc(raw_w):
        w = np.clip(raw_w, 0, None)
        total = w.sum()
        if total < 1e-9:
            return 0.0
        w = w / total
        return -accuracy(y_arr, oof_mat @ w >= 0.5)

    # 从多个初始点出发，取最优
    init_points = [np.ones(n) / n]                          # 等权
    for i in range(n):
        v = np.full(n, 0.05)
        v[i] = 0.85
        init_points.append(v)

    best_w, best_score = None, -1.0
    for x0 in init_points:
        res = minimize(neg_acc, x0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-7})
        w = np.clip(res.x, 0, None)
        w /= w.sum()
        s = -neg_acc(w)
        if s > best_score:
            best_score, best_w = s, w

    print('\n[Blend] 最优权重:')
    for name, w in zip(model_names, best_w):
        print(f'  {name}: {w:.4f}')
    print(f'  OOF acc (th=0.50): {best_score:.5f}')
    return best_w, best_score


# ── LR Stacking ─────────────────────────────────────────────

def lr_stacking(oof_mat, pred_mat, y_arr, model_names):
    """
    以 OOF 概率矩阵为特征训练 LogisticRegression 元模型。
    OOF 预测已经是 out-of-fold，可直接作为训练数据而不会泄漏。
    返回: (stack_oof_proba, stack_pred_proba, stack_oof_acc)
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(oof_mat)
    X_te = scaler.transform(pred_mat)

    lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    lr.fit(X_tr, y_arr)

    stack_oof  = lr.predict_proba(X_tr)[:, 1]
    stack_pred = lr.predict_proba(X_te)[:, 1]
    stack_acc  = accuracy(y_arr, stack_oof >= 0.5)

    print('\n[LR Stack] 系数:')
    for name, coef in zip(model_names, lr.coef_[0]):
        print(f'  {name}: {coef:.4f}')
    print(f'  OOF acc (th=0.50): {stack_acc:.5f}')
    return stack_oof, stack_pred, stack_acc


# ── 阈值优化 ────────────────────────────────────────────────

def optimize_threshold(oof_proba, y_arr, lo=0.35, hi=0.65, steps=121):
    """在 OOF 上网格搜索最优分类阈值。"""
    thresholds = np.linspace(lo, hi, steps)
    scores = [accuracy(y_arr, oof_proba >= t) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best_t   = float(thresholds[best_idx])
    best_acc = float(scores[best_idx])
    print(f'\n[Threshold] 搜索范围 [{lo:.2f}, {hi:.2f}]:')
    print(f'  最优阈值: {best_t:.4f}')
    print(f'  OOF acc:  {best_acc:.5f}')
    return best_t, best_acc


# ── 主流程 ──────────────────────────────────────────────────

def run_ensemble(exps=None, save=True):
    """
    完整集成流程：
      1. 加载 OOF + 测试集预测
      2. 各单模 OOF acc（基准）
      3. 加权 Blend（含阈值优化）
      4. LR Stacking（含阈值优化）
      5. 选最佳方案生成提交文件
    """
    print('=' * 60)
    print('Phase 5: Ensemble')
    print('=' * 60)

    y = load_labels()
    print(f'\n[data] 训练集标签: {len(y)} 条，正类比例 {y.mean():.4f}')

    print('\n[data] 加载预测文件...')
    oof_dict, pred_dict = load_oof_preds(exps)

    oof_mat, pred_mat, model_names, test_ids = align_matrices(oof_dict, pred_dict, y)
    y_arr = y.values

    # ── 1. 单模基准 ──────────────────────────────────────────
    print('\n' + '─' * 50)
    print('[Step 1] 单模 OOF 基准')
    print('─' * 50)
    single_results = {}
    for i, name in enumerate(model_names):
        acc = accuracy(y_arr, oof_mat[:, i] >= 0.5)
        t_opt, acc_opt = optimize_threshold(oof_mat[:, i], y_arr)
        single_results[name] = {'acc_0.5': acc, 'best_th': t_opt, 'acc_opt': acc_opt}
        print(f'  {name}: th=0.50 → {acc:.5f} | th={t_opt:.4f} → {acc_opt:.5f}')

    # ── 2. 加权 Blend ────────────────────────────────────────
    print('\n' + '─' * 50)
    print('[Step 2] 加权 Blend')
    print('─' * 50)
    best_w, blend_acc_05 = search_weights(oof_mat, y_arr, model_names)
    blend_oof  = oof_mat  @ best_w
    blend_pred = pred_mat @ best_w
    blend_th, blend_acc_opt = optimize_threshold(blend_oof, y_arr)

    # ── 3. LR Stacking ──────────────────────────────────────
    print('\n' + '─' * 50)
    print('[Step 3] LR Stacking')
    print('─' * 50)
    stack_oof, stack_pred, stack_acc_05 = lr_stacking(oof_mat, pred_mat, y_arr, model_names)
    stack_th, stack_acc_opt = optimize_threshold(stack_oof, y_arr)

    # ── 汇总对比 ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('汇总对比 (OOF Accuracy)')
    print('=' * 60)
    rows = []
    for name in model_names:
        r = single_results[name]
        rows.append({'方法': name, 'th=0.50': f"{r['acc_0.5']:.5f}",
                     '最优阈值': f"{r['best_th']:.4f}", 'th=最优': f"{r['acc_opt']:.5f}"})
    rows.append({'方法': 'Blend',    'th=0.50': f'{blend_acc_05:.5f}',
                 '最优阈值': f'{blend_th:.4f}', 'th=最优': f'{blend_acc_opt:.5f}'})
    rows.append({'方法': 'LR Stack', 'th=0.50': f'{stack_acc_05:.5f}',
                 '最优阈值': f'{stack_th:.4f}',  'th=最优': f'{stack_acc_opt:.5f}'})

    df_summary = pd.DataFrame(rows).set_index('方法')
    print(df_summary.to_string())

    # ── 选最佳方案生成提交 ───────────────────────────────────
    print('\n' + '─' * 50)
    candidates = {
        'blend':    (blend_oof,  blend_pred,  blend_th,  blend_acc_opt),
        'lr_stack': (stack_oof,  stack_pred,  stack_th,  stack_acc_opt),
    }
    best_name = max(candidates, key=lambda k: candidates[k][3])
    best_oof, best_pred, best_th, best_acc = candidates[best_name]

    print(f'[选定方案] {best_name}  OOF acc={best_acc:.5f}  threshold={best_th:.4f}')

    if save:
        # 保存集成 OOF 概率
        exp_tag = f'ensemble_{best_name}'
        save_preds(test_ids, best_pred, exp_tag)

        # 生成提交文件
        sub_path = save_submission(test_ids, best_pred, best_th, exp_tag)
        print(f'\n[done] 提交文件: {sub_path}')

    return best_pred, best_th, best_acc


# ── CLI ─────────────────────────────────────────────────────

if __name__ == '__main__':
    run_ensemble()
