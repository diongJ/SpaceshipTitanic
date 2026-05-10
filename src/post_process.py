# ============================================================
# 阶段 6 — 后处理规则：在 OOF 上验证，提升 >= 0.001 才应用
#
# Rule 1: CryoSleep=True + NoSpend=True + 预测不确定
#   -> 将概率推至 threshold+0.06（保守修正，不硬翻转）
#
# Rule 2: 组内置信投票
#   组内高置信成员（|prob-threshold| >= 0.18）多数投票 ->
#   低置信成员跟随多数，概率推至 threshold±0.12
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import SEED, TRAIN_CSV, TEST_CSV, OUTPUT_DIR
from data import load_raw
from preprocess import preprocess
from utils import accuracy, save_submission, seed_everything

seed_everything(SEED)

DEFAULT_EXPS = {
    'lgb': 'lgb_stratified_5fold_2seed',
    'xgb': 'xgb_stratified_5fold_2seed',
    'cat': 'cat_stratified_5fold_2seed',
}
ENSEMBLE_THRESHOLD = 0.455


# ── 辅助：重建集成 OOF 和测试集预测 ────────────────────────

def rebuild_ensemble(exps=None):
    """复现 LR Stacking，返回 (oof_prob, test_prob, train_pids, test_pids, y)"""
    if exps is None:
        exps = DEFAULT_EXPS

    y_df = pd.read_csv(TRAIN_CSV)[['PassengerId', 'Transported']]
    y_df['Transported'] = y_df['Transported'].astype(int)
    y = y_df.set_index('PassengerId')['Transported']
    train_pids = y.index.tolist()

    oof_mats, pred_mats = [], []
    for name, exp in exps.items():
        oof_path  = os.path.join(OUTPUT_DIR, 'oof',   f'{exp}.csv')
        pred_path = os.path.join(OUTPUT_DIR, 'preds', f'{exp}.csv')
        oof_s  = pd.read_csv(oof_path ).set_index('PassengerId')['Transported_Prob']
        pred_s = pd.read_csv(pred_path).set_index('PassengerId')['Transported_Prob']
        oof_mats.append(oof_s[train_pids].values)
        pred_mats.append(pred_s.values)
    test_pids = pd.read_csv(os.path.join(OUTPUT_DIR, 'preds',
                                         f'{list(exps.values())[0]}.csv'))['PassengerId'].tolist()

    oof_mat  = np.column_stack(oof_mats)
    pred_mat = np.column_stack(pred_mats)
    y_arr    = y.values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(oof_mat)
    X_te = scaler.transform(pred_mat)

    lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    lr.fit(X_tr, y_arr)

    oof_prob  = lr.predict_proba(X_tr)[:, 1]
    test_prob = lr.predict_proba(X_te)[:, 1]

    base_acc = accuracy(y_arr, oof_prob >= ENSEMBLE_THRESHOLD)
    print(f'[ensemble rebuild] OOF acc (th={ENSEMBLE_THRESHOLD}): {base_acc:.5f}')
    return oof_prob, test_prob, train_pids, test_pids, y_arr


# ── 辅助：加载规则所需元特征 ──────────────────────────────

def load_meta_features():
    """
    返回 (train_meta, test_meta)，均为 DataFrame，index=PassengerId，含列：
      GroupId, CryoSleep, NoSpend, CryoFlag
    """
    train_raw, test_raw = load_raw()
    train_c, test_c = preprocess(train_raw, test_raw)

    def extract(df):
        m = df[['PassengerId', 'GroupId', 'CryoSleep', 'TotalSpend']].copy()
        m['CryoSleep'] = m['CryoSleep'].astype(bool)
        m['CryoFlag'] = m['CryoSleep'].astype(int)
        m['NoSpend'] = (m['TotalSpend'] == 0).astype(int)
        return m.set_index('PassengerId')

    return extract(train_c), extract(test_c)


# ── Rule 1: CryoSleep + NoSpend 不确定区修正 ──────────────

def validate_cryo_rule(oof_prob, train_pids, y_arr, train_meta, threshold):
    """
    CryoSleep=True + NoSpend=True + 预测在不确定区
    (threshold-0.1, threshold+0.08) -> 概率推至 >= threshold+0.06
    """
    pids = np.array(train_pids)
    cryo_flag = train_meta.loc[pids, 'CryoFlag'].values
    no_spend  = train_meta.loc[pids, 'NoSpend'].values

    adjusted = oof_prob.copy()
    cryo_mask = (cryo_flag == 1) & (no_spend == 1)
    uncertain = (
        cryo_mask
        & (adjusted > threshold - 0.1)
        & (adjusted < threshold + 0.08)
    )
    adjusted[uncertain] = np.maximum(adjusted[uncertain], threshold + 0.06)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    base_acc = accuracy(y_arr, oof_prob >= threshold)
    new_acc  = accuracy(y_arr, adjusted >= threshold)
    gain     = new_acc - base_acc

    print(f'\n[Rule 1 — CryoSleep+NoSpend 不确定修正]')
    print(f'  CryoSleep=True & NoSpend=True 样本数: {cryo_mask.sum()}')
    print(f'  不确定区被调整: {uncertain.sum()} 个')
    print(f'  OOF acc: {base_acc:.5f} -> {new_acc:.5f}  (delta={gain:+.5f})')
    return new_acc, gain


def apply_cryo_rule(test_prob, test_pids, test_meta, threshold):
    """对测试集应用 CryoSleep+NoSpend 不确定修正"""
    pids = np.array(test_pids)
    cryo_flag = test_meta.loc[pids, 'CryoFlag'].values
    no_spend  = test_meta.loc[pids, 'NoSpend'].values

    adjusted = test_prob.copy()
    cryo_mask = (cryo_flag == 1) & (no_spend == 1)
    uncertain = (
        cryo_mask
        & (adjusted > threshold - 0.1)
        & (adjusted < threshold + 0.08)
    )
    adjusted[uncertain] = np.maximum(adjusted[uncertain], threshold + 0.06)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    print(f'  [CryoSleep] 测试集不确定区调整: {uncertain.sum()} 个')
    return adjusted


# ── Rule 2: 组内置信投票 ──────────────────────────────────

def validate_group_rule(oof_prob, train_pids, y_arr, train_meta, threshold):
    """
    组内高置信成员 (|prob - threshold| >= 0.18) 多数投票 ->
    低置信成员跟随多数，概率推至 threshold±0.12。
    """
    pids = np.array(train_pids)
    group_ids = train_meta.loc[pids, 'GroupId'].values

    adjusted = oof_prob.copy()
    n_to_true = 0
    n_to_false = 0

    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if len(idx) <= 1:
            continue
        g_probs = adjusted[idx]
        confident = (g_probs <= threshold - 0.18) | (g_probs >= threshold + 0.18)
        if not confident.any():
            continue
        majority = int((g_probs[confident] >= threshold).mean() >= 0.5)
        uncertain_idx = idx[~confident]
        if len(uncertain_idx) == 0:
            continue
        if majority:
            adjusted[uncertain_idx] = threshold + 0.12
            n_to_true += len(uncertain_idx)
        else:
            adjusted[uncertain_idx] = threshold - 0.12
            n_to_false += len(uncertain_idx)

    adjusted = np.clip(adjusted, 0.0, 1.0)

    base_acc = accuracy(y_arr, oof_prob >= threshold)
    new_acc  = accuracy(y_arr, adjusted >= threshold)
    gain     = new_acc - base_acc

    print(f'\n[Rule 2 — 组内置信投票]')
    print(f'  翻转 ->True: {n_to_true} 个, ->False: {n_to_false} 个')
    print(f'  OOF acc: {base_acc:.5f} -> {new_acc:.5f}  (delta={gain:+.5f})')
    return new_acc, gain


def apply_group_rule(prob, pids, meta, threshold):
    """对测试集应用组内置信投票规则"""
    group_ids = meta.loc[np.array(pids), 'GroupId'].values

    adjusted = prob.copy()
    n_to_true = 0
    n_to_false = 0

    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if len(idx) <= 1:
            continue
        g_probs = adjusted[idx]
        confident = (g_probs <= threshold - 0.18) | (g_probs >= threshold + 0.18)
        if not confident.any():
            continue
        majority = int((g_probs[confident] >= threshold).mean() >= 0.5)
        uncertain_idx = idx[~confident]
        if len(uncertain_idx) == 0:
            continue
        if majority:
            adjusted[uncertain_idx] = threshold + 0.12
            n_to_true += len(uncertain_idx)
        else:
            adjusted[uncertain_idx] = threshold - 0.12
            n_to_false += len(uncertain_idx)

    adjusted = np.clip(adjusted, 0.0, 1.0)
    print(f'  [Group] 测试集翻转 ->True: {n_to_true}, ->False: {n_to_false}')
    return adjusted


# ── 主流程 ────────────────────────────────────────────────

def run_postprocess(min_gain=0.001):
    print('=' * 60)
    print('Phase 6: Post-processing (confidence-based rules)')
    print('=' * 60)

    # Step 1: 重建集成 OOF
    print('\n[Step 1] 重建 LR Stack OOF...')
    oof_prob, test_prob, train_pids, test_pids, y_arr = rebuild_ensemble()

    # Step 2: 加载规则特征
    print('\n[Step 2] 加载元特征 (GroupId / CryoSleep / NoSpend)...')
    train_meta, test_meta = load_meta_features()

    threshold = ENSEMBLE_THRESHOLD
    base_acc = accuracy(y_arr, oof_prob >= threshold)
    print(f'基准 OOF acc (th={threshold}): {base_acc:.5f}')

    # Step 3: 验证规则（Rule 1 -> Rule 2 级联）
    print('\n' + '-' * 50)
    print('[Step 3] OOF 规则验证（级联）')
    print('-' * 50)

    # Rule 1
    cryo_acc, cryo_gain = validate_cryo_rule(
        oof_prob, train_pids, y_arr, train_meta, threshold)

    # Rule 2（在 Rule 1 输出上验证）
    oof_after_r1 = oof_prob.copy()
    if cryo_gain >= min_gain:
        oof_after_r1 = apply_cryo_rule(oof_prob, train_pids, train_meta, threshold)
    group_acc, group_gain = validate_group_rule(
        oof_after_r1, train_pids, y_arr, train_meta, threshold)

    # Step 4: 决策
    print('\n' + '-' * 50)
    print('[Step 4] 规则应用决策')
    print('-' * 50)

    apply_cryo  = cryo_gain  >= min_gain
    apply_group = group_gain >= min_gain

    cryo_status  = "OK 应用" if apply_cryo  else "-- 跳过(提升不足)"
    group_status = "OK 应用" if apply_group else "-- 跳过(提升不足)"
    print(f'  Rule 1 (CryoSleep+NoSpend): delta={cryo_gain:+.5f}  {cryo_status}')
    print(f'  Rule 2 (Group vote):        delta={group_gain:+.5f}  {group_status}')

    if not apply_cryo and not apply_group:
        print('\n  所有规则均未达到阈值，不修改预测。')
        return

    # Step 5: 应用到测试集
    print('\n' + '-' * 50)
    print('[Step 5] 应用规则到测试集')
    print('-' * 50)

    new_test_prob = test_prob.copy()
    tag_parts = ['post']

    if apply_cryo:
        new_test_prob = apply_cryo_rule(new_test_prob, test_pids, test_meta, threshold)
        tag_parts.append('cryo')
    if apply_group:
        new_test_prob = apply_group_rule(new_test_prob, test_pids, test_meta, threshold)
        tag_parts.append('group')

    tag = '_'.join(tag_parts)

    # Step 6: 保存提交
    print('\n' + '-' * 50)
    print('[Step 6] 生成提交文件')
    print('-' * 50)
    expected_oof_acc = group_acc if apply_group else (cryo_acc if apply_cryo else base_acc)
    print(f'预期 OOF acc (累计): {expected_oof_acc:.5f}  vs 基准 {base_acc:.5f}')
    sub_path = save_submission(test_pids, new_test_prob, threshold, tag)
    print(f'\n[done] 新提交文件: {sub_path}')


if __name__ == '__main__':
    run_postprocess()
