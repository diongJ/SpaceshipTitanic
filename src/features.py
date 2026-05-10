# ============================================================
# 阶段 3 — 特征工程：每类特征独立函数，可按组开关（对应 strategy.md 第五节）
# 输入：preprocess() 产出的 train_clean / test_clean
# 处理顺序：ID → Cabin → Spend → Age → Name → 组聚合 → 舱聚合 → 交互
# ============================================================
import numpy as np
import pandas as pd
from config import SPEND_COLS


# ── 1. ID 解析特征 ─────────────────────────────────────────
# GroupId / PersonNum / GroupSize / IsAlone 已由 preprocess 生成

def add_id_features(df):
    df = df.copy()
    df['IsLargeGroup'] = (df['GroupSize'] >= 4).astype(int)
    return df


# ── 2. Cabin 特征 ──────────────────────────────────────────

def add_cabin_features(df):
    df = df.copy()
    df['CabinNumBin'] = pd.cut(
        df['CabinNum'],
        bins=[-1, 300, 600, 900, 1200, 1500, 2000],
        labels=['0-300', '300-600', '600-900', '900-1200', '1200-1500', '1500+']
    )
    df['DeckSide'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)
    return df


# ── 3. 消费特征 ────────────────────────────────────────────
# TotalSpend / HasSpend / NumSpendCategories 已由 preprocess 生成

def add_spend_features(df):
    df = df.copy()
    for col in SPEND_COLS:
        df[f'Log{col}'] = np.log1p(df[col])
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
    df['MaxSpend'] = df[SPEND_COLS].max(axis=1)
    df['SpendStd'] = df[SPEND_COLS].std(axis=1).fillna(0)
    df['LuxurySpend'] = df[['Spa', 'VRDeck', 'RoomService']].sum(axis=1)
    df['EssentialSpend'] = df[['FoodCourt', 'ShoppingMall']].sum(axis=1)
    df['LuxuryRatio'] = df['LuxurySpend'] / (df['TotalSpend'] + 1)
    return df


# ── 4. 年龄特征 ────────────────────────────────────────────
# AgeGroup / IsChild / IsSenior 已由 preprocess 生成

def add_age_features(df):
    df = df.copy()
    df['IsTeen'] = ((df['Age'] >= 13) & (df['Age'] < 18)).astype(int)
    return df


# ── 5. 姓名特征 ────────────────────────────────────────────

def add_name_features(df):
    df = df.copy()
    df['FamilySize'] = df['SurnameGroupSize']

    # 组内成员同姓且姓名已知 → 家庭同行
    group_surname_nunique = df.groupby('GroupId')['LastName'].transform('nunique')
    df['IsSurnameInGroup'] = (
        (df['GroupSize'] > 1)
        & (group_surname_nunique == 1)
        & (df['LastName'] != 'Unknown')
    ).astype(int)
    return df


# ── 6. 组级聚合特征 ────────────────────────────────────────

def add_group_agg_features(df):
    df = df.copy()
    grp = df.groupby('GroupId')
    df['Group_TotalSpend_mean'] = grp['TotalSpend'].transform('mean')
    df['Group_TotalSpend_sum'] = grp['TotalSpend'].transform('sum')
    df['Group_Age_mean'] = grp['Age'].transform('mean')
    df['Group_Age_min'] = grp['Age'].transform('min')
    df['Group_Age_max'] = grp['Age'].transform('max')
    df['Group_HomePlanet_nunique'] = grp['HomePlanet'].transform('nunique')
    df['Group_CryoRatio'] = grp['CryoSleep'].transform(
        lambda x: x.astype(float).mean()
    )
    df['Group_VIP_any'] = grp['VIP'].transform(
        lambda x: x.astype(float).max()
    ).astype(int)
    return df


# ── 7. Cabin 级聚合特征 ────────────────────────────────────

def add_cabin_agg_features(df):
    df = df.copy()
    # CabinNum 缺失时舱室 key 无意义，使用 .loc 赋 NaN 保证 groupby 正确排除
    has_cabin = df['CabinNum'].notna() & df['Deck'].notna() & df['Side'].notna()
    df['_ck'] = (
        df['Deck'].astype(str) + '_'
        + df['CabinNum'].astype(str) + '_'
        + df['Side'].astype(str)
    )
    df.loc[~has_cabin, '_ck'] = np.nan

    df['Cabin_Size'] = df.groupby('_ck')['_ck'].transform('count')
    df['Cabin_TotalSpend_sum'] = df.groupby('_ck')['TotalSpend'].transform('sum')
    df['Cabin_CryoRatio'] = df.groupby('_ck')['CryoSleep'].transform(
        lambda x: x.astype(float).mean()
    )
    df.drop(columns=['_ck'], inplace=True)
    return df


# ── 8. 交互特征 ────────────────────────────────────────────

def add_interaction_features(df):
    df = df.copy()
    df['Cryo_x_TotalSpend'] = df['CryoSleep'].astype(float) * df['TotalSpend']
    df['Route'] = df['HomePlanet'].astype(str) + '_' + df['Destination'].astype(str)
    df['Deck_HomePlanet'] = df['Deck'].astype(str) + '_' + df['HomePlanet'].astype(str)
    df['Age_x_VIP'] = df['Age'] * df['VIP'].astype(float)
    df['IsAlone_x_TotalSpend'] = df['IsAlone'] * df['TotalSpend']
    return df


# ── 9. Fold 内目标编码（无泄漏）──────────────────────────────

def _group_stats(col, y_arr):
    """计算分组统计：返回 (agg DataFrame[sum,count], global_mean)"""
    global_mean = float(y_arr.mean())
    df = pd.DataFrame({'g': col, 'y': y_arr})
    agg = df.groupby('g')['y'].agg(['sum', 'count'])
    return agg, global_mean


def _loo_encode(col, y_arr, agg, global_mean, smooth_k):
    """训练集 LOO 编码（排除自身）"""
    g_sum = agg['sum'].reindex(col).values.astype(float)
    g_cnt = agg['count'].reindex(col).values.astype(float)
    loo_sum = g_sum - y_arr
    loo_cnt = g_cnt - 1
    return np.where(
        loo_cnt > 0,
        (loo_sum + smooth_k * global_mean) / (loo_cnt + smooth_k),
        global_mean
    )


def _apply_encode(col, agg, global_mean, smooth_k):
    """Val/Test 编码：直接用训练折统计"""
    g_sum = agg['sum'].reindex(col).fillna(0).values.astype(float)
    g_cnt = agg['count'].reindex(col).fillna(0).values.astype(float)
    return np.where(
        g_cnt > 0,
        (g_sum + smooth_k * global_mean) / (g_cnt + smooth_k),
        global_mean
    )


def compute_fold_te(te_src_tr, y_tr, te_src_val, te_src_test, smooth_k=10):
    """
    在单个 fold 的训练子集上计算目标编码，无泄漏地应用到 val 和 test。

    te_src_*: DataFrame，必须含 ['GroupId', 'LastName', 'DeckSide'] 列。
    y_tr    : numpy array，与 te_src_tr 行对齐的训练标签。
    返回 (te_tr, te_val, te_test) DataFrame，各含三列目标编码特征。
    """
    y_arr = np.array(y_tr, dtype=float)
    tr_dict, val_dict, test_dict = {}, {}, {}

    # GroupId
    col_tr  = te_src_tr['GroupId'].values
    col_val = te_src_val['GroupId'].values
    col_te  = te_src_test['GroupId'].values
    agg, gm = _group_stats(col_tr, y_arr)
    tr_dict['Group_TargetMean']   = _loo_encode(col_tr, y_arr, agg, gm, smooth_k)
    val_dict['Group_TargetMean']  = _apply_encode(col_val, agg, gm, smooth_k)
    test_dict['Group_TargetMean'] = _apply_encode(col_te,  agg, gm, smooth_k)

    # LastName（Unknown → __unk__）
    def _clean_name(arr):
        a = arr.astype(str).copy()
        a[a == 'Unknown'] = '__unk__'
        return a

    ln_tr  = _clean_name(te_src_tr['LastName'].values)
    ln_val = _clean_name(te_src_val['LastName'].values)
    ln_te  = _clean_name(te_src_test['LastName'].values)
    agg, gm = _group_stats(ln_tr, y_arr)
    tr_dict['LastName_TE']   = _loo_encode(ln_tr, y_arr, agg, gm, smooth_k)
    val_dict['LastName_TE']  = _apply_encode(ln_val, agg, gm, smooth_k)
    test_dict['LastName_TE'] = _apply_encode(ln_te,  agg, gm, smooth_k)

    # DeckSide
    ds_tr  = te_src_tr['DeckSide'].astype(str).values
    ds_val = te_src_val['DeckSide'].astype(str).values
    ds_te  = te_src_test['DeckSide'].astype(str).values
    agg, gm = _group_stats(ds_tr, y_arr)
    tr_dict['DeckSide_TE']   = _loo_encode(ds_tr, y_arr, agg, gm, smooth_k)
    val_dict['DeckSide_TE']  = _apply_encode(ds_val, agg, gm, smooth_k)
    test_dict['DeckSide_TE'] = _apply_encode(ds_te,  agg, gm, smooth_k)

    return pd.DataFrame(tr_dict), pd.DataFrame(val_dict), pd.DataFrame(test_dict)


# ── 10. TFIDF 姓名特征 ─────────────────────────────────────

def add_tfidf_name_features(df, n_tfidf=1000, n_svd=5):
    """对 LastName 做 TFIDF 字符级向量化 → TruncatedSVD 降维，生成姓名文本特征。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    names = df['LastName'].astype(str).values
    names[names == 'Unknown'] = '__unk__'

    vec = TfidfVectorizer(max_features=n_tfidf, analyzer='char_wb', ngram_range=(3, 5))
    tfidf_mat = vec.fit_transform(names)

    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    tfidf_pca = svd.fit_transform(tfidf_mat)

    for i in range(n_svd):
        df[f'Name_TFIDF_{i+1}'] = tfidf_pca[:, i]
    return df


# ── 11. 组规模分箱 + 独行儿童 ───────────────────────────────

def add_group_bin_features(df):
    df = df.copy()
    df['GroupSizeBin'] = pd.cut(
        df['GroupSize'],
        bins=[0, 1, 2, 4, 200],
        labels=['solo', 'pair', 'medium', 'large']
    )
    df['IsLoneChild'] = ((df['GroupSize'] == 1) & (df['Age'] < 13)).astype(int)
    return df


# ── 主入口 ────────────────────────────────────────────────

def build_features(train_df, test_df):
    """
    train+test 拼接后统一做特征工程，确保聚合统计（组/舱）覆盖完整。
    目标编码（compute_fold_te）需在 CV fold 内调用，不在此处计算。
    返回 (train_feat, test_feat)。
    """
    print('=== build_features ===')
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], ignore_index=True)

    print('[1/8] ID features...')
    df = add_id_features(df)
    print('[2/8] Cabin features...')
    df = add_cabin_features(df)
    print('[3/8] Spend features...')
    df = add_spend_features(df)
    print('[4/8] Age features...')
    df = add_age_features(df)
    print('[5/8] Name features...')
    df = add_name_features(df)
    print('[6/8] Group aggregation features...')
    df = add_group_agg_features(df)
    print('[7/8] Cabin aggregation features...')
    df = add_cabin_agg_features(df)
    print('[8/8] Interaction features...')
    df = add_interaction_features(df)

    train_feat = df.iloc[:n_train].reset_index(drop=True)
    test_feat  = df.iloc[n_train:].reset_index(drop=True)
    print(f'  done: train={train_feat.shape}, test={test_feat.shape}')
    return train_feat, test_feat
