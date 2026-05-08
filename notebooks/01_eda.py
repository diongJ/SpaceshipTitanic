# ============================================================
# Spaceship Titanic — EDA
# 对应文档: report/eda_report.md
# 输出图表: report/eda_overview.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# 路径基于本脚本位置，无论从哪里运行都能找到数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'spaceship-titanic')
REPORT_DIR = os.path.join(BASE_DIR, '..', 'report')
os.makedirs(REPORT_DIR, exist_ok=True)
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# ------------------------------------------------------------
# 1. 数据加载
# ------------------------------------------------------------
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')

print('=== 1. 数据形状 ===')
print(f'训练集: {train.shape}')
print(f'测试集:  {test.shape}')

print('\n=== 字段类型 ===')
print(train.dtypes)

# ------------------------------------------------------------
# 2. 缺失率 & 目标分布
# ------------------------------------------------------------
print('\n=== 2. 缺失率（训练集）===')
missing = train.isna().mean().sort_values(ascending=False)
print(missing[missing > 0].map('{:.2%}'.format))

print('\n=== 目标变量分布 ===')
print(train['Transported'].value_counts(normalize=True).map('{:.2%}'.format))

# ------------------------------------------------------------
# 3. 辅助字段解析（用于后续分析，不修改原始 DataFrame）
# ------------------------------------------------------------
def parse_fields(df):
    df = df.copy()
    df['GroupId']   = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df['TotalSpend'] = df[SPEND_COLS].sum(axis=1, skipna=True)
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck']     = cabin[0]
    df['CabinNum'] = pd.to_numeric(cabin[1], errors='coerce')
    df['Side']     = cabin[2]
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[-1, 12, 17, 25, 40, 60, 200],
        labels=['<13', '13-17', '18-25', '26-40', '41-60', '60+']
    )
    return df

train_p = parse_fields(train)

# ------------------------------------------------------------
# 4. 关键业务规律验证
# ------------------------------------------------------------
print('\n=== 3. 规律1: CryoSleep vs Transported ===')
print(train_p.groupby('CryoSleep', dropna=False)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律2: CryoSleep=True 时消费是否严格为 0 ===')
cryo_true = train_p[train_p['CryoSleep'] == True]
nonzero = (cryo_true[SPEND_COLS] > 0).any(axis=1).sum()
print(f'CryoSleep=True 且有任意消费的乘客: {nonzero} / {len(cryo_true)}')

print('\n=== 规律3: 消费总额分桶 vs Transported ===')
train_nona = train_p.dropna(subset=['TotalSpend'])
bins   = [-1, 0, 100, 500, 2000, 10000, train_nona['TotalSpend'].max()]
labels = ['0', '1-100', '101-500', '501-2k', '2k-10k', '>10k']
train_nona = train_nona.copy()
train_nona['SpendBin'] = pd.cut(train_nona['TotalSpend'], bins=bins, labels=labels)
print(train_nona.groupby('SpendBin', observed=True)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律4: HomePlanet vs Transported ===')
print(train_p.groupby('HomePlanet', dropna=False)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律5: Deck vs Transported ===')
print(
    train_p.groupby('Deck', dropna=False)['Transported'].mean()
    .sort_values(ascending=False).map('{:.2%}'.format)
)

print('\n=== 规律6: Side vs Transported ===')
print(train_p.groupby('Side', dropna=False)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律7: AgeGroup vs Transported ===')
print(train_p.groupby('AgeGroup', observed=True)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律8: VIP vs Transported ===')
print(train_p.groupby('VIP', dropna=False)['Transported'].mean().map('{:.2%}'.format))

print('\n=== 规律9: Destination vs Transported ===')
print(train_p.groupby('Destination', dropna=False)['Transported'].mean().map('{:.2%}'.format))

# ------------------------------------------------------------
# 5. 组内结构分析
# ------------------------------------------------------------
print('\n=== 4. 组内结构分析 ===')
group_sizes = train_p.groupby('GroupId').size()
print('组大小分布（前 8）:')
print(group_sizes.value_counts().sort_index().head(8))
print(f'\n独行旅客 (size=1): {(group_sizes == 1).sum()} 组')
print(f'群体旅客 (size>1): {(group_sizes > 1).sum()} 组, 占乘客 {(group_sizes[group_sizes > 1].sum() / len(train_p)):.1%}')

print('\n独行 vs 群体传送率:')
print(train_p.groupby('IsAlone' if 'IsAlone' in train_p.columns else train_p['GroupSize'].eq(1).rename('IsAlone'))['Transported'].mean().map('{:.2%}'.format))

print('\n组大小 vs 传送率:')
print(train_p[train_p['GroupSize'] <= 8].groupby('GroupSize')['Transported'].mean().map('{:.2%}'.format))

print('\n组内传送结果一致性:')
group_std = train_p.groupby('GroupId')['Transported'].std().dropna()
all_same  = (group_std == 0).mean()
print(f'组内成员传送结果完全一致的组: {all_same:.1%}')

# ------------------------------------------------------------
# 6. 交叉特征分析
# ------------------------------------------------------------
print('\n=== 5. 交叉: CryoSleep × HomePlanet → Transported ===')
ct1 = train_p.groupby(['CryoSleep', 'HomePlanet'])['Transported'].mean().unstack()
print(ct1.map('{:.1%}'.format).to_string())

print('\n=== 交叉: Deck × CryoSleep → Transported ===')
ct2 = train_p.groupby(['Deck', 'CryoSleep'])['Transported'].mean().unstack()
print(ct2.map('{:.1%}'.format).to_string())

# ------------------------------------------------------------
# 7. 消费字段统计 & 相关性
# ------------------------------------------------------------
print('\n=== 6. 消费字段描述统计 ===')
print(train_p[SPEND_COLS].describe(percentiles=[.25, .5, .75, .9, .95]).round(1).to_string())

print('\n消费字段零值比例:')
print((train_p[SPEND_COLS] == 0).mean().map('{:.1%}'.format))

print('\nSpearman 相关系数（与 Transported）:')
target = train_p['Transported'].astype(int)
for col in SPEND_COLS + ['Age', 'TotalSpend']:
    valid = train_p[[col]].join(target).dropna()
    rho, pval = ss.spearmanr(valid[col], valid['Transported'])
    print(f'  {col:15s}: rho={rho:+.3f}  (p={pval:.2e})')

print('\n消费字段间 Spearman 相关（消费>0 的行）:')
spend_nonzero = train_p[train_p['TotalSpend'] > 0][SPEND_COLS]
print(spend_nonzero.corr(method='spearman').round(2).to_string())

# ------------------------------------------------------------
# 8. Train vs Test 分布对比
# ------------------------------------------------------------
print('\n=== 7. Train vs Test 分布对比 ===')
train_p['split'] = 'train'
test_p = parse_fields(test)
test_p['split'] = 'test'
df_all = pd.concat([train_p, test_p], ignore_index=True)

fmt = lambda x: f'{x:.2%}'
for col in ['HomePlanet', 'Destination', 'CryoSleep', 'Deck']:
    tbl = df_all.groupby('split')[col].value_counts(normalize=True).unstack().T
    print(f'\n-- {col} --')
    print(tbl.map(fmt).to_string())

# ------------------------------------------------------------
# 9. 缺失值规则推断潜力
# ------------------------------------------------------------
print('\n=== 8. 缺失值规则推断潜力 ===')

def rule_inference_stats(df, label):
    df = df.copy()
    df['AllSpendZero'] = (df[SPEND_COLS].fillna(0).sum(axis=1) == 0)
    infer_cryo = df[df['CryoSleep'].isna() & df['AllSpendZero'] & (df['Age'].fillna(99) > 12)]
    cryo_spend_na = df[df['CryoSleep'] == True][SPEND_COLS].isna().sum()
    print(f'\n[{label}]')
    print(f'  CryoSleep 缺失且消费全0 (Age>12) → 可推断为 True: {len(infer_cryo)} 条')
    print(f'  CryoSleep=True 且消费缺失 → 可填 0:')
    print(cryo_spend_na[cryo_spend_na > 0].to_string())

rule_inference_stats(train, '训练集')
rule_inference_stats(test,  '测试集')

# ------------------------------------------------------------
# 10. 可视化（保存至 report/eda_overview.png）
# ------------------------------------------------------------
print('\n=== 9. 生成可视化图表 ===')

COLORS = {'True': '#3B82F6', 'False': '#F97316'}

def stacked_bar(ax, df, col, title, dropna=True, sort=False):
    grp = df.groupby(col, dropna=dropna)['Transported'].mean().dropna()
    if sort:
        grp = grp.sort_values(ascending=False)
    f_rate = 1 - grp
    cats = grp.index.astype(str)
    x = range(len(cats))
    ax.bar(x, grp.values,   color=COLORS['True'],  alpha=0.85, label='True')
    ax.bar(x, f_rate.values, bottom=grp.values, color=COLORS['False'], alpha=0.85, label='False')
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(grp.values):
        ax.text(i, v / 2, f'{v:.0%}', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

fig = plt.figure(figsize=(18, 20))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# 目标分布
ax0 = fig.add_subplot(gs[0, 0])
vals = train_p['Transported'].value_counts()
ax0.pie(vals.values, labels=['True', 'False'],
        colors=[COLORS['True'], COLORS['False']],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10})
ax0.set_title('Target Distribution', fontsize=11, fontweight='bold')

# CryoSleep
stacked_bar(fig.add_subplot(gs[0, 1]), train_p, 'CryoSleep', 'CryoSleep vs Transported')

# HomePlanet
stacked_bar(fig.add_subplot(gs[0, 2]), train_p, 'HomePlanet', 'HomePlanet vs Transported')

# Deck（按传送率降序）
stacked_bar(fig.add_subplot(gs[1, 0]), train_p, 'Deck', 'Deck vs Transported', sort=True)

# Destination（缩短标签）
ax_dest = fig.add_subplot(gs[1, 1])
dest_map = {'55 Cancri e': '55Cnc', 'PSO J318.5-22': 'PSO', 'TRAPPIST-1e': 'TRAP'}
train_dest = train_p.copy()
train_dest['Destination'] = train_dest['Destination'].map(dest_map).fillna(train_dest['Destination'])
stacked_bar(ax_dest, train_dest, 'Destination', 'Destination vs Transported')

# AgeGroup
stacked_bar(fig.add_subplot(gs[1, 2]), train_p, 'AgeGroup', 'AgeGroup vs Transported')

# Log(TotalSpend+1) 直方图
ax6 = fig.add_subplot(gs[2, 0])
for transported, color in [(True, COLORS['True']), (False, COLORS['False'])]:
    subset = train_p[train_p['Transported'] == transported]['TotalSpend'].dropna()
    ax6.hist(np.log1p(subset), bins=30, alpha=0.6, color=color,
             label=f'T={transported}', density=True)
ax6.set_title('Log(TotalSpend+1) Distribution', fontsize=11, fontweight='bold')
ax6.set_xlabel('log1p(TotalSpend)', fontsize=9)
ax6.legend(fontsize=8)

# GroupSize vs 传送率
ax7 = fig.add_subplot(gs[2, 1])
gs_rate = train_p[train_p['GroupSize'] <= 8].groupby('GroupSize')['Transported'].mean()
ax7.bar(gs_rate.index, gs_rate.values, color=COLORS['True'], alpha=0.85)
ax7.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax7.set_title('GroupSize vs Transport Rate', fontsize=11, fontweight='bold')
ax7.set_xlabel('Group Size', fontsize=9)
ax7.set_ylim(0.3, 0.75)
for i, v in zip(gs_rate.index, gs_rate.values):
    ax7.text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=8)

# VIP & Side
ax8 = fig.add_subplot(gs[2, 2])
categories = ['VIP=False', 'VIP=True', 'Side=P', 'Side=S']
rates = [
    train_p[train_p['VIP'] == False]['Transported'].mean(),
    train_p[train_p['VIP'] == True]['Transported'].mean(),
    train_p[train_p['Side'] == 'P']['Transported'].mean(),
    train_p[train_p['Side'] == 'S']['Transported'].mean(),
]
bar_colors = [COLORS['True'] if r > 0.5 else COLORS['False'] for r in rates]
ax8.bar(categories, rates, color=bar_colors, alpha=0.85)
ax8.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax8.set_title('VIP & Side vs Transported', fontsize=11, fontweight='bold')
ax8.set_ylim(0.3, 0.75)
for i, v in enumerate(rates):
    ax8.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

# Spearman 相关系数（横向条形）
ax9 = fig.add_subplot(gs[3, :])
num_cols = SPEND_COLS + ['Age', 'TotalSpend']
rhos = []
target = train_p['Transported'].astype(int)
for col in num_cols:
    valid = train_p[[col]].join(target).dropna()
    rho, _ = ss.spearmanr(valid[col], valid['Transported'])
    rhos.append(rho)
rho_series = pd.Series(rhos, index=num_cols).sort_values()
bar_cols = [COLORS['False'] if v < 0 else COLORS['True'] for v in rho_series.values]
ax9.barh(rho_series.index, rho_series.values, color=bar_cols, alpha=0.85)
ax9.axvline(0, color='black', linewidth=0.8)
ax9.set_title('Spearman Correlation with Transported', fontsize=11, fontweight='bold')
ax9.set_xlabel('Spearman rho', fontsize=9)
for i, v in enumerate(rho_series.values):
    ax9.text(
        v - 0.005 if v < 0 else v + 0.005, i,
        f'{v:.3f}', va='center',
        ha='right' if v < 0 else 'left', fontsize=9
    )

out_path = os.path.join(REPORT_DIR, 'eda_overview.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {out_path}')
plt.show()
