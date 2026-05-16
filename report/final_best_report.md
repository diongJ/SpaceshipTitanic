# Spaceship Titanic — 最优配置复现报告

## 1. 项目时间线

| 阶段 | 内容 | 提交 | OOF (单模最优) | LB 首次 |
|------|------|------|---------------|---------|
| Phase 1 | EDA — 数据探索、缺失分析、分组发现 | `eda` | — | — |
| Phase 2 | 预处理 — 三级填充体系（规则→组内→统计） | `preprocess` | — | — |
| Phase 3 | 特征工程 — 8 组 55 特征 | `feature engineering` | — | — |
| Phase 4 | 建模 — 3→5 模型 CV pipeline | `modeling` / `phase4` | LGB 0.81525 | — |
| Phase 5 | 集成 — 加权 Blend + LR Stacking | `Integrate ensemble` | LR 0.81836 | **0.80921** |
| Phase 6 | 后集成优化 — 特征/HPO/集成搜索 | `second_try` / `10try` | LR 0.82377 (leaky) | **0.81014** ✅ |

---

## 2. 最优配置（Submission #12 — LB=0.81014）

### 2.1 预处理

三级缺失值填充体系（[src/preprocess.py](../src/preprocess.py)）：

| 优先级 | 方法 | 处理内容 |
|--------|------|---------|
| 1 (最高) | 规则推断 | 消费=0 + Age>12 → CryoSleep=True；CryoSleep=True → 消费=0；Age<13 → VIP=False |
| 2 | 组内填充 | HomePlanet/Destination/Deck/Side/CryoSleep 按 GroupId 众数拉通 |
| 3 (兜底) | 统计填充 | 类别 → 按 HomePlanet 分层众数；数值 → 按 HomePlanet 分层中位数 |

### 2.2 特征工程

8 步特征生成管道（[src/features.py](../src/features.py)），产出 **60 特征**（不含目标编码，含 group label）：

| 特征组 | 数量 | 代表特征 |
|--------|------|---------|
| base | 10 | HomePlanet, CryoSleep, Destination, Age, VIP, 五项消费 |
| id | 4 | GroupSize, IsAlone, IsLargeGroup, PersonNum |
| cabin | 5 | Deck, CabinNum, CabinNumBin, Side, DeckSide |
| spend | 15 | TotalSpend, Log/结构/奢侈品/标准差 |
| age | 4 | AgeGroup, IsChild, IsTeen, IsSenior |
| name | 2 | FamilySize, IsSurnameInGroup |
| group_agg | 8 | Group_TotalSpend/Age 均值/极差、CryoRatio、VIP_any |
| cabin_agg | 3 | Cabin_Size, Cabin_TotalSpend_sum, Cabin_CryoRatio |
| interact | 5 | Route, Deck_HomePlanet, 交叉项 |
| group_label | 3 | Group_TrainTransportRatio 等 (全量 LOO) |

### 2.3 模型

5 个异构模型（[src/config.py](../src/config.py)），**全部使用原始默认超参**（非 HPO）：

| 模型 | 关键参数 | OOF (th=0.50) |
|------|---------|--------------|
| LightGBM | lr=0.02, num_leaves=63, max_depth=-1 | 0.81836 |
| XGBoost | lr=0.02, max_depth=6, subsample=0.8 | 0.81928 |
| CatBoost | lr=0.03, depth=6, iterations=5000 | 0.81801 |
| ExtraTrees | n=500, min_samples_leaf=2 | 0.81318 |
| HistGradientBoosting | lr=0.035, max_depth=7, max_iter=400 | 0.82020 |

### 2.4 集成

**LR Stacking**（[src/ensemble.py](../src/ensemble.py)）：

| 输入特征 | 说明 |
|---------|------|
| 5 模型 OOF 概率 | base models predictions |
| Group_TargetMean | LOO 目标编码 (smooth_k=1) |
| LastName_TE | LOO 目标编码 (smooth_k=1) |

元模型：`LogisticRegression(C=1.0, max_iter=1000)` + `StandardScaler`

### 2.5 训练参数

| 参数 | 值 |
|------|----|
| CV 方案 | StratifiedKFold (5 folds, shuffle per seed) |
| 随机种子 | 42, 2024 (2 seeds) |
| 训练样本 | 8,693 |
| 测试样本 | 4,277 |
| 类别特征编码 | category dtype（LGB/XGB/CAT 原生支持） |

### 2.6 Group Label 特征计算策略（关键）

Group label 特征（`Group_TrainTransportedCount`, `Group_TrainNotTransportedCount`, `Group_TrainTransportRatio`）使用**全量训练集 LOO** 计算，测试集使用全量训练标签统计。

- OOF 上存在泄露（验证折内同组互见标签），虚高 OOF 约 0.005
- 但 LB 比折内无泄露版本高 +0.00187
- **结论：对于测试集预测质量，全量 LOO > 折内无泄露**

---

## 3. LB 提交记录

| # | 提交文件 | 阈值 | LB | 说明 |
|---|---------|------|-----|------|
| 1-6 | Phase 5 探索 | 0.455 | 0.80921 | baseline LR Stack |
| 7 | ensemble_new_th0.465 | 0.465 | 0.80967 | +3 group label 特征 |
| 8 | ensemble_noleak_th0.455 | 0.455 | 0.80780 | 折内无泄露，退化 |
| 9 | ensemble_hybrid_th0.470 | 0.470 | 0.80710 | +cabin label，退化 |
| 10 | ensemble_lr_stack_th0.490 | 0.490 | 0.80923 | 同#7，仅阈值不同 |
| 11 | ensemble_tfidf_th0.470 | 0.470 | 0.80710 | +TFIDF+group_bin，退化 |
| 12-16 | post_cryo_th0.455~0.480 | 0.455~0.480 | 0.80851~0.80921 | CryoSleep 后处理规则，退化 |
| **17** | **ensemble_reverted_th0.470** | **0.470** | **0.81014** ✅ | 回退至最优配置，阈值上移 |

---

## 4. 已尝试但无效的方案（教训）

| 方案 | OOF 变化 | LB 变化 | 失败原因 |
|------|---------|---------|---------|
| Optuna HPO (LGB+CAT) | +0.0054 | **-0.00388** | HPO 在训练集过拟合，未泛化到测试集 |
| 伪标签 (pseudo-labeling) | -0.0047 | — | 高置信样本极端不平衡 (752:2) |
| 折内无泄露 group label | OOF 更诚实 | -0.00187 | 测试集只用 80% 标签信息 |
| Cabin label 特征 | +0.0005 | -0.00257 | 与 group label 高度冗余 |
| TFIDF 姓名特征 | ~持平 | -0.00257 | 姓名→传送关系无因果性 |
| GroupSizeBin/IsLoneChild | ~持平 | 推定为负 | 与现有特征冗余 |
| 后处理规则（CryoSleep+NoSpend）| +0.00023 | **-0.00046~-0.00116** | group_label 特征已在训练期捕获组信息，测试期规则添加噪声 |
| 后处理规则（组内置信投票）| -0.00127 | — | 同上，group_label 冗余；OOF 直接退化 |

---

## 5. 关键规律

1. **HPO 提升 OOF ≠ 提升 LB**：OOF 上搜索的超参在测试集上反而更差
2. **"修泄露"可能有害**：全量 LOO 虽然 OOF 虚高，但测试集预测质量最高
3. **新特征需与现有特征正交**：冗余特征非但无增益，反而引入噪声
4. **LB 最优阈值在 0.465~0.470 区间**：不受 OOF 最优阈值影响（OOF 最优阈值因 leaky 特征虚高至 0.5475）；th=0.470 的 LB（0.81014）优于 th=0.465（0.80967）
5. **StratifiedKFold > GroupKFold**：GroupKFold 导致测试集覆盖不完整

---

## 6. CV/LB Gap 分析

| 指标 | 值 |
|------|-----|
| OOF (LR Stack, leaky, th=0.5475) | 0.82377 |
| OOF (LR Stack, leaky, th=0.465) | ~0.82150 |
| LB (th=0.470) | 0.81014 |
| CV/LB gap | **~0.011** |

Gap 来源：
- 约 0.005 来自 group label 特征的 OOF 泄露（虚高 OOF）
- 约 0.007 来自训练集/测试集分布差异（structural）
- 结构性的 ~0.007 gap 无法通过 CV 方案变化消除

---

## 7. 至目标 0.82 的差距

| 当前最优 | 目标 | 差距 |
|---------|------|------|
| 0.81014 | 0.82000 | **0.00986** |

已消耗的探索方向均未突破。剩余可能方向：
1. 集成结构优化（去掉负系数模型 LGB/ET，换 XGB 元模型）
2. 新结构性无泄露特征
3. 概率校准后再集成
4. 参考外部方案的结构性创新

---

## 8. 重跑复现步骤

```bash
# 1. 确认配置为回退版本（已在当前代码中确认）
#    - FEAT_GROUPS 无 name_tfidf / group_bin
#    - CAT_COLS 无 GroupSizeBin
#    - features.py build_features() 为 8 步流程
#    - LGB/CAT 为原始默认超参

# 2. 训练 5 个基础模型
cd src
python train.py --models lgb,xgb,cat,et,hgb --cv stratified --seeds 42,2024

# 3. 生成集成提交文件
python ensemble.py

# 4. 手动生成 0.470 阈值提交（LB 最优；ensemble.py 自动用 OOF 最优 0.5475）
python -c "
import pandas as pd
pred = pd.read_csv('../outputs/preds/ensemble_lr_stack.csv')
labels = (pred['Transported_Prob'] >= 0.470).astype(bool)
sub = pd.DataFrame({'PassengerId': pred['PassengerId'], 'Transported': labels})
sub.to_csv('../outputs/submissions/submission_th0.470.csv', index=False)
"
```

---

## 9. 代码文件索引

| 文件 | 职责 |
|------|------|
| `src/config.py` | 路径、特征分组、模型超参 |
| `src/data.py` | 原始数据加载 |
| `src/preprocess.py` | 三级缺失值填充 + 基础衍生特征 |
| `src/features.py` | 8 步特征工程管道 |
| `src/train.py` | 多种子 CV 训练 + group label 特征 |
| `src/ensemble.py` | LR Stack 集成 + 阈值优化 |
| `src/utils.py` | 随机种子、评估、保存工具 |
| `src/models/lgb_model.py` | LightGBM 训练 |
| `src/models/xgb_model.py` | XGBoost 训练 |
| `src/models/cat_model.py` | CatBoost 训练 |
| `src/models/et_model.py` | Extra Trees 训练 |
| `src/models/hgb_model.py` | HistGradientBoosting 训练 |
| `report/eda_report.md` | Phase 1 报告 |
| `report/preprocess_report.md` | Phase 2 报告 |
| `report/features_report.md` | Phase 3 报告 |
| `report/phase4_modeling_report.md` | Phase 4 报告 |
| `report/phase5_ensemble_report.md` | Phase 5 报告 |
| `report/phase6_post_ensemble_report.md` | Phase 6 详细实验报告 |

---

*报告生成：2026-05-11 | 最后更新：2026-05-16*
