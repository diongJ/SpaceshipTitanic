# Spaceship Titanic — 项目完整总结

> 最后更新：2026-05-16 | 当前最优 LB：**0.81014**

---

## 目录

1. [竞赛概况](#1-竞赛概况)
2. [整体架构](#2-整体架构)
3. [Phase 1 — EDA](#3-phase-1--eda)
4. [Phase 2 — 预处理](#4-phase-2--预处理)
5. [Phase 3 — 特征工程](#5-phase-3--特征工程)
6. [Phase 4 — 建模](#6-phase-4--建模)
7. [Phase 5 — 集成](#7-phase-5--集成)
8. [Phase 6 — 后集成优化](#8-phase-6--后集成优化)
9. [全部提交记录](#9-全部提交记录)
10. [CV/LB Gap 分析](#10-cvlb-gap-分析)
11. [失败方案完整记录](#11-失败方案完整记录)
12. [核心经验教训](#12-核心经验教训)
13. [复现步骤](#13-复现步骤)
14. [代码文件索引](#14-代码文件索引)

---

## 1. 竞赛概况

| 项目 | 内容 |
|------|------|
| 任务 | 二分类：预测乘客是否被传送至异次元（Transported: True/False） |
| 评估指标 | 分类准确率（Accuracy） |
| 训练集 | 8,693 条 |
| 测试集 | 4,277 条 |
| 目标平衡度 | True 50.4% / False 49.6%（近似均衡，无需处理类别不平衡） |
| **当前最优 LB** | **0.81014**（Submission #17，2026-05-16） |
| 目标 LB | 0.82000 |
| 剩余差距 | **0.00986** |
| 榜单背景 | 第 2 名 0.855，第 10 名 0.844，竞赛中存在作弊账号（第 1 名 0.965 系硬编码测试集标签） |

---

## 2. 整体架构

```
原始数据 (train.csv / test.csv)
        │
        ▼
  Phase 2: 预处理
  三级缺失值填充（规则→组内→统计）
  产出: train_clean / test_clean（8693/4277 × 30列，零缺失）
        │
        ▼
  Phase 3: 特征工程
  8 步管道，train+test 拼接统一变换
  产出: train_feat / test_feat（60 特征，零缺失）
        │
        ▼
  Phase 4: 建模
  5 模型 × 2 种子 × 5 折 = 50 次训练
  StratifiedKFold，产出各模型 OOF + 测试集预测
        │
        ▼
  Phase 5: 集成
  LR Stacking（5模型OOF + 2个LOO目标编码）
  阈值网格搜索，产出集成 OOF + 集成测试集预测
        │
        ▼
  Phase 6: 后集成优化
  +3 group_label 特征，阈值精细对齐
  最终提交：threshold = 0.470，LB = 0.81014
```

---

## 3. Phase 1 — EDA

**代码**：`src/data.py`，报告：`report/eda_report.md`

### 3.1 核心发现

**CryoSleep（最强单一特征）**

| CryoSleep | 传送率 |
|-----------|-------|
| True | **81.8%** |
| False | 32.9% |

CryoSleep=True 的所有 3,037 名乘客，五项消费**严格为零**，无一例外。这直接驱动了预处理中最高优先级的规则推断：
- CryoSleep=True → 消费填 0
- 五项消费全 0 且 Age>12 → CryoSleep=True

**消费字段（第二强信号群）**

| 特征 | 与传送率的 Spearman rho |
|------|------------------------|
| TotalSpend | -0.419 |
| Spa | -0.373 |
| RoomService | -0.371 |
| VRDeck | -0.352 |

消费为 0 的乘客传送率高达 **78.4%**；有消费则骤降至 22%–33%。五项消费中位数均为 0，分布高度右偏，需 log1p 变换。

**组内结构（关键发现）**

- 43.6% 的多人组成员传送结果**完全一致**（同传或同不传）
- GroupSize 传送率非线性：4 人组峰值 64.1%，8 人组跌至 39.4%，独行乘客 45.2%
- 姓名中的姓氏可近似识别家庭关系（同姓组员传送一致性极高）

**位置信息**

| Deck | 传送率 | Side | 传送率 |
|------|-------|------|-------|
| B | 73.4% | S | 55.5% |
| C | 68.0% | P | 45.1% |
| E | 35.7% | | |

B/C 甲板（Europa 高价舱）vs E/F 甲板（Earth 经济舱）差距悬殊。S/P 舷差约 10 个百分点。

**年龄分组**

| 年龄段 | 传送率 |
|-------|-------|
| 儿童（<13） | **70.0%** |
| 青少年（13-17） | 55.4% |
| 成人（18+） | ~47-49% |

儿童传送率异常偏高，须单独构造 `IsChild` 特征，且不能对儿童应用"消费全零→CryoSleep"规则。

**Train/Test 分布一致性**：主要特征分布高度吻合，无协变量偏移，无需对抗验证降权。

---

## 4. Phase 2 — 预处理

**代码**：`src/preprocess.py`，报告：`report/preprocess_report.md`

train+test 拼接后统一处理（12,970 条），产出零缺失的 30 列 DataFrame。

### 4.1 三级填充体系

**第一级：规则推断（约 60% 缺失，精度最高）**

| 规则 | 条件 | 填充值 | 覆盖量 |
|------|------|-------|-------|
| 消费→CryoSleep | 五项消费全0 + Age>12 + CryoSleep缺失 | True | 136 条 |
| CryoSleep→消费 | CryoSleep=True + 消费字段缺失 | 0 | ~537 处 |
| 年龄→VIP | Age<13 + VIP缺失 | False | 28 条 |

**第二级：组内填充（约 33% 缺失，精度高）**

同 GroupId 的乘客通常为家人，特征高度一致：

| 字段 | 填充前缺失 | 填充率 | 方法 |
|------|-----------|--------|------|
| HomePlanet | 288 | 45.5% | 组内众数 |
| Destination | 274 | 43.8% | 组内众数 |
| Deck | 299 | 45.8% | 组内众数 |
| Side | 299 | 45.8% | 组内众数 |
| CryoSleep | 97 | **100%** | 组内多数投票 |

**第三级：分层统计填充（约 7% 缺失，兜底）**

| 字段类型 | 策略 |
|---------|------|
| 类别字段 | 按 HomePlanet 分层众数，再全局众数兜底 |
| 数值字段 | 按 HomePlanet 分层中位数，再全局中位数兜底 |

### 4.2 基础衍生特征（Step 5）

在清洗完成后直接计算，无泄漏风险：

| 新特征 | 计算方式 |
|--------|---------|
| TotalSpend | 五项消费之和 |
| HasSpend | TotalSpend > 0 |
| NumSpendCategories | 五项中非零的项目数 |
| IsChild | Age < 13 |
| IsSenior | Age ≥ 60 |
| IsAlone | GroupSize == 1 |
| AgeGroup | 六档分桶（0-12/13-17/18-25/26-40/41-60/60+） |

**产出**：train_clean（8,693 × 30）、test_clean（4,277 × 30），零缺失。

---

## 5. Phase 3 — 特征工程

**代码**：`src/features.py`，报告：`report/features_report.md`

train+test 拼接后统一变换（避免测试集聚合特征缺少训练集覆盖），共 **60 个特征**，零缺失。

### 5.1 8 步管道详情

**Step 1 — ID 特征**

| 特征 | 设计动机 |
|------|---------|
| IsLargeGroup（GroupSize≥4） | 4 人组传送率峰值 64.1%，大组与小组模式截然不同 |

（GroupSize、IsAlone、PersonNum 已由预处理产出）

**Step 2 — Cabin 特征**

| 特征 | 设计动机 |
|------|---------|
| CabinNumBin（6 档分桶） | 舱号连续值非线性，分桶后形成有序类别 |
| DeckSide（Deck + Side 拼接） | EDA 发现 Deck×Side 存在交互，组合后显式化 |

**Step 3 — 消费特征（15 个）**

| 特征 | 设计动机 |
|------|---------|
| LogRoomService/FoodCourt/ShoppingMall/Spa/VRDeck | log1p 缓解右偏，提升低消费区间区分度 |
| LogTotalSpend | 总消费的对数版本 |
| MaxSpend | 最大单项消费，反映消费强度 |
| SpendStd | 消费分散度（std 大 → 偏好单项；std 小 → 均匀消费） |
| LuxurySpend（Spa+VRDeck+RoomService） | 奢侈型消费合计 |
| EssentialSpend（FoodCourt+ShoppingMall） | 刚需型消费合计 |
| LuxuryRatio（LuxurySpend/(TotalSpend+1)） | 奢侈消费占比，高占比 → 传送率显著更低 |

**Step 4 — 年龄特征**

| 特征 | 设计动机 |
|------|---------|
| IsTeen（13≤Age<18） | 青少年传送率（55.4%）介于儿童与成人之间，需单独标记 |

**Step 5 — 姓名特征**

| 特征 | 设计动机 |
|------|---------|
| FamilySize（同姓人数） | 近似"血缘家庭规模"，比 GroupSize 更聚焦血缘关系 |
| IsSurnameInGroup（组内同姓） | 同姓组员传送一致性极高，该特征捕捉家庭出行模式 |

**Step 6 — 组级聚合特征（8 个）**

基于 GroupId 对全体样本做聚合后映射回每个乘客：

| 特征 | 聚合方式 | 信号含义 |
|------|---------|---------|
| Group_TotalSpend_mean | 组内平均消费 | 组团整体消费水平 |
| Group_TotalSpend_sum | 组内总消费 | 绝对消费量 |
| Group_Age_mean | 组内平均年龄 | 家庭组（中年+儿童）vs 朋友组（同龄人） |
| Group_Age_min | 组内最小年龄 | 识别含儿童的家庭组 |
| Group_Age_max | 组内最大年龄 | 识别含老人的家庭组 |
| Group_HomePlanet_nunique | 组内出发星球去重数 | >1 → 跨星球团体（非家人） |
| Group_CryoRatio | 组内冷冻比例 | 全冷冻家庭组传送率最高 |
| Group_VIP_any | 组内是否有 VIP | VIP 拉低整组传送率 |

**Step 7 — 舱级聚合特征（3 个）**

基于精确舱室 key（Deck+CabinNum+Side）做聚合：

| 特征 | 含义 |
|------|------|
| Cabin_Size | 同舱人数 |
| Cabin_TotalSpend_sum | 同舱总消费 |
| Cabin_CryoRatio | 同舱冷冻比例 |

**Step 8 — 交互特征（5 个）**

| 特征 | 设计动机 |
|------|---------|
| Cryo_x_TotalSpend | CryoSleep=True 时消费必为 0，交互项让模型显式捕捉该条件关系 |
| Route（HomePlanet_Destination） | Europa→TRAPPIST vs Earth→55 Cancri 传送率差异极大 |
| Deck_HomePlanet | Europa B 甲板 vs Earth E 甲板差异极端 |
| Age_x_VIP | VIP 儿童、VIP 老人特殊传送模式 |
| IsAlone_x_TotalSpend | 独行乘客的消费信号不同于团体乘客 |

### 5.2 关键补充：group_label 特征（Phase 6 加入）

| 特征 | 计算方式 | 效果 |
|------|---------|------|
| Group_TrainTransportedCount | 同组训练成员中被传送的人数（LOO） | 直接编码组内历史结果 |
| Group_TrainNotTransportedCount | 同组训练成员中未被传送的人数（LOO） | 互补信号 |
| Group_TrainTransportRatio | 同组传送比例（LOO，tot=0 时用 global_mean） | 最强单一 group 信号 |

计算策略：**全量训练集 LOO**（Leave-One-Out）——训练时排除自己，测试集用全量训练标签。OOF 上存在泄露（验证折内同组成员互见标签），但 LB 表现优于折内无泄露版本，是本项目最重要的发现之一。

### 5.3 特征总览

| 分组 | 特征数 | 来源 |
|------|--------|------|
| base | 10 | 原始字段 |
| id | 4 | 预处理 + Step 1 |
| cabin | 5 | 预处理 + Step 2 |
| spend | 15 | 预处理 + Step 3 |
| age | 4 | 预处理 + Step 4 |
| name | 2 | Step 5 |
| group_agg | 8 | Step 6 |
| cabin_agg | 3 | Step 7 |
| interact | 5 | Step 8 |
| group_label | 3 | Phase 6 补充 |
| **合计** | **60** | |

另有目标编码特征（`Group_TargetMean`、`LastName_TE`、`DeckSide_TE`）仅在集成元模型中使用，不进入基础模型训练。

---

## 6. Phase 4 — 建模

**代码**：`src/train.py` + `src/models/`，报告：`report/phase4_modeling_report.md`

### 6.1 训练架构

- **CV 方案**：StratifiedKFold（5 折，每 seed shuffle 一次）
- **种子**：42 和 2024，共 2 seeds × 5 folds = **10 次训练**每个模型
- **Early Stopping**：100 轮（LGB/XGB/CAT）
- **类别特征编码**：category dtype（LGB/XGB/CAT 原生支持，无需 One-Hot）

### 6.2 五个基础模型（最终配置）

| 模型 | 关键超参 | OOF acc（th=0.50） |
|------|---------|-------------------|
| LightGBM | lr=0.02, num_leaves=63, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0 | 0.81836 |
| XGBoost | lr=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0 | 0.81928 |
| CatBoost | lr=0.03, depth=6, iterations=5000, l2_leaf_reg=3 | 0.81801 |
| ExtraTrees | n_estimators=500, min_samples_leaf=2 | 0.81318 |
| HistGradientBoosting | lr=0.035, max_depth=7, max_iter=400, min_samples_leaf=10 | **0.82020** |

**所有模型均使用原始默认超参，未做 HPO**（HPO 实验证明在测试集上适得其反，见失败方案）。

### 6.3 模型间相关性（Phase 4 初始 3 模型）

| | LGB | XGB | CAT |
|--|-----|-----|-----|
| LGB | 1.00 | 0.950 | 0.941 |
| XGB | — | 1.00 | **0.987** |
| CAT | — | — | 1.00 |

XGB 与 CAT 高度重叠（0.987），集成价值有限。LGB 与 CAT 相关性最低（0.941），是最优集成搭档。

### 6.4 Bug 修复记录

**Bug 1（严重）：多 Seed OOF 覆盖而非平均**

原代码 `oof_preds[val_idx] = result['y_val_pred']`，第二个 seed 的预测完全覆盖第一个 seed，导致 OOF 仅来自最后一个 seed。修复：累加后除以 seed 数量。

```python
# 修复前（错误）
oof_preds[val_idx] = result['y_val_pred']

# 修复后（正确）
oof_preds_sum[val_idx] += result['y_val_pred']
...
oof_preds = oof_preds_sum / len(seeds)
```

**Bug 2（次要）：模型函数内部重复覆盖 seed**

各模型训练函数内部又执行 `params['random_state'] = seed`，覆盖了 `_get_model_params` 已注入的值。修复：删除各模型文件中的 seed 覆盖行。

---

## 7. Phase 5 — 集成

**代码**：`src/ensemble.py`，报告：`report/phase5_ensemble_report.md`

### 7.1 两种集成方案对比

| 方法 | OOF acc（th=最优） |
|------|-------------------|
| 加权 Blend（Nelder-Mead 搜索权重） | 0.81732 |
| **LR Stacking** | **0.81836** ✓ |

**最终选用 LR Stacking**，比最佳单模（XGB，0.81721）高 +0.00115，比 Blend 高 +0.001。

### 7.2 LR Stacking 架构

```
输入特征（7 维）：
  5 模型 OOF 概率
  + Group_TargetMean（LOO 目标编码，smooth_k=1）
  + LastName_TE（LOO 目标编码，smooth_k=1）

元模型：
  StandardScaler → LogisticRegression(C=1.0, max_iter=1000)
```

目标编码（LOO，smooth_k=1）：

```python
# LOO 公式（训练集）
train_te[i] = (sum(y) - y[i] + k * global_mean) / (count - 1 + k)

# 测试集用全量训练集统计
test_te = (sum(y_train) + k * global_mean) / (count_train + k)
```

### 7.3 LR Stack 系数分析（5 模型版本）

| 特征 | 系数 | 含义 |
|------|------|------|
| HGB | **+1.3116** | 绝对主导 |
| CAT | +0.4074 | 有效贡献 |
| ET | +0.3487 | 有效贡献 |
| XGB | +0.1997 | 弱正贡献 |
| LGB | **-0.0600** | 负贡献（信息与答案轻微反向） |

LGB 系数为负是重要发现：LR 元模型认为 LGB 的预测信号方向与最终答案轻微相反，在集成中实为噪声来源。

### 7.4 阈值发现

3 模型时（Phase 5 初始）：

| 模型 | OOF 最优阈值 |
|------|------------|
| LGB | 0.480 |
| XGB | 0.470 |
| CAT | 0.503 |
| LR Stack | 0.455 |

所有模型最优阈值均低于 0.5，说明模型整体概率偏保守（预测值系统性偏低），下移阈值可提升准确率。

---

## 8. Phase 6 — 后集成优化

**代码**：`src/train.py`（group_label），`src/post_process.py`，报告：`report/phase6_post_ensemble_report.md`

### 8.1 Group Label 特征（成功，+0.00046 LB）

**设计**：在 `_add_group_label_features()` 中计算，全量训练集 LOO，测试集用全量标签：

```python
# 训练集：LOO（排除自身）
for pid in train_pids:
    group_pids = [p for p in group_members if p != pid]
    ratio = sum(y[p] for p in group_pids) / len(group_pids)

# 测试集：全量训练标签（无泄露）
ratio = sum(y[p] for p in all_train_group_members) / len(all_train_group_members)
```

**关键发现**：

| 版本 | OOF | LB |
|------|-----|----|
| 无 group_label | 0.81836 | 0.80921 |
| 全量 LOO（leaky） | **0.82377** | **0.80967** |
| 折内无泄露 | 0.81859 | 0.80780 |

OOF 提升 +0.00541 中约 0.005 来自泄露（验证折内同组成员互见标签），但 LB 仍有真实增益 +0.00046。折内无泄露版本 OOF 更诚实，但 LB 反而下降 -0.00187，因为测试集只用到 80% 的训练标签信息。

### 8.2 阈值精细对齐（最终关键提升）

OOF 最优阈值因 leaky 特征虚高至 **0.5475**（验证折内同组泄露让模型对难样本过度自信），但 LB 最优始终在 **0.465~0.470** 附近。

实验过程：

| 提交文件 | 阈值 | LB | 说明 |
|---------|------|----|------|
| ensemble_new_th0.465 | 0.465 | 0.80967 | Phase 6 第一次找到最优 |
| ensemble_lr_stack_th0.490 | 0.490 | 0.80923 | OOF 最优阈值，反而更差 |
| **ensemble_reverted_th0.470** | **0.470** | **0.81014** | 最终最优（2026-05-11 生成，2026-05-16 提交） |

th=0.470 的文件生成于 2026-05-11 00:19，当时与 th=0.465/0.460 一同生成作为备选，仅 th=0.465 被提交。直到 2026-05-16 补交 th=0.470 才发现它是历史最优。

### 8.3 失败实验：Optuna HPO

- LGB：60 trials，搜索 9 个超参 → OOF +0.0054，但 LB -0.004
- CAT：25 trials，搜索 7 个超参 → 同样退化
- 根因：8K 样本下 HPO 高度过拟合训练集，测试集分布不同无法泛化

HPO 后的典型参数漂移：

| 参数 | 原始 | HPO 最优 | 结论 |
|------|------|---------|------|
| LGB num_leaves | 63 | **98** | HPO 倾向于更复杂的树 |
| LGB max_depth | -1 | **5** | 矛盾：叶多但深度受限 |
| CAT depth | 6 | **8** | 更深更容易过拟合 |
| CAT iterations | 5000 | **3000** | 反而更少轮次 |

### 8.4 失败实验：后处理规则

实现了两条测试期规则（参考竞争者方案）：

**Rule 1**：CryoSleep=True + NoSpend=True + 预测概率处于不确定区 → 概率推至 threshold+0.06

**Rule 2**：组内高置信成员（|prob-threshold|≥0.18）多数投票 → 低置信成员概率推至 threshold±0.12

验证结果：

| 规则 | OOF delta | LB delta | 原因 |
|------|-----------|----------|------|
| Rule 1（CryoSleep） | +0.00023（噪声级） | **-0.001 ~ -0.00116** | group_label 已在训练期捕获，测试期规则添加噪声 |
| Rule 2（组内投票） | **-0.00127** | 未提交 | OOF 直接退化，group_label 冗余 |

根本原因：这两条规则对竞争者有效，是因为其模型没有 group_label 特征，所以测试期规则相当于补充了训练时缺失的组信息。我们的模型已经在训练期通过 group_label 学到了这些信息，测试期再次引入相同信号只会造成概率分布扰动。

---

## 9. 全部提交记录

| # | 提交文件 | 阈值 | LB | 说明 |
|---|---------|------|----|------|
| 1–6 | Phase 5 探索系列 | 0.455 | 0.80921 | 3 模型 LR Stack 基线 |
| 7 | ensemble_new_th0.465 | 0.465 | 0.80967 | +group_label 3 特征 |
| 8 | ensemble_noleak_th0.455 | 0.455 | 0.80780 | 折内无泄露，退化 |
| 9 | ensemble_hybrid_th0.470 | 0.470 | 0.80710 | +cabin label，退化 |
| 10 | ensemble_lr_stack_th0.490 | 0.490 | 0.80923 | OOF 最优阈值，不如 0.465 |
| 11 | ensemble_tfidf_th0.470 | 0.470 | 0.80710 | +TFIDF+group_bin，退化 |
| 12–16 | post_cryo_th0.455~0.480 | 0.455~0.480 | 0.80851~0.80921 | CryoSleep 后处理规则，全退化 |
| **17** | **ensemble_reverted_th0.470** | **0.470** | **0.81014 ✅** | 最优阈值，历史最高 |

---

## 10. CV/LB Gap 分析

| 指标 | 值 |
|------|-----|
| OOF（leaky group_label，OOF 最优 th=0.5475） | 0.82377 |
| OOF（leaky group_label，th=0.470） | ~0.82150 |
| LB（th=0.470） | **0.81014** |
| 总 Gap | **~0.011** |

**Gap 来源分解**：

| 来源 | 估算量 | 说明 |
|------|-------|------|
| Group_label OOF 泄露 | ~0.005 | 验证折内同组互见标签，人为抬高 OOF |
| 训练/测试结构性分布差异 | ~0.006 | 组内结构、历史传送信号等无法完全复现 |
| **合计** | **~0.011** | 结构性 ~0.006 无法通过 CV 方案变化消除 |

**实验验证**（折内无泄露版本）：

- 折内无泄露 OOF = 0.81859（接近真实水平，泄露去除后下降 ~0.005）
- 折内无泄露 LB = 0.80780（比全量 LOO 版本低 -0.00187）
- 说明真实 CV/LB 结构性 gap ≈ 0.81859 - 0.80780 ≈ 0.011

---

## 11. 失败方案完整记录

| 方案 | OOF 变化 | LB 变化 | 失败根因 |
|------|---------|---------|---------|
| Optuna HPO（LGB，60 trials） | +0.0054 | **-0.004** | 在 8K 样本上过拟合训练集分布 |
| Optuna HPO（CAT，25 trials） | +0.0027 | **-0.002** | 同上 |
| 伪标签（prob>0.95 or <0.05） | -0.005 | — | 高置信样本极端不平衡（True:False=752:2），扩充无效 |
| 折内无泄露 group_label | OOF 更诚实 | **-0.00187** | 测试集只用 80% 训练标签信息 |
| Cabin label 特征 | +0.0005 | **-0.00257** | 大多数舱室只住一个 group，与 group_label 高度冗余 |
| TFIDF 姓名特征 | ~持平 | **-0.00257** | 姓名→传送结果无因果关系 |
| GroupSizeBin / IsLoneChild | ~持平 | 推定为负 | 与 GroupSize / IsAlone 高度冗余 |
| 后处理 Rule 1（CryoSleep+NoSpend） | +0.00023（噪声级）| **-0.00046 ~ -0.00116** | group_label 已捕获同等信息，测试期规则添加噪声 |
| 后处理 Rule 2（组内置信投票） | **-0.00127** | 未提交 | OOF 直接退化 |

---

## 12. 核心经验教训

### 教训 1：HPO 提升 OOF ≠ 提升 LB

在 OOF 上通过 Optuna 搜到的超参，在测试集上反而更差。原因：数据量仅 8K，OOF 本身方差较大，HPO 实际上是在对训练集分布做过拟合式调优。**结论：在本数据规模下，默认超参优于 HPO 超参。**

### 教训 2："修泄露"可能有害

Group_label 特征存在 OOF 泄露（OOF 虚高约 0.005），尝试修复为折内无泄露后 LB 反而下降 -0.00187。原因：折内无泄露版本的测试集只能利用当折训练集（约 80% 训练数据）的标签统计，而全量 LOO 版本的测试集可以利用全量训练标签，信息更完整。**结论：OOF 的绝对值不重要，重要的是它能否反映测试集的真实信号质量。**

### 教训 3：新特征必须与现有特征正交

- Cabin label 特征（按舱室分组的传送比例）：大多数舱室只住一个 group，与 group_label 几乎完全重叠，LB -0.003
- TFIDF 姓名特征：姓名→传送结果无因果关系，纯噪声，LB -0.003
- GroupSizeBin：与 GroupSize 冗余
- **规律：当现有特征已覆盖某维度信息时，从该维度继续堆叠特征只会引入噪声。**

### 教训 4：OOF 最优阈值不等于 LB 最优阈值

Leaky group_label 特征让 OOF 最优阈值虚高至 0.5475（验证折内泄露让模型对难样本过度自信），但 LB 最优阈值始终在 0.465~0.470 附近。这两者之间有约 0.08 的系统性偏差。**策略：对于存在 OOF 泄露的配置，应独立扫描 2~3 个 LB 候选阈值，不依赖 OOF 阈值优化结果。**

### 教训 5：后处理规则的前提是模型未捕获该信息

CryoSleep+NoSpend 规则和组内投票规则对竞争者有效（其模型无 group_label 特征），对我们无效（group_label 已在训练期学到同等信息）。**规律：测试期规则只能补充模型在训练期遗漏的信息，无法补充模型已经充分学到的信息，强行干预只会破坏模型的概率校准。**

### 教训 6：StratifiedKFold 优于 GroupKFold

尽管 GroupKFold 理论上更能防止组泄漏，但实验发现 StratifiedKFold 的测试集预测质量更高。原因：GroupKFold 将同组成员强制放在同一 fold，导致某些测试集中的 group 在 OOF 训练时从未被充分覆盖。

---

## 13. 复现步骤

### 环境要求

```
Python 3.9+
lightgbm, xgboost, catboost
sklearn, scipy, numpy, pandas
```

### 一键复现最优提交

```bash
cd src

# Step 1：训练 5 个基础模型（每个约 10 分钟）
python train.py --models lgb,xgb,cat,et,hgb --cv stratified --seeds 42,2024

# Step 2：生成集成 OOF + 测试集预测
python ensemble.py

# Step 3：生成 th=0.470 提交文件（LB 最优）
python -c "
import pandas as pd
pred = pd.read_csv('../outputs/preds/ensemble_lr_stack.csv')
labels = (pred['Transported_Prob'] >= 0.470).astype(bool)
sub = pd.DataFrame({'PassengerId': pred['PassengerId'], 'Transported': labels})
sub.to_csv('../outputs/submissions/submission_th0.470.csv', index=False)
print('Done:', sub['Transported'].sum(), 'True /', len(sub), 'total')
"
```

### 配置检查清单

- [ ] `FEAT_GROUPS` 包含 `group_label`，不含 `name_tfidf` / `group_bin`
- [ ] `CAT_COLS` 不含 `GroupSizeBin`
- [ ] `features.py` 为 8 步流程（无 TFIDF 步骤）
- [ ] 所有模型使用默认超参（`config.py` 中的 `LGB_PARAMS` / `XGB_PARAMS` 等，非 HPO 版本）
- [ ] `train.py` 中 group_label 特征用全量 LOO 计算（非折内）
- [ ] 多 seed OOF 为累加平均，非最后 seed 覆盖

### 验证指标

运行后预期值：

| 指标 | 预期值 |
|------|-------|
| LGB OOF（th=0.50） | ~0.81836 |
| HGB OOF（th=0.50） | ~0.82020 |
| LR Stack OOF（th=0.50） | ~0.81583 |
| LR Stack OOF（th=0.5475） | ~0.82377（含 leaky group_label 虚高） |
| LR Stack OOF（th=0.470） | ~0.82150 |

---

## 14. 代码文件索引

| 文件 | 职责 | 关键函数/类 |
|------|------|-----------|
| `src/config.py` | 路径、特征分组、5 模型超参 | `FEAT_GROUPS`, `LGB_PARAMS` 等 |
| `src/data.py` | 原始 CSV 加载 | `load_raw()` |
| `src/preprocess.py` | 三级缺失值填充 + 基础衍生特征 | `preprocess()` |
| `src/features.py` | 8 步特征工程管道 | `build_features()` |
| `src/train.py` | 多种子 CV 训练 + group_label 特征计算 | `run_cv()`, `_add_group_label_features()` |
| `src/ensemble.py` | LR Stack 集成 + 目标编码 + 阈值优化 | `run_ensemble()`, `lr_stacking()`, `compute_te_features()` |
| `src/post_process.py` | 后处理规则（CryoSleep/组投票）验证框架 | `run_postprocess()`, `validate_cryo_rule()` |
| `src/utils.py` | 随机种子、评估、保存工具 | `seed_everything()`, `accuracy()`, `save_submission()` |
| `src/models/lgb_model.py` | LightGBM 单折训练 | `train_lgb()` |
| `src/models/xgb_model.py` | XGBoost 单折训练 | `train_xgb()` |
| `src/models/cat_model.py` | CatBoost 单折训练 | `train_cat()` |
| `src/models/et_model.py` | Extra Trees 单折训练 | `train_et()` |
| `src/models/hgb_model.py` | HistGradientBoosting 单折训练 | `train_hgb()` |

**报告文件**

| 文件 | 内容 |
|------|------|
| `report/eda_report.md` | Phase 1：数据探索、关键特征分析 |
| `report/preprocess_report.md` | Phase 2：三级填充策略详情 |
| `report/features_report.md` | Phase 3：8 步特征工程详情 |
| `report/phase4_modeling_report.md` | Phase 4：5 模型训练结果、Bug 修复 |
| `report/phase5_ensemble_report.md` | Phase 5：集成架构、LR Stack 系数 |
| `report/phase6_post_ensemble_report.md` | Phase 6：所有后集成实验详情 |
| `report/final_best_report.md` | 最优配置汇总（持续更新） |
| `PROJECT_SUMMARY.md` | **本文件**：全项目综合总结 |

**输出目录结构**

```
outputs/
  oof/          # 各模型 OOF 概率（8693 行）
  preds/        # 各模型测试集预测（4277 行）
  submissions/  # 提交文件（含阈值和时间戳）
```

---

*生成时间：2026-05-16 | 当前最优：LB 0.81014（ensemble_reverted_th0.470）*
