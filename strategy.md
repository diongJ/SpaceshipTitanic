# Spaceship Titanic 竞赛完整实施方案

## 一、问题定位与目标

| 项目 | 内容 |
|------|------|
| 任务类型 | 二分类（Transported: True/False） |
| 评估指标 | Accuracy |
| 训练集规模 | ~8693 条 |
| 测试集规模 | ~4277 条 |
| Top 1% 榜单分数 | ~0.815 |
| Top 5% 榜单分数 | ~0.808 |
| **本方案目标** | 本地 5-Fold CV ≥ 0.812，公开榜 ≥ 0.810 |

---

## 二、项目工程结构

推荐采用模块化的 Python 项目结构，便于实验迭代和复现：

```
SpaceshipTitanic/
├── spaceship-titanic/          # 原始数据
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── data/
│   ├── processed/              # 清洗后的数据 (parquet 格式)
│   └── features/               # 特征工程后的数据
├── notebooks/
│   ├── 01_eda.ipynb            # 探索性分析
│   ├── 02_feature_check.ipynb  # 特征可视化与验证
│   └── 99_error_analysis.ipynb # 错例分析
├── src/
│   ├── config.py               # 全局参数配置
│   ├── data.py                 # 数据加载
│   ├── preprocess.py           # 缺失值与清洗
│   ├── features.py             # 特征工程
│   ├── models/
│   │   ├── lgb_model.py
│   │   ├── xgb_model.py
│   │   ├── cat_model.py
│   │   └── nn_model.py
│   ├── train.py                # 训练主流程（含 CV）
│   ├── ensemble.py             # 集成与 stacking
│   ├── postprocess.py          # 规则后处理
│   └── utils.py                # 工具函数（评估、IO、种子）
├── experiments/                # 每次实验的配置与结果记录
│   └── exp001_baseline_lgb.yaml
├── outputs/
│   ├── oof/                    # OOF 预测概率
│   ├── preds/                  # 测试集预测
│   ├── submissions/            # 提交文件
│   └── models/                 # 训练好的模型权重
└── strategy.md
```

**核心依赖库**（建议 `requirements.txt`）：

```
pandas, numpy, scikit-learn
lightgbm>=4.0, xgboost>=2.0, catboost>=1.2
optuna>=3.5, shap
torch>=2.0, pytorch-tabnet  (可选，用于神经网络模型)
matplotlib, seaborn          (可视化)
pyyaml                       (实验配置管理)
```

---

## 三、阶段 1：EDA 与数据理解

### 3.1 必查项（用 pandas + seaborn 完成）

| 检查项 | 实现方式 | 关注点 |
|--------|---------|-------|
| 缺失率 | `df.isna().mean()` | 每列约 2-3% 缺失，CryoSleep/Cabin 等关键列优先处理 |
| 目标分布 | `value_counts(normalize=True)` | 约 50%/50%，类别相对均衡，无需重采样 |
| 单变量与目标关系 | `groupby('feature')['Transported'].mean()` | 计算每个类别下的传送率 |
| 数值变量分布 | `sns.histplot` + `log1p` 变换观察 | 消费字段右偏严重 |
| 相关性 | `df.corr(method='spearman')` | 用 Spearman 而非 Pearson（消费字段非正态） |
| 组内一致性 | 按 GroupId 聚合检查 | 同组成员的 HomePlanet/Cabin 是否一致 |

### 3.2 关键业务规律验证（必做）

需要用代码验证以下假设，这些规律将直接驱动后续特征和后处理：

1. **CryoSleep=True → 消费全部为 0**：用 `(df['CryoSleep']==True) & (df['TotalSpend']>0)` 检查反例数量
2. **CryoSleep=True 的传送率约 81%**，远高于 False 的约 33%
3. **同 GroupId 的成员传送结果高度相关**：计算组内 Transported 标准差分布
4. **不同 Deck 的传送率差异**：B、C 甲板传送率高，E、F 偏低
5. **儿童（Age<13）的传送率显著高于成年人**

### 3.3 EDA 输出物

- 一份特征-目标关系图谱（每个特征的传送率柱状图）
- 一份缺失模式表（缺失共现关系）
- 一份业务规则验证报告（用于后处理决策）

---

## 四、阶段 2：数据清洗与缺失值处理

实施顺序非常重要，错误顺序会丢失信息。所有处理在 `preprocess.py` 中实现，**train + test 拼接后统一处理**（避免 train/test 处理不一致）。

### 4.1 处理顺序

```
Step 1: 拼接 train+test，标记 is_train
Step 2: 解析 PassengerId → GroupId, PersonNum
Step 3: 解析 Cabin → Deck, CabinNum, Side
Step 4: 解析 Name → FirstName, LastName
Step 5: 规则推断填充（基于业务逻辑）
Step 6: 组内填充（HomePlanet, Cabin 等）
Step 7: 模型填充 / 统计填充（兜底）
Step 8: 类型转换（CryoSleep/VIP → int8）
```

### 4.2 规则推断填充（最高优先级）

| 规则 | 实现 |
|------|------|
| 五项消费均为 0 且 Age > 12 → CryoSleep=True | `df.loc[(df['TotalSpend']==0) & (df['Age']>12) & df['CryoSleep'].isna(), 'CryoSleep'] = True` |
| CryoSleep=True 且消费缺失 → 填充 0 | `df.loc[df['CryoSleep']==True, spend_cols] = df.loc[...].fillna(0)` |
| Age < 13 → VIP=False（儿童不可能 VIP） | 直接规则填充 |
| Age < 13 → 通常无消费 | 用作消费字段填充辅助信号 |

### 4.3 组内填充

利用同组乘客的高度一致性：

| 字段 | 填充策略 |
|------|---------|
| HomePlanet | 同 GroupId 众数（pandas `groupby + transform`） |
| Destination | 同 GroupId 众数 |
| Cabin (Deck/Side) | 同 GroupId 众数（同组通常住相邻舱室） |
| Surname | 同 GroupId 众数 |

实现技巧：用 `df.groupby('GroupId')['HomePlanet'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan))`。

### 4.4 模型填充（针对剩余的 CryoSleep / HomePlanet）

- 用一个简单 LightGBM 分类器，以非缺失行为训练集，预测剩余缺失值
- 仅对前面规则无法覆盖的高价值字段使用

### 4.5 统计填充（兜底）

- 数值字段：按 `(HomePlanet, CryoSleep)` 分层中位数
- 类别字段：全局众数
- 消费字段：缺失即填 0（对未冷冻的乘客）或基于 `(HomePlanet, Age 桶)` 中位数

---

## 五、阶段 3：特征工程

特征工程是本赛事性能上限的决定因素。建议在 `features.py` 中按类别组织，每类特征可独立开关，便于消融实验。

### 5.1 ID 解析特征

| 特征 | 说明 |
|------|------|
| `GroupId` | PassengerId 前 4 位整数 |
| `PersonNum` | PassengerId 后 2 位 |
| `GroupSize` | 同 GroupId 成员数 |
| `IsAlone` | GroupSize == 1 |
| `IsLargeGroup` | GroupSize >= 4 |
| `PersonNumInGroup` | 在组内排序位置（可能反映长幼） |

### 5.2 Cabin 特征

| 特征 | 说明 |
|------|------|
| `Deck` | A-G, T |
| `CabinNum` | 0-1900 数值 |
| `CabinNumBin` | 分桶（0-300, 300-600, 600-900...） |
| `Side` | P / S |
| `DeckSide` | Deck + Side 组合（A_P, A_S, ...） |
| `CabinGroupSize` | 同 Cabin 完全相同的人数 |

### 5.3 消费特征（核心）

| 特征 | 说明 |
|------|------|
| `TotalSpend` | 五项之和 |
| `LogTotalSpend` | `log1p(TotalSpend)` |
| `HasSpend` | TotalSpend > 0（布尔） |
| `NumSpendCategories` | 五项中非零项的个数 |
| `MaxSpend` | 五项中最大值 |
| `SpendStd` | 五项标准差（消费均匀度） |
| `LuxurySpend` | Spa + VRDeck + RoomService（奢侈消费） |
| `EssentialSpend` | FoodCourt + ShoppingMall（基础消费） |
| `LuxuryRatio` | LuxurySpend / (TotalSpend + 1) |
| 各项 log1p | 五项各自做对数变换 |

### 5.4 年龄特征

| 特征 | 说明 |
|------|------|
| `AgeGroup` | 0-12 / 13-17 / 18-25 / 26-40 / 41-60 / 60+ |
| `IsChild` | Age < 13 |
| `IsTeen` | 13 ≤ Age < 18 |
| `IsSenior` | Age >= 60 |

### 5.5 姓名特征

| 特征 | 说明 |
|------|------|
| `LastName` | 姓 |
| `FamilySize` | 同姓氏的人数 |
| `IsSurnameInGroup` | 组内是否有同姓氏成员 |

### 5.6 组级聚合特征（高收益）

按 GroupId 聚合，回填到每个成员：

| 特征 | 说明 |
|------|------|
| `Group_TotalSpend_mean` | 组内人均消费 |
| `Group_TotalSpend_sum` | 组消费总额 |
| `Group_CryoRatio` | 组内冷冻比例 |
| `Group_Age_mean` | 组内平均年龄 |
| `Group_Age_min/max` | 组内最小/最大年龄 |
| `Group_HomePlanet_nunique` | 组内 HomePlanet 多样性 |
| `Group_VIP_any` | 组内是否有 VIP |

### 5.7 Cabin 级聚合特征

按 Cabin 聚合（同舱室乘客）：

| 特征 | 说明 |
|------|------|
| `Cabin_Size` | 同 Cabin 人数 |
| `Cabin_TotalSpend_sum` | 同 Cabin 总消费 |
| `Cabin_CryoRatio` | 同 Cabin 冷冻比例 |

### 5.8 交互特征

| 特征 | 说明 |
|------|------|
| `Cryo_x_Spend` | CryoSleep 标志 × TotalSpend（用于异常检测） |
| `HomePlanet_Destination` | 路线特征 |
| `Deck_HomePlanet` | 不同星球乘客分布在不同甲板 |
| `Age_x_VIP` | VIP 中年化分布 |
| `IsAlone_x_TotalSpend` | 独行 + 高消费 |

### 5.9 目标编码（Target Encoding）

对高基数类别特征（GroupId 不适合，但 LastName、Cabin 适合）使用 K-Fold 内目标编码，**必须在 CV 内部计算**避免泄漏：

- `LastName_TE`：同姓氏其他成员的平均传送率（leave-one-out）
- `Cabin_TE`：同 Cabin 其他成员的平均传送率
- 实现：`category_encoders.TargetEncoder` + KFold

### 5.10 特征筛选

- 训练完一轮 LightGBM 后，用 `feature_importance` 和 SHAP 删除最低 10% 重要性的特征
- 用对抗验证识别训练/测试分布差异最大的特征，对其加入 noise 或删除

---

## 六、阶段 4：建模

### 6.1 验证策略（统一规范）

- **主方案**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **替代方案**：`GroupKFold(n_splits=5, groups=GroupId)` —— 防止同组成员同时出现在 train/val
- **最终选择**：两种方案都跑，对比 CV 分数稳定性，**用 GroupKFold 作为最终评估**（更接近真实测试场景）
- 所有模型保存 OOF 预测（`outputs/oof/`），便于后续 stacking

### 6.2 LightGBM（主力模型）

| 项目 | 设置 |
|------|------|
| objective | `binary` |
| metric | `binary_error`（accuracy 互补） |
| 关键超参起点 | `learning_rate=0.02, num_leaves=63, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, min_child_samples=20, lambda_l2=1.0` |
| early_stopping | `100 rounds` |
| num_boost_round | 5000（结合 early stopping） |
| 类别特征 | 直接用 `categorical_feature` 参数，无需 one-hot |

### 6.3 XGBoost

| 项目 | 设置 |
|------|------|
| objective | `binary:logistic` |
| eval_metric | `error` |
| 关键超参 | `learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0` |
| early_stopping | `100 rounds` |
| tree_method | `hist`（速度优势） |

### 6.4 CatBoost

| 项目 | 设置 |
|------|------|
| loss_function | `Logloss` |
| eval_metric | `Accuracy` |
| 关键超参 | `iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=3, random_strength=1, bagging_temperature=1` |
| 类别特征 | 通过 `cat_features` 参数原生支持 |
| early_stopping | `od_type='Iter', od_wait=100` |

### 6.5 神经网络模型（差异化）

二选一即可：

**方案 A：MLP + Embedding（推荐）**

- 类别特征做 embedding（dim = min(50, n_unique//2)）
- 数值特征 BatchNorm + Linear
- 三层 MLP（256 → 128 → 64），Dropout=0.3
- 损失：BCEWithLogitsLoss
- 优化器：AdamW(lr=1e-3, weight_decay=1e-4)
- Scheduler：CosineAnnealingLR
- Early stopping on val accuracy

**方案 B：TabNet**

- 直接用 `pytorch-tabnet` 库
- 关键参数：`n_d=32, n_a=32, n_steps=5, gamma=1.5`
- 训练较慢但与树模型差异化明显

### 6.6 超参搜索（Optuna）

- 对 LightGBM 用 Optuna 搜索 100-200 trials
- 搜索空间：`num_leaves [15, 127], min_child_samples [5, 100], lambda_l1 [1e-8, 10], lambda_l2 [1e-8, 10], feature_fraction [0.5, 1.0]`
- 在 5-Fold CV 上优化，objective 为 OOF accuracy
- TPE sampler + MedianPruner

---

## 七、阶段 5：集成

### 7.1 集成层级

```
Level 0（基模型）              Level 1（元模型）       Level 2（最终）
┌─────────────┐
│ LightGBM ×5 │ ──┐
├─────────────┤   │
│ XGBoost  ×5 │ ──┼─→ OOF 矩阵 → Logistic Regression ─→ 加权融合 ─→ 提交
├─────────────┤   │           或 简单加权平均
│ CatBoost ×5 │ ──┤
├─────────────┤   │
│ NN       ×5 │ ──┘
└─────────────┘
```

### 7.2 Blending（先做）

- 对每个模型的 OOF 预测，搜索权重使加权 OOF 准确率最大
- 用 `scipy.optimize.minimize` 或网格搜索（步长 0.05，权重和为 1）
- 通常树模型权重 0.7-0.8，神经网络 0.2-0.3

### 7.3 Stacking（后做）

- 二级模型用 LogisticRegression（C=1.0）或浅层 LightGBM
- 输入：基模型 OOF 概率 + 少量原始强特征（CryoSleep, TotalSpend 等）
- 仍用同样的 5-Fold

### 7.4 种子平均

- LightGBM 用 5 个不同随机种子（42, 123, 2024, 7, 999）训练
- 5 次预测概率取均值，可降低方差约 0.1-0.2%

---

## 八、阶段 6：后处理

### 8.1 阈值优化

- 在 OOF 上搜索 0.40-0.60 范围内的最优阈值（步长 0.005）
- 通常最优阈值略偏离 0.5，对 Accuracy 提升约 0.05-0.15%

### 8.2 业务规则修正（谨慎使用，必须验证 OOF 提升）

| 规则 | 说明 |
|------|------|
| CryoSleep=True 且预测概率 > 0.4 → 强制为 True | 冷冻乘客传送率高，提升召回 |
| 组内一致性：组内预测均值 > 0.7 且某成员 < 0.5 → 修正为 True | 利用组内相关性 |
| Age < 13 且组内成年人均预测 True → 修正为 True | 儿童随家庭传送 |

实施前必须在 OOF 上验证：每条规则单独应用，看 OOF accuracy 是否提升 ≥ 0.001，否则不采用。

### 8.3 伪标签（最后冲刺）

- 取测试集预测概率 > 0.95 或 < 0.05 的样本（高置信度）
- 加入训练集重新训练 LightGBM
- 注意：可能引入分布偏差，需在 OOF 上严格验证
- 通常提升 0.1-0.2%

---

## 九、实验管理与复现

### 9.1 实验记录表（强制维护）

每次实验在 `experiments/expXXX.yaml` 记录配置 + 结果，主表 `experiments/log.csv`：

| exp_id | desc | features | model | cv_score | lb_score | notes |
|--------|------|----------|-------|----------|----------|-------|
| exp001 | baseline LGB | basic | lgb | 0.8021 | 0.7987 | 起点 |
| exp002 | + group features | + group_agg | lgb | 0.8064 | 0.8035 | 组特征显著 |
| exp003 | + target encoding | + te | lgb | 0.8089 | - | 仍单模 |
| exp010 | LGB+XGB+CAT blend | full | ensemble | 0.8121 | 0.8092 | ✓ 提交 |
| ... | ... | ... | ... | ... | ... | ... |

### 9.2 复现要求

- 全局固定随机种子：`numpy`, `random`, `torch`, 模型本身
- 数据处理代码确定性：避免 `set` 等无序结构影响
- 每个实验保存：模型、OOF、test 预测、配置 yaml

### 9.3 提交策略

- Kaggle 每天 5 次提交机会，**不要每次实验都提交**
- 优先提交：(1) 建立基线 (2) 单模型最佳 (3) 集成最佳 (4) 含后处理最终版
- 最终选 2 个：一个 CV 最高的、一个集成最稳的（防私榜翻车）

---

## 十、阶段化里程碑与时间预算

| 阶段 | 内容 | 预期 CV | 时间 |
|------|------|---------|------|
| **W1 D1-2** | EDA + 数据清洗 | - | 2 天 |
| **W1 D3-4** | 基础特征工程 + LGB 基线 | 0.802 | 2 天 |
| **W1 D5-7** | 完整特征 + LGB 调优 | 0.808 | 3 天 |
| **W2 D1-2** | XGB + CAT + 三模型 blend | 0.811 | 2 天 |
| **W2 D3-4** | 神经网络 + Stacking | 0.813 | 2 天 |
| **W2 D5-6** | 后处理 + 伪标签 + 种子平均 | 0.815 | 2 天 |
| **W2 D7** | 最终提交 + 选定方案 | - | 1 天 |

总计：约 14 天，每天 3-5 小时投入。

---

## 十一、风险与应对

| 风险 | 现象 | 应对 |
|------|------|------|
| CV 与公开榜分数差距大（>0.005） | 私榜翻车风险 | 改用 GroupKFold，检查目标编码泄漏 |
| 模型严重过拟合 | 训练集 acc 远高于 CV | 增大正则、减少 num_leaves、特征选择 |
| 特征过多导致训练慢 | LGB 单 fold > 5 分钟 | 用 SHAP 删除低贡献特征 |
| 集成无提升 | blend 后 CV 不升 | 检查模型多样性（Pearson 相关性 < 0.95） |
| 后处理规则在私榜失效 | OOF 提升但榜单下降 | 仅采用 OOF 提升 ≥ 0.002 的规则 |

---

## 十二、最低可用基线（MVP，半天可完成）

如果时间紧迫，最快出一个不差的基线：

1. 拼接 train+test，解析 PassengerId、Cabin
2. 简单缺失填充：数值用中位数，类别用众数
3. 基础特征：GroupSize, TotalSpend, Deck, Side, AgeGroup
4. LightGBM 默认参数 + 5-Fold CV
5. OOF 概率 > 0.5 → True，输出提交

预期分数：**0.795-0.800**（Top 30%）。在此基础上按本方案推进，逐步提升。

---

## 十三、关键代码文件职责（不写具体代码，仅说明）

| 文件 | 职责 |
|------|------|
| `src/config.py` | 路径常量、特征列表、模型超参字典、随机种子 |
| `src/data.py` | `load_raw()` → 返回 train, test；`save_processed()` |
| `src/preprocess.py` | `parse_ids()`, `fill_missing()` 等独立函数，可单测 |
| `src/features.py` | 每类特征一个函数（`add_group_features`, `add_spend_features`...），主入口 `build_features(df)` 拼装 |
| `src/models/*.py` | 每个模型实现 `train_fold(X_tr, y_tr, X_val, y_val) → model, val_pred` 接口 |
| `src/train.py` | 主流程：load → preprocess → features → CV 训练 → 保存 OOF + test 预测 |
| `src/ensemble.py` | 加载所有 OOF + test 预测，做 blend / stacking，输出最终概率 |
| `src/postprocess.py` | 阈值搜索、业务规则修正、生成提交文件 |
| `src/utils.py` | `seed_everything()`, `score()`, `save_submission()` |

每个模块设计为可独立运行 + 可单元测试，便于调试。

---

## 十四、最终检查清单（提交前）

- [ ] 提交文件格式正确：`PassengerId, Transported`，类型为 bool（True/False，非 0/1）
- [ ] 提交行数与 sample_submission 一致（4277 行）
- [ ] 无缺失值、无重复 PassengerId
- [ ] 本地 CV 分数已稳定记录，未过拟合公开榜
- [ ] 至少保留 2 个备选提交（CV 最高 + 集成最稳）
- [ ] 实验日志完整，可复现最终结果
