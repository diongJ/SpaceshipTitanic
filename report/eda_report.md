# Spaceship Titanic — EDA 报告

## 1. 数据集概览

| 项目 | 数值 |
|------|------|
| 训练集行数 | 8,693 |
| 测试集行数 | 4,277 |
| 特征数量 | 13（含 PassengerId、Name） |
| 目标变量 | Transported（bool） |

### 1.1 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| PassengerId | string | 格式 `gggg_pp`，前四位为群组编号，后两位为组内序号 |
| HomePlanet | categorical | 出发星球（Earth / Europa / Mars） |
| CryoSleep | bool | 是否冷冻休眠 |
| Cabin | string | 格式 `Deck/Num/Side`，含甲板、舱号、左右舷 |
| Destination | categorical | 目的地（TRAPPIST-1e / 55 Cancri e / PSO J318.5-22） |
| Age | float | 年龄 |
| VIP | bool | 是否 VIP |
| RoomService / FoodCourt / ShoppingMall / Spa / VRDeck | float | 五项消费金额 |
| Name | string | 姓名（含姓氏，可辅助识别家庭关系） |
| Transported | bool | **目标变量**，是否被传送到异次元 |

### 1.2 目标变量分布

目标变量高度均衡：**True 50.4% / False 49.6%**，无需处理类别不平衡，可直接使用 Accuracy 作为评估指标，也无需对损失函数做加权。

### 1.3 缺失率

所有字段缺失率约 **2.0%–2.5%**，缺失模式均匀，未发现系统性缺失关联（如某类乘客集中缺失）。

| 字段 | 缺失率 |
|------|-------|
| CryoSleep | 2.50% |
| ShoppingMall | 2.39% |
| VIP | 2.34% |
| HomePlanet | 2.31% |
| Name | 2.30% |
| Cabin | 2.29% |
| VRDeck | 2.16% |
| Spa | 2.11% |
| FoodCourt | 2.11% |
| Destination | 2.09% |
| RoomService | 2.08% |
| Age | 2.06% |

---

## 2. 关键特征分析

### 2.1 CryoSleep（最强特征）

| CryoSleep | 传送率 |
|-----------|-------|
| True | **81.8%** |
| False | 32.9% |
| 缺失 | 48.9% |

CryoSleep 是本数据集中**信息量最强的单一特征**，True/False 之间传送率相差约 49 个百分点。

**关键业务约束（已验证）**：CryoSleep=True 的所有乘客（3037 人）五项消费**严格为 0**，无一例外。这意味着：

- 若某乘客 CryoSleep=True 且消费字段缺失 → 可直接填充 0
- 若某乘客五项消费均为 0 且 CryoSleep 缺失 → 可推断 CryoSleep=True

训练集中可规则推断的样本量：**98 条**；测试集：**38 条**。另有 CryoSleep=True 的乘客消费字段缺失共 **361 处**可直接填 0，这是无泄漏的高质量缺失值处理。

### 2.2 消费字段

五项消费字段（RoomService、FoodCourt、ShoppingMall、Spa、VRDeck）与目标的 Spearman 相关系数均为**负值**，消费越高传送率越低。

| 字段 | Spearman rho |
|------|-------------|
| TotalSpend（总消费） | -0.419 |
| Spa | -0.373 |
| RoomService | -0.371 |
| VRDeck | -0.352 |
| ShoppingMall | -0.229 |
| FoodCourt | -0.187 |
| Age | -0.071 |

**消费分布特征**：

- 五项消费字段的中位数均为 0（超过 61% 的非缺失值为 0），分布呈高度右偏
- 消费为 0 的乘客传送率高达 **78.4%**；消费一旦非零，传送率骤降至 22%–33%
- 建模时须对消费字段做 `log1p` 变换以缓解右偏
- 五项消费之间存在**负相关**（如 RoomService 与 FoodCourt rho = -0.32），说明乘客消费具有偏好选择性

### 2.3 Cabin（位置特征）

Cabin 字段格式为 `Deck/Num/Side`，需解析拆分。

**甲板（Deck）传送率**（从高到低）：

| Deck | 传送率 |
|------|-------|
| B | 73.4% |
| C | 68.0% |
| G | 51.6% |
| A | 49.6% |
| F | 44.0% |
| D | 43.3% |
| E | 35.7% |
| T | 20.0%（仅 15 人，样本极少） |

B、C 甲板（通常为 Europa 高价舱）传送率远高于 E、F 甲板（通常为 Earth 经济舱）。

**侧舷（Side）**：S 舷传送率 55.5%，P 舷 45.1%，差异约 10 个百分点，是有效的位置特征。

**交叉分析**：CryoSleep=True 时，B/C/D/F 甲板传送率接近 **99%**；E/G 甲板即使冷冻，传送率仍仅约 65%，说明位置与冷冻状态存在显著交互效应。

### 2.4 HomePlanet

| HomePlanet | 传送率 |
|-----------|-------|
| Europa | **65.9%** |
| Mars | 52.3% |
| Earth | 42.4% |

Europa 乘客传送率最高，且 CryoSleep=True 的 Europa 乘客传送率接近 **99%**（全传送），而同为冷冻状态的 Earth 乘客传送率仅约 66%，说明 HomePlanet × CryoSleep 的交互特征信息量极高。

### 2.5 Destination

| Destination | 传送率 |
|------------|-------|
| 55 Cancri e | **61.0%** |
| PSO J318.5-22 | 50.4% |
| TRAPPIST-1e | 47.1% |

TRAPPIST-1e 是最主要的目的地（占约 70%），传送率最低；55 Cancri e 仅占 20% 但传送率最高。`HomePlanet × Destination` 的路线组合特征值得构造。

### 2.6 Age

| 年龄段 | 传送率 |
|-------|-------|
| 儿童（<13） | **70.0%** |
| 青少年（13-17） | 55.4% |
| 青年（18-25） | 45.8% |
| 中年（26-40） | 48.0% |
| 壮年（41-60） | 49.5% |
| 老年（60+） | 47.3% |

儿童传送率异常偏高（70%），与成年人差距约 20+ 个百分点，需单独构造 `IsChild` 布尔特征。此外儿童通常无消费，须在缺失填充时单独处理（不能用"消费全零→CryoSleep=True"规则）。

### 2.7 VIP

VIP 乘客传送率（38.2%）**低于**非 VIP（50.6%）。原因可能是 VIP 乘客消费更高，而高消费与低传送率相关。VIP 本身是一个弱负相关特征，与消费字段共线性较强。

### 2.8 GroupSize（组内结构）

| 组大小 | 传送率 |
|-------|-------|
| 1（独行） | 45.2% |
| 2 | 53.8% |
| 3 | 59.3% |
| 4 | **64.1%** |
| 5 | 59.3% |
| 6 | 61.5% |
| 7 | 54.1% |
| 8 | 39.4% |

- 约 55.3% 的乘客独行（4805 个单人组），独行乘客传送率最低
- 群体乘客（组大小 2-7）传送率整体高于独行，峰值在 4 人组（64%）
- 超大组（8 人）传送率骤降（39%），可能是特殊团体（如服务人员）
- **组内传送结果一致性**：43.6% 的多人组成员传送结果完全一致，说明组内相关性强，可利用组内聚合特征

---

## 3. Train / Test 分布对比

| 特征 | Train | Test | 结论 |
|------|-------|------|------|
| HomePlanet: Earth | 54.2% | 54.0% | ✓ 一致 |
| HomePlanet: Europa | 25.1% | 23.9% | ✓ 基本一致 |
| CryoSleep: True | 35.8% | 36.9% | ✓ 一致 |
| Destination: TRAPPIST-1e | 69.5% | 70.6% | ✓ 一致 |
| Deck: F | 32.9% | 34.6% | ✓ 基本一致 |
| Deck: G | 30.1% | 29.3% | ✓ 一致 |

**结论**：训练集和测试集分布高度一致，无明显协变量偏移，**无需对抗验证降权**，特征工程和模型可直接全量训练。

---

## 4. 缺失值处理策略

基于 EDA 发现，推荐以下处理顺序（优先级从高到低）：

### 4.1 规则推断（零泄漏，最高优先）

| 规则 | 条件 | 填充值 | 覆盖量（train+test） |
|------|------|-------|---------------------|
| 消费全零 → CryoSleep | 五项消费均为 0，Age>12，CryoSleep 缺失 | True | 136 条 |
| CryoSleep → 消费 | CryoSleep=True，消费字段缺失 | 0 | ~361 处 |
| 儿童 → VIP | Age<13，VIP 缺失 | False | 少量 |

### 4.2 组内填充

同 GroupId 的乘客通常来自同一家庭或团体，以下字段可用组内众数填充：HomePlanet、Destination、Cabin（Deck/Side）

实现：`groupby('GroupId').transform(lambda x: x.fillna(x.mode().iloc[0]))`

### 4.3 统计填充（兜底）

| 字段类型 | 填充策略 |
|---------|---------|
| 消费字段（非冷冻乘客） | 按 (HomePlanet, AgeGroup) 分层中位数 |
| Age | 按 (HomePlanet, Deck) 分层中位数 |
| HomePlanet / Destination | 全局众数 |
| CryoSleep / VIP | 全局众数（False） |

---

## 5. 主要结论与后续特征工程方向

基于 EDA，提炼以下核心结论，直接驱动特征工程设计：

1. **CryoSleep + 消费字段是模型的信息核心**：两者高度关联且可互推，围绕消费构造的聚合特征（总消费、消费类别数、消费结构占比）将是模型最重要的输入
2. **位置信息（Deck × Side × HomePlanet）有显著区分力**：需解析 Cabin 并构造 Deck×Side 组合特征
3. **乘客群组结构不可忽视**：GroupSize、组内平均消费、组内冷冻比例等组级聚合特征可捕捉家庭/团体效应
4. **年龄非线性**：儿童（<13 岁）需单独标记，不应仅作为数值特征输入
5. **交互特征高收益**：HomePlanet × CryoSleep、Deck × CryoSleep 均有接近 99% 的极端传送率，交互特征值得优先构造
6. **消费字段须做对数变换**：分布高度右偏，直接使用数值会压缩低消费乘客的区分度

---

## 6. 可视化总览

完整可视化图表见 `report/eda_overview.png`，包含：目标分布饼图、CryoSleep/HomePlanet/Deck/Destination/AgeGroup 各维度传送率堆积条形图、Log(TotalSpend) 分布直方图、GroupSize 传送率折线、VIP & Side 对比、以及所有数值特征的 Spearman 相关系数图。
