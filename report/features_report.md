# 阶段三：特征工程 报告

## 1. 处理架构

```
train_clean (8693×30) + test_clean (4277×30)
         │  ← 拼接 (12970 条统一变换)
         ▼
   ┌────┴────────────────────────────────────────────┐
   │ Step 1  ID 特征          IsLargeGroup           │
   │ Step 2  Cabin 特征        CabinNumBin, DeckSide  │
   │ Step 3  Spend 特征        对数、消费结构、奢侈品比  │
   │ Step 4  Age 特征          IsTeen                │
   │ Step 5  Name 特征         FamilySize, 组内同姓   │
   │ Step 6  组聚合特征       GroupId 层级统计       │
   │ Step 7  舱聚合特征       Cabin 层级统计         │
   │ Step 8  交互特征         组合、交叉乘积         │
   └────┬────────────────────────────────────────────┘
         ▼
train_feat (8693×55) + test_feat (4277×55)
```

所有特征均可通过 `config.py` 的 `FEAT_GROUPS` 字典按组开关，支持 config-only 消融实验。

---

## 2. 各步骤详情

### Step 1 — ID 特征

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `IsLargeGroup` | `GroupSize >= 4` | EDA 发现 4 人组传送率最高（64%），超大组（8人）急剧下降（39%），大组标识帮助模型捕捉阈值效应 |

> `GroupId`, `PersonNum`, `GroupSize`, `IsAlone` 已由 preprocess 阶段产出。

### Step 2 — Cabin 特征

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `CabinNumBin` | 舱号 6 档分桶（0-300 / 300-600 / ... / 1500+） | 舱号连续值直接输入模型效果差，分桶后形成有序类别，捕捉舱号区间的非线性 |
| `DeckSide` | `Deck + '_' + Side` 拼接 | EDA 发现 P/S 舷传送率差约 10pp，且 Deck×Side 存在交互（如 B 甲板 P 舷 vs S 舷可能不同），组合后作为一个特征输入 |

> `Deck`, `CabinNum`, `Side` 已由 preprocess 解析产出。

### Step 3 — 消费特征

消费字段分布高度右偏（中位数为 0），原始值直接使用会压制低消费区间的区分度。

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `Log{SpendCol}` × 5 | `np.log1p(col)` | 对数变换缓解右偏，使模型在 0~100 消费区间也能学习区分 |
| `LogTotalSpend` | `log1p(TotalSpend)` | 同上，总消费的对数版本 |
| `MaxSpend` | 五项消费取 max | 单一最大消费项，反映消费偏好强度 |
| `SpendStd` | 五项消费取 std（NaN→0） | 消费分散度：std 大 → 在单项大量消费；std 小 → 均匀消费 |
| `LuxurySpend` | `Spa + VRDeck + RoomService` | 三项奢侈型消费合计 |
| `EssentialSpend` | `FoodCourt + ShoppingMall` | 两项刚需型消费合计 |
| `LuxuryRatio` | `LuxurySpend / (TotalSpend + 1)` | 奢侈消费占比（分母 +1 防除零），高奢侈占比乘客传送率显著更低 |

> `TotalSpend`, `HasSpend`, `NumSpendCategories` 已由 preprocess 产出。

### Step 4 — 年龄特征

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `IsTeen` | `13 <= Age < 18` | EDA 中青少年传送率（55.4%）介于儿童（70%）和成人（~47%）之间，单独标记避免模型将 13-17 岁与儿童或成人混淆 |

> `AgeGroup`, `IsChild`, `IsSenior` 已由 preprocess 产出。

### Step 5 — 姓名特征

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `FamilySize` | `= SurnameGroupSize` | 同姓人数，近似于"家族规模"，比 GroupSize 更能标识血缘关系而非同行关系 |
| `IsSurnameInGroup` | 组内仅一种姓氏 + GroupSize>1 + 姓氏非 Unknown | 组内成员同姓意味着结为家庭出行，组内传送结果一致性极高（EDA 43.6% 一致），该特征帮助模型识别这种强一致性 |

### Step 6 — 组级聚合特征

基于 GroupId 对全体样本做聚合统计，再将统计量映射回每个乘客。train+test 拼接后统一计算，确保聚合统计覆盖完整。

| 新特征 | 聚合方式 | 设计动机 |
|--------|---------|---------|
| `Group_TotalSpend_mean` | 组内平均消费 | 组团整体消费水平——高消费组整体传送率低 |
| `Group_TotalSpend_sum` | 组内总消费 | 同上，数值尺度不同，提供互补信号 |
| `Group_Age_mean` | 组内平均年龄 | 家庭组（中年+儿童）vs 朋友组（同龄人）区分 |
| `Group_Age_min` | 组内最小年龄 | 识别含儿童的家庭组 |
| `Group_Age_max` | 组内最大年龄 | 识别含老人的家庭组 |
| `Group_HomePlanet_nunique` | 组内出发星球去重数 | >1 意味着组内有来自不同星球的成员（队友而非家人） |
| `Group_CryoRatio` | 组内 CryoSleep 比例 | 组内冷冻比例——全冷冻的家庭组传送率最高 |
| `Group_VIP_any` | 组内是否有人 VIP | 一个组内有 VIP 乘客，可能拉低整组传送率 |

### Step 7 — 舱级聚合特征

基于精确舱室 key（Deck + CabinNum + Side）做聚合。CabinNum 缺失的乘客不参与舱室聚合。

| 新特征 | 聚合方式 | 设计动机 |
|--------|---------|---------|
| `Cabin_Size` | 同舱人数 | 同舱乘客数量——多人舱 vs 单人间可能有不同的传送模式 |
| `Cabin_TotalSpend_sum` | 同舱总消费 | 同舱乘客的整体消费水平 |
| `Cabin_CryoRatio` | 同舱冷冻比例 | 同舱乘客大多冷冻 → 该舱可能在冷冻区域 |

### Step 8 — 交互特征

| 新特征 | 计算方式 | 设计动机 |
|--------|---------|---------|
| `Cryo_x_TotalSpend` | `CryoSleep(bool) * TotalSpend` | CryoSleep=True 的乘客消费均为 0，不做交互时模型难以学到"冷冻则消费无信号"的条件关系；此特征使该模式显式化 |
| `Route` | `HomePlanet + '_' + Destination` | EDA 发现路线传送率差异大（Europa→TRAPPIST vs Earth→55 Cancri），路线组合形成高信息量的类别特征 |
| `Deck_HomePlanet` | `Deck + '_' + HomePlanet` | EDA 发现 Deck×HomePlanet 交互（Europa B 甲板 vs Earth E 甲板差异极端），组合后显式化 |
| `Age_x_VIP` | `Age * VIP(bool)` | VIP 儿童、VIP 老人可能具有特殊传送模式 |
| `IsAlone_x_TotalSpend` | `IsAlone * TotalSpend` | 独行乘客的消费信号可能不同于团体乘客 |

---

## 3. 特征分组总览

对应 `config.py` 的 `FEAT_GROUPS`，共计 **55 个特征列**（不含 PassengerId 等非模型列）：

| 分组 | 特征数 | 来源 | 消融实验参看 |
|------|--------|------|-------------|
| base | 10 | preprocess 原始字段 | — |
| id | 4 | preprocess + features Step 1 | A1: 移除 id |
| cabin | 5 | preprocess + features Step 2 | A2: 移除 cabin |
| spend | 14 | preprocess + features Step 3 | A3: 移除 spend 衍生 |
| age | 4 | preprocess + features Step 4 | A4: 移除 age 衍生 |
| name | 2 | features Step 5 | A5: 移除 name |
| group_agg | 8 | features Step 6 | A6: 移除 group_agg |
| cabin_agg | 3 | features Step 7 | A7: 移除 cabin_agg |
| interact | 5 | features Step 8 | B1: 移除 interact |

---

## 4. 最终产出

| 数据集 | 行数 | 列数（近似） | 缺失 |
|--------|------|------------|------|
| train_feat | 8693 | ~55 | 0 |
| test_feat | 4277 | ~55 | 0 |

列数取决于 `FEAT_GROUPS` 的当前配置（所有组均开启约 55 列），所有列均无缺失。

---

## 5. 与竞争对手特征对比

参照 `references/competitor_analysis.md` 中 0.82137 方案的特征集（43 个特征）：

| 维度 | 本方案 | 对手 | 差异 |
|------|--------|------|------|
| 组级聚合 | **8 个** | 0 个 | 本方案显著领先 |
| 舱级聚合 | **3 个** | 0 个 | 本方案独有 |
| 消费衍生 | 14 个 | 8 个 | 本方案多消费结构分解 |
| GroupSize 级别 | IsLargeGroup | 分桶更细（5 级） | 对手略优 |
| SpendPerGroupMember | **无** | 有 | 后续可补充 |
| AvgSpendPerService | **无** | 有 | 后续可补充 |

---

## 6. 对应代码

| 文件 | 职责 |
|------|------|
| `src/features.py` | 全部特征工程逻辑（`build_features()` 主入口，8 个子函数） |
| `src/config.py` | `SPEND_COLS`, `FEAT_GROUPS`, `CAT_COLS` 定义 |

运行方式：
```bash
cd SpaceshipTitanic
python -c "
from src.data import load_raw
from src.preprocess import preprocess
from src.features import build_features
train, test = load_raw()
train_c, test_c = preprocess(train, test)
train_f, test_f = build_features(train_c, test_c)
"
```
