# AEF_SwinIR 给 Gemini 的快速投喂版

**当前状态**：V3 后处理已完成全量验证，当前最优 **14.8053 PSNR**。  
**核心任务**：围绕 **AlphaEarth 64 维无序高维特征**，调研最适合的 **分组注意力 / 置换不变建模 / 低频通道补偿 / 轻量后处理 / 光谱感知损失** 方案，并给出可落地排序。

---

## 1. 项目一句话

Landsat 8 低分辨率多时相输入，经 SwinIR 主干、云感知和时间融合后，重建到 **64 维 AlphaEarth-like 高维表示**。难点不是普通超分，而是：
- 64 维特征是 **无序集合**，不是传统按波长排序的光谱序列。
- 低频 / 非 RGB 通道明显更弱，MSE 会偏袒高方差通道。
- 复杂后处理已经有退化历史，当前更看重 **稳定、轻量、可解释** 的方案。

---

## 2. 已知系统约束

- 输入：9 波段 Landsat 8，30m，T=20，多时相。
- 附加输入：DOY 时间编码、云掩膜 indicating_mask、QA 信息。
- 输出：3× 上采样后的 64 通道特征图。
- 现有结论：
  - **V2** 中简单矩阵重排比 Gumbel 路由稳。
  - **V3** 中 `stage + gated` 是当前最有效后处理组合。
  - 历史 3b 高分不稳定，不作为主结论。

---

## 3. Gemini 需要重点查什么

### 3.1 无序高维特征的分组注意力
查这些方向：
- Group Attention
- Dynamic Grouping Attention
- Routing Attention
- Sparse Channel Attention
- Mixture-of-Experts for channels

重点回答：
- 哪些方法真的适合 **无序 64 维向量**。
- 哪些方法能把通道拆成可解释子组。
- 哪些方法适合小数据、输出端后处理，而不是重写主干。

### 3.2 置换不变性建模
查这些方向：
- Set Transformer
- Vision Permutators
- permutation-invariant attention
- set pooling / set encoding

重点回答：
- 如何把 64 维当作集合而不是序列。
- 这种方法能否替代“通道邻接”假设。

### 3.3 光谱 / 通道分组策略
查这些方向：
- hyperspectral band grouping
- multispectral band attention
- physical-prior grouping
- variance / SNR based grouping
- band-wise clustering

重点回答：
- 物理先验分组、统计先验分组、学习型分组，哪种更适合当前任务。
- 传统 RGB/IR/TIR 分组思想能否迁移到 AlphaEarth 64 维。

### 3.4 低频通道补偿
查这些方向：
- band-wise loss reweighting
- gradient balancing across channels
- channel reweighting
- focal-style regression weighting
- SE / CBAM / ECA variants

重点回答：
- 如何让低方差通道获得更多梯度。
- 哪些方法适合回归任务，而不是分类。

### 3.5 Gumbel 路由替代方案
查这些方向：
- soft grouping
- Sinkhorn-Knopp routing
- differentiable clustering
- K-means attention
- top-k routing

重点回答：
- 为什么 Gumbel 容易失效。
- 哪些替代方案更稳定、更适合小数据。

### 3.6 轻量后处理模块
查这些方向：
- residual refinement head
- lightweight enhancement block
- output correction module
- late upsampling refinement

重点回答：
- 哪些模块能直接挂在 SwinIR 的 conv_last 后面。
- 是否适合做残差补偿、门控补偿，而不是强重建。

### 3.7 光谱感知损失
查这些方向：
- SAM loss
- SID loss
- spectral-angle regression loss
- per-band weighted reconstruction loss

重点回答：
- 哪种损失更适合高维连续特征回归。
- 哪些损失能明显改善低频通道。

---

## 4. 必须比较的维度

| 维度 | 要回答什么 |
|---|---|
| 适配对象 | 无序高维向量，还是传统光谱序列 |
| 稳定性 | 是否容易训练崩，是否对初始化敏感 |
| 复杂度 | 是否足够轻量，能否用于后处理 |
| 解释性 | 分组是否可解释，是否对应物理/统计含义 |
| 小数据友好性 | 是否适合 T=20 这种小样本设置 |
| 兼容性 | 是否能接入 SwinIR late-upsampling / V2 / V3 |
| 低频帮助 | 是否有证据改善低方差通道 |

---

## 5. 需要重点找的反例

不要只找成功案例，也要找失败原因：
- Gumbel 路由为什么不稳。
- 双分支 / 多分支后处理为什么退化。
- 传统光谱卷积为什么不适合 AlphaEarth 64 维。
- stage 为什么必须配合 gated，而不能单独用。

---

## 6. 推荐关键词

### Group / Routing
- group attention
- dynamic grouping attention
- routing attention
- soft attention grouping
- sparse channel attention

### Set / Permutation
- permutation invariant attention
- set transformer channels
- unordered feature modeling
- vision permutator channel

### Spectral / Band
- hyperspectral band grouping
- multispectral band attention
- band-wise loss reweighting
- spectral angle mapper loss regression

### Stable Routing
- Sinkhorn routing
- soft routing network
- k-means attention
- differentiable clustering routing

### Post-processing
- lightweight refinement head
- residual attention refinement
- output correction module

---

## 7. 最终输出格式

请输出一份能直接用于方案决策的调研结论，必须包含：
1. **核心问题总结**
2. **文献分组对比表**
3. **失效机制与反例**
4. **可落地方案排序**
5. **下一步实验建议**

### 推荐表格
| 方案 | 核心思想 | 稳定性 | 复杂度 | 对低频通道帮助 | 是否适合当前体系 |
|---|---|---:|---:|---:|---:|

### 推荐结论
- 首选方案：
- 次选方案：
- 不推荐方案：
- 主要风险：
- 下一步实验：

---

## 8. 一句话任务描述

**请围绕“AlphaEarth 64 维无序高维特征 + 分组注意力 + 低频通道补偿 + 轻量后处理”检索最相关文献，重点给出稳定、可落地、可解释的方案排序。**
