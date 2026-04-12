# AEF_SwinIR 研究任务书（给 Gemini 的 Deep Research 版）

**用途**: 作为 Gemini 的检索与归纳任务输入，目标是快速补足“无序高维特征的分组注意力 + 后处理 + 损失设计”的文献与方案依据。  
**当前阶段**: 已完成 V3 后处理全量验证，当前最优结果为 **14.8053 PSNR**。  
**项目主线**: Landsat 8 低分辨率多时相输入 → SwinIR 主干 → 云感知/时间融合 → 64 维 AlphaEarth-like 高维输出。

---

## 1. 研究背景

### 1.1 任务目标
我们要研究的是：**如何在不修改或少改主干 SwinIR 的前提下，针对 AlphaEarth 64 维无序高维嵌入，设计更稳定、更可解释、更适合低频通道的分组注意力与后处理方案。**

### 1.2 当前系统输入输出
- 输入：Landsat 8 低分辨率多时相图像，9 波段，约 30m，时间维 T=20。
- 附加输入：DOY 时间编码、云掩膜 indicating_mask、QA 质量信息。
- 输出：64 通道高分辨率特征，空间上做 3× 上采样，目标与 AlphaEarth 兼容。
- 关键约束：AlphaEarth 64 维特征不是按波长排序的传统光谱序列，而是无序的高维融合表示。

### 1.3 已知问题
- 低频/非 RGB 通道明显更差，MSE 类损失会偏袒高方差通道。
- 传统 1D/3D 光谱卷积的“邻域相邻”假设对 AlphaEarth 64 维并不成立。
- 复杂后处理曾出现退化：Gumbel 路由、双分支、多分支 stage 组合都不稳定。
- 当前 V3 方案中，stage + gated 的残差调度是已验证的有效方向。

---

## 2. 研究问题定义

请围绕以下 6 个问题做检索与归纳，不要扩展到过宽的通用综述：

1. **无序高维特征如何做分组注意力？**
   重点找适合 set / permutation-invariant 特征的注意力、路由、分组机制。

2. **通道分组依据应该是什么？**
   重点找物理先验分组、统计先验分组、学习型自适应分组三类依据。

3. **如何解决低频/非 RGB 通道偏弱？**
   重点找跨通道梯度平衡、通道重加权、band-wise loss、重要性采样等方法。

4. **Gumbel 路由为什么容易失效，替代方案有哪些？**
   重点找更稳定的 soft grouping、Sinkhorn、K-means/EM 风格分组、top-k routing。

5. **Post-processing 是否存在更轻量、更适配 SwinIR late-upsampling 的方案？**
   重点找残差注意力、轻量细化块、局部动态校正、输出端补偿结构。

6. **哪些损失函数与分组注意力能形成互补？**
   重点找 SAM、SID、band-adaptive loss、focal-style band reweighting 及其组合方式。

---

## 3. 研究主假设

请优先围绕下面 4 个假设找证据，而不是泛泛罗列论文：

### 假设 A：置换不变性优于固定拓扑卷积
如果 64 维特征本质是无序集合，那么 set-based attention、permutation-invariant attention、动态路由会比固定邻域卷积更合理。

### 假设 B：动态分组应服从物理/统计双先验
分组不应只靠 learnable clustering，还应参考 RGB/IR/TIR 等物理族群，或高方差/低方差、SNR 等统计特征。

### 假设 C：低频通道需要显式梯度补偿
仅靠普通 MSE 会继续偏袒高方差通道，因此需要 band-wise reweighting、SAM/SID 或难例式加权。

### 假设 D：后处理应轻量且可控
最有希望的是“残差式、门控式、轻量分组式”的输出端补偿，而不是复杂重分支堆叠。

---

## 4. 需要调研的方向

### 4.1 Group Attention / 分组注意力
重点检索：
- Group Attention
- Channel Grouping Attention
- Dynamic Grouping
- Routing Attention
- Sparse / Token Grouping
- Mixture-of-Experts for channels

重点判断：
- 是否支持无序输入。
- 是否能把 64 维向量分成若干可解释子组。
- 是否适合小数据、低样本、输出端后处理场景。

### 4.2 光谱分组策略
重点检索：
- spectral grouping for hyperspectral / multispectral
- band grouping by physical priors
- band clustering by variance / SNR / mutual information
- RGB, NIR, SWIR, TIR grouping

需要明确：
- 传统光谱分组的依据能否迁移到 AlphaEarth。
- 哪些方法是“按波长顺序”，哪些是“按语义/统计分组”。
- 哪些方法更适合做后处理，而不是主干改造。

### 4.3 置换不变性建模
重点检索：
- Set Transformer
- Vision Permutators
- permutation-invariant attention
- set pooling / set encoding
- unordered vector modeling

需要重点回答：
- 如何把 64 维特征当作集合而不是序列。
- 对 token 顺序不敏感的 attention 是否可用于通道维。
- 这种方法与 group attention 的关系是什么。

### 4.4 跨通道梯度平衡
重点检索：
- channel reweighting
- band-wise loss reweighting
- gradient balancing across channels
- focal loss for regression or dense prediction
- SE / CBAM / ECA variants for importance weighting

需要重点回答：
- 如何让低方差通道获得更多优化资源。
- 哪些方法适合 regression，而不是 classification。
- 哪些方法能与分组注意力叠加。

### 4.5 Gumbel 路由替代方案
重点检索：
- soft grouping
- Sinkhorn-Knopp routing
- differentiable clustering
- K-means attention
- top-k routing / sparse routing

需要重点回答：
- 哪些替代方案比 Gumbel 更稳定。
- 哪些方案适合小数据训练。
- 哪些方案在推理成本上更可控。

### 4.6 Post-processing 轻量细化模块
重点检索：
- lightweight refinement block
- residual refinement head
- attention-based enhancement head
- output correction module
- late upsampling refinement

需要重点回答：
- 能否直接接在 SwinIR 的 conv_last 之后。
- 是否适合做残差补偿而非强重建。
- 是否可与 stage / gated 机制结合。

### 4.7 光谱感知损失
重点检索：
- SAM loss
- SID loss
- spectral-angle based regression loss
- band adaptive loss
- per-band weighted reconstruction loss

需要重点回答：
- 哪种损失更适合高维连续特征回归。
- 是否能专门改善低方差通道。
- 是否可以和分组注意力联合训练。

---

## 5. 方案对比维度

请把每篇重要文献或每种方案，都尽量按下面维度比较：

| 维度 | 需要回答的问题 |
|---|---|
| 适配对象 | 是适合无序高维向量，还是传统光谱序列 |
| 稳定性 | 是否容易训练崩、对初始化是否敏感 |
| 参数开销 | 是否能控制在轻量后处理范围内 |
| 解释性 | 分组是否可解释，是否能对应物理/统计含义 |
| 小数据友好性 | 是否适合 T=20 这种小样本设置 |
| 与当前体系兼容性 | 是否可接入 SwinIR late-upsampling / V2 / V3 |
| 对低频通道的帮助 | 是否有证据改善低方差通道 |

---

## 6. 需要重点验证的反例

请不要只找“成功案例”，还要找反例和失效原因，尤其是：

- 为什么 Gumbel 路由在这个任务里可能不稳。
- 为什么复杂双分支/多分支后处理容易退化。
- 为什么传统光谱卷积在 AlphaEarth 64 维场景里不成立。
- 为什么仅靠结构变复杂未必提升 PSNR。
- 为什么 stage 单独使用可能过度调制，必须配合 gated。

---

## 7. 检索关键词建议

请优先用这些关键词组合检索，而不是只搜“遥感超分”泛词：

### 7.1 Group / Routing / Attention
- group attention
- dynamic grouping attention
- routing attention
- soft attention grouping
- differentiable clustering attention
- sparse channel attention

### 7.2 Set / Permutation
- permutation invariant attention
- set transformer channels
- unordered feature modeling
- set pooling vision
- vision permutator channel

### 7.3 Spectral / Band
- hyperspectral band grouping
- multispectral band attention
- band-wise loss reweighting
- spectral angle mapper loss regression
- spectral information divergence loss

### 7.4 Stabilized Routing
- Sinkhorn routing
- soft routing network
- k-means attention
- differentiable clustering routing
- top-k expert routing

### 7.5 Refinement / Post-processing
- lightweight refinement head
- residual attention refinement
- output correction module
- post-processing enhancement network
- late upsampling refinement

---

## 8. 输出要求

请最终输出一份适合直接继续做方案设计的调研报告，结构必须包含：

1. **核心问题总结**
2. **架构理解与适配点**
3. **文献分组对比表**
4. **失效机制与反例总结**
5. **可落地方案排序**
6. **推荐的下一步实验计划**

### 8.1 推荐对比表格式
| 方案 | 核心思想 | 稳定性 | 复杂度 | 对低频通道帮助 | 是否适合当前体系 | 备注 |
|---|---|---:|---:|---:|---:|---|

### 8.2 推荐结论格式
- 首选方案：
- 次选方案：
- 不推荐方案：
- 主要风险：
- 下一步实验：

---

## 9. 研究边界

请严格控制研究边界，不要跑偏到以下方向：

- 不要把问题泛化为普通图像分类或检测。
- 不要只做通用 Transformer 文献综述。
- 不要把主线变成“重写主干网络”。
- 不要忽略当前已有的 V2/V3 后处理结果。
- 不要把传统光谱顺序假设直接套到 AlphaEarth 64 维。

---

## 10. 当前优先级

### P0
- 无序高维特征的置换不变性 / group attention 文献。
- Gumbel 路由失效原因与替代方案。
- 轻量后处理模块与 SwinIR late-upsampling 的兼容性。

### P1
- 光谱分组策略与 band-wise reweighting。
- SAM / SID / band-adaptive loss 与分组注意力结合。

### P2
- 更大范围的遥感超分文献扩展，但必须服务于上述主线。

---

## 11. 一句话任务描述

**请围绕“AlphaEarth 64 维无序高维特征 + 分组注意力 + 低频通道补偿 + 轻量后处理”四条主线，检索最新且最相关的文献，给出可落地的方案排序、失败原因和下一步实验建议。**
