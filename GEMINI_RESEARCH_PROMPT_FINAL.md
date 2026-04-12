# Gemini 正式 Prompt

请围绕以下主题进行 deep research，并输出一份可直接用于方案设计的调研报告。研究范围必须限定在 **可直接嵌入现有 AEF_SwinIR V3 框架** 的方案，不要扩展成重写主干网络的通用综述。

## 任务要求

1. 检索无序高维特征（Unordered High-dim Features）的置换不变性注意力机制，重点关注 Set Transformer、Vision Permutators 在通道维度建模中的应用。
2. 调研遥感领域针对多光谱/高光谱数据的通道分组（Band Grouping）策略，对比物理先验（波段属性）、统计先验（方差/SNR）与数据驱动（聚类/动态路由）的优劣。
3. 分析低频/非 RGB 通道在 MSE 损失下欠优化的数学机理，检索梯度平衡（Gradient Balancing）、通道重加权（Channel Reweighting）及难例挖掘在回归任务中的解决方案。
4. 评估 Gumbel-Softmax 路由在小样本/高维场景下的不稳定性原因，并寻找 Sinkhorn-Knopp 软路由、Top-k 稀疏路由及可微聚类（Differentiable Clustering）等稳定替代方案。
5. 针对 SwinIR late-upsampling 架构，检索轻量级图像细化模块（Refinement Block）和残差校正头（Correction Head）的设计，探讨其与 Stage/Gated 机制的结合方式。
6. 调研光谱感知损失函数（SAM、SID、光谱一致性损失）对高维嵌入重建的贡献，设计能够改善低方差通道表现的 Band-adaptive Loss 方案。
7. 综合检索结果，分析 Gumbel 路由及复杂双分支方案失效的深层原因，总结“失效机制与反例”，并明确哪些方案不推荐。
8. 整理出一份包含“核心问题总结、文献对比表、可落地算法排序、下一步实验计划”的调研报告，确保结论能直接用于现有 V3 框架的后续实现。

## 必须回答的问题

- 哪些方法最适合无序高维通道建模。
- 哪些 band grouping 依据最适合当前任务。
- 哪些方案能缓解低频/非 RGB 通道欠优化。
- 哪些稳定替代方案可替换 Gumbel 路由。
- 哪些 refinement/correction head 能直接接入 late-upsampling 输出。
- 哪些损失函数最适合和分组注意力联合使用。
- 哪些方案已经被证明不稳定或不推荐。

## 输出格式

请按以下结构输出：

1. 核心问题总结
2. 文献对比表
3. 失效机制与反例
4. 可落地算法排序
5. 下一步实验计划

## 对比维度

每个方案至少比较以下维度：

- 适配对象：无序高维向量 / 光谱序列
- 稳定性：是否易训练崩溃、是否对初始化敏感
- 复杂度：是否适合作为轻量后处理
- 解释性：分组是否可解释，是否对应物理或统计意义
- 小数据友好性：是否适合 T=20 这种场景
- 兼容性：是否可接入 SwinIR late-upsampling / V2 / V3
- 低频帮助：是否有证据改善低方差通道

## 检索关键词

- group attention
- dynamic grouping attention
- routing attention
- soft attention grouping
- permutation invariant attention
- set transformer channels
- unordered feature modeling
- hyperspectral band grouping
- multispectral band attention
- band-wise loss reweighting
- gradient balancing across channels
- sinkhorn routing
- differentiable clustering routing
- lightweight refinement head
- residual attention refinement
- SAM loss regression
- SID loss regression

## 研究边界

- 不要把问题泛化为普通图像分类或检测。
- 不要只做通用 Transformer 文献综述。
- 不要把主线变成重写主干网络。
- 不要忽略当前已有的 V2/V3 后处理结果。
- 不要把传统光谱顺序假设直接套到 AlphaEarth 64 维。
