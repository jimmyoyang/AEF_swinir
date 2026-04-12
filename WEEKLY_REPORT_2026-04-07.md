# 4-7 周报

## 1. 本周核心目标

本周主要围绕“归一化职责下沉”进行工程改造，目标是：

1. 保持现有数据与训练流程不变（旧脚本不改名、不改逻辑）
2. 单独产出新脚本，支持后续参数化归一化实验
3. 在不重新切片的前提下，先完成可切换的代码路径

在此基础上，近期进一步聚焦两个现实问题：

1. 在不重新切数据的情况下，尽量让正式实验可跑、可比、可复现
2. 明确训练提速与验证提速分别应该改什么参数，避免把 sample_num 和 batch_size 混用

---

## 2. 新增脚本（已完成）

本周已新增以下两个脚本：

1. datapipe/prepare_data_local_split_only.py
2. datapipe/datasets_with_lr_norm.py

### 2.1 datapipe/prepare_data_local_split_only.py

功能定位：

1. 只负责切片和数据划分
2. 不做任何 LR 物理归一化或值变换

这样可以确保切片阶段纯粹、职责单一，后续归一化策略改动不需要重新切片。

### 2.2 datapipe/datasets_with_lr_norm.py

功能定位：

1. 在 Dataset 读取阶段做归一化
2. 支持参数化 LR 物理归一化（可开关）
3. 保持与原数据结构兼容

本脚本新增了可配置函数：

1. apply_landsat_lr_physical_normalization(lr_img, cfg)

支持参数包括：

1. enabled
2. reflective_band_indices
3. thermal_band_indices
4. reflectance_scale
5. thermal_offset
6. thermal_scale
7. clip_to_unit
8. auto_detect_raw_input
9. raw_input_threshold

---

## 3. 兼容性与影响评估（已完成）

### 3.1 导入验证

已完成导入测试，结果通过：

1. datasets_with_lr_norm 导入成功
2. prepare_data_local_split_only 导入成功

### 3.2 对现有训练/推理代码影响

结论：默认情况下无需改 train/main/inference 脚本。

原因：

1. 新 Dataset 保持原有返回键名与张量结构兼容
2. 新增归一化参数有默认值，不启用时行为接近旧流程

---

## 4. 如何调用新脚本

### 4.1 切片脚本调用

如需使用“纯切片版”，执行时改为调用：

1. datapipe/prepare_data_local_split_only.py

说明：该脚本不做归一化，只产出切片与 split 结果。

### 4.2 训练配置切换（最小改动）

在配置文件中将 Dataset target 改为：

1. datapipe.datasets_with_lr_norm.AnytimeTemporalDataset

或（SRCNN 路径）：

1. datapipe.datasets_with_lr_norm.PreprocessedTileDataset

### 4.3 启用参数化 LR 归一化

在 data.*.params 中增加 lr_physical_norm 配置即可启用；
若不加该段，默认不强制改变现有行为。

---

## 5. 当前状态

### 5.1 已完成

1. 新脚本创建完成
2. 新脚本核心逻辑修改完成
3. 导入可用性验证完成
4. 保持旧脚本命名与原流程不变

### 5.2 待完成

1. 小样本训练冒烟验证（使用新 Dataset 路径）
2. 归一化参数敏感性对比（开关、阈值、热红外参数）
3. 决定是否进入全量实验

### 5.3 近期训练策略判断（新增）

1. 训练速度主要受 batch_size 影响，而不是 sample_num
2. sample_num 更适合用于快测或缩短验证时间，不应作为正式训练提速手段
3. 正式实验中，train 侧应尽量使用全量数据；val/test 可在调试阶段保留小样本，但最终结果汇报前需要回到全量评估
4. 对于时序模型，batch 的上限还受到时序长度、输入分辨率、通道数、额外分支和优化器状态影响

---

## 6. 风险与注意事项

1. 旧数据若已在切片阶段做过归一化，切换新 Dataset 时要避免重复归一化
2. 热红外波段索引需与实际输入通道顺序一致
3. 建议先在小规模样本上检查 loss 曲线与数值范围，再上全量训练
4. 训练集与验证集的样本截断会影响实验口径，正式对比时应统一评估协议
5. 目前已经确认 val/LR 与 val/HR 命名一一对应，但 patch_stats.log 的过滤 key 需要与 processed_data 的实际文件名兼容，否则过滤会被跳过

---

## 7. 资源评估与实验可行性（新增）

### 7.1 当前 GPU 资源

当前服务器 GPU 为 NVIDIA H100 80GB HBM3。nvidia-smi 显示：

1. 总显存约 81559 MiB
2. 当前已用约 726 MiB
3. 当前剩余约 80499 MiB

说明：当前 GPU 基本空闲，可以直接启动正式训练。

### 7.2 对 batch_size 的判断

1. batch_size 是正式训练提速和显存利用率的主要调节项
2. sample_num 不应替代 batch_size，用它来“提速训练”会改变数据量口径
3. 若 batch 受显存限制，可优先考虑梯度累积，而不是缩减训练集

### 7.3 训练/验证时间瓶颈

1. 训练集全量跑会更慢，但这是正式实验应该承担的开销
2. 验证集若频繁跑全量，会成为明显的时间瓶颈
3. 因此在冒烟验证阶段可以缩小 val/test 规模或降低 val_freq，但正式结论阶段需统一口径回到全量验证

---

## 8. 下周计划

1. 完成新 Dataset 路径的小规模训练验证
2. 记录“启用/禁用 LR 物理归一化”的定量差异
3. 若结果稳定，形成统一配置模板并推进到主实验流程
4. 进一步统一正式实验口径：训练集全量、验证集全量、batch_size 固定、只在调试阶段使用 sample_num
5. 若验证过慢，优先调整 val_freq 或采用梯度累积，而不是压缩训练数据

---

## 9. 阶段结论

本周已完成“旧流程不动、新流程并行”的目标。当前代码结构已经支持：

1. 维持现有实验连续性
2. 用新脚本快速迭代归一化策略
3. 在不重新切片的前提下推进后续验证

同时，近期对正式实验的资源和口径已形成更明确判断：

1. 当前 H100 80GB 资源充足，具备直接启动正式训练的条件
2. 正式实验应以 batch_size 为主要提速/调参点
3. sample_num 更适合调试或冒烟验证，不应长期作为训练设置
4. 训练和验证的样本选择、验证频率和过滤规则需要在各配置间统一，否则结果不可直接比较

---

## 10. 组会专题：V3 高维光谱建模与动态路由方案整合（新增）

本节用于组会汇报，目标是把“旧方案结果、失效机制、新方案替代关系、核心原理、实验落地计划”放在同一条叙事链里，便于快速判断技术路线。

### 10.1 当前共识与问题定义

在 AEF_SwinIR V3 框架下，当前已验证的结论是：

1. stage + gated 是目前后处理稳定增益的关键组合
2. 复杂动态路由与双分支在小样本（T=20）下容易退化
3. 低频/非 RGB 通道持续欠优化，是限制上限的核心瓶颈

对应的工程问题是：

1. 如何在不重写主干的前提下，提升通道维建模能力
2. 如何在 loss 层显式补偿低方差通道
3. 如何用更稳定的路由机制替代 Gumbel 类硬选择
4. 如何在 late-upsampling 输出端低开销补偿细节与光谱一致性

### 10.2 旧方案回顾（简版结果）

| # | 方案名称 | 参数增量 | 延迟增量 | 实现难度 | 框架兼容性 | 优先级 | 风险等级 |
|---|---|---:|---:|---|---|---:|---|
| 1 | Gumbel-Softmax动态路由 | ~5% | ~8% | 中 | 高 | 2 | 中 |
| 2 | 信息熵驱动重排序 | ~0% | ~2% | 低 | 极高 | 4 | 低 |
| 3 | 异构双分支残差补偿 | ~10% | ~15% | 中 | 高 | 3 | 低 |
| 4 | 置换不变集合注意力 | ~15% | ~18% | 高 | 中 | 5 | 中 |
| 5 | 非均衡几何对齐适配 | ~2% | ~1% | 极低 | 极高 | 1 | 极低 |
| 6 | 矩阵重塑与多尺度融合 | ~12% | ~20% | 中 | 中 | 6 | 中 |
| 7 | 动态图神经网络拓扑 | ~22% | ~35% | 极高 | 低 | 7 | 极高 |
| 8 | Mamba谱域全局重构 | ~18% | ~25% | 极高 | 低 | 8 | 高 |

关键结论：根据框架约束与实现成本，方案5（适配层+非均衡loss）应被作为首战突破口，方案1和3作为第二阶段补充。方案7超出框架能力，方案8理论前沿但工程风险高。方案2可作为低成本 baseline。

方案1：基于Gumbel-Softmax的动态通道路由（Gumbel Permutation Routing, GPR）

理论基础（Lee et al. ICCV 2023, Bengio et al. ICML 2014）：
通过Gumbel-Softmax重参数化使离散路由可微，将无序64维通道动态聚类为K个伪模态子空间（预设K=4：高频结构组、低频背景组、纹理边缘组、宏观气象组）。每次前向传播时，模型学习每个通道的组归属概率，并生成硬One-Hot路由掩码。

方案3：异构双分支残差补偿过滤（Asymmetric Dual-Branch Refinement, ADBR）

理论基础（ShiftLUT思想的遥感适配）：根据预先的统计分析将64维特征硬性区分为高频组（Top-32通道，MSE贡献最大）与低频组（Bottom-32通道，低变异）。两组采用完全不同的处理策略：高频分支采用标准3x3残差卷积，低频分支采用全局池化+SE再加权。

方案5：非均衡几何对齐适配层（Non-equilibrium Anisotropic Geometric Alignment, NAGA）

理论基础（Lin et al. ICCV 2017 Focal Loss, Wang et al. CVPR 2021 对比学习）：
该方案核心创新在loss层面而非网络拓扑。主干网络输出后串联一个极简的恒等初始化深度可分离卷积adapter（参数仅1-2%），配合基于超球面几何的非均衡损失函数。在反向传播中，系统动态追踪验证集per-band MSE，为优化最难的非RGB通道施加指数级的梯度权重倾斜。

方案6：矩阵重塑与多尺度群组融合（Matrix Reshaping & Spectral Odd-Even Fusion, SOEF）

理论基础（IEEE TGRS 2025 MGFF-ViT, ConvNeXt V2多尺度思想）：
通过升维手段化解序列无序难题。将原本1维无序的64通道特征向量重塑为8x8二维网格，把跨通道远距离一维关联转化为二维空间感受野关联问题，随后在该网格上应用多尺度群组卷积。

| 方案 | PSNR | 相对 V3 基线 (14.7191) | 结论 |
|---|---:|---:|---|
| v3_5e_scheme1_plus6_stage_gated | 14.8053 | +0.0862 | 当前最优 |
| v3_5e_scheme1_plus6_gated_lightweight | 14.8046 | +0.0855 | 与最优接近 |
| v3_5f_scheme3_plus6_stage_gated | 14.7713 | +0.0522 | 接近 4d 锚点 |
| v3_5a_nopost_bestpre | 14.7191 | +0.0000 | V3 基线 |
| v2_5b_scheme1_gumbel | 14.6675 | -0.0516 | 路由不稳，收益不足 |
| v2_5c_scheme3_dual_branch | 14.3000 | -0.4191 | 明显退化 |

补充说明：

1. 历史 3b 高分未稳定复现，不作为主结论
2. 当前稳定主线应以 V3 stage+gated 体系为锚点继续扩展

### 10.3 失效机制总结（为什么旧方案会退化）

| 旧问题 | 深层机理 | 观测表现 |
|---|---|---|
| Gumbel 路由不稳定 | 温度退火 + 离散采样导致梯度方差大，易单路径坍塌 | 分支利用率失衡，结果波动大 |
| 双分支退化 | 分支学习不对称，MSE 对高能量分量偏置放大 | 一支路主导，另一支路趋于失效 |
| 仅 stage 无门控 | 残差调制缺乏约束，易过调制主干输出 | PSNR 下滑、训练后期不稳 |
| 低频通道长期偏弱 | MSE 梯度与方差正相关，低方差波段梯度被淹没 | 非 RGB 通道恢复不足 |

### 10.4 旧方案与新方案的替代关系（组会重点）

| 旧方案/旧做法 | 主要问题 | 新方案替代 | 替代逻辑 |
|---|---|---|---|
| Gumbel-Softmax 硬路由 | 小样本下路由塌缩、训练震荡 | Sinkhorn 软路由 / Soft-grouping | 用连续双随机约束保持分支负载均衡 |
| 双分支硬拆分（结构先行） | 参数重、分支失衡、收益不稳 | 通道维轻量交互（ISAB/ViP-lite 后处理） | 先做通道聚合再细化，降低结构风险 |
| 纯 MSE | 低频通道欠优化 | Band-adaptive + SAM/SID 混合损失 | 在 loss 层显式提升弱通道梯度份额 |
| 固定分组或无分组 | 无法适配无序高维通道关系 | 统计初始化 + 可微 Soft-grouping | 兼顾稳定起点与动态适配 |
| 无专门输出细化 | 上采样后伪影和局部细节不足 | RRB/Gated Correction Head | 在输出端低开销做残差补偿 |

### 10.5 新方案原理说明（给决策用）

#### 10.5.1 通道建模：ISAB（Set Transformer）

原理：

1. 把通道特征看作无序集合，而不是固定顺序序列
2. 用少量诱导点先聚合再回写，复杂度从全连接注意力下降到近线性形式
3. 诱导点可理解为通道原型，对弱通道有跨通道信息借用作用

为什么适合 V3：

1. 可放在通道维后处理位置，不需要改 SwinIR 主干
2. 对通道顺序不敏感，更匹配 AlphaEarth 高维无序特征属性

#### 10.5.2 通道建模：ViP-lite（Vision Permutator 轻量化）

原理：

1. 将 H/W/C 三轴解耦混合，低成本建立跨轴交互
2. 用线性投影替代重注意力矩阵，计算开销可控

为什么适合作为备选：

1. 实现简单，部署成本低
2. 但对无序集合的归纳偏置弱于 ISAB，建议作为次选

#### 10.5.3 路由稳定化：Sinkhorn 软路由

原理：

1. 通过迭代归一化把路由矩阵约束到近似双随机空间
2. 避免硬选择导致的未选分支零梯度
3. 提升多分支负载均衡，减少单分支过载

为什么替代 Gumbel：

1. 在 T=20 小样本下更稳，收敛波动更小
2. 与现有 V3 分组/路由组件可以逐步替换，不必一次性重构

#### 10.5.4 损失重构：Band-adaptive + SAM/SID

原理：

1. 对低方差通道给予更高损失权重，缓解梯度失衡
2. SAM 约束光谱角度一致性，SID 约束分布一致性
3. 与像素重建项组合，避免只追求角度一致而丢失数值精度

为什么优先级高：

1. 不改网络结构，实施成本最低
2. 可直接验证低频通道是否得到实质提升

#### 10.5.5 输出补偿：RRB / Gated Correction Head

原理：

1. 在 late-upsampling 后增加轻量残差修正
2. 用门控选择哪些区域做强补偿、哪些区域保持平滑
3. 以低计算开销修复棋盘格伪影和纹理缺失

为什么必要：

1. 主干在 LR 空间提特征，最终 HR 细节恢复依赖输出端补偿
2. 与 stage+gated 主线机制具有天然兼容性

### 10.6 新方案对比表（重点汇报页）

| 新方案ID | 新方案 | 主要替代对象 | 复杂度 | 小样本稳定性 | 预期收益 | 主要风险 | 推荐级别 |
|---|---|---|---|---|---|---|---|
| N1 | Band-adaptive Loss | 纯 MSE | 低 | 高 | +0.10~0.25 | 权重过大导致训练震荡 | 高 |
| N2 | SAM+SID 混合损失 | 仅像素损失 | 低 | 中高 | +0.10~0.30 | 与 PSNR 目标需权重平衡 | 高 |
| N3 | GradNorm 通道平衡 | 固定损失权重 | 低 | 中高 | +0.05~0.20 | 超参敏感 | 中高 |
| N4 | 统计初始化 + Soft-grouping | 固定分组 / 无分组 | 低中 | 高 | +0.10~0.25 | 分组数与温度系数敏感 | 高 |
| N5 | ISAB 通道后处理块 | 线性通道混合 | 中 | 中高 | +0.15~0.35 | 诱导点数需调参 | 高 |
| N6 | ViP-lite 通道混合头 | 常规 MLP 混合 | 低中 | 中高 | +0.08~0.20 | 集合归纳偏置偏弱 | 中 |
| N7 | Sinkhorn 软路由 | Gumbel 硬路由（方案1） | 中 | 高 | +0.10~0.30 | 迭代带来轻微时延 | 中高 |
| N8 | RRB/Gated Correction Head | 无专门输出细化 | 低中 | 高 | +0.10~0.25 | 补偿过强会过锐化 | 高 |

#### 10.6.1 N1: Band-adaptive Loss

原理：

1. 不再对所有通道使用同一套 MSE 权重，而是根据 band-wise 方差、残差或梯度量级给不同通道分配不同损失权重
2. 低方差、低能量通道获得更高的反向梯度份额，避免高方差通道长期抢梯度
3. 本质上是把通道欠优化问题从结构层前移到优化层解决

框架适用性：

1. 直接接入现有 V3 trainer 即可，不需要改模型主干
2. 对 dataloader、forward 结构、postprocessor 都是零侵入或低侵入
3. 适合先做冒烟实验和主实验第一步

优点：

1. 实现成本最低
2. 不增加推理开销
3. 对低频通道的改善最直接，风险可控

缺点：

1. 权重设计不当会引入训练震荡
2. 如果权重过强，可能牺牲整体 PSNR 稳定性
3. 只能补偿优化偏差，不能单独解决结构表达不足

替代关系：

1. 主要替代纯 MSE
2. 可与 N2、N3 组合使用，形成损失+梯度平衡双保险

#### 10.6.2 N2: SAM+SID 混合损失

原理：

1. SAM 约束光谱向量方向一致性，关注形状是否对齐
2. SID 约束光谱分布一致性，关注概率结构是否一致
3. 与像素级重建项组合后，既保留数值精度，也补足光谱一致性

框架适用性：

1. 可直接作为 V3 辅助损失加入，工程侵入性低
2. 特别适合对 64 维高维嵌入做重建约束
3. 适合与 N1、N8 同时使用

优点：

1. 对非 RGB 通道和低频通道友好
2. 有明确光谱物理意义，组会易解释
3. 和 V3 当前重建目标兼容

缺点：

1. SAM 更偏几何方向，SID 更偏分布差异，二者权重需要调
2. 若权重配比不稳，可能出现角度对了但数值不准或相反情况
3. 对极小 batch 时统计波动可能更明显

替代关系：

1. 替代仅像素损失或纯 MSE
2. 可作为 N1 的光谱补充，形成 band-adaptive + spectral-aware 组合

#### 10.6.3 N3: GradNorm 通道平衡

原理：

1. 将各通道视作多个优化目标，自动平衡它们的梯度范数
2. 若某些通道学习过慢，则提升其损失权重；若学习过快，则适度压低
3. 让训练过程更接近通道间公平优化，而不是高方差通道优先

框架适用性：

1. 不改网络结构，只改 loss 权重调度
2. 可作为 N1 的动态版补强
3. 与 V3 训练循环兼容，适合做中期稳定化实验

优点：

1. 能缓解手工权重难统一的问题
2. 对多通道回归任务自然
3. 可以和 N1、N2 叠加使用

缺点：

1. 超参数比较敏感，尤其是平衡系数
2. 需要监控各通道梯度和 loss 曲线，调试成本高于 N1/N2
3. 对极小 batch 噪声更敏感

替代关系：

1. 替代固定损失权重
2. 可与 N1 联合，做静态 + 动态双层平衡

#### 10.6.4 N4: 统计初始化 + Soft-grouping

原理：

1. 先按波段方差、相关系数、SNR 等统计指标构造初始分组
2. 再用可微 soft-grouping 让分组在训练中微调，而不是一开始就硬拆分
3. 保留物理/统计先验，同时允许模型适应任务分布

框架适用性：

1. 适合做输入侧或瓶颈侧轻量分组模块
2. 对小样本友好，因为有稳定初始化起点
3. 适合当前不能重写主干的约束

优点：

1. 物理解释性强，组会容易讲清楚
2. 比纯数据驱动聚类更稳
3. 比固定分组更灵活

缺点：

1. 分组数、温度系数、初始化策略都要调
2. 如果统计先验选得不好，可能把有用弱信号分散
3. 单独使用时提升可能不如后处理和损失改进直接

替代关系：

1. 替代固定波段分组
2. 替代完全依赖深度聚类的初始化方式

#### 10.6.5 N5: ISAB 通道后处理块

原理：

1. 把通道视为集合，用少量诱导点先做全局压缩，再回写到原通道
2. 相当于在瓶颈处学习一组通道原型，弱通道可借用强通道信息
3. 既能建模全局通道关系，又不会像全自注意力那样计算爆炸

框架适用性：

1. 适合插在 conv_last 后或瓶颈层后，作为通道维后处理模块
2. 不需要修改 SwinIR 的窗口注意力主干
3. 是当前无序高维通道建模的强候选

优点：

1. 对无序通道贴合
2. 小样本下比复杂路由更稳
3. 计算量可控，机制清晰

缺点：

1. 诱导点数 m 需要调参
2. 作为新增模块，代码实现比纯 loss 复杂
3. 若输出层特征过弱，ISAB 收益会受限

替代关系：

1. 替代线性通道混合或简单 MLP 通道头
2. 替代把通道当序列硬卷积的做法

#### 10.6.6 N6: ViP-lite 通道混合头

原理：

1. 参考 Vision Permutator 轴分解思想，在 H/W/C 三维做轻量交互
2. 用 Permute-MLP/线性投影替代重注意力矩阵
3. 通过显式轴交换建立空间-通道耦合，不引入大规模 attention 开销

框架适用性：

1. 适合做 ISAB 轻量备选，或作为多分支增强头
2. 部署和实现简单，易快速出结果
3. 更适合增强空间-光谱联合表达，不一定最适合纯通道集合建模

优点：

1. 计算轻
2. 结构直观，工程落地容易
3. 适合作为快速 baseline 或备选路线

缺点：

1. 对通道无序性的归纳偏置不如 ISAB 强
2. 更偏位置敏感建模，对纯集合属性不够敏感
3. 作为主创新点说服力略弱于 ISAB

替代关系：

1. 可替代普通 MLP 通道头
2. 可作为 ISAB 不足时的轻量备选方案

#### 10.6.7 N7: Sinkhorn 软路由

原理：

1. 通过行列归一化把路由矩阵投影到近似双随机空间
2. 让多个专家/分支都能获得较均衡分配，而不是单一路径独占
3. 用连续可微软分配替代 Gumbel 的离散硬选择

框架适用性：

1. 适合替代现有动态路由或双分支选择层
2. 对小样本训练通常比 Gumbel 更稳
3. 可作为第二阶段结构替换验证

优点：

1. 训练稳定性更好
2. 可以缓解单分支坍塌和利用率失衡
3. 对路由可解释性较强

缺点：

1. 需要迭代归一化，推理/训练略增时延
2. 实现和调参成本高于纯 loss 改动
3. 如果分支本身质量不够，软路由也只能均衡分配失败

替代关系：

1. 替代 Gumbel-Softmax 硬路由
2. 可与 N3 配合，形成路由 + 梯度平衡组合

#### 10.6.8 N8: RRB / Gated Correction Head

原理：

1. 在 late-upsampling 后增加轻量残差修正块，对输出做细节补偿
2. 用门控机制决定哪些区域做强增强、哪些区域保持平滑
3. 目标不是重建整个图像，而是修正主干已生成 HR 输出中的局部误差

框架适用性：

1. 与当前 V3 的 stage/gated 逻辑高度兼容
2. 适合做最终输出端低成本增强
3. 是当前最容易落地且不破坏主干的方案之一

优点：

1. 参数量和计算开销都低
2. 对棋盘格伪影、边缘模糊、局部纹理不足有直接帮助
3. 与 stage/gated 主线一致，叙事连贯

缺点：

1. 如果补偿过强，可能导致过锐化或伪纹理
2. 主要改善输出端，不直接解决通道建模
3. 收益依赖主干输出质量，属于补强型模块

替代关系：

1. 替代无专门输出细化的简单后处理
2. 可作为 N1/N2 之后的最后一层补偿模块

### 10.7 建议落地顺序（保证可复现）

Phase A（低风险快收益）

1. N1 + N2：先做 loss 侧修正
2. N8：接入轻量细化头

目标：

1. 先确认低频通道是否稳定增益
2. 保持现有 V3 主干与训练脚本基本不动

Phase B（核心能力增强）

1. N4：统计初始化 + soft grouping
2. N5：ISAB 通道后处理块

目标：

1. 解决无序高维通道建模主问题
2. 在可控复杂度下获取增益上限

Phase C（结构稳定性增强）

1. N7：Sinkhorn 替换 Gumbel 路由
2. N3：GradNorm 与路由一起做稳定化

目标：

1. 降低训练波动
2. 提升分支利用率均衡性

### 10.8 组会决策建议

建议本周组会直接拍板以下三点：

1. 主线不改：继续以 V3 stage+gated 为锚点
2. 先做低风险高确定性改动：N1/N2/N8
3. 把 ISAB 与 Sinkhorn 作为第二阶段核心创新，做与旧方案的一一替代验证

预期里程碑：

1. 若 Phase A+B 顺利，整体可稳定冲击 15.0+ 区间
2. 若 Phase C 路由稳定性达标，可显著提升实验可复现性与组会说服力

### 10.9 组会讲稿版

1. 我们前面的结果已经说明，V3 里真正有效的不是更复杂，而是更稳。stage+gated 之所以能拿到 14.8053，是因为它把后处理强度控制住了，没有破坏主干已经学到的表征。
2. 之前失败的几个方向，本质上都在做同一件事：想用更复杂的路由或双分支去解决通道问题，但在 T=20 这种小样本条件下，Gumbel 路由容易塌缩，双分支容易失衡，最后反而把主干输出扰乱了。
3. 所以新方案不是继续堆结构，而是分三层替代：第一层改 loss，用 N1/N2/N3 给低频通道补梯度；第二层改通道建模，用 N4/N5/N6 解决无序高维特征交互；第三层改路由与输出，用 N7/N8 解决分支稳定性和最后一层细化。
4. 其中最适合先做的是 N1、N2、N8，因为它们不改主干、风险最低、最容易先看到低频通道提升。N4、N5 是第二阶段核心创新，真正解决无序高维通道怎么建模。N7、N3 则放在后面，专门处理动态路由和训练稳定性。
5. 换句话说，旧方案是在结构上硬拆，新方案是在损失上补、通道上聚、输出上修。目标不是把网络做得更花，而是把弱通道补回来，把路由做稳，把结果做成可复现。

如果要一句话收尾，可以说：

主线不改，继续以 V3 stage+gated 为锚点；先用 N1/N2/N8 做低风险增益，再用 N4/N5 解决无序高维通道建模，最后用 N7/N3 把路由和训练稳定性补齐。

### 10.10 参考文献（组会版，按主题整理）

A. 无序高维特征 / 置换不变性 / 通道建模

1. Lee et al., Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, ICML 2019.
2. Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition, arXiv 2021.
3. Permutation-Invariant Self-Attention, Emergent Mind topic summary.

B. 遥感波段分组 / 光谱结构 / 高光谱聚类

4. Hyperspectral Band Selection via Optimal Combination Strategy, Remote Sensing 2022.
5. Hyperspectral Band Selection via Adaptive Subspace Partition Strategy, 2019.
6. Interband Consistency-Driven Structural Subspace Clustering for Unsupervised Hyperspectral Band Selection, Sensors 2025.
7. Scalable Context-Preserving Model-Aware Deep Clustering for Hyperspectral Images, Remote Sensing 2025.

C. 损失函数 / 梯度平衡 / 回归不均衡

8. Ren et al., Balanced MSE for Imbalanced Visual Regression, CVPR 2022.
9. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, ICLR 2018.
10. Adaptive Loss Weighting for Machine Learning Interatomic Potentials, arXiv 2024.
11. Band-Sensitive Calibration of Low-Cost PM2.5 Sensors by LSTM Model with Dynamically Weighted Loss Function, Sustainability 2022.

D. 动态路由 / 稳定替代 / Soft routing

12. Gumbel-Softmax estimator / Vanilla Gumbel Softmax, 相关技术说明。
13. Sinkhorn-Knopp Algorithm / Sinkhorn-Knopp-Style Algorithm, Optimal Transport related references.
14. Hierarchical Routed Sinkformer, 2026 technical report / blog summary.
15. Differentiable Clustering Module / Learnable Clustering Module, Emergent Mind topics.

E. 后处理 / 细化模块 / SwinIR

16. Liang et al., SwinIR: Image Restoration Using Swin Transformer, ICCV Workshop 2021.
17. A Brief Analysis of the SwinIR Image Super-Resolution, IPOL 2022.
18. Lightweight Image Super-Resolution Reconstruction Network Based on Multi-Order Information Optimization, Sensors 2025.
19. R3Net: Recurrent Residual Refinement Network for Saliency Detection, IJCAI 2018.
20. Dual branch attention network for image super-resolution, PMC article.

F. 光谱一致性 / SAM / SID

21. Spectral information divergence for hyperspectral image analysis, NCKU publication.
22. Spectral Angle Mapper (SAM) related documentation and implementation notes.
23. SID-SAM hybrid method references in MATLAB / ENVI documentation.

G. 动态路由与双分支失效背景

24. The Stability Gap: Why Top-K Routing Breaks RL Optimization, technical blog.
25. A Super-Resolution Reconstruction Model for Remote Sensing Image Based on Generative Adversarial Networks, Remote Sensing 2024.
26. DBRSNet: a dual-branch remote sensing image segmentation model based on feature interaction and multi-scale feature fusion, PMC article.
