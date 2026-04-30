# 4-14 周报（详版：数据准备 LR 归一化、离线 cache、消融配置）

## 1. 一句话总结

本周工程侧围绕 **磁盘上 LR 的写入归一化**（`prepare_data_local.py`）、**Anytime 离线 cache 构建**（`scripts/build_anytime_cache.py`）以及 **消融配置中 train/val 的 cache 开关与路径**（`ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml`）三块落地；需特别注意：**当前 4f 配置为 soft mask + advanced processor 时，Dataset 代码路径实际上不会使用 tile cache**，与是否填写 `cache_dir` 无矛盾，属于「配置已就绪、主干仍为在线读 tif」状态。

cache是因圈梁数据做不动，后续查看发现在cpu上的操作（cpu利用率大于80%）
训练 2:1:1
---

## 2. 本周实验指标（定量）

以下数据均来自本仓库 `training_logs/` 下**当前磁盘上的日志与对比表**（统计截止：周报撰写时；若后续重跑，数值以最新 `training.log` / CSV 为准）。

### 2.1 四组对比实验（`run_comparison_experiments.py` → `comparison_results.csv`）

| 实验 | sample num | Final PSNR (dB) | **Best PSNR (dB)** | Final SSIM | Best SSIM（注） |
|------|------|-----------------|-------------------|------------|----------------|
| Exp1 | 2 | 14.1578 | **14.3224** | 0.2914 | 1.0 |
| Exp2 | 4 | 14.1655 | **14.2752** | 0.3162 | 1.0 |
| Exp3 | 8 | 14.1830 | **14.3951** | 0.3188 | 1.0 |
| Exp4 | 32 | Baseline + Temporal + Advanced Cloud Mask | N/A | N/A | 1.0 | 1.0 |

**解读（本周可写进汇报的结论）**：

1. 在 **Exp1–Exp3 已完成且 CSV 有有效 PSNR** 的前提下，**Best PSNR 最高为 Exp3 的 14.3951 dB**（优于 Exp1 的 14.3224、Exp2 的 14.2752），与「时相 + 简单云掩膜」设计方向一致。  
2. **Exp4** 在表中为 **N/A**，表示该次汇总时 **Exp4 未产出有效 PSNR**（未完成训练、日志解析失败或运行中断等），周报中应如实写「Exp4 待补跑或待修复对比脚本」，**不宜**把 Exp4 与前三组做数值排名。  
3. CSV 中 **Best SSIM 列为 1.0** 与验证日志常见取值不符，**更可信的是 Final SSIM**（Exp3 约 **0.3188**）；若对外汇报 SSIM，建议以各实验 `training.log` 中 `📊 Validation Metrics` 行为准，或修复 `run_comparison_experiments.py` 对 SSIM 的解析后再更新本表。

### 2.2 消融：`ablation_4f_mask_prob_or_learnable_pos_no_cross`（完整 20k iter）

**运行目录**：`training_logs/experiments/ablation_4f_mask_prob_or_learnable_pos_no_cross/2026-04-13_02-17-41/`  
**状态**：训练正常结束（`Training Finished Successfully`）。

| 指标 | 数值 | 说明 |
|------|------|------|
| **Best PSNR（验证集）** | **11.7219 dB** | 出现于 **Iter 500** 后第一次验证；此后验证 PSNR 未再超过该值，`best_ckpts/.../model.pth` 对应该轮次。 |
| 同轮 SSIM | 0.0659 | 与 Best PSNR 同一行验证日志。 |
| 同轮 Masked-PSNR | 11.7289 | 与 indicating_mask 口径一致时的 PSNR。 |
| **Iter 20000 末次验证** | PSNR **11.4859** dB，SSIM **0.0571**，ERGAS **16.4237**，SAM **0.5337**，Masked-PSNR **11.4832** | 训练损失继续下降但验证 PSNR 相对 early best **略回落**，需在下周报中讨论是否早停、学习率或数据/掩膜强度。 |

### 2.3 主配置 anytime 单次跑（对照参考）

**运行目录**：`training_logs/anytime_swinir/2026-04-12_13-20-30/`（节选）

| 指标 | 数值 |
|------|------|
| 首次验证 **Best PSNR** | **12.1954 dB**（同轮 SSIM **0.0745**，Masked-PSNR **12.4230**） |

说明：该 run 与 4f 消融 **配置与数据管线不完全相同**，**不宜与 2.1 表格直接横向比绝对数值**；仅作为「本周 anytime_swinir 目录下曾有一次验证峰值约 12.2 dB」的备忘。

### 2.4 验证可视化（定性，待改进）

当前训练过程中落盘的 **验证集可视化**（如 `.../images/val/iter_*_sample_0.png`）主观观感 **偏差**：所选样本受 **云遮挡** 等因素影响，**LR 与 HR 在有效观测上差别较大**，拼图不利于向非技术同事展示「模型是否在变好」。**数值指标仍以 `training.log` 中整集验证为准**；可视化侧的 **样本挑选 / 多格展示 / 优先低云或高有效像素 tile** 等，**留待后续修改**（与 Dataset 过滤策略或 `visualize_validation_sample` 的索引策略相关）。

### 2.5 训练可收敛 vs 验证/测试泛化、模型容量与数据（观测与后续方向）

**现象（定性）**：在当前 **小训练子集可稳定跑完** 的前提下，**训练集侧 loss 能持续下降**，但 **独立测试/推广场景下主观与部分定量反馈仍偏弱**；与验证集指标一起看，更符合「**拟合了子集分布，但对更难或更广分布未同步变好**」，而非单纯「完全训不动」。

**验证集 PSNR 轨迹（有日志支撑，4f 同一次 run：`2026-04-13_02-17-41/training.log`）**：整集验证 PSNR 在 **极早** 达到全局最高后 **整体下台阶**，中后期 **未回到峰值**，仅在 **约 11.48–11.52 dB** 区间震荡略走弱；与 **Train Loss 单调下降** 形成对照（见同 log 中 `📈 Iter … | Train Loss` 与 `📊 Validation Metrics` 交替出现）。

| 验证点（紧随其前的 Iter） | 整集 Val PSNR (dB) | 备注 |
|---------------------------|-------------------|------|
| 500 后 | **11.7219** | 全局最高（`New best metric`） |
| 1000 后 | 11.5150 | 已低于峰值约 **0.21 dB** |
| 2000 后 | **11.4796** | 本阶段较低点 |
| 5000 后 | 11.4937 | 未恢复至 11.72 |
| 10000 后 | 11.4913 | 仍低于峰值约 **0.23 dB** |
| 20000 后（末次） | **11.4859** | 相对峰值 **约 -0.24 dB**，相对 2000 步附近低点略回升但仍远低于首验 |

**Train Loss 对照（同 log，节选）**：Iter **500** 时 Train Loss **0.496617**；Iter **20000** 时 Train Loss **0.168557**。即 **训练损失明显下降**，而 **验证 PSNR 未随迭代同步上升**，可支撑汇报中「**存在过拟合或 train/val(test) 分布不一致、或优化目标与 PSNR 对齐不足**」等讨论（具体主因需结合梯度与样本再拆）。

**模型容量（存疑）**：当前 **SwinIR 规模相对「多时相 + 9→64 波段 + 大空间上下文」任务是否偏小**，能否在 **全区域、全分布** 数据上同时完成 **稳定训练与高质量推理重建**，团队内 **尚存疑**；后续建议与 **更大容量 backbone / 更强时序融合** 或 **数据分层与子集训练** 做对照，再下结论。

**测试集与数据管线（后续）**：在 **test 路径与配置完全对齐** 后应补 **独立 test 指标**；并考虑 **测试集重构**，例如：**按地物类型分层**、**精简为高质量且分布更一致子集**、与 train/val **声明同一预处理与筛选规则**，避免「训练在小而干净子集、评测在杂而难子集」导致的 **表观测试很差**。

---

## 3. 修改一：`datapipe/prepare_data_local.py` 中 LR patch 物理还原与归一化（约 140–147 行）

### 3.1 代码在流水线中的位置

在 `create_one_tile` 内，执行顺序大致为：

1. 从 Landsat / AlphaEarth 读取 **原始数值** 的 `lr_tile`、`hr_tile`。
2. 通过 `patch_passes_planning_thresholds`（与 STAGE 1 掩膜一致）筛掉明显无效块。
3. 在 **仍未做 LR 逐通道归一化** 的数组上，统计 `lr_min/lr_max`、`lr_zero_ratio`、`hr_zero_ratio` 等，用于 `pair_zero_ratio` 判定与 **`patch_stats.log` 记录**（与训练 manifest 解析口径一致：manifest 反映的是 **写盘前、归一化前** 的零占比等）。
4. **通过 pair-zero 阈值后**，再执行附件中的 LR 归一化块（140–147 行），最后把 **已归一化后的** `lr_tile` 与未在此处改写的 `hr_tile` 写入 GeoTIFF。

### 3.2 归一化在做什么（逐通道 min–max 到约 [0,1]）

对应实现要点：

1. `lr_tile` 转为 `float32`。
2. 仅处理前 `n_lr_bands = min(10, C)` 个通道（与 Landsat 常见多光谱通道数对齐；超过 10 的通道若存在则 **原样保留** 在数组中但本循环不处理）。
3. 对每个通道 `i`：
   - `lr_tile[i] -= nanmin(lr_tile[i])`：按通道去下限，消除通道内整体偏移。
   - `bmax = nanmax(lr_tile[i])`，若 `bmax > 1e-8` 则除以 `bmax`，否则除以 `1.0`：避免除零，全常数通道会落在「全零」输出。
4. 该步骤 **不是** Dataset 里的 `robust_per_image_normalize`（1%/99% 分位拉伸到 `[-1,1]`），而是 **数据准备阶段** 为磁盘产品选择的 **简单 min–max 到 [0,1] 量级** 的物理拉伸，目的是让落盘的 LR 动态范围更稳定、便于后续与 HR 联合训练。

### 3.3 与训练侧的关系与注意点

1. **训练时** `AnytimeTemporalDataset` 对从磁盘读出的 LR 仍会再做 `robust_per_image_normalize`（或带物理归一化的变体），因此：**磁盘 LR 的 min–max 与训练时分位数归一化是两级不同操作**，周报需写清，避免同事误以为「prepare 已等价于 Dataset 归一化」。
2. `patch_stats.log` 与 pair-zero 使用的是 **归一化前** 统计的 zero 比例，与 `_load_hr_zero_ratio_manifest` 的设计一致；若将来改为「只对归一化后 LR 记 log」，需同步改 manifest 解析与训练阈值语义。
3. **HR 在 `create_one_tile` 中未做与 LR 对称的 min–max**；HR 仍以读取 dtype/数值直接写盘，训练侧对 HR 使用 `robust_per_image_normalize`，口径与 4-1 周报中「HR 在 Dataset 归一化」的叙述一致。

---

## 4. 修改二：`scripts/build_anytime_cache.py`（Anytime 离线 cache）

### 4.1 目标与输入输出

1. **目标**：为每个空间 `tile_id`（由 LR 文件名中 `tile_{i}_{j}.tif` 解析）预计算并落盘，使 `AnytimeTemporalDataset._load_tile_cache` 能一次性加载整 tile 的多时相张量，减少训练时反复 `rasterio.open` 与重复归一化的开销。
2. **输入**：某 split 下的 LR 目录（例如 `.../train/LR` 或 `.../val/LR`），脚本用 `group_lr_files_by_tile` 将 `*_tile_*.tif` 按 `tile_id` 分组并 **排序**（保证时间维顺序稳定）。
3. **输出**（每个 `tile_id` 一组，与 Dataset 约定文件名一致）：
   - `tile_{tile_id}_reflectance.npy`：各时相 LR 经 `robust_per_image_normalize` 后的堆叠，存为 `float16` 以省空间，训练加载时再转 `float32`。
   - `tile_{tile_id}_hard_valid_mask.npy`：与 Dataset 非缓存路径一致的硬掩膜，`np.all(arr > 0, axis=0)` 逐像素判定。
   - `tile_{tile_id}_day_of_year.npy`：由文件名日期 `YYYYMMDD` 解析的年内日（与 Dataset 用 `datetime` 解析应一致，前提为同一命名规范）。
   - `tile_{tile_id}_file_names.txt`：每行一个 LR 文件名，与上述数组的 `T` 维严格对齐。

### 4.2 运行方式与并行

1. 命令行入口：`--lr_dir`（必填）、`--out_dir`（必填）、`--max_tiles`（0 表示全量）、`--num_workers`（`<=1` 串行，否则 `ProcessPoolExecutor` 并行）。
2. `build_one_tile_cache` 在子进程中执行，需能从项目根导入 `datapipe.datasets.robust_per_image_normalize`（脚本内已 `sys.path.insert` 项目根）。

### 4.3 与 `prepare_data_local` 的关系

1. **Cache 构建读的是已落盘的 GeoTIFF**；若 LR 在 prepare 阶段已做 140–147 的 min–max，则 cache 中的「反射率」实为 **对磁盘 LR 再做分位数归一化** 的结果，与「直接从 raw 建 cache」在数值上不同。工程上应固定一条链路：**先确定 prepare 是否写归一化 LR，再决定 cache 是否需重建**，避免混用旧 cache 与新数据。

---

## 5. 修改三：`configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml` 中 train/val 的 cache 配置

### 5.1 配置内容（当前文件）

1. **train**（示例）：
   - `cache_dir: ".../processed_data_SR_10m_32samples/cache/train"`
   - `use_tile_cache: true`
2. **val**（用户标出的 51–52 行）：
   - `cache_dir: ".../processed_data_SR_10m_32samples/cache/val"`
   - `use_tile_cache: true`

含义：**为 train/val 分别指定独立的 cache 根目录**，与 `lr_dir`/`hr_dir` 的 split 一一对应，避免 train 与 val 的 `tile_*` 文件同名冲突。

### 5.2 与 `AnytimeTemporalDataset` 实际行为的对照（必读）

`datapipe/datasets.py` 中 `can_use_cache` 条件为（逻辑与）：

1. `use_tile_cache` 且 `cache_dir` 非空；
2. `mask_type == 'hard'`；
3. **未**启用 `use_advanced_processor`；
4. **未**配置外部 `cloud_mask_dir`。

而 **ablation_4f** 在 `features.mask_band` 下配置了：

1. `use_advanced_processor: true`
2. `processor.mask_type: "soft"`

因此在 **不改动 features 配置** 的前提下，**即使 `use_tile_cache: true` 且已生成 `.npy`，训练仍不会走 `_load_tile_cache` 分支**，而是继续按 tif 在线读取并走软掩膜 + CloudMaskProcessor。周报结论：

1. 当前 yaml 中的 `cache_dir` / `use_tile_cache` 属于 **基础设施预留** 或 **为后续切 hard / 关 advanced 的子实验准备**；
2. 若希望 **立刻** 从 cache 获益，需要另存一版配置：例如临时将 `mask_type` 改为 `hard` 且 `use_advanced_processor: false`（并确认无 `cloud_mask_dir`），再对比吞吐与指标。

### 5.3 操作建议（与 build 脚本衔接）

1. 对 train：`python scripts/build_anytime_cache.py --lr_dir <train/LR> --out_dir .../cache/train [--num_workers N]`。
2. 对 val：同上，`--out_dir` 指向 `.../cache/val`。
3. 构建完成后可用 `scripts/check_cache_integrity.py`（若仓库内已存在）或人工检查每个 `tile_id` 四个文件是否齐全。

---

## 5.4（新增）Cache 操作与修改规则（重点：怎么做）

本节把「cache 到底存了什么」「如何正确开启/关闭」「什么时候必须重建」「为什么某些配置下看起来开了 cache 但实际上不会命中」写成可执行的规则与步骤，避免后续重复踩坑。

### 5.4.1 cache 里到底有什么（“存的不是原始 tif”，而是预处理后的数组）

Anytime 的 tile cache 是按 **tile_id** 预先把“多时相序列”算好并落盘。对每个 `tile_id`，cache_dir 下会有 **4 个文件**（缺一不可）：

1. `tile_{tile_id}_reflectance.npy`
   - 形状：`(T, C, H, W)`
   - 含义：对每个时相的 LR 反射率做 `robust_per_image_normalize` 后的结果（值域期望约 `[-1, 1]`），脚本写盘用 `float16` 节省空间。
2. `tile_{tile_id}_hard_valid_mask.npy`
   - 形状：`(T, H, W)`
   - 含义：硬掩膜 `np.all(arr > 0, axis=0)` 的堆叠（取值范围 `[0,1]`），用于判断有效像素与 indicating_mask（在 hard mask 路径下）。
3. `tile_{tile_id}_day_of_year.npy`
   - 形状：`(T,)`
   - 含义：由文件名 `YYYYMMDD_...` 解析得到的年内日（DOY），用来构建 timestamps/time_band。
4. `tile_{tile_id}_file_names.txt`
   - 行数：`T`
   - 含义：每个时相对应的 LR 文件名（仅文件名，不含目录），必须与上述 3 个数组的时间维严格对齐。

结论：**cache 保存的是“训练时会反复计算”的中间结果**（归一化后的 reflectance、硬掩膜、DOY、文件名列表），不是原始 tif 本体。

### 5.4.2 什么时候会命中 cache（真值表 / 必要条件）

即便配置里写了 `use_tile_cache: true` 和 `cache_dir: ...`，Dataset 也不一定会用 cache；`AnytimeTemporalDataset` 的 cache 命中条件（必须同时满足）是：

1. `use_tile_cache == true` 且 `cache_dir` 非空；
2. `mask_type == 'hard'`；
3. `use_advanced_processor == false`；
4. 没有外部 `cloud_mask_dir`（为 None）。

因此：

- **soft mask / advanced processor / 外部 cloud mask** 的任何一种启用，都会使数据管线走“在线读 tif + 在线算 mask/特征”的分支；此时 cache_dir 可以保留（为后续 hard/no-advanced 的实验准备），但训练吞吐不会因为 cache 改善。
- 如果你希望“配置含义更直观”，可以在这些实验里显式把 `use_tile_cache: false`（不是必须，但可以减少误解）。

### 5.4.3 正确构建 cache（一步一步照做）

**步骤 0：先确认 split 的 LR/HR 路径**

- 训练/验证的 `lr_dir` 与 `hr_dir` 必须对应同一个 split（train 对 train，val 对 val）。
- 代码默认用 “同名文件匹配” 找 HR：`target_hr_path = hr_dir / lr_files[0].name`。因此要求 **HR 目录下必须存在与 LR 文件同名的 HR 文件**。

**步骤 1：分别为 train/val 建 cache（不要混在一个目录）**

示例（伪路径，按你本机实际目录替换）：

1. train：
   - `python scripts/build_anytime_cache.py --lr_dir <.../train/LR> --out_dir <.../cache/train> --num_workers 8`
2. val：
   - `python scripts/build_anytime_cache.py --lr_dir <.../val/LR> --out_dir <.../cache/val> --num_workers 8`

强规则：**train 与 val 必须用不同 out_dir**。理由很简单：tile_id 会复用（同一编号在不同 split 都可能存在），混在一起会互相覆盖或污染。

**步骤 2：在配置里写回 cache_dir，并打开 use_tile_cache**

- train：`cache_dir: <.../cache/train>`，`use_tile_cache: true`
- val：`cache_dir: <.../cache/val>`，`use_tile_cache: true`

**步骤 3：完整性检查（建议每次构建后做一次）**

仓库已有脚本可做完整性与 spot check：

- `python scripts/check_cache_integrity.py --cfg_path <你的配置yaml> --phase train`
- `python scripts/check_cache_integrity.py --cfg_path <你的配置yaml> --phase val`

如果你是手动检查，最低限度要保证每个 tile_id 的 4 个文件都存在，且：

- reflectance 为 4D，mask 为 3D，doy 为 1D
- 三者的 `T` 维一致，且与 file_names.txt 行数一致

### 5.4.4 什么时候必须重建 cache（最常见踩坑点）

只要下面任一项发生变化，就按“需要重建”处理（至少对受影响的 split 重建）：

1. **LR 磁盘内容变化**：例如 `prepare_data_local.py` 改了 LR 的写盘归一化方式、换了 processed_data 目录、重新生成了 LR/HR tif。
2. **LR 文件集合变化**：比如换了 `sample_num`、过滤规则、重新划分 split，导致每个 tile_id 下包含的时相文件发生变化。
3. **文件名规则变化**：因为 DOY 与 tile_id 都来自文件名解析（日期与 `tile_{i}_{j}`）。

不需要重建的情况（但仍需确认命中条件）：

- 仅修改模型结构/损失/训练超参，而数据目录与文件未变。

### 5.4.5 为什么评估脚本要强制 need_path / 为什么要显式指定 lr_dir/hr_dir

1. `need_path: true` 的作用：让 dataset 返回 `sample['path']`，评估脚本会把它写进 per-sample CSV，便于定位“到底是哪张图异常”。
2. 显式指定 `lr_dir/hr_dir` 的作用：
   - cache 只存 file_names（文件名），运行时仍要用 `lr_dir / fname` 与 `hr_dir / fname` 重新定位真实文件。
   - 若路径没对齐（例如 val 用了 train 的 hr_dir），会出现“能读 LR，但找不到/读错 HR”的隐性错误，导致指标不可信。
3. debug 单张图时的路径规则：
   - `PreprocessedTileDataset` 的 debug 模式给一个 `debug_lr_path`，会按目录结构自动推断 HR：`.../LR/<name>.tif` → `.../HR/<name>.tif`。
   - 因此 debug 时更要确保 LR/HR 的目录结构与文件同名匹配。

### 5.4.6（补充）num_heads=1 / cross_num_heads=1 的修改规则（为什么有时必须改）

这条规则主要与 **交叉注意力 CloudCrossAttention** 的实现约束有关：模块内部要求 `internal_dim % num_heads == 0`。

因此当你修改了以下任意项时：

- `embed_dim`
- `downsample_rate`
- `cross_num_heads` / `num_heads`

就必须保证：

- `internal_dim = embed_dim // downsample_rate` 能被 `num_heads` 整除

若不满足会直接触发断言/报错。将 `num_heads=1`（或 `cross_num_heads=1`）是一个“永远能整除”的兜底改法，用于：

1. **快速跑通/定位其他问题**（先让模型能 forward）；
2. **显存/算力受限** 时临时降复杂度；

但需要强调：

- heads 改动会改变注意力层结构，通常 **无法直接加载旧 checkpoint 的对应权重**（shape mismatch 属正常现象）。
- 若目标是“继续复现/继续训练某个已有 ckpt”，heads 数必须与当时训练时一致。

---

## 6. 其它本周相关进展（简列）

1. **训练收敛与 NaN（工程侧防护）**：`trainer_mask_ablation` 对全零掩膜与 non-finite loss 的跳过逻辑；`robust_per_image_normalize` 对 NaN 波段加固；`test_lr_tile_values.py` 辅助扫目录。
2. **小样本 vs 扩大样本规模时的训练现象（观测，根因未定位）**：
   - **可收敛的设置**：在 LR/HR 经正确归一化与过滤的前提下，采用 **16 个 train sample、8 个 val、8 个 test**（即常见写法里的 **16:8:8**）时，训练能够**正常收敛**，与当前 4f 等实验日志中 `Dataset [train] size: 16`、`Dataset [val] size: 8` 的设定一致。
   - **异常现象**：将数据子集规模提高到 **32 个 sample 及以上**（例如把 `sample_num` 调到 32+）时，训练过程中会出现 **NaN**（损失或梯度 non-finite）；**根因尚未系统排查**（是否与更大子集中个别 tile 的极端值、分位归一化分母过小、掩膜全零比例、学习率或 AMP 等有关，留作后续专项）。
3. **test 集路径**：主配置中 test 与 train/val 根目录仍可能不一致，评估时需单独核对（详见上周报或本仓库 `configs/config_swinir.yaml`）。
4. **验证可视化样本**：当前默认保存的 val 图（如 `sample_0`）对应场景 **云遮挡较重**，LR 与 HR 对齐观感差，**不代表整集指标**；后期需改样本选择或展示策略（见 §2.4、§8.5）。
5. **泛化与测试侧**：训练子集上 **loss 可收敛**，但 **验证 PSNR 自早期峰值后整体未再提升**（见 §2.5 表）；**测试集表现仍偏弱**，需在路径对齐后补 **test 定量**；长期看 **test 集按地物/质量分层与精简**、以及 **模型容量对照实验** 为必要工作（§2.5、§8.6–8.7）。

---

## 7. Git 更新摘要（便于对齐代码版本）

1. **Commit**：`e9fb38a`（示例标题：`stabilize data filtering and training NaN safeguards`）
2. 与本周「数据 + 训练稳定性」相关的改动可在该提交及前后若干 commit 中追溯；推送远端需具备对应 GitHub 账号权限。

---

## 8. 下周计划（与本周三块修改强相关）


2. **cache 与 prepare 一致性**：任一调整 `prepare_data_local` 140–147 归一化策略后，**全量重建** `cache/train` 与 `cache/val`。
3. **文档化**：在 `ablation_4f` README 或本周报附录中增加「`can_use_cache` 真值表」，减少后续同学重复踩坑。
4. **32+ sample 训练 NaN**：在固定随机种子与相同配置下，对比 16 vs 32 子集的首个 NaN 迭代、该 batch 的 LR/HR/mask 统计与 `robust_per_image_normalize` 分母，必要时对问题 tile 做单步前向与梯度检查。
5. **验证可视化改进**：在 `trainer` / `visualize_validation_sample` 或数据侧增加「低云 / 高有效像素 / 固定若干 showcase tile」策略，避免长期只画 `sample_0` 等高云难例，便于汇报与定性对照。
6. **测试集与评测协议**：对齐 `configs` 中 train/val/**test** 根路径；在固定 checkpoint 上输出 **test PSNR/SSIM/SAM** 与分地物或分云量分层表；按需构建 **高质量、分布一致** 的精简 test 子集，与 val 趋势（§2.5）对照解读。
7. **模型容量消融**：在数据子集固定前提下，对比 **当前 SwinIR 宽度/深度** 与 **放大 embed_dim/depths**（或更强时序头）的 val/test，验证「容量不足」假设是否成立。
