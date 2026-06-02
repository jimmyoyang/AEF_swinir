# 2026-05-26 训练效率与显卡使用排查记录

## 当前判断

训练并不是完全跑在 CPU 上。代码路径中，模型会在 `build_model()` 里调用 `.cuda()`，每个 batch 也会在 `prepare_data()` 中把张量移动到 CUDA。此前的小规模前向、反向测试已经证明模型计算可以进入 H100 显卡。

真正慢的部分主要在数据读取和验证流程：

1. 数据集 `AnytimeTemporalDataset.__getitem__` 使用 rasterio、NumPy 和 CPU Tensor 读取 GeoTIFF，DataLoader 阶段天然在 CPU 和文件系统侧工作。
2. `train.sh` 原来每次训练前都会额外运行一次 CPU-only 的 split diagnostic，这会造成训练启动前 CPU/I/O 开销，现在改为只有设置 `RUN_SPLIT_DIAG=1` 时才运行。
3. 原训练在第 500 次迭代先进入验证，再保存普通 checkpoint。多年份验证集长度约 14081，`val_debug_max_samples` 只限制日志输出数量，并不限制实际验证样本数，所以看起来像在第 500 次迭代后卡住，而且不会及时留下 `model_500.pth`。

## 已修改内容

1. `train.sh`
   - 默认配置改为多年份训练配置。
   - 固定使用 alphaearth 环境的 Python。
   - CPU split diagnostic 改为可选：只有 `RUN_SPLIT_DIAG=1` 时才执行。

2. `trainer.py`
   - 普通 checkpoint 默认先保存再验证，避免长验证或验证卡住时丢失当前迭代模型。
   - 新增 `save_before_val` 配置开关，默认值为 `true`。
   - 新增 `val_max_batches` 和 `val_max_samples`，用于真正限制验证循环长度。
   - 新增 `train_timing_log_freq`，定期记录：
     - DataLoader 等待时间；
     - CPU 到 GPU 搬运时间；
     - 前向、反向和优化时间；
     - CUDA 显存已分配、已保留和峰值已分配。
   - 验证结束时若没有产生 PSNR，会安全跳过指标更新。

3. `configs/ablation/ablation_4f_multiyear_2018_2024.yaml`
   - 增加 `save_before_val: true`。
   - 增加 `val_max_batches: 64`，避免第 500 次迭代验证完整 14081 个样本。
   - 增加 `val_debug_max_samples: 8`、`val_compute_expensive_metrics: false`、`val_timing_log: true`。
   - 增加 `train_timing_log_freq: 50`，用于观察显卡和数据读取是否正常。

4. `configs/ablation/ablation_4f_multiyear_2018_2024_stability_2000.yaml`
   - 用于 2000 次迭代稳定性验证。
   - 保持 `val_freq: 2500`，先跑完 2000 次训练，不在中途触发完整验证。
   - 增加 `save_before_val: true`、`train_timing_log_freq: 50` 和验证上限配置，便于后续继续训练或单独验证时不再卡住。

## 后续验证计划

1. 先运行语法检查，确认 `trainer.py`、`train.sh` 和两个多年份配置没有明显错误。
2. 启动 `ablation_4f_multiyear_2018_2024_stability_2000.yaml`。
3. 同时采样 `nvidia-smi`，记录显存使用、GPU 利用率和功耗。
4. 只有在日志出现第 2000 次迭代、checkpoint 正常保存，并且监控记录显示显存和 GPU 利用率正常后，才认为本次目标完成。

## 运行记录

1. 已完成语法检查：
   - `trainer.py`、`trainer_mask_ablation.py`、`main.py` 编译通过。
   - `train.sh` 语法检查通过。
   - 两个多年份 YAML 配置均可由 OmegaConf 正常读取。
2. 当前显卡为 NVIDIA H100 80GB HBM3，启动训练前显存占用约 1 MB，GPU 利用率为 0%。
3. 已启动 2000 次迭代稳定性训练：
   - 训练进程 PID：6662。
   - 显卡监控进程 PID：6663。
   - 训练输出：`tmp/train_0526_stability.out`。
   - 显卡监控：`tmp/gpu_monitor_0526.csv`。
4. 第 50 次迭代已输出 timing 日志：
   - DataLoader 等待约 0.516 秒。
   - CPU 到 GPU 搬运约 0.001 秒。
   - 前向、反向和优化约 0.080 秒。
   - CUDA 当前已分配显存约 0.35 GB，保留显存约 6.58 GB，峰值已分配约 6.22 GB。
   - `nvidia-smi` 监控显示显卡总占用约 7.5 GB，说明模型和训练张量已经在显卡上；但 DataLoader 等待时间明显大于 GPU 计算时间，当前主要瓶颈仍然是 CPU/文件系统读取。
5. 第 100 次迭代：
   - 训练 loss 为 1.144226。
   - DataLoader 等待约 1.878 秒，前向、反向和优化约 0.081 秒。
   - `nvidia-smi` 捕捉到 GPU 利用率 58% 和 59% 的采样点，但大部分 10 秒采样仍为 0%，说明计算是短脉冲，整体受数据读取限制。
6. 由于用户要求自行测试 batch size、尽量利用 80GB 显存和提高利用率，已停止 batch=4 的小批量稳定性训练，准备进行 batch size / DataLoader worker 扫描后再重启 2000 次迭代训练。

## 继续修改

1. `trainer.py` 增加验证集按需构建逻辑：
   - 如果 `skip_val_if_not_reached=true` 且 `val_freq` 大于 `iterations`，训练启动时不再构建验证集。
   - 对当前 2000 次稳定性训练，`val_freq=2500`，因此不会提前构建 14081 个样本的 val set，减少启动时间和无效 CPU/I/O。
2. 多年份配置中将验证限制改为真正只跑 8 个样本：
   - `val_max_batches: 1`
   - `val_max_samples: 8`
   - `val_debug_max_samples: 8`
3. 本轮不运行 test mode；目标集中在训练跑满 2000 次迭代，并确认 GPU 显存和利用率正常。

## Batch size 扫描记录

1. 已停止第一轮 batch=4 的 2000 次稳定性训练：
   - 该训练已经证明模型可进入 GPU，且第 50、100、150 次迭代均能继续推进。
   - 但显存只稳定在约 7.5 GB，H100 80GB 没有被充分利用。
   - timing 日志显示主要瓶颈是 DataLoader 等待，而不是 CUDA 前向/反向。
2. 尝试使用 16 个 DataLoader workers 进行 batch size 扫描：
   - 进程能够构建完整 train dataset。
   - 但在 batch=8 的首个 CPU batch 阶段长时间无结果，GPU 保持空闲。
   - 判断该扫描方式不适合继续占用显卡，已停止相关进程。
3. 下一步改用确定性的显存扫描：
   - 使用真实数据集中读取到的样本构造 batch 张量。
   - 重点测量模型前向、反向、优化在不同 batch size 下的峰值显存和耗时。
   - 扫描完成后再把稳定 batch size 写入训练配置，并重新启动 2000 次迭代训练。
4. 使用 `num_workers=0` 的确定性扫描已开始产生结果：
   - batch=8：DataLoader 等待约 8.297 秒，CPU 到 GPU 约 0.002 秒，前向/反向/优化约 0.140 秒。
   - batch=8 峰值 CUDA 已分配显存约 11.89 GB，保留显存约 12.87 GB。
   - `nvidia-smi` 在该步捕捉到约 13.94 GB 显卡占用、71% GPU 利用率、约 234 W 功耗。
   - 继续扫描 batch=16、32、48、56、64，选择不 OOM 且留有安全余量的最大 batch 用于最终 2000 次训练。
5. 后续扫描结果：
   - batch=16：DataLoader 等待约 14.683 秒，前向/反向/优化约 0.265 秒，峰值 CUDA 已分配约 23.62 GB，保留约 25.01 GB；`nvidia-smi` 捕捉到约 26.37 GB 显卡占用、100% GPU 利用率、约 371.75 W 功耗。
   - batch=32：DataLoader 等待约 29.296 秒，前向/反向/优化约 0.490 秒，峰值 CUDA 已分配约 46.83 GB，保留约 49.53 GB；`nvidia-smi` 捕捉到约 51.49 GB 显卡占用、72% GPU 利用率、约 606.81 W 功耗。
   - 当前正在扫描 batch=48，这是优先候选值；如果稳定且不 OOM，会比 batch=32 更接近充分利用 80GB 显存。
6. batch size 扫描结论：
   - batch=48：DataLoader 等待约 46.448 秒，前向/反向/优化约 0.718 秒，峰值 CUDA 已分配约 70.21 GB，保留约 74.03 GB；`nvidia-smi` 捕捉到约 76.58 GB 显卡占用、55% GPU 利用率、约 596.60 W 功耗。
   - batch=56：发生 CUDA OOM，显卡总容量约 79.32 GiB，仅剩约 1.17 GiB 可用时还需要额外分配 1.38 GiB。
   - 因此最终训练 batch size 选择 48，这是当前最大稳定 batch，能较充分利用 80GB H100 显存，同时比 OOM 点保留一定安全余量。
7. 已将 `configs/ablation/ablation_4f_multiyear_2018_2024_stability_2000.yaml` 的训练 batch 改为 `[48, 8]`。

## batch=48 训练启动记录

1. 已重新运行语法检查，`trainer.py`、`trainer_mask_ablation.py`、`main.py`、`tmp/batch_sweep_0526.py` 均可编译。
2. 已确认稳定性配置：
   - `batch: [48, 8]`
   - `iterations: 2000`
   - `val_freq: 2500`
   - `skip_val_if_not_reached: true`
   - `val_max_samples: 8`
   - `num_workers: 4`
3. 启动 batch=48 的 2000 次迭代训练：
   - 训练进程 PID：7950。
   - 显卡监控进程 PID：7949。
   - 训练日志：`tmp/train_0526_batch48.out`。
   - 显卡监控日志：`tmp/gpu_monitor_0526_batch48.csv`。
4. 训练完成判据仍然不变：
   - 日志必须达到第 2000 次迭代。
   - 必须保存正常 checkpoint。
   - 显卡监控必须显示显存使用接近 batch=48 扫描结果，并且 GPU 利用率/功耗有正常计算峰值。
5. batch=48 真实训练启动后：
   - 验证集已按预期跳过，日志显示 `Dataset [val] skipped: val_freq=2500 is outside iterations=2000`。
   - GPU 显存从约 594 MB 上升到约 77184 MB，接近 batch=48 扫描时的 76578 MB。
   - 当前尚未到第 50 次迭代，因此还没有训练 timing 日志。
   - `nvidia-smi dmon` 在短窗口内主要看到 0% SM 利用率和约 120 W 功耗；这可能是大 batch 下 DataLoader 等待时间较长、GPU 计算是短脉冲导致，也可能说明实际训练吞吐过低。继续等第 50 次迭代 timing 后再决定是否降到 batch=32。
6. 第 50 次迭代已到达：
   - timing 日志：DataLoader 等待约 0.815 秒，CPU 到 GPU 约 0.013 秒，前向/反向/优化约 0.721 秒。
   - CUDA 当前已分配显存约 0.93 GB，保留显存约 74.62 GB，峰值已分配约 70.27 GB。
   - `nvidia-smi` 监控稳定显示显卡占用约 77184 MB，说明 batch=48 在真实训练中也接近使用完整 80GB H100。
   - 监控中已捕捉到 99% GPU 利用率、约 641.73 W 功耗，以及 20% GPU 利用率、约 181.70 W 功耗的计算采样点。
   - 当前判断 batch=48 可继续运行，不降到 batch=32。
7. 第 100 次迭代已到达：
   - 训练 loss 为 1.071990，学习率为 5.000e-05。
   - timing 日志：DataLoader 等待约 10.181 秒，CPU 到 GPU 约 0.013 秒，前向/反向/优化约 0.723 秒。
   - 显存仍稳定在约 77184 MB，CUDA 峰值已分配约 70.27 GB，保留约 74.62 GB。
   - 结论：显存利用已经接近 H100 80GB 上限，模型计算在 GPU 上正常；剩余主要瓶颈仍是 CPU/文件系统数据读取，导致 GPU 利用率呈现短脉冲。
8. 用户反馈“too slow”后停止 batch=48 训练：
   - 约 44 分钟后日志仍只到第 100 次迭代，未到第 500 次 checkpoint。
   - 显存利用正常但 SM 利用率长期异常偏低，不适合作为最终训练配置。
   - 已停止训练进程和显卡监控进程，释放 GPU。
   - 下一步对比 Git 中上一版本实现，定位当前低 GPU 利用率和数据等待过长的原因。

## Git 旧版本对比与原因定位

1. Git 当前 HEAD 中的旧配置 `configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml` 是单年份数据集：
   - `target: datapipe.datasets.AnytimeTemporalDataset`
   - train 使用 `pre_filter_sample_num: 4096` 和 `sample_num: 1024`
   - val 使用 `sample_num: 64`
   - `cache_dir` 指向单一路径，而不是多年份拼接。
2. 当前多年份配置改成：
   - `target: datapipe.datasets.MultiPathAnytimeTemporalDataset`
   - 6 个年份根目录拼接。
   - train/val 都设置 `sample_num: 0`，即使用全量样本。
   - `cache_root` 指向 `/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/multiyear_cache/ablation_4f/...`。
3. 实际检查缓存目录后发现：
   - 2018 的 `processed_data_SR_10m_filer/cache/train` 存在，约 18784 个缓存文件，大小约 8.0G。
   - 2018 的 `cache/val` 也存在，约 9316 个缓存文件。
   - 2019、2021、2022、2023、2024 的 `processed_data_SR_10m_filer/cache/train/val` 都不存在。
   - 当前配置指向的 `data/multiyear_cache/ablation_4f/train/tile` 不存在。
   - 当前多年份 hard/mask sidecar cache 基本为空，只有 HR cache 在逐步生成。
4. 结论：
   - 日志里显示 `Tile Cache: Active` 只代表配置允许使用缓存，不代表缓存文件真实命中。
   - 由于多年份 tile cache 根目录不存在，`_load_tile_cache()` 大量返回 `None`，`__getitem__` 回退到逐样本 rasterio 读取多个 GeoTIFF，并在 CPU 上生成 mask。
   - 因此 GPU 显存可以被 batch=48 填满，但 SM 利用率仍异常低；根因是 DataLoader/文件系统输入不足，不是模型没有放到 GPU。
   - 旧版本之所以相对可跑，是因为它是单年份、小样本、缓存路径更明确；当前多年份全量训练在未生成 tile cache 的情况下直接跑，会严重饿死 GPU。

## 修复方向

1. 已修改 `datapipe/datasets.py`：
   - 增加 `build_hard_mask_cache_on_miss`。
   - 增加 `build_mask_cache_on_miss`。
   - 训练时可以关闭 sidecar cache miss 后的即时构建，避免 batch 内重新读取大量原始 GeoTIFF。
2. 已修改 `configs/ablation/ablation_4f_multiyear_2018_2024_stability_2000.yaml`：
   - train 从全量改为每个年份 `sample_num: 1024`，总训练 tile 约 6144 个，接近旧版本“采样训练”的可控规模。
   - val 从全量改为每个年份 `sample_num: 8`，且稳定性训练中 `val_freq=2500`，2000 次训练内不会构建 val。
   - `cache_root` 改为每年真实的 `cache_dirs`：
     - 2018 已有 cache。
     - 2019、2021、2022、2023、2024 需要补建 cache。
   - 关闭训练期即时构建 hard/mask sidecar：
     - `build_hard_mask_cache_on_miss: false`
     - `build_mask_cache_on_miss: false`
3. 语法检查已通过：
   - `datapipe/datasets.py`
   - `trainer.py`
   - `trainer_mask_ablation.py`
   - `main.py`
   - `scripts/build_anytime_cache.py`
4. 下一步：
   - 使用 `scripts/build_anytime_cache.py` 为 2019、2021、2022、2023、2024 的 train split 各补建前 1024 个 tile cache。
   - cache 命中确认后再做短训练吞吐测试，确认 `data_wait` 是否明显下降。

## 继续排查：训练过慢后的缓存确认

1. 已确认当前 GPU 空闲：
   - `nvidia-smi` 显示 H100 80GB 当前仅约 1 MB 显存占用，GPU 利用率为 0%。
   - 没有残留的 `main.py`、`train.sh` 或显卡监控训练进程。
2. 已确认当前稳定性配置已经改为每年独立 cache 目录：
   - `2018/processed_data_SR_10m_filer/cache/train`
   - `2019/processed_data_SR_10m_filer/cache/train`
   - `2021/processed_data_SR_10m_filer/cache/train`
   - `2022/processed_data_SR_10m_filer/cache/train`
   - `2023/processed_data_SR_10m_filer/cache/train`
   - `2024/processed_data_SR_10m_filer/cache/train`
3. 实际文件数量检查结果：
   - 2018 train cache：18784 个文件，已存在。
   - 2019、2021、2022、2023、2024 train cache：目录缺失。
4. 结论：
   - 当前训练慢不是因为模型没有进入 GPU，而是多年份数据集中大部分年份没有 tile cache，DataLoader 在训练过程中回退到原始 GeoTIFF 读取和 CPU 预处理。
   - 这会造成显存看起来很高，但 GPU SM 利用率大部分时间为 0%，只在短暂前向/反向时出现计算峰值。
5. 下一步执行：
   - 为 2019、2021、2022、2023、2024 各构建前 1024 个 train tile cache。
   - 构建完成后先跑短训练吞吐测试，再决定继续使用 batch=48 还是改为 batch=32/更合适的吞吐配置。

## 缓存构建前确认

1. 已检查 2019、2021、2022、2023、2024 的 train split：
   - 每个年份 LR 与 HR 的 tile 总数一致。
   - 每个年份按排序取前 1024 个 tile 时，LR/HR tile ID 重合数都是 1024。
   - 因此用 `scripts/build_anytime_cache.py --max_tiles 1024` 构建的 LR tile cache 与当前 `sample_num: 1024` 的训练子集一致。
2. 已修改 `scripts/build_anytime_cache.py`：
   - 新增 `--skip_existing` 参数。
   - 如果某个 tile 的 `reflectance.npy`、`hard_valid_mask.npy`、`day_of_year.npy`、`file_names.txt` 四个文件已经完整存在，则跳过该 tile。
   - 这样缓存构建被中断后可以继续执行，不会重复计算已经完成的 tile。
3. 语法检查已通过：
   - `scripts/build_anytime_cache.py`
   - `datapipe/datasets.py`
   - `trainer.py`
   - `trainer_mask_ablation.py`
   - `main.py`

## 缓存构建尝试与策略调整

1. 已尝试并行补建 2019、2021、2022、2023、2024 的 train tile cache：
   - 每个年份启动 4 个 worker。
   - 日志分别写入 `tmp/cache_build_0526_<year>.out`。
2. 结果显示缓存构建本身也被底层文件系统/GeoTIFF 读取拖慢：
   - 约 1 分钟后，每个缺失年份只产生了 16 到 32 个 cache 文件，远低于预期。
   - 这说明直接补齐 5 个年份各 1024 个 tile cache 会很慢，不适合作为当前快速验证路径。
3. 已停止缓存构建相关进程，释放 CPU/IO 压力。
4. 当前更快的验证路径：
   - 先用已有完整 cache 的 2018 年数据单独跑短训练探针。
   - 如果 2018-only cached run 的 `data_wait` 明显降低且 GPU 利用率正常，就能直接证明训练慢的根因是多年份 cache miss，而不是模型没有进入 GPU。
   - 再基于这个结论决定是继续补全多年份 cache，还是先用已缓存年份/小批量年份做 2000 次稳定训练。

## cache 构建耗时估计

1. 当前需要补建的是 2019、2021、2022、2023、2024 五个年份的 train tile cache。
2. 目标规模：
   - 每年暂定构建 1024 个 tile。
   - 每个 tile 会写 4 个文件：`reflectance.npy`、`hard_valid_mask.npy`、`day_of_year.npy`、`file_names.txt`。
   - 五个年份合计约 5120 个 tile，也就是约 20480 个 cache 文件。
3. 实测速度：
   - 并行构建约 1 分钟后，每个年份只产生 16 到 32 个文件。
   - 折算约每年 4 到 8 个 tile，五个年份总计约 20 到 40 个 tile/分钟。
4. 粗略估计：
   - 只补建每年 1024 个 tile，预计约 2 到 5 小时。
   - 如果补建五个年份的全量 train cache，可能需要十几个小时甚至更久。
5. 判断：
   - cache 构建慢的原因仍然是底层文件系统读取大量 GeoTIFF 慢。
   - 为了提升 GPU 利用率，当前更优先的做法是让训练只使用已经存在 cache 的样本，避免训练中回退到原始 GeoTIFF 读取。

## GPU 利用率修复进展

1. 核心问题已经进一步明确：
   - batch=48 可以把 H100 80GB 显存推到约 76 GB。
   - 之前 GPU 利用率低，不是显存不足，也不是模型没有 `.cuda()`，而是 DataLoader 混入大量 cache miss 样本后回退到原始 GeoTIFF 读取，导致 GPU 等数据。
2. 已修改 `datapipe/datasets.py`：
   - 新增 `require_tile_cache`。
   - 新增 `filter_to_tile_cache`。
   - 当配置要求 tile cache 时，dataset 初始化阶段会过滤掉没有完整 tile cache 的 tile。
   - 如果运行中仍遇到 tile cache miss，会直接报错，避免静默回退到慢速 GeoTIFF 读取。
3. 已修改稳定性配置：
   - train/val 均设置 `require_tile_cache: true`。
   - train/val 均设置 `filter_to_tile_cache: true`。
   - train `num_workers` 从 4 调整为 8。
   - `prefetch_factor` 从 2 调整为 4。
4. cache 过滤后的 train 数据量：
   - 总计 1052 个 tile。
   - 2018：1024 个。
   - 2019：4 个。
   - 2021：8 个。
   - 2022：8 个。
   - 2023：4 个。
   - 2024：4 个。
5. 短 batch sweep 结果：
   - batch=32、workers=8：`data_wait=0.906s`，`forward/backward=0.491s`，峰值显存约 46.83GB，`nvidia-smi` 捕捉到 100% GPU 利用率。
   - batch=48、workers=8：`data_wait=0.507s`，`forward/backward=0.717s`，峰值显存约 70.21GB，`nvidia-smi` 显示约 76874MB 显存、72% GPU 利用率、617.88W。
6. 与旧的慢速训练对比：
   - 原 batch=48 在第 100 次迭代时 `data_wait=10.181s`，`forward/backward=0.723s`。
   - cache 过滤后 batch=48 的短测 `data_wait` 降到约 0.507s，已经从“GPU 长时间等数据”变成“数据等待与 GPU 计算同一量级”。
7. 下一步：
   - 使用当前 cache-gated 配置启动 2000 次迭代训练。
   - 同时记录 GPU monitor，确认实际长跑中是否能持续保持高显存占用和正常计算峰值。

## cache-gated 2000 次训练启动

1. 第一次启动失败：
   - 原因是 `train.sh` 当前不是可执行文件，直接 `./train.sh` 报 `Permission denied`。
   - 该次失败没有真正进入训练，也没有占用 GPU。
2. 已改用 `bash train.sh` 启动训练：
   - 训练日志：`tmp/train_0526_cachegated.out`
   - GPU 监控日志：`tmp/gpu_monitor_0526_cachegated.csv`
   - 训练 PID：10460
   - GPU monitor PID：10459
3. 当前训练配置：
   - batch size：48
   - `num_workers: 8`
   - `prefetch_factor: 4`
   - `require_tile_cache: true`
   - `filter_to_tile_cache: true`
   - `val_freq: 2500`，因此 2000 次稳定性训练内跳过验证集。
4. 实际启动后的数据集：
   - train 总长度：1052
   - 2018：1024
   - 2019：4
   - 2021：8
   - 2022：8
   - 2023：4
   - 2024：4
5. 实际训练第 50 次迭代结果：
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.718s`
   - `gpu_mem_reserved=74.62GB`
   - `gpu_mem_max_alloc=70.27GB`
6. GPU monitor 结果：
   - 显存稳定约 77204 MiB。
   - 多次采样显示 GPU 利用率为 99% 到 100%。
   - 功耗多次达到约 640W 到 665W。
7. 结论：
   - 当前 GPU 利用率异常的问题已经被 cache-gated 数据路径显著缓解。
   - 之前的慢速根因是 DataLoader 混入 cache miss 样本后回退到原始 GeoTIFF 读取，不是模型没有上 GPU。

## cache-gated 训练中期检查

1. 已到达第 100 次迭代：
   - loss：0.973650
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.719s`
2. 已到达第 200 次迭代：
   - loss：0.958675
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.721s`
3. 已到达第 400 次迭代：
   - loss：0.945645
   - 第 400 次出现一次 `data_wait=3.803s` 的短暂等待峰值。
   - 但 GPU monitor 同期仍大部分采样保持 99% 到 100% 利用率，显存约 77206 MiB。
4. 已到达第 500 次迭代：
   - loss：0.905034
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.722s`
   - 已保存 checkpoint：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/ckpts/model_500.pth`
5. 当前结论：
   - cache-gated 训练已经确认可以持续高 GPU 利用率运行，并且 checkpoint 保存正常。
   - 继续监控第 1000、1500、2000 次迭代和最终 checkpoint。

## cache-gated 训练 1000 次检查

1. 已到达第 800 次迭代：
   - loss：0.923076
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.722s`
2. 已到达第 900 次迭代：
   - loss：0.897358
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.723s`
3. 已到达第 1000 次迭代：
   - loss：0.899532
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.723s`
   - 已保存 checkpoint：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/ckpts/model_1000.pth`
4. GPU monitor：
   - 显存继续稳定在约 77208 MiB。
   - 多数采样为 99% 到 100% GPU 利用率。
5. 当前结论：
   - 训练已经稳定通过 1000 次迭代。
   - GPU 利用率异常问题在当前 cache-gated 配置下已经得到实测修复。
   - 继续监控第 1500 与第 2000 次 checkpoint。

## cache-gated 训练 1100 次实时检查

1. 当前训练仍在运行：
   - 主训练进程：`/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/python main.py --cfg_path configs/ablation/ablation_4f_multiyear_2018_2024_stability_2000.yaml --mode train`
   - 训练日志：`tmp/train_0526_cachegated.out`
   - GPU 监控日志：`tmp/gpu_monitor_0526_cachegated.csv`
2. 已到达第 1100 次迭代：
   - loss：0.896477
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.723s`
   - `gpu_mem_reserved=74.62GB`
   - `gpu_mem_max_alloc=70.27GB`
3. 当前 GPU 状态：
   - `nvidia-smi` 实时采样：H100 80GB，显存使用约 77208 MiB / 81559 MiB。
   - GPU 利用率实时采样为 99%。
   - GPU monitor 最近多次采样大部分为 99% 到 100%，中间少量 0% 采样属于迭代边界或日志/保存间隙。
4. cache 构建耗时回答：
   - 当前快速补建 2019、2021、2022、2023、2024 五个年份各前 1024 个 tile cache，按已观察速度估计约 2 到 5 小时。
   - 如果补建五个年份全量 train cache，预计会达到十几个小时甚至更久。
   - 当前优先目标是提升 GPU 利用率，因此先用 `filter_to_tile_cache` 和 `require_tile_cache` 避免训练时回退到原始 GeoTIFF 读取；完整多年份 cache 可以作为后续离线任务继续补齐。

## cache-gated 训练 1500 次检查

1. 已到达第 1500 次迭代：
   - loss：0.883134
   - `data_wait=0.000s`
   - `to_device=0.013s`
   - `forward_backward=0.722s`
   - `gpu_mem_reserved=74.62GB`
   - `gpu_mem_max_alloc=70.27GB`
2. 已保存 checkpoint：
   - `training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/ckpts/model_1500.pth`
3. 第 1450 次迭代出现一次短暂等待峰值：
   - `data_wait=3.512s`
   - 但第 1500 和第 1550 次迭代恢复为 `data_wait=0.000s`。
4. GPU monitor 结果：
   - 显存继续稳定约 77208 MiB。
   - 第 1500 checkpoint 附近多次采样为 99% 到 100% GPU 利用率。
5. 当前结论：
   - 已稳定通过 1500 次迭代。
   - 当前瓶颈不再是训练阶段大量等待数据，cache-gated 路径保持了高 GPU 利用率。

## cache-gated 训练最终检查

1. 训练已经完成 2000 次迭代并正常退出：
   - 日志中出现：`Training Finished Successfully!`
   - 训练曲线已生成：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/training_curves.png`
2. 后半段关键迭代：
   - 第 1800 次：loss=0.860557，`data_wait=0.000s`，`to_device=0.013s`，`forward_backward=0.721s`
   - 第 1900 次：loss=0.883152，`data_wait=0.000s`，`to_device=0.013s`，`forward_backward=0.723s`
   - 第 2000 次：loss=0.818109，`data_wait=0.000s`，`to_device=0.013s`，`forward_backward=0.725s`
   - 第 1850 次只有一次轻微等待：`data_wait=0.173s`
3. 已保存 checkpoint：
   - `model_500.pth`
   - `model_1000.pth`
   - `model_1500.pth`
   - `model_2000.pth`
   - 最终 checkpoint 路径：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/ckpts/model_2000.pth`
4. GPU 利用率证据：
   - 训练中显存稳定约 77208 MiB / 81559 MiB。
   - 有显存占用大于 70GB 的监控样本 911 个。
   - 非零利用率样本 745 个，非零样本平均 GPU 利用率约 98.2%。
   - GPU 利用率大于等于 90% 的样本 727 个。
   - 训练结束后显存释放到约 1 MiB，说明训练进程已退出并释放 GPU。
5. GPU monitor 已停止：
   - 已停止 `nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 2`。
   - 当前没有残留的 `main.py` 或 `train.sh` 训练进程。
6. 最终结论：
   - 模型和 batch tensor 确认在 GPU 上运行。
   - 之前 GPU 利用率为 0、显存低的核心原因不是模型跑在 CPU，而是多年份训练混入大量 tile cache miss，DataLoader 回退到原始 GeoTIFF 读取和 CPU 预处理，导致 GPU 等数据。
   - 当前通过 `require_tile_cache: true` 和 `filter_to_tile_cache: true` 强制使用已有 tile cache，训练阶段不再静默走慢速 GeoTIFF fallback。
   - 2000 次稳定性训练已经证明 cache-gated 路径可以把 H100 显存和计算利用起来。
   - 后续如果要恢复更完整的 2018 到 2024 多年份覆盖，需要离线补齐缺失年份的 tile cache；快速补建每个缺失年份前 1024 个 tile 约 2 到 5 小时，全量补齐预计十几个小时或更久。

## 目标完成复核

1. 已重新检查当前日志和进程状态：
   - `tmp/train_0526_cachegated.out` 明确记录第 2000 次迭代完成。
   - 日志末尾包含 `Training Finished Successfully!`。
   - 当前没有残留的 `main.py` 或 `train.sh` 训练进程。
2. 产物确认：
   - 最终 checkpoint 存在：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/ckpts/model_2000.pth`
   - 训练曲线存在：`training_logs/experiments/ablation_4f_multiyear_2018_2024_stability_2000/2026-05-26_07-19-53/training_curves.png`
3. GPU 状态复核：
   - 训练期间显存大于 70GB 的监控样本共 911 个。
   - 非零 GPU 利用率样本平均利用率约 98.2%。
   - 利用率大于等于 90% 的样本共 727 个。
   - 训练结束后 `nvidia-smi` 显示显存回到约 1 MiB，说明 GPU 已释放。
4. 目标状态：
   - 2000 次迭代已经成功完成。
   - 训练期间显存使用和 GPU 利用率均正常。
   - 本次监控目标已完成。

## 全量 train/val cache 构建启动

1. 默认训练配置 `configs/ablation/ablation_4f_multiyear_2018_2024.yaml` 已调整为可直接用于 `bash train.sh 0`：
   - cache 路径从空的 `data/multiyear_cache/ablation_4f/...` 改为每个年份自己的 `data/<year>/processed_data_SR_10m_filer/cache/<split>`。
   - train/val/test 均启用 `require_tile_cache: true` 和 `filter_to_tile_cache: true`，避免训练时静默回退到原始 GeoTIFF 读取。
   - train batch 调整为 `[48, 8]`。
   - `num_workers: 8`，`val_num_workers: 2`，`prefetch_factor: 4`，沿用 2000 次稳定训练中验证过的 H100 配置。
2. cache 工具改动：
   - `scripts/build_anytime_cache.py` 的 `.npy` 写入改为原子写入。
   - `--skip_existing` 现在会尝试读取 `.npy` 文件头，避免把损坏或中断写入的 cache 当成有效 cache。
   - 新增 `scripts/build_multiyear_anytime_cache.py`，使用一个共享进程池统一构建多年份、多 split cache，避免同时开多个独立进程池造成 IO 竞争。
3. 当前 train/val 全量 cache 计划：
   - 2018 train/val 已完整命中，跳过。
   - 2019、2021、2022、2023、2024 的 train/val 合计需要构建 35224 个 tile cache。
4. worker 数选择：
   - 小样本 benchmark 显示该任务受 GeoTIFF/文件系统 IO 限制，而不是 CPU 算力限制。
   - 24 tile 测试：2 workers 约 21 秒，4 workers 约 33 秒，8 workers 约 50 秒，16 workers 约 56 秒。
   - 48 tile 复测：2 workers 约 34 秒，4 workers 约 51 秒。
   - 因此当前全量构建使用 `--num_workers 2`。这是当前机器和共享文件系统下更快的选择；盲目提高 worker 会增加 IO 竞争并降低吞吐。
5. 已启动全量 train/val cache 后台任务：
   - PID：`12157`
   - 日志：`tmp/full_cache_train_val_0526.out`
   - PID 文件：`tmp/full_cache_train_val_0526.pid`
   - 命令：`python scripts/build_multiyear_anytime_cache.py --splits train,val --skip_existing --num_workers 2 --valid_mask_band_count 9 --reflectance_band_count 9`
6. 当前运行状态：
   - 进度条显示约 1.7 到 1.8 tile/s。
   - ETA 约 5.5 到 6 小时。
   - 当前正在构建 2019 train cache，tile 数已经从 4 增加到 70+。
   - GPU 当前空闲；cache 构建主要使用 CPU/IO，不占用 GPU。

## 全量 cache 构建运行复查

1. 运行约 6 分钟后复查：
   - 主进程 PID：`12157`
   - worker 进程数：2
   - 两个 worker 均在运行，单个 worker CPU 约 8%。
2. 当前吞吐：
   - 进度约 499 / 35224。
   - tqdm 估计剩余时间约 5.5 到 5.8 小时。
   - 2019 train cache 已增加到约 504 个 tile。
3. CPU 使用说明：
   - 机器有 128 个 CPU 线程，但 benchmark 显示该任务不是 CPU 算力瓶颈，而是共享文件系统/GeoTIFF 读取瓶颈。
   - 继续增加 worker 会让多个进程争抢同一存储路径，导致吞吐下降。
   - 因此当前 2 worker 是实测更高效的配置，而不是未使用 CPU 的配置错误。
4. GPU 状态：
   - `nvidia-smi` 显示 GPU 显存约 1 MiB，利用率 0%。
   - cache 构建不需要 GPU，当前保留 GPU 空闲，避免和后续训练互相干扰。

## 3000 次 full-cache 训练目标更新

1. 新目标：
   - 继续监督 train/val 全量 cache 构建。
   - cache 完成后执行默认命令：`bash train.sh 0`。
   - 监督 full-cache 训练直到 3000 次迭代完成，并检查显存占用和 GPU 利用率。
2. 当前 cache 状态：
   - cache 构建进程仍在运行：PID `12157`。
   - worker 进程数：2。
   - 当前正在构建 2019 train cache。
   - 2019 train cache 已从初始 4 个 tile 增加到约 956 个 tile。
   - 其他缺失年份和 val split 仍未完成，因此现在不能启动 full-cache 训练。
3. 默认训练配置已调整：
   - 文件：`configs/ablation/ablation_4f_multiyear_2018_2024.yaml`
   - `save_dir` 改为：`./training_logs/experiments/ablation_4f_multiyear_2018_2024_fullcache_3000`
   - `iterations` 改为：`3000`
   - 保持 `batch: [48, 8]`、`num_workers: 8`、`prefetch_factor: 4`。
   - 保持 `require_tile_cache: true` 和 `filter_to_tile_cache: true`，防止训练阶段回退到慢速 GeoTIFF 路径。
4. 当前判断：
   - 继续等待 cache 全量完成。
   - cache 完成前启动训练会导致数据集只使用已缓存子集，不符合 full-cache 训练目标。

## 配置文件不再修改

1. 用户要求：不要继续修改训练配置文件。
2. 已撤回刚才对 `configs/ablation/ablation_4f_multiyear_2018_2024.yaml` 的两处修改：
   - `save_dir` 已恢复为：`./training_logs/experiments/ablation_4f_multiyear_2018_2024`
   - `iterations` 已恢复为：`20000`
3. 后续执行方式：
   - 不再通过修改配置文件把训练强行改成 3000 次。
   - cache 完成后仍按用户要求执行 `bash train.sh 0`。
   - 监督该训练运行到第 3000 次迭代，并以第 3000 次迭代的日志、checkpoint、GPU 显存和 GPU 利用率作为阶段目标完成依据。
4. 当前 cache 进度：
   - 构建进程仍在运行，PID：`12157`。
   - 2019 train cache 已增加到约 1310 个 tile。
   - 其余年份和 val split 仍在等待构建。

## 20000 次训练与 3000 次监督边界

1. 用户确认：
   - 配置保持 `iterations: 20000`。
   - 训练仍按完整 20000 次迭代运行。
   - 本次监督目标只需要确认训练稳定达到第 3000 次迭代。
2. 执行策略：
   - 不修改现有配置文件。
   - cache 完成后执行 `bash train.sh 0`。
   - 监督到第 3000 次迭代，记录 loss、checkpoint、GPU 显存、GPU 利用率和 data wait 情况。
   - 第 3000 次达到后，本次监督目标可判定完成，但训练进程保持继续运行到配置中的 20000 次，除非用户另外要求停止。

## cache 构建 15 分钟复查

1. 配置确认：
   - `configs/ablation/ablation_4f_multiyear_2018_2024.yaml` 保持 `iterations: 20000`。
   - `batch: [48, 8]`、`num_workers: 8`、`prefetch_factor: 4` 保持不变。
2. cache 进程状态：
   - 主进程 PID：`12157`
   - worker 进程：2 个，仍在运行。
   - 当前进度约 `1466 / 35224`。
3. 当前 cache 覆盖：
   - 2018 train：4696/4696，已完成。
   - 2018 val：2329/2329，已完成。
   - 2019 train：约 1476/4749。
   - 2019 val 以及 2021、2022、2023、2024 的 train/val 仍待构建。
4. 当前结论：
   - cache 构建没有中断。
   - 尚未达到启动 full-cache 训练的条件。

## cache 构建 23 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `2259 / 35224`。
   - 2019 train cache 约 `2270 / 4749`。
3. 其他 split：
   - 2018 train/val 已完整。
   - 2019 val 以及 2021、2022、2023、2024 的 train/val 仍未开始或仍待完成。
4. GPU 状态：
   - cache 阶段 GPU 仍保持空闲，显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 构建稳定进行中。
   - 继续等待 train/val 全量 cache 完成后再执行 `bash train.sh 0`。

## cache 构建 34 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `3337 / 35224`，约 9%。
   - 2019 train cache 约 `3348 / 4749`。
3. 当前覆盖情况：
   - 2018 train/val 已完整。
   - 2019 train 仍有约 1401 个 tile 未完成。
   - 2019 val、2021 train/val、2022 train/val、2023 train/val、2024 train/val 仍待构建。
4. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
5. 当前结论：
   - cache 构建正常，没有失败迹象。
   - 当前不能启动训练，否则不是 full train/val cache 条件下的训练。

## cache 构建 50 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程均在运行。
2. 当前进度：
   - tqdm 约 `4952 / 35224`，约 14%。
   - 2019 train cache 已完成：`4749 / 4749`。
   - 2019 val cache 已开始：约 `215 / 2387`。
3. 当前未完成部分：
   - 2019 val 仍有约 2172 个 tile。
   - 2021、2022、2023、2024 的 train/val 仍待构建。
4. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
5. 当前结论：
   - cache 构建已经完成第一个缺失年份的 train split。
   - 继续等待全部 train/val cache 完成后再启动 `bash train.sh 0`。

## cache 构建 81 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `8211 / 35224`，约 23%。
   - 2019 train 已完成：`4749 / 4749`。
   - 2019 val 已完成：`2387 / 2387`。
   - 2021 train 已开始：约 `1097 / 4691`。
3. 当前未完成部分：
   - 2021 train 仍有约 3594 个 tile。
   - 2021 val、2022 train/val、2023 train/val、2024 train/val 仍待构建。
4. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
5. 当前结论：
   - cache 构建稳定推进，2019 年 train/val 已完整。
   - 继续等待所有 train/val cache 完成。

## cache 构建 112 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `11574 / 35224`，约 33%。
   - 2021 train cache 约 `4461 / 4691`，仅剩约 230 个 tile。
3. 已完成部分：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
4. 未完成部分：
   - 2021 train 即将完成。
   - 2021 val、2022 train/val、2023 train/val、2024 train/val 仍待构建。
5. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
6. 当前结论：
   - cache 构建稳定，预计还需要数小时。

## cache 构建 143 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `15140 / 35224`，约 43%。
   - 2021 train 已完成：`4691 / 4691`。
   - 2021 val 已完成：`2347 / 2347`。
   - 2022 train 已开始：约 `1003 / 4725`。
3. 已完成部分：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
   - 2021 train/val 完整。
4. 未完成部分：
   - 2022 train/val、2023 train/val、2024 train/val。
5. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
6. 当前结论：
   - cache 构建稳定推进，已完成 3 个年份的 train/val 覆盖。

## cache 构建 174 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `18965 / 35224`，约 54%。
   - 2022 train 已完成：`4725 / 4725`。
   - 2022 val 已开始：约 `103 / 2370`。
3. 已完成部分：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
   - 2021 train/val 完整。
   - 2022 train 完整。
4. 未完成部分：
   - 2022 val、2023 train/val、2024 train/val。
5. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
6. 当前结论：
   - cache 构建已经过半，继续等待剩余 split 完成。

## 恢复训练曲线和 VAL 可视化

1. 用户反馈：
   - 训练曲线可视化和训练过程中的 VAL 数据集可视化似乎消失。
   - 同时要求不要修改配置文件。
2. 排查结果：
   - `trainer.py` 中原有 `plot_curves()` 和 `visualize_validation_sample()` 函数仍存在。
   - 但当前多年份配置没有设置 `local_logging`，旧逻辑会导致 VAL 可视化不执行。
   - 训练曲线旧逻辑只在完整训练结束时生成；当前默认训练是 20000 iter，因此训练过程中看不到实时更新的 `training_curves.png`。
3. 已恢复的代码行为：
   - rank 0 进程始终创建 `images/val` 目录。
   - 如果配置中没有显式设置 `local_logging` / `visualize_validation` / `visualize_val`，默认启用 VAL 可视化。
   - 如果配置中显式设置 `local_logging: false`，仍尊重该显式关闭。
   - 每次 validation 的第一个 batch 默认保存 VAL 可视化图，默认保存 1 个样本，可通过 `val_visualize_max_samples` 或 `val_vis_max_samples` 控制。
   - 训练过程中按 `plot_curve_freq` / `curve_freq` 生成曲线；如果没有设置这些字段，则默认按 `val_freq` 更新，若没有 val 则按 `save_freq` 更新。
   - 训练结束时仍保留最终 `training_curves.png` 生成逻辑。
4. 可视化函数健壮性修复：
   - 支持当前 `lr_sequence` 的 `(B,T,C,H,W)` 形状。
   - 基础反射率波段优先使用 `reflectance_band_count`，兼容旧的 `num_lr_bands`。
   - 修正 error map 的通道平均维度。
   - 保存前确保 `images/val` 目录存在。
5. 验证：
   - 已执行 `python -m py_compile trainer.py trainer_mask_ablation.py`，语法检查通过。
   - 已用 CPU 假数据 smoke test：
     - 成功生成 `tmp/trainer_vis_smoke/images/val/iter_123_sample_0.png`
     - 成功生成 `tmp/trainer_vis_smoke/training_curves.png`
6. 当前 cache 状态：
   - cache 构建进程仍在运行，未因本次代码修改中断。

## cache 构建 202 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `22336 / 35224`，约 63%。
   - 2023 train cache 已开始：约 `1109 / 4693`。
3. 已完成部分：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
   - 2021 train/val 完整。
   - 2022 train/val 完整。
4. 未完成部分：
   - 2023 train/val。
   - 2024 train/val。
5. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
6. 当前结论：
   - cache 构建继续正常推进。
   - 尚未达到启动 `bash train.sh 0` 的条件。

## cache 构建 233 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `25631 / 35224`，约 73%。
   - 2023 train cache 约 `4405 / 4693`，剩余约 288 个 tile。
3. 已完成部分：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
   - 2021 train/val 完整。
   - 2022 train/val 完整。
4. 未完成部分：
   - 2023 train 即将完成。
   - 2023 val、2024 train/val 仍待构建。
5. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
6. 当前结论：
   - cache 构建稳定接近后半段完成。

## 恢复训练曲线与 VAL 可视化补充

1. 本次检查确认：
   - `trainer.py` 已恢复训练过程中的 `training_curves.png` 生成逻辑。
   - `trainer.py` 已恢复 validation 阶段保存 VAL 样本可视化图的逻辑。
   - 输出位置为实验目录下的 `training_curves.png` 和 `images/val/iter_<iter>_sample_<idx>.png`。
2. 本次补充修改：
   - 在 `trainer.py` 中显式设置 Matplotlib 使用 `Agg` 后端。
   - 这样在无图形界面的服务器训练时，也能稳定保存曲线图和 VAL 可视化 PNG，不依赖 DISPLAY。
3. 行为说明：
   - 不修改任何配置文件。
   - 如果配置没有显式设置 `local_logging` / `visualize_validation` / `visualize_val`，默认启用 VAL 可视化。
   - 如果配置显式设置关闭可视化，则仍尊重配置。
   - 曲线默认按 `plot_curve_freq` / `curve_freq` 更新；未配置时按 `val_freq` 更新，没有 val 时按 `save_freq` 更新。
4. 验证：
   - 使用训练脚本同一环境 `/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/python` 执行 `python -m py_compile trainer.py`，通过。
   - 执行 CPU smoke test，成功生成：
     - `tmp/trainer_vis_smoke_current/images/val/iter_123_sample_0.png`
     - `tmp/trainer_vis_smoke_current/training_curves.png`

## cache 构建 268 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `29467 / 35224`，约 84%。
   - 剩余约 5757 个 tile。
3. GPU 状态：
   - cache 阶段 GPU 仍空闲，显存约 1 MiB，利用率 0%。
4. 当前结论：
   - cache 仍在正常推进。
   - 尚未启动 `bash train.sh 0`，因为完整 cache 尚未完成。

## cache 构建 270 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `29618 / 35224`，约 84%。
3. cache 覆盖情况：
   - 2018 train/val 完整。
   - 2019 train/val 完整。
   - 2021 train/val 完整。
   - 2022 train/val 完整。
   - 2023 train/val 完整。
   - 2024 train：`1374 / 4642`，缺少 `3268`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 剩余 cache 约 5583 个 tile。
   - 继续等待 cache 完成，完成后再启动 `bash train.sh 0`。

## cache 构建 273 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `29986 / 35224`，约 85%。
3. cache 覆盖情况：
   - 2024 train：`1722 / 4642`，缺少 `2920`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - cache 仍在正常推进。
   - 剩余约 5235 个 tile，完成前不启动训练。

## cache 构建 279 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `30621 / 35224`，约 87%。
3. cache 覆盖情况：
   - 2024 train：`2357 / 4642`，缺少 `2285`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 仍处于 cache 构建阶段，GPU 空闲正常。
   - 继续等待 2024 train 和 2024 val cache 完成。

## cache 构建 284 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `31243 / 35224`，约 89%。
3. cache 覆盖情况：
   - 2024 train：`2981 / 4642`，缺少 `1661`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 2024 train 继续推进，尚未切换到 2024 val。
   - 完整 cache 未完成前不启动训练。

## cache 构建 290 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `31871 / 35224`，约 90%。
3. cache 覆盖情况：
   - 2024 train：`3607 / 4642`，缺少 `1035`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 2024 train 接近完成，但完整 cache 仍缺少 3350 个 tile。
   - 继续等待 cache 完成后再启动训练。

## cache 构建 296 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `32481 / 35224`，约 92%。
3. cache 覆盖情况：
   - 2024 train：`4216 / 4642`，缺少 `426`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 2024 train 即将完成。
   - 之后还需要等待 2024 val cache 完成，再启动 `bash train.sh 0`。

## cache 构建 299 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `32880 / 35224`，约 93%。
3. cache 覆盖情况：
   - 2024 train：`4616 / 4642`，缺少 `26`。
   - 2024 val：`0 / 2315`，缺少 `2315`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 2024 train 基本完成，等待切换到 2024 val。

## cache 构建 301 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `33117 / 35224`，约 94%。
3. cache 覆盖情况：
   - 2024 train：`4642 / 4642`，缺少 `0`。
   - 2024 val：`212 / 2315`，缺少 `2103`。
4. 当前结论：
   - 2024 train cache 已完整。
   - 当前已经进入最后的 2024 val cache 构建阶段。

## cache 构建 312 分钟复查

1. 进程状态：
   - 主进程 PID：`12157`，仍在运行。
   - 2 个 worker 进程仍在运行。
2. 当前进度：
   - tqdm 约 `34297 / 35224`，约 97%。
3. 全量 cache 覆盖情况：
   - 2018 train/val：缺少 `0`。
   - 2019 train/val：缺少 `0`。
   - 2021 train/val：缺少 `0`。
   - 2022 train/val：缺少 `0`。
   - 2023 train/val：缺少 `0`。
   - 2024 train：缺少 `0`。
   - 2024 val：`1412 / 2315`，缺少 `903`。
4. GPU 状态：
   - H100 显存约 1 MiB，利用率 0%。
5. 当前结论：
   - 仅剩最后的 2024 val cache 未完成。
   - 完成后执行全量覆盖复查，再启动 `bash train.sh 0`。

## cache 完成与训练启动前检查

1. cache 进程状态：
   - PID `12157` 已退出。
2. cache 日志结果：
   - `build-multiyear-cache: 100%|...| 35224/35224`
   - `[cache] finished elapsed_sec=19128.3 failures=0`
3. 全量 cache 覆盖复查：
   - 2018 train/val：缺少 `0`。
   - 2019 train/val：缺少 `0`。
   - 2021 train/val：缺少 `0`。
   - 2022 train/val：缺少 `0`。
   - 2023 train/val：缺少 `0`。
   - 2024 train/val：缺少 `0`。
4. GPU 状态：
   - 启动训练前 H100 显存约 1 MiB，利用率 0%。
5. 下一步：
   - 启动 `bash train.sh 0`。
   - 同步启动 `nvidia-smi` 监控，记录 GPU 显存、利用率和功耗。

## 训练启动记录

1. 已启动命令：
   - `bash train.sh 0`
2. 进程信息：
   - `train.sh` PID：`16009`
   - `main.py` PID：`16082`
   - GPU 监控 PID：`16010`
3. 实际训练配置：
   - 日志显示运行 `configs/ablation/ablation_4f_multiyear_2018_2024.yaml`。
   - `iterations: 20000`。
   - `batch: [48, 8]`。
   - `num_workers: 8`。
   - `val_num_workers: 2`。
   - `val_max_samples: 8`。
4. 初始 GPU 状态：
   - 当前仍处于数据集/模型初始化阶段。
   - GPU 显存约 4 MiB，利用率 0%，属于初始化阶段的正常状态。
5. 下一步：
   - 等待进入正式训练迭代。
   - 重点监控 GPU 显存、利用率、iter 日志、checkpoint、训练曲线和 VAL 可视化文件。

## 训练进入 GPU 计算阶段

1. 进程状态：
   - `train.sh` PID：`16009`，仍在运行。
   - `main.py` PID：`16082`，仍在运行。
   - GPU 监控 PID：`16010`，仍在运行。
2. 数据集状态：
   - train dataset：`28196`。
   - val dataset：`14081`。
   - 动态检测 `in_chans: 11`。
3. 训练状态：
   - 已生成 baseline VAL 可视化。
   - 已到达 `Iter 100`。
   - Iter 100 loss：`1.106594`。
4. GPU 状态：
   - 显存约 `77248 MiB / 81559 MiB`。
   - GPU 利用率约 `98% - 100%`。
   - 功耗约 `650W - 670W`。
5. 性能判断：
   - `data_wait=0.000s`，`to_device≈0.013s`，`forward_backward≈0.72s`。
   - 当前已确认模型和 batch 正常在 H100 上运行，GPU 利用率已恢复正常。

## 训练 Iter 500 复查

1. 训练状态：
   - 已到达 `Iter 500`。
   - Iter 500 loss：`1.055778`。
   - 学习率：`5.000e-05`。
2. checkpoint：
   - 已保存 `training_logs/experiments/ablation_4f_multiyear_2018_2024/2026-05-26_13-39-38/ckpts/model_500.pth`。
   - 当前 best checkpoint 已导出到 `best_ckpts/ablation_4f_multiyear_2018_2024/model.pth`。
3. validation：
   - validation 被限制为 `batches=1/1761`。
   - 实际使用 `val_max_batches=1` 和 `val_max_samples=8`。
   - Iter 500 PSNR：`12.3254`。
   - Masked-PSNR：`12.3371`。
4. 可视化：
   - 已生成 `images/val/iter_500_sample_0.png`。
   - 已更新 `training_curves.png`。
5. GPU 状态：
   - 训练阶段显存约 `77248 MiB`。
   - GPU 利用率多数采样为 `99% - 100%`。
   - checkpoint / validation 阶段有短暂利用率下降，随后恢复。
6. 当前结论：
   - 训练、checkpoint、VAL 可视化、训练曲线和 GPU 利用率均正常。
   - 继续监控到 Iter 3000，不停止 20000 iter 训练。

## 训练 Iter 1000 复查

1. 训练状态：
   - 已到达 `Iter 1000`。
   - Iter 1000 loss：`1.038759`。
   - 学习率：`5.000e-05`。
2. checkpoint：
   - 已保存 `training_logs/experiments/ablation_4f_multiyear_2018_2024/2026-05-26_13-39-38/ckpts/model_1000.pth`。
   - 当前 best checkpoint 已更新。
3. validation：
   - validation 仍限制为 `batches=1/1761`。
   - Iter 1000 PSNR：`12.4564`。
   - Masked-PSNR：`12.4461`。
4. 可视化：
   - 已生成 `images/val/iter_1000_sample_0.png`。
   - 已更新 `training_curves.png`。
5. GPU 状态：
   - 显存稳定约 `77252 MiB`。
   - GPU 利用率多数采样为 `99% - 100%`。
6. 当前结论：
   - 训练已超过 1400 iter，仍稳定运行。
   - 继续监控到 Iter 3000。

## 训练 Iter 2000 复查

1. 训练状态：
   - 已到达 `Iter 2000`。
   - Iter 2000 loss：`0.998202`。
   - 学习率：`5.000e-05`。
2. checkpoint：
   - 已保存 `model_1500.pth`。
   - 已保存 `model_2000.pth`。
   - 当前 best checkpoint 已更新。
3. validation：
   - Iter 1500 PSNR：`12.5149`。
   - Iter 2000 PSNR：`12.5684`。
   - Iter 2000 Masked-PSNR：`12.5691`。
4. 可视化：
   - 已生成 `images/val/iter_1500_sample_0.png`。
   - 已生成 `images/val/iter_2000_sample_0.png`。
   - `training_curves.png` 已在 Iter 2000 后更新。
5. GPU 状态：
   - 显存稳定约 `77252 MiB`。
   - GPU 利用率多数采样为 `99% - 100%`。
6. 当前结论：
   - 当前训练已超过 2300 iter，未见错误或 OOM。
   - 继续监控到 Iter 3000，达到后不停止 20000 iter 训练。

## 训练 Iter 3000 目标完成复查

1. 训练状态：
   - 日志已到达 `Iter 3000`。
   - Iter 3000 loss：`0.971252`。
   - 学习率：`5.000e-05`。
   - 复查时训练已经继续超过 `Iter 3300`，说明没有在 3000 停止。
2. checkpoint：
   - 已保存 `training_logs/experiments/ablation_4f_multiyear_2018_2024/2026-05-26_13-39-38/ckpts/model_3000.pth`。
   - 文件大小约 `179M`。
3. validation：
   - Iter 3000 validation 仍按要求限制为 `batches=1/1761`，即 8 个样本。
   - Iter 3000 PSNR：`12.5399`。
   - Iter 3000 Masked-PSNR：`12.5399`。
   - validation timing：`total=0.56s`。
4. 可视化：
   - 已生成 `images/val/iter_3000_sample_0.png`。
   - 已更新 `training_curves.png`。
5. GPU 状态：
   - Iter 3000 前后显存稳定约 `77252 MiB / 81559 MiB`。
   - GPU 利用率多数采样为 `99% - 100%`。
   - 只在 checkpoint / validation 阶段出现短暂下降，随后恢复正常。
6. 监控状态：
   - 按用户要求只监控到 3000 iter。
   - GPU 监控进程已停止。
   - `bash train.sh 0` 和 `main.py` 训练进程仍继续运行，保持 20000 iter 训练目标。
7. 当前结论：
   - cache 已完整。
   - 训练已稳定跑过 3000 iter。
   - batch 和模型确认在 H100 上运行，GPU 利用率和显存占用符合 80GB 卡的预期。
   - 本轮目标已完成。
