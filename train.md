# 多年份统一训练方案

这次的目标是把 2018 到 2024 的处理后数据一起送进训练，同时保留各年份自己的采样次数。根据 `scripts/filter_processed_by_exact_T.py` 的筛选逻辑，每一年都是按自己的 `target_t` 生成 filtered 数据集：

- 2018: T=20
- 2019: T=23
- 2020: T=19，当前先写进配置但注释掉
- 2021: T=21
- 2022: T=19
- 2023: T=22
- 2024: T=22

## 解决思路

原来的 `AnytimeTemporalDataset` 只接受一组 `lr_dir` 和 `hr_dir`，也就是一次只能读取一个 processed root。直接把不同年份的文件混到同一个目录里不合适，因为不同年份可能有相同的 `tile_id`，如果按 `tile_id` 分组时混在一起，会把不同年份的时间序列错误合并成一个更长的序列。

我新增了 `datapipe.datasets.MultiPathAnytimeTemporalDataset`。它接收多个数据根目录，每个根目录内部仍然用原来的 `AnytimeTemporalDataset` 单独建一个子数据集，然后在最外层把这些子数据集拼接起来。这样训练时 DataLoader 看到的是一个统一的大数据集，但每个样本仍然只来自某一个年份，2019 的 23 个时间点不会和 2024 的 22 个时间点混成同一个 tile 序列。

## 配置方式

新的默认训练配置是：

```bash
configs/ablation/ablation_4f_multiyear_2018_2024.yaml
```

核心字段如下：

```yaml
target: datapipe.datasets.MultiPathAnytimeTemporalDataset
params:
  dataset_roots:
    - ".../data/2018/processed_data_SR_10m_filer"
    - ".../data/2019/processed_data_SR_10m_filer"
    # - ".../data/2020-charles/processed_data_SR_10m_filer"
    - ".../data/2021/processed_data_SR_10m_filer"
    - ".../data/2022/processed_data_SR_10m_filer"
    - ".../data/2023/processed_data_SR_10m_filer"
    - ".../data/2024/processed_data_SR_10m_filer"
  split: train
  fixed_temporal_len: 23
```

`split` 会自动拼成每个年份下面的 `train/LR`、`train/HR`、`val/LR`、`val/HR` 或 `test/LR`、`test/HR`。`fixed_temporal_len: 23` 用 2019 的最大 T 作为统一长度：T 小于 23 的年份会在尾部补零，同时 `mask` 也补 0；模型做时间融合时可以通过 mask 忽略补出来的时间步。这样 batch 维度一致，不需要丢掉 2019 的 3 个时间点。

`sample_num: 0` 和 `pre_filter_sample_num: 0` 表示不再抽小样本，而是使用每个 filtered 年份下的全部 tile，构成真正的大训练集。

新配置里还把 `enable_hr_zero_filter` 设为 `false`。原因是这些目录已经由 `filter_processed_by_exact_T.py` 按每年的 exact T 筛过；如果训练时再启用旧的 pair-zero 过滤，某些 tile 的个别时间点会被二次删掉，实际 T 就不再等于 2019=23、2024=22、2021=21、2022=19、2023=22 这些 filtered 结果。

## train.sh 的变化

`train.sh` 的默认配置已经改成新的多年份配置：

```bash
bash train.sh 0
```

等价于：

```bash
bash train.sh 0 configs/ablation/ablation_4f_multiyear_2018_2024.yaml train
```

脚本默认使用当前机器上的环境：

```bash
/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/python
```

如果以后要临时换 Python，可以这样覆盖：

```bash
PYTHON_BIN=/path/to/python bash train.sh 0
```

## 2020 的处理

配置里已经预留了 2020：

```yaml
# - ".../data/2020-charles/processed_data_SR_10m_filer"
# - "2020_T19"
```

当前没有发现 `data/2020-charles/processed_data_SR_10m_filer` 这个 filtered root，所以先按要求注释掉。后续如果用 `scripts/filter_processed_by_exact_T.py --target-t 19` 生成 2020 的 filtered 数据，只需要取消这两行注释即可加入统一训练。
