# Cloud-Aware Super-Resolution for Generating Historical AlphaEarth-Like Imagery from Landsat
## Mid-Term Progress Report

Course: [Insert Course Name/Number]
Name: [Insert Name]
Date: [Insert Date]

## 1. Introduction and Project Goal
This project aims to reconstruct AlphaEarth-like 64-dimensional semantic embeddings at 10 m resolution from historical Landsat 8 observations (Brown et al., 2025; Google DeepMind, 2025). The task is inherently challenging because it is cross-sensor, cross-resolution, and cross-representation: a 9-band, 30 m Landsat time series must be mapped to a 64-channel, 10 m semantic target under irregular cloud cover and asynchronous acquisitions (Tang et al., 2026; Wang et al., 2023).

Our current approach builds on a SwinIR-based backbone with late upsampling, conditioned by (i) Day-of-Year temporal encoding and (ii) soft cloud probability masks (Liang et al., 2021; Pan, 2020; Zhu et al., 2015). We also explored a cloud cross-attention variant, but it is not adopted as the default model at this stage.

## 2. Data and Method Overview
### 2.1 Data and Study Area
We use Landsat 8 Path 132/Row 33 (Jinchang, Gansu, China) for 2018 as a smoke-test setting (Jiao et al., 2019; Roy et al., 2014). The area includes mountains, desert, and irrigated cropland, with approximately 35% mean cloud cover. To reduce spatial leakage, the central footprint is partitioned into geographically disjoint train/validation/test regions.

After preprocessing and QA filtering, the dataset contains 96,310 training tiles, 48,514 validation tiles, and 43,809 test tiles. Each tile has up to $T=20$ valid timestamps.

### 2.2 Method Summary and Figure Interpretation
The figure attached to this report shows the exact processing pipeline used in the experiments. The three inputs are Landsat time-series tiles, soft cloud masks derived from QA_PIXEL, and acquisition day-of-year (DOY). These inputs first pass through preprocessing and leakage-safe splitting, then enter the model as conditioning signals.

The backbone is a SwinIR-based encoder that operates on low-resolution features. Shallow feature extraction is followed by SwinIR blocks at LR, temporal fusion across the available timestamps, and PixelShuffle x3 reconstruction to produce a 64-channel, 10 m embedding tile. In the figure, the cloud cross-attention block is shown only as an exploratory ablation branch and is not part of the best configuration.

Time is encoded either as fixed sinusoidal DOY features or as a learnable-frequency variant in selected ablations. Soft cloud probabilities reduce the influence of heavily clouded pixels while still preserving partially valid observations. The SwinIR-based runs share the same reconstruction objective (L1/L2 + SSIM), and one ablation additionally tests a temporal consistency regularizer.

## 3. Experimental Setup and Results
We evaluate a single seven-configuration ablation family under the same smoke-test dataset, the same train/validation/test split, and the same training recipe. The fixed setting is Landsat Path 132/Row 33 in 2018, with Adam optimization, batch size 4, and seed 42. Across the family, the conditions that change are whether soft cloud masking is used, whether temporal encoding is fixed or learnable, and whether the exploratory cross-attention branch is included.

The comparison is designed to answer one question: under identical data and optimization conditions, do cloud-aware masking and temporal conditioning improve reconstruction over a strong SwinIR baseline?

For space, Table 1 lists the most informative reference settings from this family: the SRCNN reference, the SwinIR baseline, the soft-mask ablation, and the best cloud-aware + learnable-time setting. The full report includes the remaining ablations, but the key trend is already visible here.

Table 1 summarizes the key ablations.

| Model | Main Features | PSNR $\uparrow$ | SSIM $\uparrow$ | ERGAS $\downarrow$ |
|---|---|---:|---:|---:|
| SRCNN | none | 13.80 | 0.18 | 12.03 |
| SwinIR baseline | late upsampling only | 14.25 | 0.35 | 11.67 |
| + soft cloud mask (4c) | Prob-OR mask | 14.69 | 0.33 | 10.83 |
| + mask + learnable time (4f, best) | Prob-OR + learnable DOY | 14.76 | 0.39 | 10.74 |

The vanilla SwinIR baseline already outperforms SRCNN, confirming the benefit of a transformer-style backbone for this task. Adding soft cloud masking (4c) gives a +0.44 dB PSNR gain and lower ERGAS. The current best model (4f) combines soft masking with learnable temporal encoding and achieves +0.51 dB PSNR over the baseline. The exploratory cross-attention branch is included only as an ablation path and does not improve over the simpler temporal/mask-conditioned variants in this smoke-test setting.

## 4. Discussion, Next Steps, and Personal Contribution
Current status: the cloud-aware, time-conditioned SwinIR pipeline is fully implemented and converges stably. Ablation trends are consistent, with the strongest gains coming from soft masking and temporal encoding.

Next steps:
- Extend experiments to additional regions and years for spatial and temporal generalization.
- Compare with stronger SR baselines, including EDSR, RCAN, and Real-ESRGAN-style variants (Wang et al., 2021).
- Add a downstream evaluation (for example, land-cover or crop-type classification on reconstructed embeddings).
- Run multi-seed experiments and report mean $\pm$ standard deviation.
- Explore 64-band refinement through lightweight group-attention post-processing.

Personal contribution: I have mainly contributed to (i) implementing the QA_PIXEL-to-soft-mask pipeline, (ii) integrating DOY-based temporal conditioning in the SwinIR framework, and (iii) running and analyzing the key ablations (baseline, +mask, +time). In the next phase, I will focus on broader regional evaluation and downstream-task validation.

## References (Selected)
Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., Zhang, C., ... Kohli, P. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv. https://arxiv.org/abs/2507.22291

Google DeepMind. (2025). AlphaEarth Foundations helps map our planet in unprecedented detail. https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/

Jiao, M., Hu, M., & Xia, B. (2019). Spatiotemporal dynamic simulation of land-use and landscape-pattern in the Shiyang River basin, China. Science of the Total Environment, 506-507, 259-271.

Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image restoration using Swin Transformer. arXiv. https://arxiv.org/abs/2108.10257

Pan, H. (2020). Cloud removal for remote sensing imagery via spatial attention generative adversarial network. arXiv. https://arxiv.org/abs/2009.13015

Tang, K., Chen, X., Liu, T., Li, A., Tang, Y., Yang, P., & Chen, J. (2026). AnytimeFormer: Fusing irregular and asynchronous SAR-optical time series to reconstruct reflectance at any given time. Remote Sensing of Environment, 333, 115120.

Roy, D. P., Wulder, M. A., Loveland, T. R., Woodcock, C. E., Allen, R. G., Anderson, M. C., ... Zhu, Z. (2014). Landsat-8: Science and product vision for terrestrial global change research. Remote Sensing of Environment, 145, 154-172.

Shi, W., Caballero, J., Huszar, F., Totz, J., Aitken, A. P., Bishop, R., Rueckert, D., & Wang, Z. (2016). Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1874-1883).

Wang, C., Zhang, X., Yang, W., Wang, G., Zhao, Z., Liu, X., & Lu, B. (2023). Landsat-8 to Sentinel-2 satellite imagery super-resolution-based multiscale dilated transformer generative adversarial networks. Remote Sensing, 15(22), 5272.

Wang, Q., Shi, W., Atkinson, P. M., & Zhao, Y. (2016). Downscaling MODIS images with area-to-point regression kriging. Remote Sensing of Environment, 182, 115-128.

Wang, X., Xie, L., Dong, C., & Shan, Y. (2021). Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (pp. 1905-1914).

Zhu, Z., Wang, S., & Woodcock, C. E. (2015). Improvement and expansion of the Fmask algorithm: Cloud, cloud shadow, and snow detection for Landsats 4-7, 8, and Sentinel-2 images. Remote Sensing of Environment, 159, 269-277.

Zhu, Z., & Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all available Landsat data. Remote Sensing of Environment, 144, 152-171.
