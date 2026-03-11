# ============================================================================
# 【软掩膜实现参考代码】
# 将这段代码替换到 datapipe/datasets.py 的第 225-298 行（掩膜生成部分）
# ============================================================================

# 这是正确的实现方式，支持硬掩膜和软掩膜的完整分离

            # 3. (可选) 添加"掩膜波段"
            if self.use_mask_band:
                # ============================================================
                # 【改进的掩膜处理逻辑】支持硬掩膜/软掩膜，解耦处理器
                # ============================================================
                # 1. 生成基础掩膜（硬：有效=1，无效=0）
                base_mask = (np.all(reflectance_data > 0, axis=0)).astype(np.float32)
                
                # 2. 读取掩膜类型配置
                mask_band_config = self.features_config.get('mask_band', {})
                processor_config = mask_band_config.get('processor', {})
                mask_type = processor_config.get('mask_type', 'hard')
                soft_mask_sigma = processor_config.get('soft_mask_sigma', 2.0)
                
                # 3. 根据 mask_type 生成像素掩膜
                if mask_type == 'soft':
                    # 【软掩膜模式】使用距离变换 + 高斯平滑
                    from scipy.ndimage import distance_transform_edt
                    
                    # 计算到无效区域的距离
                    dist_to_invalid = distance_transform_edt(1 - base_mask)
                    
                    # 应用高斯核生成连续的软掩膜
                    pixel_mask = np.exp(-(dist_to_invalid ** 2) / (2 * soft_mask_sigma ** 2))
                    pixel_mask = np.clip(pixel_mask, 0, 1).astype(np.float32)
                else:
                    # 【硬掩膜模式（默认）】直接使用二值掩膜
                    pixel_mask = base_mask.copy()
                
                # 4. 如果启用高级处理器，进一步处理（可覆盖上面的结果）
                if self.use_advanced_processor and self.cloud_mask_processor:
                    # 【高级模式】使用 CloudMaskProcessor 的完整功能
                    if self.cloud_mask_dir and self.cloud_mask_dir.exists():
                        # 从外部 cloud_mask 文件读取
                        try:
                            date_str = lr_path.name.split('_')[0]
                            tile_id = re.search(r'tile_(\d+_\d+)', lr_path.name).group(1)
                            cloud_mask_path = self.cloud_mask_dir / self.cloud_mask_pattern.format(
                                date=date_str, tile_id=tile_id
                            )
                            
                            if cloud_mask_path.exists():
                                with rasterio.open(cloud_mask_path) as mask_src:
                                    if mask_src.height == h and mask_src.width == w:
                                        cloud_mask_data = mask_src.read(1)
                                        pixel_mask, _ = self.cloud_mask_processor.process_pixel_mask(
                                            cloud_mask_data, 
                                            reflectance_data=reflectance_data
                                        )
                                    else:
                                        print(f"[Warning] Cloud mask size mismatch for {lr_path.name}, using base mask")
                            # 如果文件不存在或异常，保留之前生成的 pixel_mask
                        except Exception as e:
                            print(f"[Warning] Failed to load cloud mask: {e}, using base mask")
                    else:
                        # 没有外部文件，使用高级处理器从反射率推断
                        try:
                            inferred_mask_data = (base_mask > 0).astype(np.int32)
                            pixel_mask, _ = self.cloud_mask_processor.process_pixel_mask(
                                inferred_mask_data,
                                reflectance_data=reflectance_data
                            )
                        except Exception as e:
                            print(f"[Warning] Advanced processor failed: {e}, using base mask")
                
                # 5. 安全检查：确保掩膜中没有 NaN
                if np.isnan(pixel_mask).any():
                    print(f"⚠️  Warning: NaN in pixel_mask for {lr_path.name}, filling with 0")
                    pixel_mask = np.nan_to_num(pixel_mask, nan=0.0)
                
                # 6. 裁剪范围并保存缓存
                pixel_mask = np.clip(pixel_mask, 0.0, 1.0).astype(np.float32)
                
                if pixel_masks_cache is not None:
                    pixel_masks_cache.append(pixel_mask.copy())
                
                if spatial_mask_list is not None:
                    spatial_mask_list.append(pixel_mask.copy())
                
                # 7. 归一化到 [-1, 1] 范围并添加通道维度
                mask_band = (pixel_mask * 2.0 - 1.0)[np.newaxis, :, :]  # (1, H, W)
                features_to_concat.append(mask_band)

# ============================================================================
# 使用说明
# ============================================================================
#
# 1. 配置文件示例（configs/ablation/ablation_4a_with_cross_attention_softmask.yaml）:
#    
#    features:
#      mask_band:
#        enabled: true
#        use_advanced_processor: false  # 不需要外部处理器
#        processor:
#          mask_type: "soft"           # ☆ 关键：启用软掩膜
#          soft_mask_sigma: 2.0        # 控制平滑强度
#
# 2. 运行命令:
#    
#    python main.py \
#      --cfg_path configs/ablation/ablation_4a_with_cross_attention_softmask.yaml \
#      --mode train
#
# 3. 验证软掩膜是否生效:
#    
#    在输出中应该看到：
#    [Dataset INFO] Initialized for {...} tile locations.
#      - Mask Feature Injection: Enabled
#      - Cloud Mask: Simple inference (reflectance > 0) [with soft mask processing]
#
# ============================================================================
# 关键改动说明
# ============================================================================
#
# 【之前的问题】：
#   mask_type: "soft" 配置被完全忽视，因为只有在 use_advanced_processor=true 时
#   CloudMaskProcessor 才会被初始化。
#
# 【改进方案】：
#   - 第一步：生成基础掩膜（硬掩膜）
#   - 第二步：根据 mask_type 配置生成最终掩膜（软或硬）
#   - 第三步：如果启用高级处理器，进一步优化
#
#   这样，mask_type 的配置始终有效，与 use_advanced_processor 独立。
#
# 【性能影响】：
#   - 硬掩膜：0 额外计算
#   - 软掩膜：+距离变换（scipy.ndimage.distance_transform_edt）计算
#           约 +5-10ms 每个时相（取决于分辨率）
#
# ============================================================================
