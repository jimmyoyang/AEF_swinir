import numpy as np
import torch
import torch.nn.functional as F


class CloudMaskProcessor:
    """统一处理硬掩膜/软掩膜，并输出时相可用性。"""

    def __init__(
        self,
        mask_type="hard",
        soft_mask_sigma=2.0,
        use_spatial_smoothing=False,
        smoothing_sigma=1.0,
        valid_pixel_ratio_threshold=0.5,
    ):
        self.mask_type = str(mask_type).lower()
        self.soft_mask_sigma = float(soft_mask_sigma)
        self.use_spatial_smoothing = bool(use_spatial_smoothing)
        self.smoothing_sigma = float(smoothing_sigma)
        self.valid_pixel_ratio_threshold = float(valid_pixel_ratio_threshold)

    @staticmethod
    def _to_2d_float(mask):
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
        arr = arr.astype(np.float32)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    @staticmethod
    def _normalize_to_01(arr):
        m = arr.astype(np.float32)
        m_min = float(np.min(m))
        m_max = float(np.max(m))

        if m_min >= 0.0 and m_max <= 1.0:
            return np.clip(m, 0.0, 1.0)

        # Landsat QA 置信度四档：0/1/2/3
        if m_min >= 0.0 and m_max <= 3.0:
            return np.clip(m / 3.0, 0.0, 1.0)

        if m_max - m_min < 1e-6:
            return np.zeros_like(m, dtype=np.float32)

        m = (m - m_min) / (m_max - m_min)
        return np.clip(m, 0.0, 1.0)

    @staticmethod
    def _gaussian_blur(prob_map, sigma):
        if sigma <= 0:
            return prob_map

        radius = max(1, int(round(3 * sigma)))
        kernel_size = 2 * radius + 1
        coords = torch.arange(kernel_size, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()

        weight = kernel_2d.view(1, 1, kernel_size, kernel_size)
        x = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        y = F.conv2d(x, weight)
        return y.squeeze(0).squeeze(0).numpy().astype(np.float32)

    def extract_cloud_from_qa(self, qa_band):
        qa = self._to_2d_float(qa_band)

        # 若像 QA 置信度图（0..3），先转成云概率，再阈值化硬掩膜
        if float(np.max(qa)) <= 3.0 and np.unique(np.round(qa, 2)).size <= 8:
            cloud_prob = self._normalize_to_01(qa)
            return (cloud_prob >= 0.5).astype(np.float32)

        # 二值图或其他图像：按 >0 判为云
        return (qa > 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Backward-compatible aliases for earlier documentation/version names
    # ------------------------------------------------------------------
    def _process_hard_mask(self, cloud_mask_data, reflectance_data=None):
        """Compatibility wrapper: return hard pixel mask in [0,1]."""
        if reflectance_data is not None:
            pixel_mask = (np.all(reflectance_data > 0, axis=0)).astype(np.float32)
        else:
            cloud_mask = self.extract_cloud_from_qa(cloud_mask_data)
            pixel_mask = 1.0 - cloud_mask
        return np.clip(pixel_mask.astype(np.float32), 0.0, 1.0)

    def _process_soft_mask(self, cloud_mask_data, reflectance_data=None):
        """Compatibility wrapper: return soft pixel mask in [0,1]."""
        cloud_prob = self.generate_soft_mask(cloud_mask_data, reflectance_data=reflectance_data)
        pixel_mask = 1.0 - cloud_prob
        return np.clip(pixel_mask.astype(np.float32), 0.0, 1.0)

    def generate_soft_mask(self, cloud_mask_or_qa, reflectance_data=None):
        arr = self._to_2d_float(cloud_mask_or_qa)

        # 优先识别 QA 置信度映射
        if float(np.max(arr)) <= 3.0 and np.unique(np.round(arr, 2)).size <= 8:
            cloud_prob = self._normalize_to_01(arr)
        else:
            # 视作硬掩膜（1=云），并做平滑得到软概率
            hard_cloud = (arr > 0).astype(np.float32)
            sigma = self.soft_mask_sigma if self.soft_mask_sigma > 0 else 1.0
            cloud_prob = self._gaussian_blur(hard_cloud, sigma)
            cloud_prob = self._normalize_to_01(cloud_prob)

        if self.use_spatial_smoothing:
            cloud_prob = self._gaussian_blur(cloud_prob, max(self.smoothing_sigma, 0.1))
            cloud_prob = self._normalize_to_01(cloud_prob)

        return np.clip(cloud_prob.astype(np.float32), 0.0, 1.0)

    def process_pixel_mask(self, cloud_mask_data, reflectance_data=None):
        """返回像素权重掩膜与元信息。

        返回值：
        - pixel_mask: [0,1]，1 表示更可用
        - metadata: {cloud_mask, cloud_prob, is_valid}
        """
        cloud_prob = self.generate_soft_mask(cloud_mask_data, reflectance_data=reflectance_data)
        cloud_mask = (cloud_prob >= 0.5).astype(np.float32)

        if self.mask_type == "soft":
            pixel_mask = 1.0 - cloud_prob
        else:
            if reflectance_data is not None:
                pixel_mask = (np.all(reflectance_data > 0, axis=0)).astype(np.float32)
            else:
                pixel_mask = 1.0 - cloud_mask

        pixel_mask = np.clip(pixel_mask.astype(np.float32), 0.0, 1.0)
        valid_pixel_ratio = float(np.mean(pixel_mask > 0.5))
        is_valid = bool(valid_pixel_ratio > self.valid_pixel_ratio_threshold)

        metadata = {
            "cloud_mask": cloud_mask.astype(np.float32),
            "cloud_prob": cloud_prob.astype(np.float32),
            "is_valid": is_valid,
        }
        return pixel_mask, metadata

    def __call__(self, qa_band, reflectance_data=None):
        cloud_mask = self.extract_cloud_from_qa(qa_band)
        cloud_prob = self.generate_soft_mask(qa_band, reflectance_data=reflectance_data)

        if self.mask_type == "soft":
            pixel_mask = 1.0 - cloud_prob
        else:
            pixel_mask = 1.0 - cloud_mask

        pixel_mask = np.clip(pixel_mask.astype(np.float32), 0.0, 1.0)
        valid_pixel_ratio = float(np.mean(pixel_mask > 0.5))
        is_valid = bool(valid_pixel_ratio > self.valid_pixel_ratio_threshold)

        return {
            "cloud_mask": cloud_mask.astype(np.float32),
            "cloud_prob": cloud_prob.astype(np.float32),
            "is_valid": is_valid,
        }
