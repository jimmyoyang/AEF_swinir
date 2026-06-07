# 文件路径: models/network_swinir.py
# (版本: v_anytime_flexible_fusion - 实现了可配置时序融合的最终版本)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.spectral_postprocessors import build_spectral_postprocessor

# ==============================================================================
# 1. 基础模块 (Mlp, window_partition, etc.)
#
# 【不变】这部分是SwinIR的核心组件，我们完整保留，不做任何改动。
# ==============================================================================
class Mlp(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    # ... (您提供的完整 WindowAttention 代码)
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    # ... (您提供的完整 SwinTransformerBlock 代码)
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    # ... (您提供的完整 BasicLayer 代码)
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RSTB(nn.Module):
    # ... (您提供的完整 RSTB 代码)
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim, dim, 3, 1, 1))
    def forward(self, x, x_size):
        # import pdb;pdb.set_trace()
        B,L,C=x.shape
        # return self.conv(self.residual_group(x).view(-1, *x_size, self.dim).permute(0, 3, 1, 2)) + x
        return self.conv(self.residual_group(x).view(-1, *x_size, self.dim).permute(0, 3, 1, 2)).reshape(B,L,C) + x

class PatchEmbed(nn.Module):
    # ... (您提供的完整 PatchEmbed 代码)
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    # ... (您提供的完整 PatchUnEmbed 代码)
    def __init__(self, img_size=224, patch_size=4, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.embed_dim = embed_dim
    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

class Upsample(nn.Sequential):
    # ... (您提供的完整 Upsample 代码)
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Please use 2^n or 3.')
        super(Upsample, self).__init__(*m)

# ==============================================================================
# 2. 辅助函数 (时间编码 & 时序聚合)
#
# 【不变】保留您已有的时间编码函数。
# 【新增】将不同的时序聚合策略封装成独立的函数，并创建注册表。
# ==============================================================================

# ------------------------------------------------------------------------------
# 位置编码器
# ------------------------------------------------------------------------------

class PositionEmbeddingSinCos(nn.Module):
    """公式型 2D 位置编码，类似 DETR 中的实现。

    输出张量形状为 ``(C, H, W)`` where ``C == 2 * num_pos_feats``.
    ``num_pos_feats`` 通常设置为 ``embed_dim//2``。
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, size):
        """Generate positional embedding for a grid of the given size.

        Args:
            size: tuple ``(H, W)``
        Returns:
            Tensor of shape ``(C, H, W)``.
        """
        H, W = size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 【安全检查】确保 H 和 W 都至少为 1
        if H < 1 or W < 1:
            raise ValueError(f"Image dimensions must be >= 1, but got H={H}, W={W}")
        
        # 创建坐标（支持任意大小，包括 H=1 或 W=1）
        y = torch.arange(H, device=device, dtype=torch.float32)  # (H,)
        x = torch.arange(W, device=device, dtype=torch.float32)  # (W,)
        
        # 原始化（可选）
        if self.normalize:
            y = y / max(H - 1, 1) * self.scale
            x = x / max(W - 1, 1) * self.scale
        
        # 频率维度
        num_pos_feats = self.num_pos_feats
        dim_t = torch.arange(num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / max(num_pos_feats, 1))
        
        # 计算编码：y:(H,) -> (H,1,num_pos) / (num_pos,) -> (H,num_pos) 
        # 再扩展到 (H,W,num_pos)
        pos_y = y.unsqueeze(1) / dim_t.unsqueeze(0)  # (H, num_pos_feats)
        pos_x = x.unsqueeze(1) / dim_t.unsqueeze(0)  # (W, num_pos_feats)
        
        # 应用 sin 和 cos
        pos_y = torch.cat([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=1)  # (H, 2*num_pos_feats)
        pos_x = torch.cat([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=1)  # (W, 2*num_pos_feats)
        
        # 广播到 2D 网格：pos_y (H, 2*num) -> (H, 1, 2*num) -> (H, W, 2*num)
        # pos_x (W, 2*num) -> (1, W, 2*num) -> (H, W, 2*num)
        pos_y = pos_y.unsqueeze(1).expand(H, W, -1)  # (H, W, 2*num_pos_feats)
        pos_x = pos_x.unsqueeze(0).expand(H, W, -1)  # (H, W, 2*num_pos_feats)
        
        # 【验证形状一致性】确保两个张量的前两个维度匹配
        if pos_y.shape[:2] != pos_x.shape[:2]:
            raise RuntimeError(f"Shape mismatch before cat: pos_y.shape={pos_y.shape}, pos_x.shape={pos_x.shape}. "
                             f"Expected both to have shape ({H}, {W}, *)")
        
        # 拼接 -> (H, W, 4*num_pos_feats)
        pos = torch.cat([pos_y, pos_x], dim=2)
        
        # 转为 (C, H, W)
        pos = pos.permute(2, 0, 1)
        
        return pos


# ------------------------------------------------------------------------------
# 云-光 交叉注意力模块
# ------------------------------------------------------------------------------

class CloudCrossAttention(nn.Module):
    """云掩膜→光学特征的交叉注意力模块。

    设计思路：使用 "云掩膜当提问者"，光学特征当 "知识库"。
    Q 从 mask_prob 获得，K/V 均来自 fused_feat；通过多头注意力计算
    每个像素位置在光学空间中的加权表示，然后与原始特征残差
    相加以增强云区。

    额外功能：
    * ``downsample_rate`` 控制内部 attention 维度（embedding_dim//rate），
      允许在显存受限时缩小计算。
    * ``concat_value`` 参数可选地把云特征与光学特征拼接作为 V，
      目前默认为 False（仅使用光特征）。

    模块保证输出形状和输入 fused_feat 相同，可与现有流水线无缝对接。
    """

    def __init__(self, embed_dim, num_heads, downsample_rate=1, concat_value=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 确保 embed_dim 能被 downsample_rate 整除，避免后续除法丢失
        assert embed_dim % downsample_rate == 0, "embed_dim must be divisible by downsample_rate"
        self.internal_dim = embed_dim // downsample_rate
        assert self.internal_dim % num_heads == 0, "num_heads must divide internal_dim"
        self.concat_value = concat_value

        # --------------------------------------
        # 将 mask_prob 从 1 通道升维到 internal_dim 通道
        # --------------------------------------
        # 注意：只有 Q 分支使用了 ReLU 激活，K/V 直接线性投影。
        # 这个不对称来源于实验观察：对 mask 进行非线性变换有助于
        # 提高注意力的稀疏性；如果未来需求改变，可以将激活移除或
        # 在所有分支统一添加。
        self.q_conv = nn.Sequential(
            nn.Conv2d(1, self.internal_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # K 来自 fused_feat -> internal_dim
        self.k_conv = nn.Conv2d(embed_dim, self.internal_dim, kernel_size=1)

        # 如果需要将 mask 拼接到 V 中，需要把 mask 升到 embed_dim
        # （而不是 internal_dim），以便与 fused_feat 通道对齐后再投影到 internal_dim
        if concat_value:
            self.mask_channel_mapper = nn.Sequential(
                nn.Conv2d(1, embed_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            # V 的输入通道为 2*embed_dim -> 投影到 internal_dim
            self.v_conv = nn.Conv2d(embed_dim * 2, self.internal_dim, kernel_size=1)
        else:
            # 仅使用 fused_feat 作为 V 的来源
            self.v_conv = nn.Conv2d(embed_dim, self.internal_dim, kernel_size=1)

        self.attn = nn.MultiheadAttention(self.internal_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(self.internal_dim, embed_dim)

    def forward(self, mask_prob, fused_feat, pos_encoding=None):
        """执行交叉注意力。

        Args:
            mask_prob: Tensor, shape (B*T, 1, H, W), 云概率图
            fused_feat: Tensor, shape (B*T, C, H, W), 光+时融合特征
            pos_encoding: Optional[Tensor], shape (1, internal_dim, H, W), 位置编码
        Returns:
            enhanced_feat: Tensor, same shape as fused_feat
        """
        # ensure inputs are on the same device to avoid cross‑device errors
        mask_prob = mask_prob.to(fused_feat.device)

        # fused_feat 的第一个维度实际上是 B*T，因此用 BT 更直观
        BT, C, H, W = fused_feat.shape

        # 鲁棒性检查：确保 fused_feat 通道数与 embed_dim 匹配
        assert C == self.embed_dim, f"fused_feat channel ({C}) must equal embed_dim ({self.embed_dim})"

        # ---------- Q: 由云掩膜生成 ----------
        q_feat = self.q_conv(mask_prob)                     # (B*T, internal, H, W)
        
        # 【新增】如果提供了位置编码，添加到 Q 特征上（实现 Q-KV 对称性）
        if pos_encoding is not None:
            pos_encoding = pos_encoding.to(q_feat.device)
            # 确保位置编码通道数与 q_feat 匹配
            if pos_encoding.shape[1] != q_feat.shape[1]:
                # 如果位置编码是 embed_dim，需要投影到 internal_dim
                # 为了简单，我们直接用插值或切片来适配
                # 这里使用线性插值的方式：先 flatten 再 interpolate
                pe_flat = pos_encoding.flatten(2).permute(0, 2, 1)  # (1, HW, C_pe)
                pe_resized = F.interpolate(
                    pe_flat.permute(0, 2, 1).view(1, pos_encoding.shape[1], H, W),
                    size=(H, W), mode='bilinear', align_corners=False
                )
                # 通道维度投影：使用 1x1 卷积或简单切片
                # 简单方案：如果 C_pe > internal_dim，取前 internal_dim 个通道
                if pos_encoding.shape[1] > q_feat.shape[1]:
                    pos_encoding = pos_encoding[:, :q_feat.shape[1], :, :]
                # 如果 C_pe < internal_dim，用零填充
                elif pos_encoding.shape[1] < q_feat.shape[1]:
                    padding = torch.zeros(
                        1, q_feat.shape[1] - pos_encoding.shape[1], H, W,
                        device=pos_encoding.device, dtype=pos_encoding.dtype
                    )
                    pos_encoding = torch.cat([pos_encoding, padding], dim=1)
            
            q_feat = q_feat + pos_encoding  # 添加位置编码
        
        q_seq = q_feat.flatten(2).permute(0, 2, 1)          # (B*T, HW, internal)

        # ---------- K, V: 从 fused_feat 生成 ----------
        k_feat = self.k_conv(fused_feat).flatten(2).permute(0, 2, 1)  # (B*T, HW, internal)

        if self.concat_value:
            # 将 mask 升到 embed_dim，再与 fused_feat 在通道维拼接，最后投影到 internal_dim
            mask_feat = self.mask_channel_mapper(mask_prob)  # (B*T, embed_dim, H, W)
            v_base = torch.cat([mask_feat, fused_feat], dim=1)  # (B*T, 2*embed_dim, H, W)
            v_feat = self.v_conv(v_base).flatten(2).permute(0, 2, 1)
        else:
            v_feat = self.v_conv(fused_feat).flatten(2).permute(0, 2, 1)

        # ---------- 多头 attention ----------
        attn_out, _ = self.attn(q_seq, k_feat, v_feat)      # (B*T, HW, internal)

        # ---------- 恢复空间结构并残差 ----------
        out_feat = self.out_proj(attn_out)                  # (B*T, HW, embed_dim)
        out_feat = out_feat.permute(0, 2, 1).view(BT, C, H, W)
        enhanced = fused_feat + out_feat                    # 残差连接
        return enhanced


def get_timestamp_encoding(timestamps, encoding_dim=64):
    """
    为一批时间戳（例如，年内日）生成正弦/余弦位置编码。
    """
    if encoding_dim % 2 != 0:
        raise ValueError(f"Encoding dimension must be an even number, but got {encoding_dim}")
    position = timestamps.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, encoding_dim, 2, device=timestamps.device).float() * -(np.log(10000.0) / encoding_dim))
    pe = torch.zeros(len(timestamps), encoding_dim, device=timestamps.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def temporal_fusion_mean(fused_feat, B, T, D, H, W, **kwargs):
    """
    步骤一：基线策略。使用简单的平均池化进行时序聚合。
    支持mask，只对有效时相进行平均。
    """
    # 确保 fused_feat 是 (B*T, D, H, W) 格式
    if fused_feat.dim() != 4:
        raise ValueError(f"fused_feat should be 4D (B*T, D, H, W), but got shape {fused_feat.shape}")
    
    # 重塑为 (B, T, D, H, W)
    fused_feat_reshaped = fused_feat.view(B, T, D, H, W)  # (B, T, D, H, W)
    
    mask = kwargs.get('mask', None)
    if mask is not None:
        # 确保 mask 是 (B, T) 格式
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)  # (T,) -> (1, T)
        elif mask.dim() == 2:
            pass  # 已经是 (B, T)
        else:
            raise ValueError(f"mask should be 1D (T,) or 2D (B, T), but got shape {mask.shape}")
        
        # 确保 mask 的 batch 维度匹配
        if mask.shape[0] != B:
            raise ValueError(f"mask batch dimension ({mask.shape[0]}) doesn't match B ({B})")
        if mask.shape[1] != T:
            raise ValueError(f"mask time dimension ({mask.shape[1]}) doesn't match T ({T})")
        
        # 将无效时相置为0，然后计算平均
        # mask: (B, T) -> (B, T, 1, 1, 1) 用于广播
        mask_expanded = mask.view(B, T, 1, 1, 1)  # (B, T, 1, 1, 1)
        masked_feat = fused_feat_reshaped * mask_expanded  # (B, T, D, H, W)
        sum_feat = masked_feat.sum(dim=1)  # (B, D, H, W) - 在时间维度上求和
        valid_count = mask.sum(dim=1, keepdim=False).clamp(min=1)  # (B,) 避免除零
        # 扩展 valid_count 到 (B, 1, 1, 1) 用于广播除法
        valid_count = valid_count.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        result = sum_feat / valid_count  # (B, D, H, W)
    else:
        # 没有 mask，直接对时间维度求平均
        result = fused_feat_reshaped.mean(dim=1)  # (B, D, H, W)
    
    # 确保输出是 4D (B, D, H, W)
    if result.dim() != 4:
        raise ValueError(f"temporal_fusion_mean should return 4D tensor (B, D, H, W), but got shape {result.shape}")
    
    return result

def temporal_fusion_attention(fused_feat, B, T, D, H, W, **kwargs):
    """
    【预留空间】步骤二：先进策略。使用时序注意力进行聚合。
    """
    attn_module = kwargs.get('attn_module')
    if attn_module is not None:
        # fused_feat: (B*T, D, H, W) -> reshape for attention
        feat = fused_feat.view(B, T, D, H, W).transpose(1, 2).contiguous()  # B,D,T,H,W
        feat = feat.view(B * D, T, H * W).transpose(1, 2)  # (B*D)*(H*W) tokens
        # 这里仅示范调用，如果未来需要，可根据 attn_module 实现具体逻辑
        feat = attn_module(feat)
        # 回退到原始形状
        feat = feat.transpose(1, 2).view(B, D, T, H, W).transpose(1, 2).contiguous()
        return feat.view(B * T, D, H, W)

    # 在未实现前，打印警告并回退到平均融合，以保证代码可运行
    print("【警告】temporal_fusion_attention 尚未实现，暂时回退到平均融合。")
    return temporal_fusion_mean(fused_feat, B, T, D, H, W, **kwargs)

# 【新增】创建一个函数注册表，便于动态调用
TEMPORAL_FUSION_REGISTRY = {
    'mean': temporal_fusion_mean,
    'attention': temporal_fusion_attention,
}

# ==============================================================================
# 3. 主模型 SwinIR
#
# 【修改】对SwinIR类进行微小修改，使其能够根据配置动态调用不同的时序融合函数。
# ==============================================================================
class SwinIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=9, time_encoding_dim=64,
                 embed_dim=180, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 upscale=3, upsampler='pixelshuffle', resi_connection='1conv',
                 out_channels=64,
                 # 【新增】接收新的配置参数，并提供默认值
                 temporal_fusion_mode='mean',
                 temporal_attention_params=None,
                 # cross-attention / positional embedding
                 use_cross_attention=False,
                 cross_num_heads=6,
                 cross_downsample_rate=1,
                 cross_concat_value=False,
                 use_pos_emb=False,
                 use_learnable_pos_emb=False,
                 pos_emb_dim=0,
                 use_spectral_postprocessor=False,
                 spectral_postprocessor_type='gumbel_routing',
                 spectral_postprocessor_params=None,
                 output_l2_normalize=False,
                 output_normalize_eps=1e-8,
                 **kwargs):

        super(SwinIR, self).__init__()
        # --- 1. 保存配置 ---
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.temporal_fusion_mode = temporal_fusion_mode

        # --- cross-attention & positional embedding 配置 ---
        self.use_cross_attention = use_cross_attention
        self.cross_num_heads = cross_num_heads
        self.cross_downsample_rate = cross_downsample_rate
        self.cross_concat_value = cross_concat_value
        self.use_pos_emb = use_pos_emb
        self.use_learnable_pos_emb = use_learnable_pos_emb
        # pos_emb_dim==0 表示与 embed_dim 相同
        self.pos_emb_dim = pos_emb_dim or embed_dim
        self.use_spectral_postprocessor = use_spectral_postprocessor
        self.output_l2_normalize = bool(output_l2_normalize)
        self.output_normalize_eps = float(output_normalize_eps)

        if self.use_pos_emb:
            if self.use_learnable_pos_emb:
                # 学习型位置编码
                self.pos_emb = nn.Parameter(torch.zeros(1, self.pos_emb_dim, img_size, img_size))
            else:
                # 公式型 sin/cos 编码
                self.pos_encoder = PositionEmbeddingSinCos(num_pos_feats=self.pos_emb_dim // 2)

        if self.use_cross_attention:
            self.cross_attn = CloudCrossAttention(embed_dim=embed_dim,
                                                  num_heads=self.cross_num_heads,
                                                  downsample_rate=self.cross_downsample_rate,
                                                  concat_value=self.cross_concat_value)

        # --- 2. 输入嵌入层 (与您之前的版本保持不变) ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.time_embed = nn.Linear(time_encoding_dim, embed_dim)

        # --- 3. 【预留空间】为 'attention' 模式实例化模块 ---
        self.temporal_attn_module = None
        if self.temporal_fusion_mode == 'attention':
            # TODO: 在这里根据 temporal_attention_params 实例化您的注意力模块
            # self.temporal_attn_module = YourAttentionModuleClass(**temporal_attention_params)
            print("【信息】已为 'attention' 模式初始化注意力模块（此为占位符，待实现）。")

        # --- 4. 深层特征提取 (Swin Transformer Body) (与您之前的版本保持不变) ---
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 5. 上采样与重建 (与您之前的版本保持不变) ---
        if upsampler == 'pixelshuffle':
            self.upsample = Upsample(upscale, embed_dim)
        else:
            self.upsample = nn.Identity()
        # --- 6. 最终图像重建 ---
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        # --- 6.5 可选光谱后处理模块 ---
        self.spectral_postprocessor = None
        if self.use_spectral_postprocessor:
            pp_params = spectral_postprocessor_params or {}
            if 'n_channels' not in pp_params:
                pp_params['n_channels'] = out_channels
            self.spectral_postprocessor = build_spectral_postprocessor(
                spectral_postprocessor_type,
                pp_params,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x 应该是 4D (B, C, H, W) 格式
        # 如果意外收到 5D，先处理成 4D
        if x.dim() == 5:
            # (B, T, C, H, W) -> 取第一个时间步或平均
            B, T, C, H, W = x.shape
            x = x.mean(dim=1)  # (B, C, H, W) - 平均池化时间维度
            # 确保结果是 4D
            if x.dim() != 4:
                raise ValueError(f"After reducing 5D input, expected 4D tensor, but got {x.dim()}D with shape {x.shape}")
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 4D (B,C,H,W), got shape {x.shape}")
        
        # 确保 x 是 4D (B, C, H, W)
        assert x.dim() == 4, f"forward_features input must be 4D, got {x.dim()}D with shape {x.shape}"
        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        
        # 确保输出也是 4D
        assert x.dim() == 4, f"forward_features output must be 4D, got {x.dim()}D with shape {x.shape}"
        return x

    def forward(self, data):
        """
        SwinIR 的前向传播，时序聚合部分根据配置动态调用。
        """
        # 获取数据，支持单个样本和batch
        lr_seq = data['lr_sequence']  # 可能是 (T, C, H, W) 或 (B, T, C, H, W)
        timestamps = data['timestamps']  # 可能是 (T,) 或 (B, T)

        mask = data.get('mask', None)  # 可能是 (T,) 或 (B, T)，指示哪些时相是有效的
        mask_prob = data.get('mask_prob', None)  # 可能是 (T, 1, H, W) 或 (B, T, 1, H, W)

        # 处理单个样本的情况：添加 batch 维度
        if lr_seq.dim() == 4:  # (T, C, H, W)
            lr_seq = lr_seq.unsqueeze(0)  # (1, T, C, H, W)
            timestamps = timestamps.unsqueeze(0)  # (1, T)
            if mask is not None:
                mask = mask.unsqueeze(0)  # (1, T)
            if mask_prob is not None:
                # 可能为 (T,1,H,W) -> (1,T,1,H,W)
                mask_prob = mask_prob.unsqueeze(0)
            B, T, C, H, W = lr_seq.shape
        else:  # 已经是 batch 格式 (B, T, C, H, W)
            B, T, C, H, W = lr_seq.shape
            if mask_prob is not None:
                # 如果输入是 (B*T,1,H,W)，尝试恢复
                if mask_prob.dim() == 4 and mask_prob.shape[0] == B * T:
                    mask_prob = mask_prob.view(B, T, 1, H, W)

        # 如果出现单个 batch 时重复 mask_prob
        if mask_prob is not None and mask_prob.dim() == 5 and mask_prob.shape[0] != B:
            mask_prob = mask_prob.repeat(B, 1, 1, 1, 1)

        # 确保形状为 (B, T, 1, H, W)
        if mask_prob is not None:
            assert mask_prob.dim() == 5, f"mask_prob should be 5D but got {mask_prob.shape}"

        # 1, 2, 3. 展平、嵌入、融合 (与您之前的版本完全相同)
        # 1. 将序列数据展平，以便进行2D卷积
        lr_seq_flat = lr_seq.view(B * T, C, H, W)
        
        # 2. 独立嵌入每个时相的图像特征
        optical_feat = self.conv_first(lr_seq_flat) # (B*T, embed_dim, H, W)
        
        # 3. 嵌入时间信息并融合
        time_enc = get_timestamp_encoding(timestamps.view(-1), self.time_embed.in_features)
        time_feat = self.time_embed(time_enc) # (B*T, embed_dim)
        fused_feat = optical_feat + time_feat.unsqueeze(-1).unsqueeze(-1)

        # 3.5 位置编码（如果启用）
        if self.use_pos_emb:
            if self.use_learnable_pos_emb:
                pe = self.pos_emb
            else:
                pe = self.pos_encoder((H, W)).unsqueeze(0)  # (1, C, H, W)
            fused_feat = fused_feat + pe.to(fused_feat.device)

        # 4. 云-光交叉注意力模块（在时序聚合前）
        enhanced_feat = fused_feat
        if self.use_cross_attention and mask_prob is not None:
            # 将 mask_prob 拉平成 (B*T,1,H,W)
            mp = mask_prob.view(B * T, 1, H, W)
            
            # 【新增】如果启用位置编码，准备传递给 cross_attn（实现 Q-KV 对称性）
            # 修改原因：之前只有 fused_feat (K/V) 有位置编码，mask_prob (Q) 没有
            # 这会导致注意力计算时信息不对称，影响跨模态融合效果
            # 注意：位置编码在 CloudCrossAttention 内部的 q_conv 之后添加，避免通道数不匹配
            pe_for_mask = None
            if self.use_pos_emb:
                # 获取位置编码（与 fused_feat 使用的相同）
                if self.use_learnable_pos_emb:
                    pe_for_mask = self.pos_emb
                else:
                    pe_for_mask = self.pos_encoder((H, W)).unsqueeze(0)  # (1, C, H, W)
            
            enhanced_feat = self.cross_attn(mp, fused_feat, pos_encoding=pe_for_mask)

        # 5. 【修改】动态调用时序聚合函数
        # ----------------------------------------------------
        fusion_function = TEMPORAL_FUSION_REGISTRY.get(self.temporal_fusion_mode)
        if fusion_function is None:
            raise ValueError(f"未知的时序融合模式: '{self.temporal_fusion_mode}'。请检查配置文件。")
        
        # 打包所有可能需要的参数
        fusion_kwargs = {
            'B': B, 'T': T, 'D': self.embed_dim, 'H': H, 'W': W,
            'attn_module': self.temporal_attn_module, # 将实例化的模块传入(步骤二需要)
            'mask': mask # 传递mask用于处理填充的时相
        }
        
        # 像插件一样调用选定的聚合函数
        agg_feat = fusion_function(enhanced_feat, **fusion_kwargs) # Shape: (B, D, H, W)
        # ----------------------------------------------------

        # 【安全检查】确保 agg_feat 是 4D (B, D, H, W)
        if agg_feat.dim() != 4:
            raise ValueError(
                f"temporal fusion function returned {agg_feat.dim()}D tensor with shape {agg_feat.shape}, "
                f"but expected 4D (B, D, H, W). Please check the fusion function implementation."
            )
        if agg_feat.shape[0] != B or agg_feat.shape[1] != self.embed_dim:
            raise ValueError(
                f"temporal fusion output shape mismatch: got {agg_feat.shape}, "
                f"expected (B={B}, D={self.embed_dim}, H={H}, W={W})"
            )

        # 5, 6, 7. 深层特征提取、上采样、重建 (与您之前的版本完全相同)
        # 5. 深层特征提取 (SwinIR Body)
        # agg_feat 应该是 (B, D, H, W) 格式，直接传入 forward_features
        res = self.conv_after_body(self.forward_features(agg_feat)) + agg_feat
        
        # 6. 后置上采样
        x = self.upsample(res)
        
        # 7. 最终重建
        x = self.conv_last(x)

        if self.spectral_postprocessor is not None:
            x = self.spectral_postprocessor(x)

        if self.output_l2_normalize:
            x = F.normalize(x, p=2, dim=1, eps=self.output_normalize_eps)

        return x
