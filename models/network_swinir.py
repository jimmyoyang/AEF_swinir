# 文件路径: models/network_swinir.py
# (版本: v_anytime_flexible_fusion - 实现了可配置时序融合的最终版本)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
    mask = kwargs.get('mask', None)
    if mask is not None:
        # 只对有效时相进行平均
        fused_feat_reshaped = fused_feat.view(B, T, D, H, W)  # (B, T, D, H, W)
        # 将无效时相置为0，然后计算平均
        masked_feat = fused_feat_reshaped * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, T, D, H, W)
        sum_feat = masked_feat.sum(dim=1)  # (B, D, H, W)
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1) 避免除零
        return sum_feat / valid_count.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        return fused_feat.view(B, T, D, H, W).mean(dim=1)

def temporal_fusion_attention(fused_feat, B, T, D, H, W, **kwargs):
    """
    【预留空间】步骤二：先进策略。使用时序注意力进行聚合。
    """
    # TODO: 在这里实现您的注意力逻辑。
    # 示例:
    # attn_module = kwargs.get('attn_module')
    # if attn_module is not None:
    #     # ... 实现加权融合
    #     pass

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
                 **kwargs):

        super(SwinIR, self).__init__()
        # --- 1. 保存配置 ---
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.temporal_fusion_mode = temporal_fusion_mode

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
        # import pdb;pdb.set_trace()
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, data):
        """
        SwinIR 的前向传播，时序聚合部分根据配置动态调用。
        """
        # 获取数据，支持单个样本和batch
        lr_seq = data['lr_sequence']  # 可能是 (T, C, H, W) 或 (B, T, C, H, W)
        timestamps = data['timestamps']  # 可能是 (T,) 或 (B, T)
        mask = data.get('mask', None)  # 可能是 (T,) 或 (B, T)，指示哪些时相是有效的

        # 处理单个样本的情况：添加batch维度
        if lr_seq.dim() == 4:  # (T, C, H, W)
            lr_seq = lr_seq.unsqueeze(0)  # (1, T, C, H, W)
            timestamps = timestamps.unsqueeze(0)  # (1, T)
            if mask is not None:
                mask = mask.unsqueeze(0)  # (1, T)
            B, T, C, H, W = lr_seq.shape
        else:  # 已经是batch格式 (B, T, C, H, W)
            B, T, C, H, W = lr_seq.shape

        # 1, 2, 3. 展平、嵌入、融合 (与您之前的版本完全相同)
        # 1. 将序列数据展平，以便进行2D卷积
        lr_seq_flat = lr_seq.view(B * T, C, H, W)
        
        # 2. 独立嵌入每个时相的图像特征
        optical_feat = self.conv_first(lr_seq_flat) # (B*T, embed_dim, H, W)
        
        # 3. 嵌入时间信息并融合
        time_enc = get_timestamp_encoding(timestamps.view(-1), self.time_embed.in_features)
        time_feat = self.time_embed(time_enc) # (B*T, embed_dim)
        fused_feat = optical_feat + time_feat.unsqueeze(-1).unsqueeze(-1)

        # 4. 【修改】动态调用时序聚合函数
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
        agg_feat = fusion_function(fused_feat, **fusion_kwargs) # Shape: (B, D, H, W)
        # ----------------------------------------------------

        # 5, 6, 7. 深层特征提取、上采样、重建 (与您之前的版本完全相同)
        # 5. 深层特征提取 (SwinIR Body)
        res = self.conv_after_body(self.forward_features(agg_feat)) + agg_feat
        
        # 6. 后置上采样
        x = self.upsample(res)
        
        # 7. 最终重建
        x = self.conv_last(x)
        
        return x