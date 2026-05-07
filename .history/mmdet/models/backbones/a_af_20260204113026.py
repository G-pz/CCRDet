# -*- encoding:utf-8 -*-
# !/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import cv2
import numpy as np
from ..builder import BACKBONES
from .ts_resnet import TwoStreamResNet

import matplotlib
import kornia.color as K
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.feature_analyze import save_feature_to_img

def compute_brightness_map(v_img):
    v_img = v_img.float().clamp_(0.0, 1.0)
    
    yuv = K.rgb_to_yuv(v_img)               # [B, 3, H, W]
    y = yuv[:, 0:1, :, :]                   # 取 Y 通道 [B, 1, H, W]
    
    B, _, H, W = y.shape
    flat = y.view(B, -1)
    min_v = flat.min(dim=1, keepdim=True)[0]
    max_v = flat.max(dim=1, keepdim=True)[0]
    denom = max_v - min_v + 1e-6
    norm = (flat - min_v) / denom
    brightness_map = norm.view(B, 1, H, W)
    
    return brightness_map.clamp_(0.0, 1.0) 

def sobel_gradient_edge_guided(t_img, ksize=3, normalize=True, output_range='0-1'):
    """
    Sobel Gradient Edge Guided Module (全 Torch 版本，针对 RGB 输入)
    
    输入:
        t_img (torch.Tensor): [B, 3, H, W]，float32，通常 [0,1] 范围的 RGB 图像
        ksize (int): Sobel 核大小，目前支持 3
        normalize (bool): 是否逐样本 min-max 归一化梯度幅值
        output_range (str): '0-1'（默认，float32 [0,1]） 或 '0-255'（uint8 [0,255]）
    
    输出:
        torch.Tensor: [B, 1, H, W]，梯度幅值图（边缘强度图）
    
    流程：RGB → 灰度 → Sobel 梯度 → 幅值 → (可选归一化) → 输出指定范围
    """
    assert t_img.dim() == 4 and t_img.shape[1] == 3, \
        "输入必须为 [B, 3, H, W] 的 RGB 图像"

    t_img = t_img.float()  # 确保 float32

    B, C, H, W = t_img.shape

    # RGB → 灰度（使用 BT.601 标准系数，与 cv2.COLOR_RGB2GRAY 一致）
    # gray = 0.299R + 0.587G + 0.114B
    weights = torch.tensor([0.299, 0.587, 0.114], 
                           device=t_img.device, 
                           dtype=t_img.dtype)
    gray = torch.einsum('bchw,c->bhw', t_img, weights).unsqueeze(1)  # [B, 1, H, W]

    # 或者更直观写法（性能相近）：
    # gray = (t_img[:, 0] * 0.299 + t_img[:, 1] * 0.587 + t_img[:, 2] * 0.114).unsqueeze(1)

    # 定义 3x3 Sobel 核
    if ksize != 3:
        raise ValueError("目前仅支持 ksize=3，可自行扩展到 5 或 Scharr")

    sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]]], 
                                  dtype=torch.float32, device=t_img.device)

    sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                     [ 0,  0,  0],
                                     [ 1,  2,  1]]]], 
                                  dtype=torch.float32, device=t_img.device)

    # 计算梯度
    grad_x = F.conv2d(gray, sobel_kernel_x, padding=1, groups=1)
    grad_y = F.conv2d(gray, sobel_kernel_y, padding=1, groups=1)

    # 梯度幅值
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # [B, 1, H, W]

    if normalize:
        # 逐样本 min-max 归一化
        flat = magnitude.view(B, -1)  # [B, H*W]
        min_val = flat.min(dim=1, keepdim=True)[0]
        max_val = flat.max(dim=1, keepdim=True)[0]
        denom = max_val - min_val + 1e-6
        normalized = (flat - min_val) / denom
        magnitude = normalized.view(B, 1, H, W)

    # 输出范围处理
    if output_range == '0-255':
        magnitude = (magnitude * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    elif output_range == '0-1':
        magnitude = magnitude.clamp(0.0, 1.0)
    else:
        raise ValueError("output_range 仅支持 '0-1' 或 '0-255'")

    return magnitude
    
def compute_displacement_and_warp(x, y, im_file, visualize=False, save_dir="runs/flow", step=0):
    """
    计算两个特征图之间的位移场并可视化
    输入: x, y: [B, C, H, W]
    输出: warped_feature: [B, C, H, W]
    """
    B, C, H, W = x.shape
    warped_features = []
    
    def estimate_flow(x1, y1):
        # x1, y1: [C, H, W]
        feature_A_np = x1.max(dim=0)[0].detach().cpu().numpy().astype(np.float32)  # 用最大池化
        feature_B_np = y1.max(dim=0)[0].detach().cpu().numpy().astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            feature_A_np, feature_B_np, None,
            pyr_scale=0.5, levels=2, winsize=9, iterations=5,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        return torch.from_numpy(flow).float().to(x.device)  # [H, W, 2]

    for b in range(B):
        from pathlib import Path
        file = Path(im_file[b]).stem if im_file is not None else None
        displacement_field = estimate_flow(x[b], y[b])   # [H, W, 2] * 5 6 4 
        # print(f"displacement_field max: {displacement_field.abs().max().item():.6f}")
        # 归一化位移场到 [-1, 1]
        displacement_field_normalized = displacement_field / torch.tensor([W, H], device=x.device) * 2  # [H, W, 2]
        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        warped_grid = grid + displacement_field_normalized  # [H, W, 2]
        warped_grid = warped_grid.unsqueeze(0)  # [1, H, W, 2]
        warped_grid = warped_grid.to(x.dtype)  # 保证类型一致
        # 变形特征图A
        warped_feature = F.grid_sample(
            x[b].unsqueeze(0),  # [1, C, H, W]
            warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        warped_features.append(warped_feature)
        
        #可视化部分
        if visualize:
            import os
            if save_dir is not None:
                # 分别创建三个子文件夹
                save_dir_A = os.path.join(save_dir, "featureA")
                save_dir_B = os.path.join(save_dir, "featureB")
                save_dir_W = os.path.join(save_dir, "warped_feature")
                save_dir_F = os.path.join(save_dir, "flow")
                os.makedirs(save_dir_A, exist_ok=True)
                os.makedirs(save_dir_B, exist_ok=True)
                os.makedirs(save_dir_W, exist_ok=True)
                os.makedirs(save_dir_F, exist_ok=True)

            feature_A_np = x[b].max(dim=0)[0].detach().cpu().numpy().astype(np.float32)
            feature_B_np = y[b].max(dim=0)[0].detach().cpu().numpy().astype(np.float32)
            h, _ = feature_A_np.shape

            # 保存Feature A
            if (save_dir and file is not None) and h == 64:
                # resize到(512, 640)，保持长宽比（如需pad可用INTER_NEAREST）
                feature_A_np_resized = cv2.resize(feature_A_np, (640, 512), interpolation=cv2.INTER_LINEAR)
                plt.figure(figsize=(6.4, 5.12))  # figsize单位为英寸，dpi=100时正好640x512像素
                plt.imshow(feature_A_np_resized, cmap='viridis')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除边距
                plt.savefig(f"{save_dir_A}/{file}.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

            # 保存Feature B
            if (save_dir and file is not None) and h == 64:
                feature_B_np_resized = cv2.resize(feature_B_np, (640, 512), interpolation=cv2.INTER_LINEAR)
                plt.figure(figsize=(6.4, 5.12))  # 保持长宽比，输出640x512像素
                plt.imshow(feature_B_np_resized, cmap='viridis')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f"{save_dir_B}/{file}.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
            
            # 保存Warped Feature
            if (save_dir and file is not None) and h == 64:
                warped_feature_np = warped_feature.squeeze(0).max(dim=0)[0].detach().cpu().numpy().astype(np.float32)
                warped_feature_np_resized = cv2.resize(warped_feature_np, (640, 512), interpolation=cv2.INTER_LINEAR)                
                plt.figure(figsize=(6.4, 5.12))
                plt.imshow(warped_feature_np_resized, cmap='viridis')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f"{save_dir_W}/{file}.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

            # 保存Flow
            if (save_dir and file is not None) and h == 64:
                flow_np = displacement_field.detach().cpu().numpy()
                hsv = np.zeros((H, W, 3), dtype=np.uint8)
                mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(mag * 100, 0, 255)
                rgb = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), (640, 512), interpolation=cv2.INTER_LINEAR)
                plt.figure(figsize=(6.4, 5.12))
                plt.imshow(rgb)
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f"{save_dir_F}/{file}.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
                
            if (save_dir and file is not None) and h == 64:
                flow_np = displacement_field.detach().cpu().numpy()  # [H, W, 2]
                step = 16
                # 1. 先resize flow到(512, 640)
                flow_np_resized = cv2.resize(flow_np, (640, 512), interpolation=cv2.INTER_LINEAR)
                # 2. 生成箭头坐标
                Hq, Wq = flow_np_resized.shape[:2]
                Y, X = np.mgrid[0:Hq:step, 0:Wq:step]
                U = flow_np_resized[::step, ::step, 0]
                V = flow_np_resized[::step, ::step, 1]
                # 3. 画底图和箭头
                factor = 4  # 放大箭头
                plt.figure(figsize=(6.4, 5.12))
                plt.imshow(feature_A_np_resized, cmap='viridis')
                plt.quiver(X, Y, U * factor, V * factor, color='red', angles='xy', scale_units='xy', scale=1, width=0.003)
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f"{save_dir_F}/{file}_arrow.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
           
    # print(f"全局Flow mag min/max: {global_mag_min:.6f}, {global_mag_max:.6f}")    
    warped_features = torch.cat(warped_features, dim=0)  # [B, C, H, W]
    return warped_features

class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
   
class DecomposeFusion(nn.Module):
    def __init__(self, in_channels, channel_first=False, norm_layer=nn.LayerNorm,
                 ssm_act_layer=nn.SiLU,
                 mlp_act_layer=nn.GELU, vssm_cfg=vssm_cfg):
        super().__init__()
        kwargs = vssm_cfg
        self.spatial_attention = nn.Sequential(*[
            SpatialAttention(kernel_size=7),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()])
        self.spatial_attention_rgb = nn.Sequential(*[
            SpatialAttention(kernel_size=7),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()])
        self.spatial_attention_t = nn.Sequential(*[
            SpatialAttention(kernel_size=7),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()])
        self.bright_weight = nn.Conv2d(1, 1, 3, padding=1)
        self.thermal_weight = nn.Conv2d(1, 1, 3, padding=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
    def forward(self, x, im_file):
        x, y, z = x
        _, _, H, W = x.shape
        rgb = z[:, :3, :, :]
        t = z[:, 3:, :, :]

        luminace_map = compute_brightness_map(rgb)
        luminace_map = F.interpolate(luminace_map, size=(H, W), mode='bilinear')
        luminace_map = luminace_map.to(x.dtype).to(x.device)
        thermal_map = sobel_gradient_edge_guided(t)
        thermal_map = F.interpolate(thermal_map, size=(H, W), mode='bilinear')
        thermal_map = thermal_map.to(x.dtype).to(x.device)
        rgb_weight = F.sigmoid(self.bright_weight(luminace_map)) 
        thermal_weight = F.sigmoid(self.thermal_weight(thermal_map)) 

        x = compute_displacement_and_warp(x, y, im_file)
        common_feature = self.spatial_attention((x+y))
        rgb_specific = self.spatial_attention_rgb(self.act(x-y) * (1 + rgb_weight))
        thermal_specific = self.spatial_attention_t(self.act(y-x) * (1 + thermal_weight))
        temp = self.act(self.norm(common_feature + rgb_specific + thermal_specific))
        return temp

@BACKBONES.register_module()
class AMFusionResNet(TwoStreamResNet):
    def __init__(self,
                 dims_in=[256, 512, 1024, 2048],
                 dims_lsk=[32, 32, 64, 64],
                 **kwargs
                 ):
        super(AMFusionResNet, self).__init__(**kwargs)
        self.iter = 0
        self.dims_in = dims_in
        self.dims_lsk = dims_lsk


    # forward function
    def forward(self, rgb_x, thermal_x):
        # resnet part
        rgb_x = self.vis_conv1(rgb_x)
        rgb_x = self.vis_norm1(rgb_x)
        rgb_x = self.relu(rgb_x)

        thermal_x = self.lwir_conv1(thermal_x)
        thermal_x = self.lwir_norm1(thermal_x)
        thermal_x = self.relu(thermal_x)

        rgb_x = self.maxpool(rgb_x)
        thermal_x = self.maxpool(thermal_x)

        outs_rgb, outs_thermal = [], []
        for i in range(self.num_stages):
            # resnet
            rgb_layer_name = self.vis_res_layers[i]
            rgb_res_layer = getattr(self, rgb_layer_name)
            rgb_x = rgb_res_layer(rgb_x)

            thermal_layer_name = self.lwir_res_layers[i]
            thermal_res_layer = getattr(self, thermal_layer_name)
            thermal_x = thermal_res_layer(thermal_x)

            if i < 3:
                rgb_x = compute_displacement_and_warp(rgb_x, thermal_x, im_file=None, visualize=False)

            if i in self.out_indices:
                outs_rgb.append(rgb_x)
                outs_thermal.append(thermal_x)

        return tuple(outs_rgb), tuple(outs_thermal)



