# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_neck
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

def winner_takes_all(tensor1, tensor2, k=5):
    diff = tensor1 - tensor2
    exp_diff = torch.exp(torch.clamp(k * diff, max=50))  # 限制最大值，避免溢出
    exp_neg_diff = torch.exp(torch.clamp(k * -diff, max=50))
    w1 = exp_diff / (exp_diff + exp_neg_diff)
    w2 = 1 - w1
    return w1, w2


@DETECTORS.register_module()
class GFLAF(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 tanh=None):
        super(GFLAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.tanh = tanh                      
        self.nect_t = build_neck(neck)
        self.iter = 0
        self.fuse = nn.ModuleList([Fusion(256, tanh=self.tanh) for i in range(3)]+[Fusion_CAT(256) for i in range(2)])
      
    def extract_feat(self, img):
        self.iter = self.iter + 1
        """Directly extract features from the backbone+neck."""
        v_img, t_img = img
        x, y = self.backbone(v_img, t_img)
        
        if self.with_neck:
            x = self.neck(x)
            y = self.nect_t(y)
      
        features = []       
        # Fusion
        for i in range(len(x)):
            feat = self.fuse[i](x[i], y[i])
            features.append(feat)
     
        return features


class Fusion_CAT(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2 * in_channels, in_channels, 1)

    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        temp = self.conv1x1(temp)
        return temp

class Fusion_CAT_WTA(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2 * in_channels, in_channels, 1)

    def forward(self, rgb, thermal):
        w1, w2 = winner_takes_all(rgb, thermal, k=5)
        rgb, thermal = rgb * w1, thermal * w2  
        temp = torch.cat((rgb, thermal), 1)
        temp = self.conv1x1(temp)
        return temp
    
class Fusion(nn.Module):
    def __init__(self, dim, tanh=False):
        super().__init__()
        self.dim = dim
        self.tanh = tanh
        self.Q_rgb = ModalityNorm(self.dim)
        self.Q_thermal = ModalityNorm(self.dim)
        self.K_rgb = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.K_thermal = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.V_rgb = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.V_thermal = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb, thermal):   
        _, _, H, W = rgb.shape
        rgb_Q = self.Q_rgb(rgb, thermal) 
        # rgb_Q = rgb_Q.float()
        thermal_Q = self.Q_thermal(thermal, rgb)  
        
        w1, w2 = winner_takes_all(rgb_Q, thermal_Q, k=5)
        rgb_Q, thermal_Q = rgb_Q * w1, thermal_Q * w2     
        rgb_K = self.K_rgb(rgb)
        rgb_V = self.V_rgb(rgb)
        thermal_K = self.K_thermal(thermal)
        thermal_V = self.V_thermal(thermal)

        B, C, H, W = rgb_Q.shape
        q_rgb, k_rgb, v_rgb = rgb_Q.view(B,C,-1), rgb_K.view(B,C,-1), rgb_V.view(B,C,-1)
        q_thermal, k_thermal, v_thermal = thermal_Q.view(B,C,-1), thermal_K.view(B,C,-1), thermal_V.view(B,C,-1)

        attn_rgb = (q_rgb.transpose(-2, -1) @ k_rgb) / (C ** 0.5)
        # attn_rgb = (q_rgb.transpose(-2, -1) @ k_rgb) 
        attn_rgb = self.softmax(attn_rgb)
        attn_thermal = (q_thermal.transpose(-2, -1) @ k_thermal) / (C ** 0.5)
        # attn_thermal = (q_thermal.transpose(-2, -1) @ k_thermal) 
        attn_thermal = self.softmax(attn_thermal)

        rgb_out = (v_rgb @ attn_rgb).view(B, C, H, W) + rgb
        if self.tanh:
            rgb_out = torch.tanh(rgb_out)  # 防止梯度爆炸
        thermal_out = (v_thermal @ attn_thermal).view(B, C, H, W) + thermal
        if self.tanh:
            thermal_out = torch.tanh(thermal_out)  # 防止梯度爆炸

        out = rgb_out + thermal_out
        if self.tanh:
            out = torch.tanh(out)  # 防止梯度爆炸
        return out
    
class ModalityNorm(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(ModalityNorm, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                      nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear')
            gamma = self.conv_gamma(x)
            # gamma = F.interpolate(gamma, scale_factor=2, mode='bilinear')
            beta = self.conv_beta(x)
            # beta = F.interpolate(beta, scale_factor=2, mode='bilinear')

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out