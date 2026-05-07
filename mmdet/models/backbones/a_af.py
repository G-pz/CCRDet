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

from tools.feature_analyze import save_feature_to_img


def compute_brightness_map(v_img):
    mean = np.array([115.37, 121.82, 122.63])
    std = np.array([85.13, 89.01, 88.27])
    # mean = np.array([83.20, 92.24, 97.70])
    # std = np.array([134.84, 134.84, 134.84])
   
    image = v_img.cpu().detach().numpy()
    brightness_map = []
    for i in range(len(image)):
        img = np.transpose(image[i], (1, 2, 0))
        img = img * std + mean
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        brightness_map_ = yuv_img[:, :, 0]  
        
        brightness_map_ = cv2.normalize(brightness_map_, None, 0, 255, cv2.NORM_MINMAX)
        brightness_map_ = brightness_map_/255
        brightness_map.append(brightness_map_)
        
    brightness_map = np.array(brightness_map)
    brightness_map = torch.from_numpy(brightness_map)
    # return v_img
    
    return brightness_map

class IPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_f = nn.Conv2d(1, 1, 1)
        self.conv_bright_map = nn.Conv2d(1, 1, 3, padding=1)
        self.conv_fuse = nn.Conv2d(2, 1, 3, padding=1)          

    def forward(self, rgb, bright_map):   
        _, _, H, W = rgb.shape
        bright_map = bright_map.unsqueeze(1)
        bright_map = F.interpolate(bright_map, size=(H, W), mode='bilinear')
        bright_map = bright_map.to(rgb.device)
        bright_map = bright_map.float()
        avg_out = torch.mean(rgb, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # weight = torch.cat([avg_out, max_out], dim=1)
        weight = self.conv_f(avg_out)
        bright = self.conv_bright_map(bright_map)
        weight_rgb = torch.concat((weight, bright), dim=1)
        weight_rgb = self.conv_fuse(weight_rgb)
        weight_rgb = F.sigmoid(weight_rgb)
        # weight_rgb = torch.pow(weight_rgb, 3)
        out = rgb * (1 + weight_rgb)     

        return out, weight_rgb, bright

class LSKblock(nn.Module):
    def __init__(self, dim, dim_in):
        super().__init__()
        self.conv0 = nn.Conv2d(dim_in, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 1, 7, padding=3)        

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        attn = self.conv_squeeze(agg).sigmoid()
        # attn = self.conv_squeeze(agg)
        out = x * (1 + attn)
          
        return out, attn, attn1, attn2

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
        self.ipm = nn.ModuleList()
        self.lsk = nn.ModuleList()
        self.thermal_convlist1 = nn.ModuleList()
        self.thermal_convlist2 = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.ipm.append(IPM())
            self.lsk.append(LSKblock(self.dims_lsk[i], self.dims_in[i]))

        for i in range(self.num_stages):
            self.thermal_convlist1.append(
                nn.Sequential(nn.Conv2d(dims_in[i], dims_lsk[i], 1, 1), nn.ReLU()))
            self.thermal_convlist2.append(
                nn.Sequential(nn.Conv2d(dims_lsk[i], dims_in[i], 1, 1), nn.ReLU()))


    # forward function
    def forward(self, rgb_x, thermal_x):
        self.iter = self.iter + 1
        bright_map = compute_brightness_map(rgb_x)
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
            # save_feature_to_img(rgb_x, 'rgb_in', self, iter, )
            # save_feature_to_img(thermal_x, 'thermal_in', i)

            if i < 3:

                rgb_out, weight_rgb, bright = self.ipm[i](rgb_x, bright_map)
                thermal_out, weight_t, attn1, attn2 = self.lsk[i](thermal_x)
                               
                # save_feature_to_img(rgb_x, 'rgb_in'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(thermal_x, 'thermal_in'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(weight_rgb, 'rgb_weight'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(bright, 'bright_map'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(weight_t, 'thermal_weight'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(attn1, 'attn1'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(attn2, 'attn2'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')


                # rgb_out = rgb_x + rgb_out*(1-weight_t)
                # thermal_x = thermal_x + thermal_out*(1-weight_rgb)   
                #              
                rgb_x = rgb_x*(1+weight_t)
                thermal_x = thermal_x*(2-weight_rgb)  
                # rgb_x = rgb_x + rgb_out
                # thermal_x = thermal_x + thermal_x*(1-weight_rgb)
                # rgb_x = rgb_x + rgb_x*(1-weight_t)
                # thermal_x = thermal_x + thermal_out 
                # save_feature_to_img(rgb_x, 'rgb_out'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')
                # save_feature_to_img(thermal_x, 'thermal_out'+ str(i), self.iter, output_dir = '/home/gpz/Pictures/features/afdet_module12')           
                # rgb_x = rgb_x * (1+(1-weight_t)*0.5)
                # thermal_x = thermal_x * (1+(1-weight_rgb)*0.5)

            if i in self.out_indices:
                # save_feature_to_img(rgb_x, 'rgb_out', i)
                # save_feature_to_img(thermal_x, 'thermal_out', i)
                outs_rgb.append(rgb_x)
                outs_thermal.append(thermal_x)

        return tuple(outs_rgb), tuple(outs_thermal)



