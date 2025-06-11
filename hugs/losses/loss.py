#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F

from hugs.utils.sampler import PatchSampler

from .utils import l1_loss, ssim

# 它结合了图像重建损失（L1、SSIM、LPIPS）、人体几何监督（LBS 权重回归）、以及人/场景分离监督等。
class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,
        l_lpips_w=0.0,
        l_lbs_w=0.0,
        l_humansep_w=0.0,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
    ):
        super(HumanSceneLoss, self).__init__()
        
        # ✅ 初始化传入的权重项：
        # l_ssim_w, l_l1_w, l_lpips_w：图像相似性损失的加权系数。
        # l_lbs_w：LBS 权重约束项的系数。        
        # l_humansep_w：在 human_scene 模式下用于分离人体的额外损失加权
        # num_patches, patch_size, use_patches：是否使用 PatchSampler 来进行 LPIPS 计算。
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w
        self.l_lpips_w = l_lpips_w
        self.l_lbs_w = l_lbs_w
        self.l_humansep_w = l_humansep_w
        self.use_patches = use_patches
        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
    
    # data: GT 图像、mask、bbox 等。
    # render_pkg: 渲染结果（完整图像 / 分离图像等）
    # human_gs_out: 高斯输出（含 LBS 权重）。
    # render_mode: 当前渲染类型：human / scene / human_scene。
    def forward(
        self, 
        data, 
        render_pkg,
        human_gs_out,
        render_mode, 
        human_gs_init_values=None,
        bg_color=None,
        human_bg_color=None,
    ):
        loss_dict = {}
        extras_dict = {}
        
        if bg_color is not None:
            self.bg_color = bg_color
            
        if human_bg_color is None:
            human_bg_color = self.bg_color
            
        gt_image = data['rgb']
        mask = data['mask'].unsqueeze(0)
        
        pred_img = render_pkg['render']

        #         | 渲染模式            | 操作说明                                              |
        # | --------------- | ------------------------------------------------- |
        # | `'human'`       | `gt = rgb × mask + bg_color × (1 - mask)`，只监督人体区域 |
        # | `'scene'`       | `gt = rgb × (1 - mask)`，只监督场景区域                   |
        # | `'human_scene'` | 使用原图整体进行监督                                        |
        if render_mode == "human":
            gt_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img
        elif render_mode == "scene":
            # invert the mask
            extras_dict['pred_img'] = pred_img
            
            mask = (1. - data['mask'].unsqueeze(0))
            gt_image = gt_image * mask
            pred_img = pred_img * mask
            
            extras_dict['gt_img'] = gt_image
        else:
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img
        
        if self.l_l1_w > 0.0:
            if render_mode == "human":
                Ll1 = l1_loss(pred_img, gt_image, mask)
            elif render_mode == "scene":
                Ll1 = l1_loss(pred_img, gt_image, 1 - mask)
            elif render_mode == "human_scene":
                Ll1 = l1_loss(pred_img, gt_image)
            else:
                raise NotImplementedError
            loss_dict['l1'] = self.l_l1_w * Ll1

        # 结构相似性
        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            if render_mode == "human":
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "scene":
                loss_ssim = loss_ssim * ((1 - mask).sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "human_scene":
                loss_ssim = loss_ssim
                
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim

        # 作用：感知层级上比较预测与 GT 图像是否一致（用 VGG 特征）。
        if self.l_lpips_w > 0.0 and not render_mode == "scene":
            if self.use_patches:
                if render_mode == "human":
                    bg_color_lpips = torch.rand_like(pred_img)
                    image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                    gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
                else:
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, pred_img, gt_image)
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            else:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_pred_img = pred_img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                loss_lpips = self.lpips(cropped_pred_img.clip(max=1), cropped_gt_image).mean()
                loss_dict['lpips'] = self.l_lpips_w * loss_lpips

        # 🧍‍♂️ 5. 人体区域专属监督（human_scene 模式）
        if self.l_humansep_w > 0.0 and render_mode == "human_scene":
            pred_human_img = render_pkg['human_img']
            gt_human_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            
            Ll1_human = l1_loss(pred_human_img, gt_human_image, mask)
            loss_dict['l1_human'] = self.l_l1_w * Ll1_human * self.l_humansep_w
            
            loss_ssim_human = 1.0 - ssim(pred_human_img, gt_human_image)
            loss_ssim_human = loss_ssim_human * (mask.sum() / (pred_human_img.shape[-1] * pred_human_img.shape[-2]))
            loss_dict['ssim_human'] = self.l_ssim_w * loss_ssim_human * self.l_humansep_w
            
            bg_color_lpips = torch.rand_like(pred_human_img)
            image_bg = pred_human_img * mask + bg_color_lpips * (1. - mask)
            gt_image_bg = gt_human_image * mask + bg_color_lpips * (1. - mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
            loss_lpips_human = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_human'] = self.l_lpips_w * loss_lpips_human * self.l_humansep_w

        # 比较网络预测的 LBS 权重 vs GT 权重（或初始权重）。
        # 常用于正则化控制，防止高斯绑定偏离人体结构。
        if self.l_lbs_w > 0.0 and human_gs_out['lbs_weights'] is not None and not render_mode == "scene":
            if 'gt_lbs_weights' in human_gs_out.keys():
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs'] = self.l_lbs_w * loss_lbs
        
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict
    
