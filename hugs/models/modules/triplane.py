#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

EPS = 1e-3

class TriPlane(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        
        # ✅ 定义三张特征平面作为可学习参数，每张平面都是二维特征图：
        #     plane_xy: 投影到 XY 平面
        #     plane_xz: 投影到 XZ 平面
        #     plane_yz: 投影到 YZ 平面
        # 每个平面大小为 features × H × W，维度是 [1, C, H, W]。
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))

        # 记录输入维度（3D 坐标）、输出维度（拼接后的特征为 3 × features）。
        self.dim = features
        self.n_input_dims = 3
        self.n_output_dims = 3 * features

        # 输入坐标会被归一化到 [0,1]，这两个变量用来调节归一化范围：
        self.center = 0.0
        self.scale = 2.0

    def forward(self, x):
        
        # 将输入坐标从世界空间映射到 [0, 1] 区间（用于归一化插值）。
        x = (x - self.center) / self.scale + 0.5

        # ✅ 做数值安全检查，确保所有输入点都在 [0,1] 范围内（允许浮点误差）。
        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"

        # 将坐标归一化到 [-1, 1] 区间（是 F.grid_sample 要求的坐标范围）。
        x = x * 2 - 1
        shape = x.shape

        # 调整坐标维度以匹配 grid_sample 接口（需要 [N, 1, 3]）。
        coords = x.reshape(1, -1, 1, 3)

        
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        # ✅ 三张平面分别采样：
        # 这行代码是 TriPlane 编码的关键：从二维特征平面 self.plane_xy 中根据输入的 3D 点投影到 XY 平面，并用 grid_sample 插值获取特征。
        # 输入：
            # self.plane_xy: [1, C, H, W]，一个 shape 为 [1, features, resX, resY] 的二维特征平面。
            # coords: [1, N, 1, 3]，输入点坐标已经归一化到 [-1, 1]，代表 N 个三维点。我们只取 x 和 y 分量：
        # 使用坐标 coords[..., [0, 1]] 对 self.plane_xy 特征图进行双线性采样。
        # 最终得到：
            # feat_xy: shape [N, C]
            # 每一行是一个 3D 点在 XY 平面上投影采样得到的 C 维特征向量
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)
        
        # ✅ 输出每个输入 3D 点对应的三平面特征（注意：没有激活或额外映射层，仅提供空间编码）。
        return feat
