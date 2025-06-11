# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix

# | 参数名                     | 含义                                     |
# | ----------------------- | -------------------------------------- |
# | `data`                  | 当前 batch 的元信息（相机矩阵、图像分辨率、中心点等）         |
# | `human_gs_out`          | 人体高斯的属性输出（位置、旋转、颜色等）                   |
# | `scene_gs_out`          | 场景高斯的属性输出（可以为 None）                    |
# | `render_mode`           | 当前渲染模式：`human`、`scene` 或 `human_scene` |
# | `bg_color`              | 背景颜色（用于渲染空白区域）                         |
# | `render_human_separate` | 是否在融合渲染的同时也单独渲染人体（供 supervision 用）     |
def render_human_scene(
    data, 
    human_gs_out,
    scene_gs_out,
    bg_color, 
    human_bg_color=None,
    scaling_modifier=1.0, 
    render_mode='human_scene',
    render_human_separate=False,
):

    # Step 1：根据 render_mode 合并高斯
    feats = None
    if render_mode == 'human_scene':
        feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')

    # 这个 render() 函数使用 Apple 提供的 diff_gaussian_rasterization 渲染 CUDA 核心，执行高斯点云的投影、splatted 渲染、SH shading 等。
    # render_pkg 包括：
    #     'render': 渲染图像
    #     'visibility_filter': 哪些高斯被看到
    #     'radii': 每个高斯在屏幕上的半径大小
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )

    # 🔹 Step 3：如果需要，额外单独渲染 human 图像
    if render_human_separate and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']

# Step 4：设置各部分的可见性信息
# 在 human_scene 模式中，我们需要从混合的 visibility/radii 中分离出：
# 人体部分的 visibility
# 场景部分的 visibility
    # 目的是后续 densify() 时分开处理人体和场景高斯。
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
            
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']
        
    return render_pkg


# means3D: 每个高斯的中心点，shape (N, 3)
# feats: 高斯的颜色或球谐系数，shape (N, 3) 或 (N, 3×(degree+1)²)
# opacity: 每个高斯的不透明度，shape (N, 1)
# scales: 高斯尺度（半径、协方差、轴长等）
# rotations: 高斯旋转（四元数或矩阵）
# data: 字典，包含相机内参、变换矩阵、图像尺寸等
# scaling_modifier: 调整高斯投影尺寸的系数
# bg_color: 背景颜色
# active_sh_degree: 使用球谐表示时的阶数（0 表示 RGB）
def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    
    # 初始化一个与 means3D 相同 shape 的全零变量 screenspace_points，用于在渲染过程中记录高斯在屏幕空间的位置。它不是必要输入，而是为了计算视域信息（如遮挡、屏幕位置等）。
    # screenspace_points 是为了读取渲染后的屏幕坐标而设置的 hook 变量，不是为了控制或反推 means3D。
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # 🎥 第二步：相机参数准备
    # 计算水平/垂直视场角的一半的正切，用于 FOV 投影变换。
    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    # 🎨 第三步：处理颜色 or 球谐系数
    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    # 🔧 第四步：配置高斯渲染器
    # 设置渲染器的参数，包括：
    #     分辨率（图像大小）
    #     FOV（投影范围）
    #     视图变换（世界→相机）和投影变换（相机→NDC）
    #     背景色、是否预滤波、是否启用调试
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    # 构造可微高斯渲染器对象。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 💡 第五步：调用渲染器
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )

    # 将像素值限制在 [0,1] 区间。
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    #     | 字段名                 | 含义                       |
    # | ------------------- | ------------------------ |
    # | `render`            | 最终图像，shape 为 `(3, H, W)` |
    # | `viewspace_points`  | 每个高斯的 2D 屏幕坐标（实际由渲染器写入）  |
    # | `visibility_filter` | 每个高斯是否被渲染器视锥覆盖           |
    # | `radii`             | 每个高斯在图像平面上的投影半径（越大越靠近相机） |
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }
