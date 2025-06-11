# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import numpy as np
from torch import nn
from loguru import logger
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement

from hugs.utils.spherical_harmonics import RGB2SH
from hugs.utils.general import (
    inverse_sigmoid, 
    get_expon_lr_func, 
    build_rotation,
    strip_symmetric,
    build_scaling_rotation,
)

# 🧱 初始化流程 create_from_pcd(pcd, spatial_lr_scale)
# 给定一个 Open3D 点云对象，初始化所有高斯属性：

# 🚀 forward 输出（供渲染器使用）

# 🌱 优化器 setup
# setup_optimizer(cfg) 为每个属性构建优化项并记录其学习率：

# 🌐 稠密化与裁剪 densify_and_*
# densify_and_split: 基于梯度大的点 clone 并 perturb，增加细节
# densify_and_clone: 直接复制原有点（适合不需要变化时）
# densify_and_prune: 结合最大透明度、投影尺寸裁剪点云
# prune_points: 对应张量和 optimizer 状态同步裁剪
# cat_tensors_to_optimizer: 添加新点同步到优化器中

# 💾 存取接口
# save_ply(path): 保存当前状态为 .ply 格式
# load_ply(path): 从 .ply 加载全部属性
# state_dict(): 提供可序列化的模型参数
# restore(state_dict, cfg): 用于 checkpoint 加载恢复状态

class SceneGS:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, only_rgb: bool=False):
        self.only_rgb = only_rgb
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
    
    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return save_dict
    
    def restore(self, state_dict, cfg):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self._features_dc = state_dict['features_dc']
        self._features_rest = state_dict['features_rest']
        self._scaling = state_dict['scaling']
        self._rotation = state_dict['rotation']
        self._opacity = state_dict['opacity']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def __repr__(self):
        repr_str = "SceneGS: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "features_dc: {} \n".format(self._features_dc.shape)
        repr_str += "features_rest: {} \n".format(self._features_rest.shape)
        repr_str += "scaling: {} \n".format(self._scaling.shape)
        repr_str += "rotation: {} \n".format(self._rotation.shape)
        repr_str += "opacity: {} \n".format(self._opacity.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.only_rgb:
            return features_dc.squeeze(1)
        else:
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def forward(self):
        gs_scales = self.scaling_activation(self._scaling)
        gs_rotation = self.rotation_activation(self._rotation)
        gs_xyz = self._xyz
        gs_opacity = self.opacity_activation(self._opacity)
        gs_features = self.get_features
        return {
            'xyz': gs_xyz,
            'scales': gs_scales,
            'rotq': gs_rotation,
            'shs': gs_features,
            'opacity': gs_opacity,
            'active_sh_degree': self.active_sh_degree,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 🧱 初始化流程 create_from_pcd(pcd, spatial_lr_scale)
    # 给定一个 Open3D 点云对象，初始化所有高斯属性：
    # ✅ 输入：
    #     pcd: 一个 open3d.geometry.PointCloud，含有：
    #         points: N×3，点的位置
    #         colors: N×3，RGB 颜色（范围 0~1）
    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # → 如果启用 SH（球谐），则将 RGB 转为球谐基系数（用于方向感知的颜色建模）
        if self.only_rgb:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # 1. 颜色 → 球谐系数（Spherical Harmonics）
        # features_dc: [..., 0]，表示每个点的 RGB 常量分量
        # features_rest: [..., 1:]，高阶球谐分量（先置0，训练中学习
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (N, RGB通道, SH通道数)
        features[:, :3, 0 ] = fused_color  # 填入直流项 DC：对应 SH 的第0阶（常数项）
        features[:, 3:, 1:] = 0.0   # 其余阶全部置零
        
            
        logger.info(f'Number of scene points at initialisation: {fused_point_cloud.shape[0]}')

        # 3️⃣ 初始化 scale 参数（控制高斯形状大小）
        # 对应协方差矩阵主轴方向的 scale，初始值越远，越“模糊”
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)   # 最近邻距离平方，避免为0
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)  # 取 log√d 作为尺度（三轴共享）

        # 4️⃣ 初始化 rotation 为单位四元数（无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 5️⃣ 初始化 opacity 透明度
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 6️⃣ 注册为可训练参数
        # 初始化时高斯的位置（_xyz）和输入点云的坐标完全一致。
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # 7️⃣ 初始化 2D 投影半径（用于渲染时判断遮挡/密度）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': cfg.feature, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': cfg.feature / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': cfg.opacity, "name": "opacity"},
            {'params': [self._scaling], 'lr': cfg.scaling, "name": "scaling"},
            {'params': [self._rotation], 'lr': cfg.rotation, "name": "rotation"}
        ]
        
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init * self.spatial_lr_scale,
            lr_final=cfg.position_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    # 📌 作用
    # 将新的张量（如新生成的高斯位置、颜色等）拼接（cat）到已有的张量后面，并将它们注册为可训练参数，更新优化器的参数状态字典（state_dict），避免梯度错误或优化器崩溃。
    def cat_tensors_to_optimizer(self, tensors_dict):
        # tensors_dict: 是一个字典，包含了待添加的张量（如上方 d）
        optimizable_tensors = {}
        # 遍历 self.optimizer.param_groups 中的参数组：
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # 取出该参数对应的新张量（如新位置、新颜色）：
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            # 如果该参数有优化器状态（如 Adam 的 exp_avg, exp_avg_sq），就为新张量创建对应状态向量（全 0）并拼接到原来的状态上：
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 更新 param_group 的参数，将旧参数 + 新张量拼接后的张量注册为 nn.Parameter：
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
        
    # 📌 作用
    # 将新生成的高斯点（位置、颜色、缩放、旋转、透明度等）正式注册到模型中，并同步到优化器状态，供后续训练使用。
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        
        # 把新数据打包后，调用 cat_tensors_to_optimizer 添加到已有 tensor 里，并更新优化器管理的参数列表。
        # 随后更新模型中的核心字段：
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 并重置相关累积张量：
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 这些是 每个点的累积梯度、计数器、最大投影尺寸，必须重新初始化以适应新的点。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    
        # | 维度   | `densify_and_clone` | `densify_and_split` |
        # | ---- | ------------------- | ------------------- |
        # | 类型   | 复制                  | 扰动（采样 + perturb）    |
        # | 点位置  | 不变，直接复制             | 在原点附近扰动采样           |
        # | 使用场景 | 粗糙密度增加              | 更高细节建模              |
        # | 参数 N | 无需                  | 通常设置 N=2 或更多        |
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 这一步选出梯度较大的点（通常表示这些点对最终图像影响较大）。
        # 然后进一步筛选出这些点中“当前足够稀疏的点”：
        # 含义是：
        #     ✅ 梯度足够大 且
        #     ✅ 当前的高斯尺度（范围）较小，说明尚未填满周围空间 → 可以 clone 更多点来丰富细节。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 🧪 3. 对选中的点做扰动采样（Gaussian jitter）
        # 以选中的高斯点为中心，在局部空间中生成 N 个扰动样本（用的是当前 scale 大小为 std）
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")

        # 这一操作的结果是用来扰动原始点：
        # 这里：
        #     samples: 每个点周围采样出来的随机扰动向量（高斯分布）
        #     rots: 把这些扰动从局部空间旋转到世界空间
        #     bmm(...) + ...: 得到新高斯点的位置
        # 也就是说，这是 densify_and_split 实现「在原点周围扰动采样」的关键步骤。
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 📍 4. 计算新高斯点位置（被扰动过的）
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # 🧩 6. 复制其他属性（颜色、透明度等）
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        
        # 🧩 2. 对这些点进行复制（clone）
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        #
        # ✂️ 8. 删除旧点（被分裂的原点）
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    #     | 参数名              | 说明                               |
    # | ---------------- | -------------------------------- |
    # | `grads`          | 每个高斯点的屏幕空间梯度（reflect importance） |
    # | `grad_threshold` | 复制的触发阈值（梯度大于此值才考虑 densify）       |
    # | `scene_extent`   | 当前场景尺寸（用于过滤太大的点）                 |
    # 复制那些重要性高、但物理尺寸不大的高斯点，提升密度与细节还原能力。
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 对每个点求其梯度的 L2 范数；
        # 找出大于设定阈值的点作为候选。
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 再加一重限制：只有物理尺寸不太大的点才复制；
        # self.get_scaling.max(dim=1) 提取每个点的最大尺度；
        # 如果高斯已经覆盖了场景很大区域，就不再复制。
        # 👉 确保只 densify“重要而紧凑”的点。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # ✨ 复制点属性
        # 对所有选中的点，直接复制所有属性，包括位置、颜色、尺度、旋转、透明度等；
        # 没有扰动，没有变化 —— 保持一致性；
        new_xyz = self._xyz[selected_pts_mask]
        # _features_dc 和 _features_rest 是高斯点用于颜色建模的关键特征参数，它们共同构成了每个点的球谐系数（Spherical Harmonics, SH）
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # ➕ 添加进优化器 & 参数管理
        # 将新复制的点添加进优化参数；
        # 同时更新梯度缓存、max_radii 等辅助结构；
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 控制了 高斯点的动态增密与裁剪。
    #     | 参数名               | 含义                               |
    # | ----------------- | -------------------------------- |
    # | `max_grad`        | 梯度阈值：大于此值才考虑进行复制 densify         |
    # | `min_opacity`     | 最小透明度阈值：小于这个值就会 prune 掉该点        |
    # | `extent`          | 整个场景范围（影响 pruning 和 densify）     |
    # | `max_screen_size` | 最大屏幕半径阈值，用于裁剪屏幕上占比过大的点（通常为近距离大点） |
    # | `max_n_gs`        | 高斯最大数量上限（防止 OOM）                 |
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
        
        # 🔁 第一步：计算梯度强度
            # xyz_gradient_accum 是每个点累计的屏幕空间梯度强度；
            # denom 是参与次数（用于归一化）；
            # 最后将无效项置为 0（NaN 表示从未更新）；
        # 👉 得到每个高斯点对损失的平均梯度影响强度。
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 🔒 第二步：控制最大高斯数
        # 如果未设置上限，默认允许继续增加（+1 是为了保证下一行条件成立）；
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1

        # 🌱 第三步：触发增密（densify）
        # 🔁 细节：
            # densify_and_clone(...)：复制关键点，用于整体密度提升；
            # densify_and_split(...)：扰动复制点，用于细节增加与局部调整。
        # 只有在当前高斯数未超过设定上限 max_n_gs 时才执行，防止无限膨胀。
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        # ✂️ 第四步：剪枝（prune）
        # 初始规则是：透明度太低（贡献很小）的点直接标记为裁剪对象。
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # 👉 这些点要么太远、太近、太透明，不利于高效建模，直接裁掉。
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 🧹 第五步：执行剪枝
            # 删除不再使用的点；
            # 同步 optimizer 中相关参数；
            # 更新梯度统计变量；
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    
    # 📊 为什么要记录这些信息？
    # 这些累积梯度统计用于判断高斯点是否应该 densify（复制细化）：
    # 梯度大 → 说明该点对图像误差贡献大 → 应该复制更多细节。
    # 梯度小 → 表示该点不重要，甚至可能被剪枝。
    
    # viewspace_point_tensor: 来自渲染器输入的 screenspace_points，被设为 requires_grad=True。训练中，它的 .grad 储存了每个高斯点在屏幕上的梯度（通常来源于像素 loss 的反向传播）。
    # update_filter: 一个布尔张量，指示哪些点是当前视角中被渲染（或在视锥内）的。
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 提取前两个维度（:2）：只考虑屏幕 x/y 方向的梯度，因为屏幕上是 2D。
        # 累加 L2 范数：对屏幕梯度进行范数计算（衡量该高斯对最终图像影响强度），然后逐点加到 xyz_gradient_accum 中。
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        # 每次统计后也累加一次分母，用于归一化平均（后续 densify_and_prune() 中使用）：
        self.denom[update_filter] += 1
        
