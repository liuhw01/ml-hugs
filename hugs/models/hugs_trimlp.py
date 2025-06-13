#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import trimesh
from torch import nn
from loguru import logger
import torch.nn.functional as F
from hugs.models.hugs_wo_trimlp import smpl_lbsmap_top_k, smpl_lbsweight_top_k

from hugs.utils.general import (
    inverse_sigmoid, 
    get_expon_lr_func, 
    strip_symmetric,
    build_scaling_rotation,
)
from hugs.utils.rotations import (
    axis_angle_to_rotation_6d, 
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    quaternion_multiply,
    quaternion_to_matrix, 
    rotation_6d_to_axis_angle, 
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.subdivide_smpl import subdivide_smpl_model

from .modules.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.triplane import TriPlane
from .modules.decoders import AppearanceDecoder, DeformationDecoder, GeometryDecoder


SCALE_Z = 1e-5

# 初始化时会调用：
# TriPlane 三平面体素编码器
# AppearanceDecoder 外观解码器（生成 SH 系数和透明度）
# GeometryDecoder 几何解码器（生成偏移、旋转、尺度）
# DeformationDecoder 形变解码器（生成 LBS 权重和 posedirs）



class HUGS_TRIMLP:

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
        
    # | 参数名                                | 作用                                  |
    # | ---------------------------------- | ----------------------------------- |
    # | `sh_degree`                        | 最大 spherical harmonics 阶数，用于建模高斯的颜色 |
    # | `n_subdivision`                    | SMPL 模板网格细分次数，增加高斯点数                |
    # | `init_2d`, `use_surface`           | 控制高斯初始化方式：是否用2D特征或表面法向做限制           |
    # | `use_deformer`, `disable_posedirs` | 是否使用 SMPL 的 LBS（线性 blendshape）机制    |
    # | `triplane_res`, `n_features`       | 三平面特征分辨率及通道数，用于体积编码                 |
    # | `betas`                            | SMPL shape 参数初始值                    |
    def __init__(
        self, 
        sh_degree: int, 
        only_rgb: bool=False,
        n_subdivision: int=0,  
        use_surface=False,  
        init_2d=False,
        rotate_sh=False,
        isotropic=False,
        init_scale_multiplier=0.5,
        n_features=32,
        use_deformer=False,
        disable_posedirs=False,
        triplane_res=256,
        betas=None,
    ):
        self.only_rgb = only_rgb
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self.scaling_multiplier = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = 'cuda'
        self.use_surface = use_surface
        self.init_2d = init_2d
        self.rotate_sh = rotate_sh
        self.isotropic = isotropic
        self.init_scale_multiplier = init_scale_multiplier
        self.use_deformer = use_deformer
        self.disable_posedirs = disable_posedirs
        
        self.deformer = 'smpl'
        
        if betas is not None:
            self.create_betas(betas, requires_grad=False)
        
        self.triplane = TriPlane(n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res).to('cuda')
        self.appearance_dec = AppearanceDecoder(n_features=n_features*3).to('cuda')
        self.deformation_dec = DeformationDecoder(n_features=n_features*3, 
                                                  disable_posedirs=disable_posedirs).to('cuda')
        self.geometry_dec = GeometryDecoder(n_features=n_features*3, use_surface=use_surface).to('cuda')
        
        if n_subdivision > 0:
            logger.info(f"Subdividing SMPL model {n_subdivision} times")
            # 生成更密集的高斯初始化点，通过网格细分让高斯点分布更均匀、更细致，从而提升渲染质量与形变表达能力。
            # subdivide_smpl_model(...)  ⟶  _subdivide_smpl_model(...)  ⟶  subdivide(...)
            self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=n_subdivision).to(self.device)
        else:
            self.smpl_template = SMPL(SMPL_PATH).to(self.device)
            
        self.smpl = SMPL(SMPL_PATH).to(self.device)
            
        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(), 
            faces=self.smpl_template.faces, process=False
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        
        self.init_values = {}
        self.get_vitruvian_verts()
        
        self.setup_functions()
    
    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 23*6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")
        
    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")
        
    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")
        
    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")
        
    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'triplane': self.triplane.state_dict(),
            'appearance_dec': self.appearance_dec.state_dict(),
            'geometry_dec': self.geometry_dec.state_dict(),
            'deformation_dec': self.deformation_dec.state_dict(),
            'scaling_multiplier': self.scaling_multiplier,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return save_dict
    
    def load_state_dict(self, state_dict, cfg=None):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        
        self.triplane.load_state_dict(state_dict['triplane'])
        self.appearance_dec.load_state_dict(state_dict['appearance_dec'])
        self.geometry_dec.load_state_dict(state_dict['geometry_dec'])
        self.deformation_dec.load_state_dict(state_dict['deformation_dec'])
        self.scaling_multiplier = state_dict['scaling_multiplier']
        
        if cfg is None:
            from hugs.cfg.config import cfg as default_cfg
            cfg = default_cfg.human.lr
            
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")
            
    def __repr__(self):
        repr_str = "HUGS TRIMLP: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str

    def canon_forward(self):
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        gs_opacity = appearance_out['opacity']
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
            
        return {
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'rot6d_canon': gs_rot6d,
            'shs': gs_shs,
            'opacity': gs_opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
        }

    def forward_test(
        self,
        canon_forward_out,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        xyz_offsets = canon_forward_out['xyz_offsets']
        gs_rot6d = canon_forward_out['rot6d_canon']
        gs_scales = canon_forward_out['scales']
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = canon_forward_out['opacity']
        gs_shs = canon_forward_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            lbs_weights = canon_forward_out['lbs_weights']
            posedirs = canon_forward_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        # 构造一个与 gs_xyz 同 shape 的向量组，所有法向初始设为 (0,0,1)，表示 canonical 状态下的“上”方向。
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0

        # canon_normals：将默认法向通过 canonical 旋转 gs_rotmat 变换，得到 canonical 姿态下每个高斯的真实法向。
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        # deformed_normals：再用最终的 posed 旋转 deformed_gs_rotmat 变换，得到变形后每个高斯的法向。
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        # 由于外部变换只影响空间位置和旋转，不改变 spherical harmonics 系数，直接克隆一份即可。
        deformed_gs_shs = gs_shs.clone()


        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }
         
    def forward(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        # tri_feats.shape=(N, 3×features)
        tri_feats = self.triplane(self.get_xyz)

        # return {'shs': shs, 'opacity': opacity}
        appearance_out = self.appearance_dec(tri_feats)
        
        # return {
            # 'xyz': xyz,
            # 'rotations': rotations,
            # 'scales': scales,
        # }

# 🔹阶段一：规范空间高斯构建（Canonical Gaussians）
        geometry_out = self.geometry_dec(tri_feats)
        
        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = appearance_out['opacity']
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()

# ❗ 关键区别（能力 vs 效果）
# | 方面                                    | `smpl_lbsmap_top_k` | `lbs_extra`   |
# | ------------------------------------- | ------------------- | ------------- |
# | **LBS 权重来源**                          | SMPL自带的固定模板           | 网络可学习         |
# | **pose-induced deformation**（姿态引起的形变） | ❌ 无（刚性）             | ✅ 有（非线性响应）    |
# | **posedirs 使用**                       | ❌ 未使用               | ✅ 使用          |
# | **变形精度**                              | 仅仿射刚性变换             | 动作驱动的柔性形变     |
# | **控制自由度**                             | 无法学习优化（只是应用）        | 可训练、可微、可精调    |
# | **应用场景**                              | baseline、初始化        | 真实驱动/动画/拟人化效果 |


# 🔹阶段二：是否使用 SMPL LBS 动作驱动，后续送入lbs_extra
# 使用 deformation_decoder 解码得到：
#     lbs_weights：高斯点对 SMPL 各骨骼的 LBS 权重
#     posedirs：高斯点在骨骼旋转下的形变方向
        if self.use_deformer:
            # return {
            #     'lbs_weights': lbs_weights,
            #     'posedirs': posedirs if not self.disable_posedirs else None,
            # }
            # 对于每个高斯点，它对 SMPL 模型中 各个关节 的线性混合权重（Linear Blend Skinning）是多少。
            #     lbs_weights.shape = [N, 24]（如果 SMPL 是 24 个关节）
            #     用于后续计算：
            #         deformed_xyz = lbs_extra(..., lbs_weights, posedirs, ...)
            #         ！！→ 把静态的 xyz 通过骨骼姿态变成动态位置。！！
            # ⚠️ 不是高斯偏移的原因是：
                # 高斯偏移已经在 geometry_out['xyz'] 中输出；
                # deformation_out 专注于“如何跟随 SMPL 动起来”的部分；
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            
            # 对 LBS 权重做归一化，因为 LBS 是线性加权平均，权重之和必须为 1。
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            
            # posedirs: 高斯点在关节旋转下的局部偏移响应方向
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]

# 🔹阶段三：获取 SMPL 的当前姿态参数
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        # 最终，smpl_output 包含：
        #     vertices：(1, V, 3) 顶点坐标
        #     A、T：关节变换矩阵：： 形状：A 是一个张量，通常大小 (J, 4, 4)，J 是关节数（SMPL 中为 24 或 23）。：： 形状：T 是一个张量，大小 (V, 4, 4)，V 是网格顶点数。
        #     shape_offsets、pose_offsets：blendshape 偏移
        #     full_pose：23×3 轴角表示
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )

        # — 先初始化真实权重变量，用于后面计算 gt_lbs_weights（ground-truth LBS weights）。
        gt_lbs_weights = None

        if self.use_deformer:
            
            # 取出 SMPL 输出中第 0（batch）帧的关节变换矩阵 A，形状 (J,4,4)。
            A_t2pose = smpl_output.A[0]

            # 将模板（Vitruvian）→T-pose的逆变换 inv_A_t2vitruvian 与当前姿态关节变换 A_t2pose 相乘，
            # 得到 “Vitruvian → posed” 的每个关节变换。
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian


            #### 变换位置和旋转到 posed space   lbs_extra
            # | 属性        | 来源                                   | 是否变换到 posed space |
            # | --------- | --------------------------------------- | -----------------——   |
            # | `xyz`     | `self.get_xyz + xyz_offsets → lbs → ✔`  | ✅ 是                  |
            # | `scales`  | `geometry_out['scales'] * smpl_scale`   | ✅ 跟随位移缩放          |
            # | `rotmat`  | `lbs_T[:, :3, :3] @ canonical_rotation` | ✅ 是                  |
            # | `shs`     | `appearance_out['shs']` → 不变           | ❌ 不需变（假定不随姿态） |
            # | `opacity` | `appearance_out['opacity']`             | ❌ 不变                |
            # 输入
            #     A_vitruvian2pose[None]：每个关节“模板→姿态”的 4×4 变换
            #     gs_xyz[None]：要做骨骼绑定的高斯点位置，xyz_offsets后的高斯点，如果 gs_xyz 的 shape 是 (N, 3)，那么 gs_xyz[None] 就会变成 shape (1, N, 3)。
            #     posedirs：每个点在关节旋转下的位移方向集合
            #     lbs_weights：预测得到的每个点对每个关节的权重
            #     smpl_output.full_pose：SMPL 的当前轴角参数
            # 输出
            #     deformed_xyz：骨骼绑定后高斯点的新位置 (1, N, 3)
            #     lbs_T：每个点的最终 4×4 LBS 变换矩阵 (1, N, 4, 4)
            #     中间还有其他返回值，用 _ 忽略。
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                
                # 在不计算梯度的情况下，计算“真实”LBS 权重 gt_lbs_weights，用于做正则化损失。
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)

                # 确认 gt_lbs_weights 在每个点上总和为 1，否则打警告。
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            # SMPL 作形变时的模板偏移：shape_offsets + pose_offsets，取第 0 帧。
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t
            
# 🔹阶段四：执行 LBS，计算变形高斯位置与旋转（Motion-aware Gaussians），基于lbs_weights参数
            # 根据 SMPL 自带的 lbs_weights，对 gs_xyz 上的每个点：            
            # 使用 T_vitruvian2pose（每顶点的 4×4 变换）
            # 计算其对点的 LBS 变换矩阵 lbs_T（选取 top-K 骨骼以提高效率）
            # 返回 lbs_T：形状 (1, N, 4, 4) → .squeeze(0) 得 (N,4,4)。
            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            
            # 对每个点，把 (4×4) 的 lbs_T 矩阵左乘其齐次坐标，得到变形后位置 (x',y',z',w')，取前三维 (x',y',z') 即 deformed_xyz。
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        # 全身缩放：smpl_scale 通常是一个形如 (3,) 或 (1,) 的张量，表示对整个 SMPL 变形结果做全局缩放。
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        # 全身平移：transl 是一个 (3,) 的平移向量，加上它会将所有点沿 XYZ 平移相同量。
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
            
# 🔹阶段六：更新旋转、法向、最终属性
        # 计算变形后每个高斯的旋转矩阵
        #     lbs_T 是每个点的 4×4 LBS 变换矩阵，取它的前 3×3 子矩阵 lbs_T[:, :3, :3] 得到点在 posed 状态下的刚性旋转。
        #     与 Canonical 状态下的旋转 gs_rotmat 相乘，可得到“先旋转到 canonical 朝向，再按 LBS 旋转到 posed 朝向”的复合旋转矩阵。
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        # 将复合旋转矩阵转换为四元数表示，便于后续插值或与其他旋转相乘。
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        # 应用外部仿射变换（可选）
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        # 构造一个与 gs_xyz 同 shape 的向量组，所有法向初始设为 (0,0,1)，表示 canonical 状态下的“上”方向。
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0

        # canon_normals：将默认法向通过 canonical 旋转 gs_rotmat 变换，得到 canonical 姿态下每个高斯的真实法向。
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        # deformed_normals：再用最终的 posed 旋转 deformed_gs_rotmat 变换，得到变形后每个高斯的法向。
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        # 由于外部变换只影响空间位置和旋转，不改变 spherical harmonics 系数，直接克隆一份即可。
        deformed_gs_shs = gs_shs.clone()


        #         | 状态                   | 说明                                                                                                             |
# | -------------------- | -------------------------------------------------------------------------------------------------------------- |
# | **Canonical**（规范化状态） | 顶点／高斯点处于 SMPL 模板定义的“标准”姿态下（通常是 Vitruvian pose 或 T-pose），只包含初始的形状偏移（shape\_offsets）和模型级别的模板参数，没有任何关节驱动的形变或外部变换。 |
# | **Deformed**（变形后状态）  | 顶点／高斯点应用了骨骼绑定（LBS）、pose-dirs、全局旋转缩放平移，甚至可能再叠加了一次外部仿射变换之后的位置。                                                   |

        # 返回字典
        #     xyz: 变形并应用所有缩放、平移和外部变换后的高斯中心
        #     xyz_canon: canonical（未变形）状态下的高斯中心
        #     xyz_offsets: 几何偏移（geometry decoder 输出）
        #     scales / scales_canon: 最终与 canonical 的尺度
        #     rotq / rotmat: 变形后高斯的旋转（四元数和矩阵）
        #     rotq_canon / rotmat_canon: canonical 姿态下的旋转
        #     rot6d_canon: canonical 的 6D 旋转表示
        #     shs: spherical harmonics 系数
        #     opacity: 透明度
        #     normals / normals_canon: 变形后与 canonical 下的法向
        #     active_sh_degree: 当前 SH 阶数
        #     lbs_weights, posedirs: 用于 LBS 的权重和偏移方向
        #     gt_lbs_weights: SMPL 原生权重的 ground-truth，用于正则化
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            logger.info(f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1

    @torch.no_grad()
    def get_vitruvian_verts(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        self.A_t2vitruvian = smpl_output.A[0].detach()
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.inv_A_t2vitruvian = torch.inverse(self.A_t2vitruvian)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0].detach()
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()
    
    @torch.no_grad()
    def get_vitruvian_verts_template(self):
        
        vitruvian_pose = torch.zeros(69, dtype=self.smpl_template.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl_template(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def initialize(self):
        # 1️⃣ 顶点位置初始化
        # 获取 SMPL 模板下 Vitruvian pose（即“标准姿态”）对应的所有顶点位置，作为高斯初始中心。
        t_pose_verts = self.get_vitruvian_verts_template()

        # 初始化每个高斯点的尺度倍率（可学习项）
        self.scaling_multiplier = torch.ones((t_pose_verts.shape[0], 1), device="cuda")

        # 初始化偏移为 0，意味着初始位置就是 SMPL Vitruvian 模板的顶点位置
        xyz_offsets = torch.zeros_like(t_pose_verts)
        # 初始化颜色， 所有高斯点颜色初始化为灰色 (0.5, 0.5, 0.5)
        colors = torch.ones_like(t_pose_verts) * 0.5

        # 初始化 spherical harmonics（SH）系数，维度为 [num_points, 16, 3]
            # 仅第 0 阶（常数项）为非零，设置为灰色
        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0 ] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()

        # 初始化每个高斯点的 3D 尺度为 0（将在下方计算）
        scales = torch.zeros_like(t_pose_verts)

        
        for v in range(t_pose_verts.shape[0]):
            # 查找与当前顶点 v 相连的所有边
            selected_edges = torch.any(self.edges == v, dim=-1)
            # 计算所有相连边的长度（用于估算局部几何大小）
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]], 
                dim=-1
            )
            # 缩放边长，用于控制初始化尺度大小（默认 init_scale_multiplier=0.5）
            selected_edges_len *= self.init_scale_multiplier

            # 设置高斯在 X 和 Y 方向的尺度，初始取最大边长，并取对数（因为后续会 exp 回去）
            scales[v, 0] = torch.log(torch.max(selected_edges_len))
            scales[v, 1] = torch.log(torch.max(selected_edges_len))
            
            if not self.use_surface:
                scales[v, 2] = torch.log(torch.max(selected_edges_len))
        
        if self.use_surface or self.init_2d:
            scales = scales[..., :2]
            
        scales = torch.exp(scales)
        
        if self.use_surface or self.init_2d:
            scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
            scales = torch.cat([scales, scale_z], dim=-1)

        import trimesh
        # ✅ 构建 trimesh 网格对象，用 SMPL 模板的 T-pose 顶点和面构建网格，用于计算每个顶点的表面法向。
            # t_pose_verts: 当前帧 SMPL 模板下的顶点坐标（Tensor）。
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)

        # ✅ 获取每个顶点的单位法向向量，并转为 CUDA tensor，以便后续在 GPU 上与高斯默认法向对齐。
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()

        # ✅ 构造所有高斯的默认初始法向向量，全设为 [0, 0, 1]（Z轴朝上），即假设初始高斯是垂直朝上的椭球体。
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0

        # ✅ 计算从默认法向 gs_normals 到真实表面法向 vert_normals 的旋转矩阵（每个高斯一个3×3矩阵）。
        # 用于将高斯从“朝上”姿态旋转对齐到网格真实法向。
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        # ✅ 将旋转矩阵转换为：
        #     rotq: 四元数表示（用于插值等任务）
        
        #     rot6d: 6D 旋转表示（避免欧拉角/Gimbal锁问题，适合网络学习）
        rotq = matrix_to_quaternion(norm_rotmat)
        rot6d = matrix_to_rotation_6d(norm_rotmat)

        # ✅ 存储原始默认法向（Z轴）为类成员，后续可用于对比和旋转。
        self.normals = gs_normals

        # ✅ 将默认法向通过旋转矩阵 norm_rotmat 变换为对齐后的法向，结果即为每个高斯点在初始位置下的真实法向。
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)

        # ✅ 初始化每个高斯点的不透明度（透明度）为常数 0.1，后续训练中可调。
        opacity = 0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda")

        # ✅ 获取 SMPL 模板中：
        #     posedirs: 每个顶点在关节旋转时的形变方向（用于 LBS 插值）
        #     lbs_weights: 每个顶点的 LBS 权重（对应骨骼的影响权重）
        #     用于高斯形变时驱动每个高斯点的运动。
        posedirs = self.smpl_template.posedirs.detach().clone()
        lbs_weights = self.smpl_template.lbs_weights.detach().clone()

        # ✅ 设置当前高斯点的数量为 SMPL 模板顶点数。
        self.n_gs = t_pose_verts.shape[0]

        # ✅ 设置高斯中心为 SMPL Vitruvian pose 的顶点坐标，并启用梯度更新，用于后续优化。
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))

         # ✅ 初始化每个高斯点在视图空间中的最大投影半径为0（用于视锥裁剪等渲染加速策略），后续渲染中会更新该值。
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        return {
            'xyz_offsets': xyz_offsets,
            'scales': scales,
            'rot6d_canon': rot6d,
            'shs': shs,
            'opacity': opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'deformed_normals': deformed_normals,
            'faces': self.smpl.faces_tensor,
            'edges': self.edges,
        }

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial
        
        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * cfg.smpl_spatial, "name": "xyz"},
            {'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'},
            {'params': self.geometry_dec.parameters(), 'lr': cfg.geometry, 'name': 'geometry_dec'},
            {'params': self.appearance_dec.parameters(), 'lr': cfg.appearance, 'name': 'appearance_dec'},
            {'params': self.deformation_dec.parameters(), 'lr': cfg.deformation, 'name': 'deform_dec'},
        ]
        
        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_pose, 'name': 'global_orient'})
        
        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})
            
        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})
            
        if hasattr(self, 'transl') and self.betas.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})
        
        self.non_densify_params_keys = [
            'global_orient', 'body_pose', 'betas', 'transl', 
            'v_embed', 'geometry_dec', 'appearance_dec', 'deform_dec',
        ]
        
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init  * cfg.smpl_spatial,
            lr_final=cfg.position_final  * cfg.smpl_spatial,
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
            if group["name"] in self.non_densify_params_keys:
                continue
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
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        
        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > self.percent_dense*scene_extent)
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values 
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        stds = scales[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N,1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N,1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N,1,1)
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(scales, dim=1).values <= self.percent_dense*scene_extent
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

    def densify_and_prune(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
