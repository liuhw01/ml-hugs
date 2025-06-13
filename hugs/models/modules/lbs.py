#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import NewType, Tuple, List

import torch
from smplx.lbs import (
    batch_rodrigues, 
    blend_shapes, 
    vertices2joints,
    batch_rigid_transform
    )

Tensor = NewType('Tensor', torch.Tensor)


# 功能：在 SMPL 基础上，用自定义的 LBS 权重和 pose-dirs 对一批点（v_shaped）做线性混合蒙皮（Linear Blend Skinning），并返回变形后顶点、关节变换、最终顶点变换矩阵等信息。
参数：
    # A: (B, J, 4, 4)，每个关节的齐次变换矩阵 (batch×joints)。
    # v_shaped: (B, N, 3)，要做蒙皮的 “点云” 或顶点位置；通常是带形状偏移后的模板顶点。
    # posedirs: (P, N*3) 或 (N, 3, P)，pose-dependent blendshape basis，用于生成关节驱动的偏移。
    # lbs_weights: (N, J)，每个点对每个关节的权重。
    # pose: (B, (J+1)*3) 或 (B, J+1, 3, 3)，SMPL 的关节旋转参数（axis-angle 或直接旋转矩阵）。
    # disable_posedirs: 是否跳过 pose‐dirs 偏移（只做刚性变换）。
    # pose2rot: 如果 True，把 pose 视作 axis-angle，要先转换到旋转矩阵；否则已是矩阵。
def lbs_extra(
    A: Tensor,  # (B, J, 4, 4) 每个关节的变换矩阵
    v_shaped: Tensor, # (B, N, 3) 要做LBS的点 
    posedirs: Tensor, # (P, N*3) pose blendshape basis
    lbs_weights: Tensor, # (N, J) 每个点对每个关节的权重
    pose: Tensor, # (B, (J+1)*3) 或 (B, J+1, 3, 3)
    disable_posedirs: bool = False,
    pose2rot: bool = True,
):

    batch_size = max(A.shape[0], A.shape[0])
    device, dtype = A.device, A.dtype

    # ident：3×3 单位矩阵，用于计算 pose 特征。
    ident = torch.eye(3, dtype=dtype, device=device)
    
    # pose_feature：去掉 root 关节后，把所有关节矩阵减去单位矩阵然后拼平，作为 pose-dependent 基座。
    if pose2rot:
        # 2.1 把 axis-angle 转为 3×3 旋转矩阵
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        # 2.2 生成 pose feature： (R_j − I) 拼成一长向量
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        # 2.3 用 pose_feature 和 posedirs 做矩阵乘，得到每点 pose 偏移
        if disable_posedirs:
            pose_offsets = torch.zeros_like(v_shaped)
        else:
            pose_offsets = torch.matmul(
                pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        # 如果 pose 已是旋转矩阵，略去 Rodrigues，直接做类似操作
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        if disable_posedirs:
            # pose_offsets：把 pose_feature 与 posedirs 相乘，得到每个顶点在当前姿态下的形状偏移 (B, N, 3)。
            pose_offsets = torch.zeros_like(v_shaped)
        else:
            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)
    
    if not disable_posedirs:
        # v_posed：带形变偏移的顶点位置，用于后续骨骼绑定。
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped
        
    # 5. Do skinning:
    # W is N x V x (J + 1)
    # 4.1 扩展 lbs_weights 到 batch 维
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    # 4.2 把 A 转为 (B, J, 16)，再按权重 W 做加权求和，重塑回 (B, N, 4, 4)
    num_joints = A.shape[1]
    # T：每个点的 LBS 变换矩阵 (B, N, 4, 4)，等于按权重把所有关节的 4×4 焊合起来。
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    # 5. 应用 T 变换到顶点
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    # v_posed_homo：给每个顶点加齐次坐标 → (x,y,z,1)。
    # torch.matmul(T, …)：用每个点的 4×4 变换矩阵左乘齐次坐标，得到变形后顶点。
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    
    # verts：最终变形后的顶点位置 (B, N, 3)。
    verts = v_homo[:, :, :3, 0]

        # verts：变形后顶点位置 (B, N, 3)
        # A：输入的关节变换 (B, J, 4, 4)
        # T：每点的 LBS 变换矩阵 (B, N, 4, 4)
        # v_posed：加上 pose_offsets 后的点位置 (B, N, 3)
        # v_shaped：原始输入点 (B, N, 3)
    return verts, A, T, v_posed, v_shaped


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
    disable_posedirs: bool = False,
    vert_offsets: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    shape_offsets = blend_shapes(betas, shapedirs)
    v_shaped = v_template + shape_offsets

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        if disable_posedirs:
            pose_offsets = torch.zeros_like(v_shaped)
        else:
            pose_offsets = torch.matmul(
                pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        if disable_posedirs:
            pose_offsets = torch.zeros_like(v_shaped)
        else:
            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)

    if not disable_posedirs:
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped
        
    if vert_offsets is not None:
        v_posed = v_posed + vert_offsets
            
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, T, v_posed, v_shaped, shape_offsets, pose_offsets
