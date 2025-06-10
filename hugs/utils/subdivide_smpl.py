#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch 
import trimesh 
import numpy as np 
from trimesh import grouping 
from trimesh.geometry import faces_to_edges  


from hugs.models.modules.smpl_layer import SMPL


def subdivide(
    vertices,
    faces,              
    face_index=None,               
    vertex_attributes=None
):     
    
    if face_index is None:         
        face_mask = np.ones(len(faces), dtype=bool)     
    else:         
        face_mask = np.zeros(len(faces), dtype=bool)         
        face_mask[face_index] = True      
        
    # the (c, 3) int array of vertex indices     
    faces_subset = faces[face_mask]      
    # find the unique edges of our faces subset     
    edges = np.sort(faces_to_edges(faces_subset), axis=1)     
    unique, inverse = grouping.unique_rows(edges)     
    # then only produce one midpoint per unique edge     
    mid = vertices[edges[unique]].mean(axis=1)     
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)      
    # the new faces_subset with correct winding     
    f = np.column_stack(
        [         
            faces_subset[:, 0],         
            mid_idx[:, 0],         
            mid_idx[:, 2],        
            mid_idx[:, 0],         
            faces_subset[:, 1],         
            mid_idx[:, 1],         
            mid_idx[:, 2],         
            mid_idx[:, 1],         
            faces_subset[:, 2],         
            mid_idx[:, 0],         
            mid_idx[:, 1],         
            mid_idx[:, 2]]     
    ).reshape((-1, 3))      
    
    # add the 3 new faces_subset per old face all on the end     
    # # by putting all the new faces after all the old faces     
    # # it makes it easier to understand the indexes     
    new_faces = np.vstack((faces[~face_mask], f))     
    # stack the new midpoint vertices on the end     
    new_vertices = np.vstack((vertices, mid))    
        
    if vertex_attributes is not None:       
        new_attributes = {}       
        for key, values in vertex_attributes.items():
            if key == 'v_id':             
                attr_mid = values[edges[unique][:, 0]]         
            if key == 'lbs_weights':             
                attr_mid = values[edges[unique]].mean(axis=1)         
            else:             
                attr_mid = values[edges[unique]].mean(axis=1)                      
            new_attributes[key] = np.vstack((values, attr_mid))     
    return new_vertices, new_faces, new_attributes   


def _subdivide_smpl_model(smpl=None, smoothing=False):     
    if smpl is None:         
        smpl = SMPL("data/smpl")     
    # 读取：
        # mesh 顶点、面片
        # LBS 权重、pose shape 变形方向、J regressor 等属性
    # 获取 SMPL 模板的顶点数，例如 6890（原始 SMPL 模板）
    n_verts = smpl.v_template.shape[0]     
    # 提取 SMPL 的 posedirs，即 姿态变形方向矩阵（维度约为 (N, 3, 207) 或 (N, 3, K)）
    # 表示每个顶点在关节旋转下的位移响应方向
    init_posedirs = smpl.posedirs.detach().cpu().numpy()  
    
    # | 项目 | `init_J_regressor`      | `init_lbs_weights`         |
    # | -- | ----------------------- | -------------------------- |
    # | 功能 | 定义 **关节的位置**（从顶点线性回归得到） | 定义 **关节如何影响顶点**（控制骨骼绑定与变形） |
    # | 类型 | 顶点 ➝ 关节的权重              | 关节 ➝ 顶点的影响                 |
    # | 维度 | `(6890, 24)`            | `(6890, 24)`               |
    # | 用于 | 生成初始关节点坐标 `J`           | LBS 变形时计算姿态变换              |
    # | 方向 | 顶点 → 关节                 | 关节 → 顶点                    |
    # J_regressor 中这个关节对应列非零值分布在手腕附近的顶点 → 可用于从 mesh 反推出手腕位置
    # lbs_weights 中手腕附近的顶点在该关节对应行的值高 → 表示这些点主要受手腕骨骼控制

    # 提取 SMPL 每个顶点的 LBS（Linear Blend Skinning）权重（维度 (N, 24)）
    # 表示每个顶点由哪些关节控制以及控制强度
    init_lbs_weights = smpl.lbs_weights.detach().cpu().numpy()   
    # SMPL 中用于形状建模的 shape directions（维度通常是 (N, 3, 10)）
    # reshape 成 (N, 30) 便于后续在 subdivision 中做插值
    init_shapedirs = smpl.shapedirs.detach().cpu().numpy()   
    # 顶点编号，通常是 np.arange(6890).reshape(-1, 1)
    init_v_id = smpl.v_id     
    init_shapedirs = init_shapedirs.reshape(n_verts, -1)     
    # J_regressor: 将顶点位置映射到 24 个关节点的位置的线性权重矩阵
    # 维度：通常是 (24, 6890)，转置后变为 (6890, 24)，便于像 vertex 属性一样插值处理
    init_J_regressor = smpl.J_regressor.detach().cpu().numpy().transpose(1, 0)     
    # SMPL 模板的原始顶点位置 (6890, 3)
    init_vertices = smpl.v_template.detach().cpu().numpy()     
    init_faces = smpl.faces      
    print("# vertices before subdivision:", init_vertices.shape)  

    # 会生成：
    #     细分后的顶点、三角面
    #     同步插值得到的新属性值（LBS 权重、形变方向等）
    sub_vertices, sub_faces, attr = subdivide(
        vertices=init_vertices,         
        faces=init_faces,         
        vertex_attributes={
            "v_id": init_v_id,             
            "lbs_weights": init_lbs_weights,            
            "shapedirs": init_shapedirs,             
            "J_regressor": init_J_regressor,         
        }     
    )
    
    if smoothing:
        sub_mesh = trimesh.Trimesh(vertices=sub_vertices, faces=sub_faces)         
        sub_mesh = trimesh.smoothing.filter_mut_dif_laplacian(
            sub_mesh, 
            lamb=0.5,
            iterations=5,
            volume_constraint=True,
            laplacian_operator=None
        )        
        sub_vertices = sub_mesh.vertices
                       
    new_smpl = SMPL("data/smpl")     
    new_smpl.lbs_weights = torch.from_numpy(attr["lbs_weights"]).float()
    posedirs = np.zeros((207, sub_vertices.shape[0] * 3)).astype(np.float32)
    shapedirs = attr["shapedirs"].reshape(-1, 3, 10)
    J_regressor = np.zeros_like(attr["J_regressor"].transpose(1, 0))
    J_regressor[:, :n_verts] = smpl.J_regressor
    new_smpl.posedirs = torch.from_numpy(posedirs).float()
    new_smpl.shapedirs = torch.from_numpy(shapedirs).float()
    new_smpl.v_template = torch.from_numpy(sub_vertices).float()
    new_smpl.faces_tensor = torch.from_numpy(sub_faces).long()
    new_smpl.J_regressor = torch.from_numpy(J_regressor).float()
    new_smpl.faces = sub_faces     
    new_smpl.v_id = attr["v_id"].astype(int)
    return new_smpl    

# 功能：
# 外部接口
# 支持迭代 n_iter 次细分
# 可选拉普拉斯平滑
def subdivide_smpl_model(smpl=None, smoothing=False, n_iter=1):     
    if smpl is None:         
        from hugs.cfg.constants import SMPL_PATH         
        smpl = SMPL(SMPL_PATH)          
    
    smpl.v_id = np.arange(6890)[..., None]    
    # 每轮细分后 mesh 顶点数量翻倍增长。
    for _ in range(n_iter):     
        # 输入：
        #     smpl: 原始 SMPL 模型对象
        #     smoothing: 是否进行几何平滑
        smpl = _subdivide_smpl_model(smpl, smoothing)     
    return smpl
