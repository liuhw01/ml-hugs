#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import joblib

import numpy as np
import torch
from tqdm import tqdm
import PIL
import scipy

from . import colmap_helper
from .geometry import pcd_projector
from .cameras import captures as captures_module, contents
from .scenes import scene as scene_module
from .utils import ray_utils
from .smpl import SMPL


def read_obj(path):
    vert = []
    uvs = []
    faces = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line[:2] == 'v ':
                v = line[2:].split()
                v = [float(i) for i in v]
                vert.append(np.array(v))
            elif line[:3] == 'vt ':
                uv = line[3:].split()
                uv = [float(i) for i in uv]
                uvs.append(np.array(uv))
            elif line[:2] == 'f ':
                f = line[2:].split()
                fv = [int(i.split('/')[0]) for i in f]
                ft = [int(i.split('/')[1]) for i in f]
                faces.append(np.array(fv + ft))

    vert = np.array(vert)
    uvs = np.array(uvs)
    faces = np.array(faces) - 1
    return vert, uvs, faces


class NeuManCapture(captures_module.RigRGBDPinholeCapture):
    def __init__(self, image_path, depth_path, mask_path, pinhole_cam, cam_pose, view_id, cam_id, mono_depth_path=None, keypoints_path=None, densepose_path=None):
        captures_module.RigRGBDPinholeCapture.__init__(self, image_path, depth_path, pinhole_cam, cam_pose, view_id, cam_id)
        self.captured_mask = contents.CapturedImage(
            mask_path
        )
        if mono_depth_path is not None:
            self.captured_mono_depth = contents.CapturedDepth(
                mono_depth_path
            )
            self.captured_mono_depth.dataset = 'mono'
        else:
            self.captured_mono_depth = None

        if keypoints_path is not None:
            self.keypoints = np.load(keypoints_path)
        else:
            self.keypoints = None

        if densepose_path is not None:
            self.densepose = np.load(densepose_path)
        else:
            self.densepose = None

        self._fused_depth_map = None

    def read_image_to_ram(self):
        if self.captured_mono_depth is None:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()
        else:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram() + self.captured_mono_depth.read_depth_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image.copy()
        if _mask.max() == 255:
            # Detectron2 mask
            _mask[_mask == 255] = 1
            _mask = 1 - _mask
        else:
            raise ValueError
        assert _mask.sum() > 0
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask

    @property
    def mono_depth_map(self):
        return self.captured_mono_depth.depth_map

    @property
    def fused_depth_map(self):
        if self._fused_depth_map is None:
            valid_mask = (self.depth_map > 0) & (self.mask == 0)
            x = self.mono_depth_map[valid_mask]
            y = self.depth_map[valid_mask]
            res = scipy.stats.linregress(x, y)
            self._fused_depth_map = self.depth_map.copy()
            self._fused_depth_map[~valid_mask] = self.mono_depth_map[~valid_mask] * res.slope + res.intercept
        return self._fused_depth_map


class ResizedNeuManCapture(captures_module.ResizedRigRGBDPinholeCapture):
    def __init__(self, image_path, depth_path, mask_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id, mono_depth_path=None, keypoints_path=None, densepose_path=None):
        captures_module.ResizedRigRGBDPinholeCapture.__init__(self, image_path, depth_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id)
        '''
        Note: we pass in the original intrinsic and distortion matrix, NOT the resized intrinsic
        '''
        self.captured_mask = contents.ResizedCapturedImage(
            mask_path,
            tgt_size,
            sampling=PIL.Image.NEAREST
        )
        if mono_depth_path is not None:
            self.captured_mono_depth = contents.ResizedCapturedDepth(
                mono_depth_path,
                tgt_size=tgt_size
            )
            self.captured_mono_depth.dataset = 'mono'
        else:
            self.captured_mono_depth = None
        if keypoints_path is not None:
            # raise NotImplementedError
            self.keypoints = None
        else:
            self.keypoints = None
        if densepose_path is not None:
            # raise NotImplementedError
            self.densepose = None
        else:
            self.densepose = None

    def read_image_to_ram(self):
        if self.captured_mono_depth is None:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()
        else:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram() + self.captured_mono_depth.read_depth_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image.copy()
        if _mask.max() == 255:
            # Detectron2 mask
            _mask[_mask == 255] = 1
            _mask = 1 - _mask
        else:
            raise ValueError
        assert _mask.sum() > 0
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask

    @property
    def mono_depth_map(self):
        return self.captured_mono_depth.depth_map


def create_split_files(scene_dir):
    # 10% as test set
    # 10% as validation set
    # 80% as training set
    dummy_scene = NeuManReader.read_scene(scene_dir)
    scene_length = len(dummy_scene.captures)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0
    splits = []
    for l, split in zip([train_list, val_list, test_list], ['train', 'val', 'test']):
        output = []
        save_path = os.path.join(scene_dir, f'{split}_split.txt')
        for i, cap in enumerate(dummy_scene.captures):
            if i in l:
                output.append(os.path.basename(cap.image_path))
        with open(save_path, 'w') as f:
            for item in output:
                f.write("%s\n" % item)
        splits.append(save_path)
    return splits


def read_text(txt_file):
    '''
    read the split file to a list
    '''
    assert os.path.isfile(txt_file)
    items = []
    with open(txt_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            items.append(line.strip())
    return items


class NeuManReader():
    def __init__(self):
        pass

    #     | 参数名                 | 类型            | 说明                                                 |
    # | ------------------- | ------------- | -------------------------------------------------- |
    # | `scene_dir`         | str           | 场景主目录（包含 `images/`, `sparse/`, `depth_maps/` 等子目录） |
    # | `tgt_size`          | tuple or None | 图像目标尺寸，用于 resize 图像                                |
    # | `normalize`         | bool          | 是否对场景进行归一化（调整相机坐标缩放）                               |
    # | `bkg_range_scale`   | float         | 背景区域的 near/far 范围缩放因子                              |
    # | `human_range_scale` | float         | 人体区域的 near/far 范围缩放因子（目前默认未启用）                     |
    # | `mask_dir`          | str           | 掩码图路径                                              |
    # | `smpl_type`         | str           | SMPL 参数类型，如 `'romp'`, `'optimized'`                |
    # | `keypoints_dir`     | str           | 关键点路径目录                                            |
    # | `densepose_dir`     | str           | DensePose 路径目录                                     |
    @classmethod
    def read_scene(cls, scene_dir, tgt_size=None, normalize=False, bkg_range_scale=1.1, human_range_scale=1.1, mask_dir='segmentations', smpl_type='romp', keypoints_dir='keypoints', densepose_dir='densepose'):
        def update_near_far(scene, keys, range_scale):
            # compute the near and far
            for view_id in tqdm(range(scene.num_views), total=scene.num_views, desc=f'Computing near/far for {keys}'):
                for cam_id in range(scene.num_cams):
                    cur_cap = scene.get_capture_by_view_cam_id(view_id, cam_id)
                    if not hasattr(cur_cap, 'near'):
                        cur_cap.near = {}
                    if not hasattr(cur_cap, 'far'):
                        cur_cap.far = {}
                    for k in keys:
                        if k == 'bkg':
                            pcd_2d_bkg = pcd_projector.project_point_cloud_at_capture(scene.point_cloud, cur_cap, render_type='pcd')
                            near = 0  # np.percentile(pcd_2d_bkg[:, 2], 5)
                            far = np.percentile(pcd_2d_bkg[:, 2], 95)
                        elif k == 'human':
                            pcd_2d_human = pcd_projector.project_point_cloud_at_capture(scene.verts[view_id], cur_cap, render_type='pcd')
                            near = pcd_2d_human[:, 2].min()
                            far = pcd_2d_human[:, 2].max()
                        else:
                            raise ValueError(k)
                        center = (near + far) / 2
                        length = (far - near) * range_scale
                        cur_cap.near[k] = max(0.0, float(center - length / 2))
                        cur_cap.far[k] = float(center + length / 2)
                        
        # read_captures(...) 从 images/ 和 sparse/ 中读取：
        # 每帧的 RGB 图像路径、深度图路径、相机内参、位姿等；
        # 返回 captures（一个 NeuManCapture 对象列表）；
        # 构造 RigCameraScene 对象，表示整个场景。
        # ✅ 输出：scene.captures, scene.point_cloud, scene.num_views
        captures, point_cloud, num_views, num_cams = cls.read_captures(scene_dir, tgt_size, mask_dir=mask_dir, keypoints_dir=keypoints_dir, densepose_dir=densepose_dir)
        
        # 构造一个多摄像头-多视角场景对象，用于统一管理和访问每一帧的图像、深度图、相机参数以及对应的三维点云数据。
        # scene.get_captures_by_view_id(3)     # 第4帧的图像信息（view 3）
        # scene.get_capture_by_view_cam_id(5, 0)  # 第6帧第0个摄像头的采集信息
        # scene.point_cloud   # 如果有提供点云数据
        # scene[3]            # 也可以按索引访问 captures
        scene = scene_module.RigCameraScene(captures, num_views, num_cams)
        
        scene.point_cloud = point_cloud
        
        # 根据场景中背景点云的位置，自动计算每一帧相机在渲染“背景（bkg）”时所需的 near / far clipping plane（近平面/远平面），并设置到每个 capture（帧）中。
        # 📌 最终效果
        #     每一个 capture（每帧图像）将拥有如下属性：
        # 投影到当前视图相机坐标系下，取出所有点的 z 值（深度）
        #     capture.near = {'bkg': 0.3, 'human': 0.2}
        #     capture.far  = {'bkg': 5.1, 'human': 2.3}
        #     用于后续渲染或训练过程中的视锥体裁剪计算（即 Frustum）。
        update_near_far(scene, ['bkg'], bkg_range_scale)

        if normalize:
            fars = []
            for cap in scene.captures:
                fars.append(cap.far['bkg'])
            fars = np.array(fars)
            scale = 3.14 / (np.percentile(fars, 95))
            for cap in scene.captures:
                cap.cam_pose.camera_center_in_world *= scale
                cap.near['bkg'], cap.far['bkg'] = cap.near['bkg'] * scale, cap.far['bkg'] * scale
                cap.captured_depth.scale = scale
                cap.captured_mono_depth.scale = scale
            scene.point_cloud[:, :3] *= scale
        else:
            scale = 1

        scene.scale = scale
        # smpls, world_verts, static_verts, Ts = cls.read_smpls(scene_dir, scene.captures, scale=scale, smpl_type=smpl_type)
        # scene.smpls, scene.verts, scene.static_vert, scene.Ts = smpls, world_verts, static_verts, Ts
        # _, uvs, faces = read_obj(
        #     'data/smpl/smpl_uv.obj'
        # )
        # scene.uvs, scene.faces = uvs, faces
        # update_near_far(scene, ['human'], human_range_scale)

        assert len(scene.captures) > 0

        return scene

    @classmethod
    def read_smpls(cls, scene_dir, caps, scale=1, smpl_type='romp'):
        def extract_smpl_at_frame(raw_smpl, frame_id):
            out = {}
            for k, v in raw_smpl.items():
                try:
                    out[k] = v[frame_id]
                except:
                    out[k] = None
            return out

        device = torch.device('cpu')
        body_model = SMPL(
            'data/smpl',
            gender='neutral',
            device=device
        )
        smpls = []
        static_verts = []
        world_verts = []
        Ts = []
        smpl_path = os.path.join(scene_dir, f'smpl_output_{smpl_type}.pkl')
        assert os.path.isfile(smpl_path), f'{smpl_path} is missing'
        print(f'using {smpl_type} smpl')
        raw_smpl = joblib.load(smpl_path)
        assert len(raw_smpl) == 1
        raw_smpl = raw_smpl[list(raw_smpl.keys())[0]]
        raw_alignments = np.load(os.path.join(scene_dir, 'alignments.npy'), allow_pickle=True).item()
        for cap in caps:
            frame_id = int(os.path.basename(cap.image_path)[:-4])
            # assert 0 <= frame_id < len(caps)
            temp_smpl = extract_smpl_at_frame(raw_smpl, frame_id)
            temp_alignment = np.eye(4)
            temp_alignment[:, :3] = raw_alignments[os.path.basename(cap.image_path)]
            
            # 大 pose
            da_smpl = np.zeros_like(temp_smpl['pose'][None])
            da_smpl = da_smpl.reshape(-1, 3)
            da_smpl[1] = np.array([0, 0, 1.0])
            da_smpl[2] = np.array([0, 0, -1.0])
            da_smpl = da_smpl.reshape(1, -1)

            _, T_t2pose = body_model.verts_transformations(
                return_tensor=False,
                poses=temp_smpl['pose'][None],
                betas=temp_smpl['betas'][None],
                concat_joints=True
            )
            _, T_t2da = body_model.verts_transformations(
                return_tensor=False,
                poses=da_smpl,
                betas=temp_smpl['betas'][None],
                concat_joints=True
            )
            T_da2pose = T_t2pose @ np.linalg.inv(T_t2da)
            T_da2scene = temp_alignment.T @ T_da2pose
            s = np.eye(4)
            s[:3, :3] *= scale
            T_da2scene = s @ T_da2scene

            da_pose_verts, da_pose_joints = body_model(
                return_tensor=False,
                return_joints=True,
                poses=da_smpl,
                betas=temp_smpl['betas'][None]
            )
            temp_world_verts = np.einsum('BNi, Bi->BN', T_da2scene, ray_utils.to_homogeneous(np.concatenate([da_pose_verts, da_pose_joints], axis=0)))[:, :3].astype(np.float32)
            temp_world_verts, temp_world_joints = temp_world_verts[:6890, :], temp_world_verts[6890:, :]
            temp_smpl['joints_3d'] = temp_world_joints
            temp_smpl['static_joints_3d'] = da_pose_joints
            smpls.append(temp_smpl)
            Ts.append(T_da2scene)
            static_verts.append(da_pose_verts)
            world_verts.append(temp_world_verts)
        return smpls, world_verts, static_verts, Ts

    @classmethod
    def read_captures(cls, scene_dir, tgt_size, mask_dir='segmentations', keypoints_dir='keypoints', densepose_dir='densepose'):
    # 作用是从指定场景路径中读取并封装每帧图像的相机参数、深度图、掩码、关键点、densepose 等信息，并生成 NeuManCapture 或 ResizedNeuManCapture 对象列表作为输出。
        caps = []
        # 1️⃣ 使用 COLMAP 工具加载基础相机与图像结构
        # 输入：
        # scene_dir/sparse: COLMAP 的 .txt 相机和位姿输出；
        # scene_dir/images: 图像序列所在路径；
        # tgt_size: 可选图像缩放尺寸，如 (512, 512)。
        # 输出（raw_scene 是 RigCameraScene 实例）：
        # raw_scene.captures: List[Capture]
        # raw_scene.point_cloud: np.ndarray, shape = (N, 6)  # xyzrgb
        raw_scene = colmap_helper.ColmapAsciiReader.read_scene(
            os.path.join(scene_dir, 'sparse'),
            os.path.join(scene_dir, 'images'),
            tgt_size,
            order='video',
        )

        # 2️⃣ 初始化参数
        num_views = len(raw_scene.captures)  # 视角数（等于图像数）
        num_cams = 1 # 默认单摄像头
        counter = 0 # 图像帧编号计数器
        
        for view_id in range(num_views):
            for cam_id in range(num_cams):
                raw_cap = raw_scene.captures[counter]
                
                # 4️⃣ 构造该帧对应的其他数据路径（depth, mono_depth, mask, keypoints, densepose）
                # raw_cap.image_path: '.../images/00003.jpg'
                # depth_path:         '.../depth_maps/00003.jpg.geometric.bin'
                # mono_depth_path:    '.../mono_depth/00003.jpg'

                # ✅ 1. depth_path: 多视图几何深度图（MVS Depth）
                # 来源：来自 COLMAP 等多视图立体（Multi-View Stereo, MVS）算法。 预先得到的
                # 输入数据：使用同一场景中多张图像，从不同角度三角化计算深度。
                # 优点：
                # 通常精度高，特别是表面几何结构清晰时。
                # 缺点：
                # 对纹理敏感区域效果好，但在遮挡、低纹理或人身上经常失败或缺失。
                # 存在“洞”或稀疏区域。
                # 文件格式：通常是 .geometric.bin 或 .pfm 等二进制格式。
                depth_path = raw_cap.image_path.replace('/images/', '/depth_maps/') + '.geometric.bin'
                # ✅ 2. mono_depth_path: 单目深度图（Monocular Depth） 
                # 来源：来自训练好的单目深度估计网络，如 MiDaS、DPT、LeReS。  预先得到的
                # 输入数据：只依赖单张图像，无需多视角。
                mono_depth_path = raw_cap.image_path.replace('/images/', '/mono_depth/')

                if not os.path.isfile(depth_path):
                    depth_path = raw_cap.image_path + 'dummy'
                    print(f'can not find mvs depth for {os.path.basename(raw_cap.image_path)}')
                if not os.path.isfile(mono_depth_path):
                    mono_depth_path = raw_cap.image_path + 'dummy'
                    print(f'can not find mono depth for {os.path.basename(raw_cap.image_path)}')

                
                # 这三个路径变量分别对应图像中人体区域的分割掩码（mask）、人体关键点（keypoints）和DensePose 表示（densepose）。
                # ✅ mask_path：人体掩码图（Segmentation Mask）
                #     作用：指定图像中属于“人”的区域。用于：
                #     剔除背景（训练时聚焦于人）
                #     对人体区域单独监督优化
                #     格式：
                #     npy 文件：形如 000001.jpg.npy
                #     或图像文件：000001.jpg.png
                #     内容：灰度图（一般为 0 表示背景，255 表示人体）
                mask_path = os.path.join(scene_dir, mask_dir, os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(mask_path):
                    mask_path = os.path.join(scene_dir, mask_dir, os.path.basename(raw_cap.image_path))

                # ✅ keypoints_path：人体2D关键点（e.g. COCO格式）
                #     作用：
                #     提供人体骨骼结构约束信息
                #     可用于监督姿态估计或优化初始化
                #     格式：.npy 文件（通常是 17 × 3 数组，表示17个关键点的 x, y, confidence）
                keypoints_path = os.path.join(scene_dir, keypoints_dir, os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(keypoints_path):
                    print(f'can not find keypoints for {os.path.basename(raw_cap.image_path)}')
                    keypoints_path = None

                # ✅ densepose_path：DensePose 表示（人体像素级 UV 坐标）
                #     作用：
                #     每个人体像素对应一个 SMPL 网格上的 UV 坐标和 part label
                #     提供像素级的高密度语义约束，用于精细建模
                densepose_path = os.path.join(scene_dir, densepose_dir, 'dp_' + os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(densepose_path):
                    print(f'can not find densepose for {os.path.basename(raw_cap.image_path)}')
                    densepose_path = None
                    
                # 6️⃣ 根据是否指定 tgt_size 选择使用 NeuManCapture 还是 ResizedNeuManCapture
                if tgt_size is None:
                    temp = NeuManCapture(
                        raw_cap.image_path,
                        depth_path,
                        mask_path,
                        raw_cap.pinhole_cam,
                        raw_cap.cam_pose,
                        view_id,
                        cam_id,
                        mono_depth_path=mono_depth_path,
                        keypoints_path=keypoints_path,
                        densepose_path=densepose_path
                    )
                else:
                    temp = ResizedNeuManCapture(
                        raw_cap.image_path,
                        depth_path,
                        mask_path,
                        raw_cap.pinhole_cam,
                        raw_cap.cam_pose,
                        tgt_size,
                        view_id,
                        cam_id,
                        mono_depth_path=mono_depth_path,
                        keypoints_path=keypoints_path,
                        densepose_path=densepose_path
                    )
                temp.frame_id = raw_cap.frame_id
                counter += 1
                caps.append(temp)
            
        # raw_scene.point_cloud：points3D.txt
        return caps, raw_scene.point_cloud, num_views, num_cams
