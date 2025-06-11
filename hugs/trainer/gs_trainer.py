#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import glob
import shutil
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger

from hugs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from hugs.losses.utils import ssim
from hugs.datasets import NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models import SceneGS
from hugs.utils.init_opt import optimize_init
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.vis import save_ply
from hugs.utils.image import psnr, save_image
from hugs.utils.general import RandomIndexIterator, load_human_ckpt, save_images, create_video


def get_train_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-train')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'train', 
            render_mode=cfg.mode,
            add_bg_points=cfg.scene.add_bg_points,
            num_bg_points=cfg.scene.num_bg_points,
            bg_sphere_dist=cfg.scene.bg_sphere_dist,
            clean_pcd=cfg.scene.clean_pcd,
        )
    
    return dataset


def get_val_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-val')
        dataset = NeumanDataset(cfg.dataset.seq, 'val', cfg.mode)
   
    return dataset


def get_anim_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-anim')
        dataset = NeumanDataset(cfg.dataset.seq, 'anim', cfg.mode)
    elif cfg.dataset.name == 'zju':
        dataset = None
        
    return dataset


class GaussianTrainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
        # get dataset
        if not cfg.eval:
            self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        self.anim_dataset = get_anim_dataset(cfg)
        
        self.eval_metrics = {}
        self.lpips = LPIPS(net="alex", pretrained=True).to('cuda')
        # get models
        self.human_gs, self.scene_gs = None, None
        
        if cfg.mode in ['human', 'human_scene']:
            if cfg.human.name == 'hugs_wo_trimlp':
                self.human_gs = HUGS_WO_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=cfg.human.n_subdivision,  
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                )
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                self.human_gs.initialize()
            
            #             | 参数名                          | 含义                                       |
            # | ---------------------------- | ---------------------------------------- |
            # | `name: hugs_trimlp`          | 使用的是 `HUGS_TRIMLP` 类                     |
            # | `ckpt: null`                 | 没有预加载的 checkpoint，用于从头开始训练               |
            # | `sh_degree: 0`               | 不使用 spherical harmonics（SH）来建模颜色（因为设置为0） |
            # | `n_subdivision: 2`           | 初始网格进行2次 subdivision，用于生成更密集的高斯点         |
            # | `only_rgb: false`            | 不仅仅优化 RGB，还可能涉及透明度、LBS 等属性               |
            # | `use_surface: false`         | 不使用 mesh 表面约束（默认是点云优化）                   |
            # | `use_deformer: true`         | 使用可变形模型（例如 SMPL-LBS）来驱动高斯变形              |
            # | `init_2d: false`             | 不使用2D图像初始化高斯点（用的是3D初始化）                  |
            # | `disable_posedirs: true`     | 不使用 pose blend shape（简化模型）               |
            # | `res_offset: false`          | 是否使用残差偏移建模，关闭                            |
            # | `rotate_sh: false`           | 是否让 SH 旋转以匹配姿态，这里关闭                      |
            # | `isotropic: false`           | 高斯是否为各向同性（否，保持完整协方差）                     |
            # | `init_scale_multiplier: 0.5` | 高斯初始缩放因子（影响渲染模糊度）                        |
            # | `run_init: false`            | 启动时是否初始化（后面代码里有单独控制）                     |
            # | `estimate_delta: true`       | 是否估计 shape 残差                            |
            # | `triplane_res: 256`          | 三平面体素网络分辨率                               |
            # | `optim_pose: true`           | 是否优化姿态参数                                 |
            # | `optim_betas: false`         | 是否优化 shape 参数 β（这里关闭，只使用初始的）             |
            # | `optim_trans: true`          | 优化位移参数                                   |
            # | `optim_eps_offsets: false`   | 是否优化 epsilon 偏移量（关闭）                     |
            # | `activation: relu`           | 三平面 MLP 中使用 ReLU 激活函数                    |
            # | `canon_nframes: 60`          | 用于生成 canonical pose 的帧数                  |
            # | `canon_pose_type: da_pose`   | 选择 canonical pose 类型                     |
            # | `knn_n_hops: 3`              | 用于构建 knn 图结构的 hop 数                      |
            elif cfg.human.name == 'hugs_trimlp':
                init_betas = torch.stack([x['betas'] for x in self.val_dataset.cached_data], dim=0)
                self.human_gs = HUGS_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=cfg.human.n_subdivision,  
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                    n_features=32,
                    use_deformer=cfg.human.use_deformer,
                    disable_posedirs=cfg.human.disable_posedirs,
                    triplane_res=cfg.human.triplane_res,
                    betas=init_betas[0]
                )
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                if not cfg.eval:
                    self.human_gs.initialize()
                    self.human_gs = optimize_init(self.human_gs, num_steps=7000)
        
        if cfg.mode in ['scene', 'human_scene']:
            self.scene_gs = SceneGS(
                sh_degree=cfg.scene.sh_degree,
            )
            
        # setup the optimizers
        if self.human_gs:
            self.human_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.human_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.human_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.human_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')

            if not cfg.eval:
                  # 字段包括：
                # | 参数名             | 维度     | 含义         | 功能描述                           |
                # | --------------- | ------ | ---------- | ------------------------------ |
                # | `global_orient` | N × 3  | 全局朝向       | 控制整个身体在三维空间中的旋转（轴角格式）          |
                # | `body_pose`     | N × 69 | 身体姿态       | 控制23个身体关节的局部旋转（每个关节3维轴角，共69维）  |
                # | `transl`        | N × 3  | 平移向量       | 控制人体在三维空间中的位置                  |
                # | `betas`         | N × 10 | 形状参数（个体特征） | 控制人体的个性化形状（身高、胖瘦、比例等），来自PCA主成分 |
                # | `scale`         | N × 1  | 全局缩放因子     | 对整个模型进行统一的缩放调整，以适应不同数据或场景      |
                init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
                init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])   # 把每帧的 body_pose（形状 (69,)）堆成 (N, 69)。
                init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0) # 把每帧的 transl（(3,)）堆成 (N, 3)。  
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0) 
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3), 
                                            dtype=torch.float32, device="cuda")
                
                # 在模型中创建可学习的 betas 参数
                # 取 init_betas[0]（第一个样本的形状参数）作为初始值。
                # cfg.human.optim_betas 控制是否对 betas 打开梯度及学习率。
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                
                # 在模型中逐项创建并初始化 SMPL 参数
                #     create_body_pose：将 (N,69) 的 body_pose 注册为可学习参数，梯度由 cfg.human.optim_pose 控制。
                #     create_global_orient：将 (N,3) 的 global_orient 注册为可学习参数。
                #     create_transl：将 (N,3) 的 transl 注册为可学习参数，梯度由 cfg.human.optim_trans 控制。
                # | 参数名             | 是否时序相关 | 维度               | 是否每一帧都不同？ | 说明                             |
                    # | --------------- | ------ | ---------------- | --------- | ------------------------------ |
                    # | `body_pose`     | ✅ 是    | `(N, 69)`        | ✅ 是       | 控制每帧关节姿态（如走路/坐下等），每帧都不同，必须单独优化 |
                    # | `global_orient` | ✅ 是    | `(N, 3)`         | ✅ 是       | 控制每帧整体旋转，同样随时间变化               |
                    # | `transl`        | ✅ 是    | `(N, 3)`         | ✅ 是       | 控制每帧位移，也随时间变化（走路等）             |
                    # | `betas`         | ❌ 否    | `(10,)`          | ❌ 否       | 控制“身体形状”，一旦确定（身材特征），整段序列都共享    |
                    # | `scale`         | ❌ 否    | `(1,)` or `(N,)` | 多为 ❌      | 全局缩放一般在数据预处理阶段就定好，不需要帧级优化      |
                self.human_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.human_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)
                
                self.human_gs.setup_optimizer(cfg=cfg.human.lr)
                    
        if self.scene_gs:
            logger.info(self.scene_gs)
            if cfg.scene.ckpt:
                ckpt = torch.load(cfg.scene.ckpt)
                self.scene_gs.restore(ckpt, cfg.scene.lr)
                logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*scene*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.scene_gs.restore(ckpt, cfg.scene.lr)
                    logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
                else:
                    pcd = self.train_dataset.init_pcd
                    spatial_lr_scale = self.train_dataset.radius
                    self.scene_gs.create_from_pcd(pcd, spatial_lr_scale)
                
            self.scene_gs.setup_optimizer(cfg=cfg.scene.lr)
        
        bg_color = cfg.bg_color
        if bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")
        
        if cfg.mode in ['human', 'human_scene']:
            l = cfg.human.loss

            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                l_lpips_w=l.lpips_w,
                l_lbs_w=l.lbs_w,
                l_humansep_w=l.humansep_w,
                num_patches=l.num_patches,
                patch_size=l.patch_size,
                use_patches=l.use_patches,
                bg_color=self.bg_color,
            )
        else:
            self.cfg.train.optim_scene = True
            l = cfg.scene.loss
            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                bg_color=self.bg_color,
            )
                
        if cfg.mode in ['human', 'human_scene']:
            # 通过 get_rotating_camera(...) 构造一个环绕人体旋转的相机轨迹，用于 canonical 状态下的可视化渲染或训练 supervision。
            #     参数解释：
            #     dist=5.0：相机到目标中心的距离为 5 个单位。
            #     img_size=512：渲染图像分辨率（用于相机内参设定）。
            #     nframes=cfg.human.canon_nframes：轨迹中生成多少帧（通常是 60 或 120）。
            #     device='cuda'：相机参数生成在 GPU 上。
            #     angle_limit=2π：生成 360° 环绕轨迹。
            self.canon_camera_params = get_rotating_camera(
                dist=5.0, img_size=512, 
                nframes=cfg.human.canon_nframes, device='cuda',
                angle_limit=2*torch.pi,
            )

            # ✅ 功能：获取 shape 参数 betas
            betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]

            # ✅ 功能：生成 canonical 姿态下的 SMPL 参数
            self.static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.human.canon_pose_type
            )

    def train(self):
        if self.human_gs:
            self.human_gs.train()

        pbar = tqdm(range(self.cfg.train.num_steps+1), desc="Training")
        
        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        sgrad_means, sgrad_stds = [], []
        for t_iter in range(self.cfg.train.num_steps+1):
            render_mode = self.cfg.mode
            
            if self.scene_gs and self.cfg.train.optim_scene:
                self.scene_gs.update_learning_rate(t_iter)
            
            if hasattr(self.human_gs, 'update_learning_rate'):
                self.human_gs.update_learning_rate(t_iter)
        
            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]
            
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            
            if self.scene_gs:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    scene_gs_out = self.scene_gs.forward()
                else:
                    render_mode = 'human'
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            
            
            if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                render_human_separate = True
                human_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            else:
                human_bg_color = None
                render_human_separate = False

            #         | 字段名                 | 含义                       |
            # | ------------------- | ------------------------ |
            # | `render`            | 最终图像，shape 为 `(3, H, W)` |
            # | `viewspace_points`  | 每个高斯的 2D 屏幕坐标（实际由渲染器写入）  |
            # | `visibility_filter` | 每个高斯是否被渲染器视锥覆盖           |
            # | `radii`             | 每个高斯在图像平面上的投影半径（越大越靠近相机） |
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                render_mode=render_mode,
                render_human_separate=render_human_separate,
            )
            
            if self.human_gs:
                self.human_gs.init_values['edges'] = self.human_gs.edges

            
            loss, loss_dict, loss_extras = self.loss_fn(
                data,
                render_pkg,
                human_gs_out,
                render_mode=render_mode,
                human_gs_init_values=self.human_gs.init_values if self.human_gs else None,
                bg_color=bg_color,
                human_bg_color=human_bg_color,
            )
            
            loss.backward()
            
            loss_dict['loss'] = loss
            
            if t_iter % 10 == 0:
                postfix_dict = {
                    "#hp": f"{self.human_gs.n_gs/1000 if self.human_gs else 0:.1f}K",
                    "#sp": f"{self.scene_gs.get_xyz.shape[0]/1000 if self.scene_gs else 0:.1f}K",
                    'h_sh_d': self.human_gs.active_sh_degree if self.human_gs else 0,
                    's_sh_d': self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_"+k] = f"{v.item():.4f}"
                        
                pbar.set_postfix(postfix_dict)
                pbar.update(10)
                
            if t_iter == self.cfg.train.num_steps:
                pbar.close()

            # 🖼 每 1000 步保存可视化渲染图
            if t_iter % 1000 == 0:
                with torch.no_grad():
                    pred_img = loss_extras['pred_img']
                    gt_img = loss_extras['gt_img']
                    log_pred_img = (pred_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                    save_images(log_img, f'{self.cfg.logdir}/train/{t_iter:06d}.png')

            # 🧩 Scene 高斯动态 densify（体素分裂）
            if t_iter >= self.cfg.scene.opt_start_iter:
                # 是否还在 densify 区间内：
                if (t_iter - self.cfg.scene.opt_start_iter) < self.cfg.scene.densify_until_iter and self.cfg.mode in ['scene', 'human_scene']:
                    # 提取场景的 viewspace_point（screenspace_points），用于导数判断：
                    render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']
                    render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
                    
                    # 记录梯度均值和方差（调试用）：
                    sgrad_mean, sgrad_std = render_pkg['scene_viewspace_points'].grad.mean(), render_pkg['scene_viewspace_points'].grad.std()
                    sgrad_means.append(sgrad_mean.item())
                    sgrad_stds.append(sgrad_std.item())
                    
                    with torch.no_grad():
                        # scene_densification 函数负责 动态地细化（densify）和剪枝（prune）场景中的高斯点
                        # 传入参数：
                        #     visibility_filter：一个布尔张量，标记哪些高斯在当前视角可见。
                        #     radii：当前视角下每个高斯在屏幕上的半径（投影尺度）。                            
                        #     viewspace_point_tensor：具有梯度的屏幕空间位置，用于感知哪些高斯重要。                          
                        #     iteration：当前训练迭代步数。
                        self.scene_densification(
                            visibility_filter=render_pkg['scene_visibility_filter'],
                            radii=render_pkg['scene_radii'],
                            viewspace_point_tensor=render_pkg['scene_viewspace_points'],
                            iteration=(t_iter - self.cfg.scene.opt_start_iter) + 1,
                        )
                        
            if t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                with torch.no_grad():
                    self.human_densification(
                        human_gs_out=human_gs_out,
                        visibility_filter=render_pkg['human_visibility_filter'],
                        radii=render_pkg['human_radii'],
                        viewspace_point_tensor=render_pkg['human_viewspace_points'],
                        iteration=t_iter+1,
                    )
            
            if self.human_gs:
                self.human_gs.optimizer.step()
                self.human_gs.optimizer.zero_grad(set_to_none=True)
                
            if self.scene_gs and self.cfg.train.optim_scene:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    self.scene_gs.optimizer.step()
                    self.scene_gs.optimizer.zero_grad(set_to_none=True)
                
            # save checkpoint
            if (t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0) or \
                (t_iter == self.cfg.train.num_steps and t_iter > 0):
                self.save_ckpt(t_iter)

            # run validation
            if t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
                self.validate(t_iter)

            # 📸 初始化时保存初始模型 mesh 和 canonical 渲染
            if t_iter == 0:
                if self.scene_gs:
                    self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{t_iter:06d}_splat.ply')
                if self.human_gs:
                    save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')

                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)

            # 每 anim_interval 轮执行一次：
                # 保存当前人体的 .ply
                # 执行 self.animate(t_iter)（可能是一个小动画片段）
                # 渲染 canonical pose 的图像序列（一般是环绕的 360° 摄像机）
            if t_iter % self.cfg.train.anim_interval == 0 and t_iter > 0 and self.cfg.train.anim_interval > 0:
                if self.human_gs:
                    save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                if self.anim_dataset is not None:
                    self.animate(t_iter)
                    
                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)

            # 每 1000 步增长一次 SH 表达能力（颜色更精细），调用：
            if t_iter % 1000 == 0 and t_iter > 0:
                if self.human_gs: self.human_gs.oneupSHdegree()  # self.active_sh_degree += 1
                if self.scene_gs: self.scene_gs.oneupSHdegree()  # self.active_sh_degree += 1

            # 🖼️ 渲染训练进度图像（可用于视频）
            if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0 and self.cfg.mode in ['human', 'human_scene']:
                self.render_canonical(t_iter, nframes=2, is_train_progress=True)
        
        # train progress images
        if self.cfg.train.save_progress_images:
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')
            
    def save_ckpt(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        if self.human_gs:
            torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
            
        if self.scene_gs:
            torch.save(self.scene_gs.state_dict(), f'{self.cfg.logdir_ckpt}/scene_{iter_s}.pth')
            self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply')
            
        logger.info(f'Saved checkpoint {iter_s}')

    # scene_densification 函数负责 动态地细化（densify）和剪枝（prune）场景中的高斯点
        # 传入参数：
        #     visibility_filter：一个布尔张量，标记哪些高斯在当前视角可见。
        #     radii：当前视角下每个高斯在屏幕上的半径（投影尺度）。                            
        #     viewspace_point_tensor：具有梯度的屏幕空间位置，用于感知哪些高斯重要。                          
        #     iteration：当前训练迭代步数。
    def scene_densification(self, visibility_filter, radii, viewspace_point_tensor, iteration):
        # 1️⃣ 记录最大可见屏幕半径
        self.scene_gs.max_radii2D[visibility_filter] = torch.max(
            self.scene_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        # 2️⃣ 累积 densification 统计信息
        # 这一步将 当前梯度值 和其他相关信息累计到 scene_gs 中，用于分析高斯点的重要性（如哪些点梯度较大、活跃度高）。
            # 📊 为什么要记录这些信息？
                # 这些累积梯度统计用于判断高斯点是否应该 densify（复制细化）：
                # 梯度大 → 说明该点对图像误差贡献大 → 应该复制更多细节。
                # 梯度小 → 表示该点不重要，甚至可能被剪枝。
        self.scene_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # 3️⃣ 条件触发 densify & prune
        # 在训练步数超过 densify_from_iter 后，且每隔 densification_interval 步，就执行一次高斯体素分裂和裁剪：
        if iteration > self.cfg.scene.densify_from_iter and iteration % self.cfg.scene.densification_interval == 0:
            # size_threshold 限制了屏幕上的最大半径，大于这个的高斯不会再继续 split（控制密度上限）。
            size_threshold = 20 if iteration > self.cfg.scene.opacity_reset_interval else None
            self.scene_gs.densify_and_prune(
                self.cfg.scene.densify_grad_threshold,   # 梯度分裂阈值
                min_opacity=self.cfg.scene.prune_min_opacity,  # 剪枝时最低 opacity
                extent=self.train_dataset.radius,  # 空间范围限制
                max_screen_size=size_threshold,    # 屏幕尺寸限制（optional） 
                max_n_gs=self.cfg.scene.max_n_gaussians, # 高斯点总数上限
            )
            
        # 4️⃣ 重置 Opacity（不透明度）
        is_white = self.bg_color.sum().item() == 3.
        
        if iteration % self.cfg.scene.opacity_reset_interval == 0 or (is_white and iteration == self.cfg.scene.densify_from_iter):
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            self.scene_gs.reset_opacity()
    
    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.human.densify_from_iter and iteration % self.cfg.human.densification_interval == 0:
            size_threshold = 20
            self.human_gs.densify_and_prune(
                human_gs_out,
                self.cfg.human.densify_grad_threshold, 
                min_opacity=self.cfg.human.prune_min_opacity, 
                extent=self.cfg.human.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.human.max_n_gaussians,
            )
    
    # 这段代码定义了 HUGS 模型的验证函数 validate，用于评估当前高斯模型在验证集上的表现。它会：
    # 执行正向渲染
    # 与真实图像进行比对
    # 计算评价指标（PSNR, SSIM, LPIPS）
    # 保存对比图像
    # 存储验证结果
    @torch.no_grad()
    def validate(self, iter=None):

        # iter_s: 用于保存图像和日志的标识（如果没有提供迭代次数就是 "final"）
        # bg_color: 渲染背景设为黑色 [0, 0, 0]
        iter_s = 'final' if iter is None else f'{iter:06d}'
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

        # 👤 切换人体模型为 eval 模式（不影响场景模型）
        if self.human_gs:
            self.human_gs.eval()

        # 📊 初始化指标容器
        methods = ['hugs', 'hugs_human']
        metrics = ['lpips', 'psnr', 'ssim']
        # 初始化字典，如：
        metrics = dict.fromkeys(['_'.join(x) for x in itertools.product(methods, metrics)])
        metrics = {k: [] for k in metrics}

        # 🔁 遍历验证集进行评估
        # 每帧验证数据包含：data['rgb']、smpl 参数、bbox、mask等。
        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            human_gs_out, scene_gs_out = None, None
            render_mode = self.cfg.mode
            
            if self.human_gs:
                # ✅ 1. 运行人体高斯（如果启用）
                # 使用 smpl 参数 + betas + transl 推理出当前帧的高斯位置和属性。
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'], 
                    body_pose=data['body_pose'], 
                    betas=data['betas'], 
                    transl=data['transl'], 
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if self.scene_gs:
                # ✅ 2. 场景高斯 forward（在一定迭代后才会激活）
                # 使用 render_human_scene 渲染当前帧（整合人体和场景高斯），输出为图像张量。
                if iter is not None:
                    if iter >= self.cfg.scene.opt_start_iter:
                        scene_gs_out = self.scene_gs.forward()
                    else:
                        render_mode = 'human'
                else:
                    scene_gs_out = self.scene_gs.forward()

            # ✅ 3. 渲染合成图像
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=bg_color,
                render_mode=render_mode,
            )
            
            gt_image = data['rgb']
            
            image = render_pkg["render"]
            if self.cfg.dataset.name == 'zju':
                image = image * data['mask']
                gt_image = gt_image * data['mask']
                
            # 📏 计算指标（全图）
            metrics['hugs_psnr'].append(psnr(image, gt_image).mean().double())
            metrics['hugs_ssim'].append(ssim(image, gt_image).mean().double())
            metrics['hugs_lpips'].append(self.lpips(image.clip(max=1), gt_image).mean().double())

            # 💾 保存整图对比结果
            log_img = torchvision.utils.make_grid([gt_image, image], nrow=2, pad_value=1)
            imf = f'{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png'
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)
            
            log_img = []
            # 🧍 如果是人体模式，额外裁剪 bbox 区域
            if self.cfg.mode in ['human', 'human_scene']:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_image = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                log_img += [cropped_gt_image, cropped_image]
                
                # 获取人体在图像中的 bbox，计算该区域的指标。
                metrics['hugs_human_psnr'].append(psnr(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_ssim'].append(ssim(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_lpips'].append(self.lpips(cropped_image.clip(max=1), cropped_gt_image).mean().double())
            # 保存裁剪区域的对比图：
            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(log_img, nrow=len(log_img), pad_value=1)
                torchvision.utils.save_image(log_img, f'{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png')
        
        # 📊 汇总所有指标并保存为 .pth
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')
    
    @torch.no_grad()
    def animate(self, iter=None, keep_images=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )
            
            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        if not keep_images:
            shutil.rmtree(f'{self.cfg.logdir}/anim/')
            os.makedirs(f'{self.cfg.logdir}/anim/')
    
    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]
        
        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=self.cfg.human.canon_pose_type if pose_type is None else pose_type,
        )
        
        if is_train_progress:
            progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    scaling_modifier=scale_mod,
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
            else:
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}")
            return
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/canon/')
        os.makedirs(f'{self.cfg.logdir}/canon/')
        
    def render_poses(self, camera_params, smpl_params, pose_type='a_pose', bg_color='white'):
    
        if self.human_gs:
            self.human_gs.eval()
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.val_dataset.betas[0]
        
        nframes = len(camera_params)
        
        canon_forward_out = None
        if hasattr(self.human_gs, 'canon_forward'):
            canon_forward_out = self.human_gs.canon_forward()
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        if bg_color is 'white':
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color is 'black':
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            
        imgs = []
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(smpl_params, **cam_p)

            if self.human_gs:
                if canon_forward_out is not None:
                    human_gs_out = self.human_gs.forward_test(
                        canon_forward_out,
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )
                else:
                    human_gs_out = self.human_gs.forward(
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            imgs.append(image)
        return imgs
