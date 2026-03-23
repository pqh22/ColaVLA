# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ...datasets.utils.constants import IGNORE_INDEX
from mmdet3d.models import builder
from transformers import AutoTokenizer, GenerationConfig
from ..utils.misc import load_model
from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
import numpy as np
from projects.mmdet3d_plugin.models.utils.misc import MLN
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.datasets.utils import conversation as conversation_lib
import time
from projects.mmdet3d_plugin.datasets.utils.data_utils import tokenizer_image_traj_token
import pickle


@DETECTORS.register_module()
class Petr3DImageSeqE2eCoT(MVXTwoStageDetector):
    """Petr3D."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 finetune_emb=False,
                 tokenizer=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 pretrain_dist_token_path=None,
                 use_mapemb=True,
                 e2e_head=None,
                 use_pred_traj_seq=False,
                 use_trajemb_cot=False,
                 use_gt_traj=False,
                 use_kmeans_traj=False,
                 kmeans_pad_traj=False,
                 kmeans_anchor_path=None,
                 use_xy=False,
                 use_inverse_l2=False,
                 use_rag=False,
                 use_gt_index=False,
                 use_index_0=False,
                 test_cot_cls_acc=False,
                 only_train_e2e_head=False,
                 share_proj=False,
                 pretrained=None):
        super(Petr3DImageSeqE2eCoT, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.use_pred_traj_seq = use_pred_traj_seq
        self.use_inverse_l2 = use_inverse_l2
        self.use_gt_traj = use_gt_traj
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_xy = use_xy    
        self.use_rag = use_rag
        self.use_gt_index = use_gt_index
        self.use_index_0 = use_index_0
        self.test_cot_cls_acc = test_cot_cls_acc
        self.only_train_e2e_head = only_train_e2e_head
        self.share_proj = share_proj
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156)
        self.use_mapemb = use_mapemb

        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if img_head is not None:
            self.img_head =builder.build_head(img_head)

        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            self.tokenizer.pad_token = self.tokenizer.unk_token

        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        if lm_head is not None:
            self.lm_head = load_model(lm_head, use_lora, frozen, finetune_emb)
        
        if e2e_head is not None:
            self.e2e_head = builder.build_head(e2e_head)
 
        self.test_flag = False

        plan_anchor_lidar = torch.from_numpy(np.load(kmeans_anchor_path)).cuda().float()
        plan_anchor_ego = plan_anchor_lidar.clone()
        plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
        plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
        self.plan_anchor = plan_anchor_ego[[1,0,2]] # 0: left, 1: right, 2: forward
        self.plan_anchor[2,0] = torch.zeros_like(self.plan_anchor[2,0])
        self.plan_anchor.requires_grad = False
        self.embed_dims = embed_dims
        self.plan_fut_mode = self.plan_anchor.shape[1]
        self.fut_ts = self.plan_anchor.shape[2]

        if self.use_kmeans_traj or self.use_rag:
            ego_query_pre_branch = []
            if self.use_xy:
                ego_query_pre_branch.append(nn.Linear(embed_dims*2, 1024))
            else:
                ego_query_pre_branch.append(nn.Linear(embed_dims, 1024))
            ego_query_pre_branch.append(nn.ReLU())
            ego_query_pre_branch.append(nn.Linear(1024, 4096))
            self.traj_projection = nn.Sequential(*ego_query_pre_branch)
            if self.share_proj:
                self.point_projection = self.traj_projection
            else:
                self.point_projection = nn.Sequential(*ego_query_pre_branch) 
        # elif self.use_pred_traj_seq:
        #     ego_query_pre_branch = []
        #     if self.use_xy:
        #         ego_query_pre_branch.append(nn.Linear(embed_dims*2, 1024))
        #     else:
        #         ego_query_pre_branch.append(nn.Linear(embed_dims, 1024))
        #     ego_query_pre_branch.append(nn.ReLU())
        #     ego_query_pre_branch.append(nn.Linear(1024, 4096))
        #     self.traj_projection = nn.Sequential(*ego_query_pre_branch)
        #     self.point_projection = nn.Sequential(*ego_query_pre_branch)               

        else:
            ego_query_pre_branch = []
            ego_query_pre_branch.append(nn.Linear(embed_dims * self.fut_ts, 2048))
            # ego_query_pre_branch = []
            # ego_query_pre_branch.append(nn.Linear(2 * self.fut_ts, embed_dims))
            ego_query_pre_branch.append(nn.ReLU())
            ego_query_pre_branch.append(nn.Linear(2048, 4096))
            self.traj_projection = nn.Sequential(*ego_query_pre_branch)


        if self.use_inverse_l2:
            self.traj_projection_inverse = nn.Sequential(
                nn.Linear(4096, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, 2))
            if self.use_pred_traj_seq:
                if  self.share_proj:
                    self.point_projection_inverse = self.traj_projection_inverse
                else:
                    self.point_projection_inverse = nn.Sequential(
                    nn.Linear(4096, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 2))

        if self.only_train_e2e_head:
            self.freeze_qfromer()

    def freeze_qfromer(self):
        for name, param in self.named_parameters():
            if 'e2e_head' not in name:
                param.requires_grad = False
 
        # train_params = []
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"需要梯度的参数: {name}")
        #         train_params.append(param)
        # if len(train_params) == 0:
        #     raise ValueError("没有需要训练的参数")
        self.img_head.eval()
        self.map_head.eval()
        self.pts_bbox_head.eval()

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
    
    @property
    def with_img_head(self):
        """bool: Whether the detector has a img head."""
        return hasattr(self,
                       'img_head') and self.img_head is not None
        
    
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
    
    @property
    def with_e2e_head(self):
        """bool: Whether the detector has a e2e head."""
        return hasattr(self,
                       'e2e_head') and self.e2e_head is not None
        

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        import os, ipdb
        if os.getenv("CLOSEDEBUG") == "1": ipdb.set_trace()
        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped


    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding



    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          lane_pts,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data)

        outs_roi = self.forward_roi_head(location, **data)

        pos_embed = self.position_embeding(data, location, img_metas)
        losses = dict()
        if self.with_img_head:
            query, img_ref = self.img_head(img_metas, pos_embed, **data)
            image_query = query.clone()

        if self.with_pts_bbox:
            outs, det_query,can_bus_emb = self.pts_bbox_head(image_query, img_ref, img_metas, pos_embed, **data)
            image_query = det_query.clone()
            # can_bus_embed = outs['can_bus_embed']
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            # vlm_memory_obj = outs['vlm_memory_256']
            rec_can_bus = outs['rec_can_bus']
            losses.update(self.pts_bbox_head.loss(*loss_inputs))

        if self.with_map_head:
            outs, map_query = self.map_head(image_query,img_metas, pos_embed, **data)
            # if can_bus_embed is not None:
            #     image_query = map_query.clone() + can_bus_embed
            # else:
            #     image_query = map_query.clone()
            loss_inputs = [lane_pts, outs, img_metas]
            # vlm_memory_map = outs['vlm_memory_256']
            losses.update(self.map_head.loss(*loss_inputs))
        
        if self.with_lm_head or True:
            # if self.use_mapemb:
            #     vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
            # else:
            # import pdb; pdb.set_trace()

            import os, ipdb
            if os.getenv("DEBUG") == "1": ipdb.set_trace()
            if self.use_pred_traj_seq:

                if False:
                    trajectories = None
                    ego_embeded = None
                else:

                    gt_traj_list = []
                    for i in range(B):
                        gt_traj_points = torch.from_numpy(img_metas[i]['gt_planning'][...,:2]).cuda().float()
                        if gt_traj_points.shape[1] != 6 and gt_traj_points.shape[1] != 18:
                            # Calculate velocity and direction from existing points
                            existing_points = gt_traj_points[:,:,:]  # [1, N, 2]
                            velocities = existing_points[:,1:] - existing_points[:,:-1]  # [1, N-1, 2]
                            last_velocity = velocities[:,-1:]  # Use last velocity
                            
                            # Extrapolate remaining points using velocity
                            num_missing = 6 - gt_traj_points.shape[1]
                            extrapolated_points = []
                            last_point = existing_points[:,-1:]
                            
                            for i in range(num_missing):
                                next_point = last_point + last_velocity
                                extrapolated_points.append(next_point)
                                last_point = next_point
                                
                            extrapolated_points = torch.cat(extrapolated_points, dim=1)
                            gt_traj_points = torch.cat([gt_traj_points, extrapolated_points], dim=1)
                        gt_traj_list.append(gt_traj_points)
                    gt_traj = torch.cat(gt_traj_list, dim=0)

                    if self.use_kmeans_traj or self.use_rag:
                        # import pdb; pdb.set_trace()
                        pred_anchor = []
                        for i in range(B):
                            gt_traj_points = torch.from_numpy(img_metas[i]['pred_traj2']).cuda().float()
                            pred_anchor.append(gt_traj_points)
                        pred_anchor = torch.stack(pred_anchor, dim=0) # 2,6,6,2
                        pred_anchor_end = pred_anchor[:,:,-1,:] # 2,6,2
                        end_emb = pos2posemb2d(pred_anchor_end, num_pos_feats=self.embed_dims).reshape(B, pred_anchor_end.shape[1], -1) # 2,6,512
                        # import pdb; pdb.set_trace()
                        points = self.point_projection(end_emb.cuda()) # 2,6,4096
                        min_index = [img_metas[i]['min_index'] for i in range(B)]
                        traj_anchor = pred_anchor[torch.arange(B), min_index] # [B, 6, 2]
                        _tmp = pos2posemb2d(traj_anchor.reshape(B, self.fut_ts, 2), num_pos_feats=self.embed_dims).reshape(B, self.fut_ts, -1)

                        if self.use_xy: # true
                            trajectories = _tmp # 2,6,512
                        else:
                            trajectories = _tmp.reshape(B, 12, 256)
                        trajectories = self.traj_projection(trajectories.cuda()) # 2,6,4096

                        if self.use_inverse_l2: # True
                            inverse_points = self.traj_projection_inverse(trajectories).reshape(B, self.fut_ts, 2)  # B x num_points x 2
                            l2_loss = torch.mean(torch.norm(inverse_points - traj_anchor, dim=-1))
                            losses.update(inverse_traj_loss=0.01*l2_loss)
                            inverse_points = self.point_projection_inverse(points).reshape(B, pred_anchor.shape[1], 2)  # B x num_points x 2
                            l2_loss = torch.mean(torch.norm(inverse_points - pred_anchor_end, dim=-1))
                            losses.update(inverse_point_loss=0.01*l2_loss)

                    else:
                        traj_anchor = self.plan_anchor[None].repeat(B,1,1,1,1)[torch.arange(B),data['command'].long()]
                        traj_anchor_end = traj_anchor[:,:,-1,:]
                        end_emb = pos2posemb2d(traj_anchor_end, num_pos_feats=self.embed_dims).reshape(B, traj_anchor_end.shape[1], -1)
                        points = self.point_projection(end_emb.cuda())
                        trajectories = []

                        # traj_anchor = self.plan_anchor.reshape(3*36,6,2)[None].repeat(B,1,1,1)
                        # Calculate L2 distance between gt_traj and each anchor trajectory
                        l2_dist = torch.norm(gt_traj[:,None,:,:] - traj_anchor, dim=-1).sum(dim=-1) # [B, 3*36]
                        # Find index of closest anchor trajectory
                        min_idx = torch.argmin(l2_dist, dim=-1) # [B]
                        # Select closest anchor trajectory
                        traj_anchor = traj_anchor[torch.arange(B), min_idx] # [B, 6, 2]
                        _tmp = pos2posemb2d(traj_anchor.reshape(B, 6, 2), num_pos_feats=self.embed_dims).reshape(B, 6, -1)

                        if self.use_xy:
                            trajectories = _tmp
                        else:
                            trajectories = _tmp.reshape(B, 12, 256)
                        trajectories = self.traj_projection(trajectories.cuda())

            if self.with_lm_head: # True
                vision_embeded = torch.cat([map_query.clone(), can_bus_emb], dim=1)
                vlm_loss = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, \
                                        points=points, trajectories=trajectories, ego_features=None, use_cache=False)
                losses.update(vlm_loss=vlm_loss[0])

        if self.with_e2e_head:
            vision_query = outs['vlm_memory_256']
            e2e_loss = self.e2e_head(vision_query, None, rec_can_bus, data, img_metas)
            losses.update(e2e_loss)
            
        # if self.with_img_roi_head:
        #     loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
        #     losses2d = self.img_roi_head.loss(*loss2d_inputs)
        #     losses.update(losses2d) 

        return losses

    # @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        import os, ipdb
        if os.getenv("DEBUG") == "1": ipdb.set_trace()
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      input_ids=None,
                      vlm_labels=None,
                      lane_pts=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None

        data['img_feats'] = self.extract_feat(data['img'])

        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, 
                                    depths, input_ids, vlm_labels, vlm_attn_mask, lane_pts, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data): # data dict_keys(['input_ids', 'img', 'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus'])
        # import ipdb; ipdb.set_trace()
        import os, ipdb
        if os.getenv("DEBUG") == "1": ipdb.set_trace()
        if os.getenv("CLOSEDEBUG") == "1": ipdb.set_trace()
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]: # img_metas list[list[dict]] img_metas[0][0] dict_keys(['pred_traj2', 'sample_idx', 'vlm_labels', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'])
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        #  torch.cuda.synchronize()  # Add CUDA synchronization
        #  start_time = time.time()
        import os, ipdb
        if os.getenv("DEBUG") == "1": ipdb.set_trace() 
        #check
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        if self.with_img_head:
            query, img_ref = self.img_head(img_metas, pos_embed, **data)
            image_query = query.clone()

        if self.with_pts_bbox:
            outs, det_query,can_bus_emb = self.pts_bbox_head(image_query, img_ref,img_metas, pos_embed, **data)
            rec_can_bus = outs['rec_can_bus']
            if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_metas)
                for bboxes, scores, labels in bbox_list:
                    bbox_results.append(bbox3d2result(bboxes, scores, labels))
        
        lane_results = []
        if self.with_map_head:
            outs, map_query = self.map_head(det_query, img_metas, pos_embed, **data)
            vision_emb = outs['vlm_memory_256']
            if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                lane_results = self.map_head.get_bboxes(outs, img_metas)

        generated_text = []
        cot_cls_acc = 0
        if self.with_lm_head:
            mmcv.mkdir_or_exist(self.save_path)
            vision_embeded = torch.cat([map_query.clone(), can_bus_emb], dim=1)
            B = vision_embeded.shape[0]

            import os, ipdb
            if os.getenv("CLOSEDEBUG") == "1": ipdb.set_trace()
            if self.use_pred_traj_seq:
                # Initialize conversation with same format as training
                conv = conversation_lib.default_conversation.copy()
                min_index = None
                for i, input_ids in enumerate(data['input_ids'][0]):
                    input_ids = input_ids.unsqueeze(0)
                    if i==0:
                        if False:
                            points = None
                            trajectories = None
                        else:
                            # if self.choose_from_pred or self.use_rag or self.use_kmeans_traj:
                            pred_anchor = []
                            for i in range(B):
                                pred_points = torch.from_numpy(img_metas[i]['pred_traj2']).cuda().float()
                                pred_anchor.append(pred_points)
                            pred_anchor = torch.stack(pred_anchor, dim=0)
                            pred_anchor_end = pred_anchor[:,:,-1,:]
                            end_emb = pos2posemb2d(pred_anchor_end.reshape(B,pred_anchor_end.shape[1],2), num_pos_feats=self.embed_dims).reshape(B, pred_anchor_end.shape[1], -1)
                            points = self.point_projection(end_emb.cuda())
                            trajectories = None
                                
                        # For first question
                        current_q = img_metas[0]['vlm_labels'].data[i]
                        conv.append_message(conv.roles[0], current_q)
                        prompt = conv.get_prompt() + "ASSISTANT:"  # 添加 ASSISTANT:
                        import os, ipdb
                        if os.getenv("DEBUG") == "1": ipdb.set_trace()
                        # Use tokenizer_image_traj_token instead of tokenizer
                        first_input_ids = torch.tensor(
                            tokenizer_image_traj_token(prompt, self.tokenizer, return_tensors=None),
                            dtype=torch.long
                        ).unsqueeze(0).cuda()
                        
                        output_ids = self.lm_head.generate(
                            inputs=first_input_ids,
                            images=vision_embeded,
                            trajectories=trajectories,
                            points=points,
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.75,
                            num_beams=1,
                            max_new_tokens=320,
                            use_cache=True
                        )
                        
                        # Get response and update conversation
                        current_a = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        conv.append_message(conv.roles[1], current_a)
                        
                        generated_text.append(
                            dict(
                            Q=current_q,
                            A=[current_a],
                            ))
                        # import pdb; pdb.set_trace()
                        if self.test_cot_cls_acc:
                            gt_traj_points = img_metas[0]['gt_planning'][...,:2]
                            if gt_traj_points.shape[1] != 6:
                                last_point = gt_traj_points[:,-1:].repeat(1,6-gt_traj_points.shape[1],1)
                                gt_traj_points = torch.cat([gt_traj_points, last_point], dim=1)
                            gt_traj_points = torch.from_numpy(gt_traj_points).to(dtype=vision_embeded.dtype)
                            l2_dist = torch.norm(gt_traj_points[:,None,:,:].cuda() - pred_anchor.cuda(), dim=-1).sum(dim=-1) # [B, 3*36]
                            gt_index = torch.argmin(l2_dist, dim=-1) # [B]
                            if self.use_gt_index:
                                min_index = gt_index
                            elif self.use_index_0:
                                min_index = 0
                            else:
                                try:
                                    min_index = int(''.join(filter(str.isdigit, current_a)))
                                except:
                                    import pdb; pdb.set_trace()
                            cot_cls_acc = 1 if min_index == gt_index else 0
                            return bbox_results, generated_text, lane_results, cot_cls_acc
                    else:
                        # For subsequent questions
                        current_q = img_metas[0]['vlm_labels'].data[i]
                        conv.append_message(conv.roles[0], current_q)
                        prompt = conv.get_prompt() + "ASSISTANT:"  # 添加 ASSISTANT:
                        
                        # Use tokenizer_image_traj_token instead of tokenizer
                        combined_input_ids = torch.tensor(
                            tokenizer_image_traj_token(prompt, self.tokenizer, return_tensors=None),
                            dtype=torch.long
                        ).unsqueeze(0).cuda()
                        import os, ipdb
                        if os.getenv("DEBUG") == "1": ipdb.set_trace()
                        # Process trajectories only at step 2 (i=1)
                        if i == 1:  # Only process trajectory at second step
                            if self.use_kmeans_traj or self.use_rag:
                                # import pdb; pdb.set_trace()
                                    # import pdb; pdb.set_trace()
                                # gt_traj_points = img_metas[0]['gt_planning'][...,:2]
                                # if gt_traj_points.shape[1] != 6 and gt_traj_points.shape[1] != 18:
                                #     last_point = gt_traj_points[:,-1:].repeat(1,6-gt_traj_points.shape[1],1)
                                #     gt_traj_points = torch.cat([gt_traj_points, last_point], dim=1)
                                # gt_traj_points = torch.from_numpy(gt_traj_points).to(dtype=vision_embeded.dtype)
                                # l2_dist = torch.norm(gt_traj_points[:,None,:,:].cuda() - pred_anchor.cuda(), dim=-1).sum(dim=-1) # [B, 3*36]
                                # gt_index = torch.argmin(l2_dist, dim=-1) # [B]
                                gt_index = 0
                                if self.use_gt_index:
                                    min_index = gt_index
                                elif self.use_index_0:
                                    min_index = 0
                                else:
                                    try:
                                        min_index = int(''.join(filter(str.isdigit, conv.messages[-2][1])))
                                    except:
                                        print('min_index error:',conv.messages[-2][1],'\n')
                                        min_index = 0
                                cot_cls_acc = 1 if min_index == gt_index else 0
                                traj_anchor = pred_anchor[torch.arange(B), min_index]
                                _tmp = pos2posemb2d(traj_anchor.reshape(1, self.fut_ts, 2), num_pos_feats=self.embed_dims).reshape(1, self.fut_ts, -1)

                                if self.use_xy:
                                    trajectories = _tmp
                                else:
                                    trajectories = _tmp.reshape(B, 12, 256)
                                trajectories = self.traj_projection(trajectories.cuda())

                        output_ids = self.lm_head.generate(
                            inputs=combined_input_ids,
                            images=vision_embeded,
                            trajectories=trajectories,
                            points=points,
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.75,
                            num_beams=1,
                            max_new_tokens=320,
                            use_cache=True
                        )
                        
                        # Get response and update conversation
                        current_a = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        conv.append_message(conv.roles[1], current_a)
                        
                        generated_text.append(
                            dict(
                            Q=current_q,
                            A=[current_a],
                            ))
                
                # 检查是否存在 sample_idx，如果不存在则不保存
                if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                    with open(self.save_path+img_metas[0]['sample_idx'], 'w') as file:
                        json.dump(generated_text, file)
                else:
                    print(f"Warning: No sample_idx found in img_metas[0], skipping file save for generated text: {generated_text}...")
            else:
                for i, input_ids in enumerate(data['input_ids'][0]):
                    input_ids = input_ids.unsqueeze(0)
                    output_ids = self.lm_head.generate(
                        inputs=input_ids,
                        images=vision_embeded,
                        trajectories=trajectories,
                        points=points,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True
                    )
                    generated_text.append(
                        dict(
                        Q=img_metas[0]['vlm_labels'].data[i],
                        A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                        ))

                # 检查是否存在 sample_idx，如果不存在则不保存
                if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                    with open(self.save_path+img_metas[0]['sample_idx'], 'w') as file:
                        json.dump(generated_text, file)
                else:
                    print(f"Warning: No sample_idx found in img_metas[0], skipping file save for generated text: {generated_text}...")

        # import pdb; pdb.set_trace()
        if self.with_e2e_head:
            plan_reg_pred = self.e2e_head(vision_emb, None, rec_can_bus, data, img_metas)
            if not os.path.exists(os.path.join(self.save_path, 'e2e_results')):
                os.makedirs(os.path.join(self.save_path, 'e2e_results'),exist_ok=True)
            if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                with open(os.path.join(self.save_path, 'e2e_results', img_metas[0]['sample_idx'] + '.pkl'), 'wb') as file:
                    pickle.dump((plan_reg_pred.cpu().numpy()), file)
            else:
                print(f"Warning: No sample_idx found in img_metas[0], skipping file save for e2e results: {plan_reg_pred.cpu().numpy()}...")

        # Add CUDA synchronization
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        
        # Add timing information to bbox_results
        # if len(bbox_results) > 0:
        #     bbox_results[0]['timing'] = {'simple_test_pts': elapsed_time}

        return bbox_results, generated_text, lane_results, cot_cls_acc
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        timing = {}
        start_time = time.time()
        data['img_feats'] = self.extract_img_feat(data['img'])
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, generated_text, lane_results, cot_cls_acc = self.simple_test_pts(
            img_metas, **data)
        if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None: # open-loop
            # torch.cuda.synchronize()
            timing['total'] = time.time() - start_time
            bbox_pts[0]['timing'] = timing
            bbox_pts[0]['cot_cls_acc'] = cot_cls_acc
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            bbox_list[0]['text_out'] = generated_text
            bbox_list[0]['lane_results'] = lane_results
        else: # closed-loop
            bbox_list[0]['text_out'] = generated_text
        return bbox_list
    

    