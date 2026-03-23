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
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ...datasets.utils.constants import IGNORE_INDEX
from mmdet3d.models import builder
from transformers import AutoTokenizer
from ..utils.misc import load_model
from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import numpy as np
from projects.mmdet3d_plugin.models.utils.misc import MLN
from mmdet.models.utils.transformer import inverse_sigmoid
import time
import pickle
import torch.nn.functional as F
from typing import Optional, List, Dict
from .action_head import MSARActionHeadv4, VLAPlannerMSARv4
from .classifier import ImprovedClassifier

@DETECTORS.register_module()
class Petr3DClassifyMSARPLSeqFormerv2(MVXTwoStageDetector):
    """Petr3D v2."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start=1,
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
                 tokenizer=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 use_mapemb=True,
                 e2e_head=None,
                 use_pred_traj_seq=False,
                 use_kmeans_traj=False,
                 kmeans_anchor_path=None,
                 use_xy=False,
                 use_inverse_l2=False,
                 use_rag=False,
                 use_gt_index=False,
                 use_index_0=False,
                 pretrained=None,
                 use_grpo: bool = False,
                 num_category: int = 8,
                 classification_loss_weight: float = 1.0,
                 category_latent_dim: int = 2048,
                 task_type: str = "CAUSAL_LM",
                 traj_reg_loss_weight: float = 1.0,
                 stage_index: List[List[int]] = [[5], [2, 4], [1, 3, 5], list(range(6))],
                 ms_traj_loss_weights: Optional[List[float]] = None,
                 use_checkpointing: bool = False,
                 use_pretrained_traj_queries: bool = False,
                 pretrained_traj_queries_path: Optional[str] = None,
                 topk_mode_predict: int = -1,
                 cls_correct_idx: bool = False,
                 ):
        super(Petr3DClassifyMSARPLSeqFormerv2, self).__init__(pts_voxel_layer, pts_voxel_encoder,
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
        self.use_kmeans_traj = use_kmeans_traj
        self.use_xy = use_xy
        self.use_rag = use_rag
        self.use_gt_index = use_gt_index
        self.use_index_0 = use_index_0
        self.use_grpo = use_grpo
        self.classification_loss_weight = classification_loss_weight
        self.category_latent_dim = category_latent_dim
        self.traj_reg_loss_weight = traj_reg_loss_weight
        self.stage_index = stage_index
        self.ms_traj_loss_weights = ms_traj_loss_weights
        self.cls_correct_idx = cls_correct_idx
        self.topk_mode_predict = topk_mode_predict
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
        # During __init__, execution is still on CPU; the model should be moved to CUDA later by the runner

        if lm_head is not None:
            self.lm_head = load_model(
                lm_head,
                use_lora,
                frozen,
                False,
                task_type=task_type,
                use_double_lora=False,
            )

        # Fully disable LLM checkpointing (must stay disabled to avoid DDP issues from nested checkpointing)
        if self.with_lm_head and self.lm_head is not None:
            self._close_llm_checkpointing()
        
        if e2e_head is not None:
            self.e2e_head = builder.build_head(e2e_head)
 
        self.test_flag = False

        plan_anchor_lidar = torch.from_numpy(np.load(kmeans_anchor_path)).float()
        plan_anchor_ego = plan_anchor_lidar.clone()
        plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
        plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
        plan_anchor = plan_anchor_ego[[1, 0, 2]]
        plan_anchor[2, 0] = torch.zeros_like(plan_anchor[2, 0])
        self.register_buffer('plan_anchor', plan_anchor, persistent=False)
        self.embed_dims = embed_dims
        self.plan_fut_mode = self.plan_anchor.shape[1]
        self.fut_ts = self.plan_anchor.shape[2] #6

        # # === VLA: learnable queries & trajectory head ===
        if self.with_lm_head:
            self.lm_dim = self.lm_head.hidden_size
            assert self.lm_dim == self.lm_head.hidden_size

            llm_param = next(self.lm_head.get_model().parameters()) # Note the initialization dtype and device

            self.vl_adapter = nn.Linear(self.lm_dim, category_latent_dim)
            self.ego_adapter = nn.Linear(self.lm_dim, category_latent_dim)

            self.num_category = num_category

            self.category_queries = nn.Parameter(
                torch.randn(self.num_category, category_latent_dim, device=llm_param.device, dtype=torch.float32)
            )
            nn.init.xavier_uniform_(self.category_queries)

            self.category_classifier = ImprovedClassifier(
                dim=category_latent_dim,
                num_classes=self.num_category,
                hidden_dim=category_latent_dim // 2,
                use_cls_mixer=False,
                use_cls_film=True,
            )

            # --- traj_queries initialization ---
            if use_pretrained_traj_queries:
                pretrained_traj_path = pretrained_traj_queries_path or 'ckpts/traj_queries.pt'
                try:
                    # Compatible with both dicts saved by traj_query_utils.py and directly saved tensors
                    payload = torch.load(pretrained_traj_path, map_location=llm_param.device)
                    traj_tensor = payload["traj_queries"] if isinstance(payload, dict) else payload

                    # Drop indices 4 and 5 (these categories are not needed after class merging)
                    if traj_tensor.shape[0] > self.num_category:
                        indices_to_keep = [i for i in range(traj_tensor.shape[0]) if i not in [4, 5]]
                        traj_tensor = traj_tensor[indices_to_keep]
                        print(f"[Petr3D] Filtered out indices 4 and 5 from pretrained traj_queries, shape: {traj_tensor.shape}")

                    # Basic shape validation (can be relaxed if needed)
                    expected_shape = (self.num_category, self.lm_dim)
                    if traj_tensor.shape != expected_shape:
                        raise ValueError(f"Shape mismatch: {traj_tensor.shape} vs {expected_shape}")

                    self.traj_queries = nn.Parameter(traj_tensor.to(llm_param.device), requires_grad=True)
                    print(f"[Petr3D] Loaded pretrained traj_queries from {pretrained_traj_path}")

                except Exception as err:
                    print(f"[Petr3D] WARNING: could not load pretrained traj_queries ({err}); falling back to random init")
                    self.traj_queries = nn.Parameter(
                        torch.randn(self.num_category, self.lm_dim, device=llm_param.device, dtype=torch.float32) * 0.02
                    )
            else:
                # Path is empty or file does not exist -> random initialization
                self.traj_queries = nn.Parameter(
                    torch.randn(self.num_category, self.lm_dim, device=llm_param.device, dtype=torch.float32) * 0.02
                )

            # Use MSARActionHeadv4 (v4: unified parallel decoding)
            # Key improvement: no GT/predicted-point embeddings; unified parallel decoding for training and testing
            self.action_head_msar_ctx = MSARActionHeadv4(
                fut_ts=self.fut_ts,
                d_in=self.lm_dim,
                stage_indices=self.stage_index,
                ms_traj_loss_weights=self.ms_traj_loss_weights,
                use_gradient_checkpointing=use_checkpointing,  # keep enabled to balance performance and memory usage
                topk_mode_predict=self.topk_mode_predict,
            )

            # Use VLAPlannerMSARv4 (adapted for the v4 action head)
            self.planner_msar_ctx = VLAPlannerMSARv4(
                num_category=self.num_category,
                fut_ts=self.fut_ts,
                lm_dim=self.lm_dim,
                category_queries=self.category_queries,
                category_classifier=self.category_classifier,
                traj_queries=self.traj_queries,
                ego_adapter=self.ego_adapter,
                vl_adapter=self.vl_adapter,
                action_head=self.action_head_msar_ctx,
                classification_loss_weight=self.classification_loss_weight,
                traj_reg_loss_weight=self.traj_reg_loss_weight,
                cls_loss_fn=softmax_focal_loss,
                cls_correct_idx=self.cls_correct_idx,
            )

        
        # # DDP static-graph setup flag (called after DDP wrapper)
        # self._ddp_static_graph_setup_needed = use_checkpointing
        
    def _close_llm_checkpointing(self):
        try:
            base_llm = self.lm_head.get_model()  # -> LlavaLlamaModel
        except AttributeError:
            # Compatible with certain wrapper differences
            base_llm = getattr(self.lm_head, "model", self.lm_head)

        # Fully disable checkpointing
        try:
            base_llm.gradient_checkpointing_disable()
        except Exception as e:
            print("[INFO] gradient_checkpointing_disable not available:", e)

        # Double safeguard: also turn off config switches to prevent re-enabling elsewhere
        try:
            self.lm_head.config.gradient_checkpointing = False
            base_llm.config.gradient_checkpointing = False
        except Exception:
            pass

        # If use_cache=False was set elsewhere only for checkpointing, restore the default
        try:
            self.lm_head.config.use_cache = True
            base_llm.config.use_cache = True
        except Exception:
            pass
    
    def _freeze_qfromer(self):
        if self.img_head is not None:
            for name, param in self.img_head.named_parameters():
                param.requires_grad = False

        if self.map_head is not None:
            for name, param in self.map_head.named_parameters():
                param.requires_grad = False

        if self.pts_bbox_head is not None:
            for name, param in self.pts_bbox_head.named_parameters():
                param.requires_grad = False
 
        if self.img_head is not None:
            self.img_head.eval()
        if self.map_head is not None:
            self.map_head.eval()
        if self.pts_bbox_head is not None:
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
        B = img.size(0) # torch.Size([1, 6, 3, 640, 640])
        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask and self.training:
                img = self.grid_mask(img) # torch.Size([6, 3, 640, 640])
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck: # False
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

    def _build_llm_inputs_embeds_for_vla(
        self,
        input_ids: torch.Tensor,
        vision_embeded: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        points: torch.Tensor = None,
        trajectories: torch.Tensor = None,
        image_sizes=None,
        use_traj_branch: bool = True,
        return_img_idx: bool = False,
    ):
        """
        Reuse lm_head prepare_* to map (text + vision (+trajectory/points)) to inputs_embeds/attn/pos
        """
        assert self.with_lm_head and self.lm_head is not None, "lm_head missing for VLA"
        (
            _inp, position_ids, new_attention_mask, _pkv, inputs_embeds, _labels, img_indices
        ) = self.lm_head.prepare_inputs_labels_for_multimodal_traj(
            input_ids, position_ids, attention_mask, past_key_values, labels, vision_embeded,
            points, trajectories, ego_features=None, image_sizes=image_sizes, return_img_idx=return_img_idx
        )
        return inputs_embeds, new_attention_mask, position_ids, img_indices

    def forward_learnable_queries(
        self,
        input_ids: torch.Tensor,
        vision_embeded: torch.Tensor,
        query_embeds: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        points: torch.Tensor = None,
        trajectories: torch.Tensor = None,
        image_sizes=None,
        use_traj_branch: bool = True,
        detach: bool = False,
    ):
        """
        Learnable-query forward pass: return hidden features at these query positions [B, Q, D_llm]
        """

        inputs_embeds, new_attention_mask, position_ids, img_indices = self._build_llm_inputs_embeds_for_vla(
            input_ids, vision_embeded, position_ids, attention_mask, past_key_values, labels, points, trajectories, image_sizes, use_traj_branch
        )

        # query_embeds = self.vla_queries[None, ...].expand(B, -1, -1).contiguous()  # [B,Q,D]
        query_feats = query_embeds
        updated_context = inputs_embeds
        return query_feats, updated_context #, updated, inputs_embeds, new_attention_mask # used for condition for DiT

    def forward_understand_scene(
        self,
        input_ids: torch.Tensor,
        vision_embeded: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        points: torch.Tensor = None,
        trajectories: torch.Tensor = None,
        image_sizes=None,
        use_traj_branch: bool = True,
        detach: bool = False,
        return_img_idx: bool = True,
        return_full_hidden: bool = False,
    ):
        """
        Understand the scene and return the image indices.
        """
        B, Q_vision, D = vision_embeded.shape
        inputs_embeds, new_attention_mask, position_ids, img_indices = self._build_llm_inputs_embeds_for_vla(
            input_ids, vision_embeded, position_ids, attention_mask, past_key_values, labels, points, trajectories, image_sizes, use_traj_branch, return_img_idx
        )

        outputs = self.lm_head.base_model.model.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True if return_full_hidden else False,
        )

        last_hidden = outputs.last_hidden_state  # [B, L, D]
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=1e6, neginf=-1e6)

        updated_vision_list = []
        for i in range(len(img_indices)):
            batch_idx, start_pos, end_pos = img_indices[i]
            updated_vision_list.append(last_hidden[batch_idx, start_pos:end_pos, :])
        updated_vision = torch.stack(updated_vision_list, dim=0) # [B, Q_vision, D]
        assert updated_vision.shape == (B, Q_vision, D), "Updated vision shape mismatch"

        return updated_vision.float(), inputs_embeds

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          input_ids, 
                          vlm_labels, # includes GT ids for two answers; all others are -100
                          vlm_attn_mask, # padding mask
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
        B = data['img_feats'].shape[0]
        location = self.prepare_location(img_metas, **data)

        pos_embed = self.position_embeding(data, location, img_metas)
        losses = dict()
        if self.with_img_head:
            query, img_ref = self.img_head(img_metas, pos_embed, **data)
            image_query = query.clone()

        if self.with_pts_bbox:
            outs, det_query, can_bus_emb = self.pts_bbox_head(
                image_query, img_ref, img_metas, pos_embed, **data
            )
            image_query = det_query.clone()
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))
        else:
            det_query = image_query
            can_bus_emb = det_query[:, :1, :]

        if self.with_map_head:
            outs, map_query = self.map_head(image_query, img_metas, pos_embed, **data)
            loss_inputs = [lane_pts, outs, img_metas]
            losses.update(self.map_head.loss(*loss_inputs))
        else:
            map_query = det_query

        gt_traj_list = []
        for i in range(B):
            gt_traj_points = torch.from_numpy(img_metas[i]['gt_planning'][..., :2]).to(
                device=data['img_feats'].device, dtype=torch.float32
            )
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

        vision_query = map_query.clone()

        if self.with_lm_head:
            gt_category_index = torch.tensor([m["min_index"] for m in img_metas], device=vision_query.device) if self.training else None
            # # Label merge: when num_category==6, merge label 6 into 4 and label 7 into 5
            # if gt_category_index is not None and self.num_category == 6:
            #     gt_category_index_6 = gt_category_index.clone()
            #     gt_category_index_6[gt_category_index == 6] = 4
            #     gt_category_index_6[gt_category_index == 7] = 5
            #     gt_category_index = gt_category_index_6

            outs = self.planner_msar_ctx(
                img_metas=img_metas,
                input_ids=input_ids,
                map_query=vision_query, # torch.Size([2, 384, 4096])
                can_bus_emb=can_bus_emb, # torch.Size([2, 1, 4096])
                vlm_attn_mask=vlm_attn_mask,
                vlm_labels=vlm_labels,
                forward_queries_fn=self.lm_head.forward_queries_parallel_v2,  # v4 uses parallel_v2
                forward_understand_scene_fn=self.forward_understand_scene,
                return_loss=self.training,
                save_path=getattr(self, "save_path", None),
                gt_category_index=gt_category_index,
                gt_traj=gt_traj,
                gt_traj_mask=None,
            )

            # backward loss
            if "traj_class_loss" in outs:
                losses.update(traj_cls_loss=outs["traj_class_loss"])
            if "msar_traj_loss" in outs:
                losses.update(traj_reg_loss=outs["msar_traj_loss"])

            if "loss_diversity" in outs and outs["loss_diversity"] is not None:
                losses.update(loss_diversity=outs["loss_diversity"])
            
            # Monitor multi-scale losses
            for i in range(1, 7):
                key = f"loss_s{i}"
                if key in outs:
                    losses.update({key: outs[key].clone().detach()})
                
                # Monitor the loss of the last point for each stage
                last_point_key = f"loss_s{i}_last_point"
                if last_point_key in outs:
                    losses.update({last_point_key: outs[last_point_key].clone().detach()})
            
            # Normalized mode: monitor denormalized loss of the final stage full trajectory
            if "loss_final_denorm" in outs:
                losses.update(loss_final_denorm=outs["loss_final_denorm"])

            if "loss_confidence" in outs:
                losses.update(loss_confidence=outs["loss_confidence"])

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
        # # Set DDP static graph on first call (fix repeated-parameter usage with gradient checkpointing)
        # if hasattr(self, '_ddp_static_graph_setup_needed') and self._ddp_static_graph_setup_needed:
        #     self._setup_ddp_static_graph()
        #     self._ddp_static_graph_setup_needed = False  # set only once
        
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            
            vlm_labels = torch.full_like(input_ids, IGNORE_INDEX)
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
        #check
        location = self.prepare_location(img_metas, **data)
        pos_embed = self.position_embeding(data, location, img_metas)

        # bbox_results = []
        # if self.with_pts_bbox:
        #     outs, det_query, can_bus_emb = self.pts_bbox_head(img_metas, pos_embed, **data)
        #     vision_embeded_obj = det_query.clone()
        #     can_bus_emb = can_bus_emb.unsqueeze(-2) # torch.Size([1, 1, 4096])

        #     bbox_list = self.pts_bbox_head.get_bboxes(
        #         outs, img_metas)
        #     for bboxes, scores, labels in bbox_list:
        #         bbox_results.append(bbox3d2result(bboxes, scores, labels))
        
        # lane_results = []
        # if self.with_map_head:
        #     outs, map_query = self.map_head(img_metas, pos_embed, **data)
        #     vision_embeded_map = map_query.clone()
        #     lane_results = self.map_head.get_bboxes(outs, img_metas)


        # # det and map
        # vision_query = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) 
        bbox_results = []
        if self.with_img_head:
            query, img_ref = self.img_head(img_metas, pos_embed, **data)
            image_query = query.clone()


        det_query = image_query
        can_bus_emb = det_query[:, :1, :]
        if self.with_pts_bbox:
            outs, det_query, can_bus_emb = self.pts_bbox_head(image_query, img_ref, img_metas, pos_embed, **data)
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
            for bboxes, scores, labels in bbox_list:
                bbox_results.append(bbox3d2result(bboxes, scores, labels))

        lane_results = []
        if self.with_map_head:
            outs, map_query = self.map_head(det_query, img_metas, pos_embed, **data)
            lane_results = self.map_head.get_bboxes(outs, img_metas)
        else:
            map_query = det_query

        vision_query = map_query.clone()

        generated_text = []
        if self.with_lm_head:
            B = 1
            gt_category_index = [img_metas[i]['min_index'] for i in range(B)]
            gt_category_index_ = torch.tensor(gt_category_index).to(map_query.device)
            
            # Label merge: when num_category==6, merge label 6 into 4 and label 7 into 5
            if gt_category_index_ is not None and self.num_category == 6:
                gt_category_index_6 = gt_category_index_.clone()
                gt_category_index_6[gt_category_index_ == 6] = 4
                gt_category_index_6[gt_category_index_ == 7] = 5
                gt_category_index_ = gt_category_index_6

            input_ids = data['input_ids'][0].unsqueeze(0) # (1, L)
            outs = self.planner_msar_ctx(
                img_metas=img_metas,
                input_ids=input_ids,
                map_query=vision_query,
                can_bus_emb=can_bus_emb,
                vlm_attn_mask=None,
                vlm_labels=None,
                forward_queries_fn=self.lm_head.forward_queries_parallel_v2,  # v4 uniformly uses parallel_v2
                forward_understand_scene_fn=self.forward_understand_scene,
                return_loss=False,
                save_path=getattr(self, "save_path", None),
                gt_category_index=gt_category_index_,
                gt_traj=None,
            )
            pred_traj = outs["pred_traj"]
            vla_tuple = (pred_traj.detach().cpu())

            if 'inference_wgt' in self.save_path:
                classify_acc = outs["pred_index"] == gt_category_index_
            else:
                classify_acc = outs["query_index"] == gt_category_index_

            # save vla results and classify acc for evaluation
            if 'sample_idx' in img_metas[0] and img_metas[0]['sample_idx'] is not None:
                vla_save_dir = os.path.join(self.save_path, 'vla_results')
                os.makedirs(vla_save_dir, exist_ok=True)
                vla_save_path = os.path.join(vla_save_dir, img_metas[0]['sample_idx'] + '.pkl')
                with open(vla_save_path, 'wb') as f:
                    pickle.dump(pred_traj.cpu().numpy(), f)
                with open(vla_save_path.replace('.pkl', '_classify_acc.txt'), 'w') as f:
                    f.write(f'{classify_acc}') # True or False

                # Save multi-stage trajectories in JSON format for intuitive comparison
                multi_stage_traj_dir = os.path.join(self.save_path, 'multi_stage_traj')
                os.makedirs(multi_stage_traj_dir, exist_ok=True)
                
                # Build a multi-stage trajectory data dictionary
                multi_stage_data = {
                    'sample_idx': img_metas[0]['sample_idx'],
                    # 'classify_acc': classify_acc.item() if hasattr(classify_acc, 'item') else bool(classify_acc),
                    # 'pred_index': outs.get('pred_index', outs.get('query_index', -1)).item() if hasattr(outs.get('pred_index', outs.get('query_index', -1)), 'item') else int(outs.get('pred_index', outs.get('query_index', -1))),
                    # 'num_stages': self.action_head.num_stages,
                    'trajectories': {}
                }
                
                # Add trajectory data for each stage
                for i in range(1, self.action_head_msar_ctx.num_stages + 1):
                    stage_key = f'stage{i}_traj'
                    if stage_key in outs:
                        traj_data = outs[stage_key].cpu().numpy()
                        multi_stage_data['trajectories'][f'stage_{i}'] = {
                            'shape': list(traj_data.shape),
                            'data': traj_data.tolist()
                        }
                
                # Save as a JSON file
                multi_stage_traj_path = os.path.join(multi_stage_traj_dir, img_metas[0]['sample_idx'] + '.json')
                import json
                with open(multi_stage_traj_path, 'w') as f:
                    json.dump(multi_stage_data, f, indent=2)

        return bbox_results, generated_text, lane_results, classify_acc, vla_tuple
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        timing = {}
        start_time = time.time()

        data['img_feats'] = self.extract_img_feat(data['img'])

        # if 'inference_ncap' in self.save_path:
        #     save_images(data, save_dir=self.save_path)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, generated_text, lane_results, classify_acc, vla_tuple = self.simple_test_pts(
            img_metas, **data)
        # torch.cuda.synchronize()
        timing['total'] = time.time() - start_time
        bbox_pts[0]['timing'] = timing
        bbox_pts[0]['classify_acc'] = classify_acc
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['lane_results'] = lane_results

        # === New: also include results from VLA learnable queries ===
        if vla_tuple is not None:
            vla_traj = vla_tuple[0]
        else:
            vla_traj = None

        bbox_list[0]['vla_query_feats']  = None 
        bbox_list[0]['vla_traj'] = None if vla_traj  is None else vla_traj.numpy()

        return bbox_list    


    
def softmax_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Softmax Focal Loss for multi-class classification (single-label)

    Args:
        inputs: (B, C) logits
        targets: (B,) class indices
        alpha: balance factor (float or list, shape [C])
        gamma: focusing parameter
        reduction: "none" | "mean" | "sum"
    """
    B, C = inputs.shape
    # 1. Softmax probabilities
    log_probs = F.log_softmax(inputs, dim=-1)  # (B, C)
    probs = log_probs.exp()

    # 2. Extract probabilities of target classes
    targets = targets.long()
    probs_t = probs[torch.arange(B), targets]  # (B,)
    log_p_t = log_probs[torch.arange(B), targets]  # (B,)

    # 3. Focal modulation term
    focal_factor = (1 - probs_t) ** gamma

    # 4. Class-balancing alpha
    if isinstance(alpha, (float, int)):
        alpha_t = torch.full_like(probs_t, fill_value=alpha)
    else:
        alpha = torch.tensor(alpha, device=inputs.device, dtype=inputs.dtype)
        alpha_t = alpha[targets]  # (B,)

    # 5. focal loss
    loss = -alpha_t * focal_factor * log_p_t  # (B,)

    # 6. reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
    
