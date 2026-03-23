from tkinter import N
from typing import List, Optional, Tuple, Union
import warnings
import copy
import mmcv
import numpy as np
import cv2
import torch
import torch.nn as nn

# from mmengine.registry import build_from_cfg
from mmcv.cnn import Linear
from mmdet.models import HEADS, build_loss
# from mmengine.model import bias_init_with_prob
# from mmengine.model import BaseModule
#force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
# from mmdet3d.registry import MODELS as HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

# from data_gen import planning_anchor
from ..utils.positional_encoding import pos2posemb2d, gen_sineembed_for_position
# from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
# from projects.mmdet3d_plugin.core.box3d import *

from ..utils.petr_transformer import ConstructiveTransformerLayer, PETRTransformerDecoderLayer



@HEADS.register_module()
class MotionPlanningHeadEgo(nn.Module):
    def __init__(
        self,
        ego_fut_ts=6,
        ego_fut_mode=3,
        embed_dims=256,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        kmeans_anchor_path='/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/kmeans/kmeans_plan_6.npy',
        pretrained_path=None,
        future_frame_num=2,
        future_traj_path_train=None,
        future_traj_path_val=None,
        num_extra=256,
        use_seq_query=False,
        two_layer_cross_attn=False,
        add_plan_before_cross_attn=False,
        canbus_wo_attn=False,
        add_attn_before_head=False,
        encode_all_traj=False,
        pred_res=False,
        sync_pred_traj_path_train=None,
        sync_pred_traj_path_val=None,
    ):
        super(MotionPlanningHeadEgo, self).__init__()

        self.ego_fut_ts = ego_fut_ts
        self.num_extra = num_extra
        self.ego_fut_mode = ego_fut_mode
        self.pretrained_path = pretrained_path
        self.two_layer_cross_attn = two_layer_cross_attn
        self.add_plan_before_cross_attn = add_plan_before_cross_attn
        self.canbus_wo_attn = canbus_wo_attn
        self.add_attn_before_head = add_attn_before_head
        self.encode_all_traj = encode_all_traj
        self.pred_res = pred_res
        self.sync_pred_traj = None
        self.sync_pred_traj_train = None
        self.sync_pred_traj_val = None
        if sync_pred_traj_path_train is not None:
            self.sync_pred_traj_train = mmcv.load(sync_pred_traj_path_train)
        if sync_pred_traj_path_val is not None:
            self.sync_pred_traj_val = mmcv.load(sync_pred_traj_path_val)
        if self.training:
            self.sync_pred_traj = self.sync_pred_traj_train
        else:
            self.sync_pred_traj = self.sync_pred_traj_val
        plan_anchor_lidar = torch.from_numpy(np.load(kmeans_anchor_path)).cuda().float()
        plan_anchor_ego = plan_anchor_lidar.clone()
        plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
        plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
        self.plan_anchor = plan_anchor_ego[[1,0,2]] # 0: left, 1: right, 2: forward
        self.plan_anchor.requires_grad = False
        if self.canbus_wo_attn or self.add_plan_before_cross_attn:
            self.cross_attn_obj = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)
        else:
            self.cross_attn_obj = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)
        if not use_seq_query:
            self.cross_attn_map = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)
        if two_layer_cross_attn:
            if self.canbus_wo_attn or self.add_plan_before_cross_attn:
                self.cross_attn_obj_2 = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)
            else:
                self.cross_attn_obj_2 = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)

        if self.add_attn_before_head:
            self.cross_attn_obj_3 = ConstructiveTransformerLayer(256, 8, 256*4, dropout=0.1, flash_attn=True)
        self.mlp_position_encoder = nn.Sequential(
            nn.Linear(1, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        self.planning_sampler = PlanningTarget(ego_fut_ts=6, ego_fut_mode=plan_anchor_ego.shape[1]+1  if sync_pred_traj_path_train is not None else plan_anchor_ego.shape[1])
        self.use_seq_query = use_seq_query
        input_len = 74
        self.can_bus_embed = nn.Sequential(
            nn.Linear(input_len, embed_dims), # canbus + command + egopose
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),)

        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 6 * 2),
        )

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),)

        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 74))
        if self.encode_all_traj:
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(6*embed_dims, 1, 1),
                nn.Linear(6*embed_dims, embed_dims),
            )
        else:   
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                nn.Linear(embed_dims, embed_dims),
            )

        self.future_frame_num = future_frame_num
        if future_traj_path_train is not None:  
            self.future_traj = mmcv.load(future_traj_path_train)
            self.future_traj_val = mmcv.load(future_traj_path_val)
        else:
            self.future_traj = None
            self.future_traj_val = None


    def init_weights(self):
        if self.pretrained_path is not None:
            self.load_state_dict(torch.load(self.pretrained_path)['state_dict'],False)

    def mlp_position_encoding(self):
        pos_tensor = torch.arange(self.num_extra + 1, dtype=torch.float32).cuda().reshape(-1,1)
        pos_embed = self.mlp_position_encoder(pos_tensor)
        return pos_embed[:1], pos_embed[1:]


    def forward(
        self, 
        vlm_embed_obj,
        vlm_embed_map,
        can_bus,
        data,
        img_metas=None,
    ):   
        pos_embed_zero, pos_embed_4096 = self.mlp_position_encoding()
        B = vlm_embed_obj.shape[0]
        plan_anchor = torch.tile(self.plan_anchor[None], (B, 1, 1, 1, 1))
        tokens = [img_meta['sample_idx'] for img_meta in img_metas]
        # import pdb; pdb.set_trace()
        if self.future_traj:
            plan_anchor_list = []  # Shape: (3, 6, 6, 2)
            for idx, token in enumerate(tokens):
                if self.training:
                    traj_dict = self.future_traj[token]
                else:
                    traj_dict = self.future_traj_val[token]
                if self.future_frame_num == 2:
                    future_traj = traj_dict['frame_2']
                elif self.future_frame_num == 3:
                    future_traj = traj_dict['frame_3']
                elif self.future_frame_num == 4:
                    future_traj = traj_dict['frame_4']
                else:
                    future_traj = traj_dict['current']
                if future_traj is not None:
                    future_traj = torch.from_numpy(future_traj).cuda().float()
                    plan_anchor = self.plan_anchor.clone()  # Shape: (3, 6, 6, 2)
                    # plan_anchor = torch.tile(plan_anchor[None], (B, 1, 1, 1, 1))
                    # plan_anchor[torch.arange(B), data['command'].long()] = future_traj
                    # Calculate distance between each anchor trajectory and the future trajectory
                    plan_anchor_flat = plan_anchor[data['command'][idx].long()] # (6, 6, 2)
                    dist = torch.norm(plan_anchor_flat - future_traj[None], dim=-1).mean(dim=-1)  # (6,)
                    closest_idx = dist.argmax()
                    plan_anchor[data['command'][idx].long()][closest_idx] = future_traj
                    plan_anchor_list.append(plan_anchor)
                else:
                    plan_anchor_list.append(self.plan_anchor.clone())
            plan_anchor = torch.stack(plan_anchor_list)
        # import pdb; pdb.set_trace()
        if self.sync_pred_traj is not None:
            for img_meta in img_metas:
                if img_meta['sample_idx'] in self.sync_pred_traj.keys():
                    plan_anchor_sync = torch.from_numpy(self.sync_pred_traj[img_meta['sample_idx']]).float().cuda()
                    plan_anchor_sync = plan_anchor_sync[None,None,None,:,:].repeat(B,3,1,1,1)
                else:
                    plan_anchor_sync = torch.zeros_like(plan_anchor)[:,:,:1]
            plan_anchor = torch.cat([plan_anchor, plan_anchor_sync], dim=2)
        plan_pos = gen_sineembed_for_position(plan_anchor[...,-1,:],256)

        can_bus_embed = self.can_bus_embed(can_bus)[:,None]
        if self.add_plan_before_cross_attn:
            bs_indices = torch.arange(B, device=plan_pos.device)
            cmd = data['command'].long()
            plan_embed = self.plan_anchor_encoder(plan_pos)[bs_indices, cmd]
            if self.canbus_wo_attn:
                plan_query = plan_embed.reshape(B,-1,256)
            else:
                plan_query = (can_bus_embed + plan_embed).reshape(B,-1,256)
            plan_query = self.cross_attn_obj(plan_query, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if vlm_embed_map is not None:
                plan_query = self.cross_attn_map(plan_query, vlm_embed_map, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if self.two_layer_cross_attn:
                plan_query = self.cross_attn_obj_2(plan_query, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            
            if self.canbus_wo_attn:
                plan_query =  plan_query + can_bus_embed
            plan_reg_pred = self.plan_reg_branch(plan_query).reshape(B,1,plan_anchor.shape[2],6,2)
            plan_cls = self.plan_cls_branch(plan_query).reshape(B,1,plan_anchor.shape[2])
            plan_status = self.plan_status_branch(plan_query).reshape(B,plan_anchor.shape[2],74)
            # import pdb; pdb.set_trace()
            can_bus = can_bus[:,None].repeat(1,plan_anchor.shape[2],1)
            if self.training:
                losses = self.loss_planning_v2(plan_cls,plan_reg_pred,plan_status, can_bus, data)
                return losses
            else:
                max_score_index = torch.argmax(plan_cls[torch.arange(B), data['command'].long()], dim=-1)            
                plan_reg_pred = plan_reg_pred[torch.arange(B), data['command'].long(), max_score_index].reshape(B, plan_anchor.shape[2], 2)
                return plan_reg_pred
        elif self.add_attn_before_head:
            # import pdb; pdb.set_trace()
            bs_indices = torch.arange(B, device=plan_pos.device)
            cmd = data['command'].long()
            if self.encode_all_traj:
                plan_pos = gen_sineembed_for_position(plan_anchor,256)
                plan_embed = self.plan_anchor_encoder(plan_pos.reshape(B,3,6,-1))[bs_indices, cmd]
            else:
                plan_pos = gen_sineembed_for_position(plan_anchor[...,-1,:],256)
                plan_embed = self.plan_anchor_encoder(plan_pos)[bs_indices, cmd]
            plan_query = self.cross_attn_obj(can_bus_embed, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if vlm_embed_map is not None:
                plan_query = self.cross_attn_map(plan_query, vlm_embed_map, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if self.two_layer_cross_attn:
                plan_query = self.cross_attn_obj_2(plan_query, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            plan_embed = plan_query + plan_embed
            plan_embed = self.cross_attn_obj_3(plan_embed.reshape(B,-1,256), vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            plan_reg_pred = self.plan_reg_branch(plan_embed).reshape(B,1,6,6,2)
            plan_cls = self.plan_cls_branch(plan_embed).reshape(B,1,6)
            plan_status = self.plan_status_branch(plan_query).reshape(B,74)
            if self.training:
                losses = self.loss_planning_v2(plan_cls,plan_reg_pred,plan_status, can_bus, data)
                return losses
            else:
                max_score_index = torch.argmax(plan_cls[torch.arange(B), data['command'].long()], dim=-1)            
                plan_reg_pred = plan_reg_pred[torch.arange(B), data['command'].long(), max_score_index].reshape(B, plan_anchor.shape[2], 2)
                return plan_reg_pred
        else:
            plan_embed = self.plan_anchor_encoder(plan_pos)
            plan_query = self.cross_attn_obj(can_bus_embed, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if vlm_embed_map is not None:
                plan_query = self.cross_attn_map(plan_query, vlm_embed_map, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            if self.two_layer_cross_attn:
                plan_query = self.cross_attn_obj_2(plan_query, vlm_embed_obj, pos_embed_zero[None].repeat(B,1,1), pos_embed_4096[None].repeat(B,1,1), attn_mask=None)
            plan_embed = plan_query.unsqueeze(2) + plan_embed
            plan_reg_pred = self.plan_reg_branch(plan_embed)
            plan_cls = self.plan_cls_branch(plan_embed).squeeze(-1)
            plan_status = self.plan_status_branch(plan_query).squeeze(-1)
            if self.training:
                losses = self.loss_planning(plan_cls,plan_reg_pred,plan_status, can_bus, data)
                return losses
            else:
                max_score_index = torch.argmax(plan_cls[torch.arange(B), data['command'].long()], dim=-1)            
                plan_reg_pred = plan_reg_pred[torch.arange(B), data['command'].long(), max_score_index].reshape(B, 6, 2)
                return plan_reg_pred
       
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        # if self.use_motion_loss:
        #     motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        #     loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    # @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    # @force_fp32(apply_to=("model_outs"))
    # def loss_planning(self, model_outs, data):
    def loss_planning(self, cls_scores, reg_preds, status_preds, can_bus, data):

        (
            cls,
            cls_target, 
            cls_weight, 
            reg_pred, 
            reg_target, 
            reg_weight, 
        ) = self.planning_sampler.sample(
            cls_scores,
            reg_preds,
            data['gt_planning'][...,:2], # gt trajectory
            data['gt_planning_mask'],
            data,
        )
        cls = cls.flatten(end_dim=1)
        cls_target = cls_target.flatten(end_dim=1)
        cls_weight = cls_weight.flatten(end_dim=1)
        cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight.squeeze(1))
        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        import os, ipdb
        if os.getenv("TEST_DEBUG") == "1": ipdb.set_trace() 
        reg_loss = self.plan_loss_reg(
            reg_pred, reg_target.squeeze(1), weight=reg_weight[:,0,:,:].float()
        )
        # l1_reg_loss = F.l1_loss(reg_pred, reg_target.squeeze(1)).item()
        status_loss = self.plan_loss_status(status_preds.squeeze(1), can_bus)
        loss_weight = 1.0
        loss = {}
        loss.update(e2e_cls_loss=loss_weight*cls_loss, e2e_reg_loss=loss_weight*reg_loss, e2e_status_loss=loss_weight*status_loss)
        return loss

    def loss_planning_v2(self, cls_scores, reg_preds, status_preds, can_bus, data):
        gt_reg_target = data['gt_planning'][...,:2]
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = data['gt_planning_mask'].unsqueeze(1).any(dim=-1)

        if self.pred_res:
            # Calculate distances between target and anchors
            # import pdb;pdb.set_trace()
            bs_indices = torch.arange(gt_reg_mask.shape[0], device=gt_reg_mask.device)
            cmd_indices = data['command'].long()
            plan_anchor = self.plan_anchor[None].repeat(bs_indices.shape[0],1,1,1,1)[bs_indices, cmd_indices]
            dist = torch.linalg.norm(gt_reg_target.squeeze(1) - plan_anchor, dim=-1)
            dist = dist * gt_reg_mask[:,0,0,:,None]
            dist = dist.mean(dim=-2)  # Average over timesteps
            mode_idx = torch.argmin(dist, dim=-1)  # Best matching anchor
            # Get residuals between target and best anchor

            # Select the best anchor based on command and mode_idx
            best_anchor = plan_anchor[bs_indices, mode_idx]  # Shape should match gt_reg_target
            reg_target = (gt_reg_target - best_anchor[:,None,None])  # Residual target
            
            # Calculate losses
            cls = cls_scores.flatten(end_dim=1)
            cls_target = mode_idx.flatten()
            cls_weight = gt_reg_mask.any(dim=-1).flatten()
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)
            
            reg_pred = reg_preds[bs_indices, :, None, mode_idx.squeeze(-1)]
            reg_loss = self.plan_loss_reg(reg_pred.flatten(end_dim=1),reg_target.flatten(end_dim=1),weight=gt_reg_mask[:,0].unsqueeze(-1).float())
            
        else:
            # Original non-residual implementation
            cls_target = get_cls_target(reg_preds, gt_reg_target, gt_reg_mask)
            cls_weight = gt_reg_mask.any(dim=-1)
            reg_pred = get_best_reg(reg_preds, gt_reg_target, gt_reg_mask)
            cls = cls_scores.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight.squeeze(1))
            reg_weight = gt_reg_mask.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = gt_reg_target.flatten(end_dim=1)
            reg_weight = gt_reg_mask.unsqueeze(-1)
            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target.squeeze(1), weight=reg_weight[:,0,:,:].float()
            )

        status_loss = self.plan_loss_status(status_preds.squeeze(1), can_bus)
        loss_weight = 1.0
        loss = {}
        loss.update(e2e_cls_loss=loss_weight*cls_loss, 
                   e2e_reg_loss=loss_weight*reg_loss, 
                   e2e_status_loss=loss_weight*status_loss)
        return loss


    # @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result

class PlanningTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1).any(dim=-1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['command'].long()

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def get_cls_target(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx

def get_best_reg(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg

def get_best_reg_with_idx(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds
    reg_target_cum = reg_target
    dist = torch.linalg.norm(reg_target_cum - reg_preds_cum, dim=-1)
    dist = dist * reg_weight
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg, mode_idx