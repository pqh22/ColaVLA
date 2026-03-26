# ============================================================================
# ColaVLA Inference Configuration for NeuroNCAP Closed-loop Testing
# ============================================================================
# 此配置文件专门用于 NeuroNCAP 闭环测试的推理服务
# 基于训练配置 vlm_seq_384_cot_rag5_loade6qformere2e0320_noqa_headlr20_e10_cotspeed.py
#
# 主要修改:
# - 创建专门用于推理的数据处理管线
# - 正确处理VLM prompt生成和tokenization
# - 适配NeuroNCAP的数据格式
# - 移除文件加载相关的步骤
# ============================================================================

_base_ = [
    "../../../mmdetection3d/configs/_base_/datasets/nus-3d.py",
    "../../../mmdetection3d/configs/_base_/default_runtime.py",
]

backbone_norm_cfg = dict(type="LN", requires_grad=True)
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Point cloud range and voxel size
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# nuScenes class names
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Model configuration
num_extra = 384
llm_path = "ckpts/pretrain_qformer/"

collect_keys = [
    "lidar2img",
    "intrinsics",
    "extrinsics",
    "timestamp",
    "img_timestamp",
    "ego_pose",
    "ego_pose_inv",
    "command",
    "can_bus",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

batch_size = 1

model = dict(
    type="Petr3DImageSeqE2eCoT",
    save_path="./results_planning_inference_vlm/",
    use_grid_mask=False,  # Disable grid mask for inference
    frozen=False,
    use_lora=True,
    tokenizer=llm_path,
    lm_head=llm_path,
    use_pred_traj_seq=True,
    use_xy=True,
    kmeans_anchor_path="data/nuscenes/kmeans_plan_36.npy",
    use_inverse_l2=True,
    use_kmeans_traj=True,
    use_rag=True,
    use_gt_index=False,
    use_index_0=False,
    # Vision backbone
    img_backbone=dict(
        type="EVAViT",
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        window_block_indexes=(
            list(range(0, 2))
            + list(range(3, 5))
            + list(range(6, 8))
            + list(range(9, 11))
            + list(range(12, 14))
            + list(range(15, 17))
            + list(range(18, 20))
            + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True,
        frozen=False,
    ),
    # Image head (detection)
    img_head=dict(
        type="PETRHeadImage",
        num_classes=10,
        in_channels=1024,
        out_dims=None,
        num_query=600,
        with_mask=True,
        num_extra=num_extra,
        n_control=11,
        topk_proposals=100,
        num_propagated=0,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0],
        transformer=dict(
            type="PETRTemporalTransformer",
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
    ),
    # Map head (lane detection)
    map_head=dict(
        type="PETRHeadMapWithImg",
        num_classes=1,
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=num_extra,
        n_control=11,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0],
        transformer=dict(
            type="PETRTemporalTransformer",
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
        train_cfg=dict(
            assigner=dict(
                type="LaneHungarianAssigner",
                cls_cost=dict(type="FocalLossCost", weight=1.5),
                reg_cost=dict(type="LaneL1Cost", weight=0.02),
                iou_cost=dict(type="IoUCost", weight=0.0),
            )
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.02),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.0),
    ),
    # 3D detection head
    pts_bbox_head=dict(
        type="StreamPETRHeadWithImg",
        num_classes=10,
        in_channels=1024,
        out_dims=None,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=num_extra,
        n_control=11,
        match_with_velo=False,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type="PETRTemporalTransformer",
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # End-to-end planning head
    # e2e_head=dict(
    #     type='MotionPlanningHeadEgo',
    #     plan_loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=0.5,
    #     ),
    #     plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
    #     plan_loss_status=dict(type='L1Loss', loss_weight=1.0),
    #     num_extra=num_extra,
    #     use_seq_query=True,
    #     two_layer_cross_attn=True,
    #     kmeans_anchor_path='/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/kmeans/kmeans_plan_6.npy',
    # ),
    # Model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
    test_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.8,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=4096,
            max_num=500,
        )
    ),
)

# Data augmentation for inference - MUST match original training config
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

# ============================================================================
# 专门用于NeuroNCAP推理的数据处理管线
# ============================================================================
# 注意：这个管线专门为推理设计，不包含文件加载步骤
# VLM的prompt生成和tokenization在runner.py中处理

inference_pipeline = [
    # 图像预处理步骤
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=False),
    dict(
        type="ResizeMultiview3D",
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode="value",
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # 数据格式化 - 不包含VLM处理，因为数据已经由API提供
    dict(
        type="PETRFormatBundle3D",
        collect_keys=collect_keys,
        class_names=class_names,
        with_label=False,
    ),
    # 数据收集 - 只收集必要的字段
    dict(
        type="Collect3D",
        keys=["img"] + collect_keys,
        meta_keys=(
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "scene_token",
        ),
    ),
]

# ============================================================================
# 原始训练管线（保留用于参考）
# ============================================================================
# 注意：这个管线包含LoadAnnoatationVQATestSOLVE，用于训练时的VLM处理
# 推理时不使用这个管线，因为数据格式不同

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=False),
    dict(
        type="ResizeMultiview3D",
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode="value",
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="LoadAnnoatationVQATestSOLVE",
        base_vqa_path="./data/nuscenes/vqa/val/",
        base_conv_path="./data/nuscenes/conv/val/",
        base_counter_path="./data/nuscenes/eval_cf/",
        load_type=["closed_loop"],  # planning, closed_loop
        tokenizer=llm_path,
        use_trajemb_cot=True,
        use_gt_traj=False,
        use_cot_v1=False,
        use_pred_traj_seq=True,
        use_kmeans_traj=False,
        kmeans_pad_traj=True,
        use_rag=True,
        rag_topk=5,
        cat_pred_traj=True,
        cot_with_speed=True,
        use_ego_mlp=True,
        kmeans_path="data/nuscenes/kmeans_plan_36.npy",
        max_length=2048,
        keans_div6_path="/nfs/dataset-ofs-voyager-research/pqh/ColaVLA_private/data/kmeans_plan_ego_div6.npy",
        closed_refer_num=6,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="PETRFormatBundle3D",
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type="Collect3D",
                keys=["img", "input_ids", "vlm_labels", "pred_traj2"] + collect_keys,
                meta_keys=(
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "box_mode_3d",
                    "box_type_3d",
                    "img_norm_cfg",
                    "scene_token",
                ),
            ),
        ],
    ),
]

dataset_type = "CustomNuScenesDataset"
data_root = "./data/nuscenes/"
file_client_args = dict(backend="disk")

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,  # origin 4 debug 0
    val=dict(
        type=dataset_type,
        eval_mode=["lane", "det"],
        pipeline=test_pipeline,
        ann_file=data_root + "nuscenes2d_ego_temporal_infos_val.pkl",
        classes=class_names,
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        eval_mode=["lane", "det"],
        pipeline=test_pipeline,
        ann_file=data_root + "nuscenes2d_ego_temporal_infos_val.pkl",
        classes=class_names,
        modality=input_modality,
    ),
    shuffler_sampler=dict(
        type="InfiniteGroupEachSampleInBatchSampler",
        seq_split_num=2,
        warmup_split_num=10,
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# File client
file_client_args = dict(backend="disk")

# Set to False for inference to avoid memory reset issues
find_unused_parameters = False
