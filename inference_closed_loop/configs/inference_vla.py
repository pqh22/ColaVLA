# ============================================================================
# ColaVLA Inference Configuration for NeuroNCAP Closed-loop Testing
# ============================================================================
# filededicated to NeuroNCAP inference
# training vlm_seq_384_cot_rag5_loade6qformere2e0320_noqa_headlr20_e10_cotspeed.py
#
# :
# - dedicated toinferencedataprocess
# - processVLM prompttokenization
# - NeuroNCAPdata
# - fileload
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
    type="Petr3DClassifyMSARPLSeqFormerv2",
    save_path=f"./results_planning/vla/",  # save path for vlm models.
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    tokenizer=llm_path,  # set to None if don't use llm head
    lm_head=llm_path,  # set to None if don't use llm head
    use_pred_traj_seq=True,
    use_xy=True,
    kmeans_anchor_path="data/nuscenes/kmeans_plan_36.npy",  # (3, 36, 6, 2)
    use_inverse_l2=True,
    use_kmeans_traj=True,
    use_rag=True,
    use_gt_index=False,
    use_index_0=False,
    use_grpo=False,
    traj_reg_loss_weight=80.0,
    use_pretrained_traj_queries=True,
    topk_mode_predict=3,
    cls_correct_idx=True,
    num_category=8,
    classification_loss_weight=1.0,
    category_latent_dim=4096,
    task_type="FEATURE_EXTRACTION",
    ms_traj_loss_weights=[0.5, 0.7, 1.0, 1.2, 1.5, 1.8],
    stage_index=[
        [5],
        [0, 5],
        [0, 2, 5],
        [0, 2, 4, 5],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 4, 5],
    ],  # fine index
    use_checkpointing=True,
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
    map_head=dict(
        type="PETRHeadMapWithImg",
        num_classes=1,
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,  # map query can't see vlm tokens
        topk_proposals=300,
        num_lane=1800,  # 300+1500
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
        ),  # dummy
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.02),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.0),
    ),  #
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
        n_control=11,  # align with centerline query defination
        match_with_velo=False,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
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
    # model training and testing settings
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
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
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
# dedicated toNeuroNCAPinferencedataprocess
# ============================================================================
# note inference fileload
# VLMprompttokenizationrunner.pyprocess

inference_pipeline = [
    # imagepreprocess
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=False),
    dict(
        type="ResizeMultiview3D",
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode="value",
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # dataformatting - VLMprocess dataAPI
    dict(
        type="PETRFormatBundle3D",
        collect_keys=collect_keys,
        class_names=class_names,
        with_label=False,
    ),
    # datacollection - collectionfields
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
# originaltraining kept for reference
# ============================================================================
# note LoadAnnoatationVQATestSOLVE trainingVLMprocess
# inference data

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
        type="LoadAnnoatationPUREQA",
        base_vqa_path="./data/nuscenes/vqa/train/",
        base_desc_path="./data/nuscenes/desc/train/",
        base_conv_path="./data/nuscenes/conv/train/",
        base_key_path="./data/nuscenes/keywords/train/",
        tokenizer=llm_path,
        max_length=2048,
        ignore_type=[],
        use_pred_traj=False,
        use_other_qa=False,
        use_pred_traj_seq=True,
        use_kmeans_traj=False,
        kmeans_pad_traj=True,
        use_xy=True,
        use_cot_v1=False,
        use_ego_mlp=True,
        use_rag=True,
        rag_topk=5,
        cat_pred_traj=True,
        cot_with_speed=True,
        kmeans_path="data/nuscenes/kmeans_plan_36.npy",
        lane_objs_info="./data/nuscenes/lane_obj_train.pkl",
        use_classv3=True,
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
                keys=["img", "input_ids", "vlm_labels"] + collect_keys,
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
                    "min_index",
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
