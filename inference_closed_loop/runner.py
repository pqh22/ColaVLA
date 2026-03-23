"""ColaVLA inference runner for NeuroNCAP closed-loop testing."""

import copy
import io
import os
import uuid
import sys
from typing import List, Optional

import base64
import json
import numpy as np
import projects.mmdet3d_plugin
import re
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from nuscenes.eval.common.utils import quaternion_yaw
from PIL import Image
from pyquaternion import Quaternion

from data_types import (
    NUSCENES_CAM_ORDER,
    ColaVLAInferenceInput,
    ColaVLAInferenceOutput,
    ColaVLAAuxOutputs,
)
from pathlib import Path

# VLM tokens - Simplified format
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_TRAJ_TOKEN = "<traj>"


# ------------------------
# JSON debug I/O utilities
# ------------------------
def load_colavla_input_from_json(json_path: str) -> ColaVLAInferenceInput:
    """Load a saved inference JSON and convert to ColaVLAInferenceInput.
    This loader is robust to images saved as torchserialized tensors (zip bytes) or standard PNG/JPEG.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Decode images in nuScenes camera order
    imgs = []
    for cam in NUSCENES_CAM_ORDER:
        b64 = raw["images"][cam]
        buf = base64.b64decode(b64)
        # Try torch.load first (handles zip-formatted torch tensor payloads)
        arr = None
        try:
            tensor = torch.load(io.BytesIO(buf), map_location="cpu")
            if isinstance(tensor, torch.Tensor):
                arr = tensor.numpy()
        except Exception:
            arr = None
        if arr is None:
            # Try PIL
            try:
                img = Image.open(io.BytesIO(buf))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img)
            except Exception:
                # Fallback raw buffer assuming (900,1600,3)
                raw_u8 = np.frombuffer(buf, dtype=np.uint8)
                if raw_u8.size == 900 * 1600 * 3:
                    arr = raw_u8.reshape(900, 1600, 3)
                else:
                    raise ValueError(
                        f"Unsupported image encoding for {cam}. Bytes={len(buf)}"
                    )
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Decoded image for {cam} has invalid shape: {arr.shape}")
        imgs.append(arr)
    imgs = np.stack(imgs, axis=0)

    # Ego and sensor poses
    ego2world = np.array(raw["ego2world"], dtype=np.float32)
    lidar2ego = np.array(raw["calibration"]["lidar2ego"], dtype=np.float32)
    lidar2world = ego2world @ lidar2ego

    # Per-camera intrinsics/extrinsics and lidar2img
    intrinsics = []
    extrinsics = []
    lidar2imgs = []
    for cam in NUSCENES_CAM_ORDER:
        K = np.array(raw["calibration"]["camera2image"][cam], dtype=np.float32)
        cam2ego = np.array(raw["calibration"]["camera2ego"][cam], dtype=np.float32)
        intrinsics.append(K)
        extrinsics.append(cam2ego)
        ego2cam = np.linalg.inv(cam2ego)
        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :3] = K
        lidar2cam = ego2cam @ lidar2ego
        lidar2img = cam2img @ lidar2cam
        lidar2imgs.append(lidar2img)
    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    lidar2img = np.stack(lidar2imgs, axis=0)

    # CAN bus, time, command
    can_bus = np.array(raw["canbus"], dtype=np.float32)
    timestamp_s = float(raw["timestamp"]) / 1e6
    command = int(raw["command"])

    return ColaVLAInferenceInput(
        imgs=imgs,
        ego_pose=ego2world,
        lidar_pose=lidar2world,
        lidar2img=lidar2img,
        timestamp=timestamp_s,
        can_bus_signals=can_bus,
        command=command,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def _pad_intrinsic_to_4x4(K: np.ndarray) -> np.ndarray:
    """K: (3,3) or (4,4) -> (4,4)"""
    if K is None:
        return None
    K = np.asarray(K)
    if K.shape == (4, 4):
        return K.astype(np.float32)
    assert K.shape == (3, 3), f"Intrinsic must be 3x3 or 4x4, got {K.shape}"
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = K.astype(np.float32)
    return out


def _build_cam_infos(intrinsics_44, extrinsics_44):
    """Build a minimal `cam_infos` dict and set unavailable fields to None."""
    cam_infos = {}
    for i, cam in enumerate(NUSCENES_CAM_ORDER):
        K44 = intrinsics_44[i] if intrinsics_44 is not None else None
        E = extrinsics_44[i] if extrinsics_44 is not None else None

        cam_infos[cam] = {
            "data_path": None,
            "type": "camera",
            "sample_data_token": None,
            "sensor2ego_translation": E[:3, 3].tolist()
            if isinstance(E, np.ndarray)
            else None,
            # Quaternion can be added later if needed; keep None for now.
            "sensor2ego_rotation": None,
            "ego2global_translation": None,
            "ego2global_rotation": None,
            "timestamp": None,
            "sensor2lidar_rotation": None,
            "sensor2lidar_translation": None,
            "cam_intrinsic": (
                K44[:3, :3].tolist() if isinstance(K44, np.ndarray) else None
            ),
        }
    return cam_infos


def _select_pipeline_until_pad(config):
    """Select preprocessing transforms up to PadMultiViewImage from test_pipeline."""
    assert hasattr(config, "test_pipeline"), "config is missing test_pipeline"
    # Keep transforms up to PadMultiViewImage.
    picked = []
    for transform in config.test_pipeline:
        if transform["type"] in [
            "ResizeCropFlipRotImage",
            "ResizeMultiview3D",
            "NormalizeMultiviewImage",
            "PadMultiViewImage",
        ]:
            picked.append(transform)
        if transform["type"] == "PadMultiViewImage":
            break
    return Compose(picked)


def save_nuscenes_imgs(imgs, cam_names, out_dir, frame_name):
    out_dir = Path(out_dir)
    for cam, arr in zip(cam_names, imgs):
        # 1) Ensure shape is (H, W, 3)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))  # (3,H,W) ➜ (H,W,3)

        # 2) Normalize float arrays to 0-255 and cast to uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255
        arr = arr.astype(np.uint8, copy=False)

        # 3) Convert BGR to RGB (nuScenes images are usually OpenCV/BGR)
        arr = arr[..., ::-1]

        # 4) Ensure contiguous memory and save
        arr = np.ascontiguousarray(arr)
        (out_dir / cam).mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(
            out_dir / cam / f"{frame_name}.jpg", quality=95
        )


class ColaVLARunner:
    """ColaVLA inference runner for closed-loop testing."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        model_type: str,
        enable_visualization: bool = False,
        vis_output_dir: str = "output/visualization",
        vis_scenario: str = "",
        vis_sequence: str = "",
        vis_save_images: bool = True,
        vis_save_trajectories: bool = True,
        vis_save_calibration: bool = True,
        vis_create_overlays: bool = False,
        vis_cameras: tuple = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
    ):
        """Initialize the ColaVLA runner.

        Args:
            config_path: Path to the model config file
            checkpoint_path: Path to the model checkpoint
            device: PyTorch device to run inference on
            model_type: Model type ('vla' or 'vlm')
            enable_visualization: Whether to enable visualization/data logging
            vis_output_dir: Base directory for outputs
            vis_scenario: Scenario name (e.g., 'frontal', 'side', 'stationary')
            vis_sequence: Sequence number (e.g., '103', '108')
            vis_save_images: Save raw camera images
            vis_save_trajectories: Save trajectory data as JSON
            vis_save_calibration: Save calibration parameters
            vis_create_overlays: Create trajectory overlay visualizations
            vis_cameras: Tuple of camera names to save
        """
        # Validate runtime environment
        self._check_environment()

        # Visualization configuration
        self.enable_visualization = enable_visualization
        self.vis_frame_idx = 0
        self.vis_config = {
            "output_dir": vis_output_dir,
            "scenario": vis_scenario,
            "sequence": vis_sequence,
            "save_images": vis_save_images,
            "save_trajectories": vis_save_trajectories,
            "save_calibration": vis_save_calibration,
            "create_overlays": vis_create_overlays,
            "cameras": vis_cameras,
        }

        if enable_visualization:
            from pathlib import Path

            # Build output directory structure: output_dir/scenario-sequence/
            if vis_scenario and vis_sequence:
                self.vis_output_root = (
                    Path(vis_output_dir) / f"{vis_scenario}-{vis_sequence}"
                )
            elif vis_sequence:
                self.vis_output_root = Path(vis_output_dir) / f"seq-{vis_sequence}"
            else:
                self.vis_output_root = Path(vis_output_dir) / "default"

            # Create output subdirectories
            if vis_save_images:
                for cam in vis_cameras:
                    (self.vis_output_root / "images" / cam).mkdir(
                        parents=True, exist_ok=True
                    )
            if vis_save_trajectories:
                (self.vis_output_root / "trajectories").mkdir(
                    parents=True, exist_ok=True
                )
            if vis_save_calibration:
                (self.vis_output_root / "calibration").mkdir(
                    parents=True, exist_ok=True
                )
            if vis_create_overlays:
                (self.vis_output_root / "visualization" / "traj_overlay").mkdir(
                    parents=True, exist_ok=True
                )

            print(f"✅ Visualization enabled")
            print(f"   Output: {self.vis_output_root}")
            print(f"   Scenario: {vis_scenario or 'N/A'}")
            print(f"   Sequence: {vis_sequence or 'N/A'}")
            print(f"   Save images: {vis_save_images}")
            print(f"   Save trajectories: {vis_save_trajectories}")
            print(f"   Save calibration: {vis_save_calibration}")
            print(f"   Create overlays: {vis_create_overlays}")

        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = Config.fromfile(config_path)
        self.config = config
        self.model_type = model_type
        # Build model
        try:
            self.model = build_model(
                config.model, train_cfg=None, test_cfg=config.get("test_cfg")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build model: {e}")

        # Load checkpoint
        if "none" in checkpoint_path:
            print(
                "   ⚠️ No checkpoint_path provided, using randomly initialized weights"
            )
            self.classes = list(getattr(self.config, "class_names", []))
        elif checkpoint_path is not None and len(str(checkpoint_path)) > 0:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(
                self.model,
                checkpoint_path,
                map_location="cpu",
                strict=False,
                revise_keys=[(r"^module\.", "")],
            )
            meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
            classes = meta.get("CLASSES", None)
            if classes is None and hasattr(self.config, "class_names"):
                classes = self.config.class_names
            self.classes = list(classes) if classes is not None else []
            if hasattr(self.model, "CLASSES"):
                self.model.CLASSES = self.classes
            print("   ✅ Checkpoint loaded")
        else:
            print(
                "   ⚠️ No checkpoint_path provided, using randomly initialized weights"
            )
            self.classes = list(getattr(self.config, "class_names", []))

        # Move model to device
        print(f"🔄 Moving model to device: {device}")
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        print(f"   ✅ Model moved to {device}")

        # Initialize state
        print("🔄 Initializing runner state...")
        self.reset()
        print("   ✅ Runner state initialized")

        print("🎉 ColaVLA runner setup completed successfully!")

    def _check_environment(self):
        """Validate runtime environment."""
        print("🔍 Checking environment...")

        # Check required Python packages
        try:
            import mmcv
            import mmdet3d
            import torch

            print("   ✅ Required packages available")
        except ImportError as e:
            print(f"   ❌ Missing required package: {e}")
            raise ImportError(f"Missing required package: {e}")

        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ CUDA available: {gpu_count} GPUs")
            print(f"      - GPU 0: {gpu_name}")
            print(f"      - GPU 0 Memory: {gpu_memory:.1f} GB")
        else:
            print("   ⚠️  CUDA not available, using CPU mode")

        print("   ✅ Environment check completed")

    def _build_inference_pipeline(self, config) -> Compose:
        """Build inference preprocessing pipeline."""
        # Prefer a dedicated inference pipeline when available
        if hasattr(config, "inference_pipeline"):
            print("   📋 Using inference_pipeline for NeuroNCAP inference")
            pipeline = Compose(config.inference_pipeline)
            print(
                f"   ✅ Inference pipeline built with {len(config.inference_pipeline)} stages"
            )
            return pipeline

        # Fall back to building from test_pipeline
        if hasattr(config, "test_pipeline"):
            print("   📋 Building inference pipeline from test_pipeline")
            pipeline_configs = []

            for stage_config in config.test_pipeline:
                stage_type = stage_config["type"]
                # Skip file loaders because API already provides tensors/images
                if stage_type == "LoadMultiViewImageFromFiles":
                    continue
                # Skip VQA annotation loaders during this path
                elif stage_type == "LoadAnnoatationVQATestSOLVE":
                    continue
                elif stage_type == "LoadAnnoatationPUREQA":
                    continue
                elif stage_type == "LoadAnnoatationVQATrajCF":
                    continue
                # Expand MultiScaleFlipAug3D and keep inner transforms
                elif stage_type == "MultiScaleFlipAug3D":
                    transforms = stage_config.get("transforms", [])
                    for transform in transforms:
                        if transform["type"] != "Collect3D":
                            pipeline_configs.append(transform)
                # Keep other preprocessing stages
                elif stage_type in [
                    "ResizeCropFlipRotImage",
                    "ResizeMultiview3D",
                    "NormalizeMultiviewImage",
                    "PadMultiViewImage",
                    "PETRFormatBundle3D",
                ]:
                    pipeline_configs.append(stage_config)
        else:
            # Default fallback pipeline
            ida_aug_conf = {
                "resize_lim": (0.37, 0.45),
                "final_dim": (320, 640),
                "bot_pct_lim": (0.0, 0.0),
                "rot_lim": (0.0, 0.0),
                "H": 900,
                "W": 1600,
                "rand_flip": False,
            }
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
            )

            pipeline_configs = [
                dict(
                    type="ResizeCropFlipRotImage",
                    data_aug_conf=ida_aug_conf,
                    training=False,
                ),
                dict(
                    type="ResizeMultiview3D",
                    img_scale=(640, 640),
                    keep_ratio=False,
                    multiscale_mode="value",
                ),
                dict(type="NormalizeMultiviewImage", **img_norm_cfg),
                dict(type="PadMultiViewImage", size_divisor=32),
            ]

        return Compose(pipeline_configs)

    def reset(self):
        """Reset the runner for a new sequence."""
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        # Reset visualization frame counter
        self.vis_frame_idx = 0

        # Reset model memory if it has memory mechanisms
        if hasattr(self.model, "pts_bbox_head") and hasattr(
            self.model.pts_bbox_head, "reset_memory"
        ):
            self.model.pts_bbox_head.reset_memory()
        if hasattr(self.model, "map_head") and hasattr(
            self.model.map_head, "reset_memory"
        ):
            self.model.map_head.reset_memory()

        # Reset test flag if exists
        if hasattr(self.model, "test_flag"):
            self.model.test_flag = False

    def _preprocess_canbus(self, input_data: ColaVLAInferenceInput):
        """
        Preprocess CAN bus signals to the 13-D vector used by the model.
        Expected raw layout (len=16):
            [ pos(3), orient(4), accel(3), rotation_rate(3), vel(3) ]
        Returned layout (len=13):
            [ orient(4), accel(3), rotation_rate(3), vel(3) ]
        """
        # Robustly handle list/np/torch
        can_raw = input_data.can_bus_signals
        try:
            if not isinstance(can_raw, np.ndarray):
                can_raw = np.asarray(can_raw)
        except Exception:
            pass  # assume it's already array-like

        if can_raw.shape[-1] != 16:
            raise ValueError(
                f"Expected 16-D CAN bus vector, got shape {can_raw.shape}."
            )

        # Indices given your stated format:
        # 0:3   -> pos (drop)
        # 3:7   -> orientation (keep)
        # 7:10  -> accel (keep)
        # 10:13 -> rotation_rate (keep)
        # 13:16 -> vel (keep)
        orient = can_raw[3:7]
        accel = can_raw[7:10]
        rotation_rate = can_raw[10:13]
        vel = can_raw[13:16]

        canbus_13d = np.concatenate([orient, accel, rotation_rate, vel], axis=-1)
        return canbus_13d

    def preprocess_closed_loop_for_model(
        self, input_data: ColaVLAInferenceInput
    ) -> dict:
        """Build model-ready data by running the full preprocessing pipeline."""
        pipeline_to_pad = _select_pipeline_until_pad(self.config)

        imgs = input_data.imgs
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            imgs_bgr_list = [imgs[i, :, :, ::-1].copy() for i in range(imgs.shape[0])]
        elif isinstance(imgs, list):
            imgs_bgr_list = [im[:, :, ::-1].copy() for im in imgs]
        else:
            raise ValueError(
                f"Unexpected imgs type/shape: {type(imgs)} / {getattr(imgs, 'shape', None)}"
            )

        intrinsics_44 = None
        if getattr(input_data, "intrinsics", None) is not None:
            intrinsics_44 = [_pad_intrinsic_to_4x4(K) for K in input_data.intrinsics]

        extrinsics_44 = None
        if getattr(input_data, "extrinsics", None) is not None:
            extrinsics_44 = [
                np.asarray(E, dtype=np.float32) if E is not None else None
                for E in input_data.extrinsics
            ]

        ego_pose = getattr(input_data, "ego_pose", None)
        if ego_pose is None:
            ego_pose = getattr(input_data, "lidar_pose", None)
        ego_pose = np.asarray(ego_pose, dtype=np.float32)
        ego_pose_inv = np.linalg.inv(ego_pose)

        canbus_13d = self._preprocess_canbus(input_data)
        cam_infos = _build_cam_infos(intrinsics_44, extrinsics_44)

        results = {
            "img": imgs_bgr_list,
            "sweeps": [],
            "intrinsics": intrinsics_44,
            "extrinsics": extrinsics_44,
            "lidar2img": getattr(input_data, "lidar2img", None),
            "timestamp": input_data.timestamp,
            "img_timestamp": input_data.timestamp,
            "ego_pose": ego_pose.astype(np.float32),
            "ego_pose_inv": ego_pose_inv.astype(np.float32),
            "can_bus": canbus_13d.astype(np.float32),
            "command": int(getattr(input_data, "command", 2)),
            "location": getattr(input_data, "location", "boston"),
            "scene_token": getattr(self, "scene_token", str(uuid.uuid4())),
            "sample_idx": f"inference_{getattr(self, 'scene_token', 'scene')}",
            "prev_exists": False,
            "frame_idx": 0,
            "box_type_3d": "LiDAR",
            "box_mode_3d": "LiDAR",
            "cam_infos": cam_infos,
        }

        results_after_pad = pipeline_to_pad(results)

        vqa_cfg = None
        petr_cfg = None
        collect_cfg = None

        for stage in self.config.test_pipeline:
            stage_type = stage.get("type")
            if (
                stage_type
                in (
                    "LoadAnnoatationVQATestSOLVE",
                    "LoadAnnoatationPUREQA",
                    "LoadAnnoatationVQATrajCF",
                )
                and vqa_cfg is None
            ):
                vqa_cfg = stage

            if stage_type == "MultiScaleFlipAug3D":
                for transform in stage.get("transforms", []):
                    transform_type = transform.get("type")
                    if transform_type == "PETRFormatBundle3D":
                        petr_cfg = transform
                    elif transform_type == "Collect3D":
                        collect_cfg = transform

            if stage_type == "PETRFormatBundle3D" and petr_cfg is None:
                petr_cfg = stage
            elif stage_type == "Collect3D" and collect_cfg is None:
                collect_cfg = stage

        assert vqa_cfg is not None, (
            "LoadAnnoatationVQATestSOLVE was not found in config.test_pipeline"
        )
        assert petr_cfg is not None and collect_cfg is not None, (
            "PETRFormatBundle3D / Collect3D were not found in config.test_pipeline"
        )

        results_mid = Compose([vqa_cfg])(results_after_pad)
        if not isinstance(results_mid, dict):
            raise RuntimeError("VQA preprocessing returned None")

        for key in collect_cfg.get("keys", []):
            if key in results_mid:
                continue
            if key == "img":
                raise KeyError(
                    "'img' is missing; image tensors should be produced by preprocessing"
                )
            if key == "input_ids":
                results_mid["input_ids"] = np.asarray([1, 2, 3], dtype=np.int64)
            else:
                results_mid[key] = None

        data = Compose([petr_cfg, collect_cfg])(results_mid)
        if not isinstance(data, dict):
            raise RuntimeError("Post-VQA preprocessing returned None")
        return data

    def _unwrap_datacontainers(self, data: dict) -> dict:
        """Unwrap DataContainer objects into model-consumable structures."""
        from mmcv.parallel import DataContainer

        unwrapped_data = {}
        for key, value in data.items():
            try:
                # Keep `vlm_labels` as a DataContainer structure.
                if key == "vlm_labels":
                    unwrapped_data[key] = value
                    continue

                if isinstance(value, DataContainer):
                    # Unwrap a single DataContainer.
                    unwrapped_data[key] = value.data
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], DataContainer)
                ):
                    # Unwrap a flat list of DataContainer objects.
                    unwrapped_data[key] = [item.data for item in value]
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], list)
                    and len(value[0]) > 0
                    and isinstance(value[0][0], DataContainer)
                ):
                    # Unwrap nested DataContainer lists.
                    unwrapped_data[key] = [
                        [item.data for item in sublist] for sublist in value
                    ]
                else:
                    # Keep value unchanged.
                    unwrapped_data[key] = value
            except Exception as e:
                print(f"Warning: Failed to unwrap key '{key}': {e}")
                # Keep original value if unwrapping fails.
                unwrapped_data[key] = value

        # Ensure tensors are on the same device as the model.
        try:
            device = next(self.model.parameters()).device
            for key, value in unwrapped_data.items():
                if torch.is_tensor(value):
                    if value.dtype != torch.float32:
                        unwrapped_data[key] = value.to(dtype=torch.float32)
                    if value.device != device:
                        unwrapped_data[key] = value.to(device=device)
                elif isinstance(value, list) and len(value) > 0:
                    # Handle lists containing tensors.
                    for i, item in enumerate(value):
                        if torch.is_tensor(item):
                            if item.dtype != torch.float32:
                                value[i] = item.to(dtype=torch.float32)
                            if item.device != device:
                                value[i] = item.to(device=device)
        except Exception as e:
            print(f"Warning: Failed to move tensors to device: {e}")

        return unwrapped_data

    @torch.no_grad()
    def forward_inference(
        self,
        input_data: ColaVLAInferenceInput,
        reference_trajectory: Optional[torch.Tensor] = None,
        current_objects: Optional[list] = None,
        command: Optional[int] = None,
        crashed: bool = False,
    ) -> ColaVLAInferenceOutput:
        """Run inference on the input data.

        Args:
            input_data: Input data for inference
            reference_trajectory: Reference trajectory for visualization (Nx2, world coords)
            current_objects: List of ActorTrajectory objects for visualization
            command: Driving command for visualization (0=left, 1=right, 2=straight)
            crashed: Whether a collision occurred (for visualization)

        Returns:
            ColaVLAInferenceOutput with planned trajectory
        """
        data = self.preprocess_closed_loop_for_model(input_data)
        if not isinstance(data, dict):
            raise RuntimeError("Preprocessing output must be a dict")

        data = self._unwrap_datacontainers(data)
        if "pred_traj2" in data:
            data["img_metas"]["pred_traj2"] = data["pred_traj2"]
        else:
            data["img_metas"]["pred_traj2"] = None

        data["img_metas"]["vlm_labels"] = data["vlm_labels"]
        data["rescale"] = True

        # Fill required metadata fields.
        data["img_metas"]["box_mode_3d"] = 0  # Box3DMode.LIDAR
        # Import required box class.
        from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

        data["img_metas"]["box_type_3d"] = LiDARInstance3DBoxes

        data["img_metas"] = [[data["img_metas"]]]
        device = next(self.model.parameters()).device

        data["img"] = [data["img"].unsqueeze(0).to(device=device, dtype=torch.float32)]
        if (
            "msar" in self.model.save_path
            or "clashvla" in self.model.save_path
            or "colavla" in self.model.save_path
        ):
            data["input_ids"] = [[data["input_ids"]]]
        else:
            data["input_ids"] = [
                [
                    [
                        data["input_ids"][i].to(device=device, dtype=torch.int64)
                        for i in range(len(data["input_ids"]))
                    ]
                ]
            ]

        data["lidar2img"] = [
            data["lidar2img"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["intrinsics"] = [
            data["intrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["extrinsics"] = [
            data["extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["ego_pose"] = [
            data["ego_pose"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["ego_pose_inv"] = [
            data["ego_pose_inv"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["can_bus"] = [
            data["can_bus"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["command"] = [
            data["command"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["timestamp"] = [
            data["timestamp"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        data["img_timestamp"] = [
            data["img_timestamp"].unsqueeze(0).to(device=device, dtype=torch.float32)
        ]
        if (
            "results_planning_inference_vlm" in self.model.save_path
            or "results_planning_inference_vla" in self.model.save_path
        ):
            del data["pred_traj2"]
        del data["vlm_labels"]

        # Run model inference
        self.model.eval()
        result = self.model.forward_test(**data)

        trajectory = np.zeros((0, 2), dtype=np.float32)
        if self.model_type == "vlm":
            # Extract results
            text_traj = result[
                0
            ][
                "text_out"
            ][
                1
            ][
                "A"
            ][
                0
            ]  # 'The result is [PT, (+2.86, -0.02), (+5.46, -0.03), (+7.80, -0.04), (+9.90, -0.05), (+11.80, -0.06), (+13.46, -0.07)].'

            # Use regex to find all coordinate pairs in the form (+x.xx, -y.yy)
            matches = re.findall(r"\(([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+)\)", text_traj)

            # Convert to float and store in a NumPy array
            trajectory = np.array(matches, dtype=np.float32)
        elif self.model_type == "vla":
            trajectory = result[0]["vla_traj"]  # (6,2)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Create output
        output = ColaVLAInferenceOutput(
            trajectory=trajectory,
            aux_outputs=ColaVLAAuxOutputs.empty(),
        )

        return output

    def _extract_aux_outputs(self, result: dict) -> ColaVLAAuxOutputs:
        """Extract auxiliary outputs from model result."""
        if "pts_bbox" not in result or result["pts_bbox"] is None:
            return ColaVLAAuxOutputs.empty()

        pts_bbox = result["pts_bbox"]

        # Extract detection results
        objects_in_bev = np.zeros((0, 5))
        object_classes = []
        object_scores = np.zeros((0,))
        object_ids = np.zeros((0,), dtype=int)
        future_trajs = np.zeros((0, 6, 12, 2))

        if "boxes_3d" in pts_bbox:
            boxes_3d = pts_bbox["boxes_3d"]
            scores_3d = pts_bbox.get("scores_3d", torch.zeros(len(boxes_3d)))
            labels_3d = pts_bbox.get(
                "labels_3d", torch.zeros(len(boxes_3d), dtype=torch.long)
            )

            if len(boxes_3d) > 0:
                # Convert boxes to BEV format
                objects_in_bev = self._format_boxes(boxes_3d)
                object_scores = (
                    scores_3d.cpu().numpy() if torch.is_tensor(scores_3d) else scores_3d
                )
                if hasattr(self, "classes") and len(getattr(self, "classes", [])) > 0:
                    object_classes = [
                        self.classes[int(i)]
                        if int(i) < len(self.classes)
                        else f"class_{int(i)}"
                        for i in (
                            labels_3d.cpu().numpy()
                            if torch.is_tensor(labels_3d)
                            else labels_3d
                        )
                    ]
                else:
                    object_classes = [
                        int(i)
                        for i in (
                            labels_3d.cpu().numpy()
                            if torch.is_tensor(labels_3d)
                            else labels_3d
                        )
                    ]
                object_ids = np.arange(len(boxes_3d))  # Simple ID assignment

        # Extract lane results
        lane_results = []
        if "lane_results" in result:
            lane_results = result["lane_results"]

        # Extract text output
        text_output = ""
        if "text_out" in result:
            text_output = result["text_out"]

        return ColaVLAAuxOutputs(
            objects_in_bev=objects_in_bev,
            object_classes=object_classes,
            object_scores=object_scores,
            object_ids=object_ids,
            future_trajs=future_trajs,
            lane_results=lane_results,
            text_output=text_output,
        )

    def _format_boxes(self, boxes) -> np.ndarray:
        """Format 3D boxes to BEV format [x, y, w, l, yaw]."""
        if torch.is_tensor(boxes):
            if hasattr(boxes, "bev"):
                boxes_bev = boxes.bev.cpu().numpy()
            else:
                boxes_bev = boxes.cpu().numpy()
        else:
            boxes_bev = boxes

        # Convert to [x, y, w, l, yaw] format if needed
        if boxes_bev.shape[-1] >= 5:
            return boxes_bev[:, [0, 1, 3, 4, 6]]  # x, y, l, w, yaw
        else:
            return np.zeros((0, 5))

    def _transform_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Transform trajectory coordinates if needed.

        ColaVLA may use different coordinate conventions than the API expects.
        This function handles any necessary coordinate transformations.
        """
        # For NeuroNCAP, we need ego-frame coordinates
        # If ColaVLA outputs in different coordinate frame, transform here
        return trajectory

    def _visualize_step(
        self,
        input_data: ColaVLAInferenceInput,
        output: ColaVLAInferenceOutput,
        reference_trajectory: Optional[torch.Tensor] = None,
        current_objects: Optional[list] = None,
        command: Optional[int] = None,
        crashed: bool = False,
    ):
        """Perform visualization and data logging for one inference step.

        Args:
            input_data: Input data for inference
            output: Output from inference
            reference_trajectory: Not used in this simplified version
            current_objects: Not used in this simplified version
            command: Not used in this simplified version
            crashed: Not used in this simplified version
        """
        if not self.enable_visualization:
            return

        try:
            import json
            from PIL import Image, ImageDraw, ImageFont
            from pathlib import Path
            from data_types import NUSCENES_CAM_ORDER

            timestamp = int(input_data.timestamp * 1e6)
            frame_name = f"{timestamp:016d}"
            # 1. Save raw images
            if self.vis_config["save_images"]:
                for i, cam_name in enumerate(NUSCENES_CAM_ORDER):
                    if cam_name not in self.vis_config["cameras"]:
                        continue

                    # Get image data
                    img_np = input_data.imgs[i]  # (H, W, 3)

                    # Handle different data types and ranges
                    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                        # Float image, likely in [0, 1] or normalized
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                    elif img_np.dtype != np.uint8:
                        # Other types, convert to uint8
                        img_np = img_np.astype(np.uint8)

                    # Ensure it's contiguous
                    img_np = np.ascontiguousarray(img_np)

                    # Save image
                    img_path = (
                        self.vis_output_root / "images" / cam_name / f"{frame_name}.jpg"
                    )
                    Image.fromarray(img_np).save(img_path, quality=95)
            # 2. Save trajectory data
            if self.vis_config["save_trajectories"]:
                traj_data = {
                    "timestamp": timestamp,
                    "frame_idx": self.vis_frame_idx,
                    "trajectory": output.trajectory.tolist(),  # (6, 2)
                    "command": int(getattr(input_data, "command", 2)),
                }
                traj_file = self.vis_output_root / "trajectories" / f"{frame_name}.json"
                with open(traj_file, "w") as f:
                    json.dump(traj_data, f, indent=2)

            # 3. Save calibration parameters
            if self.vis_config["save_calibration"] and self.vis_frame_idx == 0:
                # Only save once per sequence (calibration doesn't change)
                calib_data = {
                    "intrinsics": {},
                    "extrinsics": {},
                    "lidar2img": {},
                }
                for i, cam_name in enumerate(NUSCENES_CAM_ORDER):
                    if cam_name not in self.vis_config["cameras"]:
                        continue
                    calib_data["intrinsics"][cam_name] = input_data.intrinsics[
                        i
                    ].tolist()
                    calib_data["extrinsics"][cam_name] = input_data.extrinsics[
                        i
                    ].tolist()
                    calib_data["lidar2img"][cam_name] = input_data.lidar2img[i].tolist()

                calib_file = self.vis_output_root / "calibration" / "calibration.json"
                with open(calib_file, "w") as f:
                    json.dump(calib_data, f, indent=2)

            # 4. Create trajectory overlay visualization (optional)
            if (
                self.vis_config["create_overlays"]
                and "CAM_FRONT" in self.vis_config["cameras"]
            ):
                self._create_trajectory_overlay(input_data, output, frame_name)

            self.vis_frame_idx += 1

        except Exception as e:
            print(
                f"⚠️  Warning: Visualization failed for frame {self.vis_frame_idx}: {e}"
            )
            import traceback

            traceback.print_exc()

    def _create_trajectory_overlay(
        self,
        input_data: ColaVLAInferenceInput,
        output: ColaVLAInferenceOutput,
        frame_name: str,
    ):
        """Create trajectory overlay on front camera image."""
        try:
            from PIL import Image, ImageDraw
            from data_types import NUSCENES_CAM_ORDER
            import numpy as np

            # Get front camera index
            cam_idx = NUSCENES_CAM_ORDER.index("CAM_FRONT")
            img_np = input_data.imgs[cam_idx].copy()  # (H, W, 3) RGB
            img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img)

            # Get calibration
            K = input_data.intrinsics[cam_idx][:3, :3]  # (3, 3)
            cam2ego = input_data.extrinsics[cam_idx]  # (4, 4)
            ego2cam = np.linalg.inv(cam2ego)

            # Project trajectory to image
            traj = output.trajectory  # (6, 2) in ego frame
            # Add z coordinate (assume ground level)
            traj_3d = np.zeros((len(traj), 3))
            traj_3d[:, 0] = traj[:, 0]  # x
            traj_3d[:, 1] = traj[:, 1]  # y
            traj_3d[:, 2] = 0.0  # z (ground)

            # Transform to camera frame
            traj_homo = np.concatenate(
                [traj_3d, np.ones((len(traj), 1))], axis=1
            )  # (6, 4)
            traj_cam = (ego2cam @ traj_homo.T).T  # (6, 4)

            # Project to image
            traj_cam_3d = traj_cam[:, :3]  # (6, 3)
            traj_img_homo = (K @ traj_cam_3d.T).T  # (6, 3)
            traj_img = traj_img_homo[:, :2] / traj_img_homo[:, 2:3]  # (6, 2)

            # Filter points in front of camera and inside image
            valid_mask = (
                (traj_cam[:, 2] > 0)
                & (traj_img[:, 0] >= 0)
                & (traj_img[:, 0] < img.width)
                & (traj_img[:, 1] >= 0)
                & (traj_img[:, 1] < img.height)
            )

            if valid_mask.any():
                # Draw trajectory
                points = traj_img[valid_mask].tolist()
                if len(points) > 1:
                    draw.line([tuple(p) for p in points], fill=(0, 255, 0), width=5)

                # Draw points
                for point in points:
                    x, y = point
                    r = 8
                    draw.ellipse(
                        [x - r, y - r, x + r, y + r],
                        fill=(255, 0, 0),
                        outline=(0, 0, 0),
                        width=2,
                    )

            # Save overlay
            overlay_path = (
                self.vis_output_root
                / "visualization"
                / "traj_overlay"
                / f"{frame_name}.jpg"
            )
            img.save(overlay_path, quality=95)

        except Exception as e:
            print(f"⚠️  Warning: Failed to create trajectory overlay: {e}")

    def finalize_visualization(self, metrics: Optional[dict] = None):
        """Finalize visualization and save metrics.

        Args:
            metrics: Dictionary of evaluation metrics to save
        """
        if not self.enable_visualization:
            return

        import json
        from pathlib import Path

        # Save metrics if provided
        if metrics is not None:
            metrics_file = self.vis_output_root / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Metrics saved to: {metrics_file}")

        # Save summary
        summary = {
            "scenario": self.vis_config["scenario"],
            "sequence": self.vis_config["sequence"],
            "total_frames": self.vis_frame_idx,
            "config": {
                "save_images": self.vis_config["save_images"],
                "save_trajectories": self.vis_config["save_trajectories"],
                "save_calibration": self.vis_config["save_calibration"],
                "create_overlays": self.vis_config["create_overlays"],
                "cameras": list(self.vis_config["cameras"]),
            },
        }
        summary_file = self.vis_output_root / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Visualization completed")
        print(f"   Output: {self.vis_output_root}")
        print(f"   Frames: {self.vis_frame_idx}")
        print(f"   Summary: {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ColaVLARunner on a saved JSON input"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to inference config"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint .pth"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to saved JSON input"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:idx or cpu")
    parser.add_argument(
        "--model-type",
        type=str,
        default="vla",
        choices=["vla", "vlm"],
        help="Model type",
    )
    parser.add_argument(
        "--enable-vis", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "--vis-output",
        type=str,
        default="output/vis_test",
        help="Visualization output directory",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    runner = ColaVLARunner(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        device=device,
        model_type=args.model_type,
        enable_visualization=args.enable_vis,
        vis_output_dir=args.vis_output,
    )

    infer_input = load_colavla_input_from_json(args.input)
    output = runner.forward_inference(infer_input)
    traj = output.trajectory
    np.set_printoptions(precision=3, suppress=True)
    print("Trajectory (6x2):\n", traj)

    if args.enable_vis:
        runner.finalize_visualization()
