"""Data types for ColaVLA inference API."""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np
from pydantic import BaseModel, Field, Base64Bytes


# Camera order following nuScenes convention
NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class Calibration(BaseModel):
    """Calibration data."""

    camera2image: Dict[str, List[List[float]]] = Field(
        description="Camera intrinsics. The keys are the camera names."
    )
    camera2ego: Dict[str, List[List[float]]] = Field(
        description="Camera extrinsics. The keys are the camera names."
    )
    lidar2ego: List[List[float]] = Field(description="Lidar extrinsics.")


class InferenceInputs(BaseModel):
    """Input data for inference."""

    images: Dict[str, Base64Bytes] = Field(
        description="Camera images in base64-encoded PNG format. The keys are the camera names."
    )
    ego2world: List[List[float]] = Field(description="Ego pose in the world frame.")
    canbus: List[float] = Field(description="CAN bus signals.")
    timestamp: int = Field(
        description="Timestamp of the current frame in microseconds."
    )
    command: Literal[0, 1, 2] = Field(
        description="Command of the current frame. 0: right, 1: left, 2: straight"
    )
    calibration: Calibration = Field(description="Calibration data.")


class InferenceAuxOutputs(BaseModel):
    """Auxiliary outputs from inference."""

    objects_in_bev: Optional[List[List[float]]] = Field(
        default=None,
        description="Detected objects in BEV coordinates. N x [x, y, width, height, yaw]",
    )
    object_classes: Optional[List[str]] = Field(
        default=None, description="Object class names. (N,)"
    )
    object_scores: Optional[List[float]] = Field(
        default=None, description="Object detection scores. (N,)"
    )
    object_ids: Optional[List[int]] = Field(
        default=None, description="Object tracking IDs. (N,)"
    )
    future_trajs: Optional[List[List[List[List[float]]]]] = Field(
        default=None,
        description="Future trajectories. N x M modes x T timesteps x [x, y]",
    )
    lane_results: Optional[List[Dict]] = Field(
        default=None, description="Lane detection results"
    )
    text_output: Optional[str] = Field(
        default=None, description="Generated text from VLM"
    )


class InferenceOutputs(BaseModel):
    """Output/result from running the model."""

    trajectory: List[List[float]] = Field(
        description="Predicted trajectory in the ego frame. A list of (x, y) points in BEV."
    )
    aux_outputs: InferenceAuxOutputs = Field(description="Auxiliary outputs.")


@dataclass
class ColaVLAInferenceInput:
    """Internal data structure for ColaVLA inference."""

    imgs: np.ndarray
    """shape: (n-cams (6), h (900), w (1600), c (3)) | images in RGB format as uint8"""
    ego_pose: np.ndarray
    """shape: (4, 4) | ego pose in global frame"""
    lidar_pose: np.ndarray
    """shape: (4, 4) | lidar pose in global frame"""
    lidar2img: np.ndarray
    """shape: (n-cams (6), 4, 4) | lidar2img transformation matrix"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    can_bus_signals: np.ndarray
    """shape: (18,) | CAN bus signals with ego pose information"""
    command: int
    """0: right, 1: left, 2: straight"""
    intrinsics: np.ndarray
    """shape: (n-cams (6), 3, 3) | camera intrinsics"""
    extrinsics: np.ndarray
    """shape: (n-cams (6), 4, 4) | camera extrinsics (cam2ego)"""


@dataclass
class ColaVLAAuxOutputs:
    """Auxiliary outputs from ColaVLA."""

    objects_in_bev: np.ndarray
    """N x [x, y, width, height, yaw]"""
    object_classes: List[str]
    """(N,)"""
    object_scores: np.ndarray
    """(N,)"""
    object_ids: np.ndarray
    """(N,)"""
    future_trajs: np.ndarray
    """N x M modes x T timesteps x [x, y]"""
    lane_results: List[Dict]
    """Lane detection results"""
    text_output: str
    """Generated text from VLM"""

    def to_json(self) -> dict:
        """Convert to JSON-serializable format."""
        n_objects = len(self.object_classes)
        return dict(
            objects_in_bev=self.objects_in_bev.tolist() if n_objects > 0 else None,
            object_classes=self.object_classes if n_objects > 0 else None,
            object_scores=self.object_scores.tolist() if n_objects > 0 else None,
            object_ids=self.object_ids.tolist() if n_objects > 0 else None,
            future_trajs=self.future_trajs.tolist() if n_objects > 0 else None,
            lane_results=self.lane_results if self.lane_results else None,
            text_output=self.text_output if self.text_output else None,
        )

    @classmethod
    def empty(cls) -> "ColaVLAAuxOutputs":
        """Create empty auxiliary outputs."""
        return cls(
            objects_in_bev=np.zeros((0, 5)),
            object_classes=[],
            object_scores=np.zeros((0,)),
            object_ids=np.zeros((0,), dtype=int),
            future_trajs=np.zeros((0, 6, 12, 2)),
            lane_results=[],
            text_output="",
        )


@dataclass
class ColaVLAInferenceOutput:
    """Output from ColaVLA inference."""

    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in ego frame @ 2Hz"""
    aux_outputs: ColaVLAAuxOutputs
    """auxiliary outputs such as objects, lanes, and text"""
