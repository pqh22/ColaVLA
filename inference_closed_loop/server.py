"""FastAPI server for ColaVLA inference in NeuroNCAP closed-loop testing."""

import argparse
import base64
import io
from typing import Dict, List, Union
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import zipfile

from data_types import (
    NUSCENES_CAM_ORDER,
    InferenceInputs,
    InferenceOutputs,
    InferenceAuxOutputs,
    ColaVLAInferenceInput,
)
from runner import ColaVLARunner


# Create FastAPI app with explicit configuration
app = FastAPI(
    title="ColaVLA Inference Server",
    version="1.0.0",
    description="ColaVLA inference server for NeuroNCAP closed-loop testing",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Root endpoint - explicitly define as first route
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return JSONResponse(
        content={
            "service": "ColaVLA Inference Server",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "/": "GET - API information (this endpoint)",
                "/alive": "GET - Health check",
                "/infer": "POST - Run inference on multi-view images",
                "/reset": "POST - Reset server state",
                "/docs": "GET - API documentation (Swagger UI)",
                "/redoc": "GET - API documentation (ReDoc)",
            },
            "usage": {
                "health_check": "curl http://localhost:9000/alive",
                "reset_server": "curl -X POST http://localhost:9000/reset",
                "run_test": "python inference/test_inference.py --host localhost --port 9000",
            },
        }
    )


@app.get("/alive", response_class=JSONResponse)
async def alive():
    """Check if the server is alive."""
    return JSONResponse(content=True)


@app.get("/health", response_class=JSONResponse)
async def health():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "ColaVLA Inference Server",
            "version": "1.0.0",
        }
    )


@app.post("/infer", response_class=JSONResponse)
async def infer(data: InferenceInputs) -> InferenceOutputs:
    """Run inference on the given data.

    This endpoint processes multi-view camera images and related sensor data
    to produce trajectory predictions and auxiliary outputs.
    """
    try:
        # printdata
        print(f"Debug: data: {data}")

        # preprocessAPIdata ColaVLA
        colavla_input = _preprocess_api_input(data)

        print(f"Debug: colavla_input: {colavla_input}")

        # Run inference
        colavla_output = colavla_runner.forward_inference(colavla_input)

        # Convert output to API format
        # ensure
        trajectory_list = colavla_output.trajectory.tolist()
        print(f"Debug: trajectory shape: {colavla_output.trajectory.shape}")
        print(f"Debug: trajectory type: {type(trajectory_list)}")
        print(f"Debug: trajectory content: {trajectory_list}")

        # validate
        if not isinstance(trajectory_list, list) or len(trajectory_list) == 0:
            raise ValueError(f"Invalid trajectory format: {trajectory_list}")

        return InferenceOutputs(
            trajectory=trajectory_list,
            aux_outputs=InferenceAuxOutputs(),  # other ouputs are set to None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/reset", response_class=JSONResponse)
async def reset_runner():
    """Reset the runner for a new sequence."""
    try:
        colavla_runner.reset()
        return JSONResponse(content=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Add a catch-all route for debugging
@app.get("/{path:path}")
async def catch_all(path: str):
    """Catch-all route for debugging."""
    return JSONResponse(
        content={
            "error": "Not Found",
            "path": path,
            "available_endpoints": [
                "/",
                "/alive",
                "/health",
                "/infer",
                "/reset",
                "/docs",
                "/redoc",
            ],
        },
        status_code=404,
    )


def _preprocess_api_input(raw: InferenceInputs) -> ColaVLAInferenceInput:
    """Preprocess API input data to ColaVLA internal format.

    This function follows the same logic as load_colavla_input_from_json
    but works directly with API input data instead of saved JSON files.

    Args:
        raw: Input data from the API (InferenceInputs Pydantic model)

    Returns:
        ColaVLA internal input format
    """
    # Decode images in nuScenes camera order
    # imgs = []
    # for cam in NUSCENES_CAM_ORDER:
    # b64 = raw.images[cam] # use attribute access dictionary
    #     buf = base64.b64decode(b64)
    #     # Try torch.load first (handles zip-formatted torch tensor payloads)
    #     arr = None
    #     try:
    #         tensor = torch.load(io.BytesIO(buf), map_location='cpu')
    #         if isinstance(tensor, torch.Tensor):
    #             arr = tensor.numpy()
    #     except Exception:
    #         arr = None
    #     if arr is None:
    #         # Try PIL
    #         try:
    #             img = Image.open(io.BytesIO(buf))
    #             if img.mode != 'RGB':
    #                 img = img.convert('RGB')
    #             arr = np.array(img)
    #         except Exception:
    #             # Fallback raw buffer assuming (900,1600,3)
    #             raw_u8 = np.frombuffer(buf, dtype=np.uint8)
    #             if raw_u8.size == 900 * 1600 * 3:
    #                 arr = raw_u8.reshape(900, 1600, 3)
    #             else:
    #                 raise ValueError(f"Unsupported image encoding for {cam}. Bytes={len(buf)}")
    #     if arr.ndim != 3 or arr.shape[-1] != 3:
    #         raise ValueError(f"Decoded image for {cam} has invalid shape: {arr.shape}")
    #     imgs.append(arr)
    # imgs = np.stack(imgs, axis=0)
    imgs = _bytestr_to_numpy([raw.images[c] for c in NUSCENES_CAM_ORDER])

    # # checkshapedata image rendering node
    # import os
    # print("Shape:", imgs.shape) # (6, 900, 1600, 3)
    # print("Dtype:", imgs.dtype) # uint8
    # assert imgs.ndim == 4 and imgs.shape[-1] == 3, "Unexpected image shape"
    # assert imgs.dtype == np.uint8, "Unexpected dtype"

    # # savepath
    # save_dir = "/nfs/dataset-ofs-voyager-research/pqh/ColaVLA_private/img_closeloop"
    # os.makedirs(save_dir, exist_ok=True)

    # # saveimage
    # for i, img in enumerate(imgs):
    #     img_path = os.path.join(save_dir, f"cam_{i}.png")
    #     Image.fromarray(img).save(img_path)
    #     print(f"Saved: {img_path}")

    # Ego and sensor poses
    ego2world = np.array(raw.ego2world, dtype=np.float32)  # use attribute access
    lidar2ego = np.array(raw.calibration.lidar2ego, dtype=np.float32)  # use attribute access
    lidar2world = ego2world @ lidar2ego

    # Per-camera intrinsics/extrinsics and lidar2img
    intrinsics = []
    extrinsics = []
    lidar2imgs = []
    for cam in NUSCENES_CAM_ORDER:
        K = np.array(
            raw.calibration.camera2image[cam], dtype=np.float32
        )  # use attribute access
        cam2ego = np.array(
            raw.calibration.camera2ego[cam], dtype=np.float32
        )  # use attribute access
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
    can_bus = np.array(raw.canbus, dtype=np.float32)  # use attribute access
    timestamp_s = float(raw.timestamp) / 1e6  # use attribute access
    command = int(raw.command)  # use attribute access

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


def _build_colavla_input(data: InferenceInputs) -> ColaVLAInferenceInput:
    """Convert API input to ColaVLA internal format.

    Args:
        data: Input data from the API

    Returns:
        ColaVLA internal input format
    """
    # Decode images from base64
    imgs = _decode_images([data.images[c] for c in NUSCENES_CAM_ORDER])

    # Convert poses to numpy arrays
    ego2world = np.array(data.ego2world)
    lidar2ego = np.array(data.calibration.lidar2ego)

    # Compute lidar pose in world frame
    lidar2world = ego2world @ lidar2ego

    # Prepare camera intrinsics and extrinsics
    intrinsics = []
    extrinsics = []
    lidar2imgs = []

    for cam in NUSCENES_CAM_ORDER:
        # Camera intrinsics
        cam_intrinsic = np.array(data.calibration.camera2image[cam])
        intrinsics.append(cam_intrinsic)

        # Camera extrinsics (cam2ego)
        cam2ego = np.array(data.calibration.camera2ego[cam])
        extrinsics.append(cam2ego)

        # Compute lidar2img transformation
        ego2cam = np.linalg.inv(cam2ego)
        cam2img = np.eye(4)
        cam2img[:3, :3] = cam_intrinsic
        lidar2cam = ego2cam @ lidar2ego
        lidar2img = cam2img @ lidar2cam
        lidar2imgs.append(lidar2img)

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    lidar2img = np.stack(lidar2imgs, axis=0)

    # Prepare CAN bus signals
    can_bus_signals = np.array(data.canbus)

    return ColaVLAInferenceInput(
        imgs=imgs,
        ego_pose=ego2world,
        lidar_pose=lidar2world,
        lidar2img=lidar2img,
        timestamp=data.timestamp / 1e6,  # Convert to seconds
        can_bus_signals=can_bus_signals,
        command=data.command,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def _decode_images(image_base64_list: List[Union[str, bytes]]) -> np.ndarray:
    """Decode a list of base64-encoded images to numpy array.

    Args:
        image_base64_list: List of base64-encoded image payloads

    Returns:
        Numpy array of shape (n_cams, h, w, c) in RGB format
    """
    imgs = []
    for img_base64 in image_base64_list:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(img_base64)

        # Load image using PIL
        try:
            img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
        except Exception:
            # Fallback 1: check if this is a zipped image buffer
            try:
                if img_bytes.startswith(b"PK\x03\x04"):
                    with zipfile.ZipFile(io.BytesIO(img_bytes)) as zf:
                        names = [n for n in zf.namelist() if not n.endswith("/")]
                        if len(names) == 0:
                            raise ValueError("empty zip")
                        with zf.open(names[0]) as f:
                            inner_bytes = f.read()
                        # try PIL first
                        try:
                            inner_img = Image.open(io.BytesIO(inner_bytes))
                            if inner_img.mode != "RGB":
                                inner_img = inner_img.convert("RGB")
                            img_array = np.array(inner_img)
                            imgs.append(img_array)
                            continue
                        except Exception:
                            # try OpenCV
                            buf = np.frombuffer(inner_bytes, dtype=np.uint8)
                            inner_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if inner_cv is not None:
                                img_array = cv2.cvtColor(inner_cv, cv2.COLOR_BGR2RGB)
                                imgs.append(img_array)
                                continue
                            # try raw fallback
                            raw = np.frombuffer(inner_bytes, dtype=np.uint8)
                            expected = 900 * 1600 * 3
                            if raw.size == expected:
                                img_array = raw.reshape(900, 1600, 3)
                                imgs.append(img_array)
                                continue
                    # if zip path didn't continue, fallthrough to other fallbacks
            except Exception:
                pass
            # Fallback 2: try OpenCV decode directly
            try:
                buf = np.frombuffer(img_bytes, dtype=np.uint8)
                img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_cv is None:
                    raise ValueError("cv2.imdecode failed")
                # BGR->RGB
                img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            except Exception:
                # Fallback 3: interpret as raw uint8 buffer (H=900, W=1600, C=3)
                raw = np.frombuffer(img_bytes, dtype=np.uint8)
                expected = 900 * 1600 * 3
                if raw.size != expected:
                    raise
                img_array = raw.reshape(900, 1600, 3)
        imgs.append(img_array)

    return np.stack(imgs, axis=0)


def _bytestr_to_numpy(pngs: List[bytes]) -> np.ndarray:
    """Convert a list of png bytes to a numpy array of shape (n, h, w, c)."""
    imgs = []
    for png in pngs:
        # using torch load as we use torch save on rendering node
        img = torch.load(io.BytesIO(png)).clone()
        imgs.append(img.numpy())

    return np.stack(imgs, axis=0)


def _setup_colavla_runner(args) -> ColaVLARunner:
    """Setup the ColaVLA runner with the given arguments.

    Args:
        args: Command line arguments

    Returns:
        Configured ColaVLA runner
    """
    device = torch.device(args.device)

    runner = ColaVLARunner(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        model_type=args.model_type,
        enable_visualization=args.enable_vis,
        vis_output_dir=args.vis_output_dir,
        vis_scenario=args.vis_scenario,
        vis_sequence=args.vis_sequence,
        vis_save_images=args.vis_save_images,
        vis_save_trajectories=args.vis_save_trajectories,
        vis_save_calibration=args.vis_save_calibration,
        vis_create_overlays=args.vis_create_overlays,
    )

    return runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColaVLA Inference Server")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the model config file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=9000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--model_type", type=str, default="vlm", help="Model type (vlm/vla)"
    )

    # Visualization arguments
    parser.add_argument(
        "--enable-vis",
        action="store_true",
        help="Enable visualization and data logging",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=str,
        default="output/visualization",
        help="Base directory for visualization outputs",
    )
    parser.add_argument(
        "--vis-scenario",
        type=str,
        default="",
        help="Scenario name (e.g., frontal, side, stationary)",
    )
    parser.add_argument(
        "--vis-sequence", type=str, default="", help="Sequence number (e.g., 103, 108)"
    )
    parser.add_argument(
        "--vis-save-images",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Save raw camera images (default: True)",
    )
    parser.add_argument(
        "--vis-save-trajectories",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Save trajectory data as JSON (default: True)",
    )
    parser.add_argument(
        "--vis-save-calibration",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Save calibration parameters (default: True)",
    )
    parser.add_argument(
        "--vis-create-overlays",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Create trajectory overlay visualizations (default: False)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Starting ColaVLA Inference Server")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model type: {args.model_type}")
    print("=" * 60)

    # Setup global runner instance
    print("Setting up ColaVLA runner...")
    colavla_runner = _setup_colavla_runner(args)
    print("✅ Runner setup completed")

    # Start the server
    print("Starting FastAPI server...")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if args.workers > 1 else None,
        loop="asyncio",
    )
