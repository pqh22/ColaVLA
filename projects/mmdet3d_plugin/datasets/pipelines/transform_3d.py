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

from imaplib import Commands
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image
from math import factorial
import cv2
import random
import copy
from transformers import AutoTokenizer
import json
import re
import os
from nuscenes.utils.geometry_utils import view_points
from typing import List, Tuple, Union, Dict, Any, Iterable
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely.geometry import box as canvas_box
from ..utils.data_utils import preprocess_traj, preprocess
from ..utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_TRAJ_TOKEN,
    DEFAULT_POINT_TOKEN,
    DEFAULT_EGO_TOKEN,
)
import math
import pickle
from ..utils.data_utils import tokenizer_image_traj_token

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3, suppress=True)


def get_category_index(direction: str, distance: str) -> int:
    # Translated note.
    mapping = {
        ("left turn", "short"): 0,
        ("left turn", "medium"): 1,
        ("left turn", "long"): 2,
        ("slight left turn", "short"): 3,
        ("slight left turn", "medium"): 4,
        ("slight left turn", "long"): 5,
        ("straight", "short"): 6,
        ("straight", "medium"): 7,
        ("straight", "long"): 8,
        ("slight right turn", "short"): 9,
        ("slight right turn", "medium"): 10,
        ("slight right turn", "long"): 11,
        ("right turn", "short"): 12,
        ("right turn", "medium"): 13,
        ("right turn", "long"): 14,
    }

    return mapping.get((direction, distance), -1)  # return -1


Combined = str
TableKey = Union[Combined, Iterable[Combined]]

# Translated note.
MAPPING = {
    "SPEED_CONST__PATH_LANE_CHANGE_LEFT": 0,  # 11702
    (
        "SPEED_STOP__PATH_STRAIGHT",
        "SPEED_DECEL_HARD__PATH_STRAIGHT",
        "SPEED_STOP__PATH_LANE_CHANGE_RIGHT",
    ): 1,  # 4789 + 8 + 3
    "SPEED_DECEL_MILD__PATH_LANE_CHANGE_LEFT": 2,  # 4022
    "SPEED_ACCEL_MILD__PATH_LANE_CHANGE_LEFT": 3,  # 3250
    "SPEED_CONST__PATH_TURN_RIGHT_SHALLOW": 4,  # 556
    "SPEED_CONST__PATH_STRAIGHT": 5,  # 410
    "SPEED_CONST__PATH_TURN_LEFT_SHALLOW": 6,  # 369
    (
        "SPEED_ACCEL_MILD__PATH_TURN_LEFT_SHALLOW",
        "SPEED_ACCEL_HARD__PATH_TURN_LEFT_SHALLOW",
    ): 7,  # 358 + 1
    "SPEED_ACCEL_MILD__PATH_TURN_RIGHT_SHALLOW": 8,  # 334
    "SPEED_DECEL_HARD__PATH_LANE_CHANGE_LEFT": 9,  # 272
    (
        "SPEED_DECEL_MILD__PATH_STRAIGHT",
        "SPEED_DECEL_HARD__PATH_STRAIGHT",
    ): 10,  # 178 + 8
    "SPEED_ACCEL_MILD__PATH_STRAIGHT": 11,  # 144
    (
        "SPEED_ACCEL_HARD__PATH_LANE_CHANGE_LEFT",
        "SPEED_ACCEL_HARD__PATH_STRAIGHT",
    ): 12,  # 139 + 12
    (
        "SPEED_DECEL_MILD__PATH_TURN_LEFT_SHALLOW",
        "SPEED_DECEL_HARD__PATH_TURN_LEFT_SHALLOW",
    ): 13,  # 136 + 3
    "SPEED_CONST__PATH_TURN_RIGHT_UTURN": 14,  # 123
    "SPEED_ACCEL_MILD__PATH_TURN_LEFT_UTURN": 15,  # 114
    (
        "SPEED_CONST__PATH_TURN_LEFT_UTURN",
        "SPEED_ACCEL_HARD__PATH_TURN_LEFT_UTURN",
    ): 16,  # 112 + 1
    (
        "SPEED_CONST__PATH_TURN_RIGHT_SHARP",
        "SPEED_DECEL_HARD__PATH_TURN_RIGHT_SHALLOW",
        "SPEED_DECEL_MILD__PATH_TURN_RIGHT_SHARP",
        "SPEED_ACCEL_MILD__PATH_TURN_RIGHT_SHARP",
    ): 17,  # 73 + 1 + 3
    (
        "SPEED_DECEL_MILD__PATH_TURN_RIGHT_SHALLOW",
        "SPEED_ACCEL_HARD__PATH_TURN_RIGHT_SHALLOW",
    ): 18,  # 72 + 1
    "SPEED_ACCEL_MILD__PATH_TURN_RIGHT_UTURN": 19,  # 56
    (
        "SPEED_DECEL_MILD__PATH_TURN_RIGHT_UTURN",
        "SPEED_DECEL_HARD__PATH_TURN_RIGHT_UTURN",
    ): 20,  # 39 + 3
    (
        "SPEED_CONST__PATH_TURN_LEFT_SHARP",
        "SPEED_DECEL_MILD__PATH_TURN_LEFT_SHARP",
    ): 21,  # 35 + 3
    (
        "SPEED_ACCEL_MILD__PATH_TURN_LEFT_SHARP",
        "SPEED_ACCEL_HARD__PATH_TURN_LEFT_SHARP",
    ): 22,  # 32 + 1
    (
        "SPEED_DECEL_MILD__PATH_TURN_LEFT_UTURN",
        "SPEED_DECEL_HARD__PATH_TURN_LEFT_UTURN",
    ): 23,  # 29 + 3
}

# ===== v3 /load =====
# classify_traj_v3_canbus.py global_label_order.json
# __init__ load label2id default MAPPING_V3
MAPPING_V3 = {
    "SPEED_STOP__PATH_STRAIGHT": 0,  # 5875
    "SPEED_CONST__PATH_STRAIGHT": 1,  # 8505
    "SPEED_ACCEL__PATH_STRAIGHT": 2,  # 3474
    "SPEED_DECEL__PATH_STRAIGHT": 3,  # 4459
    "SPEED_ANY__PATH_TURN_LEFT_BIG": 4,  # 41
    "SPEED_ANY__PATH_TURN_RIGHT_BIG": 5,  # 55
    "SPEED_ANY__PATH_TURN_LEFT_SMALL": 6,  # 2647
    "SPEED_ANY__PATH_TURN_RIGHT_SMALL": 7,  # 3074
}


def _canon_upper(s: str) -> str:
    return " ".join(str(s).strip().split()).upper()


def _is_v3_item(results: dict) -> bool:
    """Check whether the sample belongs to the v3 scheme."""
    return any(
        k in results
        for k in (
            "combined_id_v3",
            "combined_label_v3",
            "speed_label_v3",
            "path_label_v3",
        )
    )


def _combined_v3_from_results(results: dict):
    """Build 'SPEED_*__PATH_*' from v3 fields, or use combined_label_v3 directly."""
    if results.get("combined_label_v3"):
        return _canon_upper(results["combined_label_v3"])
    s, p = results.get("speed_label_v3"), results.get("path_label_v3")
    if s and p:
        return f"{_canon_upper(s)}__{_canon_upper(p)}"
    return None


def _v3_id_from_results(results: dict, v3_label2id=None, default: int = -1) -> int:
    """
    v3 id
    1) combined_id_v3 -> return
    2) combined_label_v3 / (speed_v3, path_v3) v3_label2id MAPPING_V3
    3) return default
    """
    cid = results.get("combined_id_v3", None)
    if isinstance(cid, (int, np.integer)) and int(cid) >= 0:
        return int(cid)

    comb = _combined_v3_from_results(results)
    if comb:
        if v3_label2id is not None:
            idx = v3_label2id.get(_canon_upper(comb), None)
            if idx is not None:
                return int(idx)
        # fallback to local MAPPING_V3
        idx = MAPPING_V3.get(_canon_upper(comb), None)
        if idx is not None:
            return int(idx)

    return int(default)


def _canon(s: str) -> str:
    """Normalize by trimming, collapsing spaces, and uppercasing."""
    return " ".join(str(s).strip().split()).upper()


def build_flat_mapping(table: Dict[TableKey, int]) -> Dict[Combined, int]:
    """
    table key str str id
    return {: id}
    """
    flat = {}
    for k, idx in table.items():
        if isinstance(k, str):
            flat[_canon(k)] = int(idx)
        else:
            for kk in k:
                flat[_canon(kk)] = int(idx)
    return flat


def combined_id_from_results(
    results: Dict[str, Any],
    mapping_table: Dict[TableKey, int],
    default: int = -1,
    device=None,
) -> torch.Tensor:
    """
    results -> mapping_table return id torch.tensor
    - results['ego_labels']['combined_label']
    - results.get('combined_label')
    - return default
    """
    # Translated note.
    comb = None
    if isinstance(results.get("ego_labels"), dict):
        comb = results["ego_labels"].get("combined_label")
    elif comb is None:
        comb = results.get("combined_label")
    else:
        raise ValueError("No combined label found")

    # Translated note.
    flat = build_flat_mapping(mapping_table)
    idx = flat.get(_canon(comb) if comb is not None else "", default)
    return torch.tensor(idx, device=device, dtype=torch.long)


def post_process_coords(corner_coords, imsize=(1600, 900)):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = canvas_box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)

        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords]
            )

            # min_x, min_y, max_x, max_y
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
    else:
        return None


def analyze_position(x, y, angle_deg):
    direction = ""
    if x > 0:
        direction += "front"
    elif x < 0:
        direction += "back"

    if y > 2.5:
        direction += " left"
    elif y < -2.5:
        direction += " right"

    if abs(angle_deg) < 45:
        direction += ", same direction as you, "
    elif abs(abs(angle_deg) - 180) < 45:
        direction += ", opposite direction from you, "
    elif abs(angle_deg - 90) < 45:
        direction += ", heading from right to left, "
    elif abs(angle_deg + 90) < 45:
        direction += ", heading from left to right, "

    return direction.strip()


@PIPELINES.register_module()
class ResizeMultiview3D:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        override=False,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        new_gt_bboxes = []
        new_centers2d = []
        for i in range(len(results["img"])):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results["img"][i],
                    results["scale"],
                    return_scale=True,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results["img"][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results["img"][i],
                    results["scale"],
                    return_scale=True,
                    backend=self.backend,
                )
            results["img"][i] = img
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            # rescale the camera intrinsic
            results["intrinsics"][i][0, 0] *= w_scale
            results["intrinsics"][i][0, 2] *= w_scale
            results["intrinsics"][i][1, 1] *= h_scale
            results["intrinsics"][i][1, 2] *= h_scale

            if "gt_bboxes" in results.keys() and len(results["gt_bboxes"]) > 0:
                gt_bboxes = results["gt_bboxes"][i]
                if len(gt_bboxes) > 0:
                    gt_bboxes[:, 0] *= w_scale
                    gt_bboxes[:, 1] *= h_scale
                    gt_bboxes[:, 2] *= w_scale
                    gt_bboxes[:, 3] *= h_scale
                new_gt_bboxes.append(gt_bboxes)

            if "centers2d" in results.keys() and len(results["centers2d"]) > 0:
                centers2d = results["centers2d"][i]
                if len(gt_bboxes) > 0:
                    centers2d[:, 0] *= w_scale
                    centers2d[:, 1] *= h_scale
                new_centers2d.append(centers2d)

        results["gt_bboxes"] = new_gt_bboxes
        results["centers2d"] = new_centers2d
        results["img_shape"] = img_shapes
        results["pad_shape"] = pad_shapes
        results["scale_factor"] = scale_factors
        results["keep_ratio"] = keep_ratios

        results["lidar2img"] = [
            results["intrinsics"][i] @ results["extrinsics"][i]
            for i in range(len(results["extrinsics"]))
        ]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert "scale_factor" not in results, (
                    "scale and scale_factor cannot be both set."
                )
            else:
                results.pop("scale")
                if "scale_factor" in results:
                    results.pop("scale_factor")
                self._random_scale(results)

        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        return repr_str


@PIPELINES.register_module()
class PadMultiViewImage:
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fix_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


@PIPELINES.register_module()
class LoadAnnoatationVQA:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        lane_objs_info=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def online_vqa(self, results):
        sources = []

        gt_bboxes_2d = []
        gt_bboxes_3d = copy.deepcopy(results["gt_bboxes_3d"])
        gt_bboxes_3d_points = gt_bboxes_3d.corners
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate(
            (gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]),
            axis=1,
        )
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results["cam_infos"].items()):
                gt_bboxes_points_cam = np.matmul(
                    gt_bboxes_points, results["extrinsics"][i].T
                )
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                # img = results['img'][i]

                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]

                    corner_coords = (
                        view_points(corners_3d[:3, :], results["intrinsics"][i], True)
                        .T[:, :2]
                        .tolist()
                    )
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results["pad_shape"][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        gt_bboxes_2d.append(
                            [
                                round(min_x / width, 3),
                                round(min_y / height, 3),
                                round(max_x / width, 3),
                                round(max_y / height, 3),
                                j,
                                cam_type,
                            ]
                        )
                # cv2.imwrite(f"img_{cam_type}.jpg", img)

            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(
                    gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d))
                )
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The object is a {answer}",
                            },
                        ]
                    )

        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(
                -50, 50
            )
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (
                torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1
            )
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[: self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(
                            self.format_det_answer(i, gt_bboxes_3d, results)
                        )
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += "\n".join(objs_near)
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?",
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}",
                        },
                    ]
                )

        lane_objs = self.lane_objs_info[results["sample_idx"]]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs["all_lane_pts"]))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs["lane_objects"].keys():
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"There are no objects on this lane.",
                                },
                            ]
                        )
                    else:
                        objs = []
                        for obj in lane_objs["lane_objects"][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = "\n".join(objs)
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The objects on this lane include:\n{answer}",
                                },
                            ]
                        )

        return sources

    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(
            f"({format_number(point[0])}, {format_number(point[1])})"
            for point in bezier_lane[0]
        )
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy = vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def __call__(self, results):
        traj = None
        if "gt_planning" in results.keys():
            planning_traj = results["gt_planning"][0, :, :2]
            mask = results["gt_planning_mask"][0].any(axis=1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in planning_traj
                )
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."

        sources = self.preprocess_vqa(results, traj)
        prompt = f"You are driving in {results['location']}. "

        online_sources = self.online_vqa(results)
        sources += online_sources

        random.shuffle(sources)
        if "gt_planning" in results.keys() and len(planning_traj) == 6:
            sources = [
                [
                    {
                        "from": "human",
                        "value": "Please provide the planning trajectory for the ego car without reasons.",
                    },
                    {"from": "gpt", "value": traj},
                ]
            ] + sources

        vqa_anno = [item for pair in sources for item in pair]
        vqa_anno[0]["value"] = (
            DEFAULT_IMAGE_TOKEN + "\n" + prompt + vqa_anno[0]["value"]
        )
        vqa_converted = preprocess([vqa_anno], self.tokenizer, True)
        input_ids = vqa_converted["input_ids"][0]
        vlm_labels = vqa_converted["labels"][0]

        results["input_ids"] = input_ids
        results["vlm_labels"] = vlm_labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadAnnoatationVQATraj:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        use_classv3=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        self.use_classv3 = use_classv3
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def online_vqa(self, results):
        sources = []

        gt_bboxes_2d = []
        gt_bboxes_3d = copy.deepcopy(results["gt_bboxes_3d"])
        gt_bboxes_3d_points = gt_bboxes_3d.corners
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate(
            (gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]),
            axis=1,
        )
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results["cam_infos"].items()):
                gt_bboxes_points_cam = np.matmul(
                    gt_bboxes_points, results["extrinsics"][i].T
                )
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                # img = results['img'][i]

                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]

                    corner_coords = (
                        view_points(corners_3d[:3, :], results["intrinsics"][i], True)
                        .T[:, :2]
                        .tolist()
                    )
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results["pad_shape"][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        gt_bboxes_2d.append(
                            [
                                round(min_x / width, 3),
                                round(min_y / height, 3),
                                round(max_x / width, 3),
                                round(max_y / height, 3),
                                j,
                                cam_type,
                            ]
                        )
                # cv2.imwrite(f"img_{cam_type}.jpg", img)

            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(
                    gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d))
                )
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The object is a {answer}",
                            },
                        ]
                    )

        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(
                -50, 50
            )
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (
                torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1
            )
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[: self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(
                            self.format_det_answer(i, gt_bboxes_3d, results)
                        )
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += "\n".join(objs_near)
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?",
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}",
                        },
                    ]
                )

        lane_objs = self.lane_objs_info[results["sample_idx"]]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs["all_lane_pts"]))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs["lane_objects"].keys():
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"There are no objects on this lane.",
                                },
                            ]
                        )
                    else:
                        objs = []
                        for obj in lane_objs["lane_objects"][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = "\n".join(objs)
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The objects on this lane include:\n{answer}",
                                },
                            ]
                        )

        return sources

    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(
            f"({format_number(point[0])}, {format_number(point[1])})"
            for point in bezier_lane[0]
        )
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy = vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def trans_json_to_traj(self, traj_path):
        with open(traj_path, "r") as f:
            traj = json.load(f)
        traj = traj[-1]["A"][0]
        full_match = re.search(
            r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
            traj,
        )
        if full_match:
            coordinates_matches = re.findall(
                r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
            )
            coordinates = [
                tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                for coord in coordinates_matches
            ]
            coordinates_array = np.array(coordinates)
        return coordinates_array

    def get_meta_actions(self, traj, target_traj):
        """Determine meta actions needed to align trajectory with target"""

        # Calculate velocities and directions
        traj_velo = np.linalg.norm(traj[-1] - traj[0])
        target_velo = np.linalg.norm(target_traj[-1] - target_traj[0])

        # Determine speed meta action
        constant_eps = 0.3
        if abs(traj_velo - target_velo) < constant_eps:
            speed_meta = "maintain x-axis value"
        else:
            if traj_velo > target_velo:
                # if traj_velo > 2 * target_velo:
                #     speed_meta = "quick deceleration"
                # else:
                #     speed_meta = "gradual deceleration"
                speed_meta = "decrease %s to x-axis" % format_number(
                    traj_velo - target_velo
                )
            else:
                # if target_velo > 2 * traj_velo:
                #     speed_meta = "quick acceleration"
                # else:
                #     speed_meta = "gradual acceleration"
                speed_meta = "increase %s to x-axis" % format_number(
                    target_velo - traj_velo
                )
        # Determine steering meta action
        forward_th = 0.3
        final_lat_diff = abs(traj[-1, 1] - target_traj[-1, 1])

        if final_lat_diff < forward_th:
            steer_meta = "maintain y-axis value"
        else:
            if traj[-1, 1] < target_traj[-1, 1]:
                steer_meta = "increase %s to y-axis" % format_number(
                    target_traj[-1, 1] - traj[-1, 1]
                )
            else:
                steer_meta = "decrease %s to y-axis" % format_number(
                    traj[-1, 1] - target_traj[-1, 1]
                )

        return speed_meta, steer_meta

    def extend_traj(self, velocity, accel, dt=0.5, num_points=6):
        # Use velocity and acceleration from can_bus to generate trajectory
        pad_planning_traj = np.zeros((num_points, 2))
        # Update velocity using acceleration for first timestep only
        velocity[0] = velocity[0] + accel[0] * dt
        # Use constant velocity for all points
        for i in range(num_points):
            t = dt * (i + 1)
            pad_planning_traj[i, 0] = velocity[0] * t
            pad_planning_traj[i, 1] = velocity[1] * t

        return pad_planning_traj

    def __call__(
        self, results
    ):  # dict_keys(['can_bus', 'command', 'location', 'sample_idx', 'pts_filename', 'sweeps', 'ego_pose', 'ego_pose_inv', 'prev_idx', 'next_idx', 'scene_token', 'frame_idx', 'timestamp', 'cam_infos', 'img_timestamp', 'img_filename', 'lidar2img', 'intrinsics', 'extrinsics', 'prev_exists', 'gt_planning', 'gt_planning_mask', 'ann_info', 'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d', 'filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d', 'centers2d', 'depths', 'gt_labels_3d', 'scale', 'scale_idx', 'keep_ratio'])
        # import os, ipdb
        # if os.getenv("DEBUG") == "1": ipdb.set_trace()
        traj = None
        results["pred_traj2"] = None
        results["min_index"] = None
        if "gt_planning" in results.keys():
            if self.use_sparsedrive_traj:
                planning_traj = self.sparsedrive_traj[results["sample_idx"]]
            else:
                planning_traj = results["gt_planning"][0, :, :2]
            mask = results["gt_planning_mask"][0].any(axis=1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in planning_traj
                )
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."
            else:
                # Pad trajectory to length 6 using last point
                last_point = planning_traj[-1:]
                padding_length = 6 - len(planning_traj)
                padded_traj = np.concatenate(
                    [planning_traj, np.tile(last_point, (padding_length, 1))]
                )
                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in padded_traj
                )
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."

        sources = self.preprocess_vqa(results, traj)
        prompt = f"You are driving in {results['location']}. "

        online_sources = self.online_vqa(results)
        sources += online_sources
        random.shuffle(sources)
        command = self.command_str[results["command"]]

        if "gt_planning" in results.keys() and len(planning_traj) == 6:  # True
            if self.use_cot_v1:
                traj_index = self.closest_index[results["sample_idx"]] % 36
                gt_traj = f"the planning trajectory is [PT, {formatted_points}]."
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here are predefined planning trajectories %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Firstly, please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. "
                            + "\n"
                            + "Secondly, with the command %s, please provide the planning trajectory for the ego car using the selected predefined trajectory as a reference."
                            % command,
                        },
                        {
                            "from": "gpt",
                            "value": "Step 1: The classification result is %s, and it's trajectory is [PT, %s]. "
                            % (traj_index, formatted_traj)
                            + "\n"
                            + "Step 2: With the selected trajectory as a reference, %s"
                            % gt_traj,
                        },
                    ]
                ]

                sources = prompt_traj + sources

            elif self.use_gt_traj:  # False
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here is a predefined planning trajectory %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference.",
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj

            elif self.use_pred_traj_seq:  # True
                # traj_index = self.closest_index[results['sample_idx']] % 36
                # cls_traj = f"The result is <G{traj_index}>."
                gt_traj = f"The result is [PT, {formatted_points}]."

                if self.choose_from_pred:  # False
                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(2)]
                    ).strip()
                    traj_path1 = os.path.join(self.baseline_path, results["sample_idx"])
                    traj1 = self.trans_json_to_traj(traj_path1)
                    traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                    traj2 = self.trans_json_to_traj(traj_path2)
                    # Calculate distances between trajectories
                    # Stack trajectories first
                    trajs = np.stack([traj1, traj2])
                    results["pred_traj2"] = trajs

                    # Calculate distances and get index of minimum
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < planning_traj[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < planning_traj[..., -1, 1]
                        else "decrease"
                    )
                    results["min_index"] = traj_index
                elif self.use_rag:  # True
                    # import pdb; pdb.set_trace()
                    if self.use_ego_mlp:  # True
                        traj1_lidar = self.ego_mlp[results["sample_idx"]][
                            "final_planning"
                        ].numpy()
                        traj1 = copy.deepcopy(traj1_lidar)
                        traj1[:, 1] = -traj1_lidar[:, 0]
                        traj1[:, 0] = traj1_lidar[:, 1]
                    else:
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)

                    if results["sample_idx"] in self.rag_infos:
                        topk_indices = self.rag_infos[results["sample_idx"]][
                            : self.rag_topk
                        ]
                        topk_trajs = self.plan_anchor.reshape(-1, 6, 2)[topk_indices]
                    else:
                        # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                        # dists = np.sqrt(np.sum((extend_traj[None] - self.plan_anchor[results['command']])**2, axis=-1)).sum(-1)
                        dists = np.sqrt(
                            np.sum(
                                (
                                    planning_traj[None]
                                    - self.plan_anchor[results["command"]]
                                )
                                ** 2,
                                axis=-1,
                            )
                        ).sum(-1)
                        topk_indices = np.argsort(dists)[: self.rag_topk]
                        topk_trajs = self.plan_anchor[results["command"]][topk_indices]
                    if self.cat_pred_traj:
                        trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                    else:
                        trajs = topk_trajs
                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(len(trajs))]
                    ).strip()
                    results["pred_traj2"] = trajs
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    results["min_index"] = traj_index
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < planning_traj[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < planning_traj[..., -1, 1]
                        else "decrease"
                    )

                elif self.use_kmeans_traj:
                    trajs = self.plan_anchor[results["command"]]
                    if self.kmeans_pad_traj:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                            # traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                            results["pred_traj2"] = trajs
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                            results["pred_traj2"] = trajs
                    else:
                        results["pred_traj2"] = trajs

                    traj_tokens = " ".join(
                        [
                            f"<G{i}> {DEFAULT_POINT_TOKEN} "
                            for i in range(trajs.shape[0])
                        ]
                    ).strip()
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    results["min_index"] = traj_index
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < traj1[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < traj1[..., -1, 1]
                        else "decrease"
                    )

                speed_action, steer_action = self.get_meta_actions(
                    trajs[traj_index], planning_traj
                )

                if self.use_text_point:
                    endpoints = trajs[:, -1, :]
                    text_points = ", ".join(
                        [
                            f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                            for i, pt in enumerate(endpoints)
                        ]
                    )
                    # import pdb; pdb.set_trace()
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                elif self.add_vel:
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                elif self.add_ego:
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                else:
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                prompt_refine = [
                    [
                        {
                            "from": "human",
                            "value": "How to optimize this selected trajectory?",
                        },
                        {
                            "from": "gpt",
                            "value": "According to the current scene: "
                            + "\n"
                            + "- Velocity suggestions: %s" % speed_action
                            + "\n"
                            + "- Steering suggestions: %s" % steer_action,
                        },
                    ]
                ]

                if self.cot_with_speed:
                    # import pdb; pdb.set_trace()
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj2 = [
                        [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + f"please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": gt_traj},
                        ]
                    ]  # kmeansclusterget
                else:  # Translated note.
                    prompt_traj2 = [
                        [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + "please provide the planning trajectory for the ego car.",
                            },
                            {"from": "gpt", "value": gt_traj},
                        ]
                    ]

                if self.use_other_qa:
                    if self.use_refine_step:
                        sources = prompt_traj1 + prompt_refine + prompt_traj2 + sources
                    else:
                        sources = prompt_traj1 + prompt_traj2 + sources
                else:
                    if self.use_refine_step:
                        sources = prompt_traj1 + prompt_refine + prompt_traj2
                    else:
                        sources = prompt_traj1 + prompt_traj2

            else:
                traj_index = self.closest_index[results["sample_idx"]] % 36
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here are predefined planning trajectories %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. ",
                        },
                        {
                            "from": "gpt",
                            "value": "The classification result is %s. " % traj_index,
                        },
                        {
                            "from": "human",
                            "value": "With the command %s, please provide the planning trajectory for the ego car using the predefined trajectory [PT, %s] as a reference."
                            % (command, formatted_traj),
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj + sources

        else:
            results["loss_mask"] = False
            if self.use_gt_traj:
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here is a predefined planning trajectory %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference.",
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj

            else:
                # # Calculate L2 distance between gt_traj and plan_anchor
                # import pdb; pdb.set_trace()
                gt_traj = f"the planning trajectory is [PT, {formatted_points}]."
                # Calculate velocity and direction from existing points
                # print('traj shape', planning_traj.shape)
                existing_points = planning_traj  # [1, N, 2]
                if planning_traj.shape[0] == 0:
                    pad_planning_traj = self.extend_traj(
                        results["can_bus"][10:12], results["can_bus"][4:6]
                    )
                elif planning_traj.shape[0] == 1:
                    velocities = existing_points[-1]  # [1, N-1, 2]
                    last_velocity = velocities[-1]  # Use last velocity

                    # Extrapolate remaining points using velocity
                    num_missing = 6 - planning_traj.shape[0]
                    extrapolated_points = []
                    last_point = existing_points[-1]

                    for i in range(num_missing):
                        next_point = last_point + last_velocity
                        extrapolated_points.append(next_point)
                        last_point = next_point
                    if num_missing > 0:
                        extrapolated_points = np.stack(extrapolated_points, axis=0)
                        pad_planning_traj = np.concatenate(
                            [planning_traj, extrapolated_points], axis=0
                        )
                elif planning_traj.shape[0] > 1 and planning_traj.shape[0] < 6:
                    velocities = (
                        existing_points[-2] - existing_points[-1]
                    )  # [1, N-1, 2]
                    last_velocity = velocities[-1]  # Use last velocity

                    # Extrapolate remaining points using velocity
                    num_missing = 6 - planning_traj.shape[0]
                    extrapolated_points = []
                    last_point = existing_points[-1]

                    for i in range(num_missing):
                        next_point = last_point + last_velocity
                        extrapolated_points.append(next_point)
                        last_point = next_point
                    if num_missing > 0:
                        extrapolated_points = np.stack(extrapolated_points, axis=0)
                        pad_planning_traj = np.concatenate(
                            [planning_traj, extrapolated_points], axis=0
                        )
                anchor_trajs = self.plan_anchor[
                    results["command"]
                ]  # Get anchors for current command
                # import pdb; pdb.set_trace()
                # Calculate distances between gt_traj and each anchor trajectory
                distances = np.sqrt(
                    (
                        (
                            anchor_trajs.reshape(-1, 12)
                            - pad_planning_traj.reshape(1, 12)
                        )
                        ** 2
                    ).sum(-1)
                )

                # Get index of closest anchor trajectory
                traj_index = np.argmin(distances)
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )

                if self.use_pred_traj_seq:  # True
                    if self.choose_from_pred:  # False
                        traj_tokens = " ".join(
                            [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(2)]
                        ).strip()
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)
                        traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                        traj2 = self.trans_json_to_traj(traj_path2)
                        trajs = np.stack([traj1, traj2])
                        results["pred_traj2"] = trajs

                        # Calculate distances and get index of minimum
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                    elif self.use_rag:  # True
                        if self.use_ego_mlp:  # True
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)

                        if results["sample_idx"] in self.rag_infos:
                            topk_indices = self.rag_infos[results["sample_idx"]][
                                : self.rag_topk
                            ]
                            topk_trajs = self.plan_anchor.reshape(-1, 6, 2)[
                                topk_indices
                            ]
                        else:
                            # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                            # dists = np.sqrt(np.sum((extend_traj[None] - self.plan_anchor[results['command']])**2, axis=-1)).sum(-1)
                            dists = np.sqrt(
                                np.sum(
                                    (
                                        pad_planning_traj[None]
                                        - self.plan_anchor[results["command"]]
                                    )
                                    ** 2,
                                    axis=-1,
                                )
                            ).sum(-1)
                            topk_indices = np.argsort(dists)[: self.rag_topk]
                            topk_trajs = self.plan_anchor[results["command"]][
                                topk_indices
                            ]
                        if self.cat_pred_traj:
                            trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                        else:
                            trajs = topk_trajs
                        traj_tokens = " ".join(
                            [
                                f"<G{i}> {DEFAULT_POINT_TOKEN} "
                                for i in range(trajs.shape[0])
                            ]
                        ).strip()
                        results["pred_traj2"] = trajs
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                        # vel_adjust = 'increase' if trajs[traj_index][...,-1, 0] < planning_traj[..., -1, 0] else 'decrease'
                        # steer_adjust = 'increase' if trajs[traj_index][...,-1, 1] < planning_traj[..., -1, 1] else 'decrease'
                    elif self.use_kmeans_traj:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                            # traj1 = self.trans_json_to_traj(traj_path1)
                            # trajs = np.concatenate([traj1[None], trajs], axis=0)
                            # results['pred_traj2'] = trajs
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)

                        trajs = self.plan_anchor[results["command"]]
                        if self.kmeans_pad_traj:
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                        results["pred_traj2"] = trajs
                        traj_tokens = " ".join(
                            [
                                f"<G{i}> {DEFAULT_POINT_TOKEN} "
                                for i in range(trajs.shape[0])
                            ]
                        ).strip()
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                        # vel_adjust = 'increase' if trajs[traj_index][...,-1, 0] < planning_traj[..., -1, 0] else 'decrease'
                        # steer_adjust = 'increase' if trajs[traj_index][...,-1, 1] < planning_traj[..., -1, 1] else 'decrease'
                        # traj_tokens = ' '.join([f'<G{i}> {DEFAULT_POINT_TOKEN} ' for i in range(36)]).strip()
                    # vel = 'With the velocity of (%s, %s)' % (format_number(np.clip(results['can_bus'][4],0,1000), 2), format_number(results['can_bus'][5], 2))
                    if self.use_text_point:
                        endpoints = trajs[:, -1, :]
                        text_points = ", ".join(
                            [
                                f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                                for i, pt in enumerate(endpoints)
                            ]
                        )

                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    elif self.add_vel:
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    elif self.add_ego:
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    else:
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + "Please select the best trajectory in the current scenario.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    # if self.use_two_image:
                    #     prompt_traj = prompt_traj + prompt_traj
                    #     [{"from": 'human',
                    #     "value": "Here is a predefined planning trajectory %s. " % DEFAULT_TRAJ_TOKEN + '\n' +
                    #             "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference."},
                    #     {"from": 'gpt',
                    #     "value": gt_traj}]]

                    # else:
                    #     prompt_traj = [
                    #         [{"from": 'human',
                    #         "value": "Here are predefined planning trajectories %s. " % DEFAULT_TRAJ_TOKEN + '\n' +
                    #                 "Please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. "},
                    #         {"from": 'gpt',
                    #         "value": "The classification result is %s. " % traj_index}]]

                if self.use_other_qa:
                    sources = prompt_traj + sources
                else:
                    sources = prompt_traj
        # import pdb; pdb.set_trace()
        vqa_anno = [item for pair in sources for item in pair]
        # if self.use_pred_traj_seq:

        #     vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
        #     if not self.only_cls and len(planning_traj) == 6 and self.use_two_image:
        #         vqa_anno[2]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[2]['value']
        #         if self.use_refine_step:
        #             vqa_anno[4]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[4]['value']
        # else:
        vqa_anno[0]["value"] = (
            DEFAULT_IMAGE_TOKEN + "\n" + prompt + vqa_anno[0]["value"]
        )  # image token originalprompt
        # import pdb; pdb.set_trace()
        vqa_converted = preprocess_traj(
            [vqa_anno], self.tokenizer, True, has_traj=True
        )  # warning
        input_ids = vqa_converted["input_ids"][0]
        vlm_labels = vqa_converted["labels"][0]

        results["input_ids"] = input_ids  # torch.Size([266])
        results["vlm_labels"] = vlm_labels
        # import os, ipdb
        # if os.getenv("NEW_DEBUG") == "1": ipdb.set_trace()
        # if "combined_label_v3" in results and results["combined_label_v3"] is not None:
        #     idx = _v3_id_from_results(results, v3_label2id=getattr(self, "v3_label2id", None), default=-1)
        #     results['classv3_index'] = torch.tensor(idx, device=input_ids.device, dtype=torch.long)

        if self.use_classv3:
            # v3 v3 fields fields
            idx = _v3_id_from_results(
                results, v3_label2id=getattr(self, "v3_label2id", None), default=-1
            )
            results["meta_index"] = torch.tensor(
                idx, device=input_ids.device, dtype=torch.long
            )
        else:
            # + MAPPING
            # combined_id_from_results v3
            # fields MAPPING
            results["meta_index"] = combined_id_from_results(
                results,
                mapping_table=MAPPING,  # Translated note.
                default=-1,
                device=input_ids.device,
            )

        return results


@PIPELINES.register_module()
class LoadAnnoatationPUREQA:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        clustered_traj_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/rep_combined_ids_k6.npy",
        clustered_traj_v3_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/repres_c8_k6_classv3.npy",
        use_classv3=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

        self.use_classv3 = use_classv3
        if self.use_classv3:
            self.cluster_traj = np.load(clustered_traj_v3_path)  # (8, 6, 6, 2)
        else:
            self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

        self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

    def create_new_vlm_prompt(self, results, trajs=None):
        """
        Create a new VLM prompt based on the specified format.

        Args:
            results: The data pipeline results containing location, can_bus, etc.
            trajs: Optional predefined trajectories array
            coarse_traj: Optional coarse trajectory predicted by another model

        Returns:
            str: The formatted VLM prompt
        """
        # Extract location
        location = "singapore or boston"

        # Extract speed and acceleration from can_bus
        current_speed = results["can_bus"][-3:-1]  # Last 2 elements for velocity
        current_accel = results["can_bus"][4:6]  # Elements 4-5 for acceleration
        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

        # Create the new VLM prompt
        prompt = (
            f"You are a vehicle trajectory prediction model for autonomous driving. "
            f"Your task is to predict the ego vehicle's 3-second trajectory based on the following inputs: "
            f"multi-view images from 6 cameras, ego vehicle states (position, velocity and acceleration), "
            f"and discrete navigation commands. Now you are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
            f"Please predict the meta-action for the current scene "
            f"and provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s "
            f"and an acceleration of {accel_str} m/s^2."
        )

        vqa = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

        return vqa

    def __call__(self, results):
        import os, ipdb

        if os.getenv("DATA_DEBUG") == "1":
            ipdb.set_trace()

        vqa = self.create_new_vlm_prompt(results, trajs=None)
        prompt = vqa[0]["value"]
        input_ids = tokenizer_image_traj_token(
            prompt, self.tokenizer, return_tensors="pt"
        )

        results["input_ids"] = input_ids  # torch.Size([184])
        results["vlm_labels"] = torch.zeros_like(
            input_ids
        )  # not for use, just for compatibility

        if self.use_classv3:
            # v3 v3 fields fields
            idx = _v3_id_from_results(
                results, v3_label2id=getattr(self, "v3_label2id", None), default=-1
            )
            results["min_index"] = torch.tensor(
                idx, device=input_ids.device, dtype=torch.long
            )
        else:
            # + MAPPING
            # combined_id_from_results v3
            # fields MAPPING
            results["min_index"] = combined_id_from_results(
                results,
                mapping_table=MAPPING,  # Translated note.
                default=-1,
                device=input_ids.device,
            )

        return results


@PIPELINES.register_module()
class LoadAnnoatationVQATrajCF:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        clustered_traj_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/rep_combined_ids_k6.npy",
        clustered_traj_v3_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/repres_c8_k6_classv3.npy",
        use_classv3=False,
        closed_loop=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

        self.use_classv3 = use_classv3
        if self.use_classv3:
            self.cluster_traj = np.load(clustered_traj_v3_path)  # (8, 6, 6, 2)
        else:
            self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

        self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)
        self.closed_loop = closed_loop

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def create_new_vlm_prompt(self, results, trajs=None):
        """
        Create a new VLM prompt based on the specified format.

        Args:
            results: The data pipeline results containing location, can_bus, etc.
            trajs: Optional predefined trajectories array
            coarse_traj: Optional coarse trajectory predicted by another model

        Returns:
            str: The formatted VLM prompt
        """
        # Extract location
        location = "singapore or boston"

        # Extract speed and acceleration from can_bus
        current_speed = results["can_bus"][-3:-1]  # Last 2 elements for velocity
        current_accel = results["can_bus"][4:6]  # Elements 4-5 for acceleration
        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

        # Generate traj_tokens if trajs are provided
        num_trajs_within_cluster = trajs.shape[1]  # 6
        if trajs is not None:
            traj_tokens = " ".join(
                [
                    f"<G{i}> {DEFAULT_TRAJ_TOKEN} "
                    for i in range(num_trajs_within_cluster)
                ]
            ).strip()
        else:
            assert False, "clustered trajs is not provided"

        # Format coarse trajectory if provided
        # coarse_traj_str = f"And the coarse trajectory predicted by another strong model is {DEFAULT_TRAJ_TOKEN}. "

        # Create the new VLM prompt
        prompt = (
            f"You are a vehicle trajectory prediction model for autonomous driving. "
            f"Your task is to predict the ego vehicle's 3-second trajectory based on the following inputs: "
            f"multi-view images from 6 cameras, ego vehicle states (position, velocity and acceleration), "
            f"and discrete navigation commands. Now you are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
            f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
            f"Please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s "
            f"and an acceleration of {accel_str} m/s^2."
        )

        vqa = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

        return vqa

    def __call__(self, results):
        import os, ipdb

        if os.getenv("DATA_DEBUG") == "1":
            ipdb.set_trace()
        # traj = None
        results["pred_traj2"] = self.cluster_traj  # (24, 6, 6, 2)

        vqa = self.create_new_vlm_prompt(results, trajs=self.cluster_traj)
        prompt = vqa[0]["value"]
        input_ids = tokenizer_image_traj_token(
            prompt, self.tokenizer, return_tensors="pt"
        )

        results["input_ids"] = input_ids  # torch.Size([184])
        results["vlm_labels"] = torch.zeros_like(
            input_ids
        )  # not for use, just for compatibility

        if self.closed_loop:
            results["min_index"] = torch.tensor(
                -1, device=input_ids.device, dtype=torch.long
            )
        else:
            if self.use_classv3:
                # v3 v3 fields fields
                idx = _v3_id_from_results(
                    results, v3_label2id=getattr(self, "v3_label2id", None), default=-1
                )
                results["min_index"] = torch.tensor(
                    idx, device=input_ids.device, dtype=torch.long
                )
            else:
                # + MAPPING
                # combined_id_from_results v3
                # fields MAPPING
                results["min_index"] = combined_id_from_results(
                    results,
                    mapping_table=MAPPING,  # Translated note.
                    default=-1,
                    device=input_ids.device,
                )

        return results


@PIPELINES.register_module()
class LoadAnnoatationVQATrajTextCF:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        clustered_traj_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/rep_combined_ids_k6.npy",
        clustered_traj_v3_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/repres_c8_k6_classv3.npy",
        use_classv3=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

        self.use_classv3 = use_classv3
        if self.use_classv3:
            self.cluster_traj = np.load(clustered_traj_v3_path)  # (8, 6, 6, 2)
        else:
            self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

        self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def create_new_vlm_prompt(self, results, trajs=None):
        """
        Create a new VLM prompt based on the specified format.

        Args:
            results: The data pipeline results containing location, can_bus, etc.
            trajs: Optional predefined trajectories array
            coarse_traj: Optional coarse trajectory predicted by another model

        Returns:
            str: The formatted VLM prompt
        """
        # Extract location
        location = "singapore or boston"

        # Extract speed and acceleration from can_bus
        current_speed = results["can_bus"][-3:-1]  # Last 2 elements for velocity
        current_accel = results["can_bus"][4:6]  # Elements 4-5 for acceleration
        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

        # Generate traj_tokens if trajs are provided
        num_trajs_within_cluster = trajs.shape[1]  # 6
        if trajs is not None:
            traj_tokens = " ".join(
                [
                    f"<G{i}> {DEFAULT_TRAJ_TOKEN} "
                    for i in range(num_trajs_within_cluster)
                ]
            ).strip()
        else:
            assert False, "clustered trajs is not provided"

        # Format coarse trajectory if provided
        # coarse_traj_str = f"And the coarse trajectory predicted by another strong model is {DEFAULT_TRAJ_TOKEN}. "

        # Create the new VLM prompt
        prompt = (
            f"You are a vehicle trajectory prediction model for autonomous driving. "
            f"Your task is to predict the ego vehicle's 3-second trajectory based on the following inputs: "
            f"multi-view images from 6 cameras, ego vehicle states (position, velocity and acceleration), "
            f"and discrete navigation commands. Now you are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
            f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
            f"Please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s "
            f"and an acceleration of {accel_str} m/s^2."
        )

        vqa = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

        return vqa

    def __call__(self, results):
        import os

        if os.getenv("DATA_DEBUG") == "1":
            import ipdb

            ipdb.set_trace()

        # 1) shape (num_meta, K, 6, 2)
        results["pred_traj2"] = self.cluster_traj

        # 2) QA-CoT input_ids & vlm_labels
        input_ids, vlm_labels, meta_id = create_qacot_prompt_and_labels(
            results, self.tokenizer, self.cluster_traj, use_endpoint_text=False
        )

        # 3) results
        results["input_ids"] = input_ids  # torch.Size([L])
        results["vlm_labels"] = vlm_labels  # torch.Size([L])
        # results['min_index']  = torch.tensor(meta_id, device=input_ids.device)
        if self.use_classv3:
            # v3 v3 fields fields
            idx = _v3_id_from_results(
                results, v3_label2id=getattr(self, "v3_label2id", None), default=-1
            )
            results["min_index"] = torch.tensor(
                idx, device=input_ids.device, dtype=torch.long
            )
        else:
            # + MAPPING
            # combined_id_from_results v3
            # fields MAPPING
            results["min_index"] = combined_id_from_results(
                results,
                mapping_table=MAPPING,  # Translated note.
                default=-1,
                device=input_ids.device,
            )

        return results


@PIPELINES.register_module()
class LoadAnnoatationVQATrajTextCFTest:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        clustered_traj_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/rep_combined_ids_k6.npy",
        clustered_traj_v3_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/repres_c8_k6_classv3.npy",
        use_classv3=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

        self.use_classv3 = use_classv3
        if self.use_classv3:
            self.cluster_traj = np.load(clustered_traj_v3_path)  # (8, 6, 6, 2)
        else:
            self.cluster_traj = np.load(clustered_traj_path)  # (24, 6, 6, 2)

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def create_new_vlm_prompt(self, results, trajs=None):
        """
        Create a new VLM prompt based on the specified format.

        Args:
            results: The data pipeline results containing location, can_bus, etc.
            trajs: Optional predefined trajectories array
            coarse_traj: Optional coarse trajectory predicted by another model

        Returns:
            str: The formatted VLM prompt
        """
        # Extract location
        location = "singapore or boston"

        # Extract speed and acceleration from can_bus
        current_speed = results["can_bus"][-3:-1]  # Last 2 elements for velocity
        current_accel = results["can_bus"][4:6]  # Elements 4-5 for acceleration
        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

        # Generate traj_tokens if trajs are provided
        num_trajs_within_cluster = trajs.shape[1]  # 6
        if trajs is not None:
            traj_tokens = " ".join(
                [
                    f"<G{i}> {DEFAULT_TRAJ_TOKEN} "
                    for i in range(num_trajs_within_cluster)
                ]
            ).strip()
        else:
            assert False, "clustered trajs is not provided"

        # Format coarse trajectory if provided
        # coarse_traj_str = f"And the coarse trajectory predicted by another strong model is {DEFAULT_TRAJ_TOKEN}. "

        # Create the new VLM prompt
        prompt = (
            f"You are a vehicle trajectory prediction model for autonomous driving. "
            f"Your task is to predict the ego vehicle's 3-second trajectory based on the following inputs: "
            f"multi-view images from 6 cameras, ego vehicle states (position, velocity and acceleration), "
            f"and discrete navigation commands. Now you are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
            f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
            f"Please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s "
            f"and an acceleration of {accel_str} m/s^2."
        )

        vqa = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

        return vqa

    def __call__(self, results):
        import os

        if os.getenv("DATA_DEBUG") == "1":
            import ipdb

            ipdb.set_trace()

        # 1) shape (num_meta, K, 6, 2)
        results["pred_traj2"] = self.cluster_traj

        # 2) QA-CoT input_ids & vlm_labels
        input_ids, vlm_labels, meta_id = create_qacot_prompt_and_labels(
            results,
            self.tokenizer,
            self.cluster_traj,
            use_endpoint_text=False,
            testing=True,
        )
        # returnprefixinput ids
        vlm_labels = input_ids.clone()

        results["input_ids"] = input_ids  # torch.Size([≤L])
        results["vlm_labels"] = vlm_labels  # torch.Size([≤L])

        if self.use_classv3:
            # v3 v3 fields fields
            idx = _v3_id_from_results(
                results, v3_label2id=getattr(self, "v3_label2id", None), default=-1
            )
            results["min_index"] = torch.tensor(
                idx, device=input_ids.device, dtype=torch.long
            )
        else:
            # + MAPPING
            # combined_id_from_results v3
            # fields MAPPING
            results["min_index"] = combined_id_from_results(
                results,
                mapping_table=MAPPING,  # Translated note.
                default=-1,
                device=input_ids.device,
            )

        return results


def _format_traj_pts(traj_xy, T=6):
    """traj_xy is (t, 2); pad to T with the last point and output 'x,y' text."""
    import numpy as np

    if traj_xy.shape[0] < T:
        last = traj_xy[-1:]
        pad = np.repeat(last, T - traj_xy.shape[0], axis=0)
        traj_xy = np.concatenate([traj_xy, pad], axis=0)
    elif traj_xy.shape[0] > T:
        traj_xy = traj_xy[:T]
    formatted_points = ", ".join(
        f"({format_number(p[0], 2)}, {format_number(p[1], 2)})" for p in traj_xy
    )
    return formatted_points, traj_xy


def _pick_meta_id_gt(results, tokenizer_device):
    return int(
        combined_id_from_results(
            results, MAPPING, default=0, device=tokenizer_device
        ).item()
    )


def _build_qacot_texts(
    results, cluster_traj, meta_id, use_endpoint_text=False, testing=False
):
    """
    return (prefix_text, target_text)
    prefix_text Q1+GT_A1+Q2
    target_text A2
    """
    # Translated note.
    location = results.get("location", "singapore or boston")
    current_speed = results["can_bus"][-3:-1]
    current_accel = results["can_bus"][4:6]
    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

    # meta
    assert cluster_traj is not None, "clustered trajs is not provided"
    num_meta = cluster_traj.shape[0]
    cand_trajs = cluster_traj[meta_id]  # (K, 6, 2)
    K = cand_trajs.shape[0]

    # token
    # token
    traj_tokens = " ".join([f"<G{i}> {DEFAULT_TRAJ_TOKEN} " for i in range(K)]).strip()

    # token default False
    if use_endpoint_text:
        endpoints = cand_trajs[:, -1, :]
        text_points = ", ".join(
            [f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})" for i, pt in enumerate(endpoints)]
        )
        cand_str = f"predefined trajectories with endpoints of future 3 seconds [{text_points}]"
    else:
        cand_str = f"predefined trajectories [{traj_tokens}]"

    # ---- build Q1 / A1(GT) / Q2 ----
    q1 = (
        f"You are a vehicle trajectory prediction model for autonomous driving. "
        f"You are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
        f"The ego car's current velocity is {speed_str} m/s and acceleration is {accel_str} m/s^2. "
        f"There are {num_meta} meta action categories indexed from 0 to {num_meta - 1}. "
        f"Question 1: What is the current meta action category id?"
    )
    a1_gt = f"Answer 1: The meta action category id is {meta_id}.</s>"

    q2 = (
        f"Question 2: Based on meta action {meta_id}, here are {cand_str} for the ego car. "
        f"Please provide the final 3-second planning trajectory in the format [PT, (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]. Answer 2:"
    )

    prefix_text = " ".join([q1, a1_gt, q2])

    if testing:
        return prefix_text, None

    # ---- build A2 GT ----
    # gt_planning 6
    if "gt_planning" in results.keys():
        planning_traj = results["gt_planning"][0, :, :2]  # (T, 2)
        mask = results["gt_planning_mask"][0].any(axis=1)
        planning_traj = planning_traj[mask]
    else:
        raise ValueError("gt_planning not found in results")

    formatted_points, _ = _format_traj_pts(planning_traj, T=6)
    a2 = f"The result is [PT, {formatted_points}].</s>"

    return prefix_text, a2


def create_qacot_prompt_and_labels(
    results, tokenizer, cluster_traj, use_endpoint_text=False, testing=False
):
    """
    ()build QA-CoT
    - input_ids: prefix + target
    - vlm_labels: target part position -100
    """
    # 1) GT meta id
    prefix_device = tokenizer_image_traj_token(
        DEFAULT_IMAGE_TOKEN, tokenizer, return_tensors="pt"
    ).device
    meta_id = _pick_meta_id_gt(results, tokenizer_device=prefix_device)

    # 2) prefix/target
    prefix_text, target_text = _build_qacot_texts(
        results,
        cluster_traj,
        meta_id,
        use_endpoint_text=use_endpoint_text,
        testing=testing,
    )

    # /
    results["pred_traj2"] = cluster_traj  # (num_meta, K, 6, 2)
    results["min_index"] = torch.tensor(meta_id, device=prefix_device)

    # 3) prefix/
    prefix_ids = tokenizer_image_traj_token(
        prefix_text, tokenizer, return_tensors="pt"
    )  # (L1,)

    if testing:
        return prefix_ids, None, meta_id

    full_ids = tokenizer_image_traj_token(
        prefix_text + " " + target_text, tokenizer, return_tensors="pt"
    )  # (L2,)

    assert full_ids.ndim == 1 and prefix_ids.ndim == 1
    L1, L2 = prefix_ids.shape[0], full_ids.shape[0]
    assert L2 >= L1, "full prompt must be longer than prefix"

    # 4) target
    vlm_labels = full_ids.clone()
    vlm_labels[:L1] = -100  # Q1 / A1(GT) / Q2 token
    # note target_text token 'Answer 2: ...' [PT, ...]

    return full_ids, vlm_labels, meta_id


@PIPELINES.register_module()
class LoadAnnoatationVQATrajDualCF:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict.pkl",
        cot_with_speed=False,
        lane_objs_info=None,
        clustered_traj_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/nuscenes/process_utils/repres_c15_k5_ychange.npy",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

        self.cluster_traj = np.load(clustered_traj_path)  # (15, 5, 6, 2)

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def online_vqa(self, results):
        sources = []

        gt_bboxes_2d = []
        gt_bboxes_3d = copy.deepcopy(results["gt_bboxes_3d"])
        gt_bboxes_3d_points = gt_bboxes_3d.corners
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate(
            (gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]),
            axis=1,
        )
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results["cam_infos"].items()):
                gt_bboxes_points_cam = np.matmul(
                    gt_bboxes_points, results["extrinsics"][i].T
                )
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                # img = results['img'][i]

                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]

                    corner_coords = (
                        view_points(corners_3d[:3, :], results["intrinsics"][i], True)
                        .T[:, :2]
                        .tolist()
                    )
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results["pad_shape"][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        gt_bboxes_2d.append(
                            [
                                round(min_x / width, 3),
                                round(min_y / height, 3),
                                round(max_x / width, 3),
                                round(max_y / height, 3),
                                j,
                                cam_type,
                            ]
                        )
                # cv2.imwrite(f"img_{cam_type}.jpg", img)

            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(
                    gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d))
                )
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The object is a {answer}",
                            },
                        ]
                    )

        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(
                -50, 50
            )
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (
                torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1
            )
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[: self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(
                            self.format_det_answer(i, gt_bboxes_3d, results)
                        )
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += "\n".join(objs_near)
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?",
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}",
                        },
                    ]
                )

        lane_objs = self.lane_objs_info[results["sample_idx"]]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs["all_lane_pts"]))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs["lane_objects"].keys():
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"There are no objects on this lane.",
                                },
                            ]
                        )
                    else:
                        objs = []
                        for obj in lane_objs["lane_objects"][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = "\n".join(objs)
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The objects on this lane include:\n{answer}",
                                },
                            ]
                        )

        return sources

    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(
            f"({format_number(point[0])}, {format_number(point[1])})"
            for point in bezier_lane[0]
        )
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy = vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def trans_json_to_traj(self, traj_path):
        with open(traj_path, "r") as f:
            traj = json.load(f)
        traj = traj[-1]["A"][0]
        full_match = re.search(
            r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
            traj,
        )
        if full_match:
            coordinates_matches = re.findall(
                r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
            )
            coordinates = [
                tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                for coord in coordinates_matches
            ]
            coordinates_array = np.array(coordinates)
        return coordinates_array

    def get_meta_actions(self, traj, target_traj):
        """Determine meta actions needed to align trajectory with target"""

        # Calculate velocities and directions
        traj_velo = np.linalg.norm(traj[-1] - traj[0])
        target_velo = np.linalg.norm(target_traj[-1] - target_traj[0])

        # Determine speed meta action
        constant_eps = 0.3
        if abs(traj_velo - target_velo) < constant_eps:
            speed_meta = "maintain x-axis value"
        else:
            if traj_velo > target_velo:
                # if traj_velo > 2 * target_velo:
                #     speed_meta = "quick deceleration"
                # else:
                #     speed_meta = "gradual deceleration"
                speed_meta = "decrease %s to x-axis" % format_number(
                    traj_velo - target_velo
                )
            else:
                # if target_velo > 2 * traj_velo:
                #     speed_meta = "quick acceleration"
                # else:
                #     speed_meta = "gradual acceleration"
                speed_meta = "increase %s to x-axis" % format_number(
                    target_velo - traj_velo
                )
        # Determine steering meta action
        forward_th = 0.3
        final_lat_diff = abs(traj[-1, 1] - target_traj[-1, 1])

        if final_lat_diff < forward_th:
            steer_meta = "maintain y-axis value"
        else:
            if traj[-1, 1] < target_traj[-1, 1]:
                steer_meta = "increase %s to y-axis" % format_number(
                    target_traj[-1, 1] - traj[-1, 1]
                )
            else:
                steer_meta = "decrease %s to y-axis" % format_number(
                    traj[-1, 1] - target_traj[-1, 1]
                )

        return speed_meta, steer_meta

    def extend_traj(self, velocity, accel, dt=0.5, num_points=6):
        # Use velocity and acceleration from can_bus to generate trajectory
        trajectory = []
        current_vel = np.array(velocity)

        for i in range(num_points):
            # Calculate position at time t = i * dt
            t = i * dt
            # s = v0*t + 0.5*a*t^2
            position = current_vel * t + 0.5 * np.array(accel) * t**2
            trajectory.append(position)

        return np.array(trajectory)

    def create_new_vlm_prompt(self, results, trajs=None):
        """
        Create a new VLM prompt based on the specified format.

        Args:
            results: The data pipeline results containing location, can_bus, etc.
            trajs: Optional predefined trajectories array
            coarse_traj: Optional coarse trajectory predicted by another model

        Returns:
            str: The formatted VLM prompt
        """
        # Extract location
        location = results.get("location", "singapore or boston")

        # Extract speed and acceleration from can_bus
        current_speed = results["can_bus"][-3:-1]  # Last 2 elements for velocity
        current_accel = results["can_bus"][4:6]  # Elements 4-5 for acceleration
        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"

        # Generate traj_tokens if trajs are provided
        num_trajs_within_cluster = trajs.shape[1] + 1  # 5 + 1
        if trajs is not None:
            traj_tokens = " ".join(
                [
                    f"<G{i}> {DEFAULT_POINT_TOKEN} "
                    for i in range(num_trajs_within_cluster)
                ]
            ).strip()
        else:
            assert False, "clustered trajs is not provided"

        # Format coarse trajectory if provided
        coarse_traj_str = f"And the coarse trajectory predicted by another strong model is {DEFAULT_TRAJ_TOKEN}. "

        # Create the new VLM prompt
        prompt = (
            f"You are a vehicle trajectory prediction model for autonomous driving. "
            f"Your task is to predict the ego vehicle's 3-second trajectory based on the following inputs: "
            f"multi-view images from 6 cameras, ego vehicle states (position, velocity and acceleration), "
            f"and discrete navigation commands. Now you are driving in {location} {DEFAULT_IMAGE_TOKEN}. "
            f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
            f"{coarse_traj_str}"
            f"Please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s "
            f"and an acceleration of {accel_str} m/s^2."
        )

        vqa = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

        return vqa

    def __call__(
        self, results
    ):  # dict_keys(['can_bus', 'command', 'location', 'sample_idx', 'pts_filename', 'sweeps', 'ego_pose', 'ego_pose_inv', 'prev_idx', 'next_idx', 'scene_token', 'frame_idx', 'timestamp', 'cam_infos', 'img_timestamp', 'img_filename', 'lidar2img', 'intrinsics', 'extrinsics', 'prev_exists', 'gt_planning', 'gt_planning_mask', 'ann_info', 'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d', 'filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d', 'centers2d', 'depths', 'gt_labels_3d', 'scale', 'scale_idx', 'keep_ratio'])
        import os, ipdb

        if os.getenv("DATA_DEBUG") == "1":
            ipdb.set_trace()
        # traj = None
        results["pred_traj2"] = self.cluster_traj  # (15, 5, 6, 2)

        vqa = self.create_new_vlm_prompt(results, trajs=self.cluster_traj)
        prompt = vqa[0]["value"]
        input_ids = tokenizer_image_traj_token(
            prompt, self.tokenizer, return_tensors="pt"
        )

        results["input_ids"] = input_ids
        results["vlm_labels"] = torch.zeros_like(
            input_ids
        )  # not for use, just for compatibility

        min_index = get_category_index(
            results["direction_category"][0], results["distance_category"][0]
        )  # TODO: correspongding to 'direction_category', 'distance_category'
        results["min_index"] = torch.tensor(min_index).to(input_ids.device)

        return results


@PIPELINES.register_module()
class LoadAnnoatationVQATrajFut18:
    def __init__(
        self,
        base_vqa_path,
        base_desc_path,
        base_conv_path,
        base_key_path,
        tokenizer,
        max_length,
        n_gen=2,
        ignore_type=["v1", "v2", "v3"],
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        train_closest_path="/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/closest_train_indices_36.pkl",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline_train",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj_train",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_36_9s_train.pkl",
        use_cot_v1=False,
        use_gt_traj=False,
        use_pred_traj=False,
        use_pred_traj_seq=False,
        use_kmeans_traj=False,
        kmeans_pad_traj=False,
        use_other_qa=True,
        use_xy=False,
        only_cls=False,
        only_refine=False,
        use_two_image=False,
        use_text_traj=False,
        use_concat_point=False,
        choose_from_pred=False,
        use_refine_step=False,
        cat_pred_traj=False,
        use_rag=False,
        rag_topk=5,
        use_ego_mlp=False,
        use_sparsedrive_traj=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_train_dict_9s.pkl",
        cot_with_speed=False,
        pred_traj_60=False,
        lane_objs_info=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.use_cot_v1 = use_cot_v1
        self.use_gt_traj = use_gt_traj
        self.use_pred_traj = use_pred_traj
        self.use_other_qa = use_other_qa
        self.use_sparsedrive_traj = use_sparsedrive_traj
        self.add_vel = add_vel
        if self.use_sparsedrive_traj:
            sparsedrive_infos = pickle.load(
                open(
                    "/nfs/dataset-ofs-voyager-research/xschen/repos/SparseDrive/data/infos/nuscenes_infos_train.pkl",
                    "rb",
                )
            )
            sparsedrive_traj = {}
            for info in sparsedrive_infos["infos"]:
                gt_planning = copy.deepcopy(info["gt_ego_fut_trajs"])[:, :2]
                gt_planning[:, 1] = -info["gt_ego_fut_trajs"][:, 0]
                gt_planning[:, 0] = info["gt_ego_fut_trajs"][:, 1]
                sparsedrive_traj[info["token"]] = gt_planning.cumsum(axis=0)
            self.sparsedrive_traj = sparsedrive_traj
        self.use_xy = use_xy
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_two_image = use_two_image
        self.use_text_traj = use_text_traj
        self.use_concat_point = use_concat_point
        self.choose_from_pred = choose_from_pred
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.only_refine = only_refine
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_ego_mlp = use_ego_mlp
        self.use_text_point = use_text_point
        self.add_ego = add_ego
        self.cot_with_speed = cot_with_speed
        self.pred_traj_60 = pred_traj_60
        if self.use_ego_mlp:
            self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        plan_anchor_lidar = np.load(kmeans_path)
        if "9s" in kmeans_path:
            # plan_anchor_lidar_9s = np.load('/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/data/kmeans/kmeans_plan_9s_36.npy')
            self.plan_anchor = plan_anchor_lidar.copy()
        else:
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
        # self.plan_anchor[2,0] = np.zeros_like(self.plan_anchor[2,0])
        self.closest_index = pickle.load(open(train_closest_path, "rb"))
        self.lane_objs_info = pickle.load(open(lane_objs_info, "rb"))
        CLASSES = (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path + results["sample_idx"] + ".json"):
            with open(self.base_key_path + results["sample_idx"] + ".json", "r") as f:
                action = json.load(f)

            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": action},
                ]
            )
        if os.path.exists(self.base_desc_path + results["sample_idx"] + ".json"):
            with open(self.base_desc_path + results["sample_idx"] + ".json", "r") as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": desc["description"]},
                ]
            )
        if os.path.exists(self.base_vqa_path + results["sample_idx"] + ".json"):
            with open(self.base_vqa_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa):
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )

        if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
            with open(self.base_conv_path + results["sample_idx"] + ".json", "r") as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": "human", "value": pair["question"]},
                        {"from": "gpt", "value": pair["answer"]},
                    ]
                )
        return sources

    def online_vqa(self, results):
        sources = []

        gt_bboxes_2d = []
        gt_bboxes_3d = copy.deepcopy(results["gt_bboxes_3d"])
        gt_bboxes_3d_points = gt_bboxes_3d.corners
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate(
            (gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]),
            axis=1,
        )
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results["cam_infos"].items()):
                gt_bboxes_points_cam = np.matmul(
                    gt_bboxes_points, results["extrinsics"][i].T
                )
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                # img = results['img'][i]

                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]

                    corner_coords = (
                        view_points(corners_3d[:3, :], results["intrinsics"][i], True)
                        .T[:, :2]
                        .tolist()
                    )
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results["pad_shape"][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        gt_bboxes_2d.append(
                            [
                                round(min_x / width, 3),
                                round(min_y / height, 3),
                                round(max_x / width, 3),
                                round(max_y / height, 3),
                                j,
                                cam_type,
                            ]
                        )
                # cv2.imwrite(f"img_{cam_type}.jpg", img)

            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(
                    gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d))
                )
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The object is a {answer}",
                            },
                        ]
                    )

        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(
                -50, 50
            )
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (
                torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1
            )
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[: self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(
                            self.format_det_answer(i, gt_bboxes_3d, results)
                        )
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += "\n".join(objs_near)
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?",
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}",
                        },
                    ]
                )

        lane_objs = self.lane_objs_info[results["sample_idx"]]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs["all_lane_pts"]))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs["lane_objects"].keys():
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"There are no objects on this lane.",
                                },
                            ]
                        )
                    else:
                        objs = []
                        for obj in lane_objs["lane_objects"][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = "\n".join(objs)
                        sources.append(
                            [
                                {
                                    "from": "human",
                                    "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The objects on this lane include:\n{answer}",
                                },
                            ]
                        )

        return sources

    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(
            f"({format_number(point[0])}, {format_number(point[1])})"
            for point in bezier_lane[0]
        )
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy = vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
        else:
            answer += "."

        return answer

    def trans_json_to_traj(self, traj_path):
        with open(traj_path, "r") as f:
            traj = json.load(f)
        traj = traj[-1]["A"][0]
        full_match = re.search(
            r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
            traj,
        )
        if full_match:
            coordinates_matches = re.findall(
                r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
            )
            coordinates = [
                tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                for coord in coordinates_matches
            ]
            coordinates_array = np.array(coordinates)
        return coordinates_array

    def get_meta_actions(self, traj, target_traj):
        """Determine meta actions needed to align trajectory with target"""

        # Calculate velocities and directions
        traj_velo = np.linalg.norm(traj[-1] - traj[0])
        target_velo = np.linalg.norm(target_traj[-1] - target_traj[0])

        # Determine speed meta action
        constant_eps = 0.3
        if abs(traj_velo - target_velo) < constant_eps:
            speed_meta = "maintain x-axis value"
        else:
            if traj_velo > target_velo:
                # if traj_velo > 2 * target_velo:
                #     speed_meta = "quick deceleration"
                # else:
                #     speed_meta = "gradual deceleration"
                speed_meta = "decrease %s to x-axis" % format_number(
                    traj_velo - target_velo
                )
            else:
                # if target_velo > 2 * traj_velo:
                #     speed_meta = "quick acceleration"
                # else:
                #     speed_meta = "gradual acceleration"
                speed_meta = "increase %s to x-axis" % format_number(
                    target_velo - traj_velo
                )
        # Determine steering meta action
        forward_th = 0.3
        final_lat_diff = abs(traj[-1, 1] - target_traj[-1, 1])

        if final_lat_diff < forward_th:
            steer_meta = "maintain y-axis value"
        else:
            if traj[-1, 1] < target_traj[-1, 1]:
                steer_meta = "increase %s to y-axis" % format_number(
                    target_traj[-1, 1] - traj[-1, 1]
                )
            else:
                steer_meta = "decrease %s to y-axis" % format_number(
                    traj[-1, 1] - target_traj[-1, 1]
                )

        return speed_meta, steer_meta

    def extend_traj(self, velocity, accel, dt=0.5, num_points=6):
        # Use velocity and acceleration from can_bus to generate trajectory
        pad_planning_traj = np.zeros((num_points, 2))
        # Update velocity using acceleration for first timestep only
        velocity[0] = velocity[0] + accel[0] * dt
        # Use constant velocity for all points
        for i in range(num_points):
            t = dt * (i + 1)
            pad_planning_traj[i, 0] = velocity[0] * t
            pad_planning_traj[i, 1] = velocity[1] * t

        return pad_planning_traj

    def __call__(self, results):
        traj = None
        results["pred_traj2"] = None
        results["min_index"] = None
        if "gt_planning" in results.keys():
            if self.use_sparsedrive_traj:
                planning_traj = self.sparsedrive_traj[results["sample_idx"]]
            else:
                planning_traj = results["gt_planning"][0, :, :2]
            mask = results["gt_planning_mask"][0].any(axis=1)
            planning_traj_mask = planning_traj[mask]
            if len(planning_traj_mask) == 6:
                if self.pred_traj_60:
                    formatted_points = "".join(
                        f"({format_number(point[0], 1)}, {format_number(point[1], 1)})"
                        for point in planning_traj
                    )
                else:
                    formatted_points = ", ".join(
                        f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                        for point in planning_traj_mask
                    )
                if self.pred_traj_60:
                    formatted_points_69 = "".join(
                        f"({format_number(point[0], 1)}, {format_number(point[1], 1)})"
                        for point in planning_traj[-7:]
                    )
                else:
                    formatted_points_69 = ", ".join(
                        f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                        for point in planning_traj_mask
                    )

                traj = f"Here is the planning trajectory [PT, {formatted_points}]."
            else:
                # Pad trajectory to length 6 using last point
                last_point = planning_traj[-1:]
                padding_length = 18 - len(planning_traj)
                padded_traj = np.concatenate(
                    [planning_traj, np.tile(last_point, (padding_length, 1))]
                )
                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in padded_traj
                )
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."

        sources = self.preprocess_vqa(results, traj)
        prompt = f"You are driving in {results['location']}. "

        online_sources = self.online_vqa(results)
        sources += online_sources
        random.shuffle(sources)
        command = self.command_str[results["command"]]

        if "gt_planning" in results.keys() and len(planning_traj_mask) == 6:
            if self.use_cot_v1:
                traj_index = self.closest_index[results["sample_idx"]] % 36
                gt_traj = f"the planning trajectory is [PT, {formatted_points}]."
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here are predefined planning trajectories %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Firstly, please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. "
                            + "\n"
                            + "Secondly, with the command %s, please provide the planning trajectory for the ego car using the selected predefined trajectory as a reference."
                            % command,
                        },
                        {
                            "from": "gpt",
                            "value": "Step 1: The classification result is %s, and it's trajectory is [PT, %s]. "
                            % (traj_index, formatted_traj)
                            + "\n"
                            + "Step 2: With the selected trajectory as a reference, %s"
                            % gt_traj,
                        },
                    ]
                ]

                sources = prompt_traj + sources

            elif self.use_gt_traj:
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here is a predefined planning trajectory %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference.",
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj

            elif self.use_pred_traj_seq:
                # traj_index = self.closest_index[results['sample_idx']] % 36
                # cls_traj = f"The result is <G{traj_index}>."
                gt_traj = f"The 0~9s result is [PT, {formatted_points}]."
                gt_traj_69 = f"The 6~9s result is [PT, {formatted_points_69}]."
                if self.choose_from_pred:
                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(2)]
                    ).strip()
                    traj_path1 = os.path.join(self.baseline_path, results["sample_idx"])
                    traj1 = self.trans_json_to_traj(traj_path1)
                    traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                    traj2 = self.trans_json_to_traj(traj_path2)
                    # Calculate distances between trajectories
                    # Stack trajectories first
                    trajs = np.stack([traj1, traj2])
                    results["pred_traj2"] = trajs

                    # Calculate distances and get index of minimum
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < planning_traj[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < planning_traj[..., -1, 1]
                        else "decrease"
                    )
                    results["min_index"] = traj_index
                elif self.use_rag:
                    if self.use_ego_mlp:
                        traj1_lidar = self.ego_mlp[results["sample_idx"]][
                            "final_planning"
                        ].numpy()
                        traj1 = copy.deepcopy(traj1_lidar)
                        traj1[:, 1] = -traj1_lidar[:, 0]
                        traj1[:, 0] = traj1_lidar[:, 1]
                    else:
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)

                    if results["sample_idx"] in self.rag_infos:
                        topk_indices = self.rag_infos[results["sample_idx"]][
                            : self.rag_topk
                        ]
                        topk_trajs = self.plan_anchor.reshape(-1, 18, 2)[topk_indices]
                    else:
                        # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                        # dists = np.sqrt(np.sum((extend_traj[None] - self.plan_anchor[results['command']])**2, axis=-1)).sum(-1)
                        dists = np.sqrt(
                            np.sum(
                                (
                                    planning_traj[None]
                                    - self.plan_anchor[results["command"]]
                                )
                                ** 2,
                                axis=-1,
                            )
                        ).sum(-1)
                        topk_indices = np.argsort(dists)[: self.rag_topk]
                        topk_trajs = self.plan_anchor[results["command"]][topk_indices]
                    if self.cat_pred_traj:
                        trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                    else:
                        trajs = topk_trajs
                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(len(trajs))]
                    ).strip()
                    results["pred_traj2"] = trajs
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    results["min_index"] = traj_index
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < planning_traj[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < planning_traj[..., -1, 1]
                        else "decrease"
                    )

                elif self.use_kmeans_traj:
                    trajs = self.plan_anchor[results["command"]]
                    if self.kmeans_pad_traj:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                            # traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                            results["pred_traj2"] = trajs
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                            results["pred_traj2"] = trajs
                    else:
                        results["pred_traj2"] = trajs

                    traj_tokens = " ".join(
                        [
                            f"<G{i}> {DEFAULT_POINT_TOKEN} "
                            for i in range(trajs.shape[0])
                        ]
                    ).strip()
                    dists = np.sqrt(
                        np.sum((trajs - planning_traj[None]) ** 2, axis=-1)
                    ).sum(-1)
                    traj_index = int(dists.argmin())
                    results["min_index"] = traj_index
                    vel_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 0] < traj1[..., -1, 0]
                        else "decrease"
                    )
                    steer_adjust = (
                        "increase"
                        if trajs[traj_index][..., -1, 1] < traj1[..., -1, 1]
                        else "decrease"
                    )

                speed_action, steer_action = self.get_meta_actions(
                    trajs[traj_index], planning_traj
                )

                if self.use_text_point:
                    endpoints = trajs[:, -1, :]
                    text_points = ", ".join(
                        [
                            f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                            for i, pt in enumerate(endpoints)
                        ]
                    )
                    # import pdb; pdb.set_trace()
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                elif self.add_vel:
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                elif self.add_ego:
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                else:
                    prompt_traj1 = [
                        [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario.",
                            },
                            {
                                "from": "gpt",
                                "value": f"The best trajectory is {traj_index}.",
                            },
                        ]
                    ]
                prompt_refine = [
                    [
                        {
                            "from": "human",
                            "value": "How to optimize this selected trajectory?",
                        },
                        {
                            "from": "gpt",
                            "value": "According to the current scene: "
                            + "\n"
                            + "- Velocity suggestions: %s" % speed_action
                            + "\n"
                            + "- Steering suggestions: %s" % steer_action,
                        },
                    ]
                ]

                if self.cot_with_speed:
                    # import pdb; pdb.set_trace()
                    current_speed = results["can_bus"][-3:-1]
                    current_accel = results["can_bus"][4:6]
                    speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                    accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                    prompt_traj2 = [
                        [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + f"please provide the future 6~9s planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": gt_traj_69},
                        ]
                    ]

                    prompt_traj3 = [
                        [
                            {
                                "from": "human",
                                "value": f"Please provide the future 0~9s planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": gt_traj},
                        ]
                    ]

                else:
                    prompt_traj2 = [
                        [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + "please provide the planning trajectory for the ego car.",
                            },
                            {"from": "gpt", "value": gt_traj},
                        ]
                    ]

                if self.use_other_qa:
                    if self.use_refine_step:
                        sources = prompt_traj1 + prompt_refine + prompt_traj2 + sources
                    else:
                        sources = prompt_traj1 + prompt_traj2 + prompt_traj3 + sources
                else:
                    if self.use_refine_step:
                        sources = prompt_traj1 + prompt_refine + prompt_traj2
                    else:
                        sources = prompt_traj1 + prompt_traj2 + prompt_traj3

            else:
                traj_index = self.closest_index[results["sample_idx"]] % 36
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here are predefined planning trajectories %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. ",
                        },
                        {
                            "from": "gpt",
                            "value": "The classification result is %s. " % traj_index,
                        },
                        {
                            "from": "human",
                            "value": "With the command %s, please provide the planning trajectory for the ego car using the predefined trajectory [PT, %s] as a reference."
                            % (command, formatted_traj),
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj + sources

        else:
            results["loss_mask"] = False
            if self.use_gt_traj:
                gt_traj = f"The planning trajectory is [PT, {formatted_points}]."
                prompt_traj = [
                    [
                        {
                            "from": "human",
                            "value": "Here is a predefined planning trajectory %s. "
                            % DEFAULT_TRAJ_TOKEN
                            + "\n"
                            + "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference.",
                        },
                        {"from": "gpt", "value": gt_traj},
                    ]
                ]

                sources = prompt_traj

            else:
                # # Calculate L2 distance between gt_traj and plan_anchor
                # import pdb; pdb.set_trace()
                gt_traj = f"the planning trajectory is [PT, {formatted_points}]."
                # Calculate velocity and direction from existing points
                # print('traj shape', planning_traj.shape)
                existing_points = planning_traj  # [1, N, 2]
                if planning_traj.shape[0] == 0:
                    pad_planning_traj = self.extend_traj(
                        results["can_bus"][10:12], results["can_bus"][4:6]
                    )
                elif planning_traj.shape[0] == 1:
                    velocities = existing_points[-1]  # [1, N-1, 2]
                    last_velocity = velocities[-1]  # Use last velocity

                    # Extrapolate remaining points using velocity
                    num_missing = 6 - planning_traj.shape[0]
                    extrapolated_points = []
                    last_point = existing_points[-1]

                    for i in range(num_missing):
                        next_point = last_point + last_velocity
                        extrapolated_points.append(next_point)
                        last_point = next_point
                    if num_missing > 0:
                        extrapolated_points = np.stack(extrapolated_points, axis=0)
                        pad_planning_traj = np.concatenate(
                            [planning_traj, extrapolated_points], axis=0
                        )
                elif planning_traj.shape[0] > 1 and planning_traj.shape[0] < 6:
                    velocities = (
                        existing_points[-2] - existing_points[-1]
                    )  # [1, N-1, 2]
                    last_velocity = velocities[-1]  # Use last velocity

                    # Extrapolate remaining points using velocity
                    num_missing = 6 - planning_traj.shape[0]
                    extrapolated_points = []
                    last_point = existing_points[-1]

                    for i in range(num_missing):
                        next_point = last_point + last_velocity
                        extrapolated_points.append(next_point)
                        last_point = next_point
                    if num_missing > 0:
                        extrapolated_points = np.stack(extrapolated_points, axis=0)
                        pad_planning_traj = np.concatenate(
                            [planning_traj, extrapolated_points], axis=0
                        )
                anchor_trajs = self.plan_anchor[
                    results["command"]
                ]  # Get anchors for current command
                # import pdb; pdb.set_trace()
                # Calculate distances between gt_traj and each anchor trajectory
                distances = np.sqrt(
                    (
                        (
                            anchor_trajs.reshape(-1, 12)
                            - pad_planning_traj.reshape(1, 12)
                        )
                        ** 2
                    ).sum(-1)
                )

                # Get index of closest anchor trajectory
                traj_index = np.argmin(distances)
                formatted_traj = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in self.plan_anchor[results["command"], traj_index]
                )

                if self.use_pred_traj_seq:
                    if self.choose_from_pred:
                        traj_tokens = " ".join(
                            [f"<G{i}> {DEFAULT_POINT_TOKEN} " for i in range(2)]
                        ).strip()
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)
                        traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                        traj2 = self.trans_json_to_traj(traj_path2)
                        trajs = np.stack([traj1, traj2])
                        results["pred_traj2"] = trajs

                        # Calculate distances and get index of minimum
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                    elif self.use_rag:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)

                        if results["sample_idx"] in self.rag_infos:
                            topk_indices = self.rag_infos[results["sample_idx"]][
                                : self.rag_topk
                            ]
                            topk_trajs = self.plan_anchor.reshape(-1, 6, 2)[
                                topk_indices
                            ]
                        else:
                            # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                            # dists = np.sqrt(np.sum((extend_traj[None] - self.plan_anchor[results['command']])**2, axis=-1)).sum(-1)
                            dists = np.sqrt(
                                np.sum(
                                    (
                                        pad_planning_traj[None]
                                        - self.plan_anchor[results["command"]]
                                    )
                                    ** 2,
                                    axis=-1,
                                )
                            ).sum(-1)
                            topk_indices = np.argsort(dists)[: self.rag_topk]
                            topk_trajs = self.plan_anchor[results["command"]][
                                topk_indices
                            ]
                        if self.cat_pred_traj:
                            trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                        else:
                            trajs = topk_trajs
                        traj_tokens = " ".join(
                            [
                                f"<G{i}> {DEFAULT_POINT_TOKEN} "
                                for i in range(trajs.shape[0])
                            ]
                        ).strip()
                        results["pred_traj2"] = trajs
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                        # vel_adjust = 'increase' if trajs[traj_index][...,-1, 0] < planning_traj[..., -1, 0] else 'decrease'
                        # steer_adjust = 'increase' if trajs[traj_index][...,-1, 1] < planning_traj[..., -1, 1] else 'decrease'
                    elif self.use_kmeans_traj:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                            # traj1 = self.trans_json_to_traj(traj_path1)
                            # trajs = np.concatenate([traj1[None], trajs], axis=0)
                            # results['pred_traj2'] = trajs
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)

                        trajs = self.plan_anchor[results["command"]]
                        if self.kmeans_pad_traj:
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                        results["pred_traj2"] = trajs
                        traj_tokens = " ".join(
                            [
                                f"<G{i}> {DEFAULT_POINT_TOKEN} "
                                for i in range(trajs.shape[0])
                            ]
                        ).strip()
                        dists = np.sqrt(
                            np.sum((trajs - pad_planning_traj[None]) ** 2, axis=-1)
                        ).sum(-1)
                        traj_index = int(dists.argmin())
                        results["min_index"] = traj_index
                        # vel_adjust = 'increase' if trajs[traj_index][...,-1, 0] < planning_traj[..., -1, 0] else 'decrease'
                        # steer_adjust = 'increase' if trajs[traj_index][...,-1, 1] < planning_traj[..., -1, 1] else 'decrease'
                        # traj_tokens = ' '.join([f'<G{i}> {DEFAULT_POINT_TOKEN} ' for i in range(36)]).strip()
                    # vel = 'With the velocity of (%s, %s)' % (format_number(np.clip(results['can_bus'][4],0,1000), 2), format_number(results['can_bus'][5], 2))
                    if self.use_text_point:
                        endpoints = trajs[:, -1, :]
                        text_points = ", ".join(
                            [
                                f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                                for i, pt in enumerate(endpoints)
                            ]
                        )

                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    elif self.add_vel:
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    elif self.add_ego:
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    else:
                        prompt_traj = [
                            [
                                {
                                    "from": "human",
                                    "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                    + "Please select the best trajectory in the current scenario.",
                                },
                                {
                                    "from": "gpt",
                                    "value": f"The best trajectory is {traj_index}.",
                                },
                            ]
                        ]

                    # if self.use_two_image:
                    #     prompt_traj = prompt_traj + prompt_traj
                    #     [{"from": 'human',
                    #     "value": "Here is a predefined planning trajectory %s. " % DEFAULT_TRAJ_TOKEN + '\n' +
                    #             "Please provide the planning trajectory for the ego car with the predefined trajectory as a reference."},
                    #     {"from": 'gpt',
                    #     "value": gt_traj}]]

                    # else:
                    #     prompt_traj = [
                    #         [{"from": 'human',
                    #         "value": "Here are predefined planning trajectories %s. " % DEFAULT_TRAJ_TOKEN + '\n' +
                    #                 "Please classify the scene into one of the 36 predefined trajectories, indexed from 0 to 35. "},
                    #         {"from": 'gpt',
                    #         "value": "The classification result is %s. " % traj_index}]]

                if self.use_other_qa:
                    sources = prompt_traj + sources
                else:
                    sources = prompt_traj
        # import pdb; pdb.set_trace()
        vqa_anno = [item for pair in sources for item in pair]
        # if self.use_pred_traj_seq:

        #     vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']
        #     if not self.only_cls and len(planning_traj) == 6 and self.use_two_image:
        #         vqa_anno[2]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[2]['value']
        #         if self.use_refine_step:
        #             vqa_anno[4]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[4]['value']
        # else:
        vqa_anno[0]["value"] = (
            DEFAULT_IMAGE_TOKEN + "\n" + prompt + vqa_anno[0]["value"]
        )
        # import pdb; pdb.set_trace()
        vqa_converted = preprocess([vqa_anno], self.tokenizer, True, has_traj=True)
        input_ids = vqa_converted["input_ids"][0]
        vlm_labels = vqa_converted["labels"][0]

        results["input_ids"] = input_ids
        results["vlm_labels"] = vlm_labels

        return results


@PIPELINES.register_module()
class LoadAnnoatationVQATest:
    def __init__(
        self,
        base_conv_path,
        base_vqa_path,
        tokenizer,
        max_length,
        base_counter_path=None,
        load_type=["conv", "planning", "counter"],
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_conv_path = base_conv_path
        self.base_vqa_path = base_vqa_path
        self.base_counter_path = base_counter_path
        self.load_type = load_type
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results):
        sources = []
        if "planning" in self.load_type:  # planning trajs
            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please provide the planning trajectory for the ego car without reasons.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            )
        if "short" in self.load_type:  # short driving action
            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            )
        if "conv" in self.load_type:  # conversation
            question = random.sample(self.template, 1)[0]  # detailed description
            sources.append(
                [{"from": "human", "value": question}, {"from": "gpt", "value": ""}]
            )
            if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
                with open(
                    self.base_conv_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for pair in data_qa:
                    sources.append(
                        [
                            {"from": "human", "value": pair["question"]},
                            {"from": "gpt", "value": ""},
                        ]
                    )
            if os.path.exists(
                self.base_vqa_path + results["sample_idx"] + ".json"
            ):  # attention + action + counter * 2
                with open(
                    self.base_vqa_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for pair in data_qa:
                    sources.append(
                        [
                            {"from": "human", "value": pair["question"]},
                            {"from": "gpt", "value": ""},
                        ]
                    )
        if "counter" in self.load_type:
            all_counters = pickle.load(
                open(
                    os.path.join(
                        self.base_counter_path + results["sample_idx"] + ".pkl"
                    ),
                    "rb",
                )
            )
            for data in all_counters:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?",
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )
        return sources

    def __call__(self, results):
        sources = self.preprocess_vqa(results)
        prompt = f"You are driving in {results['location']}. "
        vlm_labels = [anno[0]["value"] for anno in sources]

        for anno in sources:
            anno[0]["value"] = DEFAULT_IMAGE_TOKEN + "\n" + prompt + anno[0]["value"]
            anno[1]["value"] = ""
        vqa_converted = preprocess(sources, self.tokenizer, True, False)
        input_ids = vqa_converted["input_ids"]
        results["input_ids"] = input_ids
        results["vlm_labels"] = vlm_labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadAnnoatationVQATestSOLVE:
    def __init__(
        self,
        base_conv_path,
        base_vqa_path,
        tokenizer,
        max_length,
        base_counter_path=None,
        load_type=["conv", "planning", "counter"],
        drivelm_path=None,
        used_drivelm_keys=None,
        used_vqa_keys=None,
        pred_res_traj=None,
        planning_with_behavior=False,
        planning_with_gt_behavior=False,
        add_dist_token=False,
        bin_info_path=None,
        yaw_info_path=None,
        accel_info_path=None,
        with_gt_behavior_context=False,
        use_gt_traj=False,
        use_cot_v1=False,
        use_trajemb_cot=False,
        pred_fut10_traj=False,
        use_kmeans_traj=False,
        use_pred_traj_seq=False,
        only_cls=False,
        use_text_traj=False,
        only_refine=False,
        cat_pred_traj=False,
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        keans_div6_path="/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/data/kmeans_plan_ego_div6.npy",
        closed_refer_num=6,
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_val.pkl",
        choose_from_pred=False,
        use_refine_step=False,
        use_two_image=False,
        use_rag=False,
        rag_topk=5,
        kmeans_pad_traj=False,
        use_qwen=False,
        use_qwenvl_25=False,
        use_ego_mlp=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        cot_with_speed=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_val_dict.pkl",
        use_classv3=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        if use_qwen or use_qwenvl_25:
            pass
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.use_qwen = use_qwen
        self.use_qwenvl_25 = use_qwenvl_25
        self.base_conv_path = base_conv_path
        self.base_vqa_path = base_vqa_path
        self.base_counter_path = base_counter_path
        self.load_type = load_type
        self.used_vqa_keys = used_vqa_keys
        self.pred_res_traj = pred_res_traj
        self.planning_with_behavior = planning_with_behavior
        self.planning_with_gt_behavior = planning_with_gt_behavior
        self.with_gt_behavior_context = with_gt_behavior_context
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.use_trajemb_cot = use_trajemb_cot
        self.use_gt_traj = use_gt_traj
        self.use_cot_v1 = use_cot_v1
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_ego_mlp = use_ego_mlp
        self.use_classv3 = use_classv3
        self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        self.pred_fut10_traj = pred_fut10_traj
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_text_traj = use_text_traj
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.choose_from_pred = choose_from_pred
        self.only_refine = only_refine
        self.use_two_image = use_two_image
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_text_point = use_text_point
        self.add_vel = add_vel
        self.add_ego = add_ego
        self.use_qwen = use_qwen
        self.cot_with_speed = cot_with_speed
        if "9s" in kmeans_path:
            plan_anchor_lidar_9s = np.load(kmeans_path)
            self.plan_anchor = plan_anchor_lidar_9s.copy()
        else:
            plan_anchor_lidar = np.load(kmeans_path)
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
            self.plan_anchor[2, 0] = np.zeros_like(self.plan_anchor[2, 0])

        if self.load_type == ["closed_loop"]:
            self.closed_refer_num = closed_refer_num
            plan_anchor_ego_div6 = np.load(keans_div6_path)
            self.kmeans_div6 = plan_anchor_ego_div6.copy()

        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results, pred_traj_str=None):
        sources = []

        if "short" in self.load_type:  # short driving action
            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            )
        if "conv" in self.load_type:  # conversation
            question = random.sample(self.template, 1)[0]  # detailed description
            sources.append(
                [{"from": "human", "value": question}, {"from": "gpt", "value": ""}]
            )
            if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
                with open(
                    self.base_conv_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for pair in data_qa:
                    sources.append(
                        [
                            {"from": "human", "value": pair["question"]},
                            {"from": "gpt", "value": ""},
                        ]
                    )

        if "counter" in self.load_type:
            all_counters = pickle.load(
                open(
                    os.path.join(
                        self.base_counter_path + results["sample_idx"] + ".pkl"
                    ),
                    "rb",
                )
            )
            for data in all_counters:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?",
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )

        if "vqa" in self.load_type:  # vqa
            if os.path.exists(
                self.base_vqa_path + results["sample_idx"] + ".json"
            ):  # attention + action + counter * 2
                with open(
                    self.base_vqa_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for idx, pair in enumerate(data_qa):
                    if "attention" in self.used_vqa_keys and idx == 0:
                        sources.append(
                            [
                                {"from": "human", "value": pair["question"]},
                                {"from": "gpt", "value": ""},
                            ]
                        )
                    if "action" in self.used_vqa_keys and idx == 1:
                        if self.planning_with_gt_behavior:
                            sources.append(
                                [
                                    {"from": "human", "value": pair["question"]},
                                    {"from": "gpt", "value": pair["answer"]},
                                ]
                            )
                        else:
                            sources.append(
                                [
                                    {"from": "human", "value": pair["question"]},
                                    {"from": "gpt", "value": ""},
                                ]
                            )
                    if "counter" in self.used_vqa_keys and idx > 1:
                        sources.append(
                            [
                                {"from": "human", "value": pair["question"]},
                                {"from": "gpt", "value": ""},
                            ]
                        )

        if "planning" in self.load_type:  # planning trajs when test only this
            if self.pred_res_traj:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": "Please predict the residual value for each point in the initial trajectory to make it a better trajectory."
                            + pred_traj_str,
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )
            else:
                if self.use_pred_traj_seq:  # True
                    if self.choose_from_pred:  # False
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)
                        traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                        traj2 = self.trans_json_to_traj(traj_path2)
                        trajs = np.stack([traj1, traj2])
                        results["pred_traj2"] = trajs
                    elif self.use_rag:  # True
                        if self.use_ego_mlp:  # True commandgetk-means
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)
                        if results["sample_idx"] in self.rag_infos:  # True
                            # import pdb; pdb.set_trace()
                            topk_indices = self.rag_infos[results["sample_idx"]][
                                : self.rag_topk
                            ]
                            topk_trajs = self.plan_anchor.reshape(-1, 6, 2)[
                                topk_indices
                            ]
                        else:
                            # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                            dists = np.sqrt(
                                np.sum(
                                    (traj1 - self.plan_anchor[results["command"]]) ** 2,
                                    axis=-1,
                                )
                            ).sum(-1)
                            topk_indices = np.argsort(dists)[: self.rag_topk]
                            topk_trajs = self.plan_anchor[results["command"]][
                                topk_indices
                            ]
                        if self.cat_pred_traj:  # True
                            trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                        else:
                            trajs = topk_trajs
                        results["pred_traj2"] = trajs  # Translated note.
                    elif self.use_kmeans_traj:  # False
                        trajs = self.plan_anchor[results["command"]]
                        # import pdb; pdb.set_trace()
                        if self.kmeans_pad_traj:
                            if self.use_ego_mlp:
                                traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                    "final_planning"
                                ].numpy()
                                traj1 = copy.deepcopy(traj1_lidar)
                                traj1[:, 1] = -traj1_lidar[:, 0]
                                traj1[:, 0] = traj1_lidar[:, 1]
                            else:
                                traj_path1 = os.path.join(
                                    self.baseline_path, results["sample_idx"]
                                )
                                traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                        results["pred_traj2"] = trajs

                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN}" for i in range(trajs.shape[0])]
                    ).strip()

                    if self.use_text_point:  # False
                        endpoints = trajs[:, -1, :]
                        text_points = ", ".join(
                            [
                                f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                                for i, pt in enumerate(endpoints)
                            ]
                        )

                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {"from": "gpt", "value": ""},
                            ]
                        ]

                    elif self.add_vel:  # False
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    elif self.add_ego:  # False
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    else:
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    prompt_refine = [
                        {
                            "from": "human",
                            "value": "How to optimize this selected trajectory?",
                        },
                        {"from": "gpt", "value": ""},
                    ]
                    if self.cot_with_speed:  # True use can_bus
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + f"please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    else:
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + "please provide the planning trajectory for the ego car.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    sources.append(prompt_cot1)

                    if self.use_refine_step:  # False
                        sources.append(prompt_refine)

                    sources.append(prompt_cot2)

                else:
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": "Please provide the planning trajectory for the ego car without reasons.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    )

        if "closed_loop" in self.load_type:
            # position if "planning" in self.load_type:
            if self.pred_res_traj:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": "Please predict the residual value for each point in the initial trajectory to make it a better trajectory."
                            + pred_traj_str,
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )
            else:
                if self.use_pred_traj_seq:  # True
                    # ==== closed-loop sample_idx ====
                    if "closed_loop" in self.load_type:
                        # kmeans_anchor
                        # shape: self.kmeans_anchor = (3, A, 6, 2) 0 command
                        cmd = int(results["command"])  # 0: LEFT, 1: RIGHT, 2: STRAIGHT
                        anchors = self.kmeans_div6[cmd]  # (A, 6, 2)
                        A = anchors.shape[0]
                        closed_refer_num = getattr(self, "closed_refer_num", 6)
                        n = min(closed_refer_num, A)
                        trajs = anchors[:n].astype(np.float32)  # (n, 6, 2)
                        results["pred_traj2"] = trajs

                    # ==== ====
                    elif self.choose_from_pred:  # False
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)
                        traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                        traj2 = self.trans_json_to_traj(traj_path2)
                        trajs = np.stack([traj1, traj2])
                        results["pred_traj2"] = trajs

                    elif self.use_rag:  # True
                        if self.use_ego_mlp:  # True
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)

                        if results["sample_idx"] in self.rag_infos:  # True
                            topk_indices = self.rag_infos[results["sample_idx"]][
                                : self.rag_topk
                            ]
                            topk_trajs = self.plan_anchor.reshape(-1, 6, 2)[
                                topk_indices
                            ]
                        else:
                            dists = np.sqrt(
                                np.sum(
                                    (traj1 - self.plan_anchor[results["command"]]) ** 2,
                                    axis=-1,
                                )
                            ).sum(-1)
                            topk_indices = np.argsort(dists)[: self.rag_topk]
                            topk_trajs = self.plan_anchor[results["command"]][
                                topk_indices
                            ]

                        if self.cat_pred_traj:  # True
                            trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                        else:
                            trajs = topk_trajs
                        results["pred_traj2"] = trajs

                    elif self.use_kmeans_traj:  # False
                        trajs = self.plan_anchor[results["command"]]
                        if self.kmeans_pad_traj:
                            if self.use_ego_mlp:
                                traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                    "final_planning"
                                ].numpy()
                                traj1 = copy.deepcopy(traj1_lidar)
                                traj1[:, 1] = -traj1_lidar[:, 0]
                                traj1[:, 0] = traj1_lidar[:, 1]
                            else:
                                traj_path1 = os.path.join(
                                    self.baseline_path, results["sample_idx"]
                                )
                                traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                        results["pred_traj2"] = trajs

                    # `trajs` build tokens prompt
                    # trajs closed-loop
                    traj_tokens = " ".join(
                        [
                            f"<G{i}> {DEFAULT_POINT_TOKEN}"
                            for i in range(results["pred_traj2"].shape[0])
                        ]
                    ).strip()

                    if self.use_text_point:  # False
                        endpoints = results["pred_traj2"][:, -1, :]
                        text_points = ", ".join(
                            [
                                f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                                for i, pt in enumerate(endpoints)
                            ]
                        )
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            [
                                {
                                    "from": "human",
                                    "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                    f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {"from": "gpt", "value": ""},
                            ]
                        ]

                    elif self.add_vel:  # False
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    elif self.add_ego:  # False
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                "Please select the best trajectory in the current scenario with ego stage <ego>.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    else:
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                "Please select the best trajectory in the current scenario.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    prompt_refine = [
                        {
                            "from": "human",
                            "value": "How to optimize this selected trajectory?",
                        },
                        {"from": "gpt", "value": ""},
                    ]

                    if self.cot_with_speed:  # True ( can_bus)
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + f"please provide the planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    else:
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + "please provide the planning trajectory for the ego car.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    sources.append(prompt_cot1)
                    if self.use_refine_step:  # False
                        sources.append(prompt_refine)
                    sources.append(prompt_cot2)

                else:
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": "Please provide the planning trajectory for the ego car without reasons.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    )

        return sources

    def extend_traj(self, velocity, accel, dt=0.5, num_points=6):
        # Use velocity and acceleration from can_bus to generate trajectory
        pad_planning_traj = np.zeros((num_points, 2))
        # Update velocity using acceleration for first timestep only
        velocity[0] = velocity[0] + accel[0] * dt
        # Use constant velocity for all points
        for i in range(num_points):
            t = dt * (i + 1)
            pad_planning_traj[i, 0] = velocity[0] * t
            pad_planning_traj[i, 1] = velocity[1] * t

        return pad_planning_traj

    def trans_json_to_traj(self, traj_path):
        with open(traj_path, "r") as f:
            traj = json.load(f)
        traj = traj[-1]["A"][0]
        full_match = re.search(
            r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
            traj,
        )
        if full_match:
            coordinates_matches = re.findall(
                r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
            )
            coordinates = [
                tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                for coord in coordinates_matches
            ]
            coordinates_array = np.array(coordinates)
        return coordinates_array

    def __call__(
        self, results
    ):  # dict_keys(['can_bus', 'command', 'location', 'sample_idx', 'pts_filename', 'sweeps', 'ego_pose', 'ego_pose_inv', 'prev_idx', 'next_idx', 'scene_token', 'frame_idx', 'timestamp', 'cam_infos', 'img_timestamp', 'img_filename', 'lidar2img', 'intrinsics', 'extrinsics', 'prev_exists', 'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d', 'filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_bboxes', 'centers2d', 'gt_labels', 'depths', 'scale', 'scale_idx', 'keep_ratio', 'pad_fix_size', 'pad_size_divisor'])
        # import ipdb; ipdb.set_trace()
        # import os, ipdb
        # if os.getenv("DEBUG") == "1": ipdb.set_trace()
        if self.pred_res_traj is not None:  # None
            if os.path.exists(self.pred_res_traj + results["sample_idx"]):
                with open(
                    self.pred_res_traj + results["sample_idx"], "r", encoding="utf8"
                ) as f:
                    pred_data = json.load(f)
                    traj = pred_data[-1]["A"][0]
                    full_match = re.search(
                        r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
                        traj,
                    )
                    if full_match:
                        coordinates_matches = re.findall(
                            r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
                        )
                        coordinates = [
                            tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                            for coord in coordinates_matches
                        ]
                        coordinates_array = np.array(coordinates)
                        pred_traj = coordinates_array

                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in pred_traj
                )
                pred_traj_str = f" Here is the initial trajectory [{formatted_points}]."
            else:
                pred_traj_str = None
        else:
            pred_traj_str = None
        # import ipdb; ipdb.set_trace()
        sources = self.preprocess_vqa(
            results, pred_traj_str
        )  # [[{'from': 'human', 'value': 'Here are predefined trajectories [<G0> <point> <G1> <point> <G2> <point> <G3> <point> <G4> <point> <G5> <point>] for the ego car. Please select the best trajectory in the current scenario.'}, {'from': 'gpt', 'value': ''}], [{'from': 'human', 'value': 'With the selected trajectory as a reference <traj>, please provide the planning trajectory for the ego car, which has a velocity of (9.38,0.00) m/s and an acceleration of (0.04,0.55) m/s^2.'}, {'from': 'gpt', 'value': ''}]]
        prompt = f"You are driving in {results['location']}. "

        for anno in sources[:1]:
            anno[0]["value"] = DEFAULT_IMAGE_TOKEN + "\n" + prompt + anno[0]["value"]
            anno[1]["value"] = ""
        # [[{'from': 'human', 'value': '<image>\nYou are driving in singapore. Here are predefined trajectories [<G0> <point> <G1> <point> <G2> <point> <G3> <point> <G4> <point> <G5> <point>] for the ego car. Please select the best trajectory in the current scenario.'}, {'from': 'gpt', 'value': ''}], [{'from': 'human', 'value': 'With the selected trajectory as a reference <traj>, please provide the planning trajectory for the ego car, which has a velocity of (9.38,0.00) m/s and an acceleration of (0.04,0.55) m/s^2.'}, {'from': 'gpt', 'value': ''}]]
        has_traj = self.use_trajemb_cot or self.use_pred_traj_seq
        vqa_converted = preprocess_traj(
            sources,
            self.tokenizer,
            True,
            False,
            has_traj=has_traj,
            use_qwen=self.use_qwen,
            use_qwenvl_25=self.use_qwenvl_25,
        )
        input_ids = vqa_converted["input_ids"]
        results["input_ids"] = input_ids
        vlm_labels = [anno[0]["value"] for anno in sources]
        results["vlm_labels"] = vlm_labels
        # import os, ipdb
        # if os.getenv("NEW_DEBUG") == "1": ipdb.set_trace()
        # if "combined_label_v3" in results and results["combined_label_v3"] is not None:
        #     idx = _v3_id_from_results(results, v3_label2id=getattr(self, "v3_label2id", None), default=-1)
        #     results['classv3_index'] = torch.tensor(idx, device=input_ids[0].device, dtype=torch.long)

        # if self.use_classv3:
        # # v3 v3 fields fields
        #     idx = _v3_id_from_results(results, v3_label2id=getattr(self, "v3_label2id", None), default=-1)
        #     results['meta_index'] = torch.tensor(idx, dtype=torch.long)
        # else:
        # # + MAPPING
        # # combined_id_from_results v3
        # # fields MAPPING
        #     results['meta_index'] = combined_id_from_results(
        #         results,
        # mapping_table=MAPPING, #
        #         default=-1,
        #         device=input_ids.device
        #     )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadAnnoatationVQATestFut18:
    def __init__(
        self,
        base_conv_path,
        base_vqa_path,
        tokenizer,
        max_length,
        base_counter_path=None,
        load_type=["conv", "planning", "counter"],
        drivelm_path=None,
        used_drivelm_keys=None,
        used_vqa_keys=None,
        pred_res_traj=None,
        planning_with_behavior=False,
        planning_with_gt_behavior=False,
        add_dist_token=False,
        bin_info_path=None,
        yaw_info_path=None,
        accel_info_path=None,
        with_gt_behavior_context=False,
        use_gt_traj=False,
        use_cot_v1=False,
        use_trajemb_cot=False,
        pred_fut10_traj=False,
        use_kmeans_traj=False,
        use_pred_traj_seq=False,
        only_cls=False,
        use_text_traj=False,
        only_refine=False,
        cat_pred_traj=False,
        kmeans_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/kmeans/kmeans_plan_36.npy",
        baseline_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/baseline",
        e24_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive/results_planning_only/load_ft_petr_e24_lidartraj",
        rag_path="/nfs/dataset-ofs-voyager-research/xschen/repos/Agent-Driver/topk_indices_dict_36_9s_val.pkl",
        choose_from_pred=False,
        use_refine_step=False,
        use_two_image=False,
        use_rag=False,
        rag_topk=5,
        kmeans_pad_traj=False,
        use_qwen=False,
        use_qwenvl_25=False,
        use_ego_mlp=False,
        use_text_point=False,
        add_vel=False,
        add_ego=False,
        cot_with_speed=False,
        only_eval_69=False,
        ego_mlp_path="/nfs/dataset-ofs-voyager-research/xschen/repos/OmniDrive-develop/data/nuscenes/ego_mlp_val_dict_9s.pkl",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        if use_qwen or use_qwenvl_25:
            pass
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.use_qwen = use_qwen
        self.use_qwenvl_25 = use_qwenvl_25
        self.base_conv_path = base_conv_path
        self.base_vqa_path = base_vqa_path
        self.base_counter_path = base_counter_path
        self.load_type = load_type
        self.used_vqa_keys = used_vqa_keys
        self.pred_res_traj = pred_res_traj
        self.planning_with_behavior = planning_with_behavior
        self.planning_with_gt_behavior = planning_with_gt_behavior
        self.with_gt_behavior_context = with_gt_behavior_context
        self.command_str = {0: "TURN LEFT", 1: "TURN RIGHT", 2: "GO STRAIGHT"}
        self.use_trajemb_cot = use_trajemb_cot
        self.use_gt_traj = use_gt_traj
        self.use_cot_v1 = use_cot_v1
        self.use_kmeans_traj = use_kmeans_traj
        self.kmeans_pad_traj = kmeans_pad_traj
        self.use_ego_mlp = use_ego_mlp
        self.ego_mlp = np.load(ego_mlp_path, allow_pickle=True)
        self.pred_fut10_traj = pred_fut10_traj
        self.use_pred_traj_seq = use_pred_traj_seq
        self.only_cls = only_cls
        self.use_text_traj = use_text_traj
        self.baseline_path = baseline_path
        self.e24_path = e24_path
        self.choose_from_pred = choose_from_pred
        self.only_refine = only_refine
        self.use_two_image = use_two_image
        self.use_refine_step = use_refine_step
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.cat_pred_traj = cat_pred_traj
        self.rag_infos = mmcv.load(rag_path)
        self.use_text_point = use_text_point
        self.add_vel = add_vel
        self.add_ego = add_ego
        self.use_qwen = use_qwen
        self.cot_with_speed = cot_with_speed
        self.only_eval_69 = only_eval_69
        if "9s" in kmeans_path:
            plan_anchor_lidar_9s = np.load(kmeans_path)
            self.plan_anchor = plan_anchor_lidar_9s.copy()
        else:
            plan_anchor_lidar = np.load(kmeans_path)
            plan_anchor_ego = plan_anchor_lidar.copy()
            plan_anchor_ego[..., 0] = plan_anchor_lidar[..., 1]
            plan_anchor_ego[..., 1] = -plan_anchor_lidar[..., 0]
            self.plan_anchor = plan_anchor_ego[
                [1, 0, 2]
            ]  # 0: left, 1: right, 2: forward
            self.plan_anchor[2, 0] = np.zeros_like(self.plan_anchor[2, 0])
        self.side = {
            "singapore": "left",
            "boston": "right",
        }
        self.template = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
        ]

    def preprocess_vqa(self, results, pred_traj_str=None):
        sources = []

        if "short" in self.load_type:  # short driving action
            sources.append(
                [
                    {
                        "from": "human",
                        "value": "Please shortly describe your driving action.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            )
        if "conv" in self.load_type:  # conversation
            question = random.sample(self.template, 1)[0]  # detailed description
            sources.append(
                [{"from": "human", "value": question}, {"from": "gpt", "value": ""}]
            )
            if os.path.exists(self.base_conv_path + results["sample_idx"] + ".json"):
                with open(
                    self.base_conv_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for pair in data_qa:
                    sources.append(
                        [
                            {"from": "human", "value": pair["question"]},
                            {"from": "gpt", "value": ""},
                        ]
                    )

        if "counter" in self.load_type:
            all_counters = pickle.load(
                open(
                    os.path.join(
                        self.base_counter_path + results["sample_idx"] + ".pkl"
                    ),
                    "rb",
                )
            )
            for data in all_counters:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?",
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )

        if "vqa" in self.load_type:  # vqa
            if os.path.exists(
                self.base_vqa_path + results["sample_idx"] + ".json"
            ):  # attention + action + counter * 2
                with open(
                    self.base_vqa_path + results["sample_idx"] + ".json", "r"
                ) as f:
                    data_qa = json.load(f)

                for idx, pair in enumerate(data_qa):
                    if "attention" in self.used_vqa_keys and idx == 0:
                        sources.append(
                            [
                                {"from": "human", "value": pair["question"]},
                                {"from": "gpt", "value": ""},
                            ]
                        )
                    if "action" in self.used_vqa_keys and idx == 1:
                        if self.planning_with_gt_behavior:
                            sources.append(
                                [
                                    {"from": "human", "value": pair["question"]},
                                    {"from": "gpt", "value": pair["answer"]},
                                ]
                            )
                        else:
                            sources.append(
                                [
                                    {"from": "human", "value": pair["question"]},
                                    {"from": "gpt", "value": ""},
                                ]
                            )
                    if "counter" in self.used_vqa_keys and idx > 1:
                        sources.append(
                            [
                                {"from": "human", "value": pair["question"]},
                                {"from": "gpt", "value": ""},
                            ]
                        )

        if "planning" in self.load_type:  # planning trajs
            if self.pred_res_traj:
                sources.append(
                    [
                        {
                            "from": "human",
                            "value": "Please predict the residual value for each point in the initial trajectory to make it a better trajectory."
                            + pred_traj_str,
                        },
                        {"from": "gpt", "value": ""},
                    ]
                )
            else:
                if self.use_pred_traj_seq:
                    if self.choose_from_pred:
                        traj_path1 = os.path.join(
                            self.baseline_path, results["sample_idx"]
                        )
                        traj1 = self.trans_json_to_traj(traj_path1)
                        traj_path2 = os.path.join(self.e24_path, results["sample_idx"])
                        traj2 = self.trans_json_to_traj(traj_path2)
                        trajs = np.stack([traj1, traj2])
                        results["pred_traj2"] = trajs
                    elif self.use_rag:
                        if self.use_ego_mlp:
                            traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                "final_planning"
                            ].numpy()
                            traj1 = copy.deepcopy(traj1_lidar)
                            traj1[:, 1] = -traj1_lidar[:, 0]
                            traj1[:, 0] = traj1_lidar[:, 1]
                        else:
                            traj_path1 = os.path.join(
                                self.baseline_path, results["sample_idx"]
                            )
                            traj1 = self.trans_json_to_traj(traj_path1)
                        if results["sample_idx"] in self.rag_infos:
                            # import pdb; pdb.set_trace()
                            topk_indices = self.rag_infos[results["sample_idx"]][
                                : self.rag_topk
                            ]
                            topk_trajs = self.plan_anchor.reshape(-1, 18, 2)[
                                topk_indices
                            ]
                        else:
                            # extend_traj = self.extend_traj(results['can_bus'][10:12], results['can_bus'][4:6])
                            dists = np.sqrt(
                                np.sum(
                                    (traj1 - self.plan_anchor[results["command"]]) ** 2,
                                    axis=-1,
                                )
                            ).sum(-1)
                            topk_indices = np.argsort(dists)[: self.rag_topk]
                            topk_trajs = self.plan_anchor[results["command"]][
                                topk_indices
                            ]
                        if self.cat_pred_traj:
                            trajs = np.concatenate([traj1[None], topk_trajs], axis=0)
                        else:
                            trajs = topk_trajs
                        results["pred_traj2"] = trajs
                    elif self.use_kmeans_traj:
                        trajs = self.plan_anchor[results["command"]]
                        # import pdb; pdb.set_trace()
                        if self.kmeans_pad_traj:
                            if self.use_ego_mlp:
                                traj1_lidar = self.ego_mlp[results["sample_idx"]][
                                    "final_planning"
                                ].numpy()
                                traj1 = copy.deepcopy(traj1_lidar)
                                traj1[:, 1] = -traj1_lidar[:, 0]
                                traj1[:, 0] = traj1_lidar[:, 1]
                            else:
                                traj_path1 = os.path.join(
                                    self.baseline_path, results["sample_idx"]
                                )
                                traj1 = self.trans_json_to_traj(traj_path1)
                            trajs = np.concatenate([traj1[None], trajs], axis=0)
                        results["pred_traj2"] = trajs

                    traj_tokens = " ".join(
                        [f"<G{i}> {DEFAULT_POINT_TOKEN}" for i in range(trajs.shape[0])]
                    ).strip()

                    if self.use_text_point:
                        endpoints = trajs[:, -1, :]
                        text_points = ", ".join(
                            [
                                f"<G{i}> ({pt[0]:.2f},{pt[1]:.2f})"
                                for i, pt in enumerate(endpoints)
                            ]
                        )

                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            [
                                {
                                    "from": "human",
                                    # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                    #         "Please select the best trajectory in the current scenario."},
                                    "value": f"Here are predefined trajectories with endpoints of future 3 seconds [{text_points}] for the ego car. "
                                    + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                                },
                                {"from": "gpt", "value": ""},
                            ]
                        ]

                    elif self.add_vel:
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot1 = [
                            {
                                "from": "human",
                                # "value": f"Here are predefined trajectories [{text_points}] for the ego car. " +
                                #         "Please select the best trajectory in the current scenario."},
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + f"Please select the best trajectory in the scenario with current speed: {speed_str} m/s and acceleration: {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    elif self.add_ego:
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario with ego stage <ego>.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    else:
                        prompt_cot1 = [
                            {
                                "from": "human",
                                "value": f"Here are predefined trajectories [{traj_tokens}] for the ego car. "
                                + "Please select the best trajectory in the current scenario.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    prompt_refine = [
                        {
                            "from": "human",
                            "value": "How to optimize this selected trajectory?",
                        },
                        {"from": "gpt", "value": ""},
                    ]
                    if self.cot_with_speed:
                        current_speed = results["can_bus"][-3:-1]
                        current_accel = results["can_bus"][4:6]
                        speed_str = f"({current_speed[0]:.2f},{current_speed[1]:.2f})"
                        accel_str = f"({current_accel[0]:.2f},{current_accel[1]:.2f})"
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + f"please provide the future 6~9s planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                        prompt_cot3 = [
                            {
                                "from": "human",
                                "value": f"Please provide the future 0~9s planning trajectory for the ego car, which has a velocity of {speed_str} m/s and an acceleration of {accel_str} m/s^2.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    else:
                        prompt_cot2 = [
                            {
                                "from": "human",
                                "value": "With the selected trajectory as a reference %s, "
                                % DEFAULT_TRAJ_TOKEN
                                + "please provide the planning trajectory for the ego car.",
                            },
                            {"from": "gpt", "value": ""},
                        ]

                    sources.append(prompt_cot1)

                    if self.use_refine_step:
                        sources.append(prompt_refine)

                    sources.append(prompt_cot2)
                    if not self.only_eval_69:
                        sources.append(prompt_cot3)

                else:
                    sources.append(
                        [
                            {
                                "from": "human",
                                "value": "Please provide the planning trajectory for the ego car without reasons.",
                            },
                            {"from": "gpt", "value": ""},
                        ]
                    )

        return sources

    def extend_traj(self, velocity, accel, dt=0.5, num_points=6):
        # Use velocity and acceleration from can_bus to generate trajectory
        pad_planning_traj = np.zeros((num_points, 2))
        # Update velocity using acceleration for first timestep only
        velocity[0] = velocity[0] + accel[0] * dt
        # Use constant velocity for all points
        for i in range(num_points):
            t = dt * (i + 1)
            pad_planning_traj[i, 0] = velocity[0] * t
            pad_planning_traj[i, 1] = velocity[1] * t

        return pad_planning_traj

    def trans_json_to_traj(self, traj_path):
        with open(traj_path, "r") as f:
            traj = json.load(f)
        traj = traj[-1]["A"][0]
        full_match = re.search(
            r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
            traj,
        )
        if full_match:
            coordinates_matches = re.findall(
                r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
            )
            coordinates = [
                tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                for coord in coordinates_matches
            ]
            coordinates_array = np.array(coordinates)
        return coordinates_array

    def __call__(self, results):

        if self.pred_res_traj is not None:
            if os.path.exists(self.pred_res_traj + results["sample_idx"]):
                with open(
                    self.pred_res_traj + results["sample_idx"], "r", encoding="utf8"
                ) as f:
                    pred_data = json.load(f)
                    traj = pred_data[-1]["A"][0]
                    full_match = re.search(
                        r"\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]",
                        traj,
                    )
                    if full_match:
                        coordinates_matches = re.findall(
                            r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0)
                        )
                        coordinates = [
                            tuple(map(float, re.findall(r"-?\d+\.\d+", coord)))
                            for coord in coordinates_matches
                        ]
                        coordinates_array = np.array(coordinates)
                        pred_traj = coordinates_array

                formatted_points = ", ".join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
                    for point in pred_traj
                )
                pred_traj_str = f" Here is the initial trajectory [{formatted_points}]."
            else:
                pred_traj_str = None
        else:
            pred_traj_str = None

        sources = self.preprocess_vqa(results, pred_traj_str)
        prompt = f"You are driving in {results['location']}. "

        for anno in sources[:1]:
            anno[0]["value"] = DEFAULT_IMAGE_TOKEN + "\n" + prompt + anno[0]["value"]
            anno[1]["value"] = ""

        has_traj = self.use_trajemb_cot or self.use_pred_traj_seq
        vqa_converted = preprocess(
            sources,
            self.tokenizer,
            True,
            False,
            has_traj=has_traj,
            use_qwen=self.use_qwen,
            use_qwenvl_25=self.use_qwenvl_25,
        )
        input_ids = vqa_converted["input_ids"]
        results["input_ids"] = input_ids
        vlm_labels = [anno[0]["value"] for anno in sources]
        results["vlm_labels"] = vlm_labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage:
    def __init__(
        self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True
    ):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf["rot_lim"] == (0.0, 0.0), (
            "Rotation is not currently supported"
        )

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d:  # sync_2d bbox labels
                gt_bboxes = results["gt_bboxes"][i]
                centers2d = results["centers2d"][i]
                gt_labels = results["gt_labels"][i]
                depths = results["depths"][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes,
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
                        gt_bboxes, centers2d, gt_labels, depths
                    )

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results["intrinsics"][i][:3, :3] = (
                ida_mat @ results["intrinsics"][i][:3, :3]
            )
        results["gt_bboxes"] = new_gt_bboxes
        results["centers2d"] = new_centers2d
        results["gt_labels"] = new_gt_labels
        results["depths"] = new_depths
        results["img"] = new_imgs
        results["lidar2img"] = [
            results["intrinsics"][i] @ results["extrinsics"][i]
            for i in range(len(results["extrinsics"]))
        ]

        return results

    def _bboxes_transform(
        self, bboxes, centers2d, gt_labels, depths, resize, crop, flip
    ):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & (
            (bboxes[:, 3] - bboxes[:, 1]) >= self.min_size
        )

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind="stable")
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage:
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results["ego_pose"] = (
            torch.tensor(results["ego_pose"]).float() @ trans_mat_inv
        ).numpy()
        results["ego_pose_inv"] = (
            trans_mat.float() @ torch.tensor(results["ego_pose_inv"])
        ).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (
                torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv
            ).numpy()

    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor(
            [
                [rot_cos, rot_sin, 0, 0],
                [-rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = torch.inverse(rot_mat)

        results["ego_pose"] = (
            torch.tensor(results["ego_pose"]).float() @ rot_mat_inv
        ).numpy()
        results["ego_pose_inv"] = (
            rot_mat.float() @ torch.tensor(results["ego_pose_inv"])
        ).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv
            ).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results["ego_pose"] = (
            torch.tensor(results["ego_pose"]).float() @ scale_mat_inv
        ).numpy()
        results["ego_pose_inv"] = (
            scale_mat @ torch.tensor(results["ego_pose_inv"]).float()
        ).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv
            ).numpy()


@PIPELINES.register_module()
class CustomPadMultiViewImage:
    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results["img"]])
        max_w = max([img.shape[1] for img in results["img"]])
        padded_img = [
            mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val)
            for img in results["img"]
        ]
        if self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in padded_img
            ]

        results["img"] = padded_img
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = None
        results["pad_size_divisor"] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class CustomParameterizeLane:
    def __init__(self, method, n_control):
        self.method = method
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results["ann_info"]["lane_pts"]
        para_centerlines = getattr(self, self.method)(centerlines, self.n_control)
        results["lane_pts"] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = (
                    self.comb(n_control - 1, j)
                    * np.power(1 - t[i], n_control - 1 - j)
                    * np.power(t[i], j)
                )
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = (
                    self.comb(n_control - 1, j)
                    * np.power(1 - t[i], n_control - 1 - j)
                    * np.power(t[i], j)
                )
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = (
            _points
            - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1)
            - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)
        )

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier_Endpointfixed(self, input_data, n_control=4):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99.

    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if np.random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
