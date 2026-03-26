from __future__ import division
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import argparse
import copy
import mmcv

import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
from os import path as osp
import glob
import re

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.utils import TORCH_VERSION, digit_version
import torch.distributed as dist


def adjust_config_for_world_size(cfg):
    """
    world_size dynamic configuration
          - cfg.num_gpus
    - cfg.batch_size
          - cfg.num_workers
          - cfg.num_iters_per_epoch
    """
    if not (dist.is_available() and dist.is_initialized()):
        print(
            "[ADJUST] torch.distributed is not initialized yet, skipping dynamic configuration."
        )
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # default print
    old_num_gpus = getattr(cfg, "num_gpus", None)
    old_bs = getattr(cfg, "batch_size", None)
    old_workers = getattr(cfg, "num_workers", None)
    old_iters = getattr(cfg, "num_iters_per_epoch", None)

    # 1) num_gpus
    cfg.num_gpus = world_size

    # 2) process batch_size
    # global batch batch_size
    # batch batch_size target_global // world_size
    # example batch_size=2 (2)
    if not hasattr(cfg, "batch_size") or cfg.batch_size is None:
        cfg.batch_size = 2  # default batch
    # world_size batch **
    # base_single_card = 2
    # cfg.batch_size = base_single_card #
    # Translated note.

    # 3) num_workers
    # workers=2
    if not hasattr(cfg, "num_workers") or cfg.num_workers is None:
        cfg.num_workers = 2
    # config total_workers
    # total_workers = getattr(cfg, 'num_workers', world_size * 2)
    # cfg.num_workers = max(1, total_workers // world_size)

    # 4) num_iters_per_epoch
    # 28130 getattr(cfg, 'train_sample_count', 28130)
    dataset_size = 28130
    per_iter_global_batch = cfg.num_gpus * cfg.batch_size
    if per_iter_global_batch <= 0:
        raise ValueError(f"Invalid per_iter_global_batch={per_iter_global_batch}")
    cfg.num_iters_per_epoch = dataset_size // per_iter_global_batch
    if cfg.num_iters_per_epoch == 0:
        cfg.num_iters_per_epoch = 1  # 1

    # 5) world_size
    # config original lr 8
    # if hasattr(cfg, 'optimizer') and 'lr' in cfg.optimizer:
    #     base_ref_cards = 8
    #     scale = world_size / base_ref_cards
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] * scale

    if rank == 0:
        print("========== [AUTO CONFIG ADJUST] ==========")
        print(f"world_size(real GPUs) : {world_size}")
        print(f"num_gpus      : {old_num_gpus} -> {cfg.num_gpus}")
        print(f"batch_size(per GPU) : {old_bs} -> {cfg.batch_size}")
        print(f"num_workers(per GPU): {old_workers} -> {cfg.num_workers}")
        print(
            f"num_iters_per_epoch: {old_iters} -> {cfg.num_iters_per_epoch} "
            f"(= {dataset_size} // ({cfg.num_gpus} * {cfg.batch_size}))"
        )
        print("==========================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use (only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use (only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "luban"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def init_dist_luban(backend="nccl"):
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = (
        os.environ.get("DISTRIBUTED_MASTER_HOSTS") or os.environ["MASTER_ADDR"]
    )
    port = os.environ["LUBAN_AVAILBLE_PORT_0"]
    dist_url = f"tcp://{master_addr}:{port}"

    num_gpus = torch.cuda.device_count()

    # ------- fallback -------
    if "DISTRIBUTED_NODE_RANK" in os.environ:
        machine_rank = int(os.environ["DISTRIBUTED_NODE_RANK"])
        machine_num = int(os.environ["DISTRIBUTED_NODE_COUNT"])
    else:  # torchrun
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        machine_rank = global_rank // num_gpus
        machine_num = world_size // num_gpus
    # ----------------------------------

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        rank=local_rank + machine_rank * num_gpus,
        world_size=num_gpus * machine_num,
    )
    return dist.get_world_size(), dist.get_rank()


def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint file in a directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "iter_*.pth"))
    if not checkpoint_files:
        return None

    # file
    iterations = []
    for checkpoint in checkpoint_files:
        match = re.search(r"iter_(\d+)\.pth", checkpoint)
        if match:
            iterations.append((int(match.group(1)), checkpoint))

    if not iterations:
        return None

    # returncheckpointfile
    return max(iterations, key=lambda x: x[0])[1]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.core.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # if args.resume_from is not None:
    resume_path = args.resume_from or os.path.join(cfg.work_dir, "latest.pth")
    if os.path.exists(resume_path):
        if os.path.isfile(resume_path):
            cfg.resume_from = resume_path
            print(f"Resuming from checkpoint: {resume_path}")
        elif os.path.isdir(resume_path):
            latest_checkpoint = get_latest_checkpoint(resume_path)
            if latest_checkpoint is not None:
                cfg.resume_from = latest_checkpoint
                print(
                    f"Auto-resuming from latest checkpoint in dir: {latest_checkpoint}"
                )
            else:
                print(f"Warning: No checkpoint found in directory: {resume_path}")
                cfg.resume_from = None
    else:
        print(f"No checkpoint found at {resume_path}, starting from scratch.")
        cfg.resume_from = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = [torch.device("cuda", id) for id in args.gpu_ids]
    else:
        cfg.gpu_ids = [
            torch.device("cuda", i) for i in range(torch.cuda.device_count())
        ]
    if (
        digit_version(TORCH_VERSION) == digit_version("1.8.1")
        and cfg.optimizer["type"] == "AdamW"
    ):
        cfg.optimizer["type"] = "AdamW2"  # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        if args.launcher == "luban":
            world_size, _ = init_dist_luban()
        else:
            dist_params = dict(backend="nccl", init_method="env://")
            init_dist(args.launcher, **dist_params)

            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
        cfg.gpu_ids = [torch.device("cuda", i) for i in range(world_size)]

    if dist.is_available() and dist.is_initialized():
        print(
            f"[DEBUG DIST] RANK_ENV={os.getenv('RANK')} "
            f"WORLD_SIZE_ENV={os.getenv('WORLD_SIZE')} "
            f"LOCAL_RANK_ENV={os.getenv('LOCAL_RANK')} "
            f"get_rank()={dist.get_rank()} get_world_size()={dist.get_world_size()}"
        )
    else:
        print("[DEBUG DIST] torch.distributed not initialized yet.")

    # # log for debug, but cannot change origin max-iter
    # adjust_config_for_world_size(cfg)

    # # wandb
    # use_wandb = os.environ.get("WANDB_DISABLED", "false").lower() != "true"
    # rank = 0
    # if dist.is_available() and dist.is_initialized():
    #     rank = dist.get_rank()

    # if use_wandb:
    #     if rank == 0:
    #         init_kwargs = dict(
    #             project=os.environ.get("WANDB_PROJECT", "mmdet3d"),
    #             name=os.environ.get("WANDB_NAME", osp.basename(args.config)),
    #             config=cfg._cfg_dict.to_dict() if hasattr(cfg, "_cfg_dict") else cfg
    #         )
    #         run_id = os.environ.get("WANDB_RUN_ID")
    #         if run_id:
    #             init_kwargs.update(dict(id=run_id, resume="allow"))
    #         wandb.init(**init_kwargs)
    #     else:
    #         os.environ["WANDB_DISABLED"] = "true"

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ["EncoderDecoder3D"]:
        logger_name = "mmseg"
    else:
        logger_name = "mmdet"
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    # if cfg.find_unused_parameters:
    #     def disable_gc(m):
    # # HF/PEFT
    #         if hasattr(m, "gradient_checkpointing_disable"):
    #             m.gradient_checkpointing_disable()
    # # config
    #         if hasattr(m, "config"):
    #             m.config.gradient_checkpointing = False
    # m.config.use_cache = False # cache

    # # build_model init_weights DDP
    #     lm = getattr(model, "lm_head", None)
    #     if lm is None and hasattr(model, "module"):
    #         lm = getattr(model.module, "lm_head", None)
    #     if lm is not None:
    #         disable_gc(lm)
    #         if hasattr(lm, "get_base_model"):
    #             disable_gc(lm.get_base_model())

    # ====== ======
    # 1) / gradient checkpointing
    # def _set_gc_non_reentrant(m):
    #     if hasattr(m, "gradient_checkpointing_enable"):
    #         m.gradient_checkpointing_enable(
    #             gradient_checkpointing_kwargs={"use_reentrant": False}
    #         )
    #     if hasattr(m, "config"):
    # m.config.use_cache = False # HF

    # # lm_head Petr3DTextOnly
    # lm = getattr(model, 'lm_head', None)
    # if lm is None and hasattr(model, 'module'):
    #     lm = getattr(model.module, 'lm_head', None)
    # if lm is not None:
    #     _set_gc_non_reentrant(lm)
    # # PEFT PeftModelForCausalLM base_model
    #     if hasattr(lm, "get_base_model"):
    #         _set_gc_non_reentrant(lm.get_base_model())
    # =====================================

    if cfg.get("SyncBN", False):
        import torch.nn as nn

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Using SyncBN")

    logger.info(f"Model:\n{model}")
    datasets = [build_dataset(cfg.data.train)]

    if not hasattr(cfg, "workflow"):
        cfg.workflow = [("train", 1)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], "PALETTE")
            else None,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # import ipdb; ipdb.set_trace()
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
