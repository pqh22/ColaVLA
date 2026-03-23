import os
import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import OptimizerHook
from torch.cuda.amp import autocast
import copy
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
    LossScaler,
    allreduce_grads,
)
from typing import Optional, Union
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch import Tensor
import logging
from collections import Counter, defaultdict
from itertools import chain

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.

    from torch.cuda.amp import GradScaler

except ImportError:
    pass


@HOOKS.register_module()
class CustomOptimizerHook(OptimizerHook):  # type: ignore
    """FP16 optimizer hook (mmcv's implementation).

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of LossScaler.
            Defaults to 512.
    """

    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        # runner.optimizer.param_groups = copy.deepcopy(
        #         runner.optimizer.param_groups.to(torch.bfloat16))
        runner.model = runner.model.to(torch.bfloat16)

    # def after_train_iter(self, runner):
    #     runner.optimizer.zero_grad()
    #     if self.detect_anomalous_params:
    #         self.detect_anomalous_parameters(runner.outputs['loss'], runner)
    #     runner.outputs['loss'].backward()

    #     if self.grad_clip is not None:
    #         grad_norm = self.clip_grads(runner.model.parameters())
    #         if grad_norm is not None:
    #             # Add grad norm to the logger
    #             runner.log_buffer.update({'grad_norm': float(grad_norm)},
    #                                      runner.outputs['num_samples'])
    #     runner.optimizer.step()


@HOOKS.register_module()
class Fp16OptimizerHookDebug(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(
        self,
        grad_clip: Optional[dict] = None,
        coalesce: bool = True,
        bucket_size_mb: int = -1,
        loss_scale: Union[float, str, dict] = 512.0,
        distributed: bool = True,
    ):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == "dynamic":
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError(
                "loss_scale must be of type float, dict, or "
                f'"dynamic", got {loss_scale}'
            )

        self._named_params_cache = None
        self._iter_cnt = 0
        self._debug_enabled = self._env_bool("GRAD_DEBUG", False)
        self._debug_log_every_n = max(1, self._env_int("GRAD_DEBUG_LOG_EVERY", 1))
        self._debug_log_topk = max(1, self._env_int("GRAD_DEBUG_TOPK", 200))
        self._debug_module_topk = max(1, self._env_int("GRAD_DEBUG_MODULE_TOPK", 50))
        self._debug_only_when_unused = self._env_bool(
            "GRAD_DEBUG_ONLY_WHEN_UNUSED", True
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value in ("1", "true", "TRUE", "yes", "YES")

    @staticmethod
    def _param_to_module_name(param_name: str) -> str:
        if "." not in param_name:
            return "<root>"
        return param_name.rsplit(".", 1)[0]

    @staticmethod
    def _log_warning(runner, message: str) -> None:
        logger = getattr(runner, "logger", None)
        if logger is not None:
            logger.warning(message)
        else:
            print(message)

    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if "fp16" in runner.meta and "loss_scaler" in runner.meta["fp16"]:
            scaler_state_dict = runner.meta["fp16"]["loss_scaler"]
            self.loss_scaler.load_state_dict(scaler_state_dict)

        m = runner.model.module if hasattr(runner.model, "module") else runner.model
        # 只缓存需要梯度的参数，减少遍历量
        self._named_params_cache = [
            (n, p) for n, p in m.named_parameters() if p.requires_grad
        ]

    def copy_grads_to_fp32(self, fp16_net: nn.Module, fp32_weights: Tensor) -> None:
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net: nn.Module, fp32_weights: Tensor) -> None:
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner) -> None:
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs["loss"]).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        if (
            self._debug_enabled
            and getattr(self, "_named_params_cache", None) is not None
        ):
            self._iter_cnt += 1
            if self._iter_cnt % self._debug_log_every_n == 0:
                unused = [
                    name for name, p in self._named_params_cache if p.grad is None
                ]
                if unused or (not self._debug_only_when_unused):
                    module_counter = Counter(
                        self._param_to_module_name(name) for name in unused
                    )
                    total_params = len(self._named_params_cache)
                    try:
                        import torch.distributed as dist

                        rank = (
                            dist.get_rank()
                            if dist.is_available() and dist.is_initialized()
                            else -1
                        )
                    except Exception:
                        rank = -1
                    self._log_warning(
                        runner,
                        f"[GradDebug][Rank {rank}] iter={self._iter_cnt} params_without_grad={len(unused)}/{total_params}",
                    )
                    for module_name, cnt in module_counter.most_common(
                        self._debug_module_topk
                    ):
                        self._log_warning(
                            runner,
                            f"[GradDebug][Rank {rank}] module={module_name} count={cnt}",
                        )
                    for param_name in unused[: self._debug_log_topk]:
                        self._log_warning(
                            runner, f"[GradDebug][Rank {rank}] param={param_name}"
                        )
                    if len(unused) > self._debug_log_topk:
                        self._log_warning(
                            runner,
                            f"[GradDebug][Rank {rank}] ... and {len(unused) - self._debug_log_topk} more params",
                        )

        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(
                    {"grad_norm": float(grad_norm)}, runner.outputs["num_samples"]
                )
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault("fp16", {})["loss_scaler"] = (
            self.loss_scaler.state_dict()
        )


# class Fp16OptimizerHookDebug(OptimizerHook):  # type: ignore
#     """FP16 optimizer hook (mmcv's implementation).

#     The steps of fp16 optimizer is as follows.
#     1. Scale the loss value.
#     2. BP in the fp16 model.
#     2. Copy gradients from fp16 model to fp32 weights.
#     3. Update fp32 weights.
#     4. Copy updated parameters from fp32 weights to fp16 model.

#     Refer to https://arxiv.org/abs/1710.03740 for more details.

#     Args:
#         loss_scale (float | str | dict): Scale factor configuration.
#             If loss_scale is a float, static loss scaling will be used with
#             the specified scale. If loss_scale is a string, it must be
#             'dynamic', then dynamic loss scaling will be used.
#             It can also be a dict containing arguments of LossScaler.
#             Defaults to 512.
#     """

#     def __init__(self,
#                     grad_clip: Optional[dict] = None,
#                     coalesce: bool = True,
#                     bucket_size_mb: int = -1,
#                     loss_scale: Union[float, str, dict] = 512.,
#                     distributed: bool = True):
#         self.grad_clip = grad_clip
#         self.coalesce = coalesce
#         self.bucket_size_mb = bucket_size_mb
#         self.distributed = distributed
#         if loss_scale == 'dynamic':
#             self.loss_scaler = LossScaler(mode='dynamic')
#         elif isinstance(loss_scale, float):
#             self.loss_scaler = LossScaler(
#                 init_scale=loss_scale, mode='static')
#         elif isinstance(loss_scale, dict):
#             self.loss_scaler = LossScaler(**loss_scale)
#         else:
#             raise ValueError('loss_scale must be of type float, dict, or '
#                                 f'"dynamic", got {loss_scale}')

#     def before_run(self, runner) -> None:
#         """Preparing steps before Mixed Precision Training.

#         1. Make a master copy of fp32 weights for optimization.
#         2. Convert the main model from fp32 to fp16.
#         """
#         # keep a copy of fp32 weights
#         old_groups = runner.optimizer.param_groups
#         runner.optimizer.param_groups = copy.deepcopy(
#             runner.optimizer.param_groups)
#         state: defaultdict = defaultdict(dict)
#         p_map = {
#             old_p: p
#             for old_p, p in zip(
#                 chain(*(g['params'] for g in old_groups)),
#                 chain(*(g['params']
#                         for g in runner.optimizer.param_groups)))
#         }
#         for k, v in runner.optimizer.state.items():
#             state[p_map[k]] = v
#         runner.optimizer.state = state
#         # convert model to fp16
#         wrap_fp16_model(runner.model)
#         # resume from state dict
#         if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
#             scaler_state_dict = runner.meta['fp16']['loss_scaler']
#             self.loss_scaler.load_state_dict(scaler_state_dict)

#     def copy_grads_to_fp32(self, fp16_net: nn.Module,
#                             fp32_weights: Tensor) -> None:
#         """Copy gradients from fp16 model to fp32 weight copy."""
#         for fp32_param, fp16_param in zip(fp32_weights,
#                                             fp16_net.parameters()):
#             if fp16_param.grad is not None:
#                 if fp32_param.grad is None:
#                     fp32_param.grad = fp32_param.data.new(
#                         fp32_param.size())
#                 fp32_param.grad.copy_(fp16_param.grad)

#     def copy_params_to_fp16(self, fp16_net: nn.Module,
#                             fp32_weights: Tensor) -> None:
#         """Copy updated params from fp32 weight copy to fp16 model."""
#         for fp16_param, fp32_param in zip(fp16_net.parameters(),
#                                             fp32_weights):
#             fp16_param.data.copy_(fp32_param.data)

#     def after_train_iter(self, runner) -> None:
#         """Backward optimization steps for Mixed Precision Training. For
#         dynamic loss scaling, please refer `loss_scalar.py`

#         1. Scale the loss by a scale factor.
#         2. Backward the loss to obtain the gradients (fp16).
#         3. Copy gradients from the model to the fp32 weight copy.
#         4. Scale the gradients back and update the fp32 weight copy.
#         5. Copy back the params from fp32 weight copy to the fp16 model.
#         6. Save loss_scaler state_dict for resume purpose.
#         """
#         # clear grads of last iteration
#         runner.model.zero_grad()
#         runner.optimizer.zero_grad()
#         # scale the loss value
#         scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
#         scaled_loss.backward()
#         # copy fp16 grads in the model to fp32 params in the optimizer

#         import os, ipdb
#         if os.getenv("GRAD_DEBUG") == "1":

#             # === DEBUG: 这里 grads 刚产生，还没被清理/覆盖，检测最准确 ===
#             if self._named_params_cache is None:
#                 # 兜底：如果没缓存，现取（一般不会走到这里）
#                 m = runner.model.module if hasattr(runner.model, 'module') else runner.model
#                 self._named_params_cache = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

#             unused = [name for name, p in self._named_params_cache if p.grad is None]
#             need_print = (len(unused) > 0) or (not self.only_when_unused)
#             if need_print and (self._iter_cnt % max(1, self.log_every_n) == 0):
#                 # 分布式 rank
#                 try:
#                     import torch.distributed as dist
#                     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
#                 except Exception:
#                     rank = -1
#                 runner.logger.warning(f"[Rank {rank}] iter={self._iter_cnt} UNUSED params: {len(unused)}")
#                 for n in unused[:self.log_topk]:
#                     runner.logger.warning(f"  {n}")
#                 if len(unused) > self.log_topk:
#                     runner.logger.warning(f"  ... and {len(unused) - self.log_topk} more.")

#         fp32_weights = []
#         for param_group in runner.optimizer.param_groups:
#             fp32_weights += param_group['params']
#         self.copy_grads_to_fp32(runner.model, fp32_weights)
#         # allreduce grads
#         if self.distributed:
#             allreduce_grads(fp32_weights, self.coalesce,
#                             self.bucket_size_mb)

#         has_overflow = self.loss_scaler.has_overflow(fp32_weights)
#         # if has overflow, skip this iteration
#         if not has_overflow:
#             # scale the gradients back
#             for param in fp32_weights:
#                 if param.grad is not None:
#                     param.grad.div_(self.loss_scaler.loss_scale)
#             if self.grad_clip is not None:
#                 grad_norm = self.clip_grads(fp32_weights)
#                 if grad_norm is not None:
#                     # Add grad norm to the logger
#                     runner.log_buffer.update(
#                         {'grad_norm': float(grad_norm)},
#                         runner.outputs['num_samples'])
#             # update fp32 params
#             runner.optimizer.step()
#             # copy fp32 params to the fp16 model
#             self.copy_params_to_fp16(runner.model, fp32_weights)
#         self.loss_scaler.update_scale(has_overflow)
#         if has_overflow:
#             runner.logger.warning('Check overflow, downscale loss scale '
#                                     f'to {self.loss_scaler.cur_scale}')

#         # save state_dict of loss_scaler
#         runner.meta.setdefault(
#             'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


# @HOOKS.register_module()
# class Fp16OptimizerHookDebug(OptimizerHook):  # type: ignore
#     """Fp16 Optimizer Hook 带未参与反传参数的调试打印。

#     与原版流程一致：
#       1) zero_grad
#       2) scaled_loss.backward()
#       3) 从 fp16 模型复制 grad 到 fp32 权重
#       4) allreduce / has_overflow / 缩放回梯度 / grad clip / step
#       5) 拷回 fp16 权重、更新 loss_scale

#     新增：
#       - 在 `scaled_loss.backward()` 之后、`optimizer.step()` 之前，
#         打印本 iter 中 **grad 为 None** 的参数名（过滤掉 requires_grad=False）。
#       - 可配置打印条数与频率。
#     """

#     def __init__(self,
#                  grad_clip: Optional[dict] = None,
#                  coalesce: bool = True,
#                  bucket_size_mb: int = -1,
#                  loss_scale: Union[float, str, dict] = 512.,
#                  distributed: bool = True,
#                  # === Debug 相关 ===
#                  log_topk: int = 50,           # 每次最多打印多少个未用参数名
#                  log_every_n: int = 1,         # 每隔多少个 iter 打印一次
#                  only_when_unused: bool = True # 仅当存在未用参数时才打印
#                  ):
#         super().__init__(grad_clip=grad_clip)
#         self.coalesce = coalesce
#         self.bucket_size_mb = bucket_size_mb
#         self.distributed = distributed

#         # loss scaler
#         if loss_scale == 'dynamic':
#             self.loss_scaler = LossScaler(mode='dynamic')
#         elif isinstance(loss_scale, float):
#             self.loss_scaler = LossScaler(init_scale=loss_scale, mode='static')
#         elif isinstance(loss_scale, dict):
#             self.loss_scaler = LossScaler(**loss_scale)
#         else:
#             raise ValueError(f'Invalid loss_scale: {loss_scale}')

#         # debug
#         self.log_topk = int(log_topk)
#         self.log_every_n = int(log_every_n)
#         self.only_when_unused = bool(only_when_unused)
#         self._iter_cnt = 0
#         self._named_params_cache = None  # (name, param) 缓存，避免每步遍历 named_parameters 开销

#     def before_run(self, runner) -> None:
#         # 1) 深拷贝 optimizer.param_groups，保留 fp32 master 权重
#         old_groups = runner.optimizer.param_groups
#         runner.optimizer.param_groups = copy.deepcopy(runner.optimizer.param_groups)

#         state: defaultdict = defaultdict(dict)
#         p_map = {
#             old_p: p
#             for old_p, p in zip(
#                 chain(*(g['params'] for g in old_groups)),
#                 chain(*(g['params'] for g in runner.optimizer.param_groups)))
#         }
#         for k, v in runner.optimizer.state.items():
#             state[p_map[k]] = v
#         runner.optimizer.state = state

#         # 2) 将模型转 fp16（保持参数对象次序稳定）
#         wrap_fp16_model(runner.model)

#         # 3) 恢复 loss_scaler
#         if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
#             self.loss_scaler.load_state_dict(runner.meta['fp16']['loss_scaler'])

#         # 4) —— 关键：构建 “fp16_param ←→ fp32_param” 的稳定配对表 ——
#         #    解包 DDP，拿到真实模块（避免 DDP 可能带来的次序差异）
#         m = runner.model.module if hasattr(runner.model, 'module') else runner.model

#         fp16_params = list(m.parameters())  # 按模块参数迭代顺序
#         fp32_params = [p for g in runner.optimizer.param_groups for p in g['params']]

#         if len(fp16_params) != len(fp32_params):
#             raise RuntimeError(f"Param count mismatch: fp16={len(fp16_params)} vs fp32={len(fp32_params)}")

#         # 可选：做一次形状核对，若不一致，马上报错定位
#         for i, (p16, p32) in enumerate(zip(fp16_params, fp32_params)):
#             if p16.shape != p32.shape:
#                 raise RuntimeError(
#                     f"Param shape mismatch at idx {i}: fp16 {tuple(p16.shape)} vs fp32 {tuple(p32.shape)}"
#                 )

#         # 保存配对表，后续复制梯度/权重只用这张表，避免“临时 zip”错位
#         self._paired_params = list(zip(fp16_params, fp32_params))

#         # 另外，缓存需要打印名字用的 named_parameters（只做一次）
#         self._named_params_cache = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

#     def copy_grads_to_fp32(self):
#         for p16, p32 in self._paired_params:
#             if p16.grad is not None:
#                 if p32.grad is None:
#                     p32.grad = p32.data.new(p32.size())
#                 p32.grad.copy_(p16.grad)

#     def copy_params_to_fp16(self):
#         for p16, p32 in self._paired_params:
#             p16.data.copy_(p32.data)


#     # ====== 关键：在 step 前做“未用参数”检测 ======
#     def after_train_iter(self, runner) -> None:
#         self._iter_cnt += 1

#         # 1) clear grads
#         runner.model.zero_grad()
#         runner.optimizer.zero_grad()

#         # 2) backward with loss scale
#         scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
#         scaled_loss.backward()

#         # === DEBUG: 这里 grads 刚产生，还没被清理/覆盖，检测最准确 ===
#         if self._named_params_cache is None:
#             # 兜底：如果没缓存，现取（一般不会走到这里）
#             m = runner.model.module if hasattr(runner.model, 'module') else runner.model
#             self._named_params_cache = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

#         unused = [name for name, p in self._named_params_cache if p.grad is None]
#         need_print = (len(unused) > 0) or (not self.only_when_unused)
#         if need_print and (self._iter_cnt % max(1, self.log_every_n) == 0):
#             # 分布式 rank
#             try:
#                 import torch.distributed as dist
#                 rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
#             except Exception:
#                 rank = -1
#             runner.logger.warning(f"[Rank {rank}] iter={self._iter_cnt} UNUSED params: {len(unused)}")
#             for n in unused[:self.log_topk]:
#                 runner.logger.warning(f"  {n}")
#             if len(unused) > self.log_topk:
#                 runner.logger.warning(f"  ... and {len(unused) - self.log_topk} more.")

#         self.copy_grads_to_fp32()  # 用稳定配对表
#         fp32_weights = [p32 for _, p32 in self._paired_params]

#         if self.distributed:
#             allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)

#         # 4) overflow 检测 / 缩放回梯度 / grad clip / step / 拷回 fp16
#         has_overflow = self.loss_scaler.has_overflow(fp32_weights)
#         if not has_overflow:
#             for param in fp32_weights:
#                 if param.grad is not None:
#                     param.grad.div_(self.loss_scaler.loss_scale)
#             if self.grad_clip is not None:
#                 grad_norm = self.clip_grads(fp32_weights)
#                 if grad_norm is not None:
#                     runner.log_buffer.update({'grad_norm': float(grad_norm)},
#                                              runner.outputs['num_samples'])
#             runner.optimizer.step()
#             self.copy_params_to_fp16(runner.model, fp32_weights)

#         self.loss_scaler.update_scale(has_overflow)
#         if has_overflow:
#             runner.logger.warning('Check overflow, downscale loss scale '
#                                   f'to {self.loss_scaler.cur_scale}')

#         # 5) 保存 scaler 状态
#         runner.meta.setdefault('fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()
