import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
from timm.models.layers import Mlp

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)
    
def gen_sineembed_for_position_2d(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """
    xy: (..., 2) where the last dimension is (x, y)
    return: (..., dim)
    """
    assert dim % 2 == 0, "pos dim must be even"
    device = xy.device
    half_dim = dim // 2
    # Use a common log-scale frequency schedule
    freq = torch.arange(half_dim, device=device, dtype=xy.dtype)
    freq = 1.0 / (10000 ** (freq / half_dim))  # (half_dim,)
    # Normalize to a reasonable range (scale can be adjusted if needed)
    x, y = xy[..., 0], xy[..., 1]
    x = x / 10.0
    y = y / 10.0
    sinx = torch.sin(x[..., None] * freq[None, ...])
    cosx = torch.cos(x[..., None] * freq[None, ...])
    siny = torch.sin(y[..., None] * freq[None, ...])
    cosy = torch.cos(y[..., None] * freq[None, ...])
    # Concatenate (sinx, cosx) and (siny, cosy), then slice to `dim`
    pos = torch.cat([sinx, cosx, siny, cosy], dim=-1)  # (..., 4*half_dim) ≥ dim
    if pos.shape[-1] != dim:
        pos = pos[..., :dim]
    return pos


def _build_mlp(in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    d = in_dim
    for i in range(num_layers - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def pairwise_traj_distance(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    a: (B, M, T, 2) predictions/anchors
    b: (B, 1, T, 2) GT (must be unsqueezed to mode dimension externally)
    mask: (B, T) True/1 means valid points
    return: (B, M) weighted average L2 distance from each mode to GT
    """
    diff = (a - b).norm(p=2, dim=-1)  # (B,M,T)
    if mask is not None:
        diff = diff * mask[:, None, :]  # (B,M,T)
        denom = mask[:, None, :].sum(dim=-1).clamp(min=1.0)  # (B,1)
        return diff.sum(dim=-1) / denom  # (B,M)
    else:
        return diff.mean(dim=-1)  # (B,M)

def get_mlp(in_dim, hidden_dim, out_dim, zero_init=False):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim)
    )
    if zero_init:
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
    return mlp

class VisionPruningRouter(nn.Module):
    """
    Use ego token to FiLM-modulate and prune vision tokens, returning key queries.

    Args:
        dim (int): feature dimension D
        hidden_dim (int): hidden dimension of the gamma/beta MLP
        keep_k (Optional[int]): number of tokens to keep; mutually exclusive with keep_ratio
        keep_ratio (Optional[float]): keep ratio (0,1]
        use_soft_weights (bool): whether to return continuous weights during forward for training
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        keep_k: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        use_soft_weights: bool = True
    ):
        super().__init__()
        assert (keep_k is None) ^ (keep_ratio is None), "keep_k and keep_ratio must be mutually exclusive"

        self.keep_k = keep_k
        self.keep_ratio = keep_ratio
        self.use_soft_weights = use_soft_weights

        # FiLM parameters
        self.scale_mlp = get_mlp(dim, hidden_dim, dim)
        self.shift_mlp = get_mlp(dim, hidden_dim, dim)

        # Router head: score for each token
        self.router_head = nn.Linear(dim, 1)

    def forward(
        self,
        ego_embeds: torch.Tensor,     # [B, 1, D]
        vision_embeds: torch.Tensor   # [B, Q_in, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, D = vision_embeds.shape
        assert ego_embeds.shape == (B, 1, D), "ego_embeds shape should be [B,1,D]"

        # 1. FiLM modulation
        ego = ego_embeds.squeeze(1)           # [B, D]
        gamma = self.scale_mlp(ego).unsqueeze(1)  # [B,1,D]
        beta  = self.shift_mlp(ego).unsqueeze(1)  # [B,1,D]
        vision_film = vision_embeds * (1 + gamma) + beta  # [B,Q,D]

        # 2. Compute routing weights
        logits = self.router_head(vision_film).squeeze(-1)  # [B, Q]
        weights = torch.sigmoid(logits) if self.use_soft_weights else None

        # 3. Select tokens to keep
        if self.keep_k is not None:
            k = min(self.keep_k, Q)
        else:
            k = max(1, int(self.keep_ratio * Q))

        # top-k indices
        topk_scores, topk_idx = torch.topk(logits, k=k, dim=1)  # [B,k]
        # Gather
        batch_idx = torch.arange(B, device=vision_embeds.device).unsqueeze(-1)
        pruned_embeds = vision_embeds[batch_idx, topk_idx]       # [B,k,D]
        selected_weights = topk_scores if not self.use_soft_weights else weights[batch_idx, topk_idx]

        return pruned_embeds, selected_weights, topk_idx
    
class VisionPruningRouterGumbel(nn.Module):
    """
    FiLM + Gumbel-Top-k differentiable pruning router
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        keep_k: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        tau: float = 1.0,                  # Gumbel-Softmax temperature
        tau_init: float = 2.0,
        tau_min: float = 0.5,
        tau_decay: float = 0.9997,
        diversity_weight: float = 0.0,
        use_anneal_tau: bool = False,
    ):
        super().__init__()
        assert (keep_k is None) ^ (keep_ratio is None), "keep_k and keep_ratio must be mutually exclusive"

        self.keep_k      = keep_k
        self.keep_ratio  = keep_ratio

        if use_anneal_tau:
            self.tau         = tau_init # initial temperature
        else:
            self.tau         = tau # fixed temperature

        self.tau_min     = tau_min # minimum temperature
        self.tau_decay   = tau_decay # temperature decay rate
        self.diversity_weight = diversity_weight
        self.use_anneal_tau = use_anneal_tau
        self.scale_mlp   = get_mlp(dim, hidden_dim, dim)
        self.shift_mlp   = get_mlp(dim, hidden_dim, dim)
        self.router_head = nn.Linear(dim, 1)

    def _anneal_tau(self):
        # Automatically anneal on each forward pass
        if self.training and self.tau > self.tau_min:
            self.tau = max(self.tau * self.tau_decay, self.tau_min)

    def _sample_gumbel(self, shape, device):
        # Gumbel(0,1) = -log(-log(U)), U~Uniform(0,1)
        return -torch.empty(shape, device=device).exponential_().log()

    def forward(
        self,
        ego_embeds: torch.Tensor,    # [B,1,D]
        vision_embeds: torch.Tensor  # [B,Q,D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, D = vision_embeds.shape
        assert ego_embeds.shape == (B, 1, D)

        if self.use_anneal_tau:
            self._anneal_tau()

        # ---------- 1. FiLM modulation ----------
        ego   = ego_embeds.squeeze(1)              # [B,D]
        gamma = self.scale_mlp(ego).unsqueeze(1)   # [B,1,D]
        beta  = self.shift_mlp(ego).unsqueeze(1)   # [B,1,D]
        vision_film = vision_embeds * (1 + gamma) + beta  # [B,Q,D]

        # ---------- 2. logits ----------
        logits = self.router_head(vision_film).squeeze(-1)  # [B,Q]

        # ---------- 3. Build mask ----------
        k = self.keep_k if self.keep_k is not None else max(1, int(self.keep_ratio * Q))

        if self.training:
            # Gumbel-Top-k (STE)
            gumbel   = self._sample_gumbel(logits.size(), logits.device)
            y        = (logits + gumbel) / self.tau           # [B,Q]
            w_soft   = F.softmax(y, dim=-1)                   # continuous probabilities
            _, idx   = w_soft.topk(k, dim=1)                  # [B,k]
            hard_mask = torch.zeros_like(w_soft).scatter_(1, idx, 1.0)  # one-hot
            mask      = hard_mask + w_soft - w_soft.detach()  # STE
        else:
            # Inference: deterministic top-k
            _, idx   = logits.topk(k, dim=1)
            mask     = torch.zeros_like(logits).scatter_(1, idx, 1.0)

        # ---------- 4. Export results ----------
        batch_idx      = torch.arange(B, device=vision_embeds.device).unsqueeze(-1)
        topk_idx       = idx                                           # [B,k]
        selected_mask  = mask[batch_idx, topk_idx]                     # [B,k]
        pruned_embeds  = vision_embeds[batch_idx, topk_idx] * selected_mask.unsqueeze(-1)  # [B,k,D]

        # Diversity: encourage selected tokens to be dissimilar (reduce cosine similarity)
        if self.training:
            K = self.keep_k
            if self.diversity_weight > 0.0 and K > 1:
                # Compute cosine similarity after normalization
                normed = F.normalize(pruned_embeds, p=2, dim=-1)         # (B, K, D)
                sim = torch.einsum('bkd,bqd->bkq', normed, normed)  # (B, K, K)
                # Remove diagonal and average
                off_diag = sim - torch.eye(K, device=sim.device).unsqueeze(0) # (B, K, K)
                loss_diversity = self.diversity_weight * off_diag.pow(2).mean() # (1,)
            else:
                loss_diversity = None
        else:
            loss_diversity = None

        return pruned_embeds, selected_mask, topk_idx, loss_diversity


class ImprovedTokenRouter(nn.Module):
    """
    Ego-conditioned L0/Hard-Concrete gating + top-k visual token router.

    Returns:
        pruned_tokens:   (B, K, D) selected K visual tokens
        selected_weights:(B, K) corresponding normalized gate weights (for downstream weighting/visualization)
        topk_idx:        (B, K) indices in the original Q sequence

    Extra:
        self.last_aux: {
            'loss_sparsity': tensor,       # sparsity regularization (weighted into total loss)
            'loss_diversity': tensor,      # diversity regularization (optional)
            'gates_mean': float,           # monitoring
            'k_used': int,                 # actual K used
            'valid_frac': float            # fraction of valid tokens (if mask provided)
        }
    """

    def __init__(
        self,
        d: int,
        keep_k: int,
        init_log_alpha: float = -2.0,
        temp: float = 2.0/3.0,
        sparsity_weight: float = 1e-4,
        diversity_weight: float = 0.01,
        normalize_scores: bool = True,
        gumbel_scale: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.keep_k = keep_k
        self.temp = temp
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.normalize_scores = normalize_scores
        self.gumbel_scale = gumbel_scale
        self.eps = eps

        # ego -> FiLM
        self.scale_mlp = get_mlp(d, max(4, d // 2), d)
        self.shift_mlp = get_mlp(d, max(4, d // 2), d)

        # Scorer
        self.scorer = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        # Learnable bias for Hard-Concrete
        self.log_alpha = nn.Parameter(torch.full((1,), init_log_alpha))

    @staticmethod
    def _sample_gumbel_like(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        u = torch.rand_like(x)
        return -torch.log(-torch.log(u + eps) + eps)

    def hard_concrete(self, logit: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Hard-Concrete gate from Louizos et al., 2018.
        Returns continuous gate values in [0,1].
        """
        if training:
            u = torch.rand_like(logit)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logit) / self.temp)
        else:
            s = torch.sigmoid(logit)  # deterministic behavior in evaluation
        gamma, zeta = -0.1, 1.1
        s = s * (zeta - gamma) + gamma
        return s.clamp(0.0, 1.0)

    def forward(
        self,
        can_bus_emb: torch.Tensor,      # (B, 1, D)
        vision_tokens: torch.Tensor,    # (B, Q, D)
        mask: torch.Tensor = None,      # (B, Q), 1 valid / 0 invalid; optional
        k_override: int = None,         # temporary keep_k override
        keep_order: bool = False,       # True: restore top-k to ascending input-index order
        return_aux: bool = False,       # True: additionally return aux dict (also stored in self.last_aux)
    ):
        B, Q, D = vision_tokens.shape
        K = int(self.keep_k if k_override is None else k_override)
        K = max(1, min(K, Q))
        # === 1) ego -> FiLM modulation ===
        ego = can_bus_emb.squeeze(1)        # (B, D)
        gamma = self.scale_mlp(ego).unsqueeze(1)  # (B, 1, D)
        beta  = self.shift_mlp(ego).unsqueeze(1)  # (B, 1, D)
        vision_film = vision_tokens * (1.0 + gamma) + beta  # (B, Q, D)

        # === 2) scoring + (optional) normalization + (optional) Gumbel noise ===
        scores = self.scorer(vision_film).squeeze(-1)  # (B, Q)
        if self.normalize_scores:
            mu = scores.mean(dim=1, keepdim=True)
            sigma = scores.std(dim=1, keepdim=True) + self.eps
            scores = (scores - mu) / sigma

        if self.training and self.gumbel_scale > 0.0:
            scores = scores + self.gumbel_scale * self._sample_gumbel_like(scores, self.eps)

        # === 3) Hard-Concrete gate ===
        gates = self.hard_concrete(scores + self.log_alpha, self.training)  # (B, Q)

        # === 4) mask support (force invalid positions to be unselected) ===
        if mask is not None:
            # gates are used for regularization/visualization; keep 0/1 (multiply by mask)
            gates = gates * mask
            # for top-k scoring, set invalid positions to -inf to prevent selection
            gates_for_topk = torch.where(mask.bool(), gates, torch.full_like(gates, -1e9))
            valid_frac = (mask.sum().float() / (mask.numel() + self.eps)).item()
        else:
            gates_for_topk = gates
            valid_frac = 1.0

        # === 5) hard top-k selection ===
        topv, topi = gates_for_topk.topk(K, dim=1)          # (B, K)
        # optional: restore ascending order by original indices
        if keep_order:
            sorted_idx = torch.sort(topi, dim=1).indices     # (B, K)
            topi = torch.gather(topi, 1, sorted_idx)
            topv = torch.gather(topv, 1, sorted_idx)

        pruned = torch.gather(
            vision_film, 1, topi.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, K, D)

        # selected weights (normalized gate weights for downstream use/visualization)
        selected_weights = topv / (topv.sum(dim=1, keepdim=True) + self.eps)  # (B, K)

        # # === 6) regularization terms ===
        # # sparsity: encourage smaller gates (overall sparser)
        # loss_sparsity = self.sparsity_weight * gates.mean()

        # diversity: encourage selected tokens to be dissimilar (reduce cosine similarity)
        if self.diversity_weight > 0.0 and K > 1:
            # cosine similarity after normalization
            normed = F.normalize(pruned, p=2, dim=-1)         # (B, K, D)
            sim = torch.einsum('bkd,bqd->bkq', normed, normed)  # (B, K, K)
            # remove diagonal and average
            off_diag = sim - torch.eye(K, device=sim.device).unsqueeze(0) # (B, K, K)
            loss_diversity = self.diversity_weight * off_diag.pow(2).mean() # (B,)
        else:
            loss_diversity = gates.new_zeros(())

        # # record aux
        # self.last_aux = dict(
        #     # loss_sparsity=loss_sparsity,
        #     loss_diversity=loss_diversity,
        #     gates_mean=float(gates.mean().detach().cpu()),
        #     k_used=K,
        #     valid_frac=valid_frac,
        # )

        return pruned, selected_weights, topi, loss_diversity


class FiLMedTokenRouter(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.router = nn.Linear(d_in, 2)

        self.scale = get_mlp(d_in, d_in // 2, d_in)
        self.shift = get_mlp(d_in, d_in // 2, d_in)

        # self.scale = nn.Linear(config.hidden_size, config.hidden_size)
        # self.shift = nn.Linear(config.hidden_size, config.hidden_size)
        # nn.init.zeros_(self.router.weight)
        # nn.init.zeros_(self.router.bias)

    def forward(self, x, ego_embeded: torch.Tensor):

        # FiLM projector
        gamma = self.scale(ego_embeded)  # [bs, dim]
        beta = self.shift(ego_embeded)  # [bs, dim]

        # Apply FilM to vision (non-replace)
        filmed_vision = x[:, 1: , :] * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])
        return filmed_vision
    



def sine_pos2d(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """
    xy: (...,2)  -> (..., dim)
    """
    assert dim % 4 == 0, "pos dim must be multiple of 4"
    device, dtype = xy.device, xy.dtype
    half = dim // 2
    freq = torch.arange(half, device=device, dtype=dtype)
    freq = 1.0 / (10000 ** (freq / half))
    x, y = xy[..., 0], xy[..., 1]
    sinx, cosx = torch.sin(x[..., None] * freq), torch.cos(x[..., None] * freq)
    siny, cosy = torch.sin(y[..., None] * freq), torch.cos(y[..., None] * freq)
    pos = torch.cat([sinx, cosx, siny, cosy], dim=-1)
    return pos

def sinusoid_tpe(T: int, dim: int, device, dtype) -> torch.Tensor:
    assert dim % 2 == 0
    pe = torch.zeros(T, dim, device=device, dtype=dtype)
    pos = torch.arange(0, T, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (T,dim)

def pos2d_embed(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """sin/cos 2D positional encoding -> dim (used for historical point memory encoding)"""
    assert dim % 4 == 0
    half = dim // 2
    freq = torch.arange(half, device=xy.device, dtype=xy.dtype)
    freq = 1.0 / (10000 ** (freq / half))
    x, y = xy[..., 0], xy[..., 1]
    sinx, cosx = torch.sin(x[..., None] * freq), torch.cos(x[..., None] * freq)
    siny, cosy = torch.sin(y[..., None] * freq), torch.cos(y[..., None] * freq)
    pe = torch.cat([sinx, cosx, siny, cosy], dim=-1)
    return pe[..., :dim]

def mlp(in_dim, hid, out_dim, layers=2, dropout=0.0) -> nn.Sequential:
    seq, d = [], in_dim
    for _ in range(layers - 1):
        seq += [nn.Linear(d, hid), nn.ReLU(inplace=True)]
        if dropout > 0: seq += [nn.Dropout(dropout)]
        d = hid
    seq += [nn.Linear(d, out_dim)]
    return nn.Sequential(*seq)


class MSARActionHeadv4(nn.Module):
    """
    V4: unified parallel decoding; both training and testing use forward_queries_parallel_v2
    
    Key improvements:
    1. Remove dependence on GT/predicted-point embeddings
    2. Use parallel decoding for both training and testing (no forward_train/forward_test split)
    3. Training: all stage outputs predict trajectories with multi-scale supervision
    4. Testing: use only the last stage output to predict the final trajectory
    5. Support normalized training mode (norm_traj_mode=True): compute loss in normalized space and monitor in original space
    """
    def __init__(
        self,
        fut_ts: int = 6,
        d_in: int = 4096,
        stage_indices: Optional[List[List[int]]] = None,
        ms_traj_loss_weights: Optional[List[float]] = None,
        use_gradient_checkpointing: bool = True,
        topk_mode_predict: int = -1,
    ):
        super().__init__()
        self.fut_ts = fut_ts
        self.d_in = d_in
        self.norm_traj_mode = False
        self.stage_indices = stage_indices or [[5], [0, 5], [0, 2, 5], [0, 2, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 4, 5]]
        self.num_stages = len(self.stage_indices)
        self.topk_mode_predict = topk_mode_predict
        # Normalization (along d_in dimension)
        self.norm_traj = nn.LayerNorm(d_in)

        # Multi-scale loss weights
        # if ms_traj_loss_weights is None:
        #     ms_traj_loss_weights = [1.0] * self.num_stages
        if ms_traj_loss_weights is None:
            self.ms_traj_loss_weights = [0.5, 0.7, 1.0, 1.2, 1.5, 1.8][:self.num_stages]
        else:
            assert len(ms_traj_loss_weights) == self.num_stages
            self.ms_traj_loss_weights = ms_traj_loss_weights
    
        # Output head (d_in -> x, y)
        if self.topk_mode_predict != -1:
            self.pred_scores_head = get_mlp(d_in, d_in // 2, 1)

        self.head = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_in // 2, 2)
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Trajectory normalization statistics (computed from the training set, used for norm_traj_mode)
        # t0-t5 correspond to 0.0s, 0.5s, 1.0s, 1.5s, 2.0s, 2.5s
        # Format: (time_step, 2), where dim=0 is X and dim=1 is Y
        traj_x_mean = torch.tensor([
            2.451756,   # t0
            4.775749,   # t1
            6.968474,   # t2
            9.026832,   # t3
            10.948921,  # t4
            12.733196,  # t5
        ], dtype=torch.float32).view(fut_ts, 1)  # (6, 1)
        
        traj_y_mean = torch.tensor([
            -0.017330,  # t0
            -0.036197,  # t1
            -0.056133,  # t2
            -0.076683,  # t3
            -0.097332,  # t4
            -0.118073,  # t5
        ], dtype=torch.float32).view(fut_ts, 1)  # (6, 1)
        
        traj_x_std = torch.tensor([
            1.830981,   # t0
            3.684343,   # t1
            5.559188,   # t2
            7.446507,   # t3
            9.338465,   # t4
            11.227656,  # t5
        ], dtype=torch.float32).view(fut_ts, 1)  # (6, 1)
        
        traj_y_std = torch.tensor([
            0.051396,   # t0
            0.199697,   # t1
            0.440962,   # t2
            0.765355,   # t3
            1.161688,   # t4
            1.619331,   # t5
        ], dtype=torch.float32).view(fut_ts, 1)  # (6, 1)
        
        # Concatenate into shape (6, 2): each row is [x_mean, y_mean]
        traj_mean = torch.cat([traj_x_mean, traj_y_mean], dim=1)  # (6, 2)
        traj_std = torch.cat([traj_x_std, traj_y_std], dim=1)      # (6, 2)
        
        # Register as buffers (not trainable, but saved/loaded with the model)
        self.register_buffer('traj_mean', traj_mean, persistent=True)
        self.register_buffer('traj_std', traj_std, persistent=True)
    
    def normalize_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory: (traj - mean) / std
        
        Args:
            traj: (B, T, 2) trajectory in original space
        
        Returns:
            norm_traj: (B, T, 2) normalized trajectory
        """
        return (traj - self.traj_mean.unsqueeze(0)) / (self.traj_std.unsqueeze(0) + 1e-6)
    
    def denormalize_traj(self, norm_traj: torch.Tensor) -> torch.Tensor:
        """
        Recover normalized trajectory: traj * std + mean
        
        Args:
            norm_traj: (B, T, 2) normalized trajectory
        
        Returns:
            traj: (B, T, 2) trajectory in original space
        """
        return norm_traj * self.traj_std.unsqueeze(0) + self.traj_mean.unsqueeze(0)
    
    def forward(
        self,
        *,
        traj_feats: torch.Tensor,        # (B, C, D_in)
        context: torch.Tensor,           # (B, Q, D_in) visual context
        can_bus_emb: torch.Tensor,       # (B, 1, D_in)
        query_index: torch.Tensor,       # (B,)
        forward_queries_parallel_v2_fn,  # calls the VLM parallel transformer (new v4 version)
        training: bool = False,
        return_loss: bool = False,
        gt_traj: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        use_grpo: bool = False,
        pred_scores: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        V4 unified forward pass: both training and testing use parallel decoding
        
        Args:
            traj_feats: (B, C, D_in) trajectory features output by VLM
            context: (B, Q, D_in) visual context (map_query)
            can_bus_emb: (B, 1, D_in)
            query_index: (B,) selected category index
            forward_queries_parallel_v2_fn: VLM parallel decoding function
            training: whether in training mode
            return_loss: whether to return loss
            gt_traj: (B, T, 2) GT trajectory
            gt_mask: (B, T) GT mask
            logits: (B, C) classification logits used for top-k mode selection
        
        Returns:
            out: dict containing pred_traj, loss, etc.
        """
        B, C, D = traj_feats.shape
        device = traj_feats.device
        T = self.fut_ts
        out = {}
        # ==================== Top-K Mode ====================
        if self.topk_mode_predict != -1 and logits is not None:
            K = self.topk_mode_predict
            
            # 1. Select top-k indices from logits
            _, topk_indices = torch.topk(logits, k=K, dim=-1)  # (B, K)
            
            # 2. Batch processing: merge B*K samples into one large batch
            # Prepare data: (B*K, ...)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, K).reshape(-1)  # (B*K,)
            topk_indices_flat = topk_indices.reshape(-1)  # (B*K,)
            
            # Select corresponding traj_feats
            cores = traj_feats[batch_indices, topk_indices_flat]  # (B*K, D_in)
            cores_n = self.norm_traj(cores)  # (B*K, D_in)
            
            # Temporal positional encoding
            tpe = sinusoid_tpe(T, self.d_in, device, cores.dtype)  # (T, D_in)
            tgt_all = cores_n.unsqueeze(1) + tpe.unsqueeze(0)  # (B*K, T, D_in)
            tgt_all = F.layer_norm(tgt_all, (self.d_in,))
            
            # Expand context and can_bus_emb
            context_expanded = context.unsqueeze(1).expand(B, K, -1, -1).reshape(B*K, -1, D)  # (B*K, Q, D)
            
            # 3. Call VLM parallel decoding (process all B*K samples in batch)
            final_tgt_output, updated = forward_queries_parallel_v2_fn(
                base_inputs_embeds=context_expanded,  # (B*K, Q, D)
                query_embeds=tgt_all,  # (B*K, T, D)
                stage_indices=self.stage_indices,
                attention_mask=None,
                position_ids=None,
                training=training,
                detach=False,
                return_full_hidden=False,
            )
            
            final_tgt_output = final_tgt_output.float()  # (B*K, Q_final, D)
            
            # 4. Split stage outputs
            stage_len_list = updated["stage_len_list"]
            stage_outputs = []
            offset = 0
            for stage_len in stage_len_list:
                stage_out = final_tgt_output[:, offset:offset+stage_len, :]  # (B*K, n_i, D)
                stage_outputs.append(stage_out)
                offset += stage_len
            
            # 5. Predict trajectories (only the last stage is needed for top-k selection)
            last_stage_output = stage_outputs[-1]  # (B*K, T, D)
            pred_trajs_flat = self.head(last_stage_output)  # (B*K, T, 2)
            pred_trajs = pred_trajs_flat.reshape(B, K, T, 2)  # (B, K, T, 2)
            
            # 6. Compute confidence scores using the last query
            last_query = last_stage_output[:, -1, :].clone()  # (B*K, D)
            scores_flat = self.pred_scores_head(last_query).squeeze(-1)  # (B*K,)
            scores = scores_flat.reshape(B, K)  # (B, K) - trajectory confidence logits
            
            # 7. Training mode: supervise by selecting the trajectory closest to GT
            if training and return_loss and gt_traj is not None:
                if gt_mask is None:
                    gt_mask = torch.ones(B, T, dtype=torch.bool, device=device)
                
                # Normalize GT (if needed)
                gt_traj_orig = gt_traj.clone() if self.norm_traj_mode else None
                if self.norm_traj_mode:
                    gt_traj_normalized = self.normalize_traj(gt_traj)  # (B, T, 2)
                else:
                    gt_traj_normalized = gt_traj
                
                # Compute the distance from each trajectory to GT
                gt_traj_expanded = gt_traj_normalized.unsqueeze(1)  # (B, 1, T, 2)
                distances = torch.norm(pred_trajs - gt_traj_expanded, dim=-1)  # (B, K, T)
                
                # Apply mask
                mask_expanded = gt_mask.unsqueeze(1).float()  # (B, 1, T)
                distances = distances * mask_expanded
                avg_distances = distances.sum(dim=-1) / (mask_expanded.sum(dim=-1) + 1e-6)  # (B, K)
                
                # Select the nearest index
                best_indices = torch.argmin(avg_distances, dim=-1)  # (B,)
                
                # Build confidence labels: nearest is 1, others are 0
                score_labels = torch.zeros(B, K, device=device, dtype=torch.float32)
                score_labels[torch.arange(B), best_indices] = 1.0
                
                # Confidence loss (BCE with logits)
                loss_confidence = F.binary_cross_entropy_with_logits(
                    scores, score_labels, reduction='mean'
                )
                out["loss_confidence"] = loss_confidence
                
                # 8. Apply multi-scale supervision to the best trajectory
                # Select stage_outputs of the best samples
                best_flat_indices = batch_indices[::K] * K + best_indices  # (B,) -> (B,) flat indices
                
                # Collect best-sample predictions for each stage
                best_stage_trajs = []
                for i, stage_out in enumerate(stage_outputs):
                    # stage_out: (B*K, n_i, D)
                    stage_out_reshaped = stage_out.reshape(B, K, -1, D)  # (B, K, n_i, D)
                    best_stage_out = stage_out_reshaped[torch.arange(B), best_indices]  # (B, n_i, D)
                    traj_pred = self.head(best_stage_out)  # (B, n_i, 2)
                    best_stage_trajs.append(traj_pred)
                
                # Compute multi-scale loss
                total_traj_loss = 0.0
                for i, traj_pred in enumerate(best_stage_trajs):
                    stage_indices_i = self.stage_indices[i]
                    gt_stage = gt_traj_normalized[:, stage_indices_i, :]  # (B, n_i, 2)
                    mask_stage = gt_mask[:, stage_indices_i]  # (B, n_i)
                    
                    loss_per_point = F.smooth_l1_loss(traj_pred, gt_stage, reduction='none', beta=1.0)
                    mask_expanded_stage = mask_stage.unsqueeze(-1).float()
                    loss_per_point = loss_per_point * mask_expanded_stage
                    loss_stage = loss_per_point.sum() / (mask_expanded_stage.sum() + 1e-6)
                    
                    total_traj_loss += self.ms_traj_loss_weights[i] * loss_stage
                    out[f"loss_s{i+1}"] = loss_stage.clone().detach()
                    
                    # Record loss of the last point (for monitoring)
                    if len(stage_indices_i) > 0:
                        last_point_idx = -1
                        pred_last_point = traj_pred[:, last_point_idx, :]  # (B, 2)
                        gt_last_point = gt_stage[:, last_point_idx, :]  # (B, 2)
                        
                        loss_last_point = F.smooth_l1_loss(pred_last_point, gt_last_point, 
                                                            reduction='none', beta=1.0)
                        
                        mask_last = mask_stage[:, last_point_idx].unsqueeze(-1).float()
                        loss_last_point = (loss_last_point * mask_last).sum() / (mask_last.sum() + 1e-6)
                        out[f"loss_s{i+1}_last_point"] = loss_last_point.clone().detach()
                
                # Total loss = trajectory loss + confidence loss
                out["loss_traj"] = total_traj_loss
                out["loss"] = total_traj_loss + loss_confidence
                
                # Record best predictions for monitoring
                best_trajs = pred_trajs[torch.arange(B), best_indices]  # (B, T, 2)
                out["pred_traj"] = best_trajs
                
            
            # 9. Test mode: choose the one with highest confidence
            else:
                score_probs = torch.sigmoid(scores)  # (B, K)
                best_indices = torch.argmax(score_probs, dim=-1)  # (B,)
                
                # Collect trajectory predictions of all stages for the best sample
                best_stage_trajs = []
                for i, stage_out in enumerate(stage_outputs):
                    # stage_out: (B*K, n_i, D)
                    stage_out_reshaped = stage_out.reshape(B, K, -1, D)  # (B, K, n_i, D)
                    best_stage_out = stage_out_reshaped[torch.arange(B), best_indices]  # (B, n_i, D)
                    traj_pred = self.head(best_stage_out)  # (B, n_i, 2)
                    
                    
                    best_stage_trajs.append(traj_pred)
                    # Save trajectories for each stage
                    out[f"stage{i}_traj"] = traj_pred
                
                # Final prediction: use the last stage (should contain all T points)
                pred_traj = best_stage_trajs[-1]  # (B, T, 2)
                
                out["pred_traj"] = pred_traj
                out["pred_scores"] = score_probs
                out["pred_topk_trajs"] = pred_trajs  # Save all top-k trajectories (last stage only)
            
            return out
        
        core = traj_feats[torch.arange(B, device=device), query_index]  # (B, D_in)
        core_n = self.norm_traj(core)

        tpe = sinusoid_tpe(T, self.d_in, device, core.dtype)  # (T, D_in)
        tgt_all = core_n.unsqueeze(1) + tpe.unsqueeze(0)     # (B, T, D_in)
        tgt_all = F.layer_norm(tgt_all, (self.d_in,))

        # 3. Call VLM parallel decoding: forward_queries_parallel_v2
        # Input: context + query_embeds; output: features of all stages
        final_tgt_output, updated = forward_queries_parallel_v2_fn(
            base_inputs_embeds=context,      # (B, Q, D)
            query_embeds=tgt_all,       # (B, T, D)
            stage_indices=self.stage_indices,
            attention_mask=None,
            position_ids=None,
            training=training,
            detach=False,
            return_full_hidden=False,
        )

        final_tgt_output = final_tgt_output.float()
        # final_tgt_output: (B, Q_final, D), ordered by stage
        # updated contains stage_len_list and related info
        
        stage_len_list = updated["stage_len_list"]
        
        # 4. Split final_tgt_output by stage
        stage_outputs = []
        offset = 0
        for i, stage_len in enumerate(stage_len_list):
            stage_out = final_tgt_output[:, offset:offset+stage_len, :]  # (B, n_i, D)
            stage_outputs.append(stage_out)
            offset += stage_len
        
        # 5. Each stage predicts trajectory points at its corresponding time steps
        # stage_out (B, n_i, D) -> directly predict coordinates of these n_i points (B, n_i, 2)
        stage_trajs = []
        for i, stage_out in enumerate(stage_outputs):
            # stage_out: (B, n_i, D), where n_i is the number of time steps for this stage
            # Example: stage 0 -> (B, 1, D), predicts only time step 5
            #       stage 5 -> (B, 6, D), predicts all 6 time steps
            traj = self.head(stage_out)  # (B, n_i, 2)
            stage_trajs.append(traj)
        
        # 6. Output: use the last stage at test time (contains all points), keep all stages at train time
        if not training or not return_loss:
            # Test mode: use only the last-stage prediction (should be full 6 points)
            for i, traj in enumerate(stage_trajs):
                out[f"stage{i}_traj"] = traj
            pred_traj_final = stage_trajs[-1]  # (B, T, 2)
            assert pred_traj_final.shape[1] == self.fut_ts, \
                f"Last stage should predict all {self.fut_ts} points, got {pred_traj_final.shape[1]}"

            
            out["pred_traj"] = pred_traj_final
        else:
            # Training mode: keep all stage predictions for multi-scale supervision
            # Note: during training, predictions stay in normalized space (if norm_traj_mode is enabled)
            out["pred_traj"] = stage_trajs[-1]  # Main prediction (full trajectory)
            for i, traj in enumerate(stage_trajs):
                out[f"stage{i}_traj"] = traj
        
        # 7. Compute loss (training mode)
        if return_loss and gt_traj is not None:
            if gt_mask is None:
                gt_mask = torch.ones(B, self.fut_ts, dtype=torch.bool, device=device)
            
            # Normalized mode: normalize GT trajectory (for training)
            # Keep original GT for monitoring the last-point loss
            gt_traj_orig = gt_traj.clone() if self.norm_traj_mode else None
            if self.norm_traj_mode:
                gt_traj = self.normalize_traj(gt_traj)  # (B, T, 2) normalized
            
            total_loss = 0.0
            for i, traj_pred in enumerate(stage_trajs):
                # traj_pred: (B, n_i, 2), where n_i is the number of time steps for this stage
                # Need to take corresponding time steps from gt_traj
                stage_indices_i = self.stage_indices[i]  # e.g., [5] or [0,5] or [0,1,2,3,4,5]
                
                # Take corresponding points from GT: (B, T, 2) -> (B, n_i, 2)
                gt_stage = gt_traj[:, stage_indices_i, :]  # (B, n_i, 2)
                
                # Take corresponding points from mask (if provided)
                if gt_mask is not None:
                    mask_stage = gt_mask[:, stage_indices_i]  # (B, n_i)
                else:
                    mask_stage = None
                
                # Compute this stage loss (in normalized space, for training)
                loss_per_point = F.smooth_l1_loss(traj_pred, gt_stage, reduction='none', beta=1.0)  # (B, n_i, 2)
                
                # Apply mask
                if mask_stage is not None:
                    mask_expanded = mask_stage.unsqueeze(-1).float()  # (B, n_i, 1)
                    loss_per_point = loss_per_point * mask_expanded
                    loss_stage = loss_per_point.sum() / (mask_expanded.sum() + 1e-6)
                else:
                    loss_stage = loss_per_point.mean()
                
                # Weighted accumulation
                total_loss += self.ms_traj_loss_weights[i] * loss_stage
                
                # Record each stage loss (for monitoring)
                out[f"loss_s{i+1}"] = loss_stage.clone().detach()
                
                # Supervise each stage's last-point loss separately (monitoring only)
                if len(stage_indices_i) > 0:
                    last_point_idx = -1  # Last point of this stage
                    pred_last_point = traj_pred[:, last_point_idx, :]  # (B, 2)
                    gt_last_point = gt_stage[:, last_point_idx, :]      # (B, 2)
                    

                    # Non-normalized mode: compute directly in original space
                    loss_last_point = F.smooth_l1_loss(pred_last_point, gt_last_point, 
                                                        reduction='none', beta=1.0)  # (B, 2)
                    
                    if mask_stage is not None:
                        mask_last = mask_stage[:, last_point_idx].unsqueeze(-1).float()  # (B, 1)
                        loss_last_point = (loss_last_point * mask_last).sum() / (mask_last.sum() + 1e-6)
                    else:
                        loss_last_point = loss_last_point.mean()
                    
                    # Record last-point loss (in original space, for monitoring)
                    out[f"loss_s{i+1}_last_point"] = loss_last_point.clone().detach()
            
            out["loss"] = total_loss

        
        return out


class VLAPlannerMSARv4(nn.Module):
    """
    V4: works with MSARActionHeadv4 for unified parallel decoding
    
    Key improvements:
    1. Call forward_queries_parallel_v2 (no GT/predicted-point embeddings)
    2. Unified training and testing logic
    """
    def __init__(
        self,
        *,
        num_category: int,
        fut_ts: int,
        lm_dim: int,
        category_queries: nn.Parameter,
        category_classifier: nn.Module,
        traj_queries: nn.Parameter,
        ego_adapter: nn.Module,
        vl_adapter: nn.Module,
        action_head: MSARActionHeadv4,      # v4 action head
        classification_loss_weight: float = 1.0,
        traj_reg_loss_weight: float = 1.0,
        cls_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        cls_correct_idx: bool = False,
    ):
        super().__init__()
        self.num_category = num_category
        self.fut_ts = fut_ts
        self.lm_dim = lm_dim
        
        self.category_queries = category_queries
        self.category_classifier = category_classifier
        self.traj_queries = traj_queries
        
        self.ego_adapter = ego_adapter
        self.vl_adapter = vl_adapter
        self.action_head = action_head
        if cls_loss_fn is None:
            raise ValueError('cls_loss_fn must be provided')
        self.cls_loss_fn = cls_loss_fn
        self.classification_loss_weight = classification_loss_weight
        self.traj_reg_loss_weight = traj_reg_loss_weight
        self.keep_k = 256
        self.cls_correct_idx = cls_correct_idx

        if self.keep_k == 384:
            self.router = None
        else:
            self.router = VisionPruningRouterGumbel(lm_dim, lm_dim // 2, keep_k=self.keep_k)

        
    @torch.no_grad()
    def _extrapolate_to_T(self, pts: torch.Tensor, T: int) -> torch.Tensor:
        N = pts.shape[1]
        if N >= T: return pts[:, :T, :]
        if N == 1:
            last = pts[:, -1:, :].expand(1, T - N + 1, 2)
            return torch.cat([pts, last[:, 1:, :]], dim=1)
        vel = pts[:, 1:] - pts[:, :-1]
        last_v = vel[:, -1:, :]
        last_p = pts[:, -1:, :]
        ext = []
        for _ in range(T - N):
            nxt = last_p + last_v
            ext.append(nxt)
            last_p = nxt
        return torch.cat([pts, torch.cat(ext, dim=1)], dim=1)
    
    def _build_gt_batch(self, img_metas: List[dict], device) -> torch.Tensor:
        batch = []
        for m in img_metas:
            gt_np = m["gt_planning"][..., :2]
            gt = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32)
            gt = self._extrapolate_to_T(gt, self.fut_ts)
            batch.append(gt)
        return torch.cat(batch, dim=0)
    
    @torch.no_grad()
    def _select_index_infer(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)
    

    def forward_global_reasoning(
        self,
        *,
        input_ids: torch.Tensor,
        vision_query: torch.Tensor,
        can_bus_emb: torch.Tensor,
        meta_query: Optional[torch.Tensor],
        vlm_attn_mask: Optional[torch.Tensor],
        forward_understand_scene_fn,
        # rethink_way: str = "original", # original prefix surfix interleave
    ) -> Dict[str, torch.Tensor]:

        # 1. understand the scene
        Q_vision = vision_query.shape[1]
        vision_ego_embeded = torch.cat([vision_query.clone(), can_bus_emb], dim=1)
        updated_vision_ego, _ = forward_understand_scene_fn(input_ids=input_ids, vision_embeded=vision_ego_embeded, attention_mask=vlm_attn_mask)
        updated_vision_query = updated_vision_ego[:, :Q_vision, :] # (B, Q_vision, lm_dim)

        # 2. recognize the critical objects
        if self.keep_k != 384:
            critical_vision_query, selected_weights, topk_idx, loss_diversity = self.router(can_bus_emb, updated_vision_query) # (B, topk_vision, lm_dim)
        else:
            critical_vision_query = updated_vision_query
            selected_weights = None
            topk_idx = None
            loss_diversity = None

        # 3. rethink the scene
        # TODO: Later, prompts can be adjusted to add cue words for key objects, then insert corresponding queries to implement interleave.
        # TODO: Explore adding supervision for visual queries: define rules to select corresponding GT labels and supervise router query selection.

        top_k = critical_vision_query.shape[1]
        num_meta = meta_query.shape[1]
        critical_vision_ego_meta = torch.cat([critical_vision_query, can_bus_emb, meta_query], dim=1) # (B, Q_vision+1+num_meta, lm_dim)
        rethinked_vision_ego_meta, new_context = forward_understand_scene_fn(input_ids=input_ids, vision_embeded=critical_vision_ego_meta, attention_mask=vlm_attn_mask) # (B, top_k+1+num_meta, lm_dim)
        final_vision_query = rethinked_vision_ego_meta[:, :top_k, :] # (B, top_k, lm_dim)
        if self.cls_correct_idx:
            final_meta_query = rethinked_vision_ego_meta[:, top_k+1:top_k+1+num_meta, :]
        else:
            final_meta_query = rethinked_vision_ego_meta[:, top_k:top_k+num_meta, :] # (B, num_meta, lm_dim)


        return final_vision_query, selected_weights, topk_idx, final_meta_query, new_context, loss_diversity
    
    def forward(
        self,
        *,
        img_metas: List[dict],
        input_ids: torch.Tensor,
        map_query: torch.Tensor,
        can_bus_emb: torch.Tensor,
        vlm_attn_mask: Optional[torch.Tensor],
        vlm_labels: Optional[torch.Tensor],
        forward_queries_fn,  # actually v4 uses forward_queries_parallel_v2
        forward_understand_scene_fn = None,
        return_loss: bool = False,
        save_path: Optional[str] = None,
        gt_category_index: Optional[torch.Tensor] = None,
        gt_traj: Optional[torch.Tensor] = None,
        gt_traj_mask: Optional[torch.Tensor] = None,
        use_grpo: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        device = map_query.device
        B = map_query.size(0)
        out: Dict[str, torch.Tensor] = {}
        # 1. GT trajectory
        if gt_traj is None and return_loss:
            gt_traj = self._build_gt_batch(img_metas, device)

        loss_diversity = None
        output_traj_feats = self.traj_queries.unsqueeze(0).expand(B, -1, -1)       # (B, C, lm_dim)
        output_category_feats = self.category_queries.unsqueeze(0).expand(B, -1, -1)  # (B, C, cat_dim)

        updated_vision_query, selected_weights, topk_idx, final_meta_query, new_context, loss_diversity = self.forward_global_reasoning(
            input_ids=input_ids,
            vision_query=map_query,
            can_bus_emb=can_bus_emb,
            meta_query=output_category_feats,
            vlm_attn_mask=vlm_attn_mask,
            forward_understand_scene_fn=forward_understand_scene_fn,
        )

        # 3. Classifier
        ego_features = self.ego_adapter(can_bus_emb)       # (B, 1, cat_dim)

        vision_features = self.vl_adapter(updated_vision_query)       # (B, Q, cat_dim)
        logits = self.category_classifier(final_meta_query, ego_features, vision_features)  # (B, C)
        out["logits"] = logits
        
        # 4. Select index
        # Selected index
        if return_loss and gt_category_index is not None:
            sel_idx = gt_category_index.to(device)
        elif save_path is not None and 'inference_wgt' in save_path:
            sel_idx = gt_category_index.to(device)
            out['pred_index'] = self._select_index_infer(logits)
        else:
            sel_idx = self._select_index_infer(logits)
        out["query_index"] = sel_idx
        
        context = new_context.float()

        # 7. Action head: uniformly use parallel decoding
        msar = self.action_head(
            traj_feats=output_traj_feats,
            context=context,
            can_bus_emb=can_bus_emb,
            query_index=sel_idx,
            forward_queries_parallel_v2_fn=forward_queries_fn,  # pass the v2 version
            training=return_loss,
            return_loss=return_loss,
            gt_traj=gt_traj if return_loss else None,
            gt_mask=gt_traj_mask,
            use_grpo=use_grpo,
            logits=logits,
        )


        for i in range(1,self.action_head.num_stages + 1):
            if f"stage{i-1}_traj" in msar:
                out[f"stage{i}_traj"] = msar[f"stage{i-1}_traj"]

        out["pred_traj"] = msar["pred_traj"]
        
        # 8. Losses
        if return_loss:
            assert gt_category_index is not None # (B,)

            # Diversity loss
            if loss_diversity is not None:
                out["loss_diversity"] = loss_diversity
            
            # Classification loss
            cls_loss = self.cls_loss_fn(logits, gt_category_index.to(device))
            out["traj_class_loss"] = self.classification_loss_weight * cls_loss
            
            # Trajectory loss
            traj_loss = msar["loss"]
            out["msar_traj_loss"] = self.traj_reg_loss_weight * traj_loss
            
            # Per-stage losses (for monitoring)
            for i in range(1, self.action_head.num_stages + 1):
                key = f"loss_s{i}"
                if key in msar:
                    out[key] = msar[key]
                
                # Propagate the last-point loss for each stage
                last_point_key = f"loss_s{i}_last_point"
                if last_point_key in msar:
                    out[last_point_key] = msar[last_point_key]


            if "loss_confidence" in msar:
                out["loss_confidence"] = msar["loss_confidence"].clone().detach()
        
        return out
