import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, context):
        out, _ = self.attn(query, context, context)
        out = query + out              # residual
        out = self.norm(out)           # LayerNorm
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query):
        out, _ = self.attn(query, query, query)
        out = query + out              # residual
        out = self.norm(out)           # LayerNorm
        return out

class ImprovedClassifier(nn.Module):
    def __init__(self, dim, num_classes, hidden_dim=512, num_heads=4, test_only_six_category=False, use_cls_mixer=False, use_cls_film=True):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.test_only_six_category = test_only_six_category
        self.use_cls_mixer = use_cls_mixer
        self.use_cls_film = use_cls_film
        if self.use_cls_mixer:
            self.cls_mixer = MixerBlock(num_classes, dim)
        # FiLM generator (ego features -> modulation parameters)
        if self.use_cls_film:
            self.film_gen = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 2 * dim)
            )

        # cross attention: modulated category queries interact with temporal features
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.self_attn = SelfAttention(dim, num_heads=num_heads)

        # classification projection
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, category_feats, ego_features, temporal_features):
        """
        category_feats: (B, num_classes, d)
        ego_features: (B, 1, d)
        temporal_features: (B, Q, d)
        return: (B, num_classes)
        """
        B = temporal_features.size(0)

        # === Step 1: ego -> FiLM modulation ===
        if self.use_cls_film:
            film_params = self.film_gen(ego_features.squeeze(1))  # (B, 2d)
            gamma, beta = film_params.chunk(2, dim=-1)  # (B, d), (B, d)

            modulated_queries = gamma.unsqueeze(1) * category_feats + beta.unsqueeze(1)  # (B, num_classes, d)
        else:
            modulated_queries = category_feats

        # === Step 2: cross attention (modulated query <-> temporal) ===
        attended_queries = self.cross_attn(modulated_queries, temporal_features)  # (B, num_classes, d)
        attended_queries = self.self_attn(attended_queries)  # (B, num_classes, d)
        if self.use_cls_mixer:
            attended_queries = self.cls_mixer(attended_queries)  # (B, num_classes, d)

        # === Step 3: classification projection ===
        logits = self.proj(attended_queries).squeeze(-1)  # (B, num_classes)

        return logits

class ImprovedClassifierV2(nn.Module):
    def __init__(self, dim, num_classes, hidden_dim=512, num_heads=4, dropout=0.1, use_mixer=False, test_only_six_category=False):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        self.use_mixer = use_mixer
        self.test_only_six_category = test_only_six_category
        if self.use_mixer:
            self.mixer = MixerBlock(num_classes, dim, drop_path_rate=dropout)

        # === FiLM generator (ego features -> modulation parameters) ===
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2 * dim)
        )

        # === normalization layers (stabilize attention) ===
        self.ln_q   = nn.LayerNorm(dim)
        self.ln_ctx = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # === attention layers (keep original structure) ===
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.self_attn  = SelfAttention(dim, num_heads=num_heads)

        # === Head A: original MLP projection ===
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        # === Head B: prototype cosine head (more stable calibration) ===
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.cosine_scale = nn.Parameter(torch.tensor(20.0))  # global temperature

        # === learnable fusion coefficient for two heads (after sigmoid in (0,1)) ===
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        # === ego -> prior bias (lightweight calibration, no structural change) ===
        self.ego_bias = nn.Linear(dim, num_classes)

    def forward(self, category_feats, ego_features, temporal_features):
        """
        category_feats:   (B, num_classes, d)
        ego_features:     (B, 1, d)
        temporal_features:(B, Q, d)
        return: logits    (B, num_classes)
        """
        B = ego_features.size(0)

        # === Step 1: ego -> FiLM modulation ===
        film_params = self.film_gen(ego_features.squeeze(1))      # (B, 2d)
        gamma, beta = film_params.chunk(2, dim=-1)                # (B, d), (B, d)
        q0 = gamma.unsqueeze(1) * category_feats + beta.unsqueeze(1)  # (B, C, d)

        # === Step 2: Cross-Attn (q <-> temporal) + residual ===
        # Use Pre-LN + residual for stable training
        q1 = q0 + self.dropout(self.cross_attn(self.ln_q(q0), self.ln_ctx(temporal_features)))  # (B, C, d)

        # === Step 3: Self-Attn (intra-class interaction) + residual ===
        q2 = q1 + self.dropout(self.self_attn(self.ln_q(q1)))     # (B, C, d)

        if self.use_mixer:
            q2 = self.mixer(q2)

        # Normalize output for classification heads
        feats = self.ln_out(q2)                                    # (B, C, d)

        # === Head A: MLP classification ===
        logits_mlp = self.proj(feats).squeeze(-1)                  # (B, C)

        # === Head B: prototype cosine classification (class-wise aligns with its own prototype) ===
        W = F.normalize(self.prototypes, dim=-1)                   # (C, d)
        X = F.normalize(feats, dim=-1)                             # (B, C, d)
        # Class-wise alignment with corresponding prototype (diagonal term)
        logits_cos = self.cosine_scale * (X * W.unsqueeze(0)).sum(-1)  # (B, C)

        # === Fuse two heads (learnable, no extra hyperparameters) ===
        mix = torch.sigmoid(self.mix_logit)                        # scalar in (0,1)
        logits = mix * logits_mlp + (1.0 - mix) * logits_cos       # (B, C)

        # === Ego prior bias (lightweight calibration) ===
        logits = logits + self.ego_bias(ego_features.squeeze(1))   # (B, C)

        return logits


class ClassifierSelfCross(nn.Module):
    def __init__(self, dim, num_classes, hidden_dim=512, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # FiLM generator (ego features -> modulation parameters)
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2 * dim)
        )

        # cross attention: modulated category queries interact with temporal features
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.self_attn = SelfAttention(dim, num_heads=num_heads)

        # classification projection
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, category_feats, ego_features, vision_features):
        """
        category_feats: (B, num_classes, d)
        ego_features: (B, 1, d)
        vision_features: (B, Q, d)
        return: (B, num_classes)
        """
        B = ego_features.size(0)

        # === Step 1: ego -> FiLM modulation ===
        film_params = self.film_gen(ego_features.squeeze(1))  # (B, 2d)
        gamma, beta = film_params.chunk(2, dim=-1)  # (B, d), (B, d)

        modulated_queries = gamma.unsqueeze(1) * category_feats + beta.unsqueeze(1)  # (B, num_classes, d)

        # === Step 2: cross attention (modulated query <-> temporal) ===
        attended_queries = self.self_attn(modulated_queries)  # (B, num_classes, d)
        attended_queries = self.cross_attn(attended_queries, vision_features)  # (B, num_classes, d)
        
        # === Step 3: classification projection ===
        logits = self.proj(attended_queries).squeeze(-1)  # (B, num_classes)

        return logits
    
class SimpleClassifier(nn.Module):
    def __init__(self, dim, num_classes, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # MLP that generates FiLM parameters
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2 * dim)
        )

        # Project to score
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, category_feats, ego_features, temporal_features):
        """
        ego_features: (B, 1, d)
        temporal_features: (B, Q, d)
        return: (B, num_classes)
        """
        B = ego_features.size(0)

        # Aggregate context (simple mean pooling)
        context = torch.cat([ego_features, temporal_features], dim=1).mean(dim=1)  # (B, d)

        # Generate FiLM parameters
        film_params = self.film_gen(context)  # (B, 2d)
        gamma, beta = film_params.chunk(2, dim=-1)  # (B, d), (B, d)

        # FiLM modulation
        modulated_queries = gamma.unsqueeze(1) * category_feats + beta.unsqueeze(1)  # (B, num_classes, d)

        # Project to score
        logits = self.proj(modulated_queries).squeeze(-1)  # (B, num_classes)

        return logits

class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention:
      y = LN(x + gate(x) ⊙ Attn(x, context))
    The gate is channel-wise (Sigmoid), predicted from the query itself,
    and adaptively adjusts the importance of different contexts.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Identity()  # Can be replaced with nn.Linear(dim, dim) for re-projection
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # query: (B, Nq, D), context: (B, Nc, D), mask: (B, Nc)  True=pad
        upd, _ = self.attn(query, context, context, key_padding_mask=key_padding_mask)  # (B,Nq,D)
        g = self.gate(query)  # (B,Nq,D)
        out = self.norm(query + g * self.proj(upd))
        return out


class ClassSelfAttention(nn.Module):
    """
    Self-attention among class tokens to model competition/suppression.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attn(x, x, x)      # (B,C,D)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class GlobalMapSummary(nn.Module):
    """
    Use learnable queries to pool 1~K global tokens from temporal features
    as coarse-grained context.
    """
    def __init__(self, dim: int, num_heads: int = 4, num_tokens: int = 1, dropout: float = 0.0):
        super().__init__()
        self.q_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, temporal: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # temporal: (B,Q,D)
        B, _, D = temporal.shape
        q = self.q_tokens.unsqueeze(0).expand(B, -1, -1)  # (B,num_tokens,D)
        out, _ = self.attn(q, temporal, temporal, key_padding_mask=key_padding_mask)  # (B,num_tokens,D)
        return self.norm(out)  # (B,num_tokens,D)
