# Vendored from VOSR's models/lightningdit.py (by Jingfeng Yao, HUST-VL;
# built on facebookresearch/DiT and willisma/SiT).
#
# Trimmed for VOSR2 v1:
# - timm's PatchEmbed/Mlp are replaced by the small local equivalents below
#   (VOSR2 targets zero new dependencies and timm is not otherwise required).
# - torch.compile decorators on module methods are dropped; ComfyUI runs on
#   accelerators/platforms (e.g. Windows without Triton) where compiling
#   arbitrary custom-node code is not a safe default.
# - Only forward_flexible's code path is kept; the fixed-size forward(),
#   sin-cos position-embedding helpers, and pos-embed interpolation were
#   dead code for this one-step model (LightningDiT is RoPE-conditioned and
#   never used its optional absolute pos_embed) and are removed.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import VisionRotaryEmbeddingFast
from .rmsnorm import RMSNorm
from .swiglu_ffn import SwiGLUFFN


class PatchEmbed(nn.Module):
    """Conv2d patchify, (N, C, H, W) -> (N, T, D). Accepts any H, W divisible by patch_size."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    @property
    def num_patches(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=False, fused_attn: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond):
        B, N, C = x.shape
        B_cond, N_cond, _ = cond.shape

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(x)


def modulate_adasin(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, use_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        z_dims=None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm)

    def forward(self, x, c, z=None, feat_rope=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + gate_msa * self.attn(modulate_adasin(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        if self.z_dims is not None:
            x = x + self.cross_attn(x, z)
        x = x + gate_mlp * self.mlp(modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LightningDiT(nn.Module):
    """One-step VOSR2 flow-matching backbone. Only forward_flexible is used at inference."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        out_channels=16,
        hidden_size=1536,
        depth=36,
        num_heads=24,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        z_dims=None,
        encdim_ratio=2,
        num_fused_layers=1,
        auxiliary_time_cond=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm, z_dims=z_dims,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, use_rmsnorm=use_rmsnorm)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.num_fused_layers = num_fused_layers
            self.layer_norm = nn.LayerNorm(z_dims)
            self.mlp_ca = Mlp(z_dims, hidden_size * encdim_ratio, out_features=hidden_size, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.auxiliary_time_cond = auxiliary_time_cond
        self.r_embedder = TimestepEmbedder(hidden_size) if auxiliary_time_cond else None

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def _get_dynamic_rope(self, hw_seq_len, device, dtype):
        if not self.use_rope:
            return None

        half_head_dim = self.hidden_size // self.num_heads // 2
        pt_seq_len = self.x_embedder.img_size[0] // self.patch_size

        from einops import repeat

        from .pos_embed import broadcat, rotate_half

        theta = 10000
        freqs = 1.0 / (theta ** (torch.arange(0, half_head_dim, 2, device=device)[: (half_head_dim // 2)].float() / half_head_dim))
        t = torch.arange(hw_seq_len, device=device).float() / hw_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).to(dtype)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).to(dtype)

        def dynamic_rope_fn(t_input):
            return t_input * freqs_cos + rotate_half(t_input) * freqs_sin

        return dynamic_rope_fn

    def forward_flexible(self, x, t, r=None, z=None):
        """Forward pass supporting variable (square) input sizes via dynamically-sized RoPE."""
        N, C, H, W = x.shape
        assert H == W, "forward_flexible currently only supports square inputs"

        current_hw_seq_len = H // self.patch_size

        x = self.x_embedder(x)
        t_raw = t
        t = self.t_embedder(t)
        r = self.r_embedder(r) * (t_raw - r).unsqueeze(-1) if self.r_embedder is not None else 0
        c = t + r
        c0 = self.t_block(c)

        if self.z_dims is not None and z is not None:
            z = z[0]
            z = self.layer_norm(z)
            z = self.mlp_ca(z)

        if self.use_rope:
            train_hw_seq_len = self.x_embedder.img_size[0] // self.patch_size
            if current_hw_seq_len != train_hw_seq_len:
                feat_rope = self._get_dynamic_rope(current_hw_seq_len, x.device, x.dtype)
            else:
                feat_rope = self.feat_rope
        else:
            feat_rope = None

        for block in self.blocks:
            x = block(x, c0, z, feat_rope)

        x = self.final_layer(x, c)
        return self.unpatchify(x)
