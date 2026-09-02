# Minimal local re-implementation of Meta's DINOv2 ViT-L/14 (facebookresearch/dinov2,
# Apache-2.0), sized to load the official `dinov2_vitl14` checkpoint directly from a
# local safetensors/pth file.
#
# VOSR2's spec forbids `torch.hub.load` at import/validate/execute time (it can
# execute cached repository code or reach the network), so the architecture is
# vendored here instead: patch embed, cls token, interpolated position encoding,
# and a stack of pre-norm transformer blocks with LayerScale. `interpolate_pos_encoding`
# below reproduces the official implementation's bicubic-resize formula verbatim
# (including its `+ 0.1` epsilon), since VOSR2 feeds arbitrary tile/image resolutions
# that rarely match the checkpoint's native 518x518 training grid.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_VITL14_CONFIG = dict(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        """Vendored verbatim from facebookresearch/dinov2 (Apache-2.0): bicubic-resizes
        the patch position embeddings to the current (w, h) grid. The `+ 0.1` offset
        before the target size and the `int()` truncation after are load-bearing --
        they sidestep a floating-point rounding edge case in `F.interpolate`'s
        `scale_factor` path that the upstream implementation works around this way."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward_intermediate_layer(self, x, layer_idx: int) -> torch.Tensor:
        """Run through `layer_idx + 1` blocks and return patch tokens (cls token dropped).

        VOSR2 only ever needs one fixed intermediate layer's patch tokens
        (`layer_dinov2b_list == [17]`, validated by the loader), so the remaining
        `depth - layer_idx - 1` blocks are skipped rather than computed and discarded.
        """
        x = self.prepare_tokens(x)
        for block in self.blocks[: layer_idx + 1]:
            x = block(x)
        return x[:, 1:]


def build_dinov2_vitl14() -> DinoVisionTransformer:
    return DinoVisionTransformer(**DINOV2_VITL14_CONFIG)
