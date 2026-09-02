# Vendored from VOSR's models/qwenimage_vae2d.py (upstream: Qwen-Image /
# Wan / HuggingFace teams, Apache-2.0).
#
# The upstream module subclasses diffusers' ModelMixin/ConfigMixin and reuses
# diffusers.DiagonalGaussianDistribution / AutoencoderKLOutput / DecoderOutput.
# diffusers is not part of a stock ComfyUI install, and VOSR2 targets zero new
# dependencies, so this version is self-contained: a plain nn.Module with a
# from_pretrained() that reads the same config.json/safetensors layout, and a
# local DiagonalGaussianDistribution with only the two ops this node needs
# (.mode(), .sample()).
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# Qwen-Image VAE per-channel latent normalization constants (from the
# upstream checkpoint's config.json defaults).
DEFAULT_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
DEFAULT_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, logvar = parameters.chunk(2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator = None) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + std * noise


@dataclass
class EncoderOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class RMSNorm2D(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class ResidualBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = RMSNorm2D(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = RMSNorm2D(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.conv2(x)
        return x + h


class AttentionBlock2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b, 3, c, h * w).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.unbind(dim=1)
        # 4-D (b, 1, N, c) single-head layout selects the O(N)-memory fused
        # attention kernel; the 3-D (b, N, c) form falls back to O(N^2) math
        # on some accelerators and OOMs on large tiles.
        x_attn = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x_attn = x_attn.permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(x_attn) + identity


class Resample2D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class MidBlock2D(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        resnets = [ResidualBlock2D(dim, dim)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock2D(dim))
            resnets.append(ResidualBlock2D(dim, dim))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class Encoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=32, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=()):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0

        self.conv_in = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock2D(in_dim, out_dim))
                if scale in attn_scales:
                    self.down_blocks.append(AttentionBlock2D(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                self.down_blocks.append(Resample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = MidBlock2D(out_dim, num_layers=1)
        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, upsample_mode=None):
        super().__init__()
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock2D(cur, out_dim))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Resample2D(out_dim, mode=upsample_mode)]) if upsample_mode is not None else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Decoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        dim_mult = list(dim_mult)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = nn.Conv2d(z_dim, dims[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock2D(dims[0], num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if i != len(dim_mult) - 1 else None
            self.up_blocks.append(UpBlock2D(in_dim, out_dim, num_res_blocks, upsample_mode))

        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class AutoencoderKLQwenImage2D(nn.Module):
    """2D-only Qwen-Image VAE. API: encode(x) -> EncoderOutput, decode(z) -> DecoderOutput."""

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        latents_mean=None,
        latents_std=None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.latents_mean = list(latents_mean) if latents_mean is not None else list(DEFAULT_LATENTS_MEAN)
        self.latents_std = list(latents_std) if latents_std is not None else list(DEFAULT_LATENTS_STD)

        self.encoder = Encoder2D(dim=base_dim, z_dim=z_dim * 2, dim_mult=dim_mult, num_res_blocks=num_res_blocks, attn_scales=attn_scales)
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)
        self.decoder = Decoder2D(dim=base_dim, z_dim=z_dim, dim_mult=dim_mult, num_res_blocks=num_res_blocks)

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        h = self.quant_conv(self.encoder(x))
        return EncoderOutput(latent_dist=DiagonalGaussianDistribution(h))

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        dec = self.decoder(self.post_quant_conv(z))
        return DecoderOutput(sample=torch.clamp(dec, -1.0, 1.0))

    @classmethod
    def from_config_dict(cls, config: dict) -> "AutoencoderKLQwenImage2D":
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", ())),
            latents_mean=config.get("latents_mean"),
            latents_std=config.get("latents_std"),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "AutoencoderKLQwenImage2D":
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        model = cls.from_config_dict(config)
        weight_path = path / "diffusion_pytorch_model.safetensors"
        state_dict = load_file(str(weight_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"VOSR2 VAE checkpoint at {weight_path} does not match the expected "
                f"architecture (missing={missing}, unexpected={unexpected})."
            )
        return model
