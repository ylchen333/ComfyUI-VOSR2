"""Torch-native color alignment, ported from VOSR's inference_vosr_onestep.py.

Upstream implements `adain_color_fix`/`wavelet_color_fix` via PIL round-trips and
OpenCV's GaussianBlur on uint8 arrays. VOSR2's contract requires float precision
throughout with no PIL/uint8 quantization and no OpenCV dependency, so both are
reimplemented here directly on BCHW float tensors in [0, 1], with a small separable
Gaussian blur standing in for `cv2.GaussianBlur`.
"""
import math

import torch
import torch.nn.functional as F

ADAIN_EPS = 1e-5
WAVELET_SIGMA = 5.0


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded, applied per-channel. x: (B, C, H, W).

    Reflect padding requires the pad width to be smaller than the dimension it
    pads, so each axis' kernel radius is capped to fit -- this only narrows the
    (already Gaussian-tapered) kernel tail on tiles/images smaller than ~8*sigma
    pixels, it never errors.
    """
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    desired_radius = max(int(math.ceil(sigma * 4.0)), 1)
    radius_h = max(min(desired_radius, h - 1), 0)
    radius_w = max(min(desired_radius, w - 1), 0)

    if radius_w > 0:
        kernel_h = _gaussian_kernel1d(sigma, radius_w, x.device, x.dtype).view(1, 1, 1, -1).expand(c, 1, 1, -1)
        x = F.pad(x, (radius_w, radius_w, 0, 0), mode="reflect")
        x = F.conv2d(x, kernel_h, groups=c)
    if radius_h > 0:
        kernel_v = _gaussian_kernel1d(sigma, radius_h, x.device, x.dtype).view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = F.pad(x, (0, 0, radius_h, radius_h), mode="reflect")
        x = F.conv2d(x, kernel_v, groups=c)
    return x


def adain_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Match target's per-channel spatial mean/std to source. Both (B, C, H, W) in [0, 1]."""
    target_mean = target.mean(dim=[2, 3], keepdim=True)
    target_std = target.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    source_mean = source.mean(dim=[2, 3], keepdim=True)
    source_std = source.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    result = (target - target_mean) / target_std * source_std + source_mean
    return torch.clamp(result, 0.0, 1.0)


def wavelet_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Combine source's low frequencies with target's high frequencies. Both (B, C, H, W) in [0, 1]."""
    source_low = gaussian_blur(source, WAVELET_SIGMA)
    target_low = gaussian_blur(target, WAVELET_SIGMA)
    target_high = target - target_low
    result = source_low + target_high
    return torch.clamp(result, 0.0, 1.0)


def apply_color_alignment(decoded: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    """decoded/reference: (B, C, H, W) float in [0, 1], same spatial size."""
    if mode == "none":
        return torch.clamp(decoded, 0.0, 1.0)
    if mode == "adain":
        return adain_color_fix(decoded, reference)
    if mode == "wavelet":
        return wavelet_color_fix(decoded, reference)
    raise ValueError(f"Unknown color_alignment mode: {mode!r}")
