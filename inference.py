"""VOSR2 inference contract: bicubic pre-scale, pad, encode, one-step denoise,
decode, color-align, crop -- with optional DiT/VAE tiling. See VOSR2.md
"Inference contract" and "Tiling".

`forward_flexible` (models/lightningdit.py) asserts a square input -- true of
every upstream tiled call (tiles are always cropped `lt_size x lt_size`), but
not of a whole non-square image's latent. To support arbitrary aspect ratios
without tiling, the untiled path additionally pads to a square before the DiT
call and crops back afterward; this is invisible in the output (still cropped
to the exact requested size) and only engages when tile_size == 0.
"""
import logging

import torch
import torch.nn.functional as F

import comfy.utils

from .color import apply_color_alignment
from .tiled_vae import _gaussian_weights, _make_tile_grid

AE_FACTOR = 8
DIT_PATCH_SIZE = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH_SIZE  # 16


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _pad_to_square(x: torch.Tensor):
    _, _, h, w = x.shape
    side = max(h, w)
    pad_h, pad_w = side - h, side - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _generate_noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """One item's worth of noise per call; a local CPU Generator keeps global RNG state untouched."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _generate_noise_batch(shape_per_item, seed: int, batch_size: int, device, dtype) -> torch.Tensor:
    return torch.stack([_generate_noise(shape_per_item, seed + i, device, dtype) for i in range(batch_size)], dim=0)


def _tile_params(tile_size: int, tile_overlap: int, lh: int, lw: int):
    """Pixel tile size/overlap -> latent-space, aligned to the DiT patch size."""
    lt_size = max((tile_size // AE_FACTOR // DIT_PATCH_SIZE) * DIT_PATCH_SIZE, DIT_PATCH_SIZE)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def _resize_to_target(images_bhwc01: torch.Tensor, upscale: int, device) -> torch.Tensor:
    x = images_bhwc01.movedim(-1, 1).to(device)  # BCHW [0, 1]
    _, _, h, w = x.shape
    target_h, target_w = h * upscale, w * upscale
    return F.interpolate(x, size=(target_h, target_w), mode="bicubic").clamp(0.0, 1.0)


def _run_untiled_batch(model, resized01: torch.Tensor, seed: int, vae_tile_size: int, vae_tile_overlap: int) -> torch.Tensor:
    b, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded01 = _pad_to_square(padded01)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    venc_fea = model.vision_features(padded01)

    noise = _generate_noise_batch(lq_latent.shape[1:], seed, b, lq_latent.device, lq_latent.dtype)
    sr_latent = model.denoise_one_step(lq_latent, noise, venc_fea)
    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)

    return decoded_pm1[:, :, :h, :w]


def _count_dit_tiles(h: int, w: int, tile_size: int, tile_overlap: int) -> int:
    """Tiles a `run_vosr2` will submit to the DiT for one already-upscaled HxW image, for progress reporting."""
    padded_h = h + (-h) % PAD_MULTIPLE
    padded_w = w + (-w) % PAD_MULTIPLE
    lh, lw = padded_h // AE_FACTOR, padded_w // AE_FACTOR
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)
    if lh <= lt_size and lw <= lt_size:
        return 1
    return len(_make_tile_grid(lh, lt_size, lt_overlap)) * len(_make_tile_grid(lw, lt_size, lt_overlap))


def _run_tiled_single(model, resized01: torch.Tensor, seed: int, tile_size: int, tile_overlap: int,
                       vae_tile_size: int, vae_tile_overlap: int, pbar: comfy.utils.ProgressBar = None) -> torch.Tensor:
    """Latent-space tiled DiT inference for one image (B=1). Ported from VOSR's tiled_latent_inference."""
    _, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    _, lc, lh, lw = lq_latent.shape
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        lq_sq = _pad_to_square(lq_latent)
        venc_fea = model.vision_features(padded01)
        noise = _generate_noise(lq_sq.shape[1:], seed, lq_sq.device, lq_sq.dtype).unsqueeze(0)
        sr_latent = model.denoise_one_step(lq_sq, noise, venc_fea)[:, :, :lh, :lw]
        if pbar is not None:
            pbar.update_absolute(pbar.current + 1)
    else:
        h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
        w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
        g_weight = _gaussian_weights(lt_size, lt_size, lc, lq_latent.device)

        tile_venc = {}
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt_size) * AE_FACTOR, padded01.shape[2])
                pw_e = min((wi + lt_size) * AE_FACTOR, padded01.shape[3])
                tile_venc[(hi, wi)] = model.vision_features(padded01[:, :, ph_s:ph_e, pw_s:pw_e])

        noise = _generate_noise(lq_latent.shape[1:], seed, lq_latent.device, lq_latent.dtype).unsqueeze(0)
        z = noise

        u_acc = torch.zeros_like(lq_latent)
        w_acc = torch.zeros_like(lq_latent)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt_size, wi + lt_size
                inp = torch.cat([lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], dim=1)
                u_tile = model.dit_velocity(inp, 1.0, 0.0, tile_venc[(hi, wi)])
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
                if pbar is not None:
                    pbar.update_absolute(pbar.current + 1)

        sr_latent = z - u_acc / w_acc

    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decoded_pm1[:, :, :h, :w]


def run_vosr2(
    model,
    images_bhwc01: torch.Tensor,
    upscale: int,
    seed: int,
    color_alignment: str,
    tile_size: int,
    tile_overlap: int,
    vae_tile_size: int,
    vae_tile_overlap: int,
) -> torch.Tensor:
    if images_bhwc01.shape[-1] != 3:
        raise ValueError(f"VOSR2Upscale expects a 3-channel RGB IMAGE, got {images_bhwc01.shape[-1]} channels.")
    if tile_size > 0 and tile_overlap >= tile_size:
        raise ValueError(f"VOSR2Upscale: tile_overlap ({tile_overlap}) must be smaller than tile_size ({tile_size}).")
    if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
        raise ValueError(f"VOSR2Upscale: vae_tile_overlap ({vae_tile_overlap}) must be smaller than vae_tile_size ({vae_tile_size}).")

    device = model.dit_patcher.load_device
    resized01 = _resize_to_target(images_bhwc01, upscale, device)
    b, _, h, w = resized01.shape

    if tile_size == 0 and (h > 512 or w > 512):
        logging.warning(
            f"VOSR2Upscale: target size {w}x{h} (after {upscale}x upscale) exceeds VOSR 2.0's "
            f"native 512px training resolution with tile_size=0 (DiT tiling disabled). VOSR2.md "
            f"states tiling is not optional above 512px -- set tile_size > 0 (e.g. 512) or "
            f"quality will likely degrade."
        )
    if vae_tile_size == 0 and (h > 1024 or w > 1024):
        logging.warning(
            f"VOSR2Upscale: target size {w}x{h} (after {upscale}x upscale) with vae_tile_size=0 "
            f"(VAE tiling disabled) decodes the full image in one pass regardless of tile_size -- "
            f"this is a common source of CUDA out-of-memory errors at this resolution. Set "
            f"vae_tile_size > 0 (e.g. 1024) to bound peak VRAM."
        )

    if tile_size > 0:
        _, _, h, w = resized01.shape
        total_tiles = b * _count_dit_tiles(h, w, tile_size, tile_overlap)
        pbar = comfy.utils.ProgressBar(total_tiles)
        outputs_pm1 = torch.cat([
            _run_tiled_single(model, resized01[i:i + 1], seed + i, tile_size, tile_overlap, vae_tile_size, vae_tile_overlap, pbar=pbar)
            for i in range(b)
        ], dim=0)
    else:
        outputs_pm1 = _run_untiled_batch(model, resized01, seed, vae_tile_size, vae_tile_overlap)

    decoded01 = (outputs_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
    aligned01 = apply_color_alignment(decoded01, resized01, color_alignment)
    return aligned01.movedim(1, -1)
