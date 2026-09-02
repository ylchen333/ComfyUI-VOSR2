"""Gaussian-blended tiled VAE encode/decode for the Qwen-Image 2D VAE.

Ported from VOSR's tiled_vae.py. Full-image VAE encode/decode of large inputs can
OOM (the Qwen VAE runs in fp32 internally); this bounds peak memory to one tile's
activations. Encoding uses `latent_dist.mode()` (deterministic) rather than
`.sample()`, since sampling draws uncorrelated noise per tile and would create
seams at tile boundaries.

Why plain Gaussian blending is sufficient here (unlike SD/LDM's pad+crop `VAEHook`
machinery): the Qwen VAE uses RMSNorm2D, a purely per-pixel/per-channel norm with
no cross-tile statistics to reconcile (no GroupNorm), so each tile's forward pass
is independent and a weighted blend of overlapping outputs is well-founded.
"""
import math

import torch
import torch.nn.functional as F

AE_FACTOR = 8  # Qwen VAE spatial compression ratio


def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device) -> torch.Tensor:
    """2-D Gaussian blend mask (1, C, tile_h, tile_w) peaked at the centre."""
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def _make_tile_grid(length: int, tile: int, overlap: int) -> list:
    """Sorted, deduplicated starting positions that cover `length`."""
    stride = max(tile - overlap, 1)
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def encode_latent(vae, x: torch.Tensor):
    """Single-shot deterministic VAE encode -> normalized latent. Returns (latent, mean, std)."""
    device = x.device
    latents_mean = torch.tensor(vae.latents_mean, device=device).view(1, -1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.latents_std, device=device).view(1, -1, 1, 1)
    z = vae.encode(x).latent_dist.mode()
    return (z - latents_mean) * latents_std, latents_mean, latents_std


def decode_latent(vae, sr_latent: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Single-shot VAE decode -> pixels in [-1, 1]."""
    sr_latent = sr_latent / latents_std + latents_mean
    return vae.decode(sr_latent).sample.clamp(-1, 1)


def _tile_params(tile_size: int, tile_overlap: int, lh: int, lw: int):
    lt_size = max(tile_size // AE_FACTOR, 1)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def tiled_encode_latent(vae, x: torch.Tensor, tile_size: int, tile_overlap: int):
    """Output latent shape equals the untiled path: ceil(h/8) x ceil(w/8)."""
    pad, orig_h, orig_w = _pad_to_multiple(x, 8)
    lh, lw = pad.shape[2] // 8, pad.shape[3] // 8
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return encode_latent(vae, pad)

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)

    b = pad.shape[0]
    acc = wacc = g = None
    mean = std = None
    for hi in h_pos:
        for wi in w_pos:
            crop = pad[:, :, hi * 8:(hi + lt_size) * 8, wi * 8:(wi + lt_size) * 8]
            z_tile, mean, std = encode_latent(vae, crop)
            if acc is None:
                lc = z_tile.shape[1]
                g = _gaussian_weights(lt_size, lt_size, lc, x.device)
                acc = torch.zeros(b, lc, lh, lw, device=x.device, dtype=z_tile.dtype)
                wacc = torch.zeros_like(acc)
            acc[:, :, hi:hi + lt_size, wi:wi + lt_size] += z_tile * g
            wacc[:, :, hi:hi + lt_size, wi:wi + lt_size] += g

    blended = acc / wacc
    out_lh, out_lw = math.ceil(orig_h / 8), math.ceil(orig_w / 8)
    return blended[:, :, :out_lh, :out_lw], mean, std


def tiled_decode_latent(vae, sr_latent: torch.Tensor, latents_mean, latents_std, tile_size: int, tile_overlap: int) -> torch.Tensor:
    """Decodes in pixel space (3ch): each latent tile maps to an exactly-8x pixel tile."""
    b, _, lh, lw = sr_latent.shape
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return decode_latent(vae, sr_latent, latents_mean, latents_std)

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)

    out_h, out_w = lh * 8, lw * 8
    g = _gaussian_weights(lt_size * 8, lt_size * 8, 3, sr_latent.device)
    acc = torch.zeros(b, 3, out_h, out_w, device=sr_latent.device, dtype=sr_latent.dtype)
    wacc = torch.zeros_like(acc)
    for hi in h_pos:
        for wi in w_pos:
            he, we = hi + lt_size, wi + lt_size
            pix = decode_latent(vae, sr_latent[:, :, hi:he, wi:we], latents_mean, latents_std)
            acc[:, :, hi * 8:he * 8, wi * 8:we * 8] += pix * g
            wacc[:, :, hi * 8:he * 8, wi * 8:we * 8] += g

    return (acc / wacc).clamp(-1, 1)


def encode_dispatch(vae, x: torch.Tensor, vae_tile_size: int, vae_tile_overlap: int):
    """encode_latent, tiled when vae_tile_size > 0."""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_encode_latent(vae, x, vae_tile_size, vae_tile_overlap)
    return encode_latent(vae, x)


def decode_dispatch(vae, sr_latent: torch.Tensor, latents_mean, latents_std, vae_tile_size: int, vae_tile_overlap: int) -> torch.Tensor:
    """decode_latent, tiled when vae_tile_size > 0."""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_decode_latent(vae, sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decode_latent(vae, sr_latent, latents_mean, latents_std)
