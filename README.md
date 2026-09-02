# ComfyUI-VOSR2

ComfyUI nodes for **VOSR 2.0** — the one-step, 1.4B-parameter image
super-resolution model (LightningDiT + Qwen-Image 2D VAE + DINOv2-L
conditioning). This is a standalone community package; it is **not** part of the
upstream [`cswry/VOSR`](https://github.com/cswry/VOSR) repository.

- Upstream code: <https://github.com/cswry/VOSR>
- Upstream weights: <https://huggingface.co/CSWRY/VOSR>
- This node package: <https://github.com/ylchen333/ComfyUI-VOSR2>

The nodes load **local files only** — nothing is ever downloaded at import,
validation, or execution time, and `torch.hub` is never called. You download the
model files once (below) and place them under your ComfyUI `models/` directory.

---

## Nodes

Both appear under **`image/upscaling/VOSR2`**.

### VOSR 2.0 Model Loader (`VOSR2ModelLoader`)

Loads a VOSR 2.0 bundle and returns a `VOSR2_MODEL` object. Split from the
upscale node so that queued images do not rebuild several GB of weights.

| Input | Default | Notes |
|---|---|---|
| `model` | — | Bundle folder under `models/vosr2/` (must contain `args.json`) |
| `vae` | — | Qwen-Image 2D VAE folder under `models/vae/VOSR2/` |
| `vision_encoder` | — | DINOv2-L `.safetensors` under `models/clip_vision/VOSR2/` |
| `dtype` | `default` | `default` / `fp16` / `bf16` for the DiT + vision encoder. The VAE always runs in fp32. |

`args.json` is validated against the fixed VOSR 2.0 architecture before anything
is constructed — an incompatible checkpoint fails loudly instead of loading
partially.

### VOSR 2.0 Upscale (`VOSR2Upscale`)

One-step super-resolution on an `IMAGE` batch.

| Input | Default | Range | Notes |
|---|---:|---|---|
| `model` | — | `VOSR2_MODEL` | From the loader |
| `image` | — | `IMAGE` | Single image or batch |
| `upscale` | `4` | `1`–`4` | Exact output multiplier |
| `seed` | `42` | ≥ 0 | Latent-noise seed; batch item *i* uses `seed + i` |
| `color_alignment` | `wavelet` | `wavelet` / `adain` / `none` | Post-process against the bicubic target |
| `tile_size` | `0` | `0`–`4096`, step 64 | DiT pixel tile; `0` disables tiling |
| `tile_overlap` | `32` | `0`–`512`, step 8 | DiT tile overlap |
| `vae_tile_size` | `0` | `0`–`8192`, step 64 | VAE pixel tile; `0` decodes the whole image in one pass |
| `vae_tile_overlap` | `32` | `0`–`512`, step 8 | VAE tile overlap |

**Tiling is not optional above 512 px.** VOSR 2.0 was trained natively at up to
512 px, so whenever the *upscaled* output exceeds 512×512 set `tile_size` (e.g.
`512`) or quality degrades. For outputs much past 1024 px also set
`vae_tile_size` (e.g. `1024`) or the full-image VAE decode will likely OOM.

---

## Model files

All four assets come from the official **[CSWRY/VOSR](https://huggingface.co/CSWRY/VOSR)**
Hugging Face repo. Download them and arrange them like this under your ComfyUI
install:

```
ComfyUI/models/
├── vosr2/
│   └── VOSR2/
│       ├── args.json
│       └── checkpoints/
│           └── ema_model.safetensors
├── vae/
│   └── VOSR2/
│       └── Qwen-Image-vae-2d/
│           ├── config.json
│           └── diffusion_pytorch_model.safetensors
└── clip_vision/
    └── VOSR2/
        └── dinov2_vitl14.safetensors
```

| File in repo | Direct download | Put it at |
|---|---|---|
| `VOSR2/args.json` | [link](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/args.json) | `models/vosr2/VOSR2/args.json` |
| `VOSR2/checkpoints/ema_model.safetensors` | [link](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/checkpoints/ema_model.safetensors) | `models/vosr2/VOSR2/checkpoints/ema_model.safetensors` |
| `Qwen-Image-vae-2d/config.json` | [link](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/config.json) | `models/vae/VOSR2/Qwen-Image-vae-2d/config.json` |
| `Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors` | [link](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors) | `models/vae/VOSR2/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors` |
| `torch_cache/checkpoints/dinov2_vitl14_pretrain.pth` | [link](https://huggingface.co/CSWRY/VOSR/resolve/main/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth) | convert → `models/clip_vision/VOSR2/dinov2_vitl14.safetensors` (see below) |

> The `model` / `vae` / `vision_encoder` dropdowns list the **folder or file
> name** shown above (`VOSR2`, `Qwen-Image-vae-2d`, `dinov2_vitl14.safetensors`).
> You can rename the `vosr2/VOSR2` and `vae/VOSR2/Qwen-Image-vae-2d` folders to
> anything you like — the loader discovers any subfolder that has the right
> files inside.

### Convert the DINOv2-L checkpoint to safetensors

The vision encoder in the upstream repo is a raw PyTorch pickle
(`dinov2_vitl14_pretrain.pth`). This node only loads `.safetensors`. Convert it
once — run this from a shell where ComfyUI's Python can `import torch`:

```python
import torch
from safetensors.torch import save_file

sd = torch.load("dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, "dinov2_vitl14.safetensors")
```

Then move `dinov2_vitl14.safetensors` to `models/clip_vision/VOSR2/`.

The key names in this checkpoint are the original Meta `facebookresearch/dinov2`
names (`blocks.N.attn.qkv.*`, `blocks.N.ls1.gamma`, …), which is exactly what the
vendored architecture in [`models/dinov2.py`](models/dinov2.py) expects — do
**not** use the Hugging Face `transformers` `facebook/dinov2-large` weights, whose
keys differ.

### One-shot download (optional)

With the `huggingface_hub` CLI:

```bash
huggingface-cli download CSWRY/VOSR \
  VOSR2/args.json \
  VOSR2/checkpoints/ema_model.safetensors \
  Qwen-Image-vae-2d/config.json \
  Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors \
  torch_cache/checkpoints/dinov2_vitl14_pretrain.pth \
  --local-dir ./vosr2_download
```

then move each file into place and run the conversion step above.

---

## Installation

### Local ComfyUI

You need a reasonably recent ComfyUI (one that ships `comfy_api.latest` — any
build from 2025 onward). No extra Python packages are required: the only imports
beyond `torch` are `safetensors` and `einops`, both of which already ship with
ComfyUI.

**Option A — ComfyUI-Manager (recommended)**

1. Open **Manager → Custom Nodes Manager → Install via Git URL**.
2. Paste `https://github.com/ylchen333/ComfyUI-VOSR2` and confirm.
3. Restart ComfyUI.

**Option B — git clone**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ylchen333/ComfyUI-VOSR2
# restart ComfyUI
```

Then place the model files as described in [Model files](#model-files) and
restart once more so the loader picks them up.

### RunComfy

Custom nodes are not yet published on Comfy.org, so on RunComfy you install this
package the same way you would any GitHub node — but it must be on a
**persistent / dedicated workspace**, because custom nodes and uploaded models do
not survive a session on the free shared machines.

1. **Start a dedicated ComfyUI workspace** on RunComfy and pick a recent ComfyUI
   version.
2. **Install the node.** In the running ComfyUI, open
   **Manager → Install via Git URL** and paste
   `https://github.com/ylchen333/ComfyUI-VOSR2`, then restart the ComfyUI process
   from the Manager. (If you have terminal access to the workspace you can
   instead `git clone` it into `ComfyUI/custom_nodes/` as above.)
3. **Add the model files.** Using RunComfy's file browser / storage manager for
   the workspace, create the folders and upload the files exactly as in
   [Model files](#model-files):
   - `ComfyUI/models/vosr2/VOSR2/…`
   - `ComfyUI/models/vae/VOSR2/Qwen-Image-vae-2d/…`
   - `ComfyUI/models/clip_vision/VOSR2/dinov2_vitl14.safetensors`

   Do the `.pth → .safetensors` conversion locally first (the workspace terminal
   works too if you have it), then upload the resulting `.safetensors`.
4. **Restart ComfyUI** and add the **VOSR 2.0 Model Loader** node — the three
   dropdowns should now be populated.

> If your RunComfy plan does not expose a git-install option or a writable
> `models/` tree, you will need to request the node/models through their support
> channel; this package cannot self-install or download anything at runtime.

---

## Usage

1. **VOSR 2.0 Model Loader** — pick `model` = `VOSR2`, `vae` =
   `Qwen-Image-vae-2d`, `vision_encoder` = `dinov2_vitl14.safetensors`, leave
   `dtype` on `default` (use `fp16`/`bf16` to save VRAM).
2. Feed an image into **VOSR 2.0 Upscale** together with the loader's `model`
   output.
3. Set `upscale` (1–4). If the result exceeds 512 px on a side, set
   `tile_size = 512`. If it exceeds ~1024 px, also set `vae_tile_size = 1024`.
4. `color_alignment` defaults to `wavelet`; `adain` or `none` are available for
   comparison.

### Recommended VOSR 2.0 Upscale settings

A solid general-purpose starting point (tiling on, for outputs past 512 px):

| Parameter | Value |
|---|---|
| `upscale` | `4` |
| `seed` | `42` |
| `color_alignment` | `wavelet` |
| `tile_size` | `512` |
| `tile_overlap` | `64` |
| `vae_tile_size` | `1024` |
| `vae_tile_overlap` | `128` |

The node's own defaults keep tiling **off** (`tile_size` / `vae_tile_size` = `0`),
which is only appropriate when the upscaled output stays at or below 512 px — set
the values above for anything larger.

Batches are first-class: item *i* is seeded with `seed + i`, so a batch result
matches running each image separately.

---

## Notes on VRAM and tiling

The upstream authors have no measured tile-size / VRAM table yet, and neither
does this package. Treat these as starting points, not guarantees:

- ≤ 512 px output: no tiling needed.
- ~2048 px output: `tile_size = 512`.
- ~4096 px output: `tile_size = 512`, `vae_tile_size = 1024`.

`fp16` / `bf16` (via the loader's `dtype`) roughly halves DiT + vision-encoder
memory; the Qwen VAE always runs in fp32.

---

## Status

In-progress v1. Done: local discovery/validation, vendored DiT/VAE/DINOv2
architectures, ComfyUI-managed load/offload via `ModelPatcher`, untiled and tiled
(DiT + VAE) inference, per-item seeded batching, and torch-native
`wavelet` / `adain` / `none` color alignment. Not yet done: numeric validation
against the upstream reference, measured VRAM/tile presets, and example workflow
JSON.

## Licensing

This node package is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)), matching upstream VOSR.

The **model weights are downloaded separately** from
[CSWRY/VOSR](https://huggingface.co/CSWRY/VOSR) and carry their own terms —
including the DINOv2 checkpoint (Meta, `facebookresearch/dinov2`) and the
Qwen-Image VAE. Review those before redistribution or commercial use.

## Credits

- **VOSR / VOSR 2.0** — Rongyuan Wu et al. ([cswry/VOSR](https://github.com/cswry/VOSR))
- **DINOv2** — Meta AI ([facebookresearch/dinov2](https://github.com/facebookresearch/dinov2))
- **Qwen-Image VAE** — Qwen team
- ComfyUI integration: this repository
