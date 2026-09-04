# VOSR 2.0 Upscale

One-step super-resolution on an `IMAGE` batch using a `VOSR2_MODEL` from the
**VOSR 2.0 Model Loader**.

## Inputs

- **model** — A `VOSR2_MODEL` from the loader.
- **image** — A single image or batch, RGB (3-channel).
- **upscale** — Exact output multiplier, `1`–`4`.
- **seed** — Latent-noise seed. Each batch item *i* uses `seed + i`, so results
  stay stable per-image as batch size changes.
- **color_alignment** — Post-process the model output against the bicubic
  target: `wavelet` (default), `adain`, or `none`.
- **tile_size** / **tile_overlap** — DiT pixel tile size and overlap. `0`
  disables tiling.
- **vae_tile_size** / **vae_tile_overlap** — VAE pixel tile size and overlap.
  `0` disables VAE tiling and decodes the whole image in one pass.

## Tiling is not optional above 512 px

VOSR 2.0 was trained natively at up to 512 px. **Whenever the upscaled output
exceeds 512×512, set `tile_size`** (e.g. `512`) — otherwise quality visibly
degrades. This isn't a performance knob at that resolution, it's required for
correct output.

For outputs well past 1024 px, also set **`vae_tile_size`** (e.g. `1024`).
Leaving it at `0` decodes the full image through the VAE in a single pass
regardless of `tile_size`, which is the most common source of CUDA
out-of-memory errors at large output sizes.

## Notes

- The node warns in the console log when the target resolution exceeds these
  thresholds with tiling disabled, but does not force tiling on automatically
  — you may still want untiled output for small images, or have a GPU with
  enough VRAM to skip VAE tiling at a given size.
- `tile_overlap` must be smaller than `tile_size` (and likewise for the VAE
  pair); the node validates this before running.
