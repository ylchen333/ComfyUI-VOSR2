# VOSR 2.0 Model Loader

Loads a VOSR 2.0 (one-step, 1.4B) bundle — a matched **LightningDiT + Qwen-Image
2D VAE + DINOv2-L** triple — and returns a `VOSR2_MODEL` for the **VOSR 2.0
Upscale** node.

VOSR 2.0 is a fixed set: the DiT only works with the exact Qwen 2D VAE it was
trained against, and its conditioning encoder is a specific DINOv2-L layer. The
VAE and vision encoder are therefore not separate inputs — they live inside the
same bundle folder as the DiT weights, under
`ComfyUI/models/vosr2/<bundle>/`.

## Inputs

- **model** — Bundle folder name under `models/vosr2/`. Leave it on `VOSR2`
  (the confirmed bundle) unless you've added your own. On first use, any part
  missing from that folder is downloaded from the pinned
  [`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR) Hugging Face repo
  (~7 GB total); once the files are present, nothing here touches the network.
- **dtype** — Compute dtype for the DiT and vision encoder:
  - `default` — follows ComfyUI's own device/dtype policy. Safe on any
    hardware; use this unless you have a specific reason not to.
  - `fp16` / `bf16` — forces that precision. Only pick these if your GPU
    actually supports them — `bf16` in particular needs an Ampere-class (RTX
    30-series/A-series) or newer NVIDIA GPU, or an equivalent ROCm/other
    device. On unsupported hardware this can silently degrade output quality
    or fail deep inside the model rather than at load time.
  - The VAE always runs in fp32 regardless of this setting — its latent space
    is precision-sensitive, so this isn't configurable.

## Notes

- Split from the upscale node on purpose: the loader builds and validates
  several GB of weights once, so a queue of images doesn't rebuild them per
  item.
- `args.json` is validated against the fixed VOSR 2.0 architecture before
  anything is constructed, and checkpoint loading is strict — an incompatible
  or malformed bundle fails with a clear error instead of loading partially.
- Offline / air-gapped installs can place the bundle files by hand instead of
  relying on the automatic download — see the package README's "Model files"
  section for the exact layout and direct download links.
