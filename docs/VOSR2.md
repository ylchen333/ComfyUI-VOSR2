# VOSR2 ComfyUI Node Specification

Status: implementation proposal  
Scope: VOSR 2.0 one-step 1.4B image super-resolution  
Upstream: `C:\Users\lorie\Documents\projects\ComfyUI\VOSR`  
License: Apache-2.0; downloaded model assets may have separate terms

## Author confirmation (Rongyuan Wu, 2026-09-01)

Contacted directly; the following are now confirmed rather than assumed, and shape several decisions below:

- **Distribution:** standalone community node package (author's stated reference point: SeedVR's approach), not a PR into the upstream repo. Matches this spec's existing direction.
- **Scope:** "VOSR 2.0" is a distinct, newer one-step 1.4B checkpoint that supersedes "VOSR 1" (the original release: 0.5B/1.4B, multi-step + one-step distilled). Prioritize VOSR 2.0 exclusively — matches this spec's existing "V1 supports only VOSR2" scoping.
- **`cfg_scale` and `weak_cond_strength_aelq` confirmed to have no effect** on VOSR 2.0 inference and should not be exposed. Matches this spec's existing exclusion of these fields (and of `infer_steps`) — can now be documented as author-confirmed rather than inferred from reading `vosr.py`.
- **Tiling is not optional:** VOSR 2.0 was natively trained at up to 512px, so tiled inference is required for anything larger. Confirms the emphasis already in this spec's Tiling section.
- **No existing tile-size/VRAM recommendations exist upstream.** The author explicitly asked for measured recommendations once available. Treat the VRAM/tile-preset work in "Device, dtype, and memory" and "Prototype questions" below as a deliverable to report back to him, not only internal QA.
- **Batch-processing demand is unknown upstream** — the author asked for a recommendation. This spec's "batch as first-class input" design (see Nodes, Tests) can be relayed directly as the answer, backed by the seed/batch test coverage already specified.
- **Still open, not resolved by his reply:** he confirmed VOSR 2.0 as conceptually distinct from VOSR 1's one-step 1.4B (`VOSR_1.4B_os`) and said to "use the name VOSR 2.0," which is consistent with the locally downloaded folder being named `VOSR2` rather than `VOSR_1.4B_os` — but he did not confirm the literal folder/file naming. **Verify the actual downloaded folder name before finalizing the "Model layout and discovery" paths below**; treat `vosr2/VOSR2-1.4B/` there as provisional.
- **New open issue from testing, not yet root-caused:** closely-inspected enlarged faces can show grid-like artifacts. The author called this out as more serious than the general softness/limited-generative-detail complaint that's otherwise the most common community feedback. Worth an explicit visual-inspection checkpoint during validation (see Tests) rather than assuming it's covered by general quality comparison against upstream.
- **Naming, display vs. code:** he asked specifically to use "VOSR 2.0" (with the space and the period) rather than "VOSR 1" in how the integration is presented. Code identifiers, folder names, and node categories below stay as `VOSR2` / `ComfyUI-VOSR2` (spaces and periods aren't practical there), but human-facing strings — node display names, README title, Registry listing name — should read "VOSR 2.0," not "VOSR2." Called out explicitly under Nodes below.

## Goal

Create a native ComfyUI extension that accepts an `IMAGE` batch and returns a VOSR2-restored `IMAGE` batch. It must use local model files, preserve upstream inference behavior, work with ComfyUI memory management, and avoid CLI/file-based image I/O.

V1 supports only VOSR2: the one-step 1.4B LightningDiT, 16-channel Qwen Image 2D VAE, and DINOv2-L layer-17 conditioning. Training, VOSR 0.5B/1.x, multi-step sampling, SD2, and the lightweight SD2 decoder are out of scope.

## Nodes

Use separate loading and execution nodes so queued images do not reconstruct several GB of models.

Display names should read **"VOSR 2.0"** (per the author's explicit naming preference), even though code identifiers, categories, and folder names use the space/period-free `VOSR2` — e.g. `VOSR2ModelLoader`'s ComfyUI display name is "VOSR 2.0 Model Loader," `VOSR2Upscale`'s is "VOSR 2.0 Upscale."

### `VOSR2ModelLoader`

Category: `image/upscaling/VOSR2`  
Output: private type `VOSR2_MODEL`

| Input | Type/default | Contract |
|---|---|---|
| `model` | combo | VOSR2 bundle discovered under `ComfyUI/models/vosr2` |
| `vae` | combo | Qwen 2D VAE discovered under `ComfyUI/models/vae/VOSR2` |
| `vision_encoder` | combo | DINOv2-L checkpoint under `ComfyUI/models/clip_vision/VOSR2` |
| `dtype` | `default` | `default`, `fp16`, or `bf16`; default follows ComfyUI policy |

The loader returns a runtime bundle containing the config and independently manageable DiT, VAE, and vision-encoder components. It never downloads files.

### `VOSR2Upscale`

Category: `image/upscaling/VOSR2`  
Input/output: ComfyUI `IMAGE`, `BHWC` RGB float `[0,1]`

| Input | Default | Range/options | Meaning |
|---|---:|---|---|
| `model` | required | `VOSR2_MODEL` | Loaded runtime bundle |
| `image` | required | `IMAGE` | Image or batch |
| `upscale` | `4` | integer `1..4` | Exact output multiplier |
| `seed` | `42` | ComfyUI seed range | Initial latent-noise seed |
| `color_alignment` | `wavelet` | `wavelet`, `adain`, `none` | Upstream postprocessing mode |
| `tile_size` | `0` | `0..4096`, step 64 | DiT pixel tile; zero disables |
| `tile_overlap` | `32` | `0..512`, step 8 | DiT overlap in pixels |
| `vae_tile_size` | `0` | `0..8192`, step 64 | VAE pixel tile; zero disables |
| `vae_tile_overlap` | `32` | `0..512`, step 8 | VAE overlap in pixels |

Do not expose `infer_steps` in v1. VOSR2 is released as a one-step model; arbitrary values imply an unsupported quality control. **`cfg_scale` and `weak_cond_strength_aelq` are likewise excluded — author-confirmed (2026-09-01) to have no effect on VOSR 2.0 inference**, not merely inferred from the reference implementation.

## Model layout and discovery

Register a `vosr2` model folder with `folder_paths.add_model_folder_path`.

```text
ComfyUI/models/
|-- vosr2/VOSR2-1.4B/
|   |-- args.json
|   `-- checkpoints/ema_model.safetensors
|-- vae/VOSR2/Qwen-Image-vae-2d/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
`-- clip_vision/VOSR2/
    `-- dinov2_vitl14.safetensors
```

⚠️ **Open item:** confirm whether the actual downloaded checkpoint folder is named `VOSR2` or something else (e.g. `VOSR_1.4B_os`) before treating `vosr2/VOSR2-1.4B/` above as final. The author's reply resolved the conceptual "VOSR 2.0 vs. VOSR 1" distinction but not the literal folder string — check the real download before locking this path in code or documentation.

Resolve combo values with `folder_paths` containment-safe helpers; workflows must not supply arbitrary paths. Weight selection is deterministic: prefer `clean_weights/ema_model.safetensors`, then `checkpoints/ema_model.safetensors`, then root `ema_model.safetensors`. Never load the first recursive safetensors match.

Validate `args.json` before construction:

- `ae_type == "qwen"`
- `dim == 1536`, `depth == 36`, `num_heads == 24`
- `patch_size == 2`, `enc_type == "dinov2l"`, `enc_dim == 1024`
- `layer_dinov2b_list == [17]`
- `auxiliary_time_cond == false`
- `distill_type == "onestep"`

Load state dicts strictly after stripping only documented prefixes. If known training-only keys exist, filter a fixed allowlist and reject all other missing/unexpected keys. Do not retain upstream `strict=False` behavior.

## Inference contract

For each image:

1. Compute exact target dimensions as input width/height times `upscale`.
2. Bicubic-resize to the target dimensions. This matches upstream behavior: scaling occurs before VAE/DiT restoration.
3. Convert `BHWC [0,1]` to `BCHW [-1,1]` on the compute device.
4. Right/bottom pad to a multiple of 16 (VAE factor 8 times DiT patch 2); remember the exact target size.
5. VAE-encode using `latent_dist.mode()`, retaining upstream Qwen latent mean/std normalization.
6. Build vision conditioning: bicubic resize to 448x448, ImageNet mean/std normalize, run DINOv2-L, and select normalized layer-17 patch tokens.
7. Create noise with a local `torch.Generator`. Item `i` uses `seed + i`; do not alter global PyTorch or NumPy RNG state.
8. Perform exactly one flow update, `t=1` to `t=0`, using `LightningDiT.forward_flexible`.
9. Decode, crop to exact target size, map to `[0,1]`, and apply color alignment against the bicubic target.
10. Return `BHWC` float on ComfyUI's intermediate/output device without PIL or uint8 quantization.

Same-sized batch items may be vectorized. Tiled mode may process items independently to cap peak memory.

## Tiling

Port upstream latent tiling without changing its math:

- Convert pixel tile size by VAE factor 8 and align it to patch size 2.
- Require overlap smaller than tile size after conversion; use the upstream minimum overlap of one eighth tile.
- Generate one full noise field and slice it so overlaps share identical noise.
- Compute DINOv2 features for each matching pixel tile.
- Gaussian-blend tile velocity predictions, then apply the one global flow update.
- If the latent fits one tile, use the untiled path.

Port the Qwen-specific VAE encode/decode tiling and Gaussian blending from `tiled_vae.py`. DiT and VAE tiling remain independent. Pad before tiling and crop only after decoding.

## Color alignment

Implement in torch to preserve float precision:

- `none`: decoded result unchanged.
- `adain`: match channel-wise spatial mean/std to the bicubic input, epsilon `1e-5`.
- `wavelet`: combine bicubic input low frequencies with decoded output high frequencies, matching upstream Gaussian sigma 5.

Prefer an existing local ComfyUI/torch blur operation. Otherwise add a small separable Gaussian implementation, not OpenCV. Clamp once at the end.

## Device, dtype, and memory

- Use `comfy.model_management.get_torch_device()` and ComfyUI dtype/offload policy; never hard-code CUDA.
- Keep tensor code device-neutral for CPU, ROCm, XPU, and other supported devices where operations work.
- Give DiT, VAE, and DINOv2 ComfyUI-compatible loading/offload wrappers so only the active phase must be resident.
- Load checkpoint tensors on the offload device and avoid duplicate GPU copies.
- Adapt DiT attention to ComfyUI optimized attention rather than choosing a private backend.
- Use existing ComfyUI/Comfy Kitchen operations when their exact math/layout contract matches.
- Do not add `torch.no_grad()` or `torch.inference_mode()`; ComfyUI owns inference policy.
- Do not synchronize CUDA or call `torch.cuda.empty_cache()` from the node.
- Scope DINO features, tile accumulators, VAE statistics, and noise to one execution.

Actual minimum VRAM and recommended tiles must be measured during implementation, not guessed in the UI. **This measurement is now an explicit, named ask from the upstream author (2026-09-01), not only internal QA — write up tile-size/VRAM/quality tradeoffs after implementation and send them back to him.**

## Package structure

```text
ComfyUI-VOSR2/
|-- __init__.py
|-- nodes.py
|-- loader.py
|-- inference.py
|-- color.py
|-- tiled_vae.py
|-- models/
|   |-- lightningdit.py
|   |-- pos_embed.py
|   |-- qwenimage_vae2d.py
|   |-- rmsnorm.py
|   `-- swiglu_ffn.py
|-- requirements.txt
|-- pyproject.toml
|-- examples/
|   |-- vosr2_basic.json
|   `-- vosr2_basic_api.json
|-- LICENSE
`-- README.md
```

Vendor only runtime code required by VOSR2. Remove training losses, datasets, Accelerate stubs, CLI parsing, filesystem image loops, progress bars, W&B, evaluation code, SD2 support, and multi-step paths.

Use the supplied `comfy_api.latest` API: both nodes subclass `io.ComfyNode`, define `io.Schema`, and register through one `ComfyExtension`/`comfy_entrypoint`. No routes or frontend JavaScript are needed.

The package must also include Comfy Registry metadata in `pyproject.toml` (`name`, `version`, description, license, repository URL, dependencies, and `[tool.comfy]`). Keep `requirements.txt` and `project.dependencies` synchronized. Ship one editable UI workflow and one API-format workflow; the ComfyCV examples demonstrate why both forms are useful to local users and programmatic clients.

The runtime bundle should expose narrow operations such as `encode`, `vision_features`, `denoise_one_step`, and `decode`. The node must not probe child attributes for capabilities.

## Registry & ComfyUI-Manager publishing (researched 2026-09-01)

Two distinct mechanisms exist; use the first as primary, don't rely on the second:

**Comfy Registry (registry.comfy.org) — the modern, self-service path, and what this spec's `pyproject.toml` requirement already targets:**
1. Create a Registry account and a **Publisher ID** (unique identity after the `@` in your profile).
2. Generate an **API key** for that publisher at registry.comfy.org/nodes.
3. Run `comfy node init` to scaffold `pyproject.toml` with the required fields: `name` (immutable, unique), `description`, `version` (semver), `PublisherId`, `DisplayName` (this is where "VOSR 2.0" as a display string, per the author's naming request, actually belongs), `Icon` (SVG/PNG/JPG/GIF, up to 800x400), `Repository` URL, and `dependencies` (populated from `requirements.txt`).
4. Publish either manually (`comfy node publish`, prompts for the API key; packages all git-tracked files by default — use a `.comfyignore` to exclude tests/dev assets the way `.gitignore` works) or automatically via CI: store the API key as a `REGISTRY_ACCESS_TOKEN` repo secret and add `Comfy-Org/publish-node-action` as a GitHub Actions workflow that fires on `pyproject.toml` changes to `main` — every version bump auto-republishes.
5. **The Registry is described by ComfyUI's own docs as what now powers ComfyUI-Manager's node discovery** — publishing here is meant to make the node automatically installable through Manager without a separate submission.

**ComfyUI-Manager's `custom-node-list.json` — an older, manual PR-based path:**
Separately, ComfyUI-Manager's own docs describe adding a node by submitting a pull request against `Comfy-Org/ComfyUI-Manager` editing `custom-node-list.json` directly, with `requirements.txt`/`install.py` as the expected lifecycle files and a note to keep dependency version constraints as loose as possible to avoid conflicts with other installed nodes.

⚠️ These two descriptions come from two different official doc pages and read as slightly inconsistent about how central the manual PR route still is — worth a direct check of the current `docs.comfy.org` pages (or asking in the ComfyUI dev Discord/server, where this whole collaboration started) at publish time to confirm whether a `custom-node-list.json` PR is still a required or recommended second step in addition to Registry publishing, or whether it's now legacy. Don't assume either doc page is stale without checking — this is exactly the kind of thing that changes between when this was researched and when VOSR2 actually ships.

## Dependencies

Do not copy upstream `requirements.txt`: it pins PyTorch/CUDA and contains many training-only packages that could break ComfyUI.

Target zero new dependencies. Use ComfyUI/PyTorch equivalents for Accelerate, einops, OpenCV, torchvision transforms, and timm. If the Qwen VAE cannot be cleanly adapted to native ComfyUI code, the only candidate addition is a ComfyUI-compatible `diffusers` version after checking the installed environment.

DINOv2 must be constructed locally and loaded from the chosen checkpoint. Never call `torch.hub.load`; it can execute cached repository code or access the internet.

## Reference integration review

The following working integrations were inspected on 2026-08-31:

- [CreativeInquiry/ComfyCV `lorie`](https://github.com/CreativeInquiry/ComfyCV/tree/main/lorie)
- [ylchen333/ComfyUI-LocateAnything](https://github.com/ylchen333/ComfyUI-LocateAnything)
- [ylchen333/ComfyUI-YOLO](https://github.com/ylchen333/ComfyUI-YOLO)

They are implementation references, not code dependencies. Adopt these demonstrated patterns:

| Pattern | Reference evidence | VOSR2 decision |
|---|---|---|
| Loader node returns a private model type | LocateAnything returns `LOCATEANYTHING_MODEL`; YOLO returns `ULTRALYTICS_MODEL` | Keep `VOSR2_MODEL`, separate from execution |
| Models live below `folder_paths.models_dir` | LocateAnything registers `models/LocateAnything`; YOLO registers `models/ultralytics` | Register `models/vosr2`; use `get_filename_list` and safe full-path resolution |
| Comfy images cross the library boundary explicitly | YOLO permutes Comfy `BHWC` tensors to library `BCHW` | Centralize VOSR2 layout/range conversion in `inference.py` |
| Image batches are first-class inputs | YOLO contains separate batch handling; ComfyCV ships batched-video workflows | Define stable per-item seed semantics and test batches directly |
| Package metadata belongs in `pyproject.toml` | LocateAnything and YOLO declare Registry metadata and dependencies | Add a complete `pyproject.toml`; do not rely only on README installation steps |
| Examples include executable workflow JSON | LocateAnything ships a basic workflow; ComfyCV pairs UI and API JSON workflows | Ship minimal UI and API VOSR2 workflows |
| Optional help UI is isolated under `web/` | LocateAnything exports `WEB_DIRECTORY` for its help popup | Omit web assets in v1 because VOSR2 needs no custom UI behavior |

### Round 2 (2026-09-01): SeedVR2 and HYPIR — the author's own reference point, plus a closer size analog

Rongyuan named **SeedVR2** (`numz/ComfyUI-SeedVR2_VideoUpscaler`) directly as the distribution model to follow, so it's now a primary reference, not just a nice-to-have. Also reviewed **HYPIR-ComfyUI** (`EricRollei/HYPIR-ComfyUI`) as a closer size/complexity analog — a single diffusion-based restoration model, not a large video model like SeedVR2.

| Pattern | Reference evidence | VOSR2 consideration |
|---|---|---|
| Split loader nodes per component, not one combined loader | SeedVR2 has separate **Load DiT**, **Load VAE**, and (optional) **Torch Compile Settings** nodes feeding one execution node, enabling a "global model cache" shared across multiple upscaler instances | **Open question, not yet decided:** this spec currently has one `VOSR2ModelLoader` covering DiT+VAE+vision-encoder via three combo widgets. SeedVR2's split lets a user swap just the VAE or just the DiT without reloading everything else, and matches the exact pattern the author pointed to. Worth deciding deliberately whether to split `VOSR2ModelLoader` into `VOSR2DiTLoader` + `VOSR2VAELoader` (+ vision-encoder loader) before locking the Nodes section, rather than defaulting to the simpler combined loader just because it's already written here. |
| Per-component, independent offload targets | SeedVR2: DiT, VAE, and intermediate tensors each independently configurable to GPU / CPU / secondary GPU; "BlockSwap" streams transformer blocks between GPU/CPU for large models | This spec's existing "give DiT, VAE, and DINOv2 ComfyUI-compatible loading/offload wrappers so only the active phase must be resident" already points the same direction — SeedVR2 confirms this is the expected level of granularity for a Registry-quality node, not over-engineering. |
| Local imports to avoid ComfyUI environment conflicts | SeedVR2 explicitly documents "local imports prevent conflicts with other ComfyUI custom nodes" and graceful fallback chains for optional deps (Flash/Sage attention → PyTorch SDPA) | Matches this spec's "target zero new dependencies" / vendor-only-what's-needed stance under Dependencies. Adopt the fallback-chain idea specifically for anything optional (e.g. if a faster attention backend is available, use it; otherwise fall back cleanly rather than hard-requiring it). |
| Minimal single-node pattern for a smaller model | HYPIR-ComfyUI ships one execution node (`HYPIR Image Restore`), model file discovered via a dropdown scan of a `models/` folder, tiled VAE decode with user-adjustable tile/stride | Useful lower bound — if the DiT/VAE loader split above turns out to be more complexity than VOSR2 needs at this size (1.4B, single model family, no video/temporal dimension like SeedVR2), HYPIR's simpler shape is the fallback to compare against. |
| Registry metadata plus a `node_list.json` for non-standard registration | HYPIR ships both `pyproject.toml` and `node_list.json` | Confirms this spec's existing `pyproject.toml` requirement; `node_list.json` only needed if VOSR2's nodes don't register through the standard `NODE_CLASS_MAPPINGS`/`comfy_api.latest` pattern already specified here — shouldn't be needed given the extension approach already chosen. |

Do not copy these reference behaviors:

- LocateAnything uses `from_pretrained(..., trust_remote_code=True)` and may resolve remote models. VOSR2 must load audited local files only.
- YOLO contains an HTTP model downloader. VOSR2 must never download during node import, validation, or execution.
- Both examples expose explicit device choices or CUDA-oriented paths. VOSR2 delegates device selection to `comfy.model_management` and does not offer a manual device widget.
- LocateAnything manually calls `torch.cuda.empty_cache()`. VOSR2 leaves cache and unload policy to ComfyUI.
- The examples use legacy `NODE_CLASS_MAPPINGS`. VOSR2 uses the supplied `comfy_api.latest` extension entrypoint unless compatibility testing proves the installed ComfyUI version requires a legacy registration shim. Do not maintain two registration systems without that concrete requirement.
- Their third-party libraries justify broader dependencies such as Transformers, Accelerate, Ultralytics, OpenCV, and Requests. Those packages are not evidence that VOSR2 needs them.

The ComfyCV `lorie` directory is primarily workflow/documentation material rather than another VOSR2-like loader implementation. Its most relevant contribution is deployment practice: keep UI workflow JSON distinct from API-format JSON, document model placement, and test batch-oriented workflows as users actually run them.

## Errors

Fail with short actionable messages for missing component files, an incompatible VOSR2 config, malformed state dicts, invalid image channels, invalid tile overlap, impractical tensor dimensions, or unsupported device/dtype operations. Never continue with a partially loaded model.

## Tests

Unit coverage:

- layout/range conversion round-trip
- pad-to-16 and exact crop for odd sizes
- stable local noise for identical seeds and distinct batch-index noise
- AdaIN statistics and wavelet range
- complete tile-grid coverage and nonzero blend denominators
- safe path resolution and deterministic checkpoint selection
- VOSR2 config signature rejection/acceptance

Integration coverage:

- load all supplied assets with network disabled
- compare a fixed image/seed with upstream using `color_alignment=none`
- compare float wavelet output before upstream PNG quantization
- exact 1x, 2x, 3x, and 4x dimensions for odd inputs
- batch result equals separate item runs under the defined seed rule
- tiled and untiled outputs are numerically/visually close (not necessarily bit-identical)
- repeated queues do not reconstruct models
- ComfyUI unload/reload returns every component to the proper device
- **visually inspect enlarged/upscaled face regions for grid-pattern artifacts across a few test images** — author-flagged (2026-09-01) as a known, more-serious-than-usual open issue currently under investigation upstream; document whether reproduced locally, and don't treat a clean result on one test image as resolving it

Measure time and peak VRAM for 512, 2048 (DiT tile 512), and 4096 targets (DiT 512, VAE 1024), for fp16/bf16 only where hardware supports them.

## Acceptance criteria

V1 is complete when:

1. Both nodes load without custom routes or frontend code.
2. All components load only from configured local model folders.
3. An image batch returns exact-size results for every exposed scale.
4. Identical image/settings/seed reproduce identical output.
5. Fixed untiled output matches upstream within an agreed pre-encoding tolerance.
6. Both tiling modes have full coverage, no obvious seams, and no persistent tensor cache.
7. ComfyUI can offload/reload components without reconstructing them.
8. No path downloads, uses `torch.hub`, writes output images, changes global RNG, or replaces the user's PyTorch installation.

## Implementation sequence

1. Scaffold extension and safe local model discovery.
2. Reduce/vendor VOSR2-only model definitions and strict loaders.
3. Add ComfyUI model/offload wrappers.
4. Implement untiled float-tensor inference and validate against upstream.
5. Add seed/batch semantics and odd-size padding/cropping.
6. Add torch-native color alignment.
7. Add DiT tiling, then VAE tiling, validating each separately.
8. Add tests, measured VRAM presets, attribution, and a minimal workflow.

## Prototype questions

- Can ComfyUI's native Qwen Image VAE load the supplied 2D-extracted checkpoint, or is a small adapter required?
- Does an existing ComfyUI DINOv2 implementation expose normalized layer-17 patch tokens in the required shape?
- Which current model-patcher wrapper best handles three sequential, independently offloadable components?
- Which ComfyUI/Comfy Kitchen attention and norm operations exactly match VOSR2 in fp16 and bf16?
- What tile presets fit common 8, 12, 16, and 24 GB GPUs based on measured peak allocation? **(Feeds the tile-size/VRAM recommendations the author explicitly asked for — see "Author confirmation" above.)**