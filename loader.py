"""Discovery and strict loading of a VOSR 2.0 bundle.

A bundle is a self-contained folder under ``ComfyUI/models/vosr2/`` -- VOSR 2.0
is a fixed DiT/VAE/DINOv2 triple (see the layout comment below), so the three
pieces are not independently selectable and live together:

    ComfyUI/models/vosr2/<bundle>/
        args.json
        checkpoints/ema_model.safetensors            (or clean_weights/, or bundle root)
        Qwen-Image-vae-2d/{config.json, diffusion_pytorch_model.safetensors}
        dinov2_vitl14.safetensors

The confirmed bundle is fetched from the pinned Hugging Face repo ``CSWRY/VOSR``
on first use, but only for parts that are missing -- once the files are in place
nothing here touches the network. See ``ensure_vosr2_files``.

The ``model`` combo value is re-joined against ``_VOSR2_ROOT`` via
``_safe_child_dir``, never taken as -- or resolved through -- an arbitrary path.
"""
import json
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import comfy.model_management
import comfy.model_patcher
import folder_paths

from .models.dinov2 import build_dinov2_vitl14
from .models.lightningdit import LightningDiT
from .models.qwenimage_vae2d import AutoencoderKLQwenImage2D

VOSR2_FOLDER_KEY = "vosr2"

_VOSR2_ROOT = Path(folder_paths.models_dir) / "vosr2"
_VOSR2_ROOT.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path(VOSR2_FOLDER_KEY, str(_VOSR2_ROOT))

# VOSR 2.0 is one indivisible artifact: the DiT runs entirely in *this* Qwen 2D
# VAE's latent space (its channel count, scale factor, and per-channel
# latents_mean/std), and is conditioned on *this* DINOv2-L feature extractor.
# None of the three can be swapped independently, so they live together inside a
# single bundle folder rather than in ComfyUI's shared vae/ and clip_vision/
# trees. A bundle is a subdirectory of models/vosr2/ laid out as:
#     <bundle>/args.json
#     <bundle>/checkpoints/ema_model.safetensors      (or clean_weights/, or bundle root)
#     <bundle>/Qwen-Image-vae-2d/{config.json,diffusion_pytorch_model.safetensors}
#     <bundle>/dinov2_vitl14.safetensors
HF_REPO_ID = "CSWRY/VOSR"
KNOWN_MODEL = "VOSR2"           # the one confirmed bundle; also the combo default
_VAE_SUBDIR = "Qwen-Image-vae-2d"       # diffusers VAE dir inside the bundle
_VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
_VISION_FILENAME = "dinov2_vitl14.safetensors"  # converted DINOv2-L, inside the bundle

# Paths of each piece inside the pinned HF repo. The DiT files carry a "VOSR2/"
# prefix that becomes the bundle dir; the VAE files are fetched into the bundle.
_DIT_HF_FILES = ("VOSR2/args.json", "VOSR2/checkpoints/ema_model.safetensors")
_VAE_HF_FILES = tuple(f"{_VAE_SUBDIR}/{name}" for name in _VAE_FILES)
# DINOv2-L ships upstream as a raw torch pickle; the loader converts it to
# safetensors on download (this package only ever ``load_file``s .safetensors).
_DINOV2_HF_FILE = "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth"

# Required args.json values for a VOSR 2.0 (one-step 1.4B) checkpoint. See
# VOSR2.md "Author confirmation" -- these fields are confirmed to have no
# effect, or to be architecturally fixed, for this specific release.
REQUIRED_ARGS = {
    "ae_type": "qwen",
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "patch_size": 2,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "auxiliary_time_cond": False,
    "distill_type": "onestep",
}

# DiT construction params that must be present in args.json but whose value is
# taken from the checkpoint rather than pinned by REQUIRED_ARGS.
DIT_ARG_KEYS = ("mlp_ratio", "use_qknorm", "use_swiglu", "use_rope", "use_rmsnorm", "encdim_ratio", "resolution")

# Known training-only key patterns that may be present in an EMA/distillation
# checkpoint but have no corresponding module in the inference-only LightningDiT
# above (e.g. an EMA wrapper's step counter). Extend this list, don't relax
# strict loading, if a real checkpoint surfaces another such key.
_TRAINING_ONLY_KEY_PATTERNS = (
    re.compile(r"^n_averaged$"),
    re.compile(r"^step_count$"),
    re.compile(r"^decay$"),
)
_STRIPPABLE_PREFIXES = ("module.", "_orig_mod.", "ema_model.")


class VOSR2LoadError(RuntimeError):
    """Raised for any missing file, incompatible config, or malformed state dict."""


def _safe_child_dir(root: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise VOSR2LoadError(f"Invalid VOSR2 selection: {name!r}")
    candidate = (root / name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise VOSR2LoadError(f"Invalid VOSR2 selection: {name!r}")
    return candidate


def list_model_bundles() -> list:
    if not _VOSR2_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in _VOSR2_ROOT.iterdir()
        if p.is_dir() and (p / "args.json").is_file()
    )


def model_options() -> list:
    """Bundle combo options: whatever is on disk, but always offering ``KNOWN_MODEL``.

    ComfyUI snapshots combo ``options`` when the schema is built, so on a fresh
    install ``list_model_bundles()`` returns ``[]`` and the dropdown would be
    empty and unselectable. Seeding it with the confirmed name lets the user pick
    it and run; the loader then downloads that bundle on first execute.
    """
    found = list_model_bundles()
    return found if KNOWN_MODEL in found else [KNOWN_MODEL, *found]


def _convert_dinov2_pth_to_safetensors(src_pth: Path, dest: Path) -> None:
    state = torch.load(str(src_pth), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise VOSR2LoadError(
            f"Unexpected DINOv2 checkpoint format at {src_pth}: expected a flat state dict."
        )
    tensors = {k: v.contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise VOSR2LoadError(f"DINOv2 checkpoint at {src_pth} contained no tensors.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(dest)


def ensure_vosr2_files(model_name: str) -> None:
    """Fetch the confirmed VOSR 2.0 bundle from ``CSWRY/VOSR`` for missing parts only.

    A no-op once the files are present -- it never contacts the network then.
    Only ``KNOWN_MODEL`` can be auto-fetched; a custom bundle name that is absent
    is left for the normal "not found" error in ``load_vosr2``, since this code
    cannot know where such a bundle would come from.
    """
    if model_name != KNOWN_MODEL:
        return

    bundle = _VOSR2_ROOT / KNOWN_MODEL
    vae_dir = bundle / _VAE_SUBDIR

    dit_missing = _find_dit_weight(bundle) is None or not (bundle / "args.json").is_file()
    vae_missing = not all((vae_dir / name).is_file() for name in _VAE_FILES)
    vision_missing = not (bundle / _VISION_FILENAME).is_file()

    if not (dit_missing or vae_missing or vision_missing):
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise VOSR2LoadError(
            "VOSR 2.0 model files are missing and huggingface_hub is not available "
            "to download them. Install huggingface_hub (it normally ships with "
            "ComfyUI), or place the files manually -- see the README 'Model files' "
            "section."
        ) from exc

    if dit_missing:
        logging.info("[VOSR2] downloading DiT bundle from %s ...", HF_REPO_ID)
        for f in _DIT_HF_FILES:
            hf_hub_download(HF_REPO_ID, f, local_dir=str(_VOSR2_ROOT))
    if vae_missing:
        logging.info("[VOSR2] downloading Qwen-Image 2D VAE from %s ...", HF_REPO_ID)
        vae_dir.mkdir(parents=True, exist_ok=True)
        for f in _VAE_HF_FILES:
            hf_hub_download(HF_REPO_ID, f, local_dir=str(bundle))
    if vision_missing:
        logging.info("[VOSR2] downloading + converting DINOv2-L encoder from %s ...", HF_REPO_ID)
        src = hf_hub_download(HF_REPO_ID, _DINOV2_HF_FILE)
        _convert_dinov2_pth_to_safetensors(Path(src), bundle / _VISION_FILENAME)


def _load_args_json(bundle_dir: Path) -> dict:
    args_path = bundle_dir / "args.json"
    if not args_path.is_file():
        raise VOSR2LoadError(f"VOSR2 model bundle at {bundle_dir} is missing args.json.")
    with open(args_path, "r") as f:
        args = json.load(f)

    for key, expected in REQUIRED_ARGS.items():
        actual = args.get(key)
        if actual != expected:
            raise VOSR2LoadError(
                f"VOSR2 model bundle at {bundle_dir} has an incompatible config: "
                f"expected {key}={expected!r}, got {actual!r}. VOSR2 v1 supports only "
                f"the VOSR 2.0 one-step 1.4B checkpoint."
            )
    missing = [k for k in DIT_ARG_KEYS if k not in args]
    if missing:
        raise VOSR2LoadError(
            f"VOSR2 model bundle at {bundle_dir}'s args.json is missing required field(s): {missing}."
        )
    return args


def _find_dit_weight(bundle_dir: Path):
    for candidate in (
        bundle_dir / "clean_weights" / "ema_model.safetensors",
        bundle_dir / "checkpoints" / "ema_model.safetensors",
        bundle_dir / "ema_model.safetensors",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_dit_weight_path(bundle_dir: Path) -> Path:
    weight = _find_dit_weight(bundle_dir)
    if weight is None:
        raise VOSR2LoadError(
            f"No ema_model.safetensors found under {bundle_dir} "
            f"(looked in clean_weights/, checkpoints/, and the bundle root)."
        )
    return weight


def _clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        stripped = key
        for prefix in _STRIPPABLE_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        if any(pattern.match(stripped) for pattern in _TRAINING_ONLY_KEY_PATTERNS):
            continue
        cleaned[stripped] = value
    return cleaned


def _strict_load(module: torch.nn.Module, state_dict: dict, source: Path):
    state_dict = _clean_state_dict_keys(state_dict)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise VOSR2LoadError(
            f"VOSR2 checkpoint at {source} does not match the expected {type(module).__name__} "
            f"architecture (missing={missing}, unexpected={unexpected})."
        )


def _resolve_dtype(dtype: str, device) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "default":
        return comfy.model_management.unet_dtype(device=device)
    raise VOSR2LoadError(f"Unknown dtype option: {dtype!r}")


class VOSR2Model:
    """Runtime bundle: independently offloadable DiT/VAE/vision-encoder plus config.

    Exposes narrow operations (`vision_features`, `encode`, `denoise_one_step`,
    `decode`) rather than the raw modules, so callers never need to probe for
    capabilities on the wrapped components.
    """

    def __init__(self, dit_patcher, vae_patcher, vision_patcher, args: dict):
        self.dit_patcher = dit_patcher
        self.vae_patcher = vae_patcher
        self.vision_patcher = vision_patcher
        self.args = args
        self.vision_layer_index = args["layer_dinov2b_list"][0]
        self.dinov2_size = args.get("dinov2_size", 448)

    def _load(self, patcher):
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        return patcher.model, patcher.load_device

    def vision_features(self, lq_bchw_01: torch.Tensor) -> list:
        """lq_bchw_01: (B, 3, H, W) in [0, 1]. Returns a one-item list (fixed-index layer)."""
        model, device = self._load(self.vision_patcher)
        x = torch.nn.functional.interpolate(lq_bchw_01, size=(self.dinov2_size, self.dinov2_size), mode="bicubic").clamp(0.0, 1.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x.to(device) - mean) / std
        feats = model.forward_intermediate_layer(x.to(model.pos_embed.dtype), self.vision_layer_index)
        return [feats]

    def encode(self, x_bchw_neg1_1: torch.Tensor):
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        return tiled_vae.encode_latent(model, x_bchw_neg1_1.to(device))

    def encode_tiled(self, x_bchw_neg1_1: torch.Tensor, tile_size: int, tile_overlap: int):
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        return tiled_vae.encode_dispatch(model, x_bchw_neg1_1.to(device), tile_size, tile_overlap)

    def decode(self, latent: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        return tiled_vae.decode_latent(model, latent.to(device), latents_mean, latents_std)

    def decode_tiled(self, latent: torch.Tensor, latents_mean, latents_std, tile_size: int, tile_overlap: int) -> torch.Tensor:
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        return tiled_vae.decode_dispatch(model, latent.to(device), latents_mean, latents_std, tile_size, tile_overlap)

    def dit_velocity(self, inp: torch.Tensor, t_cur: float, t_next: float, venc_fea: list) -> torch.Tensor:
        """Single DiT forward: predicted flow velocity for `inp = cat([lq_latent, z], dim=1)`."""
        model, device = self._load(self.dit_patcher)
        compute_dtype = model.t_embedder.mlp[0].weight.dtype
        inp = inp.to(device=device, dtype=compute_dtype)
        venc_fea = [f.to(device=device, dtype=compute_dtype) for f in venc_fea]
        b = inp.shape[0]
        t_cur_t = torch.full((b,), t_cur, device=device, dtype=compute_dtype)
        t_next_t = torch.full((b,), t_next, device=device, dtype=compute_dtype)
        return model.forward_flexible(inp, t_cur_t, t_next_t, venc_fea)

    def denoise_one_step(self, lq_latent: torch.Tensor, noise: torch.Tensor, venc_fea: list) -> torch.Tensor:
        """One Euler flow-matching update from t=1 to t=0 (VOSR2 is a one-step model)."""
        device = self.dit_patcher.load_device
        z = noise.to(device)
        u = self.dit_velocity(torch.cat([lq_latent.to(device), z], dim=1), 1.0, 0.0, venc_fea)
        return z - u


def load_vosr2(model_name: str, dtype: str) -> VOSR2Model:
    ensure_vosr2_files(model_name)

    bundle_dir = _safe_child_dir(_VOSR2_ROOT, model_name)
    if not bundle_dir.is_dir():
        raise VOSR2LoadError(f"VOSR2 model bundle not found: {model_name!r}")
    args = _load_args_json(bundle_dir)
    dit_weight_path = _resolve_dit_weight_path(bundle_dir)

    # The VAE and vision encoder are part of the bundle -- VOSR 2.0 is a fixed
    # DiT/VAE/DINOv2 triple, not independently selectable components.
    vae_dir = bundle_dir / _VAE_SUBDIR
    if not (vae_dir / "config.json").is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} is missing its Qwen-Image 2D VAE "
            f"({vae_dir}/config.json). The VAE's latent space is tied to this DiT "
            f"and cannot be substituted."
        )

    vision_path = bundle_dir / _VISION_FILENAME
    if not vision_path.is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} is missing its DINOv2-L encoder ({vision_path})."
        )

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    compute_dtype = _resolve_dtype(dtype, load_device)

    base_channels = 16  # Qwen 2D VAE latent channels
    dit = LightningDiT(
        input_size=args["resolution"] // 8,
        patch_size=args["patch_size"],
        in_channels=2 * base_channels,
        out_channels=base_channels,
        hidden_size=args["dim"],
        depth=args["depth"],
        num_heads=args["num_heads"],
        mlp_ratio=args["mlp_ratio"],
        z_dims=args["enc_dim"],
        encdim_ratio=args["encdim_ratio"],
        auxiliary_time_cond=args["auxiliary_time_cond"],
        use_qknorm=args["use_qknorm"],
        use_swiglu=args["use_swiglu"],
        use_rope=args["use_rope"],
        use_rmsnorm=args["use_rmsnorm"],
        num_fused_layers=len(args["layer_dinov2b_list"]),
    )
    dit_state_dict = load_file(str(dit_weight_path))
    _strict_load(dit, dit_state_dict, dit_weight_path)
    dit = dit.eval().to(compute_dtype)
    for p in dit.parameters():
        p.requires_grad_(False)

    vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
    vae = vae.eval().float()  # Qwen VAE runs in fp32; see VOSR2.md "Device, dtype, and memory".
    for p in vae.parameters():
        p.requires_grad_(False)

    vision_encoder = build_dinov2_vitl14()
    vision_state_dict = load_file(str(vision_path))
    _strict_load(vision_encoder, vision_state_dict, vision_path)
    vision_encoder = vision_encoder.eval().to(compute_dtype)
    for p in vision_encoder.parameters():
        p.requires_grad_(False)

    dit_patcher = comfy.model_patcher.ModelPatcher(dit, load_device=load_device, offload_device=offload_device)
    vae_patcher = comfy.model_patcher.ModelPatcher(vae, load_device=load_device, offload_device=offload_device)
    vision_patcher = comfy.model_patcher.ModelPatcher(vision_encoder, load_device=load_device, offload_device=offload_device)

    return VOSR2Model(dit_patcher, vae_patcher, vision_patcher, args)
