import sys

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import loader
from .inference import run_vosr2
from .loader import VOSR2Model


class VOSR2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VOSR2ModelLoader",
            display_name="VOSR 2.0 Model Loader",
            category="image/upscaling/VOSR2",
            description="Load a VOSR 2.0 (one-step 1.4B) bundle: matched LightningDiT + Qwen-Image 2D VAE + DINOv2-L. The whole bundle is downloaded from the CSWRY/VOSR Hugging Face repo on first run if absent.",
            inputs=[
                io.Combo.Input("model", options=loader.model_options(), default=loader.KNOWN_MODEL, tooltip="VOSR 2.0 bundle folder under models/vosr2/ (DiT + its matched VAE + vision encoder). Downloaded from CSWRY/VOSR on first run if absent."),
                io.Combo.Input("dtype", options=["default", "fp16", "bf16"], default="default", tooltip="Compute dtype for the DiT and vision encoder. The VAE always runs in fp32."),
            ],
            outputs=[
                io.Custom("VOSR2_MODEL").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, dtype) -> io.NodeOutput:
        bundle = loader.load_vosr2(model, dtype)
        return io.NodeOutput(bundle)


class VOSR2Upscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VOSR2Upscale",
            display_name="VOSR 2.0 Upscale",
            category="image/upscaling/VOSR2",
            description="One-step VOSR 2.0 image super-resolution.",
            inputs=[
                io.Custom("VOSR2_MODEL").Input("model", tooltip="Loaded VOSR 2.0 bundle."),
                io.Image.Input("image"),
                io.Int.Input("upscale", default=4, min=1, max=4, tooltip="Exact output multiplier."),
                io.Int.Input("seed", default=42, min=0, max=sys.maxsize, control_after_generate=io.ControlAfterGenerate.fixed, tooltip="Initial latent-noise seed."),
                io.Combo.Input("color_alignment", options=["wavelet", "adain", "none"], default="wavelet"),
                io.Int.Input("tile_size", default=0, min=0, max=4096, step=64, tooltip="DiT pixel tile size; 0 disables tiling. VOSR 2.0 was trained natively at up to 512px, so tiling is required (e.g. 512) whenever the upscaled output exceeds 512x512 -- otherwise quality degrades."),
                io.Int.Input("tile_overlap", default=32, min=0, max=512, step=8),
                io.Int.Input("vae_tile_size", default=0, min=0, max=8192, step=64, tooltip="VAE pixel tile size; 0 disables VAE tiling and decodes the full image in one pass regardless of tile_size -- a common source of CUDA OOM once the upscaled output goes much past 1024px. Set a nonzero value (e.g. 1024) for large outputs."),
                io.Int.Input("vae_tile_overlap", default=32, min=0, max=512, step=8),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, model, image, upscale, seed, color_alignment, tile_size, tile_overlap, vae_tile_size, vae_tile_overlap):
        if tile_size > 0 and tile_overlap >= tile_size:
            return f"tile_overlap ({tile_overlap}) must be smaller than tile_size ({tile_size})."
        if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
            return f"vae_tile_overlap ({vae_tile_overlap}) must be smaller than vae_tile_size ({vae_tile_size})."
        return True

    @classmethod
    def execute(cls, model: VOSR2Model, image, upscale, seed, color_alignment, tile_size, tile_overlap, vae_tile_size, vae_tile_overlap) -> io.NodeOutput:
        result = run_vosr2(
            model, image, upscale, seed, color_alignment,
            tile_size, tile_overlap, vae_tile_size, vae_tile_overlap,
        )
        return io.NodeOutput(result)


class VOSR2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VOSR2ModelLoader,
            VOSR2Upscale,
        ]


async def comfy_entrypoint() -> VOSR2Extension:
    return VOSR2Extension()
