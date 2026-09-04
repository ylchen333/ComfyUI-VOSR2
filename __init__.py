"""Package entrypoint.

Runs an explicit dependency/version check *before* importing anything that
touches torch, einops, or safetensors at module scope (``nodes_vosr`` ->
``loader``/``inference`` -> ``models/*``). Without this, an incompatible
environment (old torch missing ``torch.compiler``, a stripped-down install
missing ``einops``) fails deep inside a vendored architecture file with an
opaque ``AttributeError``/``ImportError``, and ComfyUI just logs "module
failed to load" with that traceback -- the node pack silently vanishes from
the UI with no actionable message. Failing loudly and clearly here, at the
top of the package, keeps that diagnosis one log line instead of a stack
trace hunt.
"""
import logging

_MIN_TORCH = (2, 1)  # torch.compiler.disable (models/pos_embed.py) needs >=2.1;
                      # F.scaled_dot_product_attention needs >=2.0.


def _check_dependencies() -> None:
    try:
        import torch
    except ImportError as exc:
        msg = (
            "ComfyUI-VOSR2 requires PyTorch, but it could not be imported. "
            "This normally ships with ComfyUI itself -- check your ComfyUI "
            "install."
        )
        logging.error("[VOSR2] %s", msg)
        raise ImportError(msg) from exc

    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if version < _MIN_TORCH:
        msg = (
            f"ComfyUI-VOSR2 requires PyTorch >= {'.'.join(map(str, _MIN_TORCH))} "
            f"(uses torch.compiler and scaled_dot_product_attention), found "
            f"{torch.__version__}. Upgrade PyTorch, or update ComfyUI to a build "
            f"that ships a newer torch."
        )
        logging.error("[VOSR2] %s", msg)
        raise ImportError(msg)

    for pkg in ("einops", "safetensors"):
        try:
            __import__(pkg)
        except ImportError as exc:
            msg = (
                f"ComfyUI-VOSR2 requires the '{pkg}' package, which normally "
                f"ships with ComfyUI core but was not found in this environment. "
                f"Install it with `pip install {pkg}` in ComfyUI's Python "
                f"environment."
            )
            logging.error("[VOSR2] %s", msg)
            raise ImportError(msg) from exc


_check_dependencies()

from .nodes_vosr import comfy_entrypoint

# Serves web/docs/<node_id>.md as each node's in-UI help page. See
# https://docs.comfy.org/custom-nodes/backend/help_page
WEB_DIRECTORY = "web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
