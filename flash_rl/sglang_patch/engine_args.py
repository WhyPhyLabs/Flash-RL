import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _map_flashrl_to_sglang_args() -> Dict[str, Any]:
    """
    Map FLASHRL_CONFIG to SGLang Engine keyword arguments.

    - fp8 -> {quantization: 'fp8', kv_cache_dtype: 'fp8_e5m2'} (unless disabled)
    - bf16 -> {} (logprob patch only)
    - YAML path -> (parse later; currently degrade to bf16 with a warning)
    """
    cfg = os.environ.get("FLASHRL_CONFIG", "").strip()
    if not cfg:
        return {}

    if cfg.lower() == "bf16":
        return {}

    if cfg.lower() == "fp8":
        # Defer FP8 specifics to SGLang defaults. Only set kv_cache_dtype if explicitly requested.
        args: Dict[str, Any] = {"quantization": "fp8"}
        kv_dtype = os.environ.get("FLASHRL_KV_CACHE_DTYPE")
        disable_kv = os.environ.get("FLASHRL_DISABLE_FP8_KV", "0") == "1"
        if kv_dtype and not disable_kv:
            args["kv_cache_dtype"] = kv_dtype
        return args

    # YAML profile support can be extended here.
    if cfg.endswith(".yaml") or cfg.endswith(".yml"):
        logger.warning(
            "YAML profiles for SGLang are not yet mapped; running as bf16 (logprob patch only)."
        )
        return {}

    # Fallback
    return {}


def patch_sglang_engine_init() -> bool:
    """
    Monkey-patch SGLang Engine __init__ to inject Flash-RL quantization args
    if the user didn't already provide them.

    We guard heavily so that any import/attribute error degrades gracefully.
    """
    try:
        # SGLang engine path has evolved; try a few candidates.
        Engine = None
        for path in (
            "sglang.srt.server.engine.Engine",
            "sglang.server.engine.Engine",
            "sglang.engine.Engine",
        ):
            try:
                module_path, cls_name = path.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[cls_name])
                Engine = getattr(mod, cls_name)
                break
            except Exception:
                continue

        if Engine is None:
            logger.debug("Could not locate SGLang Engine class; skip engine patch.")
            return False

        if getattr(Engine, "__flashrl_patched__", False):
            return True

        orig_init = Engine.__init__
        flashrl_defaults = _map_flashrl_to_sglang_args()

        def wrapped_init(self, *args, **kwargs):  # type: ignore[no-redef]
            # Only inject missing keys so user-specified kwargs win.
            for k, v in flashrl_defaults.items():
                kwargs.setdefault(k, v)
            return orig_init(self, *args, **kwargs)

        Engine.__init__ = wrapped_init  # type: ignore[assignment]
        Engine.__flashrl_patched__ = True
        logger.info("SGLang Engine init patched with Flash-RL defaults: %s", flashrl_defaults)
        return True
    except Exception as e:
        logger.warning("Failed to patch SGLang Engine init: %s", e)
        return False
