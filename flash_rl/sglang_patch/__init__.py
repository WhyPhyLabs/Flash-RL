import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _is_sglang_installed() -> bool:
    try:
        import sglang  # noqa: F401
        return True
    except Exception:
        return False


def _read_flashrl_config() -> Optional[str]:
    return os.environ.get("FLASHRL_CONFIG")


def _warn_if_irrelevant_envs():
    # vLLM-specific knobs should be no-ops for SGLang to avoid confusion
    if os.environ.get("FLASHRL_LMHEAD_FP32", "0") == "1":
        logger.warning(
            "FLASHRL_LMHEAD_FP32 is only applicable to vLLM; ignoring for SGLang."
        )


def _maybe_patch_engine_args():
    try:
        from .engine_args import patch_sglang_engine_init

        patched = patch_sglang_engine_init()
        logger.debug("SGLang engine init patched: %s", patched)
    except Exception as e:
        logger.warning("Failed to patch SGLang engine args: %s", e)


def _maybe_patch_sampler():
    try:
        from .sampler_patch import patch_sglang_sampler

        patched = patch_sglang_sampler()
        logger.debug("SGLang sampler patched: %s", patched)
    except Exception as e:
        logger.warning("Failed to patch SGLang sampler: %s", e)


def auto_patch() -> Dict[str, Any]:
    """
    Auto-activate the Flash-RL SGLang adapter when FLASHRL_CONFIG is present.

    Returns a small dict with status booleans for observability.
    """
    status = {"enabled": False, "engine": False, "sampler": False}

    cfg = _read_flashrl_config()
    if not cfg:
        logger.debug("FLASHRL_CONFIG not set; skipping SGLang auto-patch.")
        return status

    if not _is_sglang_installed():
        logger.debug("SGLang not installed; skipping SGLang auto-patch.")
        return status

    _warn_if_irrelevant_envs()

    # Patch engine args first (quantization / kv cache dtype mapping)
    try:
        from .engine_args import patch_sglang_engine_init

        status["engine"] = bool(patch_sglang_engine_init())
    except Exception as e:
        logger.warning("Engine args patch failed: %s", e)

    # Patch sampler for sampling-consistent post-filter logprobs
    try:
        from .sampler_patch import patch_sglang_sampler

        status["sampler"] = bool(patch_sglang_sampler())
    except Exception as e:
        logger.warning("Sampler patch failed: %s", e)

    status["enabled"] = status["engine"] or status["sampler"]
    if status["enabled"]:
        logger.info("Flash-RL SGLang adapter active (engine=%s, sampler=%s)", status["engine"], status["sampler"])
    else:
        logger.debug("Flash-RL SGLang adapter not activated.")

    return status
