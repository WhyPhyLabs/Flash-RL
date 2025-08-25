import logging
import os

# Get logging configuration from environment
log_file = os.getenv("FLASHRL_LOGGING_FILE")
log_level = os.getenv("FLASHRL_LOGGING_LEVEL", "INFO")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level.upper()))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

if log_file:
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
logger = logging.getLogger(__name__)

def check_vllm_installed():
    """Check if vllm is installed"""
    try:
        import vllm
        return True
    except ImportError:
        return False

def check_sglang_installed():
    """Check if sglang is installed"""
    try:
        import sglang  # noqa: F401
        return True
    except ImportError:
        return False

def check_dist_initialized():
    """Check if distributed environment is initialized"""
    try:
        from torch.distributed import is_initialized
        return is_initialized()
    except ImportError:
        return False

def _warn_sglang_version():
    try:
        import sglang  # noqa: F401
        ver = getattr(sglang, "__version__", "0")
        logger.debug("flash-rl: detected sglang version: %s", ver)
        try:
            from packaging import version as _ver
            if _ver.parse(ver) < _ver.parse("0.3.6"):
                logger.warning(
                    "flash-rl: sglang<0.3.6 detected; features may degrade (logprob patch only)."
                )
        except Exception:
            # packaging not installed or parse failure; skip strict compare
            pass
    except Exception:
        # sglang not installed; skip
        pass


def _activate_vllm():
    if not check_vllm_installed():
        logger.debug("vLLM not installed; skipping vLLM patching.")
        return
    from .vllm_patch import (
        patch_vllm_llm,
        patch_vllm_process_weights_after_loading,
    )
    # patch optional new upstream feature if available
    try:
        from .vllm_patch import patch_vllm_fp8_create_weight  # type: ignore
    except Exception:
        patch_vllm_fp8_create_weight = None

    process_weights_status = patch_vllm_process_weights_after_loading()
    logger.debug(
        f"Patching vllm process_weights_after_loading... status: {process_weights_status}"
    )

    patch_status = patch_vllm_llm()
    logger.debug(
        f"Patching the vllm LLM to enable flash_rl quantization... status: {patch_status}"
    )

    if patch_vllm_fp8_create_weight is not None:
        try:
            fp8w_status = patch_vllm_fp8_create_weight()
            logger.debug(f"Patching the vllm fp8 linear... status: {fp8w_status}")
        except Exception as e:
            logger.debug(f"Failed to patch vllm fp8 create_weights: {e}")

    if 'FLASHRL_TEST_RELOAD' in os.environ:
        from .vllm_patch import patch_vllm_llm_test_reload

        reload_status = patch_vllm_llm_test_reload()
        logger.debug(
            f"Patching vllm LLM init to test reload... status: {reload_status}"
        )

    if os.environ.get('FLASHRL_LMHEAD_FP32', '0') == '1':
        from .vllm_patch import patch_vllm_lmhead_to_fp32

        patch_status = patch_vllm_lmhead_to_fp32()
        logger.debug(f"Patching vllm lmhead to fp32... status: {patch_status}")


def _activate_sglang():
    if not check_sglang_installed():
        logger.debug("SGLang not installed; skipping SGLang patching.")
        return
    try:
        from .sglang_patch import auto_patch as _sglang_auto_patch

        _warn_sglang_version()
        _sglang_status = _sglang_auto_patch()
        logger.debug(f"SGLang adapter status: {_sglang_status}")
    except Exception as e:
        logger.warning(f"Failed to activate SGLang adapter: {e}")


def _activate_backends():
    # Check if patching is needed based on environment variables
    if 'FLASHRL_CONFIG' not in os.environ:
        logger.debug("FLASHRL_CONFIG not set; skipping backend patching")
        return

    backend = os.environ.get('FLASHRL_BACKEND', 'auto').lower()
    if backend == 'vllm':
        _activate_vllm()
        return
    if backend == 'sglang':
        _activate_sglang()
        return

    # auto (default): try both guardedly
    _activate_vllm()
    _activate_sglang()


_activate_backends()
    
