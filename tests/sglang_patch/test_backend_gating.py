import importlib
import sys
from types import ModuleType


def test_backend_gate_sglang_skips_vllm(monkeypatch):
    """
    When FLASHRL_BACKEND=sglang and both 'sglang' and 'vllm' are importable,
    Flash-RL should only activate the SGLang path and skip calling vLLM patchers.

    We simulate importable 'vllm' and assert that vllm patch entrypoints are not called.
    """
    # Env: flashrl enabled and gate to sglang
    monkeypatch.setenv("FLASHRL_CONFIG", "bf16")
    monkeypatch.setenv("FLASHRL_BACKEND", "sglang")

    # Provide a dummy sglang module so check passes
    sys.modules["sglang"] = ModuleType("sglang")

    # Provide a dummy vllm module so the import check passes
    sys.modules["vllm"] = ModuleType("vllm")

    # Intercept vllm patch function to detect unexpected calls
    called = {"llm": False}

    def _sentinel():
        called["llm"] = True
        return False

    # Import vllm_patch so that __init__ can resolve it if it tried
    import flash_rl.vllm_patch as vp

    monkeypatch.setattr(vp, "patch_vllm_llm", _sentinel, raising=True)

    # Now import flash_rl, which triggers activation at import time
    importlib.invalidate_caches()
    import importlib as _imp

    # Reload to ensure import-time code runs with our env and monkeypatches
    mod = _imp.import_module("flash_rl")
    _imp.reload(mod)

    # Ensure vllm patcher was not called due to backend gate
    assert called["llm"] is False

