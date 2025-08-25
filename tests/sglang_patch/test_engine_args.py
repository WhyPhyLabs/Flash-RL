import importlib


def test_engine_args_fp8_mapping(monkeypatch):
    monkeypatch.setenv("FLASHRL_CONFIG", "fp8")
    # by default, we do not set kv_cache_dtype; defer to SGLang
    monkeypatch.delenv("FLASHRL_DISABLE_FP8_KV", raising=False)
    monkeypatch.delenv("FLASHRL_KV_CACHE_DTYPE", raising=False)
    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)
    args = mod._map_flashrl_to_sglang_args()
    assert args.get("quantization") == "fp8"
    assert "kv_cache_dtype" not in args


def test_engine_args_bf16_mapping(monkeypatch):
    monkeypatch.setenv("FLASHRL_CONFIG", "bf16")
    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)
    args = mod._map_flashrl_to_sglang_args()
    assert args == {}


def test_engine_args_disable_kv_fp8(monkeypatch):
    monkeypatch.setenv("FLASHRL_CONFIG", "fp8")
    monkeypatch.setenv("FLASHRL_DISABLE_FP8_KV", "1")
    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)
    args = mod._map_flashrl_to_sglang_args()
    assert args.get("quantization") == "fp8"
    assert "kv_cache_dtype" not in args


def test_engine_args_explicit_kv_dtype(monkeypatch):
    monkeypatch.setenv("FLASHRL_CONFIG", "fp8")
    monkeypatch.setenv("FLASHRL_KV_CACHE_DTYPE", "fp8_e4m3")
    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)
    args = mod._map_flashrl_to_sglang_args()
    assert args.get("quantization") == "fp8"
    assert args.get("kv_cache_dtype") == "fp8_e4m3"


def test_engine_kwargs_precedence(monkeypatch):
    """
    Verify that user-provided engine kwargs override Flash-RL defaults.
    We simulate an importable SGLang Engine class and ensure that when the
    user passes quantization="bf16", the patch does not overwrite it with fp8.
    """
    import importlib
    import sys
    from types import ModuleType

    # Ensure our mapping would choose fp8 by default
    monkeypatch.setenv("FLASHRL_CONFIG", "fp8")

    # Build a fake module tree: sglang.srt.server.engine
    root = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    server = ModuleType("sglang.srt.server")
    engine_mod = ModuleType("sglang.srt.server.engine")

    captured = {}

    class Engine:  # noqa: N801 - mimic external class
        __flashrl_patched__ = False

        def __init__(self, *args, **kwargs):  # noqa: D401
            nonlocal captured
            captured = kwargs.copy()

    engine_mod.Engine = Engine

    sys.modules["sglang"] = root
    sys.modules["sglang.srt"] = srt
    sys.modules["sglang.srt.server"] = server
    sys.modules["sglang.srt.server.engine"] = engine_mod

    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)

    # Apply patch and instantiate with an explicit conflicting kwarg
    assert mod.patch_sglang_engine_init() is True
    from sglang.srt.server.engine import Engine as PatchedEngine  # type: ignore

    PatchedEngine(quantization="bf16", kv_cache_dtype="bf16")
    assert captured.get("quantization") == "bf16"
    # If user specified kv_cache_dtype, it should remain as provided
    assert captured.get("kv_cache_dtype") == "bf16"
