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
