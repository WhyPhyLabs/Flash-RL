import importlib


def test_engine_args_fp8_mapping(monkeypatch):
    monkeypatch.setenv("FLASHRL_CONFIG", "fp8")
    # ensure kv cache enabled by default
    monkeypatch.delenv("FLASHRL_DISABLE_FP8_KV", raising=False)

    mod = importlib.import_module("flash_rl.sglang_patch.engine_args")
    importlib.reload(mod)
    args = mod._map_flashrl_to_sglang_args()
    assert args.get("quantization") == "fp8"
    assert args.get("kv_cache_dtype") == "fp8_e5m2"


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

