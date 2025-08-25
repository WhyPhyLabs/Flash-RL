import importlib
import math

import pytest

torch = None
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow environments without torch
    pass


pytestmark = pytest.mark.skipif(
    torch is None, reason="torch is required for sampler math tests"
)


def test_post_filter_distribution_topk():
    mod = importlib.import_module("flash_rl.sglang_patch.sampler_patch")
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # increasing
    probs = mod.compute_post_filter_distribution(logits, temperature=1.0, top_k=2)
    # Only two largest kept (indices 2 and 3)
    kept = probs.nonzero(as_tuple=True)[-1].tolist()
    assert set(kept) == {2, 3}
    # Renormalized over the kept set
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)


def test_post_filter_distribution_topp():
    mod = importlib.import_module("flash_rl.sglang_patch.sampler_patch")
    # Create distribution where top two exceed 0.9 cumulative
    logits = torch.tensor([[0.0, 0.0, 2.0, 3.0]])
    probs = mod.compute_post_filter_distribution(logits, top_p=0.9)
    # Expect top indices kept; sum to 1
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)


def test_post_filter_distribution_minp():
    mod = importlib.import_module("flash_rl.sglang_patch.sampler_patch")
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    base = torch.softmax(logits, dim=-1)
    maxp = float(base.max())
    probs = mod.compute_post_filter_distribution(logits, min_p=0.5)
    # Keep tokens with prob >= 0.5 * max_prob
    kept_mask = base >= (0.5 * maxp)
    assert torch.equal((probs > 0), kept_mask)
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)
