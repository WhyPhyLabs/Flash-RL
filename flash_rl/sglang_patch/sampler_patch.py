import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _softmax(logits):
    # numerically stable softmax over last dimension

    max_val, _ = logits.max(dim=-1, keepdim=True)
    exp = (logits - max_val).exp()
    denom = exp.sum(dim=-1, keepdim=True)
    return exp / denom.clamp_min(1e-20)


def compute_post_filter_distribution(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
):
    """
    Given logits for a step (shape [..., V]), compute the post-filter sampling
    distribution with temperature, top-k, top-p, min-p and renormalization.
    Returns probs of same shape as logits.
    """
    import torch

    if temperature is None or temperature <= 0:
        temperature = 1.0
    logits = logits / float(temperature)
    probs = _softmax(logits)

    # Create a mask initialized to all True
    mask = torch.ones_like(probs, dtype=torch.bool)

    if top_k is not None and top_k > 0:
        # keep only top_k tokens
        topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        keep = torch.zeros_like(probs, dtype=torch.bool)
        keep.scatter_(-1, topk_idx, True)
        mask = mask & keep

    if top_p is not None and 0.0 < top_p < 1.0:
        # nucleus sampling: sort by prob desc, keep minimal prefix reaching top_p
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum <= top_p
        # ensure at least one token kept
        cutoff[..., 0] = True
        keep = torch.zeros_like(probs, dtype=torch.bool)
        keep.scatter_(-1, sorted_idx, cutoff)
        mask = mask & keep

    if min_p is not None and 0.0 < min_p < 1.0:
        # keep tokens with prob >= min_p * max_prob
        max_prob, _ = probs.max(dim=-1, keepdim=True)
        thresh = min_p * max_prob
        keep = probs >= thresh
        mask = mask & keep

    # Apply mask and renormalize
    masked = probs * mask
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    renorm = masked / denom
    return renorm


def compute_logprob_of_token(probs, token_id):
    import torch

    # gather probability and take log
    p = probs.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
    return torch.log(p.clamp_min(1e-20))


def patch_sglang_sampler() -> bool:
    """
    Patch SGLang sampler to ensure per-token logprobs used in RL are computed
    from the post-filter distribution (after top-k/top-p/min-p and renorm).

    Strategy:
      1) If SGLang exposes a `return_logprob` that already reflects post-filter
         numerics, keep behavior and do nothing.
      2) Otherwise, wrap Sampler.forward to compute post-filter distribution
         and store logprob for the chosen token on the output object when possible.

    Returns True if a patch was applied or confirmed unnecessary.
    """
    try:
        # Try to locate a stable Sampler implementation (paths evolved over time)
        Sampler = None
        for path in (
            "sglang.srt.layers.sampler.Sampler",
            "sglang.srt.sampler.Sampler",
        ):
            try:
                module_path, cls_name = path.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[cls_name])
                Sampler = getattr(mod, cls_name)
                break
            except Exception:
                continue

        if Sampler is None:
            logger.debug("Could not locate SGLang Sampler class; skip sampler patch.")
            return False

        if getattr(Sampler, "__flashrl_patched__", False):
            return True

        orig_forward = Sampler.forward

        def wrapped_forward(self, *args, **kwargs):  # type: ignore[no-redef]
            # Call original forward to get chosen tokens and (possibly) logprobs
            out = orig_forward(self, *args, **kwargs)

            # If upstream provides post-filter logprobs, leave as-is.
            # Heuristic: presence of attribute 'post_filter_logprobs' or metadata
            try:
                if hasattr(out, "post_filter_logprobs"):
                    return out
            except Exception:
                pass

            # Try to recompute using inputs available in self/kwargs
            try:

                logits = getattr(out, "logits", None)
                token_ids = getattr(out, "token_ids", None)

                # Fallback: check common names in kwargs
                if logits is None:
                    logits = kwargs.get("logits", None)
                if token_ids is None:
                    token_ids = kwargs.get("token_ids", None)

                if logits is None or token_ids is None:
                    return out  # not enough info, return unchanged

                sampling_params = getattr(self, "sampling_params", None)
                temperature = getattr(sampling_params, "temperature", 1.0) if sampling_params else 1.0
                top_k = getattr(sampling_params, "top_k", None) if sampling_params else None
                top_p = getattr(sampling_params, "top_p", None) if sampling_params else None
                min_p = getattr(sampling_params, "min_p", None) if sampling_params else None

                probs = compute_post_filter_distribution(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p
                )
                logprobs = compute_logprob_of_token(probs, token_ids)

                # Attach for downstream consumers
                try:
                    out.post_filter_logprobs = logprobs
                except Exception:
                    pass
            except Exception as e:  # be conservative, never break user inference
                logger.debug("Flash-RL sampler patch compute failed: %s", e)
                return out

            return out

        Sampler.forward = wrapped_forward  # type: ignore[assignment]
        Sampler.__flashrl_patched__ = True
        logger.info("SGLang Sampler patched for post-filter logprobs.")
        return True
    except Exception as e:
        logger.warning("Failed to patch SGLang sampler: %s", e)
        return False
