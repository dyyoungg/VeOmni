"""Training-only muDNN FlashAttention adapter for torch-musa 2.11.

This uses ATen's existing autograd registration, not MATE/TileLang. Supported
semantics are unpadded self-attention, optionally packed, with zero dropout.
Metadata values are checked once per tensor/version (a bounded weak cache),
so repeated transformer layers do not synchronize metadata with the host.
No FA3 forward state is reused: muDNN owns both forward and backward.
"""

import math
import sys
import weakref
from collections import Counter, OrderedDict
from numbers import Integral, Real

import torch


_STATS = Counter()
_VALIDATED = OrderedDict()
_CACHE_LIMIT = 64
_BACKEND_CHECKED = False


def get_mudnn_call_stats(reset=False):
    """Return per-process Python dispatch counts, not GPU timing or bwd counts."""
    result = dict(_STATS)
    if reset:
        _STATS.clear()
    return result


def _check_parallel():
    # Avoid importing the entire model registry in standalone kernel tests.
    state_module = sys.modules.get("veomni.distributed.parallel_state")
    if state_module is None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError("muDNN attention needs VeOmni parallel state for distributed execution.")
        return
    state = state_module.get_parallel_state()
    if state is None:
        return
    if state.sp_enabled or state.ulysses_size != 1 or state.cp_size != 1:
        raise NotImplementedError("muDNN attention currently supports DP/FSDP2 only; SP/CP are unsupported.")


def _check_backend():
    global _BACKEND_CHECKED
    if not _BACKEND_CHECKED:
        for name in ("aten::_flash_attention_forward", "aten::_flash_attention_backward"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "PrivateUse1"):
                raise RuntimeError(f"Installed torch-musa does not provide {name} for PrivateUse1.")
        _BACKEND_CHECKED = True


def _cache_key(kind, tensors, extra):
    try:
        versions = tuple(tensor._version for tensor in tensors)
    except RuntimeError:
        # Inference tensors have no version counter: never cache their values.
        return None
    return (kind, tuple(id(tensor) for tensor in tensors), versions, extra)


def _cached_validation(kind, tensors, extra, validate):
    key = _cache_key(kind, tensors, extra)
    refs = _VALIDATED.get(key) if key is not None else None
    if refs is not None and all(ref() is tensor for ref, tensor in zip(refs, tensors)):
        _VALIDATED.move_to_end(key)
        _STATS[f"{kind}_cache_hits"] += 1
        return
    validate()
    _STATS[f"{kind}_validations"] += 1
    if key is not None:
        _VALIDATED[key] = tuple(weakref.ref(tensor) for tensor in tensors)
        _VALIDATED.move_to_end(key)
        while len(_VALIDATED) > _CACHE_LIMIT:
            _VALIDATED.popitem(last=False)


def _check_options(
    dropout_p, softmax_scale, window_size, softcap, alibi_slopes, deterministic, return_attn_probs, kwargs
):
    if kwargs:
        raise TypeError(f"Unsupported muDNN attention arguments: {sorted(kwargs)}")
    if dropout_p != 0.0:
        raise NotImplementedError("muDNN adapter requires dropout_p=0.")
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError("muDNN adapter does not support sliding-window attention.")
    if softcap not in (None, 0.0) or alibi_slopes is not None:
        raise NotImplementedError("muDNN adapter does not support softcap or ALiBi.")
    if deterministic or return_attn_probs:
        raise NotImplementedError("muDNN adapter does not support deterministic=True or attention probabilities.")
    if softmax_scale is not None and (
        not isinstance(softmax_scale, Real) or not math.isfinite(softmax_scale) or softmax_scale <= 0
    ):
        raise ValueError("softmax_scale must be a finite positive Python number or None.")


def _check_qkv(q, k, v, ndim):
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim for tensor in (q, k, v)):
        raise ValueError(f"Expected {ndim}-dimensional Q/K/V tensors.")
    if q.device.type != "musa" or any(tensor.device != q.device for tensor in (k, v)):
        raise ValueError("Q/K/V must be on the same MUSA device.")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(tensor.dtype != q.dtype for tensor in (k, v)):
        raise ValueError("Q/K/V must share BF16 or FP16 dtype.")
    if any(size <= 0 for tensor in (q, k, v) for size in tensor.shape):
        raise ValueError("Empty attention dimensions are unsupported.")
    if q.shape[:-2] != k.shape[:-2] or k.shape != v.shape:
        raise ValueError("Only self-attention with matching token lengths and K/V shapes is supported.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] > 128 or q.shape[-1] % 8:
        raise ValueError("Q/K/V need equal head dimensions, divisible by 8 and at most 128.")
    if q.shape[-2] % k.shape[-2]:
        raise ValueError("Q head count must be an integer multiple of K/V head count.")


def _check_metadata(q, k, cu_q, cu_k, max_q, max_k):
    for name, cu in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
        if not isinstance(cu, torch.Tensor) or cu.ndim != 1 or cu.dtype != torch.int32:
            raise ValueError(f"{name} must be a one-dimensional int32 tensor.")
        if cu.device != q.device or not cu.is_contiguous() or cu.numel() < 2:
            raise ValueError(f"{name} must be contiguous, on the Q device, with at least two entries.")
    if cu_q.numel() != cu_k.numel():
        raise ValueError("Q/K cumulative lengths must have the same number of sequences.")
    if any(not isinstance(length, Integral) or isinstance(length, bool) or length <= 0 for length in (max_q, max_k)):
        raise ValueError("max_seqlen_q/k must be positive Python integers, precomputed outside the model layers.")

    def validate():
        q_values = cu_q.detach().cpu().tolist()
        k_values = q_values if cu_q is cu_k else cu_k.detach().cpu().tolist()
        if q_values != k_values:
            raise ValueError("Only self-attention with identical Q/K sequence boundaries is supported.")
        if q_values[0] != 0 or q_values[-1] != q.shape[0] or k_values[-1] != k.shape[0]:
            raise ValueError("Cumulative lengths must begin at zero and end at the packed token count.")
        lengths = [end - start for start, end in zip(q_values, q_values[1:])]
        if min(lengths) <= 0:
            raise ValueError("Cumulative lengths must be strictly increasing; empty sequences are unsupported.")
        if max(lengths) > min(max_q, max_k):
            raise ValueError("max_seqlen_q/k is smaller than an actual packed sequence.")

    _cached_validation("metadata", (cu_q, cu_k), (q.shape[0], k.shape[0], max_q, max_k), validate)


def mudnn_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """Compute packed self-attention [T,H,D], preserving GQA and pack boundaries."""
    _check_options(
        dropout_p, softmax_scale, window_size, softcap, alibi_slopes, deterministic, return_attn_probs, kwargs
    )
    _check_parallel()
    _check_qkv(q, k, v, 3)
    _check_metadata(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    _check_backend()
    # Only the innermost dimension must be contiguous; avoid copying normal
    # [S,H,D] views produced by transposing Qwen's [B,H,S,D] tensors.
    q, k, v = (tensor if tensor.stride(-1) == 1 else tensor.contiguous() for tensor in (q, k, v))
    output = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        0.0,
        causal,
        False,
        scale=softmax_scale,
    )[0]
    _STATS["native_varlen_forward_calls"] += 1
    return output


def mudnn_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """Compute no-padding self-attention with FA-style [B,S,H,D] tensors."""
    _check_options(
        dropout_p, softmax_scale, window_size, softcap, alibi_slopes, deterministic, return_attn_probs, kwargs
    )
    _check_qkv(q, k, v, 4)
    batch, sequence = q.shape[:2]
    cu = torch.arange(0, (batch + 1) * sequence, sequence, device=q.device, dtype=torch.int32)
    # These boundaries are constructed from host-known shapes and are valid by
    # construction. Seed the validation cache without a device-to-host copy.
    _cached_validation("metadata", (cu, cu), (batch * sequence, batch * sequence, sequence, sequence), lambda: None)
    output = mudnn_flash_attn_varlen_func(
        q.reshape(-1, *q.shape[-2:]),
        k.reshape(-1, *k.shape[-2:]),
        v.reshape(-1, *v.shape[-2:]),
        cu,
        cu,
        sequence,
        sequence,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    _STATS["dense_facade_calls"] += 1
    return output.reshape(batch, sequence, q.shape[-2], q.shape[-1])


def mudnn_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    skip_ulysses=False,
    **kwargs,
):
    """Transformers facade: [B,H,S,D] in, [B,S,H,D] out; DP/FSDP2 only."""
    _check_parallel()
    if sliding_window is not None:
        raise NotImplementedError("muDNN adapter does not support sliding-window attention.")
    if kwargs.pop("output_attentions", False) or kwargs.pop("head_mask", None) is not None:
        raise NotImplementedError("muDNN adapter cannot return attention weights or apply head masks.")
    causal = kwargs.pop("is_causal", getattr(module, "is_causal", True))
    cu_q = kwargs.pop("cu_seq_lens_q", None)
    cu_k = kwargs.pop("cu_seq_lens_k", None)
    max_q = kwargs.pop("max_length_q", None)
    max_k = kwargs.pop("max_length_k", None)
    position_ids = kwargs.pop("position_ids", None)
    if kwargs.pop("use_cache", False):
        raise NotImplementedError("muDNN adapter does not support inference KV caches.")
    # Qwen2 passes this training bookkeeping through every decoder layer. It
    # has no masking role once cache use is disabled and cu_seqlens are set.
    kwargs.pop("cache_position", None)
    # The customer collator forwards loss/packing bookkeeping through Qwen2.
    # Full attention uses only the validated cu_seq_lens_* boundaries above;
    # the linear-attention alias is for GatedDeltaNet, absent from Qwen2.
    for name in (
        "seq_lens",
        "attention_mask_len",
        "linear_attn_cu_seq_lens_q",
        "tail_padding_length",
        "output_hidden_states",
        # Qwen3-MoE forwards this model-level bookkeeping flag through the
        # Transformers attention interface.  Router logits are produced by
        # the MoE block, so muDNN has no attention work to do for this value.
        "output_router_logits",
    ):
        kwargs.pop(name, None)
    _check_options(dropout, scaling, None, softcap, None, False, False, kwargs)
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Transformers muDNN facade requires [B,H,S,D] tensors.")
    batch, _, sequence, _ = query.shape
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != (batch, sequence):
            raise ValueError("Only None or a two-dimensional all-ones padding mask is supported.")
        if attention_mask.requires_grad:
            raise ValueError("Differentiable attention masks are unsupported.")

        def validate_mask():
            if not bool(torch.all(attention_mask.detach().cpu() == 1)):
                raise ValueError("Padding/additive masks are unsupported; use prepacked unpadded self-attention.")

        _cached_validation("mask", (attention_mask,), (batch, sequence), validate_mask)
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    if any(item is not None for item in (cu_q, cu_k, max_q, max_k)):
        if any(item is None for item in (cu_q, cu_k, max_q, max_k)) or batch != 1:
            raise ValueError("Packed attention requires B=1 and all four precomputed sequence metadata fields.")
        output = mudnn_flash_attn_varlen_func(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            cu_q,
            cu_k,
            max_q,
            max_k,
            softmax_scale=scaling,
            causal=causal,
        ).unsqueeze(0)
    else:
        if position_ids is not None:
            raise ValueError(
                "Precompute packed metadata when position_ids are supplied; never infer boundaries per layer."
            )
        output = mudnn_flash_attn_func(q, k, v, softmax_scale=scaling, causal=causal)
    _STATS["transformers_facade_calls"] += 1
    return output, None
