"""Standalone muDNN checks; bypasses VeOmni's heavyweight pytest conftest.

Run ``python tests/ops/test_mudnn_attention.py --long-iters 3`` on S5000.
``--cpu-only`` checks option rejection without requiring torch-musa.
The 4096-token run checks repeated forward/backward finiteness, not complete
FP32 equivalence or end-to-end distributed training stability.
"""

import argparse
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


_ADAPTER_PATH = Path(__file__).resolve().parents[2] / "veomni/ops/kernels/attention/mudnn.py"
_SPEC = importlib.util.spec_from_file_location("standalone_mudnn_attention", _ADAPTER_PATH)
mudnn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mudnn)
_LONG_ITERS = 3
_CPU_ONLY = False


def _has_musa():
    return not _CPU_ONLY and hasattr(torch, "musa") and torch.musa.is_available()


def _reference(q, k, v, boundaries, causal, scale):
    outputs = []
    groups = q.shape[1] // k.shape[1]
    for start, end in zip(boundaries, boundaries[1:]):
        q_part = q[start:end].transpose(0, 1)
        k_part = k[start:end].repeat_interleave(groups, dim=1).transpose(0, 1)
        v_part = v[start:end].repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = (q_part @ k_part.transpose(-1, -2)) * scale
        if causal:
            mask = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        outputs.append((scores.softmax(dim=-1) @ v_part).transpose(0, 1))
    return torch.cat(outputs)


def _inputs(dtype, heads_q, heads_kv, dim, total=48):
    torch.manual_seed(173)
    cpu = [(torch.randn(total, heads, dim) * 0.2).to(dtype) for heads in (heads_q, heads_kv, heads_kv)]
    device = [tensor.to("musa").requires_grad_(True) for tensor in cpu]
    reference = [tensor.float().requires_grad_(True) for tensor in cpu]
    return device, reference


def _metrics(reference, actual):
    actual = actual.detach().float().cpu()
    reference = reference.detach().float()
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "rel_l2": (delta.norm() / reference.norm().clamp_min(1e-12)).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


class TestMuDNNOptions(unittest.TestCase):
    def test_unsupported_options_fail_before_dispatch(self):
        q = torch.empty(8, 4, 64)
        cu = torch.tensor([0, 8], dtype=torch.int32)
        base = (q, q, q, cu, cu, 8, 8)
        invalid = (
            {"dropout_p": 0.1},
            {"window_size": (32, 0)},
            {"alibi_slopes": torch.ones(4)},
            {"softcap": 1.0},
            {"deterministic": True},
            {"return_attn_probs": True},
            {"unexpected_arg": None},
            {"softmax_scale": math.nan},
        )
        for options in invalid:
            with self.subTest(options=list(options)), self.assertRaises((ValueError, TypeError, NotImplementedError)):
                mudnn.mudnn_flash_attn_varlen_func(*base, **options)

    def test_cpu_tensors_are_rejected(self):
        q = torch.empty(8, 4, 64, dtype=torch.bfloat16)
        cu = torch.tensor([0, 8], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "MUSA"):
            mudnn.mudnn_flash_attn_varlen_func(q, q, q, cu, cu, 8, 8)

    def test_sequence_parallel_is_rejected(self):
        state = SimpleNamespace(sp_enabled=True, ulysses_size=2, cp_size=1)
        state_module = SimpleNamespace(get_parallel_state=lambda: state)
        with patch.dict(sys.modules, {"veomni.distributed.parallel_state": state_module}):
            with self.assertRaisesRegex(NotImplementedError, "SP/CP"):
                mudnn._check_parallel()


class TestMuDNNDevice(unittest.TestCase):
    def setUp(self):
        if not _has_musa():
            self.skipTest("A usable MUSA device is required.")

    def test_small_numeric(self):
        for dtype in (torch.bfloat16, torch.float16):
            for heads_q, heads_kv, dim, causal in ((40, 8, 128, True), (16, 16, 80, False)):
                with self.subTest(dtype=dtype, heads=(heads_q, heads_kv), dim=dim, causal=causal):
                    device, reference = _inputs(dtype, heads_q, heads_kv, dim)
                    if dtype == torch.bfloat16:
                        # Match Qwen's transposed packed views, rather than
                        # testing contiguous random tensors exclusively.
                        device = [
                            tensor.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_(True)
                            for tensor in device
                        ]
                    q, k, v = device
                    boundaries = [0, 17, 48]
                    cu_q = torch.tensor(boundaries, dtype=torch.int32, device="musa")
                    cu_k = cu_q.clone()
                    output = mudnn.mudnn_flash_attn_varlen_func(q, k, v, cu_q, cu_k, 31, 31, causal=causal)
                    self.assertIn("FlashAttentionBackward", type(output.grad_fn).__name__)
                    expected = _reference(*reference, boundaries, causal, dim**-0.5)
                    grad = (torch.randn_like(expected) * 0.2).to(dtype)
                    expected.backward(grad.float())
                    output.backward(grad.to("musa"))
                    torch.musa.synchronize()
                    results = {
                        name: _metrics(ref, got)
                        for name, ref, got in zip(
                            ("output", "dq", "dk", "dv"),
                            (expected, *(tensor.grad for tensor in reference)),
                            (output, *(tensor.grad for tensor in device)),
                        )
                    }
                    print(
                        json.dumps(
                            {
                                "event": "numeric",
                                "dtype": str(dtype),
                                "heads": [heads_q, heads_kv],
                                "dim": dim,
                                "causal": causal,
                                "metrics": results,
                            }
                        ),
                        flush=True,
                    )
                    for result in results.values():
                        self.assertTrue(result["finite"])
                        self.assertLess(result["rel_l2"], 0.02 if dtype == torch.bfloat16 else 0.003)

    def test_dense_and_transformers_facades(self):
        device, _ = _inputs(torch.bfloat16, 40, 8, 128)
        q, k, v = device
        cu = torch.tensor([0, 17, 48], dtype=torch.int32, device="musa")
        expected = mudnn.mudnn_flash_attn_varlen_func(q, k, v, cu, cu, 31, 31, causal=True)
        before = mudnn.get_mudnn_call_stats().get("metadata_validations", 0)
        result, weights = mudnn.mudnn_attention_forward(
            SimpleNamespace(is_causal=True),
            *(tensor.unsqueeze(0).transpose(1, 2) for tensor in device),
            attention_mask=torch.ones(1, 48, device="musa"),
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=31,
            max_length_k=31,
            position_ids=torch.arange(48, device="musa").unsqueeze(0),
            use_cache=False,
            cache_position=torch.arange(48, device="musa"),
            seq_lens=torch.tensor([17, 31], device="musa"),
            linear_attn_cu_seq_lens_q=cu,
            attention_mask_len=[17, 31],
            tail_padding_length=0,
            output_hidden_states=False,
            output_router_logits=True,
        )
        self.assertIsNone(weights)
        torch.testing.assert_close(result.squeeze(0), expected, atol=0.01, rtol=0.01)
        self.assertEqual(mudnn.get_mudnn_call_stats().get("metadata_validations", 0), before)
        result.sum().backward()
        self.assertTrue(all(tensor.grad is not None for tensor in device))
        dense = [tensor.detach().reshape(2, 24, *tensor.shape[-2:]).requires_grad_(True) for tensor in device]
        actual = mudnn.mudnn_flash_attn_func(*dense, causal=False)
        refs = [tensor.detach().float().cpu().reshape(48, *tensor.shape[-2:]) for tensor in dense]
        expected = _reference(*refs, [0, 24, 48], False, 128**-0.5)
        self.assertLess(_metrics(expected, actual.reshape_as(expected))["rel_l2"], 0.02)

    def test_malformed_inputs_and_metadata_mutation(self):
        device, _ = _inputs(torch.bfloat16, 40, 8, 128)
        q, k, v = device
        cu = torch.tensor([0, 17, 48], dtype=torch.int32, device="musa")
        fn = mudnn.mudnn_flash_attn_varlen_func
        fn(q, k, v, cu, cu, 31, 31)
        bad_cases = (
            (q, k, v, cu.long(), cu, 31, 31),
            (q, k, v, cu, cu, 16, 31),
            (q[:, :39], k, v, cu, cu, 31, 31),
            (q.float(), k, v, cu, cu, 31, 31),
            (q, k, v, cu, cu[:2], 31, 31),
            (q, k, v, torch.tensor([0, 0, 48], device="musa", dtype=torch.int32), cu, 48, 48),
        )
        for args in bad_cases:
            with self.assertRaises(ValueError):
                fn(*args)
        cu[-1] = 47
        with self.assertRaisesRegex(ValueError, "packed token count"):
            fn(q, k, v, cu, cu, 31, 31)
        with self.assertRaisesRegex(ValueError, "Padding/additive"):
            mudnn.mudnn_attention_forward(
                SimpleNamespace(is_causal=True),
                *(tensor.unsqueeze(0).transpose(1, 2) for tensor in device),
                attention_mask=torch.zeros(1, 48),
            )

    def test_z_long_repeated_backward(self):
        if _LONG_ITERS <= 0:
            self.skipTest("Long-sequence probe disabled.")
        for heads_q, heads_kv, dim, causal in ((40, 8, 128, True), (16, 16, 80, False)):
            device, _ = _inputs(torch.bfloat16, heads_q, heads_kv, dim, total=4096)
            q, k, v = device
            cu = torch.tensor([0, 4096], dtype=torch.int32, device="musa")
            grad = torch.randn_like(q) * 0.01
            baseline = None
            for iteration in range(_LONG_ITERS):
                for tensor in device:
                    tensor.grad = None
                output = mudnn.mudnn_flash_attn_varlen_func(q, k, v, cu, cu, 4096, 4096, causal=causal)
                output.backward(grad)
                torch.musa.synchronize()
                self.assertTrue(
                    all(bool(torch.isfinite(tensor).all()) for tensor in (output, *(x.grad for x in device)))
                )
                if baseline is None:
                    baseline = output.detach().clone()
                else:
                    torch.testing.assert_close(output, baseline, rtol=0.02, atol=0.02)
                print(
                    json.dumps(
                        {
                            "event": "long_fwd_bwd_finite",
                            "heads": [heads_q, heads_kv],
                            "dim": dim,
                            "total_tokens": 4096,
                            "boundaries": [0, 4096],
                            "causal": causal,
                            "iteration": iteration + 1,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--long-iters", type=int, default=3)
    arguments = parser.parse_args()
    _CPU_ONLY, _LONG_ITERS = arguments.cpu_only, arguments.long_iters
    if not _CPU_ONLY:
        import torch_musa  # noqa: F401

    if not _CPU_ONLY and not _has_musa():
        raise SystemExit("MUSA unavailable; use --cpu-only explicitly for option-only checks.")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "event": "mudnn_test_summary",
                "success": result.wasSuccessful(),
                "cpu_only": _CPU_ONLY,
                "stats": mudnn.get_mudnn_call_stats(),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
