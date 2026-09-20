"""Single-GPU Qwen3-VL decoder benchmark; run from the repository root.

Measures training-mode forward and forward+backward, without optimizer, data
loading, loss computation, FSDP or SP. Both paths use identical weights, inputs
and FlashAttention. First-call latency includes any compilation/cache loading.
"""

import argparse
import copy
import json
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=2816)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--freeze-weights", action="store_true", help="Backward computes input gradients only.")
    parser.add_argument(
        "--check-only", action="store_true", help="Report numerical differences without timing rounds."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON results file.")
    args = parser.parse_args()
    for key in ("seq_len", "hidden_size", "intermediate_size", "heads", "kv_heads", "warmup", "iterations", "rounds"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.hidden_size % args.heads or args.heads % args.kv_heads:
        parser.error("hidden-size must be divisible by heads; heads must be divisible by kv-heads")
    if (args.hidden_size // args.heads) % 8:
        parser.error("head dimension must be a multiple of 8 for this FlashAttention benchmark")
    return args


def comparison_metrics(actual, reference):
    """Measure error against eager in FP64, outside the timed GPU section."""
    import torch

    actual, reference = actual.double(), reference.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    mismatch = ~torch.isclose(actual, reference, rtol=2e-2, atol=2e-2)
    result = {"elements": reference.numel(), "mismatched": int(mismatch.sum()), "finite": finite}
    if finite:
        difference = actual - reference
        error_norm = difference.norm().item()
        reference_norm = reference.norm().item()
        result.update(
            max_abs=difference.abs().max().item(),
            rmse=difference.square().mean().sqrt().item(),
            reference_rms=reference.square().mean().sqrt().item(),
            relative_l2=error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else None),
        )
    return result


def save_results(path, results):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(f"Saved {path}")


def main():
    args = parse_args()
    import torch
    from torch.utils.checkpoint import checkpoint

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this benchmark on an idle GPU in your training environment.")

    from veomni.data.data_collator import add_flash_attention_kwargs_from_position_ids
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.distributed.torch_compile import CompileConfig, compile_decoder_blocks
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLTextConfig,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    torch.manual_seed(1234)
    device = torch.device("cuda", 0)
    dtype = getattr(torch, args.dtype)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.hidden_size // args.heads,
        max_position_embeddings=args.seq_len,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    eager = Qwen3VLTextDecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).train()
    if args.freeze_weights:
        eager.requires_grad_(False)
    compiled = copy.deepcopy(eager)
    if args.checkpoint:
        for layer in (eager, compiled):
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    container = torch.nn.Module()
    container._no_split_modules = ["Qwen3VLTextDecoderLayer"]
    container.config = SimpleNamespace(model_type="qwen3_vl", vision_config=SimpleNamespace())
    container.layer = compiled
    inputs = torch.randn(1, args.seq_len, args.hidden_size, device=device, dtype=dtype, requires_grad=True)
    positions = torch.arange(args.seq_len, device=device).unsqueeze(0)
    rotary = Qwen3VLTextRotaryEmbedding(config).to(device)
    with torch.no_grad():
        position_embeddings = rotary(inputs, positions)
    batch = {"position_ids": positions}
    add_flash_attention_kwargs_from_position_ids(batch)
    kwargs = {k: batch[k] for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")}
    upstream = torch.randn_like(inputs)
    models = {"eager": eager, "compiled": compiled}

    def clear():
        inputs.grad = None
        for model in models.values():
            model.zero_grad(set_to_none=True)

    def step(model, backward):
        model.zero_grad(set_to_none=True)
        inputs.grad = None
        output = model(inputs, position_embeddings=position_embeddings, **kwargs)
        if backward:
            output.backward(upstream)
        return output

    def snapshot(model):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = step(model, True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        values = {"output": output.detach().cpu(), "input_grad": inputs.grad.detach().cpu()}
        values.update(
            {name: param.grad.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
        )
        clear()
        return elapsed, values

    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "scope": "one decoder, batch=1, training forward/backward; no optimizer, loss, FSDP or SP",
    }
    with use_parallel_state(SimpleNamespace(async_enabled=False)):
        assert compile_decoder_blocks(container, CompileConfig(enable=True, fullgraph=True, dynamic=False)) == 1
        print("Checking outputs/gradients; first Inductor call may take minutes.", flush=True)
        eager_first, reference = snapshot(eager)
        compiled_first, actual = snapshot(compiled)
        metrics, failures = {}, []
        for name in reference:
            metrics[name] = comparison_metrics(actual[name], reference[name])
            print(f"{name}: {json.dumps(metrics[name], allow_nan=False)}", flush=True)
            try:
                torch.testing.assert_close(actual[name], reference[name], rtol=2e-2, atol=2e-2)
            except AssertionError as error:
                failures.append(f"{name}: {error}")
        del reference, actual
        results["first_forward_backward_seconds"] = {"eager": eager_first, "compiled": compiled_first}
        results["accuracy"] = {"rtol": 2e-2, "atol": 2e-2, "tensors": metrics, "failures": failures}
        if failures:
            save_results(args.output, results)
            raise SystemExit("Correctness FAILED; no speed measurement.\n" + "\n".join(failures))
        print(f"Correctness PASS. First F+B: eager={eager_first:.3f}s, compiled={compiled_first:.3f}s", flush=True)
        if args.check_only:
            save_results(args.output, results)
            return

        for label, backward in (("forward", False), ("forward_backward", True)):
            timings = {name: [] for name in models}
            peaks = {name: [] for name in models}
            for model in models.values():
                for _ in range(args.warmup):
                    step(model, backward)
            torch.cuda.synchronize()
            for round_idx in range(args.rounds):
                # Alternate order to reduce thermal/order bias. Synchronize only
                # around each batch of steps so normal CPU/GPU overlap remains.
                order = list(models) if round_idx % 2 == 0 else list(reversed(models))
                for name in order:
                    clear()
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        step(models[name], backward)
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000 / args.iterations)
                    peaks[name].append((torch.cuda.max_memory_allocated(device) - baseline) / 2**20)
            medians = {name: statistics.median(values) for name, values in timings.items()}
            speedup = medians["eager"] / medians["compiled"]
            results[label] = {
                "median_ms": medians,
                "round_ms": timings,
                "speedup": speedup,
                "latency_reduction_percent": (1 - medians["compiled"] / medians["eager"]) * 100,
                "peak_incremental_allocated_mib": {name: max(values) for name, values in peaks.items()},
            }
            print(f"{label}: eager={medians['eager']:.3f}ms, compiled={medians['compiled']:.3f}ms, {speedup:.3f}x")
            print(f"  Additional peak allocated MiB: {results[label]['peak_incremental_allocated_mib']}", flush=True)
    save_results(args.output, results)


if __name__ == "__main__":
    main()
