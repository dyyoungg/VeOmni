r"""Compare the generated Qwen3 MoE reference loss with the Triton implementation.

Example (run in a CUDA environment with torch and triton installed)::

    python scripts/benchmark_load_balancing_loss.py \
        --layers 48 --tokens 8192 --experts 128 --top-k 8 --dtype float32

Tokens are per layer, per rank (batch size is one). Forward uses autograd;
backward includes propagation through cat into the per-layer leaf gradients.
CUDA-event timings include launch gaps, exclude input construction and gradient
clearing, and exclude JIT compilation via warmup. No CUDA graphs or autocast.
The reference is extracted verbatim via AST from the generated source, avoiding
unrelated model imports. No generated files are modified.
"""

import argparse
import ast
import importlib.util
import statistics
from pathlib import Path


def load_reference(path, torch):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_balancing_loss_func"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name]


def load_triton(path):
    spec = importlib.util.spec_from_file_location("benchmark_lb_triton", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_balancing_loss_triton


def clear_gradients(logits):
    for tensor in logits:
        tensor.grad = None


def check_correctness(torch, implementations, logits, experts, top_k, mask, rtol, atol):
    reference_loss = None
    reference_grads = None
    passed = True
    for name, function in implementations.items():
        clear_gradients(logits)
        loss = function(logits, experts, top_k, mask)
        loss.backward()
        # Keep comparison tensors on CPU so they cannot inflate benchmark memory.
        value = loss.detach().float().cpu()
        gradients = [tensor.grad.detach().float().cpu() for tensor in logits]
        print(f"{name} loss: {value.item():.9g}")
        if reference_loss is None:
            reference_loss, reference_grads = value, gradients
        else:
            max_error = max(
                (actual - expected).abs().max().item() for actual, expected in zip(gradients, reference_grads)
            )
            print(f"Gradient max absolute difference: {max_error:.6g}")
            try:
                torch.testing.assert_close(value, reference_loss, rtol=rtol, atol=atol)
                # Normalize tiny auxiliary-loss gradients before applying tolerance;
                # otherwise an all-zero gradient could pass a raw absolute check.
                scale = max(expected.abs().max().item() for expected in reference_grads)
                if scale > 0:
                    print(f"Gradient max error / reference max magnitude: {max_error / scale:.6g}")
                for actual, expected in zip(gradients, reference_grads):
                    if scale == 0:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(actual / scale, expected / scale, rtol=rtol, atol=rtol)
            except AssertionError as error:
                passed = False
                print(f"CORRECTNESS FAILED: {error}")
        del loss
    clear_gradients(logits)
    print(f"Correctness: {'PASS' if passed else 'FAIL (timings are diagnostic only)'}")
    return passed


def benchmark(torch, function, logits, experts, top_k, mask, warmup, iterations):
    for _ in range(warmup):
        clear_gradients(logits)
        loss = function(logits, experts, top_k, mask)
        loss.backward()
        del loss
    clear_gradients(logits)
    torch.cuda.synchronize()
    samples = []
    start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    for _ in range(iterations):
        clear_gradients(logits)
        start.record()
        loss = function(logits, experts, top_k, mask)
        middle.record()
        loss.backward()
        end.record()
        end.synchronize()
        samples.append((start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)))
        del loss
    clear_gradients(logits)
    return [statistics.median(sample[index] for sample in samples) for index in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--mask", choices=["none", "full", "padded"], default="none")
    parser.add_argument("--valid-fraction", type=float, default=0.8, help="Valid token fraction for padded mask")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-4, help="Loss rtol and normalized gradient rtol/atol")
    parser.add_argument("--atol", type=float, default=1e-8, help="Loss absolute tolerance")
    args = parser.parse_args()
    if min(args.layers, args.tokens, args.experts, args.top_k, args.warmup, args.iterations) < 1:
        parser.error("Shapes, warmup and iterations must be positive")
    if args.top_k > args.experts or not 0 < args.valid_fraction <= 1:
        parser.error("Require top-k <= experts and 0 < valid-fraction <= 1")
    if args.rtol < 0 or args.atol < 0:
        parser.error("Tolerances must be nonnegative")

    import torch
    import triton

    if not torch.cuda.is_available():
        parser.error("A working CUDA device is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    implementations = {
        "hf_generated": load_reference(
            root / "veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py", torch
        ),
        "triton": load_triton(root / "veomni/ops/kernels/load_balancing_loss/triton.py"),
    }
    logits = tuple(
        torch.randn(args.tokens, args.experts, device="cuda", dtype=getattr(torch, args.dtype), requires_grad=True)
        for _ in range(args.layers)
    )
    mask = None
    if args.mask != "none":
        mask = torch.ones(1, args.tokens, device="cuda", dtype=torch.long)
        if args.mask == "padded":
            mask[:, max(1, int(args.tokens * args.valid_fraction)) :] = 0
    print(f"GPU: {torch.cuda.get_device_name()} | torch={torch.__version__} | triton={triton.__version__}")
    print(vars(args))
    print("Reference softmax uses input dtype; Triton uses FP32 internally. Low-precision top-k ties may differ.")
    passed = check_correctness(torch, implementations, logits, args.experts, args.top_k, mask, args.rtol, args.atol)
    results = {}
    print(f"{'implementation':<18} {'forward ms':>12} {'backward ms':>12} {'fwd+bwd ms':>12} (medians)")
    for name, function in implementations.items():
        results[name] = benchmark(
            torch, function, logits, args.experts, args.top_k, mask, args.warmup, args.iterations
        )
        print(f"{name:<18} " + " ".join(f"{value:12.4f}" for value in results[name]), flush=True)
    ratios = [reference / optimized for reference, optimized in zip(results["hf_generated"], results["triton"])]
    print("HF / Triton speedup (forward, backward, total): " + ", ".join(f"{ratio:.3f}x" for ratio in ratios))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
