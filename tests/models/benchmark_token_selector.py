"""Benchmark DynamicTokenSelectorProjector against dynamic average pooling.

Run from any directory with the repository's environment:

    conda run -n llava python tests/models/benchmark_token_selector.py --device 0

To use two GPUs concurrently, split the cases between two terminals:

    conda run -n llava python tests/models/benchmark_token_selector.py \
        --device 0 --cases small_r4 video_r1 video_r4 video_r12
    conda run -n llava python tests/models/benchmark_token_selector.py \
        --device 1 --cases mixed_pack large_r4

The default input does not require gradients, matching a frozen vision tower.
Pass ``--features-require-grad`` to measure an unfrozen vision tower instead.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ENCODER_HIDDEN = 4608
OUTPUT_HIDDEN = 2048
SELECTOR_HIDDEN = 128
CASE_NAMES = (
    "small_r4",
    "video_r1",
    "video_r4",
    "video_r12",
    "mixed_pack",
    "large_r4",
)


def synchronize() -> None:
    torch.cuda.synchronize()


def measure_latency(module, features, grids, ratios, *, training, warmup, iterations):
    module.train(training)

    def step():
        if training:
            output, _ = module(features, grids, downsample_ratios=ratios)
            output.sum().backward()
        else:
            with torch.no_grad():
                module(features, grids, downsample_ratios=ratios)

    for _ in range(warmup):
        step()
        if training:
            features.grad = None
            for parameter in module.parameters():
                parameter.grad = None
    synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        synchronize()
        samples.append(start.elapsed_time(end))
        if training:
            features.grad = None
            for parameter in module.parameters():
                parameter.grad = None

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.9))],
    }


def measure_memory(module, features, grids, ratios):
    module.train()
    features.grad = None
    for parameter in module.parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    output, seq_len = module(features, grids, downsample_ratios=ratios)
    synchronize()
    retained_after_forward = torch.cuda.memory_allocated()
    forward_peak_allocated = torch.cuda.max_memory_allocated()
    output.sum().backward()
    synchronize()

    result = {
        "output_tokens": sum(seq_len),
        "retained_after_forward_mib": (retained_after_forward - base_allocated) / 2**20,
        "forward_peak_allocated_mib": (forward_peak_allocated - base_allocated) / 2**20,
        "working_peak_allocated_mib": (torch.cuda.max_memory_allocated() - base_allocated) / 2**20,
        "working_peak_reserved_mib": (torch.cuda.max_memory_reserved() - base_reserved) / 2**20,
    }
    del output
    features.grad = None
    for parameter in module.parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    synchronize()
    return result


def parameter_stats(module):
    parameters = sum(parameter.numel() for parameter in module.parameters())
    return {
        "parameters": parameters,
        "bf16_weights_mib": parameters * 2 / 2**20,
        "bf16_gradients_mib": parameters * 2 / 2**20,
        "fp32_adam_states_mib": parameters * 8 / 2**20,
    }


def build_case(case_name, device):
    if case_name == "small_r4":
        grids = [[16, 28, 28]]
        ratios = [4.0]
    elif case_name == "video_r1":
        grids = [[64, 28, 28]]
        ratios = [1.0]
    elif case_name == "video_r4":
        grids = [[64, 28, 28]]
        ratios = [4.0]
    elif case_name == "video_r12":
        grids = [[64, 28, 28]]
        ratios = [12.0]
    elif case_name == "large_r4":
        grids = [[120, 28, 28]]
        ratios = [4.0]
    elif case_name == "mixed_pack":
        grids = [
            [1, 48, 48],
            [16, 28, 50],
            [32, 28, 28],
            [1, 56, 42],
            [16, 28, 28],
        ]
        ratios = [1.0, 2.0, 8.0, 12.0, 4.0]
    else:
        raise ValueError(case_name)

    grids_tensor = torch.tensor(grids, dtype=torch.long, device=device)
    ratios_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
    input_tokens = sum(time * (height // 2) * (width // 2) for time, height, width in grids)
    return grids_tensor, ratios_tensor, input_tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=CASE_NAMES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--features-require-grad", action="store_true")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"device must be in [0, {torch.cuda.device_count() - 1}], got {args.device}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    import veomni.models.custom.llava_qwen3moe.projector as avgpool_module
    import veomni.models.custom.llava_qwen3moe.projector_token_selector as selector_module

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # A training process has initialized ParallelState. Avoid measuring the
    # warning path used only by this isolated benchmark.
    avgpool_module.get_parallel_state = lambda: None
    selector_module.get_parallel_state = lambda: None

    baseline = avgpool_module.DynamicAvgPoolProjector(ENCODER_HIDDEN, OUTPUT_HIDDEN, 4).to(
        device=device, dtype=torch.bfloat16
    )
    selector = selector_module.DynamicTokenSelectorProjector(
        ENCODER_HIDDEN,
        OUTPUT_HIDDEN,
        4,
        selector_hidden=SELECTOR_HIDDEN,
    ).to(device=device, dtype=torch.bfloat16)
    selector.mlp.load_state_dict(baseline.mlp.state_dict())

    result = {
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "features_require_grad": args.features_require_grad,
        "parameter_stats": {
            "avgpool": parameter_stats(baseline),
            "selector": parameter_stats(selector),
        },
        "cases": {},
    }

    for case_name in args.cases:
        grids, ratios, input_tokens = build_case(case_name, device)
        features = torch.randn(
            input_tokens,
            ENCODER_HIDDEN,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=args.features_require_grad,
        )
        with torch.no_grad():
            _, _, selector_aux = selector.select_tokens(features, grids, downsample_ratios=ratios)
        candidate_entries = selector_aux.candidate_indices.numel()
        del selector_aux
        case_result = {
            "input_tokens": input_tokens,
            "ratios": ratios.cpu().tolist(),
            "selector_candidate_entries": candidate_entries,
            "candidate_amplification": candidate_entries / input_tokens,
        }
        for module_name, module in (("avgpool", baseline), ("selector", selector)):
            memory = measure_memory(module, features, grids, ratios)
            inference = measure_latency(
                module,
                features,
                grids,
                ratios,
                training=False,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            train = measure_latency(
                module,
                features,
                grids,
                ratios,
                training=True,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            case_result[module_name] = {
                **memory,
                "forward": inference,
                "forward_backward": train,
            }
        result["cases"][case_name] = case_result
        del features
        torch.cuda.empty_cache()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
