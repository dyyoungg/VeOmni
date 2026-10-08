#!/usr/bin/env python
"""Benchmark SVD and Muon-style owner all-to-all for spectral entropy.

Single GPU:
  python scripts/benchmark_spectral_entropy.py --device cuda

Distributed owner path:
  torchrun --nproc_per_node=8 scripts/benchmark_spectral_entropy.py \
      --device cuda --distributed --sizes 1536x2048 2048x2048 6144x2048
"""

import argparse
import time

import torch
import torch.distributed as dist


def parse_size(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.lower().split("x")
        return int(rows), int(cols)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"matrix size must look like ROWSxCOLS, got {value!r}") from exc


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _entropy(matrix: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(matrix.float())
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
    return -(probabilities * probabilities.clamp_min(torch.finfo(energy.dtype).tiny).log()).sum() / torch.log(
        torch.tensor(float(singular_values.numel()), device=matrix.device)
    )


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
        torch.npu.reset_max_memory_allocated()


def _peak_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        return (
            torch.cuda.max_memory_allocated(device) / 1024**2,
            torch.cuda.max_memory_reserved(device) / 1024**2,
        )
    if device.type == "npu" and hasattr(torch, "npu"):
        return torch.npu.max_memory_allocated() / 1024**2, torch.npu.max_memory_reserved() / 1024**2
    return 0.0, 0.0


def benchmark_svd(
    rows: int, cols: int, device: torch.device, warmup: int, iterations: int
) -> tuple[float, float, float, float]:
    matrix = torch.randn(rows, cols, device=device, dtype=torch.float32)
    for _ in range(warmup):
        _entropy(matrix)
    synchronize(device)
    _reset_peak(device)
    start = time.perf_counter()
    if device.type == "cuda":
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    for _ in range(iterations):
        _entropy(matrix)
    if device.type == "cuda":
        event_end.record()
        event_end.synchronize()
        elapsed_ms = event_start.elapsed_time(event_end) / iterations
    else:
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
    allocated_mib, reserved_mib = _peak_memory_mib(device)
    matrix_mib = matrix.numel() * matrix.element_size() / 1024**2
    del matrix
    return elapsed_ms, matrix_mib, allocated_mib, reserved_mib


def benchmark_owner_all2all(
    rows: int, cols: int, device: torch.device, warmup: int, iterations: int
) -> tuple[float, float, float, float]:
    """Benchmark row gather all-to-all + owner SVD + scalar all-to-all."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    shard_rows = (rows + world - 1) // world
    local_rows = min(shard_rows, max(rows - rank * shard_rows, 0))
    # Every rank owns one row shard for every logical parameter. Parameter j is
    # owned by rank j, matching Muon's bucket assignment.
    local_parameters = [
        torch.randn(local_rows, cols, device=device, dtype=torch.float32) for _ in range(world)
    ]
    group = dist.group.WORLD

    def run_once() -> None:
        send = [local_parameters[index].contiguous() for index in range(world)]
        recv = [
            torch.empty(min(shard_rows, max(rows - source * shard_rows, 0)), cols, device=device)
            for source in range(world)
        ]
        dist.all_to_all(recv, send, group=group)
        owner_value = torch.zeros(1, device=device, dtype=torch.float32)
        if rank < world:
            full = torch.cat(recv, dim=0)
            owner_value = _entropy(full).reshape(1)
            del full
        scalar_send = [owner_value for _ in range(world)]
        scalar_recv = [torch.empty(1, device=device) for _ in range(world)]
        dist.all_to_all(scalar_recv, scalar_send, group=group)

    for _ in range(warmup):
        run_once()
    dist.barrier()
    _reset_peak(device)
    start = time.perf_counter()
    if device.type == "cuda":
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    for _ in range(iterations):
        run_once()
    if device.type == "cuda":
        event_end.record()
        event_end.synchronize()
        elapsed_ms = event_start.elapsed_time(event_end) / iterations
    else:
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
    allocated_mib, reserved_mib = _peak_memory_mib(device)
    dist.barrier()
    summary = torch.tensor([elapsed_ms, allocated_mib, reserved_mib], device=device)
    dist.all_reduce(summary, op=dist.ReduceOp.MAX)
    if rank == 0:
        elapsed_ms, allocated_mib, reserved_mib = summary.tolist()
    del local_parameters
    return elapsed_ms, rows * cols * 4 / 1024**2, allocated_mib, reserved_mib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=parse_size,
        nargs="+",
        default=[(1024, 1024), (1536, 2048), (2048, 2048), (6144, 2048)],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if args.distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        mode = "owner-all2all" if args.distributed else "single-device"
        print(f"mode={mode}, device={device}, torch={torch.__version__}")
        print(f"{'shape':>14} {'matrix MiB':>12} {'time ms':>12} {'peak alloc MiB':>16} {'peak reserved MiB':>19}")
    for rows, cols in args.sizes:
        if args.distributed:
            result = benchmark_owner_all2all(rows, cols, device, args.warmup, args.iterations)
        else:
            result = benchmark_svd(rows, cols, device, args.warmup, args.iterations)
        if rank == 0:
            elapsed_ms, matrix_mib, allocated_mib, reserved_mib = result
            print(f"{rows}x{cols:<8} {matrix_mib:12.1f} {elapsed_ms:12.3f} {allocated_mib:16.1f} {reserved_mib:19.1f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
