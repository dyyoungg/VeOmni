---
name: torch-perf-optimize
description: "Torch training performance optimization methodology. Covers the full diagnostic pipeline: baseline → Nsight Systems (GPU bubble, comm/compute overlap, CPU/GIL) → PyTorch Profiler (hotspot ops) → Nsight Compute (roofline, SOL, occupancy, warp stall). Use when diagnosing training slowness, analyzing nsys/ncu reports, or planning optimization strategy. Trigger: 'training slow', 'nsys', 'ncu', 'nsight', 'bubble', 'overlap', 'roofline', 'SOL', 'occupancy', 'warp stall', 'GIL', 'step time', 'GPU idle', 'communication overhead', 'kernel optimization'."
---

## Overview

Performance problems split into three layers. Always work top-down:

1. **System layer** — Why is the GPU not working continuously?
   - Tool: Nsight Systems → Bubble, CPU stall, H2D, sync, NCCL, straggler
2. **Operator layer** — What runs when the GPU is busy?
   - Tool: PyTorch Profiler → hotspot ops, short kernels, shapes, call stacks
3. **Kernel layer** — Why is this hotspot slow, and how much headroom remains?
   - Tool: Nsight Compute → Roofline, SOL, occupancy, warp stall, memory access

Never start from the bottom. Running `ncu` on a full training run produces a massive, slow, hard-to-interpret report. Find the critical path in nsys first, then target 1–5 kernel launches.

---

## 0. Establish Baseline

Fix these before profiling: model version, effective batch / non-padding tokens, sequence length distribution, precision, parallelism strategy, GPU model, software versions.

Warm up until step time stabilizes, then record:

- **Effective throughput**: samples/s or non-padding tokens/s
- **Step time**: median, P90, P99
- **MFU**
- **Peak memory**: `torch.cuda.max_memory_allocated()`
- **Correctness**: loss curve, gradient norms, or short-run trajectory

### Timing GPU regions correctly

CUDA is asynchronous. Never wrap GPU code with `time.time()`. Use CUDA Events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
loss = model(inputs).loss
end.record()
torch.cuda.synchronize()  # only sync when reading the result
elapsed_ms = start.elapsed_time(end)
```

Do NOT insert `torch.cuda.synchronize()` around every sub-region — it breaks H2D/compute and comm/compute overlap that may exist in production.

### NVTX markers

Tag business phases for easier trace reading:

```python
torch.cuda.nvtx.range_push("forward")
try:
    output = model(inputs)
finally:
    torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("backward")
try:
    output.loss.backward()
finally:
    torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("optimizer")
try:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
finally:
    torch.cuda.nvtx.range_pop()
```

In distributed training, also tag DP/TP/EP collectives with logical names. Otherwise dozens of `ncclAllReduce` calls are indistinguishable.

---

## 1. Nsight Systems: Find GPU Bubbles

### 1.1 What is a Bubble

A bubble is any interval on the critical path of a steady-state step where the GPU is not executing computation, copy, or communication that advances that step.

It is NOT simply "a stream with no kernel". Multi-stream programs may have one stream idle while another works. Diagnose by selecting 5–20 stable steps and viewing GPU streams, CUDA API, CPU thread state, and NVTX ranges together.

Minimum collection command:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --output=profile \
  python train.py
```

- `cuda`: CPU CUDA API + GPU workload
- `nvtx`: maps workload back to forward/backward
- `osrt`: reveals locks, condition variables, thread scheduling

Capture window must cover steady-state iterations. Exclude initialization, JIT compilation, and checkpoint saves.

### 1.2 Reading the timeline: start from blank space, trace upward

Select a blank region on the GPU CUDA streams, then look up the timeline. It typically falls into one of four categories:

| What you see before/after the bubble | Likely cause | Next verification |
|--------------------------------------|-------------|-------------------|
| CPU thread has no CUDA API calls for a long time | Dataloader, tokenize, Python logic, GIL, or CPU launch can't keep up | CPU sampling, py-spy, dataloader segment timing |
| CPU stuck in `cudaStreamSynchronize`, `cudaEventSynchronize`, `.item()` | Implicit sync breaks the async pipeline | Find the call stack, remove unnecessary scalar reads/syncs |
| H2D copy right before compute, copy engine not overlapped with compute | Pageable memory, no prefetch stream, compute window too short | Pin memory, `non_blocking=True`, separate stream + Event |
| Collective finishes then next kernel starts, or one rank arrives much later | Exposed communication, late gradient ready, straggler, load imbalance | Multi-rank traces, comm/compute overlap, idle breakdown |

Common misdiagnosis: a short CUDA API call does NOT mean the GPU has received work. CPU launch, driver scheduling, stream dependencies, and preceding kernels can all delay launch-to-execution. Always cross-check CPU CUDA API row and GPU stream row.

### 1.3 The most commonly missed bubble: CPU is busy, but busy doing the wrong thing

When GPU is idle, CPU thread state showing "Running" does NOT mean CPU is healthy. It may be doing per-sample Python collation, expensive serialization, or launching floods of tiny kernels. Conversely, CPU showing "Waiting" is more likely I/O, locks, sync API, or waiting on another rank.

Use nsys CPU sampling / call stacks to locate the function first, then decide if `perf` is needed.

- `dataloader_iter_next` high → break down into read, decode, tokenize, collate, IPC
- Floods of short CUDA launches → consider `torch.compile` or CUDA Graph
- Don't blindly increase `num_workers` from 4 to 32 — too many workers create contention and serialization overhead

### Quantifying Bubble Ratio

```
Bubble_ratio = Σ(GPU idle intervals in steady-state window) / total_steady_state_window
```

This is a before/after diagnostic metric for the same workload, NOT a cross-model KPI. For multi-stream and multi-rank, compute over the critical path, not a single stream, to avoid double counting.

---

## 2. Communication and Compute Overlap

### 2.1 Timeline overlap ≠ performance gain

Let a collective's GPU interval be M, and the parallel compute interval be C.

- **Time overlap**: `overlap = intersect(M, C)`
- **Exposed communication** (still on critical path): `exposed = M - overlap`

Even with temporal overlap, comm and compute may contend for SM, HBM, NVLink, or PCIe. "More overlap" can make step time worse.

Three gates for real overlap:

1. **Temporally concurrent**: NCCL activity and useful compute on different streams at the same time
2. **Dependency-covered**: collective finishes before its consumer actually needs the result
3. **End-to-end beneficial**: exposed comm or step time decreased; not trading slower compute for a pretty overlap

### 2.2 How to verify in nsys

Align backward NVTX ranges, NCCL row, and GPU compute streams per rank:

- **DDP**: first all-reduce should appear soon after the first gradients are ready. If it only appears after backward fully completes → bucket too large or gradient-ready order doesn't provide a window.
- **Watch the last collective's tail**: how much it extends beyond the last backward kernel is roughly the upper bound of exposed communication.
- **Compare all ranks**: which rank enters the first/last collective latest? Other ranks waiting is usually NOT slow network — it's that rank's earlier data/compute path being slower.
- Copy-engine collectives may not show as normal GPU kernels. Don't only watch one NCCL kernel row — combine CUDA/NCCL API with corresponding GPU activity.

HTA (Holistic Trace Analysis) can summarize across ranks:

```python
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="./traces")
analyzer.get_temporal_breakdown()
analyzer.get_idle_time_breakdown()
analyzer.get_comm_comp_overlap()
```

HTA finds "who's slowest" and "did coverage change". Final attribution still requires nsys to see what dependency constrains that collective.

### 2.3 Tuning knobs

- **DDP `bucket_cap_mb`**: no magic default. Too large → first all-reduce late, no overlap window. Too small → flood of tiny collectives and launch overhead. A/B test a few values; measure first-collective start time, tail exposure, step time, throughput.
- **FSDP wrap granularity**: determines all-gather/reduce-scatter size and timing. One big unit → no overlap opportunity. Too fine → collective count explodes. Start from Transformer block boundaries, adjust by trace.

---

## 3. Confirming GIL as the Bottleneck

GIL is CPython's "only one thread executes Python bytecode at a time" lock. It is NOT synonymous with "low CPU utilization", and it is NOT a shared lock between `DataLoader` worker processes (those are separate processes).

### 3.1 py-spy usage

Two short samples on the training process:

```bash
# All Python stacks
py-spy record --pid <PID> --duration 20 -o all.svg

# Only stacks holding the GIL
py-spy record --pid <PID> --duration 20 --gil -o gil.svg
```

Decision logic:

| Observation | Conclusion |
|-------------|-----------|
| `all.svg` CPU time in Python collate/string/JSON/per-sample loops AND `gil.svg` concentrates on the same thread and stack | GIL serialization is suspicious |
| `all.svg` hotspot mainly in C/C++ extensions, `gil.svg` is not hot | Extension likely released GIL; do NOT misdiagnose as GIL problem |
| `gil.svg` is hot, but GPU has no bubble | May not be worth fixing — optimize only what's on the critical path |

Note: `--gil` only keeps samples where the sampled thread holds the GIL. It misses running native extensions that have released it. It confirms Python serial sections but CANNOT alone measure total GIL wait time.

### 3.2 Cross-validate with nsys

nsys cannot directly label an interval as "GIL wait". Its value is aligning CPU thread state, CUDA API, and GPU bubbles:

1. During GPU bubble, is the training main thread in a Python hotspot?
2. Are other CPU threads waiting long while only one thread executes Python logic?
3. After converting the hotspot to batched/vectorized/C++/CUDA or process-parallel, did bubble ratio and step time decrease?

All three pieces of evidence together → "GIL is an end-to-end bottleneck". A py-spy flame graph alone is insufficient.

---

## 4. Occupancy: A Clue, Not a Scoreboard

Achieved occupancy in Nsight Compute: ratio of actually active warps to the theoretical maximum active warps on an SM. Constrained by block size, registers, shared memory, block count, and architecture.

High occupancy lets the SM switch to another warp when one stalls on memory or instruction dependency. But it is NOT a performance target:

- Large GEMMs may saturate Tensor Cores even with moderate occupancy
- Bandwidth-bound kernels may hit HBM limits regardless of occupancy
- Forcing lower register usage to raise occupancy can cause spills → slower
- Tiny grids may not have enough blocks to fill all SMs regardless of per-block config

**The right question**: Does this kernel have hideable latency AND insufficient active warps to hide it?

| NCU observation | Likely explanation | Action |
|----------------|-------------------|--------|
| Low occupancy, but near compute or memory roofline | Resource usage is a reasonable tradeoff; usually not the first thing to fix | Keep implementation, check other end-to-end bottlenecks first |
| Low occupancy, far from both rooflines, warp state shows long latency | Active warps may be insufficient to hide latency | Check register/shared memory limits, block size, memory access, dependency chains |
| High occupancy, DRAM throughput near roof | Bandwidth is the primary limiter | Fuse ops, reduce intermediate tensors, improve data reuse |
| High occupancy, throughput still low, many short kernels | Single-kernel metrics are misleading | `torch.compile`, CUDA Graph, fusion |

In the NCU report: check Occupancy and Launch Statistics for the limiting resource, then Warp State Statistics for the wait cause. Don't just tweak block size and watch occupancy go up — validate with kernel time and end-to-end throughput.

---

## 5. Is This Kernel Near SOL (Speed of Light)?

SOL in NCU: hardware resource utilization high-level view. Practical definition: **Is this kernel near the achievable ceiling set by its dominant resource, for this shape, dtype, and GPU?**

The ceiling comes from the Roofline:

```
Achievable_FLOPS = min(Peak_FLOPS, Peak_BW × Arithmetic_Intensity)
where Arithmetic_Intensity = FLOPs / Bytes_transferred
```

The dominant roof can change with different batch sizes or matrix shapes for the same code.

### 5.1 Minimum NCU collection

First identify the kernel name, stable launch count, and NVTX range from PyTorch Profiler or nsys. Then collect a few launches:

```bash
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section SpeedOfLight_RooflineChart \
  --kernel-name regex:"gemm|matmul|xmma" \
  --launch-skip 20 \
  --launch-count 5 \
  -o kernel_report \
  python train.py
```

Section names vary by NCU version. Run `ncu --list-sections` to check. Do NOT use `--set full` on an entire training run — hardware counter replay massively inflates runtime.

### 5.2 Decision table: roof first, then percentages

| Report pattern | Conclusion | Priority action |
|---------------|-----------|-----------------|
| Roofline point near the slope, DRAM throughput high | Near memory roof; "SOL" = bandwidth ceiling | Reduce read/write bytes, fuse, improve data reuse |
| Roofline point near the plateau, Tensor Core/SM throughput high | Near compute roof | Check shape alignment, precision, algorithmic FLOPs; otherwise don't force-optimize |
| Point far from both roofs AND low occupancy, visible warp stalls | Non-roofline bottleneck exists | Check dependencies, registers, cache misses, memory coalescing, block/grid size |
| Point far from both roofs, but timeline shows dense short kernels | Per-kernel metrics are limited in meaning | Prioritize fusion, compile, or CUDA Graph |

Quantify "near SOL" as the achieved fraction of the dominant roof:

```
SOL% = measured_throughput / dominant_roof_throughput × 100%
```

Compare within the same GPU, software version, and measurement setup. Don't set a universal "90% threshold" — library implementation, shape, cache hierarchy, instruction mix, and counter methodology all affect it. The most reliable comparison: same kernel before vs after a change, and vs a mature library implementation for the same problem.

### Two commonly confused concepts

- **`nvidia-smi` GPU Util**: sampling-window "is the GPU busy" — good for direction, bad for root cause
- **NCU SOL / Roofline**: this specific kernel's utilization vs hardware ceiling — good for deciding whether to keep optimizing

A training job can have 95% GPU Util while being bottlenecked by inefficient communication or bandwidth-bound kernels. Conversely, GPU Util can be low while the only hotspot GEMM is already near compute roof.

---

## 6. Symptom → Action Decision Table

| Symptom | First tool | Action after sufficient evidence |
|---------|-----------|--------------------------------|
| GPU has large bubbles, CPU has no CUDA API calls | nsys + CPU sampling / py-spy | Optimize data pipeline, batch Python logic, reduce launches, or address GIL |
| H2D copy blocks compute | nsys copy engine / streams | Pin memory, `non_blocking=True`, separate prefetch stream; use Event for dependencies |
| Multi-GPU throughput far below linear scaling | Multi-rank nsys + HTA | Reduce exposed communication; tune bucket/wrap/prefetch; locate straggler rank |
| Top time in elementwise/dropout/norm short kernels | PyTorch Profiler + nsys | `torch.compile`, fusion, CUDA Graph; first confirm shape stability |
| Large GEMM hotspot | NCU Roofline | If compute-bound: check Tensor Core usage and matrix alignment; if already near roof: change optimization direction |
| Kernel low occupancy and far from roofline | NCU occupancy + warp state | Adjust implementation based on limiting resource and stall cause; don't chase occupancy alone |
| OOM / high memory | PyTorch memory snapshot | Check activation lifetimes, then choose checkpointing, sharding, or offload |

---

## 7. Validation: Not a Pretty Screenshot

Change one variable at a time. After each change (e.g., adjusting DDP bucket), compare at minimum:

- p50 / P90 step time and effective throughput
- Last collective's exposed tail
- Bubble ratio or data-wait time
- Peak memory
- Loss / gradient / short-run training consistency

A kernel that is 20% faster but was only 5% of step time, or that breaks comm/compute overlap, can make end-to-end throughput worse. **Performance optimization is not score-chasing on a single metric — it's shortening the critical path.**

### Maintain a benchmark

Keep a fixed-input benchmark in the repo with structured output: commit hash, config, hardware/software versions, throughput, step time distribution, peak memory, correctness result, and profile link. This is the only way to know whether a regression after upgrading torch/compile/CUDA/NCCL/model is real or a baseline shift.
