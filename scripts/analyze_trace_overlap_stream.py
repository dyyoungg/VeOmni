"""
流式分析 Perfetto trace: FSDP2 通信 vs 计算 overlap。
使用 ijson 流式解析，不会 OOM。

python scripts/analyze_trace_overlap_stream.py profile_json/seq24k_rank0.pt.trace.json.gz
"""
import gzip
import sys
import ijson
from collections import defaultdict
import bisect


def stream_events(path):
    """流式 yield trace events"""
    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")

    # Chrome trace format: {"traceEvents": [...]} or just [...]
    # Try traceEvents first
    try:
        parser = ijson.items(f, "traceEvents.item")
        count = 0
        for evt in parser:
            count += 1
            yield evt
        if count > 0:
            f.close()
            return
    except Exception:
        f.seek(0)

    # Try as plain array
    parser = ijson.items(f, "item")
    for evt in parser:
        yield evt
    f.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile_json/seq24k_rank0.pt.trace.json.gz"

    # Collect intervals by category
    # We focus on stream-level GPU events
    nccl_ag = []       # all-gather intervals (start_us, end_us)
    nccl_rs = []       # reduce-scatter
    nccl_other = []    # other nccl
    compute = []       # compute kernels on GPU
    fsdp_markers = []  # FSDP CPU markers
    step_markers = []  # ProfilerStep markers

    # Stats
    total_events = 0
    categories = defaultdict(int)
    name_samples = defaultdict(int)

    print(f"Streaming {path} ...")

    for evt in stream_events(path):
        total_events += 1
        if total_events % 500000 == 0:
            print(f"  processed {total_events} events...", flush=True)

        ph = evt.get("ph", "")
        if ph != "X":
            # Only duration events
            continue

        name = evt.get("name", "")
        cat = evt.get("cat", "")
        ts = float(evt.get("ts", 0))
        dur = float(evt.get("dur", 0))

        if dur <= 0:
            continue

        categories[cat] += 1
        start = ts
        end = ts + dur

        # Classify
        name_lower = name.lower()

        # ProfilerStep
        if "ProfilerStep" in name:
            step_markers.append((start, end, name))
            continue

        # NCCL kernels (GPU-side)
        if "nccl" in name_lower:
            if "allgather" in name_lower or "all_gather" in name_lower:
                nccl_ag.append((start, end))
            elif "reducescatter" in name_lower or "reduce_scatter" in name_lower:
                nccl_rs.append((start, end))
            else:
                nccl_other.append((start, end))
            name_samples[name] += 1
            continue

        # FSDP CPU-side markers
        if "FSDP" in name:
            fsdp_markers.append((start, end, name))
            continue

        # GPU compute kernels
        if cat in ("kernel", "gpu_memcpy", "cuda_runtime"):
            compute.append((start, end))
            continue

        # Also catch known compute kernel names even without "kernel" cat
        if any(kw in name_lower for kw in [
            "gemm", "matmul", "flash", "attention", "layernorm",
            "elementwise", "softmax", "silu", "rmsnorm", "group_gemm",
            "vectorized", "reduce_kernel", "ampere", "cutlass", "sm80", "sm90",
        ]):
            compute.append((start, end))

    print(f"\nTotal events: {total_events}")
    print(f"\nCategories (top 15):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat:<40} {count:>8}")

    print(f"\n{'='*80}")
    print(f"COLLECTED INTERVALS:")
    print(f"  NCCL All-Gather:      {len(nccl_ag):>6}  total = {sum(e-s for s,e in nccl_ag)/1e6:.3f}s")
    print(f"  NCCL Reduce-Scatter:  {len(nccl_rs):>6}  total = {sum(e-s for s,e in nccl_rs)/1e6:.3f}s")
    print(f"  NCCL Other:           {len(nccl_other):>6}  total = {sum(e-s for s,e in nccl_other)/1e6:.3f}s")
    print(f"  Compute kernels:      {len(compute):>6}  total = {sum(e-s for s,e in compute)/1e6:.3f}s")
    print(f"  FSDP markers:         {len(fsdp_markers):>6}")
    print(f"  Step markers:         {len(step_markers):>6}")

    # NCCL name distribution
    print(f"\nNCCL kernel names (top 10):")
    for name, count in sorted(name_samples.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<70} {count:>5}")

    # Overlap calculation
    def merge_intervals(intervals):
        """合并重叠区间"""
        if not intervals:
            return []
        sorted_iv = sorted(intervals)
        merged = [sorted_iv[0]]
        for s, e in sorted_iv[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def compute_overlap(comm_intervals, comp_intervals):
        """
        计算 comm 区间和 compute 区间的 overlap 时间（us）。
        使用 sweep line 算法。
        """
        if not comm_intervals or not comp_intervals:
            return 0

        # Merge both interval sets
        comm_merged = merge_intervals(comm_intervals)
        comp_merged = merge_intervals(comp_intervals)

        overlap = 0
        j = 0
        for cs, ce in comm_merged:
            # Move j to first comp interval that ends after cs
            while j < len(comp_merged) and comp_merged[j][1] <= cs:
                j += 1
            # Scan comp intervals that overlap with [cs, ce]
            k = j
            while k < len(comp_merged) and comp_merged[k][0] < ce:
                ov_start = max(cs, comp_merged[k][0])
                ov_end = min(ce, comp_merged[k][1])
                if ov_end > ov_start:
                    overlap += (ov_end - ov_start)
                k += 1

        return overlap

    # Merge and compute overlap
    ag_merged = merge_intervals(nccl_ag)
    rs_merged = merge_intervals(nccl_rs)
    compute_merged = merge_intervals(compute)

    ag_total_us = sum(e - s for s, e in ag_merged)
    rs_total_us = sum(e - s for s, e in rs_merged)
    compute_total_us = sum(e - s for s, e in compute_merged)

    ag_overlap_us = compute_overlap(ag_merged, compute_merged)
    rs_overlap_us = compute_overlap(rs_merged, compute_merged)

    print(f"\n{'='*80}")
    print(f"OVERLAP ANALYSIS (after merging overlapping intervals):")
    print(f"  All-Gather (merged):")
    print(f"    Wall time:           {ag_total_us/1e6:.3f}s")
    print(f"    Overlapped compute:  {ag_overlap_us/1e6:.3f}s")
    print(f"    Overlap ratio:       {ag_overlap_us/ag_total_us*100:.1f}%" if ag_total_us > 0 else "    N/A")
    print(f"    EXPOSED (no overlap): {(ag_total_us - ag_overlap_us)/1e6:.3f}s ({(1-ag_overlap_us/ag_total_us)*100:.1f}%)" if ag_total_us > 0 else "")
    print()
    print(f"  Reduce-Scatter (merged):")
    print(f"    Wall time:           {rs_total_us/1e6:.3f}s")
    print(f"    Overlapped compute:  {rs_overlap_us/1e6:.3f}s")
    print(f"    Overlap ratio:       {rs_overlap_us/rs_total_us*100:.1f}%" if rs_total_us > 0 else "    N/A")
    print(f"    EXPOSED (no overlap): {(rs_total_us - rs_overlap_us)/1e6:.3f}s ({(1-rs_overlap_us/rs_total_us)*100:.1f}%)" if rs_total_us > 0 else "")

    # Total timeline
    if step_markers:
        first_step_start = min(s for s, _, _ in step_markers)
        last_step_end = max(e for _, e, _ in step_markers)
        total_wall = (last_step_end - first_step_start) / 1e6
    else:
        all_starts = [s for s, _ in nccl_ag + nccl_rs + compute]
        all_ends = [e for _, e in nccl_ag + nccl_rs + compute]
        if all_starts and all_ends:
            total_wall = (max(all_ends) - min(all_starts)) / 1e6
        else:
            total_wall = 0

    total_exposed = (ag_total_us - ag_overlap_us + rs_total_us - rs_overlap_us) / 1e6

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total wall time:        {total_wall:.2f}s")
    print(f"  Compute (merged):       {compute_total_us/1e6:.3f}s ({compute_total_us/1e6/total_wall*100:.1f}%)" if total_wall > 0 else "")
    print(f"  Comm exposed (AG+RS):   {total_exposed:.3f}s ({total_exposed/total_wall*100:.1f}%)" if total_wall > 0 else "")
    print(f"  Other (gap/overhead):   {total_wall - compute_total_us/1e6 - total_exposed:.3f}s" if total_wall > 0 else "")

    # Per-layer analysis using FSDP markers
    print(f"\n{'='*80}")
    print(f"FSDP MARKER BREAKDOWN:")
    fsdp_by_name = defaultdict(list)
    for s, e, n in fsdp_markers:
        fsdp_by_name[n].append(e - s)
    for name, durs in sorted(fsdp_by_name.items(), key=lambda x: -sum(x[1]))[:20]:
        total = sum(durs) / 1e6
        avg = (sum(durs) / len(durs)) / 1e3  # ms
        print(f"  {name:<55} cnt={len(durs):>4}  total={total:.3f}s  avg={avg:.2f}ms")


if __name__ == "__main__":
    main()
