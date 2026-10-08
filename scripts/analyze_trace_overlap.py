"""
分析 Perfetto trace 中 FSDP2 通信 vs 计算的 overlap 情况。
流式解析 json.gz，不会 OOM。

用法: python scripts/analyze_trace_overlap.py profile_json/seq24k_rank0.pt.trace.json.gz
"""
import gzip
import json
import sys
from collections import defaultdict


def load_trace_events(path):
    """流式加载 trace events"""
    print(f"Loading {path} ...")
    if path.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")

    data = json.load(f)
    f.close()

    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    else:
        events = data
    print(f"Loaded {len(events)} events")
    return events


def classify_event(evt):
    """将 event 分类为 compute / allgather / reducescatter / other_comm"""
    name = evt.get("name", "")
    cat = evt.get("cat", "")

    # NCCL communication
    if "nccl" in name.lower() or "nccl" in cat.lower():
        if "allgather" in name.lower() or "all_gather" in name.lower():
            return "allgather"
        elif "reducescatter" in name.lower() or "reduce_scatter" in name.lower():
            return "reducescatter"
        elif "allreduce" in name.lower() or "all_reduce" in name.lower():
            return "allreduce"
        elif "alltoall" in name.lower() or "all_to_all" in name.lower():
            return "alltoall"
        else:
            return "other_comm"

    # FSDP operations (CPU-side markers)
    if "FSDP" in name:
        if "all_gather" in name:
            return "fsdp_allgather_marker"
        elif "reduce_scatter" in name or "post_backward" in name:
            return "fsdp_reducescatter_marker"
        elif "all_gather_copy_out" in name:
            return "fsdp_copy_out"
        return "fsdp_other"

    return None


def analyze_stream_overlap(events):
    """
    分析方法：
    1. 找出 GPU kernel 级别的 NCCL 通信 events (在 CUDA stream 上)
    2. 找出 GPU 上的 compute kernels
    3. 计算通信时间窗口内有多少 compute 在并行执行
    """
    # 只看 GPU kernel events (ph='X' 表示 complete event)
    gpu_events = []
    for evt in events:
        if evt.get("ph") not in ("X",):
            continue
        # GPU events have tid that corresponds to stream
        cat = evt.get("cat", "")
        if "kernel" in cat.lower() or "gpu" in cat.lower() or "cuda" in cat.lower():
            gpu_events.append(evt)
        # Also capture events with dur > 0 on device
        elif evt.get("dur", 0) > 0 and "args" in evt:
            args = evt.get("args", {})
            if "stream" in str(args) or "device" in str(args):
                gpu_events.append(evt)

    print(f"\nGPU kernel-level events: {len(gpu_events)}")

    # Categorize by stream/tid
    stream_events = defaultdict(list)
    for evt in gpu_events:
        tid = evt.get("tid", 0)
        stream_events[tid].append(evt)

    print(f"Streams found: {list(stream_events.keys())[:20]}")

    # Print top streams by event count
    for tid, evts in sorted(stream_events.items(), key=lambda x: -len(x[1]))[:10]:
        sample_names = set(e.get("name", "")[:50] for e in evts[:5])
        total_dur = sum(e.get("dur", 0) for e in evts) / 1e6  # us -> s
        print(f"  Stream {tid}: {len(evts)} events, total {total_dur:.2f}s, samples: {sample_names}")


def analyze_by_name_pattern(events):
    """
    更直接的方法：按 event name 模式分类，不依赖 stream ID。
    统计 NCCL kernels 的时间窗口和 compute kernels 的时间窗口的 overlap。
    """
    nccl_intervals = []  # (start, end, name)
    compute_intervals = []  # (start, end, name)
    fsdp_markers = []  # CPU-side FSDP markers

    for evt in events:
        ph = evt.get("ph", "")
        if ph != "X":
            continue

        name = evt.get("name", "")
        ts = evt.get("ts", 0)  # microseconds
        dur = evt.get("dur", 0)

        if dur <= 0:
            continue

        start = ts
        end = ts + dur

        # NCCL kernel on GPU
        if "nccl" in name.lower():
            nccl_intervals.append((start, end, name))
        # FSDP markers (CPU side)
        elif "FSDP" in name:
            fsdp_markers.append((start, end, name))
        # GPU compute kernels - common patterns
        elif any(kw in name.lower() for kw in [
            "gemm", "matmul", "flash", "attention", "layernorm", "layer_norm",
            "elementwise", "softmax", "gelu", "silu", "rmsnorm", "rms_norm",
            "embedding", "rotary", "rope", "group_gemm", "moe",
            "vectorized", "reduce_kernel", "cast", "copy",
        ]):
            compute_intervals.append((start, end, name))

    print(f"\n{'='*80}")
    print(f"EVENT COUNTS:")
    print(f"  NCCL kernel events: {len(nccl_intervals)}")
    print(f"  Compute kernel events: {len(compute_intervals)}")
    print(f"  FSDP CPU markers: {len(fsdp_markers)}")

    if not nccl_intervals:
        print("\nNo NCCL events found! Trying broader search...")
        # Try broader comm pattern
        for evt in events:
            if evt.get("ph") != "X":
                continue
            name = evt.get("name", "")
            cat = evt.get("cat", "")
            if any(kw in name.lower() or kw in cat.lower() for kw in ["allgather", "reduce", "broadcast", "scatter"]):
                ts = evt.get("ts", 0)
                dur = evt.get("dur", 0)
                if dur > 0:
                    nccl_intervals.append((ts, ts + dur, name))
        print(f"  After broader search: {len(nccl_intervals)} comm events")

    # Sub-categorize NCCL
    ag_intervals = [(s, e, n) for s, e, n in nccl_intervals if "allgather" in n.lower() or "all_gather" in n.lower()]
    rs_intervals = [(s, e, n) for s, e, n in nccl_intervals if "reducescatter" in n.lower() or "reduce_scatter" in n.lower()]
    other_nccl = [(s, e, n) for s, e, n in nccl_intervals if (s,e,n) not in ag_intervals and (s,e,n) not in rs_intervals]

    print(f"\n  All-Gather NCCL kernels: {len(ag_intervals)}")
    print(f"  Reduce-Scatter NCCL kernels: {len(rs_intervals)}")
    print(f"  Other NCCL kernels: {len(other_nccl)}")

    # Total time
    ag_total = sum(e - s for s, e, _ in ag_intervals) / 1e6
    rs_total = sum(e - s for s, e, _ in rs_intervals) / 1e6
    compute_total = sum(e - s for s, e, _ in compute_intervals) / 1e6

    print(f"\n{'='*80}")
    print(f"TOTAL DURATION (sum of kernel durations, may overlap):")
    print(f"  All-Gather:      {ag_total:.3f}s")
    print(f"  Reduce-Scatter:  {rs_total:.3f}s")
    print(f"  Compute:         {compute_total:.3f}s")

    # Calculate overlap: how much compute time overlaps with communication
    def calc_overlap(comm_intervals, comp_intervals):
        """计算 comm 时间窗口内有多少 compute 同时在运行"""
        if not comm_intervals or not comp_intervals:
            return 0.0, 0.0

        # Sort both by start time
        comm_sorted = sorted(comm_intervals, key=lambda x: x[0])
        comp_sorted = sorted(comp_intervals, key=lambda x: x[0])

        # For each comm interval, find overlapping compute time
        total_overlap = 0.0
        total_comm = 0.0

        comp_idx = 0
        for cs, ce, _ in comm_sorted:
            total_comm += (ce - cs)
            # Find compute intervals that overlap with [cs, ce]
            # Reset comp_idx if needed (comm is sorted so we can use a sweep)
            while comp_idx > 0 and comp_sorted[comp_idx - 1][1] > cs:
                comp_idx -= 1

            local_overlap = 0.0
            j = comp_idx
            while j < len(comp_sorted) and comp_sorted[j][0] < ce:
                overlap_start = max(cs, comp_sorted[j][0])
                overlap_end = min(ce, comp_sorted[j][1])
                if overlap_end > overlap_start:
                    local_overlap += (overlap_end - overlap_start)
                j += 1

            total_overlap += local_overlap

        return total_overlap / 1e6, total_comm / 1e6  # convert to seconds

    ag_overlap, ag_comm = calc_overlap(ag_intervals, compute_intervals)
    rs_overlap, rs_comm = calc_overlap(rs_intervals, compute_intervals)

    print(f"\n{'='*80}")
    print(f"COMMUNICATION-COMPUTE OVERLAP ANALYSIS:")
    print(f"  All-Gather:")
    print(f"    Total comm time:     {ag_comm:.3f}s")
    print(f"    Overlapped compute:  {ag_overlap:.3f}s")
    print(f"    Overlap ratio:       {ag_overlap/ag_comm*100:.1f}%" if ag_comm > 0 else "    N/A")
    print(f"  Reduce-Scatter:")
    print(f"    Total comm time:     {rs_comm:.3f}s")
    print(f"    Overlapped compute:  {rs_overlap:.3f}s")
    print(f"    Overlap ratio:       {rs_overlap/rs_comm*100:.1f}%" if rs_comm > 0 else "    N/A")

    # FSDP marker analysis
    if fsdp_markers:
        print(f"\n{'='*80}")
        print(f"FSDP CPU MARKERS (top by duration):")
        fsdp_by_name = defaultdict(list)
        for s, e, n in fsdp_markers:
            fsdp_by_name[n].append(e - s)
        for name, durs in sorted(fsdp_by_name.items(), key=lambda x: -sum(x[1]))[:15]:
            total = sum(durs) / 1e6
            avg = (sum(durs) / len(durs)) / 1e3  # ms
            print(f"  {name:<55} count={len(durs):>5}  total={total:.3f}s  avg={avg:.2f}ms")

    # Sample NCCL event names
    nccl_names = defaultdict(int)
    for _, _, n in nccl_intervals:
        nccl_names[n] += 1
    if nccl_names:
        print(f"\n{'='*80}")
        print(f"NCCL KERNEL NAME DISTRIBUTION (top 15):")
        for name, count in sorted(nccl_names.items(), key=lambda x: -x[1])[:15]:
            print(f"  {name:<70} {count:>5}")

    # Per-step analysis
    print(f"\n{'='*80}")
    print(f"PER-STEP BREAKDOWN:")
    step_markers = [(s, e, n) for s, e, n in fsdp_markers if "ProfilerStep" in n]
    if not step_markers:
        # Try to find step boundaries from other markers
        step_markers = [(evt.get("ts",0), evt.get("ts",0)+evt.get("dur",0), evt.get("name",""))
                       for evt in events if evt.get("ph") == "X" and "ProfilerStep" in evt.get("name","")]

    if step_markers:
        for i, (ss, se, sn) in enumerate(step_markers[:5]):
            step_ag = [(s,e) for s,e,_ in ag_intervals if s >= ss and e <= se]
            step_rs = [(s,e) for s,e,_ in rs_intervals if s >= ss and e <= se]
            step_comp = [(s,e) for s,e,_ in compute_intervals if s >= ss and e <= se]

            step_dur = (se - ss) / 1e6
            ag_dur = sum(e-s for s,e in step_ag) / 1e6
            rs_dur = sum(e-s for s,e in step_rs) / 1e6
            comp_dur = sum(e-s for s,e in step_comp) / 1e6

            print(f"  Step {i}: total={step_dur:.2f}s  AG={ag_dur:.3f}s  RS={rs_dur:.3f}s  Compute={comp_dur:.3f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace_overlap.py <trace.json.gz>")
        sys.exit(1)

    path = sys.argv[1]
    events = load_trace_events(path)

    # Quick overview: all unique categories
    cats = defaultdict(int)
    for evt in events:
        cat = evt.get("cat", "unknown")
        cats[cat] += 1
    print(f"\nEvent categories:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cat:<30} {count:>8}")

    # Main analysis
    analyze_by_name_pattern(events)


if __name__ == "__main__":
    main()
