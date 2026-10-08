"""
细粒度分析：找出 backward 阶段中 compute stream 空闲但 all-gather 在跑的时间段。
区分 forward vs backward，找出 backward 中的 bubble。

python scripts/analyze_backward_gaps.py profile_json/seq24k_overlap_rank0.pt.trace.json.gz
"""
import gzip
import sys
import ijson
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile_json/seq24k_overlap_rank0.pt.trace.json.gz"

    print(f"Streaming {path} ...")

    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")

    # Collect events by stream/category
    # We need to identify: compute stream events, comm stream events, FSDP markers
    compute_kernels = []   # GPU compute kernels (kernel category)
    nccl_kernels = []      # NCCL kernels (kernel category, nccl in name)
    fsdp_markers = []      # FSDP CPU markers for phase identification
    step_markers = []      # ProfilerStep markers
    all_gather_markers = []  # FSDP::all_gather CPU markers
    backward_markers = []   # autograd/backward markers

    total = 0
    parser = ijson.items(f, "traceEvents.item")

    for evt in parser:
        total += 1
        if total % 1000000 == 0:
            print(f"  {total}...", flush=True)

        ph = evt.get("ph", "")
        if ph != "X":
            continue

        name = evt.get("name", "")
        cat = evt.get("cat", "")
        ts = float(evt.get("ts", 0))
        dur = float(evt.get("dur", 0))
        if dur <= 0:
            continue

        start = ts
        end = ts + dur

        # Step markers
        if "ProfilerStep" in name:
            step_markers.append((start, end, name))
            continue

        # FSDP markers (CPU)
        if "FSDP" in name:
            fsdp_markers.append((start, end, name))
            if "all_gather" in name.lower() and "::" in name:
                all_gather_markers.append((start, end, name))
            continue

        # Backward phase markers
        if "backward" in name.lower() or "autograd" in cat.lower():
            backward_markers.append((start, end, name))
            continue

        # NCCL kernels
        if "nccl" in name.lower():
            nccl_kernels.append((start, end, name))
            continue

        # GPU compute kernels
        if cat == "kernel":
            compute_kernels.append((start, end, name))
            continue

        # Also catch compute from other categories
        if cat in ("gpu_memcpy", "gpu_memset"):
            compute_kernels.append((start, end, name))

    f.close()
    print(f"\nTotal events: {total}")
    print(f"Compute kernels: {len(compute_kernels)}")
    print(f"NCCL kernels: {len(nccl_kernels)}")
    print(f"FSDP markers: {len(fsdp_markers)}")
    print(f"Steps: {len(step_markers)}")

    # Separate NCCL into AG vs RS
    nccl_ag = [(s, e) for s, e, n in nccl_kernels if "allgather" in n.lower() or "all_gather" in n.lower()]
    nccl_rs = [(s, e) for s, e, n in nccl_kernels if "reducescatter" in n.lower() or "reduce_scatter" in n.lower()]

    print(f"NCCL AG kernels: {len(nccl_ag)}")
    print(f"NCCL RS kernels: {len(nccl_rs)}")

    # Sort everything by start time
    compute_kernels.sort()
    nccl_ag.sort()
    nccl_rs.sort()

    # Find backward phase boundaries using FSDP markers
    # In FSDP2, backward is triggered by autograd.
    # We can identify backward by looking at "post_backward" or by the pattern:
    # Forward: FSDP::pre_forward → FSDP::all_gather → compute → next layer
    # Backward: reverse order, with FSDP::post_backward_reduce (reduce-scatter)

    # Strategy: Use the FIRST step's timeline to identify forward vs backward boundary
    # Forward ends roughly when the last layer's forward completes
    # Backward starts from there

    if not step_markers:
        print("No step markers found!")
        return

    # Analyze first complete step (skip step 0 which might have warmup)
    step_markers.sort()
    target_step = step_markers[1] if len(step_markers) > 1 else step_markers[0]
    step_start, step_end, step_name = target_step
    print(f"\n{'='*80}")
    print(f"Analyzing {step_name}: [{step_start/1e6:.3f}s, {step_end/1e6:.3f}s] duration={((step_end-step_start)/1e6):.3f}s")

    # Get events within this step
    step_compute = [(s, e) for s, e, _ in compute_kernels if s >= step_start and e <= step_end]
    step_ag = [(s, e) for s, e in nccl_ag if s >= step_start and e <= step_end]
    step_rs = [(s, e) for s, e in nccl_rs if s >= step_start and e <= step_end]
    step_fsdp = [(s, e, n) for s, e, n in fsdp_markers if s >= step_start and e <= step_end]

    print(f"  Compute kernels in step: {len(step_compute)}")
    print(f"  AG kernels in step: {len(step_ag)}")
    print(f"  RS kernels in step: {len(step_rs)}")

    # Identify forward/backward boundary
    # In a typical training step: forward → backward → optimizer
    # The RS (reduce-scatter) only happens in backward (grad aggregation)
    # The AG happens in both forward and backward (param unshard)
    # Forward: AG → compute → AG → compute → ... (no RS)
    # Backward: AG → compute → RS → AG → compute → RS → ...

    # Find the first RS event - backward starts before that
    if step_rs:
        first_rs = step_rs[0][0]
        # Backward boundary: find the last AG before first RS that's "close" to it
        # Actually better: look for FSDP post_backward markers
        post_backward_markers = [(s, e, n) for s, e, n in step_fsdp if "post_backward" in n.lower()]
        if post_backward_markers:
            backward_start = post_backward_markers[0][0]
        else:
            # Estimate: backward starts a bit before first RS
            backward_start = first_rs - 1000  # 1ms before first RS

        print(f"\n  Forward phase: [{step_start/1e6:.3f}s, {backward_start/1e6:.3f}s] = {(backward_start-step_start)/1e6:.3f}s")
        print(f"  Backward phase: [{backward_start/1e6:.3f}s, {step_end/1e6:.3f}s] = {(step_end-backward_start)/1e6:.3f}s")
    else:
        backward_start = (step_start + step_end) / 2  # fallback

    # Backward analysis: find gaps in compute stream during AG
    bwd_compute = [(s, e) for s, e in step_compute if s >= backward_start]
    bwd_ag = [(s, e) for s, e in step_ag if s >= backward_start]
    bwd_rs = [(s, e) for s, e in step_rs if s >= backward_start]

    print(f"\n  Backward compute kernels: {len(bwd_compute)}")
    print(f"  Backward AG kernels: {len(bwd_ag)}")
    print(f"  Backward RS kernels: {len(bwd_rs)}")

    # Find gaps in compute stream during backward
    # A "gap" is a period where no compute kernel is running
    bwd_compute_sorted = sorted(bwd_compute)

    gaps = []
    for i in range(1, len(bwd_compute_sorted)):
        prev_end = bwd_compute_sorted[i-1][1]
        curr_start = bwd_compute_sorted[i][0]
        if curr_start > prev_end + 1:  # gap > 1us
            gaps.append((prev_end, curr_start))

    # Filter: gaps that coincide with AG activity
    gaps_during_ag = []
    ag_idx = 0
    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start
        if gap_dur < 100:  # skip tiny gaps < 0.1ms
            continue
        # Check if any AG is active during this gap
        overlapping_ag = False
        for ag_s, ag_e in bwd_ag:
            if ag_s < gap_end and ag_e > gap_start:
                overlapping_ag = True
                break
        if overlapping_ag:
            gaps_during_ag.append((gap_start, gap_end, gap_dur))

    print(f"\n{'='*80}")
    print(f"BACKWARD COMPUTE GAPS DURING ALL-GATHER:")
    print(f"  Total gaps > 0.1ms: {len([g for g in gaps if g[1]-g[0] > 100])}")
    print(f"  Gaps overlapping with AG: {len(gaps_during_ag)}")

    if gaps_during_ag:
        total_gap_time = sum(d for _, _, d in gaps_during_ag)
        print(f"  Total gap time during AG: {total_gap_time/1e3:.2f} ms")

        # Distribution
        gap_durs = sorted([d for _, _, d in gaps_during_ag], reverse=True)
        print(f"\n  Gap duration distribution (top 30):")
        print(f"  {'Duration (ms)':<15} {'Count':<8}")

        # Bucket by size
        buckets = [(0.1, 0.5), (0.5, 1), (1, 5), (5, 10), (10, 50), (50, 100), (100, 500), (500, 9999)]
        for lo, hi in buckets:
            count = len([d for d in gap_durs if lo <= d/1000 < hi])
            if count > 0:
                total_in_bucket = sum(d for d in gap_durs if lo <= d/1000 < hi) / 1000
                print(f"  {lo:.1f}-{hi:.0f} ms:     {count:>5} gaps,  total = {total_in_bucket:.1f} ms")

        # Show largest gaps
        print(f"\n  Largest 20 gaps (ms):")
        for i, (gs, ge, gd) in enumerate(sorted(gaps_during_ag, key=lambda x: -x[2])[:20]):
            # What AG was running during this gap?
            concurrent_ag = [(s, e) for s, e in bwd_ag if s < ge and e > gs]
            ag_info = f"{len(concurrent_ag)} AG(s)" if concurrent_ag else ""
            print(f"    {i+1:>2}. {gd/1000:.2f} ms  @ t={gs/1e6:.4f}s  {ag_info}")

    # Also analyze: what fraction of backward AG time has no concurrent compute
    print(f"\n{'='*80}")
    print(f"BACKWARD AG: FRACTION WITHOUT COMPUTE OVERLAP:")

    # For each backward AG kernel, check if compute is running concurrently
    ag_exposed_time = 0
    ag_total_time = 0
    ag_exposed_events = []

    for ag_s, ag_e in bwd_ag:
        ag_dur = ag_e - ag_s
        ag_total_time += ag_dur

        # Find compute overlap with this specific AG
        overlap = 0
        for cs, ce in bwd_compute_sorted:
            if cs >= ag_e:
                break
            if ce <= ag_s:
                continue
            ov_s = max(ag_s, cs)
            ov_e = min(ag_e, ce)
            if ov_e > ov_s:
                overlap += (ov_e - ov_s)

        exposed = ag_dur - overlap
        if exposed > 100:  # > 0.1ms exposed
            ag_exposed_time += exposed
            ag_exposed_events.append((ag_s, ag_e, exposed, ag_dur))

    print(f"  Backward AG total time: {ag_total_time/1e3:.2f} ms")
    print(f"  Backward AG exposed (no compute): {ag_exposed_time/1e3:.2f} ms ({ag_exposed_time/ag_total_time*100:.1f}%)" if ag_total_time > 0 else "")
    print(f"  Backward AG events with >0.1ms exposure: {len(ag_exposed_events)}/{len(bwd_ag)}")

    if ag_exposed_events:
        print(f"\n  Top 20 most-exposed backward AG events:")
        for i, (s, e, exp, dur) in enumerate(sorted(ag_exposed_events, key=lambda x: -x[2])[:20]):
            print(f"    {i+1:>2}. exposed={exp/1000:.2f}ms / total={dur/1000:.2f}ms ({exp/dur*100:.0f}%)  @ t={s/1e6:.4f}s")

    # Forward AG analysis for comparison
    fwd_ag = [(s, e) for s, e in step_ag if s < backward_start]
    fwd_compute = [(s, e) for s, e in step_compute if s < backward_start]
    fwd_compute_sorted = sorted(fwd_compute)

    fwd_ag_exposed = 0
    fwd_ag_total = 0
    for ag_s, ag_e in fwd_ag:
        ag_dur = ag_e - ag_s
        fwd_ag_total += ag_dur
        overlap = 0
        for cs, ce in fwd_compute_sorted:
            if cs >= ag_e:
                break
            if ce <= ag_s:
                continue
            ov_s = max(ag_s, cs)
            ov_e = min(ag_e, ce)
            if ov_e > ov_s:
                overlap += (ov_e - ov_s)
        fwd_ag_exposed += (ag_dur - overlap)

    print(f"\n{'='*80}")
    print(f"COMPARISON: Forward vs Backward AG exposure:")
    print(f"  Forward AG:  total={fwd_ag_total/1e3:.1f}ms  exposed={fwd_ag_exposed/1e3:.1f}ms ({fwd_ag_exposed/fwd_ag_total*100:.1f}%)" if fwd_ag_total > 0 else "")
    print(f"  Backward AG: total={ag_total_time/1e3:.1f}ms  exposed={ag_exposed_time/1e3:.1f}ms ({ag_exposed_time/ag_total_time*100:.1f}%)" if ag_total_time > 0 else "")


if __name__ == "__main__":
    main()
