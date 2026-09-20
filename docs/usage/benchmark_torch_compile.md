# Decoder compile benchmark

Run from the repository root in a CUDA training environment with FlashAttention
installed, selecting an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_torch_compile \
  --seq-len 2048 --dtype bfloat16 --freeze-weights --iterations 50 --rounds 5 \
  --output /tmp/compile_benchmark.json
```

This measures one randomly initialized Qwen3-VL text decoder, batch size 1, FP16 by default,
using the same FlashAttention implementation, weights, inputs and upstream
gradients for eager and Inductor. Shape options are `--hidden-size`,
`--intermediate-size`, `--heads`, `--kv-heads` and `--seq-len`.
`--dtype bfloat16` measures BF16 with the same correctness thresholds. The script
does not bind optimized RMSNorm/MLP/RoPE OpSlots: those use their default model
implementations. Results may differ from training with fused ops enabled.
Use `--checkpoint` for non-reentrant activation checkpointing.
Use `--freeze-weights` for input-gradient-only backward, as in a frozen backbone
that propagates gradients to a trainable upstream projector.

The benchmark first checks outputs, input gradients and trainable parameter
gradients. A mismatch aborts measurement. First forward+backward latency includes
initialization, compilation or loading existing compiler caches; it is not a
measurement of pure cold compilation time. Caches are not deleted.

After warmup, each round measures a batch of iterations with CUDA synchronization
at the boundaries. Eager and compiled alternate measurement order. The result
reports median milliseconds per iteration, all round timings, and speedup
(`eager_ms / compiled_ms`; below 1 means slower). Forward runs with autograd
enabled; forward+backward includes gradient clearing but no optimizer or loss
computation. Inputs, positional embeddings and attention metadata are prebuilt.
Changing shapes between runs requires new measurements and may trigger compilation.

Memory is the peak additional PyTorch-allocated memory above each measurement's
baseline. Both models remain resident. It excludes cached/reserved memory and
is not the total VRAM required for training.

These are single-decoder results, not end-to-end LLaVA/Qwen3-MoE measurements.
They exclude other layers, vision/projector computation, data loading and
FSDP/Ulysses communication. They do not establish compile support for a custom
multimodal model. Compare actual trainer step time separately before enabling
compilation in production training.

## Example measurement

Measured on an idle NVIDIA A100-SXM4-80GB with PyTorch 2.8.0+cu126,
sequence length 2048, hidden size 1024, intermediate size 2816, 8 query heads,
2 KV heads, frozen weights, no checkpointing, 10 warmup iterations and
5 alternating rounds of 50 iterations. Output and input-gradient comparisons
passed with `rtol=atol=0.02`.

| Precision | Operation | Eager ms | Compiled ms | Speedup |
|-----------|-----------|----------|-------------|---------|
| FP16 | Forward | 1.483 | 0.739 | 2.01x |
| FP16 | Forward + backward | 3.754 | 1.490 | 2.52x |
| BF16 | Forward | 1.435 | 0.689 | 2.08x |
| BF16 | Forward + backward | 3.643 | 1.359 | 2.68x |

The BF16 first compiled forward+backward took 5.219 seconds with the existing
compiler cache state. Additional allocated memory peaked at 107.53 MiB eager
versus 70.16 MiB compiled during forward+backward.

Full-parameter-gradient runs at the same shape failed the elementwise gradient
check in both precisions, so no full-parameter-training speedup is reported.
For example, FP16 `mlp.down_proj.weight` had 26 of 2,883,584 entries outside
tolerance; the largest absolute difference among mismatches was 0.02791.
That numerical discrepancy needs separate investigation before drawing a
full-parameter-training conclusion. To exercise that check, omit
`--freeze-weights`; the benchmark stops if correctness fails.

## Numerical differences

Use `--check-only --output /tmp/compile_accuracy.json` to collect per-tensor
errors without timing rounds. The JSON is also saved on a failed correctness
check, and the command exits nonzero. Error statistics are computed in FP64 on
CPU after collecting the eager and compiled outputs and gradients. Relative L2
is `norm(compiled - eager) / norm(eager)`; it is not task accuracy or loss change.
An undefined relative error (zero reference norm with nonzero error) is `null`.

For the same A100 setup, seed and shapes described above:

| Tensor | FP16 relative L2 | BF16 relative L2 | FP16 max absolute | BF16 max absolute |
|--------|------------------|------------------|-------------------|-------------------|
| Output, frozen weights | 0.0367% | 0.2935% | 0.00390625 | 0.03125 |
| Input gradient, frozen weights | 0.0269% | 0.2110% | 0.00390625 | 0.03125 |
| All parameter gradients concatenated, trainable weights | 0.0792% | 0.6394% | 0.03515625 | 0.28125 |

Output and input-gradient errors in the trainable run are essentially the same
as in the frozen run. No output or input-gradient elements exceeded
`abs(compiled - eager) <= 0.02 + 0.02 * abs(eager)`.
Of 11,274,496 parameter-gradient elements, FP16 had 26 mismatches (0.000231%),
all in `mlp.down_proj.weight`; BF16 had 760,890 mismatches (6.7488%).
The largest absolute error over all elements is different from the largest
error among mismatches: large reference values permit larger absolute errors.

These compare two low-precision execution paths for one random input/weight
sample, not either path against an FP32 reference. They do not establish which
path is closer to exact arithmetic or whether long-run convergence changes.
