# Experimental frame attention projector

`veomni.models.custom.llava_qwen3moe.projector_pooled_query_attention` provides
projectors selectable through `build_image_projector("pooled_query_attention", ...)`
and `build_image_projector("pooling_residual_attention", ...)`.
The existing `dynamic_avgpool` factory option is unchanged.
The full-feature baseline is named `PooledQueryAttentionProjector`. Update old
checkpoint configs from `frame_varlen_attention` to `pooled_query_attention`;
the old module, class name and factory option have been removed. Parameter keys
are unchanged, so complete full-feature checkpoints retain their behavior.

For the custom Qwen3.5 vision encoder, set these fields inside
`encoder_config.image_config` in the checkpoint config:

```json
{
  "image_projector_type": "pooled_query_attention",
  "image_projector_num_attention_heads": 32,
  "image_projector_attention_dropout": 0.0
}
```

The builder defaults to 32 heads: the Qwen3.5 merger concatenates 2x2 patches
of width 1152 into width 4608, giving head dimension 144. The standalone class
default remains 8 heads; pass 32 explicitly when using it with width 4608.

```python
from veomni.models.custom.llava_qwen3moe.projector_pooled_query_attention import (
    PooledQueryAttentionProjector,
)

projector = PooledQueryAttentionProjector(
    encoder_hidden=1024,
    out_hidden=2048,
    downsample_ratio=4,
    num_attention_heads=8,
).cuda().bfloat16()
tokens, seq_len = projector(
    images_feature, images_thw,
    merge_size=merge_sizes,
    downsample_ratios=ratios,
)
```

Inputs follow `DynamicAvgPoolProjector`: features are flattened, spatially merged
vision features in grid/time/row/column order. `images_thw` describes the grids
before spatial merging. Merge sizes and compression ratios can vary per grid.
Each temporal grid step is an independent attention sequence; this need not be
one raw video frame when the encoder performs temporal merging.

Adaptive average pooling produces queries. Each query attends to all dense
patches in its own temporal step, with separate query and KV prefix sums in one
FlashAttention varlen call. Learned normalized time/row/column embeddings and a
log2 compression-rate embedding condition the attention. The residual output
projection starts at zero. Enhanced tokens pass through the original-style
two-layer projector MLP; there is no additional residual FFN. Output token count,
order, and per-temporal-step `seq_len` match average pooling.

## Pooling residual experiment

Set `image_projector_type` to `pooling_residual_attention` to build
`PoolingResidualAttentionProjector` from the same `projector_pooled_query_attention.py` module.
The encoder already forwards this option to the shared image-projector factory;
no changes to the ViT, LLM positions, placeholder lengths or collators are needed.

The source X is the final ViT feature after the encoder's norm-and-concatenate
spatial merger, before the projector MLP. This experiment uses:

```text
B = AdaptiveAvgPool2d(X)
R = X - U(B)
Q = query_proj(LN(B)) + query_position + compression_rate
K = key_proj(LN(X)) + key_position
V = value_proj(R)
Z = B + zero_init_output(Attention(Q, K, V))
tokens = mlp(Z)
```

U broadcasts each coarse feature over its actual adaptive-pooling window.
On integer compression grids this repeats the region mean. On non-divisible
grids a dense position may belong to several windows: it receives the average
of their coarse features. Broadcasting preserves time and channel dimensions;
it is a reference coarse representation, not an inverse of pooling. For scale
1, R is exactly zero. Do not expect the zero-mean-cell property on overlapping
windows. The class also accepts the baseline's positive ratios below 1, but
those enlarge the output grid and are not compression experiments.

Residual values have a bias-free linear projection and no per-token LayerNorm,
so their relative amplitudes are retained. Pooling and broadcasting remain in
the autograd graph. Query/key normalization, frame boundaries, positions, output
MLP and zero-initialized output projection match the full-feature baseline.
At initialization Q/K/V receive zero gradients; the output projection learns
first. With `return_attention_aux=True`, `aux.dense_residual_tokens` exposes
the [N, D] input pooling differences, whereas `aux.residual_tokens` is the
[C, D] attention correction. The full-feature baseline leaves the former unset.

Compare both modes at the same realized token budget. Reuse the same pretrained
MLP and initialize attention parameters consistently for a controlled experiment.
Although their parameter schemas match, a trained full-feature attention branch
is not a trained residual branch: changing the config alone changes Value
semantics. Start the new branch from a baseline MLP with newly initialized
attention, or load a checkpoint trained specifically for residual values.

Requires FlashAttention 2, CUDA fp16/bf16 Q/K/V and head dimension at most 256.
There is no CPU/SDPA fallback. Training with Ulysses sequence parallelism raises
`NotImplementedError`. Empty visual inputs retain parameter autograd edges, but
the caller must still execute the projector and connect its output to the loss
(including a zero-valued dependency for text-only inputs). Packed variable
lengths are supported; full distributed FSDP2 training has not been validated.

Use `return_attention_aux=True` for base/residual tensors and frame boundaries,
or `enhance_tokens()` to inspect features before the final MLP. Copy an existing
projector's `mlp.state_dict()` into `projector.mlp` to preserve its learned mapping;
the new attention parameters require their own initialization/checkpoint state.
VeOmni initializes missing projector parameters through the projector's own
`_init_weights`: Linear weights use std 0.02, biases are zero, LayerNorm weights
are one, and the attention residual output weight is zero. A complete checkpoint
restores trained attention weights without reinitializing them.

To create an independent checkpoint with all image projector weights removed:

```bash
python scripts/prepare_frame_attention_checkpoint.py SOURCE DESTINATION
```

This script selects the `pooled_query_attention` full-feature baseline.
It copies unaffected shards byte-for-byte, rewrites only shards containing
`image_encoder.mm_projector.*`, and updates the index and the three config fields
above. Audio projector, vision backbone, LLM weights and other config values are
preserved. Per-shard readback hashes are recorded in
`frame_attention_conversion.json`. Existing output directories are refused.
Use the new directory for both `model_path` and `config_path`; do not resume a
distributed checkpoint with the old projector structure.

The prepared experiment checkpoint is based on
`ckpt/0904_llavaomni_30A3B_dynamic_downsample_st0_gametext_4e5/checkpoints/ckpt_20000/hf_ckpt`.
Its sibling output directory is `hf_ckpt_frame_attention_no_image_projector`.
This removes all four old image-projector MLP tensors as well, so the whole new
projector starts fresh. All audio-projector tensors remain in the checkpoint.

Tests: `tests/models/test_frame_varlen_attention_projector.py`. GPU tests compare
FlashAttention with explicit float32 softmax attention, check frame isolation,
dynamic lengths, initialization equivalence, and backward gradients.
