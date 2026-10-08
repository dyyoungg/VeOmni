# Spectral entropy monitoring

Enable the callback in the training configuration. For a model with 32 blocks,
this selects the first, middle and final blocks (indices start at zero):

```yaml
train:
  spectral_entropy_monitor_interval: 1000
  spectral_entropy_monitor_layers: [0, 15, 31]
  spectral_entropy_monitor_max_slices: 4
  spectral_entropy_monitor_log_per_param: true
```

The callback matches the first `layers.N`, `layer.N`, `blocks.N`, `block.N`, or
`h.N` component in each parameter name. The filter applies across model towers:
language and vision blocks with the same index both qualify. Image projector
parameters are included even though they are standalone and have no layer index;
other parameters without a layer index (for example an embedding) are excluded.
Only trainable 2D/3D parameters are monitored; a frozen router is excluded too.
Audio encoder parameters, including parameters named under `audio_encoder`,
`audio_tower`, or `audio_projector`, are excluded even when their layer index
matches the requested list. This also excludes projectors inside the audio tower.
The startup log lists the actual selected parameter names and warns about
requested indices with no matching trainable parameters.

A nonempty list selects all eligible parameters in those blocks. An empty list
selects no parameters. Negative, fractional and string indices are rejected.
Duplicate indices are ignored. Set the interval to zero to disable monitoring.

Metrics are grouped under `spectral_entropy/overall`,
`spectral_entropy/image_encoder`, and `spectral_entropy/llm`. Each scope has
`mean`, `min`, and `max` metrics plus type means at
`{scope}/type/{type}`. Scope is `image_encoder` or `llm`; names containing
`image_encoder`, `vision_tower`, `vision_encoder`,
`vision_model`, `visual`, `image_projector`, or `mm_projector` belong to the
image encoder scope. Types are `projector`, `attention_qo`, `attention_kv`,
`router`, `expert`, and `dense`. A mean weights each selected parameter equally. Layer
metrics are intentionally omitted because the selected layers are usually
similar. If per-parameter logging is enabled, metrics are placed under the
corresponding scope's `param` group.

Compatible 2D FSDP2 row-sharded parameters are grouped by shard mesh and dtype.
Each batch holds at most the shard group's size in parameters. Rank j gathers
parameter j's rows and computes its SVD; a second all-to-all distributes the
scalar entropy. A partial batch uses one-element dummy tensor slots because some
multi-node NCCL versions can hang on zero-element split tensors; idle owners
participate in both collectives but perform no SVD and contribute no metric.
Thus on 8 ranks, 19 compatible parameters use batches of 8, 8, and 3 (five
dummy slots).
All ranks must select the same parameter names in the same order.

This owner path applies to 2D row-sharded matrices. Other layouts use the existing
fallback. In particular, with `spectral_entropy_monitor_gather_3d: false` (the
default), 3D expert tensors report a local-shard proxy averaged over the
parameter's shard mesh. A matrix-dimension shard's entropy is not the entropy
of the full expert matrix and depends on the sharding layout. Setting this flag
to true gathers the entire 3D tensor and can require substantial memory; selecting
fewer expert slices limits SVD work but does not reduce that full gather.

To trace a stall, set `VEOMNI_SPECTRAL_ENTROPY_DEBUG=1` in the training
environment. Every rank logs begin/end markers around full-tensor gathers, SVDs,
3D shard reductions, and both 2D owner all-to-all operations. The last marker
without a matching end identifies the operation where that rank stopped.
