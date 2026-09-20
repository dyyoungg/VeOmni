# Ulysses omni dataloader

The custom `llava_trainer.VLMTrainer` selects `make_ulysses_train_dataloader`
when Ulysses is enabled. This pipeline is separate from the native text/VLM
`build_dataloader` implementation.

## Sample identity and synchronization

Each worker emits `(raw_sample_index, chunk_index, terminal, sample)` records.
Indices refer to the shuffled, per-DP-rank epoch stream. Every raw sample ends
with a terminal record, including an empty generator or a processor failure.
Chunks emitted before a failure keep their original indices.

The existing heap-based reorderer retains its buffering and overflow behavior
and passes the record identity through to synchronization. Ulysses
peers exchange record identities and content fingerprints through their CPU
Gloo group. Only matching identities with matching fingerprints enter packing.
Peers ahead of the smallest identity retain their current record while others
advance. A missing chunk is dropped; it is never compared with the next video's
chunk. The common prefix of a partially decoded video can still be trained.
EOF peers continue participating until all peers reach EOF. Iterator, packing
and collation errors propagate through this same metadata exchange. A final
exchange covers errors while collating the final pack.

Each exchange carries four integers: `[raw_id, chunk_id, state, fingerprint]`.
`RecordState` encodes `DATA=0`, `TERMINAL=1`, `EOF=2`, `ERROR=3`, `STOP=4`.
Errors take precedence over stop requests. A nonnegative fingerprint denotes
valid data; `-1` denotes no data. Tensor shapes are included in the fingerprint,
so separate token-length and validity fields are unnecessary.

Fingerprints include input tokens, labels, sequence boundaries, visual/audio
payloads and downsample ratios. Computing them adds a CPU pass over each sample;
they are content hashes, not a mathematical proof of tensor equality.
The group is captured during loader construction. Missing Ulysses Gloo groups
and combined Ulysses/context parallelism are rejected.

## Batches and epochs

The trainer owns epoch advancement. `set_epoch(epoch)` followed by iteration
produces only that epoch. Repeating `set_epoch` for a restored epoch preserves
its cursor. The trainer calls `set_epoch` only at epoch entry and closes every
loader before epoch-end callbacks. The next epoch switches the loader's epoch.

`train.per_device_train_batch_size` is the number of packed micro-batches per
optimizer step. Each packed sequence is collated separately; the loader returns
`list[dict]`, and the trainer accumulates their gradients. Incomplete final
accumulation groups are dropped so all DP peers execute the same number of
forward/backward calls. As with the existing training loop, the epoch stops when
any DP peer is exhausted or returns an empty batch. All loaders also use the
same estimated step upper bound (`init_data_size`); packing and video chunking
do not make that bound exact. A read exception is logged locally and synchronized
before the next training step; all ranks close their loaders and finish training
without raising a new trainer exception or starting another epoch. Epoch-end
checkpoint/evaluation callbacks are skipped on this failure path.

The LR schedule uses the global raw-sample count per epoch multiplied by the
number of epochs once. A long video's chunk count does not multiply this total.
The current raw-sample progress is conservative while training inside a video:
it advances with the committed raw cursor rather than counting chunks as videos.

## Checkpoint and resume

The loader implements `state_dict()` and `load_state_dict()`. The trainer calls
`mark_batch_consumed()` after a successful optimizer update. Checkpoints record
the raw sample index of the last trained packed sequence. Fetching or prefetching
a batch does not commit progress. A batch rejected because another DP peer is
exhausted is not marked consumed.

Resume is intentionally coarse: the current raw sample is read again from its
beginning. Some chunks of a long video can be trained again. There is no saved
chunk cursor, packing buffer, or attempt to reproduce the exact packed batch
boundary. This trades a small amount of repeated training for a simple resume
contract. The checkpoint validates source path, shard length, seed,
DP/Ulysses topology, micro-batch count and maximum sequence length.
Epoch-end checkpoints retain the current epoch, step and coarse data cursor.
Restoring one may revisit that epoch's stopping condition before advancing to
the next epoch. The loader is not advanced merely to save a checkpoint.

The sample RNG is derived from the training seed, epoch and sample index;
Ulysses media preprocessing uses a single executor worker to retain augmentation
order across peers. This is independent of the coarse checkpoint contract.

For direct loader users:

```python
loader.set_epoch(epoch)
try:
    for micro_batches in loader:
        train_optimizer_step(micro_batches)
        loader.mark_batch_consumed()
finally:
    loader.close()
```

## Regression tests

`pytest tests/data/test_ulysses_dataloader.py` uses CPU tensors, real DataLoader
workers and two-process Gloo. It covers missing/empty/failed streams, unequal
EOF, processor RNG replay, content fingerprints, partial accumulation tails,
epoch accounting, coarse raw-sample checkpoint recovery, producer shutdown and
rank-local collation errors. Optional media/model imports are replaced when
loading the legacy class definitions. Full accelerator training remains a
separate integration check.

## Real-data consistency smoke test

In an environment that can import the trainer and media processors, run:

```bash
conda activate llava
ULYSSES_TEST_MAX_SAMPLES=64 ULYSSES_TEST_MAX_STEPS=0 \
  torchrun --standalone --nproc_per_node=2 -m veomni.data.ulysess_dataloader \
  exp_data/0913_stage2_mm_projector_balancedgame/30A3B_qwen35encoder_fsdp2_freeze_router_auxloss_dynamic_downsample_frameattention_projector.yaml \
  --train.accelerator.ulysses_size 2 \
  --train.per_device_train_batch_size 2 \
  --train.dataloader_num_workers 2
```

`test_ulysess()` uses the tokenizer and processor assets from the trainer YAML;
no model weights are loaded. It selects equally spaced rows of the offset file
into a temporary subset, preserving the original file mapping and source files.
`ULYSSES_TEST_MAX_SAMPLES` defaults to 128, `ULYSSES_TEST_MAX_STEPS` to 10, and
`ULYSSES_TEST_SP_SIZE` to 2. Set the step limit to zero to exhaust the subset.
The sample limit is global, before DP sharding.

The test fails on identity, chunk count, EOF or content differences **before**
the loader can filter mismatched samples. It also compares SHA-256 fingerprints
of every packed field (tensor dtype, shape and bytes) **before** SP slicing,
for every returned micro-batch. Consumer checks use a separate process group
from the background synchronization protocol. Already detected prefetch errors
are retained even when the step limit stops consumption. Empty runs fail.

The final `ULYSSES_CONSISTENCY_PASS` report includes checked micro-batches,
matched chunks, empty raw samples, and the number of packs containing each
modality. Empty samples can include unavailable media or processing failures
shared by all ranks; a pass checks consistency, not source-data completeness.
Prefetch and a dropped incomplete accumulation group may make the reported
processed chunk/pack counts larger than the number of consumed micro-batches.
A subset only covers the modalities it actually contains; long-video chunking
requires records with `subtitles`. This test does not run forward/backward.
