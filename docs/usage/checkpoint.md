# Checkpoint layout

This branch retains the checkpoint format used before the upstream merge.
Checkpoint callbacks own model, optimizer and trainer-state saving together;
the retained model runtime is unwrapped at this boundary.

```
output_dir/
├── checkpoints/global_step_{N}/
│   ├── .metadata
│   ├── __*.distcp                  # model and optimizer in the same DCP
│   ├── extra_state/
│   │   └── extra_state_rank_{R}.pt # cursor, scheduler, RNG and meters
│   └── hf_ckpt/                    # full-model HF export
├── global_step_{N}/                # LoRA adapter export
└── model_assets/
```

`extra_state` contains `global_step`, `start_epoch`, `start_step`,
`train_dataloader`, `lr_scheduler`, `environ_meter`, `channel_loss_callback`
and `torch_rng_state`. Each rank writes and restores its own file. Full-model
DCP resumes restore model and optimizer; LoRA DCP saves trainable parameters
only and still needs the pretrained base weights.

`train.checkpoint.load_path` points directly to the step directory. `auto`
discovers a `global_step_*` directory with a root `.metadata` marker.
`scripts/merge_dcp_to_hf.py --load-dir` takes that same DCP directory.

Saving retains `save_async`, `dcp_save_to_lowest_rank`, the DCP/HF save
cadences, and `save_total_limit` cleanup. The local trainer's epoch-end DCP
save remains unconditional, as before the merge. The split `model/ckpt`,
`model/optimizer`, `loader` and manifest layout is not enabled; neither are
local staging or `save_timeout_seconds`.

Existing checkpoints from before this merge remain the resume format. Do not
point this branch at a checkpoint produced by the deferred split-layout writer.
