# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU tests of the legacy omni pipeline without optional media/model imports.

Compile the actual class definitions so these tests can exercise worker, packing,
checkpoint and collective behavior without loading remote storage clients or model
kernels. Tensor operations, DataLoader workers and Gloo collectives are real.
"""

import ast
import dataclasses
import hashlib
import heapq
import importlib.util
import logging
import os
import queue
import random
import threading
import time
import traceback
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "veomni/data/ulysses_protocol.py"
spec = importlib.util.spec_from_file_location("ulysses_protocol_under_test", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def parallel_state():
    size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return SimpleNamespace(dp_rank=0, dp_size=1, ulysses_size=size, sp_size=size, sp_rank=rank, cp_size=1)


def extract_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


NS = dict(
    torch=torch,
    dist=dist,
    os=os,
    random=random,
    hashlib=hashlib,
    heapq=heapq,
    queue=queue,
    threading=threading,
    time=time,
    traceback=traceback,
    types=types,
    IterableDataset=IterableDataset,
    get_worker_info=get_worker_info,
    get_parallel_state=parallel_state,
    get_ulysses_sequence_parallel_cpu_group=lambda: dist.group.WORLD if dist.is_initialized() else None,
    get_image_video_audio_placeholder=lambda tokenizer: (-200, -201, -202),
    logger=logging.getLogger(__name__),
    synchronize_records=protocol.synchronize_records,
    SynchronizedDataError=protocol.SynchronizedDataError,
    RecordState=protocol.RecordState,
    dataclass=dataclasses.dataclass,
    field=dataclasses.field,
    IGNORE_INDEX=-100,
    F=F,
)
extract_classes(
    ROOT / "veomni/data/ulysess_dataloader.py",
    {
        "UlyssesStreamingDataset",
        "ReorderingDataLoader",
        "MultimodalPacker",
        "PrefetchingPackedLoader",
        "UlysessOmniProcessor",
        "_UlyssesTestAudit",
    },
    NS,
)
extract_classes(ROOT / "veomni/data/data_collator.py", {"UlysessOmniDataSharderCollator"}, NS)
Dataset = NS["UlyssesStreamingDataset"]
Reorder = NS["ReorderingDataLoader"]
Loader = NS["PrefetchingPackedLoader"]
Collator = NS["UlysessOmniDataSharderCollator"]
Audit = NS["_UlyssesTestAudit"]


def test_consistency_audit_rejects_equal_sum_different_content():
    first = torch.tensor([1, 2, 3])
    second = torch.tensor([3, 2, 1])
    assert first.sum() == second.sum()
    assert Audit.fingerprint(first) != Audit.fingerprint(second)
    assert Audit.fingerprint(first) != Audit.fingerprint(first.reshape(1, 3))
    assert Audit.fingerprint(first) != Audit.fingerprint(first.to(torch.int32))


@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_consistency_audit_fails_before_filtering(field):
    metadata = [5, 0, protocol.RecordState.DATA, 123]
    peer = metadata.copy()
    peer[field] += 1
    audit = Audit(lambda value: [value, peer], None)
    with pytest.raises(protocol.SynchronizedDataError, match="Pre-filter Ulysses mismatch"):
        audit.exchange_metadata(metadata)
    assert audit.error is not None


def test_consistency_audit_retains_prefetch_error_after_stop():
    metadata = [5, 0, protocol.RecordState.DATA, 123]
    peer = metadata.copy()
    peer[3] += 1
    audit = Audit(lambda value: [value, peer], None)
    with pytest.raises(protocol.SynchronizedDataError):
        audit.exchange_metadata(metadata)
    metadata[2] = protocol.RecordState.STOP
    audit.exchange_metadata(metadata)
    assert "Pre-filter Ulysses mismatch" in audit.error


def test_consistency_audit_retains_collator_error():
    def fail(features):
        raise protocol.SynchronizedDataError("collator failed")

    audit = Audit(None, fail)
    with pytest.raises(protocol.SynchronizedDataError):
        audit.collate([{"input_ids": torch.tensor([1, 2])}])
    assert "collator failed" in audit.error


def test_consistency_audit_hashes_full_batch_before_collator_mutates_it():
    raw = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])}
    expected = Audit.fingerprint(raw["input_ids"])

    def collate(features):
        features[0]["input_ids"] = features[0]["input_ids"][:1]
        return features[0]

    output = Audit(None, collate).collate([raw])
    assert output["_ulysses_test_fingerprints"]["input_ids"] == expected
    assert output["input_ids"].numel() == 1


def sample(value, length=3):
    return SimpleNamespace(
        input_ids=torch.full((length,), value, dtype=torch.long),
        labels=torch.full((length,), value, dtype=torch.long),
        attention_mask_len=[length],
    )


class Processor:
    def __call__(self, data, sample_idx):
        for chunk in range(data):
            yield sample(10 * sample_idx + chunk + 1)


def make_loader(batch_size=1, workers=0, counts=(3, 2, 1), max_seq_len=6, prefetch=2):
    raw = Dataset.__new__(Dataset)
    raw.data_list = list(counts)
    raw.epoch = 0
    raw.skip_samples_count = 0
    raw.dp_rank = 0
    raw.use_offset = False
    raw.training_args = SimpleNamespace(seed=42)
    raw.data_args = SimpleNamespace(train_path="test-data")
    raw._processor = Processor()
    raw._init_processor = lambda: None
    stream = DataLoader(raw, batch_size=None, num_workers=workers)
    return Loader(
        Reorder(stream),
        SimpleNamespace(model_max_length=max_seq_len),
        SimpleNamespace(),
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        collate_fn=Collator(),
        prefetch_batches=prefetch,
        num_train_epochs=3,
    )


def consume(loader):
    result = []
    for batches in loader:
        result.append([batch["input_ids"].tolist() for batch in batches])
        loader.mark_batch_consumed()
    return result


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_checkpoint_restarts_current_raw_sample(workers, batch_size):
    original = make_loader(batch_size, workers, counts=(12, 12, 12), max_seq_len=3)
    try:
        iterator = iter(original)
        # Advance into raw sample 1, then partway through that video's chunks.
        for _ in range(12 // batch_size + 1):
            next(iterator)
            original.mark_batch_consumed()
        state = original.state_dict()
        assert state["resume_index"] == 1
        assert "cursor" not in state
        next(iterator)
        assert original.state_dict() == state  # Prefetch/delivery is uncommitted.
    finally:
        original.close()
    restored = make_loader(batch_size, workers, counts=(12, 12, 12), max_seq_len=3)
    restored.load_state_dict(state)
    restored.set_epoch(0)
    try:
        first = next(iter(restored))
        # Coarse resume intentionally repeats the first chunk of raw sample 1.
        assert first[0]["input_ids"][0, 0].item() == 11
        assert len(first) == batch_size
        restored.mark_batch_consumed()
    finally:
        restored.close()


def test_single_epoch_and_fresh_epoch_state():
    loader = make_loader(batch_size=2)
    try:
        first_epoch = consume(loader)
        assert len(first_epoch) == 1  # 3 micro-batches: one full step, one dropped tail.
        assert loader.epoch == 0
        assert loader.samples_consumed == 3
    finally:
        loader.close()
    loader.set_epoch(1)
    assert loader.state_dict()["resume_index"] == 0
    assert loader.samples_consumed == 0
    try:
        assert consume(loader)
        assert loader.epoch == 1
    finally:
        loader.close()


def test_restore_rejects_changed_batch_configuration():
    source = make_loader(batch_size=1)
    restored = make_loader(batch_size=2)
    with pytest.raises(ValueError, match="configuration"):
        restored.load_state_dict(source.state_dict())


def test_fingerprint_checks_contents_and_boundaries():
    loader = make_loader()
    a, b = sample(1), sample(2)
    assert loader._sample_fingerprint(a) != loader._sample_fingerprint(b)
    b = sample(1)
    b.attention_mask_len = [1, 2]
    assert loader._sample_fingerprint(a) != loader._sample_fingerprint(b)
    b = sample(1)
    a.pixel_values = torch.ones(2, 2, dtype=torch.bfloat16)
    b.pixel_values = torch.zeros(2, 2, dtype=torch.bfloat16)
    assert loader._sample_fingerprint(a) != loader._sample_fingerprint(b)


def test_empty_and_partially_failed_generators_have_unique_terminal_records():
    loader = make_loader(counts=(0, 3))
    raw = loader.raw_dataset

    class FailingProcessor:
        def __call__(self, data, sample_idx):
            if data:
                yield sample(1)
                raise ValueError("decode failure")

    raw._processor = FailingProcessor()
    records = list(raw)
    for raw_id in range(2):
        group = [record for record in records if record[0] == raw_id]
        assert group[-1][2] is True
        assert [record[1] for record in group] == list(range(len(group)))
    assert sum(record[2] for record in records) == 2


def test_reordering_keeps_original_order_and_passes_identity():
    records = [
        [0, 0, False, "A0"],
        [1, 0, False, "B0"],
        [0, 1, False, "A1"],
        [1, 1, True, None],
        [0, 2, True, None],
    ]
    ordered = list(Reorder(records))
    assert ordered == [
        (0, 0, False, "A0"),
        (0, 1, False, "A1"),
        (0, 2, True, None),
        (1, 0, False, "B0"),
        (1, 1, True, None),
    ]


def test_wrapper_rng_is_independent_of_previous_samples():
    wrapper = NS["UlysessOmniProcessor"].__new__(NS["UlysessOmniProcessor"])
    wrapper.training_args = SimpleNamespace(seed=123)
    wrapper.epoch = 0
    wrapper.init_image_processor = lambda: None

    class RandomProcessor:
        def process(self, data, idx):
            return [self._rng.random() for _ in range(data.get("draws", 1))]

    wrapper.processor = RandomProcessor()
    wrapper.longvideo_processor = RandomProcessor()
    expected = wrapper({}, 12)
    wrapper({"draws": 100}, 4)
    assert wrapper({}, 12) == expected
    wrapper.epoch = 1
    assert wrapper({}, 12) != expected


def _protocol_worker(rank, init_file, output_dir, case):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:

        def exchange(metadata):
            assert len(metadata) == 4
            tensor = torch.tensor(metadata, dtype=torch.long)
            peers = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(peers, tensor)
            return [peer.tolist() for peer in peers]

        records = [(0, 0, False, 10), (0, 1, True, None), (1, 0, False, 20), (1, 1, True, None)]
        if case == "missing" and rank == 0:
            records = [(0, 0, False, 10), (0, 1, False, 99), (0, 2, True, None), *records[2:]]
        elif case == "empty" and rank == 1:
            records = [(0, 0, True, None), *records[2:]]
        elif case == "eof" and rank == 1:
            records = records[:2]
        elif case == "invalid" and rank == 1:
            records[0] = (0, 0, False, None)
        elif case == "content" and rank == 1:
            records[0] = (0, 0, False, 11)
        elif case == "zero":
            records[0] = (0, 0, False, 0)
        elif case == "error_stop":

            def broken():
                raise ValueError("error must take precedence over stop")
                yield

            if rank == 1:
                records = broken()
        elif case == "error" and rank == 1:

            def broken():
                yield (0, 0, False, 10)
                raise ValueError("reader failed")

            records = broken()
        try:
            result = list(
                protocol.synchronize_records(
                    records,
                    exchange,
                    lambda value: value if value is not None else -1,
                    lambda: (case == "stop" and rank == 1) or case == "error_stop",
                )
            )
            result = [record[3] for record in result]
        except RuntimeError:
            if case not in ("error", "error_stop"):
                raise
            result = "error"
        Path(output_dir, f"{rank}.txt").write_text(repr(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing", [10, 20]),
        ("invalid", [20]),
        ("content", [20]),
        ("zero", [0, 20]),
        ("error_stop", "error"),
        ("empty", [20]),
        ("eof", [10]),
        ("error", "error"),
        ("stop", []),
    ],
)
def test_gloo_identity_alignment_and_termination(tmp_path, case, expected):
    mp.spawn(_protocol_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), case), nprocs=2, join=True)
    assert ast.literal_eval((tmp_path / "0.txt").read_text()) == expected
    assert ast.literal_eval((tmp_path / "1.txt").read_text()) == expected


def extract_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def test_longvideo_expansion_uses_selected_ratio():
    namespace = dict(torch=torch, logger=logging.getLogger(__name__), OmniSample=SimpleNamespace)
    build = extract_method(
        ROOT / "veomni/data/llavaomni_processor.py", "LongVideoProcessor", "_build_chunk", namespace
    )
    selected = 4
    grid = torch.tensor([[1, 8, 8]])

    def expand(**kwargs):
        # Projector receives selected=4; placeholders must use the same value,
        # rather than the fixed config's 16.
        ratio = kwargs.get("downsample_ratio", 16)
        tokens = torch.full((16 // ratio,), 42)
        return tokens, tokens.clone(), {"video": tokens.numel()}

    processor = SimpleNamespace(
        video_merge_size=2,
        _expand_multimodal_tokens=expand,
        process_image_videos=lambda **kwargs: {"pixel_values_videos": torch.ones(64, 8), "video_grid_thw": grid},
    )
    frame = torch.zeros(1, 2, 2, 3, dtype=torch.uint8).numpy()
    vr = SimpleNamespace(get_batch=lambda _: SimpleNamespace(asnumpy=lambda: frame))
    result = build(
        processor,
        [torch.tensor([-201])],
        [torch.tensor([-100])],
        ["video"],
        [((2, 2), [0])],
        vr,
        "unused",
        {"backend": "decord"},
        selected,
    )
    assert result.input_ids.numel() == 4
    assert result.video_downsample_ratios.tolist() == [selected]


def test_trainer_sample_schedule_counts_epochs_once():
    namespace = dict(
        torch=torch,
        dist=dist,
        os=os,
        PrefetchingPackedLoader=Loader,
        get_parallel_state=parallel_state,
        get_data_parallel_world_size=lambda: 1,
        build_lr_scheduler=lambda *args, **kwargs: kwargs,
    )
    build = extract_method(ROOT / "veomni/trainer/llava_trainer.py", "VLMTrainer", "_build_lr_scheduler", namespace)
    optimizer_args = SimpleNamespace(
        lr=1,
        lr_min=0,
        lr_decay_style="linear",
        lr_decay_ratio=1,
        lr_warmup_ratio=0.1,
        lr_start=0,
    )
    trainer = SimpleNamespace(
        train_dataloader=make_loader(counts=(1,) * 100),
        optimizer=None,
        device=torch.device("cpu"),
        args=SimpleNamespace(
            train=SimpleNamespace(remote_dataloader=False, num_train_epochs=3, optimizer=optimizer_args)
        ),
    )
    build(trainer)
    assert trainer.init_data_size == 100
    assert trainer.train_steps == 300
    assert trainer.lr_scheduler["train_steps"] == 300


def test_consumer_cursor_counts_raw_videos_not_chunks():
    loader = make_loader(batch_size=1, counts=(6,), max_seq_len=3)
    try:
        iterator = iter(loader)
        for _ in range(3):
            next(iterator)
            loader.mark_batch_consumed()
        assert loader.samples_consumed == 0  # Still inside the first raw video.
        assert loader.state_dict()["resume_index"] == 0
    finally:
        loader.close()


def test_incomplete_microbatch_group_is_dropped():
    loader = make_loader(batch_size=2, counts=(3,), max_seq_len=3)
    try:
        outputs = consume(loader)
    finally:
        loader.close()
    assert len(outputs) == 1
    assert len(outputs[0]) == 2


def _loader_worker(rank, init_file, output_dir, case):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    loader = make_loader(batch_size=1, counts=(10,), max_seq_len=3, prefetch=1)
    try:
        original_collator = loader.collate_fn
        call_count = 0

        def collate(features):
            nonlocal call_count
            call_count += 1
            fail_on = 10 if case == "last_error" else 2
            if rank == 0 and case.endswith("error") and call_count == fail_on:
                raise ValueError("injected rank-local collation failure")
            return original_collator(features)

        loader.collate_fn = collate
        failed = False
        if case == "close":
            iterator = iter(loader)
            next(iterator)
            loader.mark_batch_consumed()
            if rank == 1:
                time.sleep(0.1)
            loader.close()
            loader.set_epoch(1)
            result = consume(loader)
            assert len(result) == 10
        else:
            try:
                consume(loader)
            except (ValueError, protocol.SynchronizedDataError):
                failed = True
            assert failed
        loader.close()
        Path(output_dir, f"{rank}.txt").write_text("ok")
    finally:
        if loader.is_launched:
            loader.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["middle_error", "last_error", "close"])
def test_gloo_loader_collate_error_and_close(tmp_path, case):
    mp.spawn(_loader_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), case), nprocs=2, join=True)
    assert (tmp_path / "0.txt").read_text() == "ok"
    assert (tmp_path / "1.txt").read_text() == "ok"


@pytest.mark.parametrize("step_limit,steps_per_epoch", [(1, 1), (20, 4)])
def test_trainer_owns_epochs_and_keeps_coarse_checkpoint(step_limit, steps_per_epoch):
    namespace = dict(
        torch=torch,
        PrefetchingPackedLoader=Loader,
        dist=SimpleNamespace(is_initialized=lambda: False, barrier=lambda: None),
        helper=SimpleNamespace(print_device_mem_info=lambda *args: None),
        logger=logging.getLogger(__name__),
        synchronize=lambda: None,
    )
    train = extract_method(ROOT / "veomni/trainer/llava_trainer.py", "VLMTrainer", "train", namespace)
    loader = make_loader(batch_size=2, counts=(9,), max_seq_len=3)
    trainer = SimpleNamespace(
        train_dataloader=loader,
        args=SimpleNamespace(
            train=SimpleNamespace(remote_dataloader=False, num_train_epochs=2, local_rank=0, global_rank=0)
        ),
        state=SimpleNamespace(global_step=0),
        train_steps=2,
        init_data_size=step_limit,
        device=torch.device("cpu"),
        on_train_begin=lambda: None,
        on_epoch_begin=lambda: None,
        on_train_end=lambda: None,
        destroy_distributed=lambda: None,
    )
    steps = []
    checkpoints = []
    epoch_calls = []
    set_epoch = loader.set_epoch

    def record_epoch(epoch):
        epoch_calls.append(epoch)
        set_epoch(epoch)

    loader.set_epoch = record_epoch

    def train_step(epoch, batches):
        assert len(batches) == 2
        steps.append(epoch)
        loader.mark_batch_consumed()
        trainer.current_step += 1
        trainer.state.global_step += 1

    trainer.train_step = train_step

    def on_epoch_end():
        assert not loader.is_launched
        checkpoints.append((trainer.current_epoch, trainer.current_step, loader.state_dict()))

    trainer.on_epoch_end = on_epoch_end
    train(trainer)
    assert steps == [0] * steps_per_epoch + [1] * steps_per_epoch
    assert epoch_calls == [0, 1]
    assert [(epoch, step) for epoch, step, _ in checkpoints] == [(0, steps_per_epoch), (1, steps_per_epoch)]
    assert [state["epoch"] for _, _, state in checkpoints] == [0, 1]
    assert all(state["resume_index"] == (1 if step_limit == 20 else 0) for _, _, state in checkpoints)


def _trainer_stop_worker(rank, rendezvous, output_dir, case):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    events = []

    class ScriptedLoader:
        def set_epoch(self, epoch):
            events.append(("epoch", epoch))

        def launch(self):
            pass

        def __iter__(self):
            yield [{"input_ids": torch.tensor([1, 2])}]
            if rank == 1:
                if case in ("error", "error_and_eof"):
                    raise RuntimeError("rank-local read failure")
                if case == "empty":
                    yield []
                elif case == "none":
                    yield None
                return
            if case != "error_and_eof":
                yield [{"input_ids": torch.tensor([3, 4])}]

        def close(self):
            events.append(("close",))

    namespace = dict(
        torch=torch,
        dist=dist,
        helper=SimpleNamespace(print_device_mem_info=lambda *a: None),
        logger=logging.getLogger(__name__),
        synchronize=lambda: None,
    )
    train = extract_method(ROOT / "veomni/trainer/llava_trainer.py", "VLMTrainer", "train", namespace)
    trainer = SimpleNamespace(
        train_dataloader=ScriptedLoader(),
        args=SimpleNamespace(
            train=SimpleNamespace(remote_dataloader=False, num_train_epochs=2, local_rank=rank, global_rank=rank)
        ),
        state=SimpleNamespace(global_step=0),
        train_steps=10,
        init_data_size=5,
        device=torch.device("cpu"),
        on_train_begin=lambda: None,
        on_epoch_begin=lambda: None,
        on_epoch_end=lambda: events.append(("epoch_end",)),
        on_train_end=lambda: events.append(("train_end",)),
        destroy_distributed=lambda: events.append(("destroy",)),
    )

    def train_step(epoch, batches):
        events.append(("step", epoch))
        trainer.current_step += 1
        trainer.state.global_step += 1

    trainer.train_step = train_step
    try:
        train(trainer)
        expected_epochs = 1 if case in ("error", "error_and_eof") else 2
        assert [e for e in events if e[0] == "step"] == [("step", e) for e in range(expected_epochs)]
        assert events.count(("close",)) == expected_epochs
        assert events.count(("epoch_end",)) == (0 if expected_epochs == 1 else 2)
        assert events[-2:] == [("train_end",), ("destroy",)]
        Path(output_dir, f"{rank}.txt").write_text(repr(events))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["error", "error_and_eof", "eof", "empty", "none"])
def test_trainer_gloo_stops_all_ranks_without_raising(tmp_path, case):
    mp.spawn(_trainer_stop_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), case), nprocs=2, join=True)
    assert (tmp_path / "0.txt").read_text() == (tmp_path / "1.txt").read_text()
