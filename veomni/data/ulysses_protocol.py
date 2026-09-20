"""Identity-based synchronization for independently processed Ulysses data streams."""

from collections.abc import Callable, Iterable, Iterator
from enum import IntEnum
from typing import Any


# Every raw sample ends with a terminal record, including empty/failed samples.
Record = tuple[int, int, bool, Any]


class RecordState(IntEnum):
    DATA = 0  # 当前记录是数据；指纹为 -1 时表示无有效 sample
    TERMINAL = 1  # 当前原始样本结束
    EOF = 2  # 本 rank 的整个数据流耗尽
    ERROR = 3  # 发生异常，所有 rank 一起退出；优先于 STOP
    STOP = 4  # 请求停止，例如 loader.close()


class SynchronizedDataError(RuntimeError):
    """An error already acknowledged by all peers in a metadata exchange."""


def synchronize_records(
    records: Iterable[Record],
    exchange: Callable[[list[int]], list[list[int]]],
    fingerprint: Callable[[Any], int],
    stopped: Callable[[], bool],
) -> Iterator[Record]:
    """Intersect ordered streams by identity, including explicit EOF participation.

    Peers ahead of the minimum identity retain their record. Missing chunks are
    discarded, never paired with a different sample. EOF peers continue exchanging
    metadata until everybody reaches EOF. A local iterator failure is propagated to
    all peers on the next exchange instead of stranding them in a collective.
    """
    iterator = iter(records)
    current = None
    advance = True
    eof = False
    while True:
        error = None
        # 同组 rank 交换的同步信息
        metadata = [
            0,  # [0] raw_id：当前 epoch 打乱、DP 分片后的原始样本序号
            0,  # [1] chunk_id：该样本内部的 chunk 序号，从 0 开始
            RecordState.EOF,  # [2] 状态：数据 / 样本结束 / 流结束 / 异常 / 停止
            -1,  # [3] 内容指纹：包含张量形状和内容；非负为有效哈希，-1 为无数据
        ]
        try:
            if advance and not eof:
                current = next(iterator, None)
                eof = current is None
            if not eof:
                raw_id, chunk_id, terminal, sample = current
                metadata[:3] = [raw_id, chunk_id, RecordState.TERMINAL if terminal else RecordState.DATA]
                if not terminal:
                    metadata[3] = fingerprint(sample)
        except Exception as exc:
            error = exc
        if error is not None:
            metadata[2] = RecordState.ERROR
        elif stopped():
            metadata[2] = RecordState.STOP
        try:
            peers = exchange(metadata)
        except SynchronizedDataError as exc:
            if error is not None:
                raise exc from error
            raise
        if any(peer[2] == RecordState.ERROR for peer in peers):
            raise RuntimeError("A Ulysses data peer failed while reading its sample stream") from error
        if any(peer[2] == RecordState.STOP for peer in peers) or all(peer[2] == RecordState.EOF for peer in peers):
            return
        active = [peer for peer in peers if peer[2] != RecordState.EOF]
        identity = min(tuple(peer[:2]) for peer in active)
        advance = not eof and tuple(metadata[:2]) == identity
        if (
            len(active) == len(peers)
            and all(tuple(peer[:2]) == identity for peer in peers)
            and all(peer[2] == RecordState.DATA and peer[3] >= 0 for peer in peers)
            and all(peer[3] == peers[0][3] for peer in peers)
        ):
            yield current
