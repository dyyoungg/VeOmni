
import os
import json
import time
import random
from typing import Optional, List, Dict, Any, Tuple, Iterator
import types
import traceback
import heapq
import queue
import threading
import datetime
import pickle
import hashlib


import math
from aoss_client.client import Client as CephClient
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import torch.distributed as dist

from veomni.utils.constants import (
    get_image_video_audio_placeholder,
    _CHAT_TEMPLATES,

)
from veomni.data.multimodal.image_utils import get_adaptive_pool_size
from veomni.data.data_collator import UlysessOmniDataSharderCollator
from veomni.data.llavaomni_processor import OmniSampleProcessor, LongVideoProcessor
from veomni.distributed.sequence_parallel import get_data_parallel_rank, get_data_parallel_world_size, get_ulysses_sequence_parallel_cpu_group
from veomni.distributed.parallel_state import get_parallel_state, _init_parallel_state
from veomni.utils.logging import get_logger
from veomni.utils.helper import read_data
from veomni.data.ulysses_protocol import RecordState, SynchronizedDataError, synchronize_records

try:
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.services.bos.bos_client import BosClient
    from baidubce.retry.retry_policy import BackOffRetryPolicy

    ACCESS_KEY_ID = os.environ.get("BAIDU_AK", "")
    SECRET_ACCESS_KEY = os.environ.get("BAIDU_SK", "")
    BOS_HOST = "https://bj.bcebos.com"
    global_config = BceClientConfiguration(
        credentials=BceCredentials(ACCESS_KEY_ID, SECRET_ACCESS_KEY),
        endpoint=BOS_HOST,
        retry_policy=BackOffRetryPolicy(max_error_retry=3, max_delay_in_millis=20000),
    )
except Exception:
    global_config = None
    BosClient = None

AOSS_FILE = "/mnt/afs/yangdeyu/aoss_ydy_game.conf"
logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class UlysessOmniProcessor:
    def __init__(self, tokenizer, data_args, training_args, model_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.model_args = model_args
        self.rank = int(os.environ.get("RANK", 0))
        self.ceph_client = CephClient(AOSS_FILE)
        try:
            self.bos_client = BosClient(global_config)
        except Exception:
            self.bos_client = None
        self.processor: Optional[OmniSampleProcessor] = None
        self.longvideo_processor = None
        self.standard_processor = None

    def build_inputs_token(
        self,
        input_str: str = "",
        input_type: Optional[str] = None,
        return_tensor: bool = True,
    ):
        arc = self.model_args.model_arc
        templates = _CHAT_TEMPLATES.get(arc)
        if templates is None:
            raise NotImplementedError(f"Unsupported model_arc: {arc!r}")

        template = templates.get(input_type)
        if template is None:
            raise ValueError(
                f"Unknown input_type {input_type!r}. "
                f"Valid options: {list(templates.keys())}"
            )

        if input_type == "assistant_prefix":
            text = template
        else:
            text = template.format(input_str)

        tokens = self.tokenizer(text)["input_ids"]
        return torch.tensor(tokens, dtype=torch.long) if return_tensor else tokens

    def init_image_processor(self) -> None:
        if self.processor is None:
            self.processor = OmniSampleProcessor(
                tokenizer=self.tokenizer,
                model_args=self.model_args,
                data_args=self.data_args,
                training_args=self.training_args,
                ceph_client=self.ceph_client,
                bos_client=self.bos_client,
                rank=self.rank,
                build_inputs_token_fn=self.build_inputs_token,
                preprocess_workers=1,
            )
            self.processor.init_image_processor()

        if self.longvideo_processor is None:

            self.longvideo_processor = LongVideoProcessor(tokenizer=self.tokenizer,
                    model_args=self.model_args,
                    data_args=self.data_args,
                    training_args=self.training_args,
                    ceph_client=self.ceph_client,
                    bos_client=self.bos_client,
                    rank=self.rank,
                    build_inputs_token_fn=self.build_inputs_token,
                    preprocess_workers=1,)
            self.longvideo_processor.init_image_processor()


    
    def __call__(self, sample_data: Dict[str, Any], sample_idx) -> Optional[Dict[str, Any]]:
        self.init_image_processor()
        # A raw sample owns its RNG, so worker scheduling and earlier failures do
        # not change its augmentation or chunking when replayed after a checkpoint.
        seed = f"{getattr(self.training_args, 'seed', 42)}:{self.epoch}:{sample_idx}"
        self.processor._rng = random.Random(seed)
        self.longvideo_processor._rng = random.Random(seed)
        if "subtitles" in sample_data:
            # Generator返回 generator 对象，不执行内部代码
            return self.longvideo_processor.process(sample_data, sample_idx)
        else:
            # 普通短视频
            # print(self.dp_rank, sample_idx)
            return self.processor.process(sample_data, sample_idx)
        


class UlyssesStreamingDataset(IterableDataset):
    """
    Iterable dataset that:
      * shards data files across dp_rank / num_workers
      * processes each sample via UlysessOmniProcessor
      * yields tuples (global_idx, sub_idx, is_last, sample_or_None)
 
    The tuple wrapper lets ReorderingDataLoader reconstruct strict global order
    when num_workers > 1 scrambles delivery.
    """
 
    def __init__(self, tokenizer, data_args, model_args, training_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        self.epoch = 0
        self.skip_samples_count = 0
        self.offline_split = getattr(data_args, "offline_dataset_split", False)

        # offset-based lazy reading mode
        offset_path = getattr(data_args, "offset_file_path", "")
        mapping_path = getattr(data_args, "file_maping_path", "")
        self.use_offset = bool(offset_path and mapping_path
                               and os.path.exists(offset_path)
                               and os.path.exists(mapping_path))
        self._all_offsets = None      # np.ndarray [N, 2] (mmap)
        self.file_mapping = None      # List[str]

        self.dp_rank, self.dp_world_size = self._detect_distribution_mode()
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.data_list = self._load_data_list(data_args.train_path)

        try:
            ps = get_parallel_state()
            self._sp_rank = ps.ulysses_rank if (ps is not None and ps.sp_size > 1) else 0
        except Exception:
            self._sp_rank = 0
            logger.info("get sp rank failed. set sp_rank=0 by default.")
        logger.info(f"[DP Rank {self.dp_rank} SP Rank {self._sp_rank}] "
                     f"Loaded {len(self.data_list)} samples (offset_mode={self.use_offset}).")
        # Processor is lazily initialized inside each DataLoader worker
        self._processor: Optional[UlysessOmniProcessor] = None
 
    # ── Data loading ─────────────────────────────────────────────────────────
 
    def _load_data_list(self, data_path: str):
        # ── Offset-based lazy reading mode ────────────────────────────────
        if self.use_offset:
            offset_path = self.data_args.offset_file_path
            mapping_path = self.data_args.file_maping_path
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.file_mapping = json.load(f)
            self._all_offsets = np.load(offset_path, mmap_mode='r')  # [N, 2], uint64
            # 按 dp_rank 确定性分片 → 同一 DP group 内所有 SP rank 拿到相同子集
            total = len(self._all_offsets)
            indices = list(range(self.dp_rank, total, self.dp_world_size))
            logger.info(
                f"[DP Rank {self.dp_rank}] Offset mode: {total} total samples, "
                f"{len(indices)} assigned to this DP rank."
            )
            return indices  # data_list 存的是 offset 数组的 global index

        # ── Original full-load mode ───────────────────────────────────────
        if not self.offline_split:
            assert isinstance(data_path, str), "offline spilt is False, data path must in json or jsonl format!"
            full = read_data(data_path)
            return full[self.dp_rank :: self.dp_world_size]

        assert os.path.isdir(data_path), (
            f"offline_dataset_split=True requires data_path to be a directory, got: {data_path}"
        )
        shard_path = os.path.join(data_path, f"train_{self.dp_rank}.jsonl")
        logger.info(f"[DP Rank {self.dp_rank}] Loading shard: {shard_path}")
        return read_data(shard_path)
 
    # ── Distribution helpers ──────────────────────────────────────────────────
 
    @property
    def _is_sp_mode(self) -> bool:
        if dist.is_initialized():
            ps = get_parallel_state()
            return ps is not None and getattr(ps, "sp_size", 1) > 1
        return False
 
    def _detect_distribution_mode(self):
        if not dist.is_initialized():
            return 0, 1
        if self._is_sp_mode:
            return get_data_parallel_rank(), get_data_parallel_world_size()
        return dist.get_rank(), dist.get_world_size()
 
    # ── Epoch / resume API ───────────────────────────────────────────────────
 
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # logger.info(f"[DP Rank {self.dp_rank}] Epoch set to {epoch}.")
 
    def set_consumed_samples(self, n: int):
        self.skip_samples_count = n
        if n > 0:
            logger.info(f"[DP Rank {self.dp_rank}] Will skip first {n} samples.")
 
    def __len__(self):
        return len(self.data_list)
 
    # ── Processor (lazy, per-worker) ─────────────────────────────────────────
 
    def _init_processor(self):
        if self._processor is None:
            torch.set_num_threads(1)
            self._processor = UlysessOmniProcessor(
                tokenizer=self.tokenizer,
                data_args=self.data_args,
                training_args=self.training_args,
                model_args=self.model_args,
            )
            self._processor.init_image_processor()
            # logger.info(
            #     f"[DP Rank {self.dp_rank} PID {os.getpid()}] UlyssesOmniProcessor initialized."
            # )
 
    # ── Iteration helpers ─────────────────────────────────────────────────────
 
    def _get_shuffled_data(self) -> List:
        g = torch.Generator()
        g.manual_seed(self.epoch + getattr(self.training_args, "seed", 42))
        perm = torch.randperm(len(self.data_list), generator=g).tolist()
        return [self.data_list[i] for i in perm]

    # ── Offset-based lazy reading helpers ─────────────────────────────────

    def _get_file_handle(self, file_path: str):
        """Per-worker file handle cache to avoid repeated open/close."""
        if not hasattr(self, '_file_cache'):
            self._file_cache = {}
        if file_path not in self._file_cache:
            if len(self._file_cache) >= 100:
                oldest = next(iter(self._file_cache))
                self._file_cache[oldest].close()
                del self._file_cache[oldest]
            self._file_cache[file_path] = open(file_path, "r", encoding="utf-8")
        return self._file_cache[file_path]

    def _read_sample_by_offset(self, offset_idx: int) -> Dict:
        """Seek to byte offset in the source JSONL and read one sample."""
        file_id, byte_offset = self._all_offsets[offset_idx]
        file_path = self.file_mapping[int(file_id)]
        fh = self._get_file_handle(file_path)
        fh.seek(int(byte_offset))
        line = fh.readline()
        return json.loads(line)
 
    def __iter__(self):
        self._init_processor()
        self._processor.epoch = self.epoch
        data = self._get_shuffled_data()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        for global_idx in range(self.skip_samples_count, len(data)):
            if global_idx % num_workers != worker_id:
                continue
            sub_idx = 0
            try:
                item = data[global_idx]
                sample_data = self._read_sample_by_offset(item) if self.use_offset else item
                result = self._processor(sample_data, sample_idx=global_idx)
                if result is not None:
                    if not isinstance(result, (list, types.GeneratorType)):
                        result = [result]
                    for sample in result:
                        yield (global_idx, sub_idx, False, sample)
                        sub_idx += 1
            except Exception:
                logger.warning(
                    f"[DP Rank {self.dp_rank} Worker {worker_id}] "
                    f"Processing failed at sample {global_idx}: {traceback.format_exc()}"
                )
            # Never omit an empty sample or reuse an already emitted chunk ID.
            yield (global_idx, sub_idx, True, None)


# ---------------------------------------------------------------------------
# Reordering wrapper (restores strict global order after multi-worker shuffle)
# ---------------------------------------------------------------------------
 
class ReorderingDataLoader:
    """
    Consumes tuples (global_idx, sub_idx, is_last, data) from a DataLoader
    driven with num_workers > 1 and re-emits data in strict (global, sub) order.
 
    A min-heap buffers out-of-order arrivals; if the buffer grows beyond
    MAX_BUFFER_SIZE the head is forcibly emitted to avoid unbounded memory use.
    """
 
    MAX_BUFFER_SIZE = 200
 
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._start_idx = 0
        self.rank = dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", 0))
 
    @property
    def dataset(self):
        return self.dataloader
 
    def set_consumed_samples(self, n: int):
        self._start_idx = n
 
    def __iter__(self):
        iterator = iter(self.dataloader)
        next_global = self._start_idx
        next_sub    = 0
        self._start_idx = 0
 
        heap: List = []
        def get_next_item():
            start_t = time.time()
            try:
                item = next(iterator)
                duration = time.time() - start_t
                if duration > 60:
                    logger.warning(f"[Rank {self.rank}] WARNING: PyTorch DataLoader took {duration:.2f}s to yield an item!")
                return item
            except StopIteration:
                return None
 
        def _advance(data, is_last):
            nonlocal next_global, next_sub
            yield (next_global, next_sub, is_last, data)
            if is_last:
                next_global += 1
                next_sub = 0
            else:
                next_sub += 1
 
        def _drain():
            while heap and heap[0][0] == next_global and heap[0][1] == next_sub:
                _, _, b_is_last, b_data = heapq.heappop(heap)
                yield from _advance(b_data, b_is_last)
                
 
        while True:
            batch = get_next_item()
            if batch is None:
                break
            g_idx, s_idx, is_last, data = batch
 
            if g_idx == next_global and s_idx == next_sub:
                yield from _advance(data, is_last)
                del data
                yield from _drain()
 
            elif g_idx >= next_global:
                heapq.heappush(heap, (g_idx, s_idx, is_last, data))

            elif g_idx < next_global:
                logger.error(
                    f"[Rank {self.rank}] FATAL SP DESYNC: Received late sample g_idx={g_idx}, "
                    f"but stream already advanced to next_global={next_global}. "
                    "This will permanently break Sequence Parallelism!"
                )
 
            # Evict if buffer is too large
            while len(heap) > self.MAX_BUFFER_SIZE:
                logger.warning(
                    f"[Rank {self.rank}] Buffer size exceeded {self.MAX_BUFFER_SIZE}. "
                    f"Forcing eviction. Expected next_global={next_global}, "
                    f"but jumping to {heap[0][0]}."
                )
                b_g, b_s, b_is_last, b_data = heapq.heappop(heap)
                next_global, next_sub = b_g, b_s
                yield from _advance(b_data, b_is_last)
                del b_data
                yield from _drain()
 
        # Drain remaining buffer at end of epoch
        while heap:
            b_g, b_s, b_is_last, b_data = heapq.heappop(heap)
            next_global, next_sub = b_g, b_s
            yield from _advance(b_data, b_is_last)
            del b_data  


# ---------------------------------------------------------------------------
# Multimodal packer
# ---------------------------------------------------------------------------
 
class MultimodalPacker:
    """
    Greedy bin-packing of samples up to max_seq_len.
    Consumes the raw per-sample dicts (after ReorderingDataLoader) and emits
    packed dicts whose `input_ids` length ≤ max_seq_len.  The packed dict uses
    `sample_lens` (a 1-D tensor of per-sample lengths) instead of a 2-D
    attention mask.
 
    Expected per-sample keys
    ------------------------
        input_ids           : [L]
        labels              : [L]
        attention_mask_len  : scalar or 1-element tensor  ← L
        pixel_values        : [P, D] | None
        image_grid_thw      : [M, 3] | None
        pixel_values_video  : [V, D] | None
        video_grid_thw      : [K, 3] | None
        audio_features      : [A, F] | None
        audio_features_lens : [Na]   | None
    """
 
    _MULTIMODAL_KEYS = (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_video",
        "video_grid_thw",
        "audio_features",
        "audio_features_lens",
        "image_downsample_ratios",
        "video_downsample_ratios",
    )
 
    def __init__(
        self,
        source_iterator: Iterator[Dict],
        tokenizer,
        model_args,
        max_seq_len: int,
    ):
        self.source = source_iterator
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.model_args = model_args
 
        # Token IDs for modality placeholders (used for over-length sample check)
        self._image_token_id, self._video_token_id, self._audio_token_id = get_image_video_audio_placeholder(tokenizer)
 
        self._reset_buffer()
 
    def _reset_buffer(self):
        self._buf: Dict[str, List] = {
            "input_ids": [],
            "labels": [],
            "sample_lens": [],
            **{k: [] for k in self._MULTIMODAL_KEYS},
        }
        self._cur_len = 0
        self._resume_sample = None

    # ── Length estimation helpers ─────────────────────────────────────────────
 
    def _image_tokens(self, grid_thw: Optional[torch.Tensor], downsample_ratio: Optional[float] = None) -> int:
        if grid_thw is None:
            return 0
        if downsample_ratio is None:
            downsample_ratio = getattr(self.model_args, "mm_downsample_ratio", 16)
        total = 0
        for thw in grid_thw:
            t, h, w = thw
            mh, mw = get_adaptive_pool_size(
                int(h) // 2, int(w) // 2,
                downsample_ratio,
            )
            total += int(t) * mh * mw
        return total
 
    def _audio_tokens(self, lens: Optional[torch.Tensor]) -> int:
        if lens is None:
            return 0
        r = getattr(self.model_args, "audio_downsample_ratio", 10)
        return sum((int(l) + r - 1) // r for l in lens)
 
    # ── Pack & yield ─────────────────────────────────────────────────────────
 
    def _flush(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
 
        if self._buf["input_ids"]:
            out["input_ids"]  = torch.cat(self._buf["input_ids"],  dim=0)
            out["labels"]     = torch.cat(self._buf["labels"],     dim=0)
            out["sample_lens"] = torch.cat(self._buf["sample_lens"], dim=0).to(torch.int32)
 
        for key in self._MULTIMODAL_KEYS:
            valid = [t for t in self._buf[key] if t is not None and t.numel() > 0]
            out[key] = torch.cat(valid, dim=0) if valid else None
 
        out["_resume_sample"] = self._resume_sample
        return out

    def __iter__(self):
        for sample in self.source:
            if sample is None:
                continue
 
            # Normalise attention_mask_len to a plain int
            attn_len_raw = sample.attention_mask_len
            if attn_len_raw is None:
                attn_len_raw = [sample.input_ids.numel()]
            if isinstance(attn_len_raw, torch.Tensor):
                sample_len = int(attn_len_raw.sum().item()) if attn_len_raw.numel() > 1 else int(attn_len_raw.item())
            else:
                sample_len = int(attn_len_raw) if not hasattr(attn_len_raw, "__iter__") else sum(attn_len_raw)
 
            # ── Handle over-length samples ────────────────────────────────────
            if sample_len > self.max_seq_len:
                # Truncate if all visual tokens are still present after crop
                trunc_ids = sample.input_ids[:self.tokenizer.model_max_length]
                trunc_lbl = sample.labels[:self.tokenizer.model_max_length]
 
                # Use per-sample downsample ratio when available (dynamic compression)
                img_ratio = float(sample.image_downsample_ratios[0]) if getattr(sample, 'image_downsample_ratios', None) is not None and sample.image_downsample_ratios.numel() > 0 else None
                vid_ratio = float(sample.video_downsample_ratios[0]) if getattr(sample, 'video_downsample_ratios', None) is not None and sample.video_downsample_ratios.numel() > 0 else None
                img_ok  = (trunc_ids == self._image_token_id).sum() == self._image_tokens(sample.image_grid_thw, img_ratio)
                vid_ok  = (trunc_ids == self._video_token_id).sum() == self._image_tokens(sample.video_grid_thw, vid_ratio)
                aud_ok  = (trunc_ids == self._audio_token_id).sum() == self._audio_tokens(sample.audio_features_lens)
 
                if img_ok and vid_ok and aud_ok:
                    sample.input_ids = trunc_ids
                    sample.labels    = trunc_lbl
                    sample_len          = self.tokenizer.model_max_length
                    sample.attention_mask_len = torch.tensor(
                        [self.tokenizer.model_max_length], dtype=torch.long
                    )
                else:
                    # Cannot safely truncate – discard
                    logger.warning(f"Discarding over-length sample (len={sample_len}) that cannot be safely truncated.")
                    continue
 
            # ── Flush buffer if adding this sample would overflow ─────────────
            if self._cur_len + sample_len > self.max_seq_len:
                if self._cur_len > 0:
                    yield self._flush()
                    time.sleep(0.001)
                    self._reset_buffer()
 
            # ── Append to buffer ──────────────────────────────────────────────
            self._cur_len += sample_len
            self._resume_sample = sample._ulysses_sample_index
            self._buf["input_ids"].append(sample.input_ids)
            self._buf["labels"].append(sample.labels)
 
            # sample_lens: store per-sample length as a 1-element int32 tensor
            if isinstance(attn_len_raw, torch.Tensor) and attn_len_raw.numel() > 1:
                self._buf["sample_lens"].append(attn_len_raw.to(torch.int32))
            else:
                self._buf["sample_lens"].append(torch.tensor([sample_len], dtype=torch.int32))
 
            for key in self._MULTIMODAL_KEYS:
                val = getattr(sample, key, None)
                if val is not None:
                    self._buf[key].append(val)
 
        # ── Emit remaining buffer ─────────────────────────────────────────────
        if self._cur_len > 0:
            yield self._flush()
            self._reset_buffer() 


# ---------------------------------------------------------------------------
# Prefetching loader (main process background thread)
# ---------------------------------------------------------------------------
 
class PrefetchingPackedLoader:
    """One epoch of packed micro-batches with coarse raw-sample checkpointing.

    Resume re-reads the current raw sample from its beginning. Chunks of a long
    video may be trained again; packing buffers and chunk cursors are not saved.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        dataset,
        tokenizer,
        model_args,
        max_seq_len,
        batch_size,
        collate_fn=None,
        prefetch_batches=2,
        start_epoch=0,
        num_train_epochs=1,
    ):
        if batch_size < 1 or prefetch_batches < 1:
            raise ValueError("batch_size and prefetch_batches must be positive")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.prefetch_batches = prefetch_batches
        self.epoch = start_epoch
        self.num_train_epochs = num_train_epochs
        self.samples_consumed = 0
        self._resume_index = 0
        self._pending_sample = None
        self.is_launched = False
        self.queue = None
        self.stop_event = threading.Event()
        self.producer_thread = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        ps = get_parallel_state()
        self.dp_rank = ps.dp_rank
        self.dp_size = ps.dp_size
        self.sp_size = ps.ulysses_size
        self.cpu_group = get_ulysses_sequence_parallel_cpu_group() if self.sp_size > 1 else None
        if self.sp_size > 1 and self.cpu_group is None:
            raise RuntimeError("Ulysses data synchronization requires the CPU Gloo group")
        if getattr(ps, "cp_size", 1) > 1:
            raise ValueError("The Ulysses omni dataloader does not support combined context parallelism")

    @property
    def raw_dataset(self):
        obj = self.dataset
        while hasattr(obj, "dataset"):
            obj = obj.dataset
        return obj

    @property
    def data_list(self):
        return self.raw_dataset.data_list

    @property
    def raw_samples_consumed(self):
        return self.samples_consumed

    def _propagate(self, obj, method, value):
        if hasattr(obj, method):
            getattr(obj, method)(value)
        if hasattr(obj, "dataset"):
            self._propagate(obj.dataset, method, value)

    def set_epoch(self, epoch):
        if epoch == self.epoch:
            return  # Preserve a restored mid-epoch sample index.
        if self.is_launched:
            raise RuntimeError("Close the Ulysses dataloader before changing epochs")
        self.epoch = epoch
        self._resume_index = 0
        self._pending_sample = None
        self.samples_consumed = 0

    def mark_batch_consumed(self):
        """Commit progress only after the trainer successfully updates parameters."""
        if self._pending_sample is None:
            raise RuntimeError("No Ulysses batch is pending consumption")
        self._resume_index = self._pending_sample
        self._pending_sample = None
        self.samples_consumed = self._resume_index

    def _configuration(self):
        raw = self.raw_dataset
        return {
            "data_path": raw.data_args.train_path,
            "num_samples": len(self.data_list),
            "seed": getattr(raw.training_args, "seed", 42),
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "sp_size": self.sp_size,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
        }

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "epoch": self.epoch,
            "resume_index": self._resume_index,
            "samples_consumed": self.samples_consumed,
            "configuration": self._configuration(),
        }

    def load_state_dict(self, state):
        if self.is_launched:
            raise RuntimeError("Restore the Ulysses dataloader before launching it")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("Unsupported Ulysses dataloader checkpoint version")
        if state["configuration"] != self._configuration():
            raise ValueError("Ulysses dataloader configuration differs from its checkpoint")
        resume_index = int(state["resume_index"])
        if not 0 <= resume_index <= len(self.data_list):
            raise ValueError("Invalid Ulysses dataloader resume index")
        self.epoch = state["epoch"]
        self._resume_index = resume_index
        self.samples_consumed = resume_index

    def launch(self):
        if self.is_launched:
            return
        self._propagate(self.dataset, "set_epoch", self.epoch)
        self._propagate(self.dataset, "set_consumed_samples", self._resume_index)
        self.queue = queue.Queue(maxsize=self.prefetch_batches)
        self.stop_event.clear()
        self.is_launched = True
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)
        self.producer_thread.start()

    def close(self):
        if not self.is_launched:
            return
        self.stop_event.set()
        # Drain to unblock a producer waiting to enqueue while an SP peer waits
        # in the next metadata exchange. Stop is agreed in that same exchange.
        deadline = time.monotonic() + 30
        while self.producer_thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.get(timeout=0.1)
            except queue.Empty:
                pass
            self.producer_thread.join(timeout=0.1)
        if self.producer_thread.is_alive():
            raise RuntimeError("Ulysses producer did not stop; refusing to reuse its queue or process group")
        self.producer_thread = None
        self.is_launched = False

    def _sample_fingerprint(self, item):
        if item is None:
            return -1
        digest = hashlib.blake2b(digest_size=8)
        for key in ("input_ids", "labels", "attention_mask_len", *MultimodalPacker._MULTIMODAL_KEYS):
            value = getattr(item, key, None)
            digest.update(key.encode())
            values = value if isinstance(value, (list, tuple)) else [value]
            for tensor in values:
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.detach().cpu().contiguous()
                    digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                else:
                    digest.update(repr(tensor).encode())
        return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)

    def _exchange_metadata(self, metadata):
        if self.cpu_group is None:
            peers = [metadata]
        else:
            local = torch.tensor(metadata, dtype=torch.long)
            gathered = [torch.empty_like(local) for _ in range(self.sp_size)]
            try:
                dist.all_gather(gathered, local, group=self.cpu_group)
            except RuntimeError as exc:
                raise SynchronizedDataError("Ulysses metadata collective failed") from exc
            peers = [peer.tolist() for peer in gathered]
        if any(peer[2] == RecordState.ERROR for peer in peers):
            raise SynchronizedDataError("A Ulysses peer failed during reading, packing or collation")
        return peers

    def _create_cpu_synced_iterator(self, raw_iterator):
        for raw_id, _, _, sample in synchronize_records(
            raw_iterator, self._exchange_metadata, self._sample_fingerprint, self.stop_event.is_set
        ):
            sample._ulysses_sample_index = raw_id
            yield sample

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _producer(self):
        try:
            source = self._create_cpu_synced_iterator(iter(self.dataset))
            packer = MultimodalPacker(source, self.tokenizer, self.model_args, self.max_seq_len)
            batch_buf = []
            for packed in packer:
                sample_index = packed.pop("_resume_sample")
                # Each packed sequence is a micro-batch; the trainer accumulates
                # the returned list before one optimizer step.
                batch = self.collate_fn([packed]) if self.collate_fn else packed
                batch_buf.append(batch)
                if len(batch_buf) == self.batch_size:
                    self._put((batch_buf, sample_index))
                    batch_buf = []
            # Keep the number of forwards identical across DP ranks. An
            # incomplete accumulation group is dropped, as in the omni loader.
            if batch_buf and not self.stop_event.is_set():
                logger.info(f"Dropping {len(batch_buf)}/{self.batch_size} trailing Ulysses micro-batches")
            # A final exchange covers failures while collating the last pack,
            # after the record iterator has already agreed on EOF.
            self._exchange_metadata([0, 0, RecordState.STOP if self.stop_event.is_set() else RecordState.EOF, -1])
        except Exception as exc:
            if not isinstance(exc, SynchronizedDataError):
                try:
            
                    self._exchange_metadata([0, 0, RecordState.ERROR, -1])
                except SynchronizedDataError:
                    pass
            self._put(exc)
        finally:
            self._put(None)

    def __iter__(self):
        self.launch()
        return self._consumer()

    def _consumer(self):
        while True:
            try:
                item = self.queue.get(timeout=600)
            except queue.Empty as exc:
                raise TimeoutError(f"[Rank {self.rank}] Ulysses dataloader queue timeout") from exc
            if isinstance(item, Exception):
                raise item
            if item is None:
                self._resume_index = len(self.data_list)
                self.samples_consumed = len(self.data_list)
                return
            batches, sample_index = item
            self._pending_sample = sample_index
            yield batches


def make_ulysses_train_dataloader(data_args, training_args, model_args, tokenizer):
    """
    Build the complete Ulysses SP training dataloader.
 
    Args:
        model_args      : model configuration namespace
        data_args       : data configuration namespace
        training_args   : HuggingFace TrainingArguments (or compatible)
        tokenizer       : pre-built tokenizer (already configured)
 
    Returns:
        PrefetchingPackedLoader: one epoch of lists of collated micro-batches.
        Call mark_batch_consumed() after each successful optimizer step.
    """
   
    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_dataset = UlyssesStreamingDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )
 
    # ── Multi-worker DataLoader (raw, un-collated) ────────────────────────────
    stream_loader = DataLoader(
        raw_dataset,
        batch_size=None,           # disable auto-batching; items are already dicts
        num_workers=getattr(training_args, "dataloader_num_workers", 2),
        prefetch_factor=(getattr(training_args, "dataloader_prefetch_factor", 2)
                         if getattr(training_args, "dataloader_num_workers", 2) > 0 else None),
        persistent_workers=False,
    )
 
    # ── Reorder across workers ────────────────────────────────────────────────
    reorder_loader = ReorderingDataLoader(stream_loader)
 
    # ── Collator ──────────────────────────────────────────────────────────────
    collator = UlysessOmniDataSharderCollator(pad_token_id=tokenizer.pad_token_id)
 
    # ── Prefetching packer ────────────────────────────────────────────────────
    train_loader = PrefetchingPackedLoader(
        dataset=reorder_loader,
        tokenizer=tokenizer,
        model_args=model_args,
        max_seq_len=training_args.model_max_length,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        prefetch_batches=getattr(training_args, "dataloader_prefetch_batches", 2),
        num_train_epochs=int(getattr(training_args, "num_train_epochs", 1)),
    )
 
    return train_loader



def check_sp_consistency(tensor: torch.Tensor, sp_group, name: str, step: int = -1):
    global_rank = dist.get_rank()
    sp_rank = dist.get_rank(group=sp_group)
    sp_world_size = dist.get_world_size(group=sp_group)
    
    
    group_start_rank = (global_rank // sp_world_size) * sp_world_size
    peer_ranks = list(range(group_start_rank, group_start_rank + sp_world_size))
    local_checksum = tensor.sum().reshape(1).to(tensor.device)

    group_size = dist.get_world_size(group=sp_group)
    gathered_checksums = [torch.zeros_like(local_checksum) for _ in range(group_size)]
   
    try:
        
        dist.all_gather(gathered_checksums, local_checksum, group=sp_group)
    except Exception as e:
       
        print(f"\n❌ [Rank {global_rank}] CRASHED during '{name}' check!\n"
              f"   I was waiting for peers: {peer_ranks}\n"
              f"   One of them likely exited early or died.\n", flush=True)
        raise e
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
  
    ref_val = gathered_checksums[0]
    is_consistent = True
    
    current_rank = dist.get_rank()
    
    for i, val in enumerate(gathered_checksums):
        
        if not torch.allclose(ref_val, val, rtol=1e-5, atol=1e-5):
            is_consistent = False
            if dist.get_rank(group=sp_group) == 0:
                print(f"❌ [Mismatch] {name}: SP_Rank 0 sum={ref_val.item()}, SP_Rank {i} sum={val.item()}")

    return is_consistent

class _UlyssesTestAudit:
    """Test-only checks before sample filtering and before SP slicing."""

    def __init__(self, exchange, collator):
        self.exchange = exchange
        self.collator = collator
        self.error = None
        self.samples = 0
        self.empty_samples = 0
        self.modalities = {"image": 0, "video": 0, "audio": 0}

    @staticmethod
    def fingerprint(value):
        """Include dtype, shape and every byte; equal sums are insufficient."""
        digest = hashlib.sha256()

        def update(item):
            digest.update(type(item).__name__.encode())
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item):
                    update(key)
                    update(item[key])
            elif isinstance(item, (list, tuple)):
                digest.update(str(len(item)).encode())
                for child in item:
                    update(child)
            else:
                digest.update(repr(item).encode())

        update(value)
        return digest.hexdigest()

    def exchange_metadata(self, metadata):
        try:
            peers = self.exchange(metadata)
        except Exception as exc:
            self.error = repr(exc)
            raise
        # A deliberate bounded stop can interrupt peers at different records.
        if any(peer[2] == RecordState.STOP for peer in peers):
            return peers
        if any(peer != peers[0] for peer in peers):
            self.error = f"Pre-filter Ulysses mismatch (sample/chunk/terminal/EOF/content): {peers}"
            raise SynchronizedDataError(self.error)
        if metadata[2] == RecordState.TERMINAL:
            self.empty_samples += int(metadata[1] == 0)
        elif metadata[2] == RecordState.DATA and metadata[3] >= 0:
            self.samples += 1
        return peers

    def collate(self, features):
        raw = features[0]
        fingerprints = {key: self.fingerprint(value) for key, value in raw.items()}
        for modality, key in (("image", "pixel_values"), ("video", "pixel_values_video"), ("audio", "audio_features")):
            value = raw.get(key)
            self.modalities[modality] += int(value is not None)
        try:
            batch = self.collator(features)
        except Exception as exc:
            self.error = repr(exc)
            raise
        batch["_ulysses_test_fingerprints"] = fingerprints
        return batch


def test_ulysess(args=None):
    """Run with torchrun and a trainer YAML; never initialize model weights.

    ULYSSES_TEST_MAX_SAMPLES (128) bounds the offset subset, MAX_STEPS (10)
    bounds iteration, and SP_SIZE (2) selects the Ulysses group size. All three
    names have the ULYSSES_TEST_ prefix. A zero MAX_STEPS runs to subset EOF.
    """
    import copy
    import tempfile

    from transformers import AutoTokenizer

    if args is None:
        from veomni.arguments import parse_args
        from veomni.trainer.llava_trainer import VeOmniVLMArguments

        args = parse_args(VeOmniVLMArguments)
    model_args, data_args, training_args = args.model, copy.copy(args.data), args.train
    max_samples = int(os.environ.get("ULYSSES_TEST_MAX_SAMPLES", "128"))
    max_steps = int(os.environ.get("ULYSSES_TEST_MAX_STEPS", "10"))
    sp_size = int(os.environ.get("ULYSSES_TEST_SP_SIZE", "2"))
    if max_samples < 1 or max_steps < 0 or sp_size < 2:
        raise ValueError("Require MAX_SAMPLES > 0, MAX_STEPS >= 0 and SP_SIZE >= 2")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", timeout=datetime.timedelta(minutes=5))
    train_loader = None
    audit_group = None
    try:
        world_size = dist.get_world_size()
        if world_size % sp_size:
            raise ValueError(f"world_size={world_size} must be divisible by SP_SIZE={sp_size}")
        _init_parallel_state(
            dp_size=world_size // sp_size,
            ulysses_size=sp_size,
            device_type="cuda" if use_cuda else "cpu",
        )
        ps = get_parallel_state()
        if ps.ulysses_size != sp_size or ps.dp_size != world_size // sp_size or ps.cp_size != 1:
            raise ValueError("Existing parallel state does not match the requested test topology")
        # Consumer checks must not interleave with producer metadata collectives.
        audit_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(minutes=5))

        def gather(value):
            peers = [None] * world_size
            dist.all_gather_object(peers, value, group=audit_group)
            return peers

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.config_path or model_args.model_path,
            model_max_length=training_args.model_max_length,
        )
        with tempfile.TemporaryDirectory(prefix="ulysses-consistency-") as scratch:
            offsets = np.load(data_args.offset_file_path, mmap_mode="r")
            if len(offsets) == 0:
                raise ValueError("The offset file contains no samples")
            # Deterministically cover the mapping without materializing/shuffling
            # tens of millions of indices just to run a bounded smoke test.
            rows = np.linspace(0, len(offsets) - 1, min(max_samples, len(offsets)), dtype=np.int64)
            data_args.offset_file_path = os.path.join(scratch, "offsets.npy")
            np.save(data_args.offset_file_path, offsets[rows])
            train_loader = make_ulysses_train_dataloader(data_args, training_args, model_args, tokenizer)
            audit = _UlyssesTestAudit(train_loader._exchange_metadata, train_loader.collate_fn)
            train_loader._exchange_metadata = audit.exchange_metadata
            train_loader.collate_fn = audit.collate
            iterator = iter(train_loader)
            steps = micro_batches_checked = 0
            try:
                while max_steps == 0 or steps < max_steps:
                    try:
                        batches = next(iterator)
                        local = {
                            "dp": ps.dp_rank,
                            "state": "batch",
                            "fingerprints": [batch.pop("_ulysses_test_fingerprints") for batch in batches],
                        }
                    except StopIteration:
                        local = {"dp": ps.dp_rank, "state": "eof"}
                    except Exception as exc:
                        local = {"dp": ps.dp_rank, "state": "error", "error": repr(exc)}
                    peers = gather(local)
                    errors = [peer for peer in peers if peer["state"] == "error"]
                    if errors:
                        raise AssertionError(f"Ulysses consistency test failed: {errors}")
                    for dp_rank in range(ps.dp_size):
                        group = [peer for peer in peers if peer["dp"] == dp_rank]
                        if len(group) != sp_size:
                            raise AssertionError(f"Expected {sp_size} Ulysses peers, got {len(group)}")
                        if any(peer != group[0] for peer in group):
                            raise AssertionError(f"Pre-slice batch mismatch at step {steps}: {group}")
                    if any(peer["state"] == "eof" for peer in peers):
                        break
                    train_loader.mark_batch_consumed()
                    steps += 1
                    micro_batches_checked += len(batches)
                    if dist.get_rank() == 0:
                        print(f"Ulysses consistent: step={steps}, micro_batches={micro_batches_checked}", flush=True)
            finally:
                train_loader.close()
            reports = gather(
                {
                    "rank": dist.get_rank(),
                    "steps": steps,
                    "micro_batches": micro_batches_checked,
                    "matched_chunks": audit.samples,
                    "empty_samples": audit.empty_samples,
                    "packed_modalities": audit.modalities,
                    "error": audit.error,
                }
            )
            if any(report["error"] is not None for report in reports):
                raise AssertionError(f"Ulysses prefetch failed: {reports}")
            if any(report["micro_batches"] == 0 for report in reports):
                raise AssertionError(f"No complete micro-batch was checked: {reports}")
            if dist.get_rank() == 0:
                print("ULYSSES_CONSISTENCY_PASS " + json.dumps(reports, sort_keys=True), flush=True)
            return reports
    finally:
        try:
            if train_loader is not None:
                train_loader.close()
        finally:
            if audit_group is not None:
                dist.destroy_process_group(audit_group)
            if owns_group:
                dist.destroy_process_group()
                from veomni.distributed.parallel_state import clear_parallel_state

                clear_parallel_state()


if __name__ == "__main__":
    test_ulysess()
