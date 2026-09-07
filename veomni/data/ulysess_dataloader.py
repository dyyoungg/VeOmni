
import os
import json
import time
import random
from typing import Optional, List, Dict, Any, Callable, Tuple, Iterator
import types
import traceback
import heapq
import queue
import threading
import datetime
import pickle
import gc


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
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.utils.logging import get_logger
from veomni.utils.helper import read_data
from veomni.utils.constants import get_image_video_audio_placeholder

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
                    preprocess_workers=getattr(self.data_args, "preprocess_workers", 2),)
            self.longvideo_processor.init_image_processor()


    
    def __call__(self, sample_data: Dict[str, Any], sample_idx) -> Optional[Dict[str, Any]]:
        self.init_image_processor()
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
        g.manual_seed(self.epoch + 42)
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
 
    @staticmethod
    def _iterate_with_lookahead(iterable):
        """Yield (item, is_last) pairs; is_last=True for the final element."""
        it = iter(iterable)
        try:
            prev = next(it)
        except StopIteration:
            return
        for item in it:
            yield prev, False
            prev = item
        yield prev, True
 
 
    def __iter__(self):
       
        self._init_processor()
 
        data = self._get_shuffled_data()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id   = worker_info.id          if worker_info else 0

        total_shards    = num_workers
        current_shard   = worker_id

        # Resume support: skip already-consumed samples
        data = data[self.skip_samples_count :]
        global_idx    = self.skip_samples_count
        local_counter = self.skip_samples_count
        self.skip_samples_count = 0  # reset so next epoch starts fresh
 
        for i, item in enumerate(data):
            try:
                if global_idx % total_shards == current_shard:

                    try:
                        # offset 模式：item 是 offset 数组的 global index，按需读取
                        sample_data = self._read_sample_by_offset(item) if self.use_offset else item
                        result = self._processor(sample_data, sample_idx=local_counter)
                        if result is not None:
                            if isinstance(result, (list, types.GeneratorType)):
                                sub_idx = 0
                                for sub_item, is_last in self._iterate_with_lookahead(result):
                                    yield (global_idx, sub_idx, is_last, sub_item)
                                    sub_idx += 1
                                result = None
                            else:
                                yield (global_idx, 0, True, result)
                                result = None
                        else:
                            yield (global_idx, 0, True, None)
                            result = None
                        # print(f"rank {self.rank}, sample idx: {global_idx}")
                        local_counter += 1
                        if local_counter % 50 == 0:
                            time.sleep(0.001)  # yield GIL briefly
                            gc.collect()
    
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning(
                            f"[DP Rank {self.dp_rank} Worker {worker_id}] "
                            f"Processing failed at idx {i}: {e}"
                        )
                        yield (global_idx, 0, True, None)
                        local_counter += 1
    
                global_idx += 1
 
            except Exception as e:
                logger.error(f"Outer loop error at sample {i}: {e}")
                continue


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
            yield data
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
 
        return out
 
    def __iter__(self):
        for sample in self.source:
            if sample is None:
                continue
 
            # Normalise attention_mask_len to a plain int
            attn_len_raw = sample.attention_mask_len
            if attn_len_raw is None:
                attn_len_raw = sample.input_ids
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
    """
    Background-thread loader that continuously:
      1. Iterates UlyssesStreamingDataset (via ReorderingDataLoader)
      2. Syncs samples across SP ranks using CPU Gloo all_gather
      3. Packs samples with MultimodalPacker
      4. Colates batches with OmniDataSharderCollator
      5. Puts finished batches in a queue for the main training loop
 
    The loader is infinite: after each epoch it increments self.epoch and
    starts over, so the training loop never sees StopIteration.
    """
 
    def __init__(
        self,
        dataset,             # ReorderingDataLoader wrapping a DataLoader
        tokenizer,
        model_args,
        max_seq_len: int,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        prefetch_batches: int = 2,
        start_epoch: int = 0,
        num_train_epochs: int = 1,
    ):
        self.dataset        = dataset
        self.tokenizer      = tokenizer
        self.model_args     = model_args
        self.max_seq_len    = max_seq_len
        self.batch_size     = batch_size
        self.collate_fn     = collate_fn
        self.prefetch_batches = prefetch_batches
        self.epoch          = start_epoch
        self.num_train_epochs = num_train_epochs
        self.samples_consumed = 0
 
        self.is_launched    = False
        self.queue: Optional[queue.Queue] = None
        self.stop_event     = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None
 
        self.rank    = dist.get_rank()    if dist.is_initialized() else int(os.getenv("RANK", 0))
        ps           = get_parallel_state()
        self.dp_rank = ps.dp_rank if ps is not None else self.rank
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def launch(self):
        if self.is_launched:
            return
        if self.samples_consumed > 0:
            self._propagate(self.dataset, "set_consumed_samples", self.samples_consumed)
 
        # logger.info(f"[Rank {self.rank}] Launching PrefetchingPackedLoader (epoch {self.epoch})…")
        self.queue = queue.Queue(maxsize=self.prefetch_batches)
        self.stop_event.clear()
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)
        self.producer_thread.start()
        self.is_launched = True
 
    def close(self):
        self.stop_event.set()
        time.sleep(1)
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        if self.producer_thread is not None:
            self.producer_thread.join(timeout=5.0)
            if self.producer_thread.is_alive():
                logger.warning(f"[Rank {self.rank}] Producer thread did not exit cleanly.")
        self.producer_thread = None
        self.is_launched = False
 
    @property
    def raw_samples_consumed(self) -> int:
        return self.samples_consumed
 
    @property
    def data_list(self) -> List:
        """Walk the dataset chain to find the underlying OmniStreamingDataset."""
        obj = self.dataset
        while hasattr(obj, "dataset"):
            obj = obj.dataset
        return getattr(obj, "data_list", [])
 
    # ── Helpers ───────────────────────────────────────────────────────────────
 
    def _propagate(self, obj, method: str, value):
        """Recursively call `method(value)` on every layer of the dataset chain."""
        if hasattr(obj, method):
            getattr(obj, method)(value)
        if hasattr(obj, "dataset"):
            self._propagate(obj.dataset, method, value)
 
    # ── CPU consistency sync ──────────────────────────────────────────────────
 
    def _sample_fingerprint(self, item: Optional[Dict]) -> List[int]:
        """
        Compute a 3-element integer fingerprint [input_len, modal_hash, is_valid]
        used to detect rank divergence before a sample is processed.
        """
        if item is None:
            return [-1, -1, -1]
 
        input_len = len(item.input_ids)
        h = 0
 
        def _shape_sum(key):
            val = getattr(item, key)
            if val is None:
                return 0
            if isinstance(val, torch.Tensor):
                return sum(val.shape)
            if isinstance(val, (list, tuple)):
                return sum(sum(t.shape) for t in val if isinstance(t, torch.Tensor))
            return 0
 
        def _val_sum(key):
            val = getattr(item, key)
            if val is None:
                return 0
            if isinstance(val, torch.Tensor):
                return int(val.float().sum().item())
            if isinstance(val, (list, tuple)):
                return sum(int(t.float().sum().item()) for t in val if isinstance(t, torch.Tensor))
            return 0
 
        h += _shape_sum("pixel_values")
        h += _val_sum("image_grid_thw")
        h += _shape_sum("pixel_values_video")
        h += _val_sum("video_grid_thw")
        h += _shape_sum("audio_features")
        h += _val_sum("audio_features_lens")
 
        return [input_len, h, 1]
 
    def _process_sync_buffer(
        self, buffer: List[Optional[Dict]], cpu_group
    ) -> Iterator[Dict]:
        """
        all_gather fingerprints for every item in buffer, then yield only
        those where all SP ranks agree on (input_len, modal_hash, is_valid).
        """
        meta = [self._sample_fingerprint(item) for item in buffer]
        device     = "cpu"
        local_t    = torch.tensor(meta, dtype=torch.long, device=device)  # [B, 3]
        world_size = dist.get_world_size(group=cpu_group)
        gathered   = [torch.zeros_like(local_t) for _ in range(world_size)]
 
        try:
            if not self.stop_event.is_set():
                dist.all_gather(gathered, local_t, group=cpu_group)
            else:
                return
        except Exception as e:
            logger.error(f"[Rank {self.rank}] CPU Gloo all_gather failed: {e}")
            return
 
        dropped = 0
        for i, item in enumerate(buffer):
            metas = [g[i] for g in gathered]
            ref_len, ref_fp, ref_valid = metas[0].tolist()
 
            ok = ref_valid == 1
            if ok:
                for m in metas[1:]:
                    ml, mf, mv = m.tolist()
                    if ml != ref_len or mf != ref_fp or mv != ref_valid:
                        ok = False
                        break
 
            if ok and item is not None:
                yield item
            else:
                dropped += 1
 
        if dropped:
            logger.debug(
                f"[Rank {self.rank}] SP sync dropped {dropped}/{len(buffer)} samples."
            )
 
    def _create_cpu_synced_iterator(
        self, raw_iterator: Iterator, chunk_size: int = 1
    ) -> Iterator[Dict]:
        """
        Wrap raw_iterator with SP-consistency gating.
 
        When sp_data_group is available, samples are processed in chunks of
        `chunk_size` and fingerprints are compared across SP ranks.  Ranks
        that fail to produce matching data (e.g., missing modality due to a
        corrupt file) have those samples dropped uniformly so every SP rank
        always sees the exact same sequence of packed samples.
        """
        cpu_group = None
        sp_size   = 1
        try:
            ps = get_parallel_state()
            if ps is not None and ps.sp_size > 1:
                cpu_group = get_ulysses_sequence_parallel_cpu_group()
                sp_rank   = ps.ulysses_rank
                sp_size   = ps.ulysses_size
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not retrieve sp_data_group: {e}")
 
        if cpu_group is None or sp_size <= 1:
            logger.error(
                f"[Rank {self.rank}] CRITICAL WARNING: SP CPU Gloo group is None! "
                "Data divergence across SP ranks will NOT be caught!"
            )
            for item in raw_iterator:
                if item is not None:
                    yield item
            return
     
        buffer: List = []
        for item in raw_iterator:
            buffer.append(item)
            if len(buffer) >= chunk_size:
                yield from self._process_sync_buffer(buffer, cpu_group)
                buffer = []
 
        if buffer:
            yield from self._process_sync_buffer(buffer, cpu_group)
 
    # ── Producer thread ───────────────────────────────────────────────────────
 
    def _producer(self):
        # Make sure CUDA device is set correctly inside the thread
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
 
        try:
            while not self.stop_event.is_set():
                # Check epoch limit
                if self.epoch >= self.num_train_epochs:
                    logger.info(
                        f"[Rank {self.rank}] All {self.num_train_epochs} epoch(s) finished. "
                        f"Producer stopping."
                    )
                    break

                self._propagate(self.dataset, "set_epoch", self.epoch)
 
                raw_iter    = iter(self.dataset)
                synced_iter = self._create_cpu_synced_iterator(raw_iter)
                packer      = MultimodalPacker(
                    synced_iter, self.tokenizer, self.model_args, self.max_seq_len
                )
 
                batch_buf: List = []
                for packed in packer:
                    if self.stop_event.is_set():
                        return
 
                    batch_buf.append(packed)
                    del packed  
                    if len(batch_buf) == self.batch_size:
                        self._emit(batch_buf)
                        batch_buf.clear()
 
                if batch_buf and not self.stop_event.is_set():
                    self._emit(batch_buf)
                    batch_buf.clear()
 
                logger.info(
                    f"[Rank {self.rank}] Epoch {self.epoch} finished "
                    f"({self.epoch + 1}/{self.num_train_epochs})."
                )
                self.epoch += 1
 
        except Exception as e:
            if not self.stop_event.is_set():
                logger.error(f"[Rank {self.rank}] Producer error: {e}\n{traceback.format_exc()}")
                self.queue.put(e)
        finally:
            # Signal consumer to stop, whether we finished all epochs or were stopped
            self.queue.put(None)
 
    def _emit(self, batch_buf: List):
        """Collate and enqueue a batch; propagate exceptions to consumer."""
        try:
            if self.collate_fn is not None:
                batch = self.collate_fn(batch_buf)
            else:
                batch = batch_buf
            while not self.stop_event.is_set():
                try:
                    self.queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Collation error: {e}\n{traceback.format_exc()}")
            while not self.stop_event.is_set():
                try:
                    self.queue.put(e, timeout=0.1)
                    break
                except queue.Full:
                    pass
 
    # ── Consumer (main thread) ────────────────────────────────────────────────
 
    def __iter__(self):
        if not self.is_launched:
            self.launch()
        return self._consumer()
 
    def _consumer(self):
        while True:
            try:
                item = self.queue.get(timeout=600)
            except queue.Empty:
                alive = self.producer_thread is not None and self.producer_thread.is_alive()
                raise TimeoutError(
                    f"[Rank {self.rank}] DataLoader queue timeout. "
                    f"Producer {'alive' if alive else 'DEAD'}."
                )
 
            if isinstance(item, Exception):
                raise item
 
            if item is None:
                break

            # Count consumed raw samples for resume
            if isinstance(item, dict):
                if "seq_lens" in item:
                    self.samples_consumed += item["seq_lens"].numel()
                elif "input_ids" in item:
                    self.samples_consumed += item["input_ids"].size(0)
            elif isinstance(item, list):
                for s in item:
                    if isinstance(s, dict) and "sample_lens" in s:
                        self.samples_consumed += len(s["sample_lens"])
                    else:
                        self.samples_consumed += 1
 
            yield item
            item = None
            self.queue.task_done()
 


def make_ulysses_train_dataloader(data_args, training_args, model_args, tokenizer):
    """
    Build the complete Ulysses SP training dataloader.
 
    Args:
        model_args      : model configuration namespace
        data_args       : data configuration namespace
        training_args   : HuggingFace TrainingArguments (or compatible)
        tokenizer       : pre-built tokenizer (already configured)
 
    Returns:
        PrefetchingPackedLoader – finite iterator yielding collated batches
                                  for num_train_epochs epochs
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
        prefetch_factor=getattr(training_args, "dataloader_prefetch_factor", 2),
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

def test_ulysess():
    from veomni.trainer.llava_trainer import VeOmniVLMArguments
    from transformers import AutoTokenizer
    from veomni.arguments import parse_args
    args = parse_args(VeOmniVLMArguments)
    model_args, data_args, training_args = args.model, args.data, args.train

    # Must set CUDA device and init dist BEFORE init_parallel_state
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path,
                                              model_max_length=training_args.model_max_length)
    init_parallel_state(ulysses_size=2)

    rank = dist.get_rank()
    dp_rank = get_parallel_state().dp_rank
    print(f"RANK: {rank}, dp rank: {dp_rank}")
    device = torch.device("cuda", local_rank)

    train_loader = make_ulysses_train_dataloader(data_args, training_args, model_args, tokenizer)

    for i, data in enumerate(train_loader):
        sp_group = get_parallel_state().ulysses_group
        sp_rank = get_parallel_state().sp_rank
        device_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        # Check seq_lens (NOT sp-sliced) for data consistency across SP ranks
        seq_lens = device_data["seq_lens"]
        seq_consist = check_sp_consistency(seq_lens, sp_group, "seq_lens", step=i)

        # Check full input_ids by all_gather-ing the SP chunks back together
        input_ids = device_data["input_ids"].squeeze(0)  # [T_pad // sp_size]
        sp_size = get_parallel_state().sp_size
        gathered = [torch.zeros_like(input_ids) for _ in range(sp_size)]
        dist.all_gather(gathered, input_ids, group=sp_group)
        full_ids = torch.cat(gathered, dim=0)
        id_consist = check_sp_consistency(full_ids, sp_group, "full_input_ids", step=i)

        status = "✅" if (seq_consist and id_consist) else "❌"
        msg = (f"{status} RANK: {rank} DP_Rank {dp_rank} SP_Rank {sp_rank} | "
               f"Shape: {device_data['input_ids'].shape} | "
               f"seq_lens: {seq_lens.tolist()} | "
               f"Consumed: {train_loader.samples_consumed}")
        print(msg)
        del device_data, full_ids, gathered




if __name__ == "__main__":
    test_ulysess()
    