import copy
import json
import os
import psutil
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.multiprocessing import Lock, Manager
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from typing import Iterator

try:
    from aoss_client.client import Client as CephClient
except ImportError:
    from petrel_client.client import Client as CephClient

from veomni.data.base_dataloader import BaseDataLoader
from veomni.utils.constants import (
    IGNORE_INDEX,
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
    _CHAT_TEMPLATES,
    REMOTE_SERVER_PORT
)
from veomni.data.multimodal.image_utils import  tokenizer_audio_token
from veomni.data.llavaomni_processor import OmniSampleProcessor, OmniSample, LongVideoProcessor
from veomni.utils.constants import get_image_video_audio_placeholder
from veomni.utils import helper

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
logger = helper.create_logger(__name__)


def set_env_cpu_limit(cpu_num: int = 1) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(cpu_num)

def create_packed_causal_mask_4D(
    seq_lengths: list[int], 
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:

    max_len = sum(seq_lengths)
    lengths_tensor = torch.tensor(seq_lengths, device=device, dtype=torch.long)
    sequence_ids = torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device), 
        lengths_tensor
    )
    ids_col = sequence_ids.unsqueeze(1)
    ids_row = sequence_ids.unsqueeze(0)
    mask_block_diag = (ids_col == ids_row)
    mask_causal = torch.tril(
        torch.ones(max_len, max_len, dtype=torch.bool, device=device)
    )
    
    mask_2d = mask_block_diag & mask_causal

    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
    
    return mask_4d


def Qwen25VLcollatorFunc(batch_data, tokenizer):
    batch_data = [x for x in batch_data if x is not None]
    if len(batch_data) == 0:
        # Return a dummy batch with a single padding token instead of raising.
        # This keeps all ranks in sync during distributed eval – if one rank's
        # sample fails while others succeed, a raise here causes that rank's
        # DataLoader worker to die, leading to NCCL timeout / CUDA errors.
        dummy_id = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
        return {
            "input_ids": dummy_id,
            "labels": torch.tensor([[IGNORE_INDEX]], dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "audio_features": None,
            "audio_features_lens": None,
            "image_downsample_ratios": None,
            "video_downsample_ratios": None,
            "category": ["dummy_skip"],
            "raw_data": [{}],
            "options_num": torch.tensor([0], dtype=torch.long),
        }

    # 1. 基础文本 Token 与 Label 的 Padding 和截断
    input_ids = [instance["input_ids"] for instance in batch_data]
    labels = [instance["labels"] for instance in batch_data]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )[:, :tokenizer.model_max_length]
    
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX # 请确保 IGNORE_INDEX 在外部已定义
    )[:, :tokenizer.model_max_length]

    multimodal_keys = [
        "pixel_values", "image_grid_thw",
        "pixel_values_video", "video_grid_thw",
        "audio_features", "audio_features_lens",
        "image_downsample_ratios", "video_downsample_ratios"
    ]
    collected_tensors = {k: [] for k in multimodal_keys}
    
    attention_mask = []
    attention_mask_4d = []
    seq_lens_list = [] # 用于 bs=1 的 Flash Attention

    for instance in batch_data:
        for k in multimodal_keys:
            if instance.get(k) is not None:
                collected_tensors[k].append(instance[k])
        
      
        attn_len = instance["attention_mask_len"]
        seq_lens_list.append(torch.tensor(attn_len))
    
        total_len = sum(attn_len) 
        attention_mask.append(torch.ones(total_len, dtype=torch.long))
        # attention_mask_4d.append(create_packed_causal_mask_4D(attn_len))

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )[:, :tokenizer.model_max_length]

    # batch_size = len(attention_mask_4d)
    # max_len = attention_mask.shape[1]
    # final_attention_mask_4d = torch.zeros(
    #     batch_size, 1, max_len, max_len, dtype=torch.bool
    # )

    # for i, mask_4d in enumerate(attention_mask_4d):
    #     L_i = min(mask_4d.shape[2], max_len)
    #     final_attention_mask_4d[i, 0, :L_i, :L_i] = mask_4d[0, 0, :L_i, :L_i]

    def safe_cat(t_list):
        return torch.cat(t_list, dim=0) if len(t_list) > 0 else None

    batch_inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        # "attention_mask_4d": final_attention_mask_4d, # 传seq len 代替，
        "pixel_values": safe_cat(collected_tensors["pixel_values"]),
        "image_grid_thw": safe_cat(collected_tensors["image_grid_thw"]),
        "pixel_values_videos": safe_cat(collected_tensors["pixel_values_video"]),
        "video_grid_thw": safe_cat(collected_tensors["video_grid_thw"]),
        "audio_features": safe_cat(collected_tensors["audio_features"]),
        "audio_features_lens": safe_cat(collected_tensors["audio_features_lens"]),
        "image_downsample_ratios": safe_cat(collected_tensors["image_downsample_ratios"]),
        "video_downsample_ratios": safe_cat(collected_tensors["video_downsample_ratios"]),
    }

    # packing
    if len(batch_data) == 1:
        attn_len = batch_data[0]["attention_mask_len"]
        lengths = torch.tensor(attn_len, dtype=torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            lengths.cumsum(dim=0).to(torch.int32)
        ])
        max_seqlen = int(lengths.max().item())

        batch_inputs["cu_seq_lens_q"] = cu_seqlens
        batch_inputs["cu_seq_lens_k"] = cu_seqlens
        batch_inputs["max_length_q"] = max_seqlen  # int，不是 tensor
        batch_inputs["max_length_k"] = max_seqlen
    if len(seq_lens_list) == 1:
        batch_inputs["seq_lens"] = safe_cat(seq_lens_list).to(torch.int32)

    metadata_configs = [
        ("audio_ground_truth_text", ("", "zh")),
        ("category", None),
        ("raw_data", {})
    ]
    
    for key, default in metadata_configs:
       
        if any(key in d for d in batch_data): 
            batch_inputs[key] = [d.get(key, default) for d in batch_data]

    if any("options_num" in d for d in batch_data):
        options = [d["options_num"] for d in batch_data if "options_num" in d]
        batch_inputs["options_num"] = torch.cat(options, dim=0)

    return batch_inputs
    

class OmniDataloader(BaseDataLoader):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        training_args,
        model_args,
        eval_mode: bool,
    ):
        super().__init__(tokenizer, data_args, training_args, model_args, eval_mode)
     
        self.ceph_client = CephClient(AOSS_FILE)
        try:
            self.bos_client = BosClient(global_config)
        except Exception:
            self.bos_client = None
        self.processor: Optional[OmniSampleProcessor] = None
        self.image_token_id, self.video_token_id, self.audio_token_id  = get_image_video_audio_placeholder(tokenizer)
        self._trained_index = 0

    def init_image_processor(self) -> None:

        self.processor = OmniSampleProcessor(
            tokenizer=self.tokenizer,
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            ceph_client=self.ceph_client,
            bos_client=self.bos_client,
            rank=self.rank,
            build_inputs_token_fn=self.build_inputs_token,
            preprocess_workers=getattr(self.data_args, "preprocess_workers", 2),
        )
        self.processor.init_image_processor()
        
        # 实例化长视频字幕的 Processor
        self.long_video_processor = LongVideoProcessor(
            tokenizer=self.tokenizer,
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            ceph_client=self.ceph_client,
            bos_client=self.bos_client,
            rank=self.rank,
            build_inputs_token_fn=self.build_inputs_token,
            preprocess_workers=getattr(self.data_args, "preprocess_workers", 2),
        )
        # 共享底层的 vision_processor 和 threadpool 防止显存/内存 OOM
        self.long_video_processor.image_processor = self.processor.image_processor
        self.long_video_processor.image_config = getattr(self.processor, "image_config", None)
        self.long_video_processor._threadpool = self.processor._threadpool
    
    def state_dict(self) -> Dict:
 
        total = len(self.data_list)
        remaining = self.data_queue.qsize() if hasattr(self, "data_queue") else 0
        trained_index = total - remaining
   
        return {
            "data_path":     self.data_args.train_path,
            "trained_index": max(trained_index, 0),
            "remote_data_index": self.remote_data_index.value
        }

    def load_state_dict(self, state: Dict) -> None:
        saved_path   = state.get("data_path", "")
        resume_index = state.get("trained_index", 0)
        remote_index = state.get("remote_data_index", 0)
        self.remote_data_index.value = remote_index

        if not self.training_args.remote_dataloader:
            if saved_path and saved_path != self.data_args.train_path:
                logger.warning(
                    f"[Dataloader] data_path mismatch: "
                    f"checkpoint='{saved_path}' vs current='{self.data_args.train_path}'. "
                    f"Starting from scratch."
                )
                return

            if resume_index <= 0:
                return

            total = len(self.data_list)
            if resume_index >= total:
                logger.warning(
                    f"[Dataloader] trained_index={resume_index} >= total={total}, "
                    f"data exhausted, starting from scratch."
                )
                return

            self.data_list = self.data_list[resume_index:]
            logger.info(
                f"[Dataloader] Resumed: skipped {resume_index}/{total}, "
                f"{len(self.data_list)} samples remaining."
            )

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

    def launch(self) -> None:
        self.data_lock = Lock()
        self.image_merge_sizes: List = []
        self._clear_pack_buffer()
        self.worker_metrics_queue = torch.multiprocessing.Queue()
        if getattr(self.data_args, "save_token_counted_data", False):
            self.save_info_queue = torch.multiprocessing.Queue()
        super().launch()

        if getattr(self.data_args, "save_token_counted_data", False):
            self.save_info_path = os.path.join(
                self.data_args.tokencounted_data_save_dir,
                os.path.basename(self.data_args.train_path),
            )
            os.makedirs(self.data_args.tokencounted_data_save_dir, exist_ok=True)
            self.save_thread = threading.Thread(target=self._save_info_worker, daemon=True)
            self.save_thread.start()
        print(f"{self.rank} successful launch dataloader")
    

    def _save_info_worker(self):
        with open(self.save_info_path, 'w', encoding='utf-8') as f:
            while not self.end_signal:
                try:
                    item = self.save_info_queue.get(timeout=2.0) 
                except queue.Empty:
                
                    continue
            
                if item is None: 
                    break
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()


    def _generate_fake_sample(self) -> Tuple:
        """Generate a single fake multimodal sample with random modality combination.

        Randomly selects which modalities to include (image/video/audio/text-only)
        with random sizes, producing an OmniSample-like tuple that can be fed into
        _pack_and_enqueue for realistic packing behavior.

        Returns:
            (input_ids, labels, caption_len, resources, token_counts)
        """
        import math
        import random
        from veomni.data.multimodal.image_utils import get_adaptive_pool_size

        max_len = self.tokenizer.model_max_length
        downsample_ratio = self.model_args.mm_downsample_ratio
        merge_size = 2
        vocab_size = len(self.tokenizer)

        # Randomly pick modality combination
        # Weights: text-only 10%, image 25%, video 25%, audio 10%,
        #          image+audio 10%, video+audio 10%, all 10%
        combo = random.choices(
            ["text", "image", "video", "audio",
             "image+audio", "video+audio", "all"],
            weights=[10, 25, 25, 10, 10, 10, 10],
            k=1,
        )[0]

        has_image = combo in ("image", "image+audio", "all")
        has_video = combo in ("video", "video+audio", "all")
        has_audio = combo in ("audio", "image+audio", "video+audio", "all")

        pixel_values = None
        image_grid_thw = None
        image_downsample_ratios = None
        num_image_tokens = 0

        pixel_values_video = None
        video_grid_thw = None
        video_downsample_ratios = None
        num_video_tokens = 0

        audio_features = None
        audio_features_lens = None
        actual_audio_feature_len = []
        num_audio_tokens = 0

        # --- Image ---
        if has_image:
            num_images = random.randint(1, 4)
            all_pixels, all_thw = [], []
            for _ in range(num_images):
                # Random spatial: h,w in {16,20,24,28,32} (multiples of 2 for merge)
                img_h = random.choice([16, 20, 24, 28, 32, 48])
                img_w = random.choice([16, 20, 24, 28, 32, 48])
                img_t = 1
                n_patches = img_t * img_h * img_w
                all_pixels.append(torch.randn(n_patches, 1176, dtype=torch.float32))
                all_thw.append([img_t, img_h, img_w])
                m_h, m_w = get_adaptive_pool_size(img_h // merge_size, img_w // merge_size, downsample_ratio)
                num_image_tokens += img_t * m_h * m_w

            pixel_values = torch.cat(all_pixels, dim=0)
            image_grid_thw = torch.tensor(all_thw, dtype=torch.int64)
            image_downsample_ratios = torch.full((num_images,), downsample_ratio, dtype=torch.float32)

        # --- Video ---
        if has_video:
            # Random temporal: 2~16 temporal patches; random spatial
            vid_t = random.choice([8, 16, 32])
            vid_h = random.choice([12, 16, 20, 24, 48, 96])
            vid_w = random.choice([12, 16, 20, 24, 48, 96])
            vid_num_patches = vid_t * vid_h * vid_w
            pixel_values_video = torch.randn(vid_num_patches, 1176, dtype=torch.float32)
            video_grid_thw = torch.tensor([[vid_t, vid_h, vid_w]], dtype=torch.int64)
            video_downsample_ratios = torch.tensor([downsample_ratio], dtype=torch.float32)
            v_m_h, v_m_w = get_adaptive_pool_size(vid_h // merge_size, vid_w // merge_size, downsample_ratio)
            num_video_tokens = vid_t * v_m_h * v_m_w

        # --- Audio ---
        if has_audio:
            audio_downsample_ratio = self.model_args.audio_downsample_ratio
            audio_frame_length = getattr(self.model_args, "audio_frame_length", 320)
            num_mel_bins = getattr(self.model_args, "num_mel_bins", 128)
            max_chunk_samples = 480000  # 30s at 16kHz
            # Random 1~3 audio chunks (simulating 1~90s audio)
            num_chunks = random.randint(1, 3)
            mel_list, len_list = [], []
            for _ in range(num_chunks):
                # Random duration per chunk: 1s~30s
                dur_samples = random.randint(16000, max_chunk_samples)
                chunk_frames = math.ceil(dur_samples / audio_frame_length)
                mel_list.append(torch.randn(1, num_mel_bins, 3000, dtype=torch.float32))
                len_list.append(chunk_frames)  # whisper CNN downsamples 3000 mel frames; effective len = chunk_frames
                feat_len = math.ceil(chunk_frames / audio_downsample_ratio)
                actual_audio_feature_len.append(feat_len)
                num_audio_tokens += feat_len
            audio_features = torch.cat(mel_list, dim=0)
            audio_features_lens = torch.tensor(len_list, dtype=torch.int64)

        # --- Guard: if total mm tokens too large for max_len, drop some modalities ---
        total_mm_tokens = num_image_tokens + num_video_tokens + num_audio_tokens
        min_text_tokens = 64
        if total_mm_tokens + min_text_tokens > max_len:
            # Too many mm tokens — fall back to text-only for this sample
            has_image = has_video = has_audio = False
            pixel_values = image_grid_thw = image_downsample_ratios = None
            pixel_values_video = video_grid_thw = video_downsample_ratios = None
            audio_features = audio_features_lens = None
            actual_audio_feature_len = []
            num_image_tokens = num_video_tokens = num_audio_tokens = 0
            total_mm_tokens = 0

        # --- Build input_ids ---
        total_mm_tokens = num_image_tokens + num_video_tokens + num_audio_tokens
        # Random text length: 64~1024, capped to fit max_len
        text_budget = max(64, max_len - total_mm_tokens)
        text_len = random.randint(64, min(1024, text_budget))
        total_len = total_mm_tokens + text_len

        input_ids = torch.randint(0, vocab_size, (total_len,), dtype=torch.long)
        labels = torch.randint(0, vocab_size, (total_len,), dtype=torch.long)

        # Fill multimodal pad token positions at the front
        pos = 0
        if has_image:
            input_ids[pos:pos + num_image_tokens] = self.image_token_id
            labels[pos:pos + num_image_tokens] = IGNORE_INDEX
            pos += num_image_tokens
        if has_video:
            input_ids[pos:pos + num_video_tokens] = self.video_token_id
            labels[pos:pos + num_video_tokens] = IGNORE_INDEX
            pos += num_video_tokens
        if has_audio:
            input_ids[pos:pos + num_audio_tokens] = self.audio_token_id
            labels[pos:pos + num_audio_tokens] = IGNORE_INDEX
            pos += num_audio_tokens

        resources = {
            "image_pixels": pixel_values,
            "image_thw": image_grid_thw,
            "video_pixels": pixel_values_video,
            "video_thw": video_grid_thw,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "actual_audio_feature_len": actual_audio_feature_len,
            "image_downsample_ratios": image_downsample_ratios,
            "video_downsample_ratios": video_downsample_ratios,
        }
        token_counts = {
            "image": num_image_tokens,
            "video": num_video_tokens,
            "audio": num_audio_tokens,
        }
        return input_ids, labels, total_len, resources, token_counts

    def _worker_loop_fake(self, status_event) -> None:
        """Worker loop that produces fake multimodal data for speed benchmarking.

        Each call to _generate_fake_sample produces a random modality combination
        with random sizes. Samples go through _pack_and_enqueue for realistic
        sequence packing behavior.

        Workers run indefinitely until the trainer sets workers_done_event
        (triggered by max_steps or num_train_epochs). The counter is for
        logging only.
        """
        set_env_cpu_limit(cpu_num=1)
        torch.set_num_threads(1)
        self._clear_pack_buffer()

        ppid = os.getppid()
        local_count = 0

        while os.getppid() == ppid and not self.workers_done_event.is_set():
            # Back-pressure: wait for consumer to drain before producing more
            if self.result_queue.qsize() > 4:
                time.sleep(0.5)
                continue

            input_ids, labels, caption_len, resources, token_counts = self._generate_fake_sample()
            self._pack_and_enqueue(input_ids, labels, caption_len, resources, token_counts)

            local_count += 1
            if local_count % 500 == 0 and self.rank == 0:
                logger.info(f"[FakeData] worker generated {local_count} samples")

        # Flush remaining packed buffer before exiting
        if len(self.new_input_ids) > 0:
            self._flush_pack_buffer()
            self._clear_pack_buffer()

        status_event.set()
        if self.workers_done_event.is_set() or os.getppid() != ppid:
            self.result_queue.cancel_join_thread()


    def worker_loop(self, status_event) -> None:
        # --- Fake data fast path ---
        if getattr(self.training_args, "use_fake_data", False):
            self._worker_loop_fake(status_event)
            return

        if self.processor is None:
            self.init_image_processor()
        import gc
        gc.enable()
        set_env_cpu_limit(cpu_num=1)
        torch.set_num_threads(1)
        self.tokenizer.add_bos_token = False

        ppid = os.getppid()
        http_addr: Optional[str] = None
        sample_index = 0
        proc = psutil.Process(os.getpid())

        file_cache = {}
        MAX_CACHE_SIZE = 100  
        session = requests.Session()
        def get_file_line(file_path: str, offset: int) -> str:
            if file_path not in file_cache:
                if len(file_cache) >= MAX_CACHE_SIZE:
                    oldest_path = next(iter(file_cache))
                    file_cache[oldest_path].close()
                    del file_cache[oldest_path]
                file_cache[file_path] = open(file_path, "r", encoding="utf-8")
            
            f = file_cache[file_path]
            f.seek(offset)
            return f.readline()

        while os.getppid() == ppid and not self.workers_done_event.is_set():
            if self.result_queue.qsize() > 4:
                time.sleep(1)
                gc.collect()
                continue

            if self.training_args.remote_dataloader:
                if http_addr is None:
                    master_addr = os.environ.get("MASTER_ADDR")
                    
                    if master_addr is None:
                        # single node
                        master_addr = socket.gethostname()
                    
                    http_addr = f"http://{master_addr}:{REMOTE_SERVER_PORT}/ask_data"
                   
                try:
                    response = session.post(http_addr, json={}, timeout=5)
                    data_index = response.json()["index"]
                except Exception as e:
                    # logger.error(f"ask data error: {e!r}")
                    time.sleep(3)
                    continue
                if data_index >= len(self.data_list) - 1:
                    print(f"rank {dist.get_rank()}, data index: {data_index}")
                    if hasattr(self, 'new_input_ids') and len(self.new_input_ids) > 0:
                        self._flush_pack_buffer()
                        self._clear_pack_buffer()
                    
                    status_event.set()
                    time.sleep(10)
                    continue
                file_path = None
                try:
                    file_id, offset = self.data_list[data_index]
                    file_path = self.file_mapping[int(file_id)]
                    line = get_file_line(file_path, int(offset))
                    sample_data = json.loads(line)
                    sample_index += 1
                except Exception as e:
                    logger.error(f"Read Error at index {data_index}: {e}")
                    if file_path is not None and file_path in file_cache:
                        try:
                            file_cache[file_path].close()
                        except:
                            pass
                        del file_cache[file_path]
                    
                    time.sleep(1) 
                    continue
            
            else:
                try:
                    sample_data = json.loads(self.data_queue.get(timeout=5))
                    sample_index += 1
                except queue.Empty:
                    status_event.set()
                    continue

            if isinstance(sample_data, dict):
                self.worker_loop_finetune(sample_data, sample_index)
                
            elif isinstance(sample_data, list):
                # Susbtitle List 格式：[video_path, subtitle_path, (可选的 system_prompt)]
                formatted_data = {
                    "video": sample_data[0],
                    "subtitle_path": sample_data[1]
                }
                if len(sample_data) > 2:
                    formatted_data["system_prompt"] = sample_data[2]
                
                self.worker_loop_finetune(formatted_data, sample_index, is_long_video=True)
            else:
                raise NotImplementedError("Unsupported sample_data format.")
            
            if sample_index % 50 == 0:
                mem_mb = proc.memory_info().rss / 1024**2
                gen0, gen1, gen2 = gc.get_count()
                if self.rank == 0:
                    try:
                        self.worker_metrics_queue.put_nowait({
                            "pid": os.getpid(),
                            "rss_mb": mem_mb,
                            "gc_gen0": gen0,
                            "gc_gen1": gen1,
                            "gc_gen2": gen2,
                        })
                    except Exception:
                        pass
                    logger.info(
                        f"[Worker pid={os.getpid()}] step={sample_index} "
                        f"rss={mem_mb:.0f}MB "
                        f"gc=({gen0},{gen1},{gen2})"
                    )
                gc.collect()
        try:
            for f in file_cache.values():
                f.close()
        except:
            pass

        if self.workers_done_event.is_set() or os.getppid() != ppid:
            self.result_queue.cancel_join_thread()

    def _save_token_counts(self, sample_data: Dict, system_token: List[int], counts: Dict, total_len: int) -> None:
        try:
            save_item = copy.deepcopy(sample_data)
            save_item.update({
                "computed_total_tokens": total_len,
                "computed_image_tokens": counts["image"],
                "computed_video_tokens": counts["video"],
                "computed_audio_tokens": counts["audio"],
                "system_prompt": self.tokenizer.decode(system_token) if hasattr(self, 'tokenizer') else str(system_token),
            })
            self.save_info_queue.put(save_item)
        except Exception as e:
            print(f"Error saving token counted data: {e}")

    def _validate_truncation(self, input_ids: torch.Tensor, counts: Dict) -> bool:
        
        img_ok = sum(input_ids == self.image_token_id).item() == counts.get("image", 0)
        vid_ok = sum(input_ids == self.video_token_id).item() == counts.get("video", 0)
        aud_ok = sum(input_ids == self.audio_token_id).item() == counts.get("audio", 0)
        return img_ok and vid_ok and aud_ok
    
    def _pack_and_enqueue(
        self, cur_input_ids: torch.Tensor, cur_labels: torch.Tensor, 
        cur_caption_len: int, resources: Dict, token_counts: Dict
    ) -> None:
        
        max_len = self.tokenizer.model_max_length
        # 首先检查长度截断
        if len(cur_input_ids) >= max_len:
            cur_input_ids = cur_input_ids[:max_len]
            cur_labels = cur_labels[:max_len]
            if not self._validate_truncation(cur_input_ids, token_counts):
                print(f"WARNING: Skipping sample. Truncation to {max_len} cut off multimodal tokens.")
                return
            cur_caption_len = max_len
      
        if not self.training_args.pack_seq:
            self._enqueue_packed_result(cur_input_ids, cur_labels, [cur_caption_len], resources)
            return
        image_num = 0
        if len(self.new_images_thw):
            image_num = torch.cat(self.new_images_thw)[:,0].sum().item()
        
        if (self.new_caption_len + cur_caption_len > max_len and len(self.new_input_ids) > 0) or image_num > self.training_args.target_image_num:
            # 缓冲区溢出，把当前的缓存打包发出
            self._flush_pack_buffer()
            self._clear_pack_buffer()


        self._append_to_pack_buffer(cur_input_ids, cur_labels, cur_caption_len, resources, token_counts)


    def _flush_pack_buffer(self) -> None:
        packed_ids = torch.cat(self.new_input_ids).clone()
        packed_labels = torch.cat(self.new_labels).clone()
        packed_counts = {
            "image": sum(self.new_image_tokens),
            "video": sum(self.new_video_tokens),
            "audio": sum(self.new_audio_tokens),
        }
        if not self._validate_truncation(packed_ids, packed_counts):
            print("WARNING: Packed batch failed validation. Dropping.")
            return
 
        def _safe_cat(lst):
            return torch.cat(lst, dim=0).clone() if lst else None
 
        packed_resources = {
            "image_thw":            _safe_cat(self.new_images_thw),
            "image_pixels":         _safe_cat(self.new_images_list),
            "video_thw":            _safe_cat(self.new_video_thw),
            "video_pixels":         _safe_cat(self.new_video_list),
            "audio_features_lens":  _safe_cat(self.audio_feature_len_list),
            "audio_features":       _safe_cat(self.audio_feature_list),
            "image_downsample_ratios": _safe_cat(self.new_image_downsample_ratios),
            "video_downsample_ratios": _safe_cat(self.new_video_downsample_ratios),
        }
        self._enqueue_packed_result(
            packed_ids, packed_labels,
            list(self.attention_mask_len),
            packed_resources,
        )

    def _enqueue_packed_result(self, input_ids, labels, attn_mask_len, resources) -> None:
        self.result_queue.put({
            "input_ids":            input_ids,
            "labels":               labels,
            "pixel_values":         resources.get("image_pixels"),
            "image_grid_thw":       resources.get("image_thw"),
            "pixel_values_video":   resources.get("video_pixels"),
            "video_grid_thw":       resources.get("video_thw"),
            "audio_features":       resources.get("audio_features"),
            "audio_features_lens":  resources.get("audio_features_lens"),
            "image_downsample_ratios": resources.get("image_downsample_ratios"),
            "video_downsample_ratios": resources.get("video_downsample_ratios"),
            "attention_mask_len":   attn_mask_len,
        })

    def _append_to_pack_buffer(self, cur_input_ids, cur_labels, cur_caption_len, resources, token_counts):
        self.new_caption_len += cur_caption_len
        self.new_input_ids.append(cur_input_ids)
        self.new_labels.append(cur_labels)
        self.attention_mask_len.append(cur_caption_len)

        if resources.get("image_pixels") is not None and resources.get("image_thw") is not None:
            self.new_images_list.append(resources["image_pixels"])
            self.new_images_thw.append(resources["image_thw"])
            self.new_image_tokens.append(token_counts["image"])
            if resources.get("image_downsample_ratios") is not None:
                self.new_image_downsample_ratios.append(resources["image_downsample_ratios"])

        if resources.get("video_pixels") is not None and resources.get("video_thw") is not None:
            self.new_video_list.append(resources["video_pixels"])
            self.new_video_thw.append(resources["video_thw"])
            self.new_video_tokens.append(token_counts["video"])
            if resources.get("video_downsample_ratios") is not None:
                self.new_video_downsample_ratios.append(resources["video_downsample_ratios"])

        if resources.get("audio_features") is not None and resources.get("audio_features_lens") is not None:
            self.audio_feature_list.append(resources["audio_features"])
            self.audio_feature_len_list.append(resources["audio_features_lens"])
            self.new_audio_tokens.append(sum(resources["actual_audio_feature_len"]))


    def _clear_pack_buffer(self) -> None:
        self.new_caption_len = 0
        self.new_input_ids, self.new_labels = [], []
        self.attention_mask_len = []
        self.new_images_list, self.new_images_thw = [], []
        self.new_video_list, self.new_video_thw = [], []
        self.audio_feature_list, self.audio_feature_len_list = [], []
        self.new_image_tokens, self.new_video_tokens, self.new_audio_tokens = [], [], []
        self.new_image_downsample_ratios = []
        self.new_video_downsample_ratios = []

    
    # ------------------------------------------------------------------
    # Finetune worker: dispatch + batch packing
    # ------------------------------------------------------------------    
    def worker_loop_finetune(self, sample_data: Dict, sample_idx: int, is_long_video: bool = False) -> None:        
        # 根据标记选择处理器
        processor = self.long_video_processor if is_long_video else self.processor
        
        samples = processor.process(sample_data, sample_idx)
        if samples is None:
            return
            
        # 兼容单样本(短数据)和迭代器(长视频数据)两种返回格式
        if isinstance(samples, Iterator):
            try:
                for sample in samples:
                    if sample is None:
                        continue
                    self._process_single_sample(sample_data, sample, processor)
            finally:
                if hasattr(samples, 'close'):
                    samples.close()
        else:
            self._process_single_sample(sample_data, samples, processor)

    def _process_single_sample(self, sample_data: Dict, sample: OmniSample, processor: Any) -> None:
        if getattr(self.data_args, "save_token_counted_data", False):
            system_token = processor._build_system_token(sample_data, 0)
            self._save_token_counts(
                sample_data, system_token, sample.token_counts, sample.caption_len
            )

        self._pack_and_enqueue(
            cur_input_ids=sample.input_ids,
            cur_labels=sample.labels,
            cur_caption_len=sample.caption_len,
            resources={
                "image_pixels": sample.pixel_values,
                "image_thw": sample.image_grid_thw,
                "video_pixels": sample.pixel_values_video,
                "video_thw": sample.video_grid_thw,
                "audio_features": sample.audio_features,
                "audio_features_lens": sample.audio_features_lens,
                "actual_audio_feature_len": sample.actual_audio_feature_len,
                "image_downsample_ratios": sample.image_downsample_ratios,
                "video_downsample_ratios": sample.video_downsample_ratios,
            },
            token_counts=sample.token_counts,
        )
       
        

    def fetch_data_loop(self) -> None:
        device = torch.device(f"cuda:{self.local_rank}")
        while not self.end_signal:
            batch_data = []
            batch_count = 0
            data = None
            while batch_count < self.training_args.per_device_train_batch_size and not self.end_signal:
                while True:  # add loop for gracefully exit
                    try:
                        data = self.result_queue.get(timeout=2)
                        batch_data.append(data)
                        batch_count += 1
                        break
                    except queue.Empty:
                        if self.end_signal:
                            break
                        else:
                            continue
            if self.end_signal:
                break

            batch_inputs = Qwen25VLcollatorFunc(batch_data, self.tokenizer)
            del batch_data
            bf16_keys = ["images", "pixel_values", "pixel_values_videos", "audio_features"]
            for k, v in batch_inputs.items():
                if isinstance(v, torch.Tensor):
                    if k in bf16_keys:
                        batch_inputs[k] = v.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    else:
                        batch_inputs[k] = v.to(device=device, non_blocking=True)
                    del v  
            
            while True: # add loop for gracefully exit
                try:
                    self.batch_data_queue.put(batch_inputs, timeout=1)
                
                    break
                except queue.Full:
                    pass
                if self.end_signal:
                    break
            
            if self.end_signal:
                break
                
    

class Qwen25VLEvaluationDataset(Dataset, OmniDataloader):
    def __init__(
        self,
        tokenizer,
        data_args,
        training_args,
        model_args,
    ):
        super().__init__(
            tokenizer=tokenizer, 
            data_args=data_args, 
            training_args=training_args, 
            model_args=model_args, 
            eval_mode=True
        )
        self.image_merge_size = 2
        self.video_merge_size = 2
        self.data_list = self._load_eval_data(data_args.eval_path)
        self.categories = {d.get('category', 'unknown') for d in self.data_list}
        
        print(f"Eval dataset initialized. Total size: {len(self.data_list)}")
   
  
    def _load_eval_data(self, path: str) -> List[Dict]:
        if path.endswith(".jsonl"):
            with open(path) as f:
                return [json.loads(line) for line in f]
        elif path.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        raise ValueError(f"Unsupported eval data format: {path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.processor is None:
            self.init_image_processor()

        sample_data = self.data_list[index]
        try:
            if "audio" in sample_data and "image" not in sample_data and "video" not in sample_data:
                return self.process_audio_eval(sample_data)
            else:
                return self.process_image_text_eval(sample_data)
        except Exception as e:
            print(f"[Dataset Error] index {index} failed: {e}")
            import traceback
            traceback.print_exc()
            return None 
        
    def process_audio_eval(self, sample_data: Dict) -> Optional[Dict]:

        audio_data_list = self._load_and_pad_audios(sample_data)
        if not audio_data_list:
            return None

        audio_mel, actual_audio_len, raw_audio_len, audio_chunk_counts = self.processor._extract_audio_features(audio_data_list)

        # 2. 构建 Prompt Token
        category = sample_data["category"]
        language = sample_data.get("language", "en")
        text = sample_data.get("text", None)

        if category in ["aishell", "librispeech", "cantonese", "commonvoice_ja"]:
            query_map = {
                "zh": "请转录这段语音的内容。",
                "en": "Please transcribe the audio content.",
                "zh_yue": "请使用粤语转录这段音频内容。",
                "jap": "请使用日语转录这段音频内容。"
            }
            query = query_map.get(language, "Please transcribe the audio content.")
            input_ids = self._build_audio_query_tokens(query)
            label_tokens = [IGNORE_INDEX] * len(input_ids)

        elif category in ["MMAU", "vocalsound"]:
            question = sample_data.get("question", "")
            answer = sample_data.get("answer", "")
            answer_idx = 0
            for idx, c in enumerate(sample_data['candidates']):
                question += f"\n({chr(ord('A') + idx)}) {c}\n" if idx == 0 else f"({chr(ord('A') + idx)}) {c}\n"
                if c == answer: answer_idx = idx

            question_tokens = self._build_audio_query_tokens(question)
            text_no_loss = 'My best option is ('
            text_with_loss = f"{chr(ord('A') + answer_idx)}"
            mask_len = len(self.tokenizer(text_no_loss)['input_ids'])
            answer_tokens = self.tokenizer(text_no_loss + text_with_loss)['input_ids']

            input_ids = question_tokens + answer_tokens
            label_tokens = [IGNORE_INDEX] * len(question_tokens) + [IGNORE_INDEX] * mask_len + answer_tokens[-(len(answer_tokens) - mask_len):]
        else:
            return None

        input_ids, labels, _ = self.processor._expand_multimodal_tokens(
            input_ids, label_tokens,
            audio_feature_len=actual_audio_len,
            audio_chunk_counts=audio_chunk_counts
        )

        return self._build_return_dict(
            input_ids=input_ids,
            labels=labels,
            category=category,
            audio_features=audio_mel,
            audio_features_lens=raw_audio_len,
            text=text,
            language=language,
            options_num=len(sample_data.get("candidates", [])),
            raw_data=sample_data
        )

    def build_inputs_qwen2(self, prompt_list: List[Dict], system_prompt:str="You are a helpful assistant."):
        prompt = ""
     
        if self.model_args.model_arc in ["qwen2",'qwen3']:
            user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            assistant_format = "{content}<|im_end|>" 
        else:
            raise NotImplementedError(f"Unsupported model_arc: {self.model_args.model_arc!r}")

        for message in prompt_list:
            content = message["value"]
            if message['role'].lower() in ['human',"user"]:
                prompt += user_format.format(content=content)
            elif message["role"].lower() in ["assistant", "gpt"]:
                prompt += assistant_format.format(content=content)
            else:
                pass
        return self.tokenizer(prompt)["input_ids"]

   
    def process_image_text_eval(self, sample_data: Dict) -> Optional[Dict]:
        system_prompt = 'You are a helpful assistant.'
        system_token = self.build_inputs_token(system_prompt, input_type='system', return_tensor=False)

        category = sample_data['category']
        question = sample_data['question']
        answer = sample_data['answer']
        

        has_visual = "path" in sample_data or "image_path" in sample_data or 'wukong' in category
        audio_list = []
        
        if has_visual:
            answer_idx = -1
            for idx, c in enumerate(sample_data.get('candidates', [])):
                question += f"\n({chr(ord('A') + idx)}) {c}\n" if idx == 0 else f"({chr(ord('A') + idx)}) {c}\n"
                if c == answer: 
                    answer_idx = idx
            
            if "audio" not in sample_data:
                question_tokens = self.build_inputs_token(question, input_type="query_format", return_tensor=False)
            else:
                question_tokens, audio_list = self._build_mixed_audio_tokens(sample_data, question, category)
                if not question_tokens: 
                    return None
                
            no_loss_txt = 'My best option: ('
            with_loss_txt = f"{chr(ord('A') + answer_idx)}"
            mask_len = len(self.tokenizer(no_loss_txt)['input_ids'])
            answer_tokens = self.tokenizer(no_loss_txt + with_loss_txt)['input_ids']
            
            subtitle_tokens = question_tokens + answer_tokens
            label_tokens = [IGNORE_INDEX] * len(question_tokens) + [IGNORE_INDEX] * mask_len + answer_tokens[-(len(answer_tokens) - mask_len):]
        else:
            # 纯文本
            question_tokens = self.build_inputs_qwen2(question, system_prompt=system_prompt) 
            label_tokens = [IGNORE_INDEX] * len(question_tokens)
            text_with_loss = answer
            answer_tokens = self.tokenizer(text_with_loss)['input_ids']
            subtitle_tokens = question_tokens + answer_tokens
            label_tokens += answer_tokens

     
        audio_mel, actual_audio_len, raw_audio_len, audio_chunk_counts = self.processor._extract_audio_features(audio_list)

        max_image_tokens = self.tokenizer.model_max_length - len(subtitle_tokens) - len(system_token) - sum(actual_audio_len)
        pixels, thw, visual_mode = self._load_visual_features(sample_data, max_image_tokens)

        if visual_mode == "error":
            return None

        # 添加 Vision 前缀 Token
        if thw is not None:
            image_num = thw[0][0].item() if visual_mode == "video" else len(thw)
            subtitle_tokens, label_tokens = self.processor.make_image_subtitle_label_tokens(
                image_num, subtitle_tokens, label_tokens, mode=visual_mode
            )

        cur_input_ids = system_token + subtitle_tokens
        cur_labels = [IGNORE_INDEX] * len(system_token) + label_tokens

        downsample_ratio = self.model_args.mm_downsample_ratio
        input_ids, labels, _  = self.processor._expand_multimodal_tokens(
            cur_input_ids, cur_labels,
            image_thw=thw if visual_mode == "image" else None,
            video_thw=thw if visual_mode == "video" else None,
            audio_feature_len=actual_audio_len,
            audio_chunk_counts=audio_chunk_counts,
            downsample_ratio=downsample_ratio,
        )

        # Skip samples that exceed model_max_length after token expansion,
        # otherwise the collator truncates input_ids causing image/video token
        # count to mismatch projector output (CUDA OOB) and answer labels to
        # be cut off (target_id becomes IGNORE_INDEX=-100).
        if len(input_ids) > self.tokenizer.model_max_length:
            n_images = len(thw) if thw is not None and visual_mode == "image" else 0
            n_video_frames = thw[0][0].item() if thw is not None and visual_mode == "video" else 0
            print(f"[Eval Warning] Sample exceeds model_max_length after expansion "
                  f"({len(input_ids)} > {self.tokenizer.model_max_length}), skipping. "
                  f"category={category}, visual_mode={visual_mode}, n_images={n_images}, "
                  f"n_video_frames={n_video_frames}, "
                  f"image_path={sample_data.get('image_path', 'N/A')}, "
                  f"video_path={sample_data.get('path', 'N/A')}")
            return None

        # Build per-grid downsample ratio tensors (same as training path)
        image_downsample_ratios = None
        video_downsample_ratios = None
        if thw is not None:
            if visual_mode == "image":
                image_downsample_ratios = torch.full((thw.shape[0],), downsample_ratio, dtype=torch.float32)
            elif visual_mode == "video":
                video_downsample_ratios = torch.full((thw.shape[0],), downsample_ratio, dtype=torch.float32)

        return self._build_return_dict(
            input_ids=input_ids,
            labels=labels,
            category=category,
            pixel_values=pixels if visual_mode == "image" else None,
            image_grid_thw=thw if visual_mode == "image" else None,
            pixel_values_video=pixels if visual_mode == "video" else None,
            video_grid_thw=thw if visual_mode == "video" else None,
            audio_features=audio_mel,
            audio_features_lens=raw_audio_len,
            image_downsample_ratios=image_downsample_ratios,
            video_downsample_ratios=video_downsample_ratios,
            options_num=len(sample_data.get("candidates", [])),
            raw_data=sample_data
        )

    # ==========================================
    # 可复用的辅助方法拆分
    # ==========================================
    def _build_audio_query_tokens(self, query: str) -> List[int]:
        if self.model_args.use_audio_start_end_token:
            prompt = f"<|im_start|>user\n{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}{query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n<audio>{query}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer_audio_token(prompt=prompt, tokenizer=self.tokenizer, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None)

    def _build_mixed_audio_tokens(self, sample_data: Dict, question: str, category: str) -> Tuple[List[int], List]:
        audio_str = f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}" if self.model_args.use_audio_start_end_token else DEFAULT_AUDIO_TOKEN
        
        if category == "mvbench_audio":
            cur_text = f"\n<|im_start|>user\n{audio_str}<|im_end|>\n<|im_start|>assistant\n"
        else:
            cur_text = f"\n<|im_start|>user\n{audio_str}{question}<|im_end|>\n<|im_start|>assistant\n"
            
        question_tokens = []
        audio_chunks = cur_text.split(DEFAULT_AUDIO_TOKEN)
        for i, chunk in enumerate(audio_chunks):
            question_tokens.extend(self.tokenizer(chunk)["input_ids"])
            if i < len(audio_chunks) - 1:
                question_tokens.append(AUDIO_TOKEN_INDEX)
                
        audio_list = self._load_and_pad_audios(sample_data)
        return question_tokens, audio_list

    def _load_and_pad_audios(self, sample_data: Dict) -> List[torch.Tensor]:
        audio_field = sample_data.get("audio", [])
        paths = [audio_field] if isinstance(audio_field, str) else audio_field
        res = []
        for p in paths:
           
            t, _ = self.processor.get_audio_from_local_or_remote(p)
            if t is None: 
                return []
            res.append(t)
        return res


    def _load_visual_features(self, sample_data: Dict, max_tokens: int) -> Tuple[Any, Any, str]:
        """统一视频与图像的特征加载流，复用基类方法"""
        video_path = sample_data.get("path")
        image_path = sample_data.get("image_path")
        selected_downsample_ratio = self.model_args.mm_downsample_ratio
        if video_path:
            video_file = self.processor.get_video_path({"video": video_path})
            if not video_file: 
                return None, None, "error"
            
            vformat = "gif" if video_path.endswith(".gif") else ""
            start, end = sample_data.get('start'), sample_data.get('end')
            
            # 复用 OmniDataloader 的 get_video_frames
            if start is not None and end is not None:
                imgs, _, _, _ = self.processor.get_video_frames(video_file, max_tokens, selected_downsample_ratio, start, end)
            else:
                imgs, _, _, _ = self.processor.get_video_frames(video_file, max_tokens, selected_downsample_ratio, format=vformat)
                
            if not imgs:
                print(f"eval data extract video failed!!!!!!!!check video process!!：{video_path}") 
                return None, None, "error"
            
            inputs = self.processor.process_image_videos(video=imgs, merge_size=self.video_merge_size)
            return inputs["pixel_values_videos"], inputs['video_grid_thw'], "video"
            
        elif image_path:
            try:
                if isinstance(image_path, str):
                    image_path = [image_path]
                if not isinstance(image_path, list):
                    print(f"image path is not list, check data:{sample_data}")
                    return None, None, "none"
                imgs = self.processor.get_image_list_from_paths(image_path, selected_downsample_ratio)
                if not imgs:
                    print("eval data extract image failed!!!!!!!!check video process!!")
                    return None, None, "error"

                # Limit total image tokens to max_tokens (analogous to the video branch).
                # Each image's token count after projector pooling:
                #   h_patches = img.height / image_factor,  w_patches = img.width / image_factor
                #   merged: h = h_patches / merge_size,  w = w_patches / merge_size
                #   after adaptive pool: Mh, Nw = get_adaptive_pool_size(h, w, scale)
                #   tokens = Mh * Nw
                from veomni.data.multimodal.image_utils import get_adaptive_pool_size
                merge = self.image_merge_size
                factor = self.processor.image_patch_size
                kept_imgs = []
                total_tokens = 0
                for img in imgs:
                    h = img.height // factor // merge
                    w = img.width  // factor // merge
                    mh, mw = get_adaptive_pool_size(h, w, scale=selected_downsample_ratio)
                    n_tokens = mh * mw
                    if total_tokens + n_tokens > max_tokens and kept_imgs:
                        break
                    kept_imgs.append(img)
                    total_tokens += n_tokens
                if len(kept_imgs) < len(imgs):
                    print(f"[Eval Info] Truncated images from {len(imgs)} to {len(kept_imgs)} "
                          f"to fit max_tokens ({total_tokens}/{max_tokens})")
                imgs = kept_imgs

                inputs = self.processor.process_image_videos(images=imgs, merge_size=self.image_merge_size)
                return inputs["pixel_values"], inputs['image_grid_thw'], "image"
            except Exception as e:
                print(f"[Visual Load Error] {e}")
                return None, None, "error"
                
        return None, None, "none"


    def _build_return_dict(self, **kwargs) -> Dict:
        
        base = {
            "input_ids": None, "labels": None, "text": None,
            "audio_features": None, "audio_features_lens": None,
            "attention_mask_len": None, "language": "en", "category": "unknown",
            "pixel_values": None, "image_grid_thw": None,
            "pixel_values_video": None, "video_grid_thw": None,
            "audio_ground_truth_text": None, "options_num": torch.tensor([0], dtype=torch.long),
            "raw_data": None
        }
        base.update(kwargs)
        
        if base["input_ids"] is not None:
            base["attention_mask_len"] = [len(base["input_ids"])]
            if not isinstance(base["input_ids"], torch.Tensor):
                base["input_ids"] = torch.tensor(base["input_ids"], dtype=torch.long)
            if not isinstance(base["labels"], torch.Tensor):
                base["labels"] = torch.tensor(base["labels"], dtype=torch.long)
            
        if base["text"] is not None:
            base["audio_ground_truth_text"] = (base["text"], base["language"])
            
        if isinstance(base["options_num"], int):
            base["options_num"] = torch.tensor([base["options_num"]], dtype=torch.long)
            
        return base
    


@dataclass
class Qwen25VLDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    collator_func: Callable[[list, transformers.PreTrainedTokenizer], dict]
    def __call__(self, batch_data):
        batch_inputs = self.collator_func(batch_data, self.tokenizer)
        return batch_inputs



def get_eval_dataloader(tokenizer, data_args, training_args, model_args):

    eval_dataset = Qwen25VLEvaluationDataset(tokenizer, 
                                            data_args,
                                            training_args,
                                            model_args)
    
    if dist.is_available() and dist.is_initialized():
        eval_sampler = DistributedSampler(
                                        eval_dataset,
                                        shuffle=True, 
                                        drop_last=False 
                                )
    
    else:
        eval_sampler = None

    eval_collator = Qwen25VLDataCollator(tokenizer=tokenizer, collator_func=Qwen25VLcollatorFunc)
    eval_dataloader = DataLoader(
                                eval_dataset,
                                batch_size=1,
                                sampler=eval_sampler, 
                                num_workers=2,
                                collate_fn=eval_collator,
                                prefetch_factor=training_args.dataloader_prefetch_factor,
                                pin_memory=True
                            )
    eval_dataloader.categories = getattr(eval_dataset, 'categories', None)
    return eval_dataloader

def get_train_dataloader(data_args, training_args, model_args, tokenizer):

    train_dataloader = OmniDataloader(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            model_args=model_args,
            eval_mode=False
        )
    return train_dataloader




if __name__ == "__main__":
    from veomni.trainer.llava_trainer import VLMTrainingArguments, VLMMDataArguments, VLMMModelArguments
    parser = transformers.HfArgumentParser(
        (VLMMModelArguments, VLMMDataArguments, VLMTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.max_seq_len = 4096
    training_args.model_max_length = 4096
    training_args.num_train_epochs = 1
    model_args.model_path = "/mnt/afs/share/llava_qwen30B_A3B-qwen35encoder_veomni-down16"
    model_args.vision_tower = "/mnt/afs/share/Qwen35_A3B_vision_encoder"
    data_args.eval_path = "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/videommmu_mme_lvbench.json"
    
    training_args.dataloader_num_workers = 0
    model_args.mm_downsample_ratio = 16
    model_args.model_arc = "qwen2"
    training_args.per_device_train_batch_size = 1
    
    # subtitle
    data_args.sample_fps = 2
    data_args.compress_fps = True
    data_args.high_resolution_interval = 4
    data_args.max_subtitle_token_num = 2048
    data_args.minimum_image_token_num = 60
    model_args.mm_image_size = [644, 364]
    
    # training_args.remote_dataloader = True
    # data_args.offset_file_path = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0511_stage1_puretext/offsets.npy"
    # data_args.file_maping_path = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0511_stage1_puretext/file_mapping.json"
    data_args.train_path = "/mnt/afs/jiayi/code/LLaVA_hub/scripts/test_json/test_sub.jsonl"
    # data_args.file_maping_path = "/mnt/afs/jiayi/code/LLaVA_hub/scripts/test_json/test_sub.jsonl"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/afs/share/llava_qwen30B_A3B-qwen35encoder_veomni-down16",
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    
    

    dataloader = get_eval_dataloader(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
     
    )
    eval_data = Qwen25VLEvaluationDataset(tokenizer, 
                                                data_args,
                                                training_args,
                                                model_args)
    eval_data.init_image_processor()
    with open(data_args.eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for line in data:
        if line["category"] == "videommmu":
            new_data.append(line)

    data = new_data
    def check_single_item(line):
        try:
            eval_data._load_visual_features(line, max_tokens=4096)
            return True, line, None
        except Exception as e:
           
            video_identifier = line.get("video") or line.get("id") or str(line)
            return False, video_identifier, str(e)

    failed_items = []
    num_workers = 16  
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"开始多线程校验，共 {len(data)} 条数据...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
       
        futures = [executor.submit(check_single_item, line) for line in data]
        
        for future in tqdm(as_completed(futures), total=len(data), desc="Validating Videos"):
            success, video_id, err_msg = future.result()
            if not success:
                failed_items.append({"video": video_id, "error": err_msg})


    print(f"\n校验完成！共计 {len(failed_items)} / {len(data)} 个视频无法正常加载。")

    if failed_items:
        print("\n[失败列表示例]:")
        for item in failed_items[:10]:  # 仅展示前10个
            print(f"路径/标识: {item['video']} | 错误原因: {item['error']}")
        
        # 将失败的视频列表保存到本地 json
        output_fail_log = "failed_videos_log.json"
        with open(output_fail_log, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
        print(f"\n所有失败视频的完整日志已保存至: {output_fail_log}")