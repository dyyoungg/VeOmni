
import io
import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import av
import imageio.v3 as iio
import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from whisper.audio import pad_or_trim, log_mel_spectrogram
from snowflake import SnowflakeGenerator
from transformers import AutoConfig

from veomni.utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    IMAGE_PACTH_SIZE,
    AUDIO_TOKEN_INDEX,
    VIDEO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN,
    MIN_PIXELS_SEQ,
    MAX_PIXELS_SEQ,
    SYSTEM_PROMPTS,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_VISION_START_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    _CHAT_TEMPLATES
)
from veomni.data.multimodal.image_utils import (
    qwen25vl_image_preprocess,
    smart_resize,
    Qwen25VLProcessor
)
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import get_data_parallel_rank
from veomni.data.multimodal.image_utils import get_adaptive_pool_size, jpeg_degrade, smart_resize
from veomni.utils.constants import get_image_video_audio_placeholder
from veomni.utils.logging import get_logger
from decord import VideoReader, cpu


logger = get_logger(__name__)


@dataclass
class OmniSample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    caption_len: int
 
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
 
    pixel_values_video: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
 
    audio_features: Optional[torch.Tensor] = None
    audio_features_lens: Optional[torch.Tensor] = None
    actual_audio_feature_len: List[int] = field(default_factory=list)
 
    attention_mask_len: List[int] = field(default_factory=list)
    token_counts: Dict[str, int] = field(
        default_factory=lambda: {"image": 0, "video": 0, "audio": 0}
    )

class OmniSampleProcessor:
    """
    单条样本处理器。每个 worker 进程各自持有一个独立实例。 
    """
    def __init__(
        self,
        tokenizer,
        model_args,
        data_args,
        training_args,
        *,
        ceph_client,
        bos_client,
        rank: int,
        build_inputs_token_fn,
        preprocess_workers: int = 4,
    ):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
 
        self.ceph_client = ceph_client
        self.bos_client = bos_client
        self.rank = rank
        self._build_inputs_token = build_inputs_token_fn
 
        self.mel_bins = model_args.num_mel_bins
        self.jpeg_degrade_qualities = list(range(75, 101))
        self.preprocess_workers = preprocess_workers
 
        self.video_id_generator = SnowflakeGenerator(instance=rank)
 
        # 在 init_image_processor() 中延迟初始化（worker 进程启动后调用）
        self.image_processor: Optional[Qwen25VLProcessor] = None
        self._threadpool: Optional[ThreadPoolExecutor] = None
        self.image_merge_size = 2
        self.video_merge_size = 2
        self.image_patch_size = IMAGE_PACTH_SIZE
        self.image_factor = IMAGE_PACTH_SIZE * 2
        self._rng = random.Random()
 
        self.image_token_id, self.video_token_id, self.audio_token_id  = get_image_video_audio_placeholder(tokenizer)
        if dist.is_initialized() and get_parallel_state() is not None:
            self.dp_rank = get_data_parallel_rank()
        else:
            self.dp_rank = rank
 

    def init_image_processor(self) -> None:
        """在 worker 进程启动后调用一次，加载 image_processor 并建线程池。"""
       
        try:
            self.image_processor = Qwen25VLProcessor.from_pretrained(self.model_args.vision_tower)
            self.image_config = AutoConfig.from_pretrained(self.model_args.vision_tower)
        except Exception as e:
            # print(traceback.print_exc())
            raise ValueError(f"error loading processor: {e}")
        self.image_patch_size = getattr(self.image_config, "patch_size")
        self.image_factor = self.image_patch_size * 2
        # print("image patch size:", self.image_patch_size)
        self._threadpool = ThreadPoolExecutor(max_workers=self.preprocess_workers)
        
 
    # ------------------------------------------------------------------
    # I/O：视频
    # ------------------------------------------------------------------

    def get_video_path(self, sample_data: Dict) -> Optional[str]:
        if self.training_args.dataloader_debug:
            return "s3://video"
        try:
            video = sample_data["video"]
            if "chatgpt-videos" in video:
                return self._resolve_chatgpt_video(sample_data)
            elif "s3://" in video:
                return self._fetch_s3_video(video)
            elif "bos:/" in video:
                return self._fetch_bos_video(video)
            else:
                if os.path.exists(video):
                    return video
                print(f"video path does not exist: {video}")
                return None
        except Exception as e:
            if "s3://chatgpt-videos" in sample_data.get("video", ""):
                return None
            print(f"get video fail {sample_data} {e!r}", flush=True)
            return None
 
    def _resolve_chatgpt_video(self, sample_data: Dict) -> Optional[str]:
        base = sample_data["video"]
        vid = sample_data["id"]
        for stem in (vid[2:], vid):
            for ext in ("mp4", "webm"):
                path = f"{base}/{stem}.{ext}"
                if self.ceph_client.contains(path):
                    return self.ceph_client.Get(path)
        return None
 
    def _fetch_s3_video(self, video: str) -> Optional[str]:
        data = self.ceph_client.Get(video)
        if data is None or data == "<none>":
            print(f"video path does not exist: {video}")
            return None
        ext = video.split(".")[-1]
        video_id = next(self.video_id_generator)
        path = f"/dev/shm/{video_id}.{ext}"
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return path
 
    def _fetch_bos_video(self, bos_uri: str) -> Optional[str]:
        if bos_uri.startswith("bos://"):
            path_body = bos_uri[6:]
        elif bos_uri.startswith("bos:/"):
            path_body = bos_uri[5:]
        else:
            print(f"[Warn] Malformed BOS URI: {bos_uri}")
            return None
        try:
            bucket, key = path_body.split("/", 1)
            ext = key.split(".")[-1]
            video_id = next(self.video_id_generator)
            path = f"/dev/shm/{video_id}.{ext}"
            self.bos_client.get_object_to_file(bucket, key, path)
            return path
        except Exception as e:
            print(f"[Error] Failed to download video from BOS: {bos_uri}, {e}")
            local = locals().get("path")
            if local and os.path.exists(local):
                os.remove(local)
            return None

    # ------------------------------------------------------------------
    # I/O：图像
    # ------------------------------------------------------------------
 
    def _read_image_bytes(self, image_path: str) -> Optional[bytes]:
        """Return raw bytes for an image regardless of storage backend."""
        if "bos:/" in image_path:
            try:
                body = image_path.split("bos:/")[-1].lstrip("/")
                bucket, key = body.split("/", 1)
                return self.bos_client.get_object(bucket, key).data.read()
            except Exception as e:
                print(f"[Error] Failed to load image from BOS: {image_path} | {e}")
                return None
        elif "s3://" in image_path:
            return self.ceph_client.Get(image_path)
        else:
            with open(image_path, "rb") as f:
                return f.read()
            
    def get_image_list_from_paths(
        self, image_paths: List[str]
    ) -> List[torch.Tensor]:
        images_list = []
        for image_path in image_paths:
            raw = self._read_image_bytes(image_path)
            if raw is None:
                continue
            buf = io.BytesIO(np.frombuffer(raw, np.uint8))
            with Image.open(buf) as img:
                image = img.convert("RGB")
            images_list.extend(
                self.preprocess_image(image, image_merge_size=self.image_merge_size)
            )
        return images_list
    
   # ------------------------------------------------------------------
    # I/O：音频
    # ------------------------------------------------------------------
 
    def get_audio_from_local_or_remote(
        self, audio_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        ext = os.path.basename(audio_path).split(".")[-1]
        assert ext in ("wav", "mp3", "flac"), f"Unsupported audio format: {ext}"
        try:
            if "s3://" in audio_path:
                raw = self.ceph_client.Get(audio_path)
                y, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
            elif "bos:/" in audio_path:
                body = audio_path.split("bos:/")[-1].lstrip("/")
                bucket, key = body.split("/", 1)
                raw = self.bos_client.get_object(bucket, key).data.read()
                y, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
            else:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            return None, None

        max_len = self.model_args.audio_max_duration * sr if self.model_args.audio_max_duration else None
        if max_len and len(y) > max_len:
            y = y[-max_len:]
        return torch.from_numpy(y), sr
    
    def get_audio_data_list(self, data_dict: Dict) -> List[torch.Tensor]:
        """Load all audio tensors referenced in *data_dict*."""
        audio_field = data_dict["audio"]
        paths = [audio_field] if isinstance(audio_field, str) else audio_field

        result = []
        for path in paths:
            tensor, _ = self.get_audio_from_local_or_remote(path)
            if tensor is not None:
                result.append(tensor)
        return result
    
    # ------------------------------------------------------------------
    # 图像 / 视频预处理
    # ------------------------------------------------------------------
 
    def preprocess_image(
        self, image: Image.Image, image_merge_size: Optional[int] = None, fix_sample_fps:bool=False
    ) -> List[Image.Image]:
        if image_merge_size is None:
            image_merge_size = self.image_merge_size
        if self.training_args.jpeg_image_augmentation:
            quality = random.choice(self.jpeg_degrade_qualities)
            image = jpeg_degrade(image, quality)
        fix_size = (
            self.model_args.mm_image_size if self.training_args.fix_image_size else None
        )
        min_side = math.ceil(math.sqrt(getattr(self.model_args, "mm_downsample_ratio", 4))) * self.image_factor
        if fix_sample_fps:
            w, h = min_side * 5, min_side * 3
            fix_size = (w, h)

        return qwen25vl_image_preprocess(
            image,
            mm_downsample_ratio=getattr(self.model_args, "mm_downsample_ratio", 4),
            size_factor=self.image_factor,
            executor=self._threadpool,
            min_side=min_side,
            force_fixed_size=fix_size,
        )

    def process_image_videos(
        self,
        images=None,
        video=None,
        return_tensors: str = "pt",
        merge_size: int = 2,
        **kwargs,
    ):
        """Unified interface for image and video preprocessing."""
        if images is None and video is None:
            raise ValueError("One of 'images' or 'video' must be provided.")
        if images is not None and video is not None:
            raise ValueError("Only one of 'images' or 'video' should be provided.")

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            return self.image_processor(
                images=images,
                return_tensors=return_tensors,
                merge_size=merge_size,
                **kwargs,
            )

        # # video branch
        # if not isinstance(video, list) or not all(
        #     isinstance(f, Image.Image) for f in video
        # ):
        #     raise TypeError("'video' must be a list of PIL.Image.Image objects")
        # return self.image_processor(
        #     videos=[video],
        #     return_tensors="pt",
        #     merge_size=merge_size,
        #     **kwargs,
        # )
        
        # video branch
        if not isinstance(video, list):
            raise TypeError("'video' must be a list (List[Image.Image] or List[List[Image.Image]]).")

        # 检查是单视频还是多视频段
        if isinstance(video[0], list):
            # 多视频段模式: List[List[Image.Image]] (LongVideoProcessor 多视频)
            if not all(isinstance(f, Image.Image) for sublist in video for f in sublist):
                raise TypeError("Elements of sublists in 'video' must be PIL.Image.Image objects.")
            videos_to_process = video
        elif isinstance(video[0], Image.Image):
            # 单视频模式: List[Image.Image] 
            if not all(isinstance(f, Image.Image) for f in video):
                raise TypeError("Elements of 'video' must be PIL.Image.Image objects.")
            videos_to_process = [video]
        else:
            raise TypeError("Unrecognized video format. Must be List[Image.Image] or List[List[Image.Image]].")

        return self.image_processor(
            videos=videos_to_process,
            return_tensors="pt",
            merge_size=merge_size,
            **kwargs,
        )


    def calculate_video_each_frame_token(
        self, height: int, width: int, merge_size: int = 2, fix_sample_fps:bool=False
    ) -> Tuple[int, Tuple[int, int]]:
        if self.training_args.fix_image_size:
            rw, rh = self.model_args.mm_image_size
        else:
            min_side = math.ceil(math.sqrt(getattr(self.model_args, "mm_downsample_ratio", 4))) * self.image_factor
            if fix_sample_fps:
                rh, rw = min_side* 3, min_side * 5

            else:
                rh, rw = smart_resize(
                    height=height,
                    width=width,
                    factor=self.image_patch_size * merge_size,
                    min_pixels=MIN_PIXELS_SEQ * self.image_factor **2,
                    max_pixels=MAX_PIXELS_SEQ * self.image_factor **2,
                    min_side=min_side
                )
        resolution = (rw, rh)
        proj = self.model_args.image_projector_type

        if "avgpool" in proj or "dual_conv" in proj:
            scale = self.model_args.mm_downsample_ratio
            M = resolution[0] / self.image_patch_size / merge_size
            N = resolution[1] / self.image_patch_size / merge_size
            m, n = get_adaptive_pool_size(M, N, float(scale))
            if isinstance(self.image_processor, Qwen25VLProcessor):
                each = math.ceil(m * n / 2) + 2
            else:
                raise NotImplementedError
        elif "mlp" in proj:
            if isinstance(self.image_processor, Qwen25VLProcessor):
                each = int(math.prod(resolution) / (self.image_factor ** 2) / 2) + 2
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(f"Unknown projector type: {proj!r}")

        return each, resolution
    
    @staticmethod
    def _count_gif_frames(gif: Image.Image) -> int:
        gif.seek(0)
        frames = 0
        while True:
            try:
                frames += 1
                gif.seek(gif.tell() + 1)
            except EOFError:
                break
        return frames
    
    
    def get_seq_frames(
        self,
        total_num_frames: int,
        desired_num_frames: int,
        start_frame: int,
        end_frame: int,
        framerate: float,
        fix_sample_fps: bool=False
    ) -> List[int]:
        if self.data_args.use_finetune_fps and fix_sample_fps:
            sample_fps = 4
            step = max(1, int(framerate / sample_fps))
            seq = list(range(start_frame, end_frame, step))
            if len(seq) > desired_num_frames:
                seq = sorted(random.sample(seq, desired_num_frames))
        else:
            seg = float((total_num_frames - 1) / desired_num_frames)
            seq = []
            for i in range(desired_num_frames):
                idx = (int(seg * i) + int(seg * (i + 1))) // 2 + start_frame
                seq.append(min(idx, end_frame))
            seq = sorted(set(seq))
        return seq

    @staticmethod
    def _cleanup_shm(video_file) -> None:
        if isinstance(video_file, str) and video_file.startswith("/dev/shm"):
            try:
                os.remove(video_file)
            except Exception:
                pass


    def _load_video_inputs(
        self,
        sample_data: Dict,
        video_file: str,
        system_token: List[int],
        vformat: str,
        extra_reserved_tokens: int = 0,
    ) -> Optional[Tuple]:
        """
        Shared video-frame loading logic used by video_data_process,
        video_option_data_process, and video_audio_process.

        Returns (image_pixels, image_thw, image_num, merge_size,
                 total_image_tokens) or None on failure.
        """
        subtitle_len = 0  # caller adds their own text length
        res_token_num = (
            self.tokenizer.model_max_length
            - subtitle_len
            - len(system_token)
            - extra_reserved_tokens
        )
        fix_sample_fps = False
        if isinstance(sample_data, dict):
            fix_sample_fps = sample_data.get("sample_fps", False)

        try:
           
            imgs, _, total_tokens, _ = self.get_video_frames(
                video_file, 
                res_token_num,
                sample_data.get("start",None), 
                sample_data.get("end",None),
                method=self.training_args.video_decode_method,
                format=vformat,
                fix_sample_fps=fix_sample_fps
            )
            
            if not imgs:
                return None
            inputs = self.process_image_videos(video=imgs, merge_size=self.video_merge_size)
        except Exception as e:
            print(traceback.print_exc())
            return None

        pixels = inputs["pixel_values_videos"]
        thw = inputs["video_grid_thw"]
        image_num = thw[0][0].item()
        merge_size = self._get_merge_size_from_inputs(inputs)
        return pixels, thw, image_num, merge_size, total_tokens
    
    def extract_imagelist_from_videobytes(
        self,
        video_file,
        max_image_tokens: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        method: str = "decord",
        format: str = "",
        merge_size: int = 2,
        fix_sample_fps: bool=False
    ) -> Optional[Tuple[List[Image.Image], int, int, Tuple[int, int]]]:
        
        raw_img_list: List[Image.Image] = []
        video_io = io.BytesIO(video_file) if isinstance(video_file, bytes) else video_file
        
        video_iter = None 
        vr = None
        try:
           
            resolution = (640, 480)
            framerate = 1.0
            frame_count = 0
            
            is_gif = format.lower() == "gif"
            if not is_gif:
                if isinstance(video_file, str) and video_file.lower().endswith(".gif"):
                    is_gif = True
                elif isinstance(video_file, bytes) and video_file.startswith(b"GIF"):
                    is_gif = True
                
            if is_gif:
                method = "imageio"  
                gif_obj = Image.open(video_io)
                resolution = gif_obj.size
                frame_count = self._count_gif_frames(gif_obj)
                framerate = getattr(self.data_args, 'sample_fps', 2.0)
                gif_obj.close()
            else:
                if isinstance(video_io, io.BytesIO):
                    video_io.seek(0)
                
                with av.open(video_io) as container:
                    meta = iio.immeta(video_file, index=None)
                    if "duration" not in meta:
                        meta["duration"] = (
                            container.duration / 1_000_000
                            if container.duration is not None
                            else None
                        )
                
                video_iter = iio.imiter(video_file, plugin="pyav", thread_count=1)
                first = next(video_iter)
                resolution_actual = (first.shape[1], first.shape[0])
                resolution_meta = meta.get("size") or meta.get("source_size")
                resolution = (
                    resolution_meta
                    if resolution_meta and resolution_meta == resolution_actual
                    else resolution_actual
                )
                
                framerate = meta.get("fps", 24.0)
            
                duration = meta.get("duration", 0)
                frame_count = int(duration * framerate) if duration else 0
                
                if hasattr(video_iter, 'close'):
                    video_iter.close()
                video_iter = None

            if frame_count <= 0:
                logger.warning("[WARN] 无法获取有效的总帧数。")
                return None

            start_frame = int(framerate * start_time) if start_time is not None else 0
            end_frame = int(framerate * end_time) if end_time is not None else frame_count
            valid_frame_count = max(0, end_frame - start_frame)
            
            if valid_frame_count == 0:
                start_frame, end_frame = 0, frame_count
                valid_frame_count = end_frame - start_frame

            each_token, resized_res = self.calculate_video_each_frame_token(
                height=resolution[1], width=resolution[0], merge_size=merge_size, fix_sample_fps=fix_sample_fps
            )
            
            max_frames = int(max_image_tokens / each_token)
            if not fix_sample_fps:
                desired = (
                    min(getattr(self.data_args, 'finetune_sample_frames', max_frames), max_frames)
                    if getattr(self.data_args, 'finetune_sample_frames', 0) > 0
                    else max_frames
                )
            else:
                desired = max(max_frames, 200)
            
            frame_seq = self.get_seq_frames(
                valid_frame_count, desired, start_frame, end_frame, framerate, fix_sample_fps=fix_sample_fps
            )

            if not frame_seq:
                return None

            if method == "decord":
                try:
                    if isinstance(video_io, io.BytesIO):
                        video_io.seek(0)
                    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
                    frames_np = vr.get_batch(frame_seq).asnumpy()
                    raw_img_list = [Image.fromarray(img).convert("RGB") for img in frames_np]
                except Exception as e:
                    logger.warning(f"[WARN] Decord 抽取失败，降级使用 imageio 重新抽取: {e}")
                    method = "imageio"  
                
            if method == "imageio":
                if isinstance(video_io, io.BytesIO):
                    video_io.seek(0)
                video_iter = iio.imiter(video_file, plugin="pyav", thread_count=1)
                seq_sorted = sorted(frame_seq)
                target_idx = 0
                
                for current_idx, image in enumerate(video_iter):
                    if current_idx % 20 == 0:
                        time.sleep(0.001)
                    if current_idx == seq_sorted[target_idx]:
                        raw_img_list.append(Image.fromarray(image).convert("RGB"))
                        target_idx += 1
                        if target_idx >= len(seq_sorted):
                            break

        except Exception as e:
            logger.warning(f"[ERROR] 视频帧抽取发生异常: {e}")
            # traceback.print_exc()
            return None
            
        finally:
          
            if vr is not None:
                del vr
                
            if video_iter is not None and hasattr(video_iter, 'close'):
                try:
                    video_iter.close()
                    del video_iter
                except Exception:
                    pass
            self._cleanup_shm(video_file)
            

        if not raw_img_list:
            logger.warning("[WARN] 未能成功提取到任何图像帧。")
            return None

        return raw_img_list, valid_frame_count, each_token, resized_res

    def get_video_frames(
        self,
        video_file,
        max_image_tokens: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        method: str = "decord",
        format: str = "",
        merge_size: int = 2,
        fix_sample_fps:bool=False
    ) -> Tuple[List, int, int, Tuple[int, int]]:
        if self.training_args.dataloader_debug:
            fake = [Image.new("RGB", (644, 364), (200, 200, 200))] * 30
            return (
                self.preprocess_image(fake, image_merge_size=self.image_merge_size),
                20,
                560,
                (644, 364),
            )

        results = self.extract_imagelist_from_videobytes(
            video_file, max_image_tokens, start_time, end_time,
            method=method, format=format, merge_size=merge_size,
            fix_sample_fps=fix_sample_fps,
        )
        if results is None:
            return [], 0, 0, (0, 0)

        raw_imgs, frame_count, each_token, resolution = results
        raw_imgs = self.preprocess_image(raw_imgs, image_merge_size=self.image_merge_size, fix_sample_fps=fix_sample_fps)
        # Qwen25VL requires an even number of frames
        if len(raw_imgs) % 2 == 1 and raw_imgs:
            raw_imgs.pop()
        return raw_imgs, frame_count, len(raw_imgs) * each_token, resolution
    
    # ------------------------------------------------------------------
    # 音频特征提取
    # ------------------------------------------------------------------
 
    def _extract_audio_features(self, audio_list: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], List[int], Optional[torch.Tensor]]:
        if not audio_list:
            return None, [], None

        CHUNK_SAMPLES = 480000  # 30s * 16kHz
        frame_len = self.model_args.audio_frame_length
        compress_ratio = self.model_args.audio_downsample_ratio

        all_chunks = []
        chunk_frame_lens = []  # 每个 chunk 的有效帧长（conv2 stride=2 之后）

        for audio in audio_list:
            n_samples = len(audio)
            n_chunks = math.ceil(n_samples / CHUNK_SAMPLES)
            for i in range(n_chunks):
                start = i * CHUNK_SAMPLES
                end = min((i + 1) * CHUNK_SAMPLES, n_samples)
                chunk = audio[start:end]

                # 该 chunk 的有效帧数
                chunk_frames = math.ceil(len(chunk) / frame_len)
                chunk_frame_lens.append(chunk_frames)

                # pad_or_trim 到标准 30s 长度，再提取 mel
                chunk_padded = pad_or_trim(chunk.unsqueeze(0))  # [1, 480000]
                chunk_mel = log_mel_spectrogram(chunk_padded, n_mels=self.mel_bins)  # [1, mel_bins, 3000]
                all_chunks.append(chunk_mel)

        audio_mel = torch.cat(all_chunks, dim=0)  # [total_chunks, mel_bins, 3000]
        raw_len = torch.tensor(chunk_frame_lens)
        actual_len = [(l + compress_ratio - 1) // compress_ratio for l in raw_len]
        return audio_mel, actual_len, raw_len
    
    def _process_audio_features(self, resources: Dict) -> Dict:
        audio_data_list = resources.get("audio", [])
        
        audio_mel, actual_len, raw_len = self._extract_audio_features(audio_data_list)
        
        resources["audio_features"] = audio_mel
        resources["audio_features_lens"] = raw_len  
        resources["actual_audio_feature_len"] = actual_len
            
        return resources
    
    
    def calculate_audio_tokens(self, audio_data_list: List[torch.Tensor]) -> int:
        CHUNK_SAMPLES = 480000  # 30s * 16kHz
        frame_len = self.model_args.audio_frame_length
        ratio = self.model_args.audio_downsample_ratio

        total = 0
        for a in audio_data_list:
            n_samples = len(a)
            n_chunks = math.ceil(n_samples / CHUNK_SAMPLES)
            for i in range(n_chunks):
                chunk_samples = min(CHUNK_SAMPLES, n_samples - i * CHUNK_SAMPLES)
                chunk_frames = math.ceil(chunk_samples / frame_len)
                total += (chunk_frames + ratio - 1) // ratio

        if self.model_args.use_audio_start_end_token:
            total += 2 * len(audio_data_list)
        return total

    # ------------------------------------------------------------------
    # 模态分发
    # ------------------------------------------------------------------
    @staticmethod
    def _swap_image_to_video_keys(result: Optional[Tuple]) -> None:
        if result is None:
            return
        _, _, _, resources = result
        resources["video_pixels"] = resources.pop("image_pixels", None)
        resources["video_thw"] = resources.pop("image_thw", None)
        resources["image_pixels"] = None
        resources["image_thw"] = None

    def _dispatch_modality(self, sample_data: Dict, sample_index) -> Optional[Tuple]:
        """Route sample to the right modality processor."""
        has_video = "video" in sample_data
        has_audio = "audio" in sample_data
        has_image = "image" in sample_data
        has_option = sample_data.get("option")
        has_conv = "conversations" in sample_data

        video_file: Optional[str] = None
        if has_video:
            video_file = self.get_video_path(sample_data)
            if video_file is None:
                return None

        vformat = "gif" if sample_data.get("video", "").endswith(".gif") else ""
        system_token = self._build_system_token(sample_data, sample_index)
        result = None


        if has_video and not has_audio and has_conv:
            result = self.video_data_process(sample_data, video_file, system_token, vformat)
            self._swap_image_to_video_keys(result)
           
        elif has_option and not has_conv:
            result = self.video_option_data_process(sample_data, video_file, system_token, vformat)
            self._swap_image_to_video_keys(result)

        elif has_image and not has_audio:
            result = self.image_data_process(sample_data)
        
        elif has_audio and not has_video and not has_image:
            result = self.audio_asr_process(sample_data)

        elif has_audio and has_video:
            result = self.video_audio_process(sample_data, video_file, system_token, vformat)
            self._swap_image_to_video_keys(result)

        elif has_audio and has_image:
            result = self.image_audio_process(sample_data)

        else:
            subtitle_tokens, label_tokens, _ = self.process_conversations(sample_data)
            result = (
                    torch.tensor(subtitle_tokens, dtype=torch.long),
                    torch.tensor(label_tokens, dtype=torch.long),
                    len(subtitle_tokens),
                    {"image_pixels": None, "image_thw": None, "audio": [], "merge_sizes": None}
                )
            
        return result

    def video_data_process(
        self,
        sample_data: Dict,
        video_file: str,
        system_token: List[int],
        vformat: str,
    ) -> Optional[Tuple]:
        subtitle_tokens, label_tokens, _ = self.process_conversations(sample_data)
        res = self._load_video_inputs(sample_data, video_file, system_token, vformat,
                                      extra_reserved_tokens=len(subtitle_tokens))
        if res is None:
            return None
        pixels, thw, image_num, merge_size, total_tokens = res
        subtitle_tokens, label_tokens = self.make_image_subtitle_label_tokens(
            image_num, subtitle_tokens, label_tokens, mode="video"
        )
        cap_len = len(subtitle_tokens) + total_tokens
        resources = {"image_pixels": pixels, "image_thw": thw,
                     "merge_sizes": [merge_size], "audio": []}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )

    def video_option_data_process(
        self,
        sample_data: Dict,
        video_file: str,
        system_token: List[int],
        vformat: str,
    ) -> Optional[Tuple]:
        answer = sample_data["answer"]
        question = sample_data["question"]
        answer_idx = -1
        for idx, c in enumerate(sample_data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx

        question_tokens = self._build_inputs_token(
            question, input_type="query_format", return_tensor=False
        )
        label_tokens = [IGNORE_INDEX] * len(question_tokens)

        no_loss_text = "My best option: ("
        with_loss_text = f"{chr(ord('A') + answer_idx)}) {answer}."
        no_loss_len = len(self.tokenizer(no_loss_text)["input_ids"])
        answer_tokens = self.tokenizer(no_loss_text + with_loss_text)["input_ids"] + [151645]
        subtitle_tokens = question_tokens + answer_tokens
        label_tokens += [IGNORE_INDEX] * no_loss_len + answer_tokens[no_loss_len:]
        assert len(label_tokens) == len(subtitle_tokens)

        res = self._load_video_inputs(sample_data, video_file, system_token, vformat,
                                      extra_reserved_tokens=len(subtitle_tokens))
        if res is None:
            return None
        pixels, thw, image_num, merge_size, total_tokens = res
        subtitle_tokens, label_tokens = self.make_image_subtitle_label_tokens(
            image_num, subtitle_tokens, label_tokens, mode="video"
        )
        cap_len = len(subtitle_tokens) + total_tokens
        resources = {"image_pixels": pixels, "image_thw": thw,
                     "merge_sizes": [merge_size], "audio": []}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )

    def image_data_process(self, sample_data: Dict) -> Optional[Tuple]:
        subtitle_tokens, label_tokens, _ = self.process_conversations(sample_data)
        try:
            paths = sample_data["image"]
            if isinstance(paths, str):
                paths = [paths]
            images_list = self.get_image_list_from_paths(paths)
        except Exception as e:
            print(f"get image {sample_data['image']} error: {e!r}")
            return None
        if len(paths) != len(images_list):
            return None

        time.sleep(0.001)
        inputs = self.process_image_videos(images=images_list, merge_size=self.image_merge_size)
        pixels = inputs["pixel_values"]
        thw = inputs["image_grid_thw"]
        image_num = thw.shape[0]
        merge_size = self._get_merge_size_from_inputs(inputs, n=image_num)

        subtitle_tokens, label_tokens = self.make_image_subtitle_label_tokens(
            image_num, subtitle_tokens, label_tokens, mode="image"
        )
        cap_len = len(subtitle_tokens)
        for t, ms in zip(thw, merge_size):
            _, h, w = t
            m, n = get_adaptive_pool_size(
                (h / ms).item(), (w / ms).item(), self.model_args.mm_downsample_ratio
            )
            cap_len += m * n + 2

        if cap_len > self.tokenizer.model_max_length:
            return None

        resources = {"image_pixels": pixels, "image_thw": thw,
                     "merge_sizes": [merge_size], "audio": []}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )

    def audio_asr_process(self, sample_data: Dict) -> Optional[Tuple]:
        audio_data_list = self.get_audio_data_list(sample_data)
        if not audio_data_list:
            return None
        subtitle_tokens, label_tokens, audio_count = self.process_conversations(sample_data)
        if audio_count != len(audio_data_list):
            print(
                f"audio token count {audio_count} != audio data count "
                f"{len(audio_data_list)}: {sample_data}"
            )
            return None
        cap_len = len(subtitle_tokens) + self.calculate_audio_tokens(audio_data_list)
        resources = {"image_pixels": None, "image_thw": None, "merge_sizes": [], "audio": audio_data_list}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )
    
    def _get_merge_size_from_inputs(
        self, inputs: Dict, n: int = 1
    ) -> torch.Tensor:
        """Extract merge_size tensor from processor output, falling back to default."""
        if "merge_sizes" in inputs:
            return inputs["merge_sizes"]
        default = getattr(self.image_processor, "merge_size", 2)
        return torch.tensor([default] * n)

    def video_audio_process(
        self, sample_data: Dict, video_file: str, system_token: List[int], vformat: str
    ) -> Optional[Tuple]:
        audio_data_list = self.get_audio_data_list(sample_data)
        if not audio_data_list:
            return None
        subtitle_tokens, label_tokens, audio_count = self.process_conversations(sample_data)
        if audio_count != len(audio_data_list):
            print(f"audio token count {audio_count} != audio data count {len(audio_data_list)}")
            return None
        audio_cap = self.calculate_audio_tokens(audio_data_list)
        res = self._load_video_inputs(
            sample_data, video_file, system_token, vformat,
            extra_reserved_tokens=len(subtitle_tokens) + audio_cap,
        )
        if res is None:
            return None
        pixels, thw, image_num, merge_size, total_tokens = res
        subtitle_tokens, label_tokens = self.make_image_subtitle_label_tokens(
            image_num, subtitle_tokens, label_tokens, mode="video"
        )
        cap_len = len(subtitle_tokens) + audio_cap + total_tokens
        resources = {"image_pixels": pixels, "image_thw": thw,
                     "merge_sizes": [merge_size], "audio": audio_data_list}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )

    def image_audio_process(self, sample_data: Dict) -> Optional[Tuple]:
        audio_data_list = self.get_audio_data_list(sample_data)
        if not audio_data_list:
            return None
        subtitle_tokens, label_tokens, audio_count = self.process_conversations(sample_data)
        if audio_count != len(audio_data_list):
            print(f"audio token count {audio_count} != audio data count {len(audio_data_list)}")
            return None
        try:
            paths = sample_data["image"]
            if isinstance(paths, str):
                paths = [paths]
            images_list = self.get_image_list_from_paths(paths)
        except Exception as e:
            print(f"get image {sample_data['image']} error: {e!r}")
            return None
        if not images_list:
            return None

        time.sleep(0.001)
        inputs = self.process_image_videos(images=images_list, merge_size=self.image_merge_size)
        pixels = inputs["pixel_values"]
        thw = inputs["image_grid_thw"]
        image_num = thw.shape[0]
        merge_size = self._get_merge_size_from_inputs(inputs, n=image_num)

        subtitle_tokens, label_tokens = self.make_image_subtitle_label_tokens(
            image_num, subtitle_tokens, label_tokens, mode="image"
        )
        cap_len = len(subtitle_tokens) + self.calculate_audio_tokens(audio_data_list)
        for t, ms in zip(thw, merge_size):
            _, h, w = t
            m, n = get_adaptive_pool_size(
                (h / ms).item(), (w / ms).item(), self.model_args.mm_downsample_ratio
            )
            cap_len += m * n + 2

        if cap_len > self.tokenizer.model_max_length:
            return None

        resources = {"image_pixels": pixels, "image_thw": thw,
                     "merge_sizes": [merge_size], "audio": audio_data_list}
        return (
            torch.tensor(subtitle_tokens, dtype=torch.long),
            torch.tensor(label_tokens, dtype=torch.long),
            cap_len,
            resources,
        )
    
     # ------------------------------------------------------------------
    # 对话 token 化
    # ------------------------------------------------------------------
    

    def _build_system_token(self, sample_data: Dict, sample_idx) -> List[int]:
        if 'system' in sample_data or "system_prompt" in sample_data:
            system_prompt = sample_data.get("system", sample_data.get("system_prompt", ""))
        else:
            seed_idx = sample_idx + self.dp_rank
            system_prompt = SYSTEM_PROMPTS[seed_idx % len(SYSTEM_PROMPTS)]
        return self._build_inputs_token(system_prompt, input_type="system", return_tensor=False)
    
    def process_conversations(self, sample_data: Dict):
        """Tokenise a conversation dict into (input_ids, labels, audio_count)."""
        arc = self.model_args.model_arc
        if arc in ("qwen2", "qwen3"):
            user_fmt = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            asst_fmt = "{content}<|im_end|>"
        else:
            raise NotImplementedError(f"Unsupported model_arc: {arc!r}")

        assert not self.tokenizer.add_bos_token
        img_pat = r"(\n<image>)|(<image>\n)|(<image>)"
        vid_pat = r"(\n<video>)|(<video>\n)|(<video>)"

        text_tokens: List[int] = []
        label_tokens: List[int] = []
        audio_count = 0
        has_audio = "audio" in sample_data
        has_image = "image" in sample_data
        has_video = "video" in sample_data

        for turn_idx, c in enumerate(sample_data["conversations"]):
            # Determine speaker
            if "from" in c:
                speaker = c["from"].lower()
                if speaker in ("human", "user"):
                    is_human = True
                elif speaker in ("gpt", "assistant"):
                    is_human = False
                elif speaker == "system":
                    sys_toks = self._build_inputs_token(
                        c.get("value", c.get("content", "")),
                        input_type="system_in_middle",
                        return_tensor=False,
                    )
                    text_tokens += sys_toks
                    label_tokens += [IGNORE_INDEX] * len(sys_toks)
                    continue
                else:
                    raise NotImplementedError(f"Unknown speaker: {c['from']!r}")
            else:
                is_human = (turn_idx % 2 == 0)

            value = c.get("value", c.get("content", ""))

            if is_human:
                if has_image or has_video:
                    value = re.sub(img_pat, "", value)
                    value = re.sub(vid_pat, "", value)
                cur_text = user_fmt.format(content=value)
                audio_count += cur_text.count(DEFAULT_AUDIO_TOKEN)

                if has_audio:
                    cur_tokens = self._tokenize_with_audio_placeholders(cur_text)
                else:
                    cur_tokens = self.tokenizer(cur_text)["input_ids"]

                text_tokens += cur_tokens
                label_tokens += [IGNORE_INDEX] * len(cur_tokens)
            else:
                # assistant
                if has_video and "chatgpt-videos" in sample_data.get("video", ""):
                    value = value.split("\n")[0]
                if has_image:
                    value = re.sub(img_pat, "", value)
                cur_text = asst_fmt.format(content=value)
                cur_tokens = self.tokenizer(cur_text)["input_ids"]
                text_tokens += cur_tokens

                if c.get("infer"):
                    label_tokens += [IGNORE_INDEX] * len(cur_tokens)
                elif c.get("first_token"):
                    new_labels = [IGNORE_INDEX] * len(cur_tokens)
                    new_labels[:2] = cur_tokens[:2]
                    label_tokens += new_labels
                else:
                    label_tokens += cur_tokens

        assert len(label_tokens) == len(text_tokens), (
            f"label/token length mismatch: {len(label_tokens)} vs {len(text_tokens)}"
        )
        return text_tokens, label_tokens, audio_count
    
    def _tokenize_with_audio_placeholders(self, text: str) -> List[int]:
        """Replace DEFAULT_AUDIO_TOKEN occurrences with AUDIO_TOKEN_INDEX."""
        DUMMY = "<|image_pad|>"
        dummy_id = self.tokenizer.convert_tokens_to_ids(DUMMY)
        if self.model_args.use_audio_start_end_token:
            start = self.tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_START_TOKEN)
            end = self.tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_END_TOKEN)
            replacement = [start, AUDIO_TOKEN_INDEX, end]
        else:
            replacement = [AUDIO_TOKEN_INDEX]

        processed = text.replace(DEFAULT_AUDIO_TOKEN, DUMMY)
        tokens = []
        for tok_id in self.tokenizer(processed)["input_ids"]:
            if tok_id == dummy_id:
                tokens.extend(replacement)
            else:
                tokens.append(tok_id)
        return tokens
    
    # ------------------------------------------------------------------
    # 多模态 token 展开
    # ------------------------------------------------------------------
    def _expand_multimodal_tokens(
        self, input_ids: List[int], labels: List[int],
        image_thw=None, video_thw=None, audio_feature_len: List[int]=None
    ) -> Tuple[List[int], List[int]]:
        """Qwen2.5-VL 核心的占位符展开逻辑"""
        new_ids, new_labels = [], []
        i_idx, v_idx, a_idx = 0, 0, 0
        total_image_token, total_video_token, total_audio_token = 0, 0, 0

        for i, token_id in enumerate(input_ids):
            if token_id == IMAGE_TOKEN_INDEX and image_thw is not None:
                t, h, w = image_thw[i_idx]
                m_h, m_w = get_adaptive_pool_size(h//2, w//2, self.model_args.mm_downsample_ratio)
                num = (t * m_h * m_w).item()
                new_ids.extend([self.image_token_id] * num)
                new_labels.extend([IGNORE_INDEX] * num)
                i_idx += 1
                total_image_token += num
                
            elif token_id == VIDEO_TOKEN_INDEX and video_thw is not None:
                t, h, w = video_thw[v_idx]
                m_h, m_w = get_adaptive_pool_size(h//2, w//2, self.model_args.mm_downsample_ratio)
                num = (t * m_h * m_w).item()
                new_ids.extend([self.video_token_id] * num)
                new_labels.extend([IGNORE_INDEX] * num)
                v_idx += 1
                total_video_token += num
                
            elif token_id == AUDIO_TOKEN_INDEX and audio_feature_len:
                num = audio_feature_len[a_idx].item()
                new_ids.extend([self.audio_token_id] * num)
                new_labels.extend([IGNORE_INDEX] * num)
                a_idx += 1
                total_audio_token += num
            else:
                new_ids.append(token_id)
                new_labels.append(labels[i])

        cur_input_ids = torch.tensor(new_ids, dtype=torch.long)
        cur_labels = torch.tensor(new_labels, dtype=torch.long)
        token_counts = {"image": total_image_token, "video": total_video_token, "audio": total_audio_token}

        return cur_input_ids, cur_labels, token_counts

    def _prepend_system_prompt(
        self, system_token: List[int], input_ids: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sys_t = torch.tensor(system_token, dtype=torch.long)
        sys_label = torch.tensor([IGNORE_INDEX] * len(system_token), dtype=torch.long)
        return torch.cat([sys_t, input_ids]), torch.cat([sys_label, labels])
    
    def make_image_subtitle_label_tokens(
        self,
        image_num: int,
        subtitle_tokens: List[int],
        label_tokens: List[int],
        mode: str = "video",
    ) -> Tuple[List[int], List[int]]:
       
        start = self.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_START_TOKEN)
        end = self.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN)
        ignore3 = [IGNORE_INDEX] * 3

        if mode == "image":
            prefix_ids: List[int] = []
            prefix_labels: List[int] = []
            for _ in range(image_num):
                prefix_ids.extend([start, IMAGE_TOKEN_INDEX, end])
                prefix_labels.extend(ignore3)
        elif mode == "video":
            prefix_ids = [start] + [VIDEO_TOKEN_INDEX] + [end]
            prefix_labels = ignore3
        else:
            raise NotImplementedError(f"Unknown mode: {mode!r}")
        

        return prefix_ids + subtitle_tokens, prefix_labels + label_tokens


    def process(self, sample_data: Dict, sample_idx: int) -> Optional[OmniSample]:
       
        result = self._dispatch_modality(sample_data, sample_idx)
        if result is None:
            return
        cur_input_ids, cur_labels, cur_caption_len, multimodal_resources = result

        multimodal_resources = self._process_audio_features(multimodal_resources)

        image_thw = multimodal_resources.get("image_thw")
        video_thw = multimodal_resources.get("video_thw")
        actual_audio_lens = multimodal_resources.get("actual_audio_feature_len", [])

        cur_input_ids, cur_labels, token_counts = self._expand_multimodal_tokens(
            cur_input_ids, cur_labels, image_thw, video_thw, actual_audio_lens
        )

        system_token = self._build_system_token(sample_data, sample_idx)
        cur_input_ids, cur_labels = self._prepend_system_prompt(
            system_token, cur_input_ids, cur_labels
        )
        cur_caption_len = len(cur_input_ids)

        return OmniSample(
            input_ids=cur_input_ids,
            labels=cur_labels,
            caption_len=cur_caption_len,
            pixel_values=multimodal_resources.get("image_pixels"),
            image_grid_thw=multimodal_resources.get("image_thw"),
            pixel_values_video=multimodal_resources.get("video_pixels"),
            video_grid_thw=multimodal_resources.get("video_thw"),
            audio_features=multimodal_resources.get("audio_features"),
            audio_features_lens=multimodal_resources.get("audio_features_lens"),
            actual_audio_feature_len=multimodal_resources.get("actual_audio_feature_len", []),
            attention_mask_len=[cur_caption_len],
            token_counts=token_counts,
        )
    
class LongVideoProcessor(OmniSampleProcessor):
    def __init__(self, tokenizer, model_args, data_args, training_args, *, ceph_client, bos_client, rank, build_inputs_token_fn, preprocess_workers=4):
        super().__init__(tokenizer, 
                         model_args, 
                         data_args, 
                         training_args, 
                         ceph_client=ceph_client, 
                         bos_client=bos_client, 
                         rank=rank, 
                         build_inputs_token_fn=build_inputs_token_fn, 
                         preprocess_workers=preprocess_workers)
    
    def process_subtitle_srt(self, subtitle_bytes):
        def encode_time(time_string):
            hours, minutes, seconds = time_string.split(':')
            seconds, milliseconds = seconds.split(',')
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
            return total_seconds

        content = subtitle_bytes.decode('utf-8')
        blocks = content.split('\n\n')  # split different parts
        
        subtitles = []
        for block in blocks:
            if block.strip():
                lines = block.split('\n')  # split every line 
                index = lines[0]
                times = lines[1].split(' --> ')
                start_time = encode_time(times[0])
                end_time = encode_time(times[1])
                text = ' '.join(lines[2:])  # subtitle may includes many lines
                
                subtitle = {
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                }   
                subtitles.append(subtitle)
        return subtitles
    
    def _get_subtitle_data(self, subtitle_path: str) -> List[Dict]:
        """读取并解析 subtitle (支持 json、srt 格式)"""
        try:
            if "s3://" in subtitle_path:
                raw_data = self.ceph_client.Get(subtitle_path)
            elif "bos:/" in subtitle_path:
                body = subtitle_path.split("bos:/")[-1].lstrip("/")
                bucket, key = body.split("/", 1)
                raw_data = self.bos_client.get_object(bucket, key).data.read()
            else:
                with open(subtitle_path, 'rb') as f:
                    raw_data = f.read()
            
            if subtitle_path.endswith('.srt'):
                subtitle_list = self.process_subtitle_srt(raw_data)
            else:
                json_file = json.loads(raw_data)
                subtitle_list = []
                ori_sub_list = json_file['body'] if "body" in json_file else json_file
                
                for i, s in enumerate(ori_sub_list):
                    item = {
                        'index': i, 
                        'start_time': s['from'], 
                        'end_time': s['to'], 
                        'text': s['content']
                    }
                    if "ignore" in s: item['ignore'] = s['ignore']
                    if 'image_line' in s: item['image_line'] = s['image_line']
                    elif 'active_line' in s: item['text'] = s['active_line']
                    if 'fps4' in s: item['fps4'] = s['fps4']
                    subtitle_list.append(item)
            return subtitle_list
        except Exception as e:
            logger.error(f"[Subtitle Error] Failed to read {subtitle_path}: {e}")
            return []

    def _get_required_interval(self, time_until_start: float, is_fps4: bool) -> float:
        """动态FPS压缩逻辑"""
        if time_until_start < 10 and not is_fps4:
            return 1.0 / 1.0  # 1 FPS
        elif time_until_start < 10 and is_fps4:
            return 1.0 / 4.0  # 4 FPS
        elif time_until_start < 20:
            return 1.0 / 0.5  # 0.5 FPS
        else:
            return 1.0 / 0.1  # 0.1 FPS
            
    def resize_to_max_side(self, resolution, max_side=644):
        # 找出最大边（长边）
        (width, height) = resolution
        long_side = max(width, height)
        if long_side < max_side:
            return resolution
        # 比例因子
        scale = max_side / long_side
        # 缩放后的新尺寸，保留整数
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        return new_width, new_height


    def process(self, sample_data: Dict, sample_idx: int) -> Iterator[OmniSample]:
        video_file = self.get_video_path(sample_data)
        subtitle_path = sample_data.get("subtitle_path", "")
        if not video_file or not subtitle_path:
            return

        subtitle_list = self._get_subtitle_data(subtitle_path)
        if not subtitle_list:
            return

        system_prompt = sample_data.get("system_prompt", "You are a helpful assistant.")
        if self.model_args.model_arc in ['qwen2', 'qwen3']:
            sys_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n生成该视频的字幕<|im_end|>\n<|im_start|>assistant\n"
        else:
            sys_text = "Human: 生成该视频的字幕 Assistant:"
            
        system_tokens = torch.tensor(self.tokenizer(sys_text)["input_ids"], dtype=torch.long)

        vr = None
        reader_state = {
            "backend": "decord",
            "video_iter": None,
            "current_idx": -1,
            "last_frame_array": None  # 存最后一次解码的原始 numpy 数组，应付重复帧
        }
        
        
        try:
            # 优先尝试 Decord
            vr = VideoReader(video_file, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            init_resolution = (vr[0].shape[1], vr[0].shape[0])
            
        except Exception as e:
            logger.warning(f"[WARN] Decord 初始化失败: {e}。开始降级使用 imageio 解析基础信息。{sample_data}")
            reader_state["backend"] = "imageio"
            vr = None
            try:
                # 如果 Decord 炸了，降级尝试用 av 和 imageio 获取视频元信息
                with av.open(video_file) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else 24.0
                    
                    # 取第一帧来确定绝对真实的分辨率
                    video_iter = iio.imiter(video_file, plugin="pyav", thread_count=1)
                    first_frame = next(video_iter)
                    init_resolution = (first_frame.shape[1], first_frame.shape[0])
                    if hasattr(video_iter, 'close'):
                        video_iter.close()
                    
                    # 获取总帧数
                    if stream.frames and stream.frames > 0:
                        total_frames = stream.frames
                    else:
                        duration = container.duration / 1_000_000.0 if container.duration else 0.0
                        if duration == 0.0 and stream.duration:
                            duration = stream.duration * float(stream.time_base)
                        total_frames = int(duration * fps)
                        
                if total_frames <= 0 or fps <= 0:
                    raise ValueError(f"无效的视频属性: fps={fps}, frames={total_frames}")
                    
            except Exception as e2:
                # 连 imageio 都解析不出来，说明视频彻底损坏，放弃
                logger.error(f"[ERROR] ImageIO 兜底解析失败，视频 {video_file} 彻底损坏: {e2} {sample_data}")
                self._cleanup_shm(video_file)
                return

        safe_max_frame = max(0, total_frames - 5) 
        
        try:
            # 分辨率计算
            if self.data_args.sub_native_resolution:
                input_resolution = init_resolution
            else:
                if init_resolution == (1280, 720):
                    input_resolution = (self.model_args.mm_image_size[0], self.model_args.mm_image_size[1])
                else:
                    input_resolution = self.resize_to_max_side(init_resolution, self.model_args.mm_image_size[0])
            
            # 预估 Token (为了切分 Chunk 判断)
            each_low_res_token, resized_resolution = self.calculate_video_each_frame_token(
                height=input_resolution[1], width=input_resolution[0], merge_size=self.video_merge_size
            )
            each_high_res_token, _ = self.calculate_video_each_frame_token(
                height=init_resolution[1], width=init_resolution[0], merge_size=self.video_merge_size
            )
            
            if self.data_args.high_resolution_interval > 0.0:
                denominator = self.data_args.high_resolution_interval * self.data_args.sample_fps
                ratio = (init_resolution[0] * init_resolution[1]) / (input_resolution[0] * input_resolution[1])
                numerator = self.data_args.high_resolution_interval * self.data_args.sample_fps - 2.0 + ratio * 2.0
                each_res_token = int(each_low_res_token * numerator / denominator)

            max_len = self.tokenizer.model_max_length
            video_token_tensor = torch.tensor([VIDEO_TOKEN_INDEX], dtype=torch.long)
            ignore_tensor = torch.tensor([IGNORE_INDEX], dtype=torch.long)
            start_img_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_START_TOKEN)], dtype=torch.long)
            end_img_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN)], dtype=torch.long)

            # ====== 状态维护 ======
            tokens = [system_tokens]
            labels = [torch.full((system_tokens.shape[0],), IGNORE_INDEX, dtype=torch.long)]
            types = ["system"]  # 用于清理: system, vision_start, video, vision_end, text
            video_segments = [None] # 只对 "video" 有效，保存 (resolution, frame_indices)

            token_num = len(system_tokens)
            text_token_num = len(system_tokens)
            global_valid_image_count = 0
            valid_text_count = 0 
            minimum_image_token_num = getattr(self.data_args, 'minimum_image_token_num', 60)
            cur_time = 0
            for sub_idx, sub in enumerate(subtitle_list):
                start_time = sub['start_time']
                end_time = min(sub['end_time'], start_time + 10)
                
                # 动态 FPS 帧抽取
                frame_indices = []
                if not sub.get('ignore', False):
                    sample_fps = 4 if (sub.get('fps4', False) or 'image_line' in sub) else self.data_args.sample_fps
                    
                    raw_times = []
                    while cur_time < end_time:
                        raw_times.append(cur_time)
                        cur_time += 1.0 / sample_fps

                    if getattr(self.data_args, 'compress_fps', True):
                        last_kept_time = -float('inf')
                        for t in raw_times:
                            time_until_start = end_time - t
                            req_interval = self._get_required_interval(time_until_start, sub.get('fps4', False))
                            if (t >= last_kept_time + req_interval - 1e-9) or (sub.get('fps4', False) and t >= last_kept_time + req_interval - 0.03):
                                frame_indices.append(min(int(t * fps), safe_max_frame))
                                last_kept_time = t
                    else:
                        frame_indices = [min(int(t * fps), safe_max_frame) for t in raw_times]

                    cur_gap_token_est = len(frame_indices) * each_res_token # 取中间保守估计
                    
                    if valid_text_count == 0 or cur_gap_token_est > max_len - self.data_args.max_subtitle_token_num:
                        if len(frame_indices) > minimum_image_token_num:
                            frame_indices = frame_indices[-minimum_image_token_num:]

                # 分辨率交错判断并打包视频段
                cur_tokens, cur_labels, cur_types, cur_segments = [], [], [], []
                cur_token_num = 0

                if frame_indices:
                    groups = [] 
                    current_group_res = None
                    current_group_frames = []

                    for idx in frame_indices:
                        if not getattr(self.data_args, 'no_image', False):
                            global_valid_image_count += 1
                            # 还原交错逻辑
                            if self.data_args.high_resolution_interval > 0.0:
                                interval = int(self.data_args.high_resolution_interval * sample_fps)
                                pattern = [0] * (interval - 2) + [1, 1]
                                res_mode = pattern[(global_valid_image_count - 1) % len(pattern)]
                                res = init_resolution if res_mode == 1 else resized_resolution
                            else:
                                res = resized_resolution

                            if current_group_res is None: current_group_res = res
                            if res != current_group_res:
                                groups.append((current_group_res, current_group_frames))
                                current_group_res = res
                                current_group_frames = [idx]
                            else:
                                current_group_frames.append(idx)
                    
                    if current_group_frames:
                        groups.append((current_group_res, current_group_frames))

                    # 过滤空组
                    valid_groups = []
                    for res, f_indices in groups:
                        if len(f_indices) % 2 != 0:
                            f_indices.append(f_indices[-1]) # 补帧
                        if f_indices:
                            valid_groups.append((res, f_indices))

                    if valid_groups:
                        # 整个 subtitle 段落只加一次 start end
                        if getattr(self.model_args, 'split_img_token', True) and not getattr(self.data_args, 'no_image', False):
                            cur_tokens.append(start_img_tensor)
                            cur_labels.append(ignore_tensor)
                            cur_types.append("vision_start")
                            cur_segments.append(None)
                            cur_token_num += 1

                        # 中间变分辨率只追加 <video> token 占位符
                        for res, f_indices in valid_groups:
                            seg_tokens = len(f_indices) * (each_high_res_token if res == init_resolution else each_low_res_token)
                            cur_tokens.append(video_token_tensor)
                            cur_labels.append(ignore_tensor)
                            cur_types.append("video")
                            cur_segments.append((res, f_indices))
                            cur_token_num += seg_tokens

                        if getattr(self.model_args, 'split_img_token', True) and not getattr(self.data_args, 'no_image', False):
                            cur_tokens.append(end_img_tensor)
                            cur_labels.append(ignore_tensor)
                            cur_types.append("vision_end")
                            cur_segments.append(None)
                            cur_token_num += 1

                # 追加文本
                sub_text = random.choice(sub['image_line']) if 'image_line' in sub else sub['text']
                text_tensor = torch.tensor(self.tokenizer(sub_text)["input_ids"], dtype=torch.long)
                
                cur_tokens.append(text_tensor)
                if sub.get('ignore', False):
                    cur_labels.append(torch.full((text_tensor.shape[0],), IGNORE_INDEX, dtype=torch.long))
                else:
                    cur_labels.append(text_tensor)
                cur_types.append("text")
                cur_segments.append(None)
                cur_token_num += len(text_tensor)

                # 判断 Chunk 切分并清理过期 Token
                if token_num + cur_token_num > max_len or sub_idx == len(subtitle_list) - 1:
                    
                    # 尝试构建 Chunk
                    chunk = self._build_chunk(tokens, labels, types, video_segments, vr, video_file, reader_state)

                    
                    # 如果遇到无法处理的视频文件，构建失败，丢弃整个视频直接返回
                    if chunk is None:
                        logger.error(f"[Discard Video] Discarding whole session for sample_data {sample_data} video {video_file} due to extraction failure.")
                        return  
                    
                    yield chunk
                    del chunk
                    
                    # ========= 清理机制 =========
                    valid_text_count = 0
                    remove_indices = []
                    for i in range(1, len(tokens)): # 保留 system_tokens (索引0)
                        if types[i] in ["vision_start", "video", "vision_end"]:
                            remove_indices.append(i)
                        elif types[i] == "text":
                            # 过去的文本不再计算 Loss，仅作为历史 Context
                            labels[i] = labels[i] * 0 + IGNORE_INDEX
                            # 超出最大历史长度则弹出
                            if text_token_num > getattr(self.data_args, 'max_subtitle_token_num', 2048):
                                remove_indices.append(i)
                                text_token_num -= len(tokens[i])

                    for i in reversed(remove_indices):
                        tokens.pop(i)
                        labels.pop(i)
                        types.pop(i)
                        video_segments.pop(i)
                    token_num = text_token_num
                    # =================================================

                # 合并当前步
                tokens.extend(cur_tokens)
                labels.extend(cur_labels)
                types.extend(cur_types)
                video_segments.extend(cur_segments)
                token_num += cur_token_num
                text_token_num += len(text_tensor)
                valid_text_count += len(text_tensor)
                
        except Exception as e:
            logger.error(f"Error processing sample_data {sample_data} video {video_file}:  {e}")
            return
        finally:
            # 统一关闭清理
            if vr is not None:
                del vr
            if reader_state["video_iter"] is not None:
                try:
                    reader_state["video_iter"].close()
                    del reader_state["video_iter"]
                except Exception:
                    pass
            self._cleanup_shm(video_file)
            import gc
            gc.collect()

    def _build_chunk(self, tokens: List[torch.Tensor], labels: List[torch.Tensor], types: List[str], video_segments: List[Any], vr, video_file: str, reader_state: Dict) -> Optional[OmniSample]:
        """将积累的视频帧和 Tokens 打包成 Qwen2.5-VL 需要的格式"""
        from PIL import Image
        video_clips = []

        for seg in video_segments:
            if seg is not None:
                res, f_indices = seg
                pil_frames = []
                
                # 尝试用 decord 
                if reader_state["backend"] == "decord" and vr is not None:
                    try:
                        frames_np = vr.get_batch(f_indices).asnumpy()
                        pil_frames = [Image.fromarray(img).convert("RGB").resize(res, Image.Resampling.BICUBIC) for img in frames_np]
                    except Exception as e:
                        logger.warning(f"[WARN] Decord 失败，全局切换到 imageio: {e}")
                        reader_state["backend"] = "imageio"
                
                # decord失败或全局已经是imageio，执行流式兜底提取
                if reader_state["backend"] == "imageio":
                    if reader_state["video_iter"] is None:
                        reader_state["video_iter"] = iio.imiter(video_file, plugin="pyav", thread_count=1)
                        reader_state["current_idx"] = -1
                        reader_state["last_frame_array"] = None

                    v_iter = reader_state["video_iter"]

                    for target in f_indices:
                        frame_found = False
                        
                        # 目标帧就是刚读过的上一帧（即重复帧），直接复用
                        if target == reader_state["current_idx"] and reader_state["last_frame_array"] is not None:
                            # 拿原始数组重新按照当前要求的 res 缩放 (应对跨段落分辨率不同的情况)
                            img_pil = Image.fromarray(reader_state["last_frame_array"]).convert("RGB").resize(res, Image.Resampling.BICUBIC)
                            pil_frames.append(img_pil)
                            continue
                            
                        # 发生回退乱序（你确认不会发生，防范性 break）
                        if target < reader_state["current_idx"]:
                            break
                            
                        # 目标帧在后面，正常向后遍历直到追上 target
                        while reader_state["current_idx"] < target:
                            try:
                                image_array = next(v_iter)
                                reader_state["current_idx"] += 1
                                
                                if reader_state["current_idx"] % 20 == 0:
                                    time.sleep(0.001)
                                
                                # 追上目标帧了
                                if reader_state["current_idx"] == target:
                                    reader_state["last_frame_array"] = image_array # 留个底，以备下一个是重复帧
                                    img_pil = Image.fromarray(image_array).convert("RGB").resize(res, Image.Resampling.BICUBIC)
                                    pil_frames.append(img_pil)
                                    frame_found = True
                                    break
                            except StopIteration:
                                break # 视频读完了提前 EOF
                        
                        # 如果没找到（比如 EOF 或者遇到乱序），直接放弃当前段落
                        if not frame_found:
                            break

                # 校验：如果抽取出来的数量和要求的数量对不上，说明 EOF 或损坏
                if len(pil_frames) < len(f_indices):
                    logger.error(f"[Discard Segment] Expected {len(f_indices)} frames but got only {len(pil_frames)}. Marking whole video as failed.")
                    return None
                
                video_clips.append(pil_frames)

        pixel_values_videos = None
        video_grid_thw = None
        # Qwen2.5-VL 特征处理
        if video_clips:
            inputs = self.process_image_videos(video=video_clips, merge_size=self.video_merge_size)
            pixel_values_videos = inputs["pixel_values_videos"]
            video_grid_thw = inputs["video_grid_thw"]
            
            del video_clips
            del inputs


        # Token 拼接与展开
        flat_tokens = torch.cat(tokens, dim=0).tolist()
        flat_labels = torch.cat(labels, dim=0).tolist()

        cur_input_ids, cur_labels, token_counts = self._expand_multimodal_tokens(
            input_ids=flat_tokens, 
            labels=flat_labels, 
            image_thw=None, 
            video_thw=video_grid_thw, 
            audio_feature_len=[]
        )

        return OmniSample(
            input_ids=cur_input_ids,
            labels=cur_labels,
            caption_len=len(cur_input_ids),
            pixel_values_video=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask_len=[len(cur_input_ids)],
            token_counts=token_counts,
        )

