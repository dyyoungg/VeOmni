"""
Unified multimodal evaluation via SGLang server.

Supports audio / image / video inputs, MCQ and free-form generation tasks,
with pluggable metrics (accuracy, CER, WER, edit distance).

Task and metric are auto-resolved per sample when not specified:
  - audio (zh) -> generate + cer
  - audio (en) -> generate + wer
  - image/video -> mcq + accuracy
  - category ending with '_generate' -> generate + edit_distance
Use --task / --metric to override globally.

Usage:
    # Auto-resolve task & metric per sample (mixed dataset)
    python eval_sglang.py --dataset_path /path/to/mixed_data.json

    # Audio ASR with auto metric (cer/wer by language)
    python eval_sglang.py --eval_dataset aishell

    # Video MCQ (explicit)
    python eval_sglang.py --eval_dataset mvbench --task mcq --metric accuracy

    # Image MCQ with custom sampling params
    python eval_sglang.py --eval_dataset mmstar \
        --temperature 0.1 --top_k 20 --top_p 0.8

    # Custom dataset, multiple servers
    python eval_sglang.py --dataset_path /path/to/data.json \
        --task generate --metric wer --urls 10.0.0.1 10.0.0.2

    # S3 audio data
    python eval_sglang.py --dataset_path s3://bucket/eval.json
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time
import uuid
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
import jiwer
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from tqdm import tqdm
import warnings



import av
import imageio
import imageio.v3 as iio
from decord import VideoReader
from decord import cpu as decord_cpu

from veomni.data.multimodal.image_utils import qwen25vl_image_preprocess

from aoss_client.client import Client as CepthClient

_cepthclient = CepthClient("/mnt/afs/yangdeyu/aoss_ydy_game.conf")


warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".gif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".svg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".pcm"}

# ── Dataset Registry ─────────────────────────────────────────────────────────

DATASETS: Dict[str, dict] = {
    # Audio ASR datasets
    "librispeech_test_clean": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_test_clean_new.json",
        "task": "generate",
        "metric": "wer",
    },
    "librispeech_test_other": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_test_other_new.json",
        "task": "generate",
        "metric": "wer",
    },
    "librispeech_dev_clean": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_dev_clean_new.json",
        "task": "generate",
        "metric": "wer",
    },
    "librispeech_dev_other": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_dev_other_new.json",
        "task": "generate",
        "metric": "wer",
    },
    "wenet": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/wenet_test.json",
        "task": "generate",
        "metric": "cer",
    },
    "aishell": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/aishell_test.json",
        "task": "generate",
        "metric": "cer",
    },
    "commonvoice15_zh": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/commonvoice15_test.json",
        "task": "generate",
        "metric": "cer",
    },
    "commonvoice17_zh": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/commonvoice17/commonvoice17_zh_test.json",
        "task": "generate",
        "metric": "cer",
    },
    "commonvoice17_en": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/academic_mix/commonvoice17/json/commonvoice_test.json",
        "task": "generate",
        "metric": "wer",
    },
    "test_meeting": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/test_meeting.json",
        "task": "generate",
        "metric": "cer",
    },
    "test_net": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/test_net.json",
        "task": "generate",
        "metric": "cer",
    },
    "online_badaudio": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/online_badaudio.json",
        "task": "generate",
        "metric": "cer",
    },
    # Image benchmarks
    "mmstar": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/MMStar/mmstar_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "mmmu-dev": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/MMMU/mmmu-dev.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "RealWorldQA": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/RealWorldQA/RealWorldQA_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "CRPE": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/CRPE/crpe_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "MMhalBench": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/DPO_data/MMhal_Bench/mmhal-bench_without_image.jsonl",
        "task": "generate",
        "metric": "accuracy",
    },
    "RefoMB": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/DPO_data/RefoMB/train.jsonl",
        "task": "generate",
        "metric": "accuracy",
    },
    "ocr": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/ocr_eval.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "SyncDoc": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/syncdoc.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "SyncDoc_complex": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/syncdoc_new.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "chartqa": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/ChartQA/chartqa_test_final.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    # Video benchmarks
    "mvbench": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/mvbench_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "lvbench": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/lvbench_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "videomme": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/videomme_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
    "videommmu": {
        "path": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/videommmu_test.json",
        "task": "mcq",
        "metric": "accuracy",
    },
}


# ── Metrics ──────────────────────────────────────────────────────────────────


def _mixed_language_chars(text: list[str]) -> list[list[str]]:
    return [re.findall(r"[a-zA-Z0-9]+|[一-鿿]", s) for s in text]


def calculate_cer(references: list[str], hypotheses: list[str]) -> list[float]:
    return [
        jiwer.cer(
            ref,
            hyp,
            reference_transform=_mixed_language_chars,
            hypothesis_transform=_mixed_language_chars,
        )
        for ref, hyp in zip(references, hypotheses)
    ]


def calculate_wer(references: list[str], hypotheses: list[str]) -> list[float]:
    from jiwer import transforms as tr

    wer_transform = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ExpandCommonEnglishContractions(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    return [
        jiwer.wer(
            ref,
            hyp,
            reference_transform=wer_transform,
            hypothesis_transform=wer_transform,
        )
        for ref, hyp in zip(references, hypotheses)
    ]


def calculate_edit_distance(references: list[str], hypotheses: list[str]) -> list[float]:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if not ref and not hyp:
            scores.append(0.0)
            continue
        m, n = len(ref), len(hyp)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                if ref[i - 1] == hyp[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        norm = max(m, n, 1)
        scores.append(dp[n] / norm)
    return scores


def calculate_accuracy(references: list[str], hypotheses: list[str]) -> list[float]:
    return [1.0 if r.strip() == h.strip() else 0.0 for r, h in zip(references, hypotheses)]


METRIC_FUNCTIONS: Dict[str, Callable] = {
    "cer": calculate_cer,
    "wer": calculate_wer,
    "edit_distance": calculate_edit_distance,
    "accuracy": calculate_accuracy,
}


# ── Media Type Detection ─────────────────────────────────────────────────────


def detect_media_type(sample: dict) -> Tuple[str, Union[str, list]]:
    if "audio" in sample or "audio_path" in sample:
        path = sample.get("audio") or sample.get("audio_path", "")
        if path:
            return "audio", path

    if "video_path" in sample or "path" in sample:
        path = sample.get("video_path") or sample.get("path", "")
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return "video", path

    if "image_path" in sample or "image" in sample:
        path = sample.get("image_path") or sample.get("image", "")
        ext = os.path.splitext(path)[1].lower() if isinstance(path, str) else ""
        if ext in IMAGE_EXTENSIONS or isinstance(path, list):
            return "image", path

    for key in ("path", "video_path", "image_path", "image", "audio", "audio_path"):
        path = sample.get(key, "")
        if not path:
            continue
        p = path if isinstance(path, str) else (path[0] if isinstance(path, list) and path else "")
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return "video", path
        if ext in IMAGE_EXTENSIONS:
            return "image", path
        if ext in AUDIO_EXTENSIONS:
            return "audio", path

    return "unknown", sample.get("path", sample.get("image_path", sample.get("video_path", "")))


# ── Media Loading ────────────────────────────────────────────────────────────


def load_audio(
    audio_path: str,
    gain_db: float = 0.0,
    sr: int = 16000,
) -> Tuple[Union[str, bytes], Optional[float], Optional[float]]:
    """Load audio, optionally apply gain. Returns (audio_input, original_db, adjusted_db)."""
    if audio_path.startswith("s3://") and _cepthclient is not None:
        audio_bytes = _cepthclient.Get(audio_path)
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    elif os.path.exists(audio_path):
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    else:
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    rms = np.sqrt(np.mean(audio**2))
    original_db = 20 * np.log10(max(rms, 1e-10))

    if gain_db == 0.0:
        if audio_path.startswith("s3://"):
            return audio_bytes, original_db, original_db
        return audio_path, original_db, original_db

    gain_linear = 10 ** (gain_db / 20.0)
    audio = np.clip(audio * gain_linear, -1.0, 1.0)
    adjusted_rms = np.sqrt(np.mean(audio**2))
    adjusted_db = 20 * np.log10(max(adjusted_rms, 1e-10))

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), original_db, adjusted_db


def load_video_frames(
    video_path: str,
    total_sample_frames: int = 64,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    method: str = "decord",
    format: str = "",
    mm_downsample_ratio: int = 1,
    size_factor: int = 28,
) -> Tuple[list, int]:
    """Extract and preprocess frames from a video file."""
    assert iio is not None, "imageio is required for video loading"

    if os.path.exists(video_path):
        video_file = video_path
        target_input = video_path
    elif video_path.startswith("s3://") and _cepthclient is not None:
        video_bytes = _cepthclient.Get(video_path)
        target_input = f"/dev/shm/{uuid.uuid4().hex}.mp4"
        with open(target_input, "wb") as f:
            f.write(video_bytes)
        video_file = target_input
    else:
        print(f"video path does not exist: {video_path}")
        return [], 0

    is_gif = (format == "gif") or video_path.lower().endswith(".gif")
    start_frame = 0

    if is_gif:
        try:
            video = iio.imread(video_file, index=None)
            frame_count = video.shape[0]
        except Exception:
            gif_obj = Image.open(video_file)
            gif_obj.seek(0)
            fc = 0
            while True:
                try:
                    fc += 1
                    gif_obj.seek(gif_obj.tell() + 1)
                except EOFError:
                    break
            frame_count = fc
            video = iio.imiter(video_file)
        end_frame = frame_count
        framerate = 4
    else:
        container = av.open(video_file)
        meta_data = iio.immeta(video_file, index=None)
        if "duration" not in meta_data:
            meta_data["duration"] = (container.duration or 0) / 1_000_000
        container.close()
        video = None
        framerate = meta_data["fps"]
        frame_count = int(meta_data["duration"] * framerate)
        end_frame = frame_count
        if start_time is not None:
            start_frame = int(framerate * start_time)
        if end_time is not None:
            end_frame = int(framerate * end_time)
        frame_count = end_frame - start_frame
        if frame_count == 0:
            start_frame = 0
            end_frame = frame_count
            frame_count = end_frame - start_frame

    desired = min(total_sample_frames, frame_count)
    seg_size = float(frame_count - 1) / desired if desired > 0 else 1
    frame_seq = []
    for i in range(desired):
        idx = (int(seg_size * i) + int(seg_size * (i + 1))) // 2 + start_frame
        frame_seq.append(min(idx, end_frame))
    frame_seq = sorted(set(frame_seq))

    raw_img_list = []
    if method == "decord" and not is_gif and VideoReader is not None:
        try:
            vr = VideoReader(target_input, ctx=decord_cpu(0))
            images_np = vr.get_batch(frame_seq).asnumpy()
            del vr
            raw_img_list = [Image.fromarray(img).convert("RGB") for img in images_np]
        except Exception as e:
            print(f"[Warning] decord failed ({e}), falling back to imageio: {video_path}")
            raw_img_list = []

    if not raw_img_list:
        if video is None:
            video = iio.imiter(video_file, plugin="pyav", thread_count=1)
        img_index = 0
        for idx, image in enumerate(video):
            if img_index < len(frame_seq) and idx == frame_seq[img_index]:
                raw_img_list.append(Image.fromarray(image).convert("RGB"))
                img_index += 1
                if img_index == len(frame_seq):
                    break

    processed_images = qwen25vl_image_preprocess(
        raw_img_list, mm_downsample_ratio=mm_downsample_ratio, size_factor=size_factor
    )
    if len(processed_images) % 2 != 0:
        processed_images.pop()

    if target_input.startswith("/dev/shm"):
        try:
            os.remove(target_input)
        except OSError:
            pass

    return processed_images, frame_count


def load_image_frames(
    image_path: Union[str, list],
    mm_downsample_ratio: int = 1,
    size_factor: int = 28,
) -> list:
    """Load and preprocess image(s). Each image is duplicated (qwen vit processes in pairs)."""
    assert qwen25vl_image_preprocess is not None, "veomni image_utils required for image loading"

    if isinstance(image_path, str):
        image_paths = [image_path]
    else:
        image_paths = image_path

    images_list = []
    for path in image_paths:
        value = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                value = f.read()
        elif "s3://" in path and _cepthclient is not None:
            value = _cepthclient.Get(path)

        if value is not None:
            buff = io.BytesIO(np.frombuffer(value, np.uint8))
            with Image.open(buff) as image:
                image = image.convert("RGB")
            processed = qwen25vl_image_preprocess(
                image, mm_downsample_ratio=mm_downsample_ratio, size_factor=size_factor
            )
            # duplicate by design: qwen vit processes images in pairs
            images_list.extend(processed)
            images_list.extend(processed)

    return images_list


# ── SGLang Client ────────────────────────────────────────────────────────────


class MultiModalClient:
    def __init__(self, urls: list[str], sampling_params: dict):
        self.urls = urls
        self.sampling_params = sampling_params

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _encode_audio(audio: Union[np.ndarray, str, bytes]) -> str:
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                data = f.read()
        elif isinstance(audio, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, audio)
            data = buf.getvalue()
        elif isinstance(audio, bytes):
            data = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        return base64.b64encode(data).decode("utf-8")

    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        audios: Optional[list[Union[np.ndarray, str, bytes]]] = None,
        image_downsample_ratios: Optional[list[int]] = None,
    ) -> str:
        url = random.choice(self.urls)

        payload: dict = {
            "text": prompt,
            "sampling_params": self.sampling_params,
            "stream": True,
        }

        if images:
            encoded = [self._encode_image(img) for img in images]
            payload["image_data"] = encoded if len(encoded) > 1 else encoded[0]
        if audios:
            encoded = [self._encode_audio(a) for a in audios]
            payload["audio_data"] = encoded if len(encoded) > 1 else encoded[0]
        if image_downsample_ratios is not None:
            payload["image_downsample_ratios"] = image_downsample_ratios

        all_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Accept": "text/event-stream"}, json=payload) as response:
                if response.status != 200:
                    err = await response.text()
                    raise RuntimeError(f"Request failed ({response.status}): {err}")

                prev_len = 0
                async for line in response.content:
                    if not line:
                        continue
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data:"):
                        continue
                    if decoded == "data: [DONE]":
                        break
                    json_data = json.loads(decoded[5:].strip())
                    full_text = json_data.get("text", "")
                    new_text = full_text[prev_len:]
                    if new_text:
                        all_text += new_text
                        prev_len = len(full_text)

        return all_text


# ── Prompt Construction ──────────────────────────────────────────────────────

MCQ_INSTRUCTION_VIDEO = "\nPlease answer the question use a single option like (A) or (B) or...etc."
MCQ_INSTRUCTION_IMAGE = "\nAnswer the question using single option word (e.g. A or B or C or D) and nothing else."


def build_prompt(
    question: Union[str, list],
    media_type: str,
    task: str,
    num_images: int = 0,
    language: str = "en",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>"

    # vision tokens go between system and first user turn
    vision_prefix = ""
    if media_type in ("image", "video") and num_images > 0:
        vision_prefix = "<|vision_start|>" + "<image>" * num_images + "<|vision_end|>"

    if isinstance(question, list):
        for i, (role, text) in enumerate(question):
            if role.lower() in ("human", "user"):
                if i == 0 and vision_prefix:
                    prompt += vision_prefix
                prompt += "\n<|im_start|>user\n"
                if i == 0 and media_type == "audio":
                    prompt += f"{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}\n"
                prompt += text + "<|im_end|>\n"
            else:
                prompt += f"<|im_start|>assistant\n{text}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    else:
        if vision_prefix:
            prompt += vision_prefix
        prompt += "\n<|im_start|>user\n"
        if media_type == "audio":
            prompt += f"{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}\n"
        prompt += question + "<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def _parse_conversation_question(question_list: list) -> list[Tuple[str, str]]:
    """Convert [{"role": "Human", "value": "..."}, ...] into [(role, text), ...]."""
    turns = []
    for item in question_list:
        if isinstance(item, dict) and "role" in item and "value" in item:
            turns.append((item["role"], item["value"]))
    return turns


def prepare_question(
    sample: dict, task: str, media_type: str, language: str = "en",
) -> Tuple[Union[str, list], str]:
    """Build the question text and extract the ground-truth answer.

    Returns (question_text_or_turns, gt_answer).
    question_text_or_turns is a str for simple questions, or a list of
    (role, text) tuples for multi-turn conversation-format questions.
    """
    if media_type == "audio":
        gt = sample.get("text", "")
        if isinstance(gt, list):
            gt = " ".join(str(t) for t in gt)
        if task == "generate":
            q = "请转录这段音频，不要输出多余解释" if language == "zh" else "transcribe the audio directly."
        else:
            q = sample.get("question", "")
            if isinstance(q, list) and q and isinstance(q[0], dict):
                q = _parse_conversation_question(q)
            elif isinstance(q, list):
                q = "\n".join(str(x) for x in q)
        return q, str(gt)

    question = sample.get("question", "")
    is_conversation = isinstance(question, list) and question and isinstance(question[0], dict)

    if is_conversation:
        turns = _parse_conversation_question(question)
    else:
        if isinstance(question, list):
            question = "\n".join(str(q) for q in question)

    answer = sample.get("answer", sample.get("gt_answer", ""))
    if isinstance(answer, list):
        answer = answer[0] if len(answer) == 1 else "\n".join(str(a) for a in answer)

    if task == "mcq":
        answer_option = ""
        if is_conversation:
            if "candidates" in sample:
                last_role, last_text = turns[-1]
                for idx, c in enumerate(sample["candidates"]):
                    option_char = chr(ord("A") + idx)
                    last_text += f"\n({option_char}) {c}"
                    if c == answer:
                        answer_option = option_char
                last_text += MCQ_INSTRUCTION_IMAGE
                turns[-1] = (last_role, last_text)
            return turns, answer_option or str(answer)
        else:
            if "candidates" in sample:
                for idx, c in enumerate(sample["candidates"]):
                    option_char = chr(ord("A") + idx)
                    question += f"\n({option_char}) {c}"
                    if c == answer:
                        answer_option = option_char
            if media_type == "video":
                question += MCQ_INSTRUCTION_VIDEO
            else:
                question += MCQ_INSTRUCTION_IMAGE
            return question, answer_option or str(answer)
    else:
        if is_conversation:
            return turns, str(answer)
        return question, str(answer)


def extract_mcq_answer(text: str) -> str:
    match = re.search(r"([A-Z])", text)
    return match.group(1) if match else ""


def resolve_sample_task_metric(
    sample: dict,
    media_type: str,
    global_task: str = "",
    global_metric: str = "",
) -> Tuple[str, str]:
    """Per-sample task and metric resolution.

    Rules (applied when no global override is given):
    - category ending with '_generate' -> generate task, edit_distance metric
    - audio -> generate task, cer (zh) or wer (en)
    - image/video -> mcq task, accuracy metric
    """
    category = sample.get("category", sample.get("task", ""))

    if not global_task and isinstance(category, str) and category.endswith("_generate"):
        task = "generate"
    elif global_task:
        task = global_task
    elif media_type == "audio":
        task = "generate"
    else:
        task = "mcq"

    if not global_metric and isinstance(category, str) and category.endswith("_generate"):
        metric = "edit_distance"
    elif global_metric:
        metric = global_metric
    elif media_type == "audio":
        language = sample.get("language", "en")
        metric = "cer" if language == "zh" else "wer"
    else:
        metric = "accuracy"

    return task, metric


# ── Evaluation ───────────────────────────────────────────────────────────────


async def evaluate_sample(
    client: MultiModalClient,
    sample: dict,
    global_task: str,
    global_metric: str,
    media_type: str,
    media_path: Union[str, list],
    eval_frames: int = 64,
    mm_downsample_ratio: int = 1,
    size_factor: int = 28,
    volume_gain_db: float = 0.0,
) -> dict:
    language = sample.get("language", "en")
    category = sample.get("category", sample.get("task", "unknown"))
    system_prompt = sample.get("system", sample.get("system_prompt", "You are a helpful assistant."))

    task, metric = resolve_sample_task_metric(sample, media_type, global_task, global_metric)
    question_text, gt_answer = prepare_question(sample, task, media_type, language)

    images = None
    audios = None
    image_downsample_ratios = None
    extra_info: dict = {}

    if media_type == "audio":
        audio_input, original_db, adjusted_db = await asyncio.to_thread(load_audio, media_path, volume_gain_db)
        audios = [audio_input]
        extra_info["original_db"] = round(original_db, 2) if original_db is not None else None
        extra_info["adjusted_db"] = round(adjusted_db, 2) if adjusted_db is not None else None

    elif media_type == "video":
        images_list, frame_count = await asyncio.to_thread(
            load_video_frames,
            media_path,
            eval_frames,
            sample.get("start", None),
            sample.get("end", None),
            "decord",
            "",
            mm_downsample_ratio,
            size_factor,
        )
        images = images_list
        image_downsample_ratios = [mm_downsample_ratio] * (len(images_list) // 2)
        extra_info["frame_count"] = frame_count

    elif media_type == "image":
        images_list = await asyncio.to_thread(load_image_frames, media_path, mm_downsample_ratio, size_factor)
        images = images_list
        image_downsample_ratios = [mm_downsample_ratio] * (len(images_list) // 2)

    num_images = len(images) if images else 0
    prompt = build_prompt(question_text, media_type, task, num_images, language, system_prompt)

    raw_answer = await client.generate(
        prompt, images=images, audios=audios, image_downsample_ratios=image_downsample_ratios
    )
    raw_answer = raw_answer.strip()

    if task == "mcq":
        pred = extract_mcq_answer(raw_answer)
    else:
        pred = raw_answer.replace("The answer is ", "").replace(".", "").strip()

    return {
        "gt": gt_answer,
        "pred": pred,
        "raw_answer": raw_answer,
        "question": question_text,
        "media_type": media_type,
        "media_path": media_path if isinstance(media_path, str) else str(media_path),
        "category": category,
        "language": language,
        "task": task,
        "metric": metric,
        **extra_info,
    }


async def evaluate_batch(
    client: MultiModalClient,
    dataset: list[dict],
    global_task: str = "",
    global_metric: str = "",
    max_concurrency: int = 8,
    **kwargs,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[Optional[dict]] = [None] * len(dataset)
    pbar = tqdm(total=len(dataset), desc="Evaluating", unit="sample")

    async def _run(idx: int, sample: dict):
        media_type = sample["_media_type"]
        media_path = sample["_media_path"]
        async with semaphore:
            try:
                result = await evaluate_sample(
                    client, sample, global_task, global_metric, media_type, media_path, **kwargs
                )
                result["status"] = "success"
            except Exception as e:
                result = {
                    "status": "error",
                    "error": str(e),
                    "media_path": str(media_path),
                    "category": sample.get("category", "unknown"),
                }
            results[idx] = result
            pbar.update(1)

    tasks = [_run(i, s) for i, s in enumerate(dataset)]
    await asyncio.gather(*tasks)
    pbar.close()
    return [r for r in results if r is not None]


# ── Reporting ────────────────────────────────────────────────────────────────


def compute_metrics(results: list[dict]) -> dict:
    """Compute per-sample metrics, grouping by the metric each sample resolved to."""
    valid = [r for r in results if r.get("status") == "success"]
    if not valid:
        return {"total": len(results), "valid": 0, "by_metric": {}}

    # Group by metric name, compute scores per group
    by_metric: Dict[str, list] = {}
    for r in valid:
        by_metric.setdefault(r.get("metric", "accuracy"), []).append(r)

    metric_summaries: Dict[str, dict] = {}
    for metric_name, group in sorted(by_metric.items()):
        metric_fn = METRIC_FUNCTIONS[metric_name]
        gts = [r["gt"] for r in group]
        preds = [r["pred"] for r in group]
        scores = metric_fn(gts, preds)

        for r, s in zip(group, scores):
            r["metric_score"] = s

        overall = sum(scores) / len(scores) if scores else 0.0

        # per-category breakdown within this metric
        categories: Dict[str, list] = {}
        for r in group:
            cat = r.get("category", "unknown")
            categories.setdefault(cat, []).append(r["metric_score"])
        cat_scores = {cat: sum(vals) / len(vals) for cat, vals in sorted(categories.items())}

        # per-media-type breakdown within this metric
        media_types: Dict[str, list] = {}
        for r in group:
            mt = r.get("media_type", "unknown")
            media_types.setdefault(mt, []).append(r["metric_score"])
        mt_scores = {mt: sum(vals) / len(vals) for mt, vals in sorted(media_types.items())}

        metric_summaries[metric_name] = {
            "overall": overall,
            "count": len(group),
            "by_category": cat_scores,
            "by_media_type": mt_scores,
        }

    return {
        "total": len(results),
        "valid": len(valid),
        "by_metric": metric_summaries,
    }


def print_report(metrics: dict, dataset_name: str):
    print(f"\n{'=' * 60}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  Samples:  {metrics['valid']} / {metrics['total']}")

    for metric_name, summary in metrics.get("by_metric", {}).items():
        print(f"\n  [{metric_name}] overall: {summary['overall']:.4f}  (n={summary['count']})")

        if len(summary.get("by_media_type", {})) > 1:
            for mt, val in summary["by_media_type"].items():
                print(f"    media={mt}: {val:.4f}")

        if summary.get("by_category"):
            for cat, val in summary["by_category"].items():
                print(f"    {cat}: {val:.4f}")

    print(f"{'=' * 60}\n")


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_dataset(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args):
    # Resolve dataset path
    if args.dataset_path and os.path.exists(args.dataset_path):
        data_path = args.dataset_path
        dataset_name = args.eval_dataset or os.path.splitext(os.path.basename(data_path))[0]
    elif args.eval_dataset and args.eval_dataset in DATASETS:
        data_path = DATASETS[args.eval_dataset]["path"]
        dataset_name = args.eval_dataset
    else:
        raise FileNotFoundError(
            f"Dataset not found. Use --dataset_path or --eval_dataset from: {list(DATASETS.keys())}"
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Resolve task and metric: empty string means per-sample auto-resolution
    ds_info = DATASETS.get(dataset_name, {})
    global_task = args.task or ds_info.get("task", "")
    global_metric = args.metric or ds_info.get("metric", "")

    if global_metric and global_metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric '{global_metric}'. Available: {list(METRIC_FUNCTIONS.keys())}")

    dataset = load_dataset(data_path)

    # Detect media type and annotate
    video_count, image_count, audio_count = 0, 0, 0
    for sample in dataset:
        media_type, media_path = detect_media_type(sample)
        sample["_media_type"] = media_type
        sample["_media_path"] = media_path
        if media_type == "video":
            video_count += 1
        elif media_type == "image":
            image_count += 1
        elif media_type == "audio":
            audio_count += 1

    random.shuffle(dataset)

    print(f"Loaded {len(dataset)} samples from {data_path}")
    print(f"  Media: video={video_count}, image={image_count}, audio={audio_count}")
    print(f"  Task: {global_task or 'auto'}, Metric: {global_metric or 'auto'}")
    print(f"  Server URLs: {args.urls}")

    # Build client
    generate_urls = [f"http://{u}:{args.port}/generate_stream" for u in args.urls]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    client = MultiModalClient(urls=generate_urls, sampling_params=sampling_params)

    t0 = time.time()
    results = asyncio.run(
        evaluate_batch(
            client,
            dataset,
            global_task=global_task,
            global_metric=global_metric,
            max_concurrency=args.max_concurrency,
            eval_frames=args.eval_frames,
            mm_downsample_ratio=args.mm_downsample_ratio,
            size_factor=args.size_factor,
            volume_gain_db=args.volume_gain_db,
        )
    )
    elapsed = time.time() - t0
    print(f"Evaluation done in {elapsed:.1f}s")

    metrics = compute_metrics(results)
    metrics["elapsed_seconds"] = elapsed
    print_report(metrics, dataset_name)

    # Save
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"results_{dataset_name}_{args.model_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified multimodal evaluation via SGLang")

    # Server
    parser.add_argument("--urls", nargs="+", default=["127.0.0.1"], help="SGLang server IPs")
    parser.add_argument("--port", type=int, default=18003)

    # Dataset
    parser.add_argument(
        "--eval_dataset", type=str, default="", help=f"Predefined dataset: {', '.join(DATASETS.keys())}"
    )
    parser.add_argument("--dataset_path", type=str, default="", help="Custom JSON/JSONL path (overrides eval_dataset)")

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="",
        choices=["mcq", "generate", ""],
        help="Task type (default: inferred from dataset)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="",
        choices=["accuracy", "cer", "wer", "edit_distance", ""],
        help="Metric (default: inferred from dataset)",
    )

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # Media
    parser.add_argument("--eval_frames", type=int, default=64, help="Video frame count")
    parser.add_argument("--mm_downsample_ratio", type=int, default=1)
    parser.add_argument("--size_factor", type=int, default=32)
    parser.add_argument("--volume_gain_db", type=float, default=0.0, help="Audio volume gain (dB)")

    # Concurrency
    parser.add_argument("--max_concurrency", type=int, default=2)

    # Output
    parser.add_argument("--output_path", type=str, default="./exp_data")
    parser.add_argument("--model_name", type=str, default="llava")

    args = parser.parse_args()
    main(args)
