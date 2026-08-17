"""
Audio ASR evaluation via SGLang server.

Usage:
    python eval_audio_asr_sglang.py \
        --server_url http://localhost:30000/generate \
        --data_path /path/to/test.json \
        --eval_dataset aishell \
        --output_path ./results
"""

import asyncio
import aiohttp
import base64
import io
import json
import argparse
import os
import re
import time

import soundfile as sf
import jiwer
import librosa
import numpy as np
from PIL import Image
from typing import AsyncGenerator, List, Optional, Tuple, Union

DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"
# ── Dataset paths ────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = {
    "librispeech_test_clean": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_test_clean_new.json",
    "librispeech_test_other": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_test_other_new.json",
    "librispeech_dev_clean": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_dev_clean_new.json",
    "librispeech_dev_other": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/librispeech_dev_other_new.json",
    "wenet": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/wenet_test.json",
    "aishell": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/aishell_test.json",
    "commonvoice15_zh": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/commonvoice15_test.json",
    "commonvoice17_zh": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/commonvoice17/commonvoice17_zh_test.json",
    "commonvoice17_en": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/academic_mix/commonvoice17/json/commonvoice_test.json",
    "test_meeting": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/test_meeting.json",
    "test_net": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/evaluation_dataset/test_net.json",
    "online_badaudio": "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/online_badaudio.json"
}


# ── SGLang Client ────────────────────────────────────────────────────────────
class MultiModalClient:
    def __init__(self, url: str, default_sampling_params: dict, logger=None):
        self.url = url
        self.default_sampling_params = default_sampling_params
        self._logger = logger or print

    @staticmethod
    def encode_audio_to_base64(audio: Union[np.ndarray, str, bytes]) -> str:
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio_data = f.read()
        elif isinstance(audio, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, audio)
            audio_data = buffer.getvalue()
        elif isinstance(audio, bytes):
            audio_data = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        return base64.b64encode(audio_data).decode("utf-8")

    async def generate(
        self,
        prompt: str,
        audios: Optional[List[Union[np.ndarray, str, bytes]]] = None,
    ) -> str:
        """Non-streaming generation — returns full text."""
        audio_data_list = []
        if audios:
            for audio in audios:
                audio_b64 = self.encode_audio_to_base64(audio)
                audio_data_list.append(audio_b64)

        payload = {
            "text": prompt,
            "sampling_params": self.default_sampling_params,
            "stream": False,
        }

        if audio_data_list:
            payload["audio_data"] = (
                audio_data_list if len(audio_data_list) > 1 else audio_data_list[0]
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    err_msg = await response.text()
                    raise Exception(
                        f"Request failed with status {response.status}: {err_msg}"
                    )
                result = await response.json()
                return result.get("text", "")

    async def generate_stream(
        self,
        prompt: str,
        audios: Optional[List[Union[np.ndarray, str, bytes]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation — yields incremental text chunks."""
        audio_data_list = []
        if audios:
            for audio in audios:
                audio_b64 = self.encode_audio_to_base64(audio)
                audio_data_list.append(audio_b64)

        payload = {
            "text": prompt,
            "sampling_params": self.default_sampling_params,
            "stream": True,
        }

        if audio_data_list:
            payload["audio_data"] = (
                audio_data_list if len(audio_data_list) > 1 else audio_data_list[0]
            )
        all_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers={"Accept": "text/event-stream"}, json=payload
            ) as response:
                if response.status != 200:
                    err_msg = await response.text()
                    raise Exception(
                        f"Request failed with status {response.status}: {err_msg}"
                    )

                prev_len = 0
                async for line in response.content:
                    if line:
                        line_dec = line.decode("utf-8").strip()
                        if line_dec.startswith("data:"):
                            if line_dec == "data: [DONE]":
                                break
                            json_data = json.loads(line_dec[5:].strip())
                            full_text = json_data.get("text", "")
                            new_text = full_text[prev_len:]
                            if new_text:
                                all_text += new_text
                                prev_len = len(full_text)
        return all_text


# ── Volume helpers ──────────────────────────────────────────────────────────
def compute_rms_db(audio: np.ndarray) -> float:
    """Compute RMS volume in dB (relative to full-scale)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def adjust_volume(audio_path: str, gain_db: float, sr: int = 16000) -> bytes:
    """Load audio, apply gain in dB, return WAV bytes.

    gain_db > 0 amplifies, < 0 attenuates, == 0 returns original loudness.
    Output is clipped to [-1, 1] to avoid clipping distortion.
    """
    audio, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    original_db = compute_rms_db(audio)

    if gain_db != 0:
        gain_linear = 10 ** (gain_db / 20.0)
        audio = audio * gain_linear
        audio = np.clip(audio, -1.0, 1.0)

    adjusted_db = compute_rms_db(audio)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    return wav_bytes, original_db, adjusted_db


# ── Metric helpers ───────────────────────────────────────────────────────────
DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"


def mixed_language_transform(text: list[str]):
    word_list = []
    for s in text:
        item_tokens = re.findall(r"[a-zA-Z0-9]+|[一-鿿]", s)
        word_list.append(item_tokens)
    return word_list


def calculate_cer(ground_truth, hypothesis):
    return jiwer.cer(
        ground_truth,
        hypothesis,
        reference_transform=mixed_language_transform,
        hypothesis_transform=mixed_language_transform,
    )


def calculate_wer(ground_truth, hypothesis):
    from jiwer import transforms as tr

    wer_default = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ExpandCommonEnglishContractions(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    return jiwer.wer(
        ground_truth,
        hypothesis,
        reference_transform=wer_default,
        hypothesis_transform=wer_default,
    )


# ── Build the prompt (same chat template as training) ────────────────────────
def build_prompt(language: str, meta_instruction: str = "You are a helpful assistant.") -> str:
    prompt_text = (
        "请转录这段音频，不要输出多余解释"
        if language == "zh"
        else "transcribe the audio directly."
    )

    prompt = ""
    if meta_instruction:
        prompt += f"<|im_start|>system\n{meta_instruction}<|im_end|>\n"
    prompt += (
        f"<|im_start|>user\n"
        f"{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}\n"
        f"{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# ── Evaluate one sample ─────────────────────────────────────────────────────
async def evaluate_sample(
    client: MultiModalClient,
    audio_path: str,
    ground_truth_text: str,
    language: str,
    meta_instruction: str = "You are a helpful assistant.",
    volume_gain_db: float = 0.0,
) -> Tuple[float, str, float, float]:
    prompt = build_prompt(language, meta_instruction)

    # Apply volume adjustment and get dB stats
    if volume_gain_db != 0:
        wav_bytes, original_db, adjusted_db = adjust_volume(audio_path, volume_gain_db)
        audio_input = wav_bytes
    else:
        # Still compute original dB for diagnostics
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        original_db = compute_rms_db(audio)
        adjusted_db = original_db
        audio_input = audio_path

    out_text = await client.generate_stream(prompt=prompt, audios=[audio_input])
    out_text = out_text.strip()

    if language == "zh":
        metric = calculate_cer(ground_truth_text, out_text)
    else:
        metric = calculate_wer(ground_truth_text, out_text)

    return metric, out_text, original_db, adjusted_db


# ── Batch evaluation with concurrency control ───────────────────────────────
async def evaluate_batch(
    client: MultiModalClient,
    dataset: list,
    max_concurrency: int = 8,
    volume_gain_db: float = 0.0,
) -> list:
    """Evaluate all samples with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    results = [None] * len(dataset)

    async def _eval_one(idx: int, data: dict):
        async with semaphore:
            audio_path = data["audio"]
            text = data["text"]
            language = data["language"]
            category = data["category"]
            try:
                metric, out_text, original_db, adjusted_db = await evaluate_sample(
                    client, audio_path, text, language,
                    volume_gain_db=volume_gain_db,
                )
            except Exception as e:
                print(f"[{idx}] Error: {e}")
                metric, out_text, original_db, adjusted_db = None, "", None, None

            results[idx] = {
                "audio": audio_path,
                "text": text,
                "language": language,
                "category": category,
                "metric": metric,
                "prediction": out_text,
                "original_db": round(original_db, 2) if original_db is not None else None,
                "adjusted_db": round(adjusted_db, 2) if adjusted_db is not None else None,
            }
            if (idx + 1) % 2 == 0 or idx == len(dataset) - 1:
                db_info = f"vol={original_db:.1f}dB" if original_db is not None else ""
                print(
                    f"  Processed {idx + 1}/{len(dataset)}, {db_info}, "
                    f"prediction: {out_text[:80]}"
                )

    tasks = [_eval_one(i, d) for i, d in enumerate(dataset)]
    await asyncio.gather(*tasks)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    # Load dataset
    data_path = args.data_path
    if not os.path.exists(data_path):
        data_path = DEFAULT_DATASET_PATH.get(args.eval_dataset, "")
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path} / {args.eval_dataset}"
        )

    with open(data_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {data_path}")
    print(f"SGLang server: {args.server_url}")
    if args.volume_gain_db != 0:
        print(f"Volume gain: {args.volume_gain_db} dB")

    # Build client
    sampling_params = {
        "temperature": 0.1,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": 1.05,
        "top_k": 20,
        "top_p": 0.8,
    }
    client = MultiModalClient(url=args.server_url, default_sampling_params=sampling_params)

    # Run evaluation
    t0 = time.time()
    results = asyncio.run(
        evaluate_batch(client, dataset, max_concurrency=args.max_concurrency,
                       volume_gain_db=args.volume_gain_db)
    )
    elapsed = time.time() - t0
    print(f"Evaluation done in {elapsed:.1f}s ({len(dataset)} samples)")

    # Filter out failures
    valid_results = [r for r in results if r is not None and r["metric"] is not None]

    # Compute metrics
    cer_list = [r["metric"] for r in valid_results if r["language"] == "zh"]
    wer_list = [r["metric"] for r in valid_results if r["language"] == "en"]

    overall_cer = sum(cer_list) / len(cer_list) if cer_list else None
    overall_wer = sum(wer_list) / len(wer_list) if wer_list else None

    print(f"\n{'='*50}")
    print(f"  Dataset:     {args.eval_dataset}")
    print(f"  Samples:     {len(valid_results)} / {len(dataset)}")
    if overall_cer is not None:
        print(f"  Overall CER: {overall_cer:.4f}")
    if overall_wer is not None:
        print(f"  Overall WER: {overall_wer:.4f}")

    # Volume diagnostics
    db_values = [r["original_db"] for r in valid_results if r.get("original_db") is not None]
    if db_values:
        quiet_threshold = -30.0
        quiet_samples = [r for r in valid_results if r.get("original_db") is not None and r["original_db"] < quiet_threshold]
        print(f"  Volume stats: min={min(db_values):.1f}dB, max={max(db_values):.1f}dB, "
              f"mean={np.mean(db_values):.1f}dB")
        print(f"  Quiet samples (<{quiet_threshold}dB): {len(quiet_samples)}/{len(db_values)}")
        if quiet_samples:
            quiet_metrics = [r["metric"] for r in quiet_samples if r["metric"] is not None]
            normal_metrics = [r["metric"] for r in valid_results
                              if r.get("original_db") is not None and r["original_db"] >= quiet_threshold
                              and r["metric"] is not None]
            if quiet_metrics and normal_metrics:
                print(f"    Quiet avg error:  {np.mean(quiet_metrics):.4f}")
                print(f"    Normal avg error: {np.mean(normal_metrics):.4f}")
        if args.volume_gain_db != 0:
            print(f"  Volume gain applied: {args.volume_gain_db} dB")

    print(f"{'='*50}\n")

    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(
        args.output_path, f"results_{args.eval_dataset}_sglang_{args.model_name}.json"
    )

    output_data = {
        "metrics": {
            "overall_cer": overall_cer,
            "overall_wer": overall_wer,
            "total_samples": len(dataset),
            "valid_samples": len(valid_results),
            "elapsed_seconds": elapsed,
            "volume_gain_db": args.volume_gain_db,
        },
        "results": valid_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio ASR evaluation via SGLang")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:18004/generate_stream",
        help="SGLang server generate endpoint",
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--eval_dataset", type=str, default="aishell")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=10,
        help="Max concurrent requests to SGLang server",
    )
    parser.add_argument(
        "--volume_gain_db",
        type=float,
        default=0.0,
        help="Volume gain in dB to apply to all audio before eval. "
             "E.g. 10 to amplify by 10dB, 20 for 20dB. 0 = no change.",
    )
    parser.add_argument("--model_name", type=str, default="llava")
    args = parser.parse_args()
    main(args)
