"""
Vision Encoder Inference Speed Comparison
==========================================
Compares Qwen2.5-VL ViT (patch_size=14) vs Qwen3.5-MoE ViT (patch_size=16).

Image processor is loaded from each encoder's own checkpoint path so that
patch size / normalization are handled correctly.  Both encoders receive the
same list of images (identical pixel content), so the comparison is fair even
though the resulting token counts differ slightly due to different patch sizes.

Usage
-----
python profile_encoder.py \
    --qwen25_path  /path/to/qwen25_vision_encoder \
    --qwen35_path  /path/to/qwen35_vision_encoder \
    [--image_size   448]    # source image H == W in pixels
    [--image_num    1]      # number of images per forward pass
    [--dtype        bfloat16]
    [--warmup       3]
    [--runs         20]
    [--device       cuda:0]
"""

import argparse
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import (
    Qwen25ViTPretrainedModel,
    BeeBeeVLVisionModelConfig,   # side-effect: registers with AutoConfig
)
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import (
    Qwen3_5MoeViTPretrainedModel,
    BeeBeeVLQwen35MoeVisionModelConfig,  # side-effect: registers with AutoConfig
)
from veomni.data.multimodal.image_utils import Qwen25VLProcessor

# ──────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────

@contextmanager
def cuda_timer(result: list):
    """Appends elapsed ms to `result` after the block exits."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    result.append(start.elapsed_time(end))


def benchmark(model, inputs: dict, warmup: int, runs: int) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) over `runs` timed forward passes."""
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)
    torch.cuda.synchronize()

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            with cuda_timer(timings):
                model(**inputs)

    arr = np.array(timings)
    return float(arr.mean()), float(arr.std())


# ──────────────────────────────────────────────
# Image / input helpers
# ──────────────────────────────────────────────

def make_dummy_images(size: int, num: int) -> list:
    """
    Returns a list of `num` random RGB images (size × size px).
    Each image uses a distinct seed to avoid cache effects.
    """
    images = []
    for i in range(num):
        rng = np.random.default_rng(42 + i)
        pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        images.append(Image.fromarray(pixels))
    return images


def prepare_inputs(
    processor,
    images: list,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Both Qwen2.5-VL and Qwen3.5-MoE processors accept a list of images:
        pixel_values  : [total_patches_across_all_images, C * temporal * pH * pW]
        image_grid_thw: [num_images, 3]
    Remapped to lm_encode(features=..., grid_thw=...).
    """
    out = processor(images=images, return_tensors="pt")
    return {
        "hidden_states": out["pixel_values"].to(device=device, dtype=dtype),
        "grid_thw": out["image_grid_thw"].to(device=device, dtype=torch.int32),
    }


# ──────────────────────────────────────────────
# VRAM helpers
# ──────────────────────────────────────────────

def reset_peak_memory(device: torch.device):
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Qwen2.5 vs Qwen3.5MoE vision encoder"
    )
    p.add_argument("--qwen25_path", required=True,
                   help="Qwen2.5 vision encoder checkpoint dir (patch_size=14)")
    p.add_argument("--qwen35_path", required=True,
                   help="Qwen3.5MoE vision encoder checkpoint dir (patch_size=16)")
    p.add_argument("--image_size", type=int, default=672,
                   help="Source image H=W in pixels. "
                        "Tip: use a multiple of lcm(14,16)=112 — e.g. 448, 672, 1120.")
    p.add_argument("--image_num", type=int, default=1,
                   help="Number of images per forward pass (batch list length).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs",   type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ──────────────────────────────────────────────
# Per-encoder runner
# ──────────────────────────────────────────────

def run_one_encoder(
    tag: str,
    step_a: str,
    step_b: str,
    model_cls,
    model_path: str,
    processor_path: str,
    images: list,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict:
    """Load → prepare inputs → benchmark → unload. Returns a stats dict."""
    print(f"\n{step_a} Loading processor  '{processor_path}' …")
    processor = Qwen25VLProcessor.from_pretrained(processor_path)

    print(f"{step_b} Loading {tag} …")
    model = model_cls.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()

    inputs = prepare_inputs(processor, images, device, dtype)
    total_patches = inputs["hidden_states"].shape[0]
    print(f"      images       : {inputs['grid_thw'].shape[0]}")
    print(f"      pixel_values : {list(inputs['hidden_states'].shape)}")
    print(f"      grid_thw     : {inputs['grid_thw'].tolist()}")

    reset_peak_memory(device)
    mean_ms, std_ms = benchmark(model, inputs, warmup=warmup, runs=runs)
    vram_mb = peak_memory_mb(device)

    del model
    torch.cuda.empty_cache()

    return {
        "tag":           tag,
        "total_patches": total_patches,
        "mean_ms":       mean_ms,
        "std_ms":        std_ms,
        "vram_mb":       vram_mb,
    }


# ──────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────

def print_summary(results: list, image_num: int, runs: int):
    W = 74
    print()
    print("=" * W)
    print(f"Results  (image_num={image_num}, runs={runs})")
    print("=" * W)
    print(
        f"{'Encoder':<26} {'Images':>6} {'Patches':>8}"
        f" {'Mean ms':>10} {'Std ms':>8} {'VRAM MB':>9}"
    )
    print("-" * W)
    for r in results:
        print(
            f"{r['tag']:<26}"
            f"{image_num:>6}"
            f"{r['total_patches']:>8}"
            f"{r['mean_ms']:>10.2f}"
            f"{r['std_ms']:>8.2f}"
            f"{r['vram_mb']:>9.1f}"
        )
    print("-" * W)

    if len(results) == 2:
        a, b = results          # a = Qwen2.5,  b = Qwen3.5MoE
        speedup = a["mean_ms"] / b["mean_ms"]
        faster, slower = (b, a) if speedup >= 1 else (a, b)
        ratio = speedup if speedup >= 1 else 1 / speedup
        print(f"  Speedup : {faster['tag']} is {ratio:.2f}× faster than {slower['tag']}")
        print(
            f"  Per-img : {a['tag']} {a['mean_ms']/image_num:.2f} ms  |  "
            f"{b['tag']} {b['mean_ms']/image_num:.2f} ms"
        )
    print("=" * W)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    args   = parse_args()
    dtype  = getattr(torch, args.dtype)
    device = torch.device(args.device)

    print("=" * 74)
    print("Vision Encoder Speed Benchmark")
    print("=" * 74)
    print(f"  image_size : {args.image_size} × {args.image_size} px")
    print(f"  image_num  : {args.image_num}")
    print(f"  dtype      : {args.dtype}")
    print(f"  warmup     : {args.warmup}  |  runs : {args.runs}")
    print(f"  device     : {device}")

    # Build image list once — the same list is passed to both encoders
    images = make_dummy_images(args.image_size, args.image_num)

    results = []

    results.append(run_one_encoder(
        tag            = "Qwen2.5  (patch=14)",
        step_a         = "[1/4]",
        step_b         = "[2/4]",
        model_cls      = Qwen25ViTPretrainedModel,
        model_path     = args.qwen25_path,
        processor_path = args.qwen25_path,
        images         = images,
        dtype          = dtype,
        device         = device,
        warmup         = args.warmup,
        runs           = args.runs,
    ))

    results.append(run_one_encoder(
        tag            = "Qwen3.5MoE (patch=16)",
        step_a         = "[3/4]",
        step_b         = "[4/4]",
        model_cls      = Qwen3_5MoeViTPretrainedModel,
        model_path     = args.qwen35_path,
        processor_path = args.qwen35_path,
        images         = images,
        dtype          = dtype,
        device         = device,
        warmup         = args.warmup,
        runs           = args.runs,
    ))

    print_summary(results, image_num=args.image_num, runs=args.runs)


if __name__ == "__main__":
    main()