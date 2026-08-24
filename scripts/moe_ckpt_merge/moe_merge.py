"""
Memory-efficient MoE checkpoint merge.

Instead of loading all weights into memory at once (which OOMs for 235B+ models),
this implementation:
1. Builds a lightweight index (key -> shard file) by scanning safetensors headers only
2. Plans output shards and merged keys without holding tensor data
3. Writes each output shard by loading only the tensors needed for that shard on-demand
Peak memory ≈ one output shard (~5GB) + one layer's expert tensors during stacking.
"""

import gc
import json
import os
import re
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Dict, List, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig

from veomni.models import build_processor, build_tokenizer

SHARD_SIZE_BYTES = 8_000_000_000  # 8GB per output shard
IO_WORKERS = 8  # threads for parallel safetensors reads


# --------------------------------------------------------------------------- #
# MoE config detection (unchanged logic)
# --------------------------------------------------------------------------- #


def _get_moe_config(config) -> Tuple[int, int]:
    """Extract num_experts and first_k_dense_replace from a config object."""
    if hasattr(config, "num_experts"):
        num_experts = config.num_experts
    elif hasattr(config, "n_routed_experts"):
        num_experts = config.n_routed_experts
    else:
        raise RuntimeError(f"could not find num_experts in config: {type(config)}")

    if hasattr(config, "first_k_dense_replace"):
        moe_layer_start_idx = config.first_k_dense_replace
    else:
        moe_layer_start_idx = 0

    return num_experts, moe_layer_start_idx


def _detect_moe_groups(config) -> List[dict]:
    """Detect MoE groups from config, supporting both flat and nested (omni) models."""
    groups = []

    # Case 1: Qwen3-Omni style with thinker_config / talker_config
    if hasattr(config, "thinker_config") or hasattr(config, "talker_config"):
        if hasattr(config, "thinker_config"):
            thinker_cfg = config.thinker_config
            text_cfg = thinker_cfg.text_config if hasattr(thinker_cfg, "text_config") else thinker_cfg
            num_experts, moe_start = _get_moe_config(text_cfg)
            groups.append(
                dict(
                    prefix="thinker.model.layers",
                    num_hidden_layers=text_cfg.num_hidden_layers,
                    num_experts=num_experts,
                    moe_layer_start_idx=moe_start,
                )
            )

        if hasattr(config, "talker_config"):
            talker_cfg = config.talker_config
            text_cfg = talker_cfg.text_config if hasattr(talker_cfg, "text_config") else talker_cfg
            if hasattr(text_cfg, "num_experts") or hasattr(text_cfg, "n_routed_experts"):
                num_experts, moe_start = _get_moe_config(text_cfg)
                groups.append(
                    dict(
                        prefix="talker.model.layers",
                        num_hidden_layers=text_cfg.num_hidden_layers,
                        num_experts=num_experts,
                        moe_layer_start_idx=moe_start,
                    )
                )

        return groups

    # Case 2: Flat model (qwen3moe, deepseek, etc.)
    text_cfg = config.text_config if hasattr(config, "text_config") else config
    num_experts, moe_start = _get_moe_config(text_cfg)
    num_hidden_layers = (
        text_cfg.num_hidden_layers if hasattr(text_cfg, "num_hidden_layers") else config.num_hidden_layers
    )
    groups.append(
        dict(
            prefix="model.layers",
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            moe_layer_start_idx=moe_start,
        )
    )
    return groups


# --------------------------------------------------------------------------- #
# Lightweight shard index (no tensor data loaded)
# --------------------------------------------------------------------------- #


def _dtype_byte_size(dtype_str: str) -> int:
    """Byte size per element for safetensors dtype strings."""
    mapping = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "BOOL": 1,
    }
    return mapping.get(dtype_str, 2)  # default bf16


def _build_shard_index(shard_files: List[str]) -> Dict[str, dict]:
    """
    Build {key: {"file": path, "shape": tuple, "dtype": str, "nbytes": int}}
    by reading safetensors headers only (no tensor data loaded into RAM).
    """
    index = {}
    for filepath in tqdm(shard_files, desc="Indexing shards"):
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                slice_obj = f.get_slice(key)
                shape = slice_obj.get_shape()
                dtype = slice_obj.get_dtype()
                numel = 1
                for s in shape:
                    numel *= s
                nbytes = numel * _dtype_byte_size(dtype)
                index[key] = {"file": filepath, "shape": tuple(shape), "dtype": dtype, "nbytes": nbytes}
    return index


def _load_tensors_from_file(filepath: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Load multiple tensors from a single safetensors file (one open/close)."""
    result = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in keys:
            result[key] = f.get_tensor(key)
    return result


def _load_keys_parallel(index: Dict[str, dict], keys: List[str], max_workers: int = 8) -> Dict[str, torch.Tensor]:
    """
    Load a batch of tensors in parallel, grouped by source file.
    Each source file is opened once and all its requested keys are read in one pass.
    Multiple source files are read concurrently via threads.
    """
    # Group keys by source file
    file_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key in keys:
        file_to_keys[index[key]["file"]].append(key)

    result = {}

    if len(file_to_keys) == 1:
        # Single file: no threading overhead
        filepath, file_keys = next(iter(file_to_keys.items()))
        result = _load_tensors_from_file(filepath, file_keys)
    else:
        # Multiple files: read in parallel
        with ThreadPoolExecutor(max_workers=min(max_workers, len(file_to_keys))) as executor:
            futures = {
                executor.submit(_load_tensors_from_file, filepath, file_keys): filepath
                for filepath, file_keys in file_to_keys.items()
            }
            for future in as_completed(futures):
                result.update(future.result())

    return result


# --------------------------------------------------------------------------- #
# Expert key analysis
# --------------------------------------------------------------------------- #

# Pattern: {prefix}.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight
EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.+\.layers)\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>\w+)\.weight$"
)


def _build_expert_merge_plan(
    index: Dict[str, dict], moe_groups: List[dict]
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
        passthrough_keys: keys that don't need merging (in original order)
        merged_key_map: {new_merged_key: [expert_0_key, expert_1_key, ...]}
    """
    # Build set of valid (prefix, layer) pairs that are MoE layers
    moe_layer_set: Set[Tuple[str, int]] = set()
    group_info: Dict[str, dict] = {}  # prefix -> group
    for g in moe_groups:
        group_info[g["prefix"]] = g
        for layer_idx in range(g["moe_layer_start_idx"], g["num_hidden_layers"]):
            moe_layer_set.add((g["prefix"], layer_idx))

    expert_keys: Set[str] = set()
    # merged_key -> list of source keys (ordered by expert index)
    merged_key_map: Dict[str, List[str]] = {}

    for key in index:
        m = EXPERT_KEY_RE.match(key)
        if m:
            prefix = m.group("prefix")
            layer = int(m.group("layer"))
            if (prefix, layer) in moe_layer_set:
                expert_idx = int(m.group("expert"))
                proj = m.group("proj")
                merged_key = f"{prefix}.{layer}.mlp.experts.{proj}"
                if merged_key not in merged_key_map:
                    num_experts = group_info[prefix]["num_experts"]
                    merged_key_map[merged_key] = [None] * num_experts
                merged_key_map[merged_key][expert_idx] = key
                expert_keys.add(key)

    # Passthrough keys: everything that's not an individual expert weight
    passthrough_keys = [k for k in index if k not in expert_keys]

    return passthrough_keys, merged_key_map


# --------------------------------------------------------------------------- #
# Output key ordering and shard planning
# --------------------------------------------------------------------------- #


def _plan_output_keys(
    passthrough_keys: List[str], merged_key_map: Dict[str, List[str]], index: Dict[str, dict]
) -> List[Tuple[str, int]]:
    """
    Determine output key order and their sizes.
    Returns [(output_key, nbytes), ...] in the order they should appear.

    Strategy: insert merged keys at the position where the first expert key
    would have appeared in the original order. Non-expert keys keep original order.
    """
    # Original key order
    all_original_keys = list(index.keys())
    key_to_pos = {k: i for i, k in enumerate(all_original_keys)}

    # For each merged key, find where it should be inserted (position of expert 0)
    merged_key_positions: Dict[str, int] = {}
    for merged_key, expert_keys in merged_key_map.items():
        first_expert_key = expert_keys[0]
        merged_key_positions[merged_key] = key_to_pos[first_expert_key]

    # Compute merged key sizes: stacking num_experts tensors along dim 0
    merged_key_sizes: Dict[str, int] = {}
    for merged_key, expert_keys in merged_key_map.items():
        # Size = num_experts * size_of_one_expert
        one_expert_key = expert_keys[0]
        merged_key_sizes[merged_key] = index[one_expert_key]["nbytes"] * len(expert_keys)

    # Build output list: passthrough keys + merged keys, sorted by original position
    output_entries = []
    for key in passthrough_keys:
        output_entries.append((key, index[key]["nbytes"], key_to_pos[key]))

    for merged_key in merged_key_map:
        output_entries.append((merged_key, merged_key_sizes[merged_key], merged_key_positions[merged_key]))

    # Sort by original position
    output_entries.sort(key=lambda x: x[2])

    return [(k, nbytes) for k, nbytes, _ in output_entries]


def _assign_shards(output_keys_with_sizes: List[Tuple[str, int]], shard_size: int) -> List[List[str]]:
    """Assign output keys to shards based on size limit."""
    shards = []
    current_shard = []
    current_size = 0

    for key, nbytes in output_keys_with_sizes:
        if current_shard and current_size + nbytes > shard_size:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(key)
        current_size += nbytes

    if current_shard:
        shards.append(current_shard)

    return shards


# --------------------------------------------------------------------------- #
# Main: memory-efficient merge and save
# --------------------------------------------------------------------------- #


def main(raw_hf_path: str, merge_hf_path: str):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(merge_hf_path, exist_ok=True)

    # Load config and tokenizer (lightweight)
    config = AutoConfig.from_pretrained(raw_hf_path)
    tokenizer = build_tokenizer(raw_hf_path)
    try:
        processor = build_processor(raw_hf_path)
    except Exception:
        processor = None

    # Step 1: Build lightweight index (no tensor data)
    shard_files = sorted(glob(os.path.join(raw_hf_path, "*.safetensors")))
    assert shard_files, f"No safetensors files found in {raw_hf_path}"
    index = _build_shard_index(shard_files)
    print(f"Indexed {len(index)} tensors across {len(shard_files)} shards")

    # Step 2: Detect MoE groups and plan the merge
    moe_groups = _detect_moe_groups(config)
    print(f"Detected {len(moe_groups)} MoE group(s)")
    for g in moe_groups:
        print(f"  - {g['prefix']}: layers {g['moe_layer_start_idx']}-{g['num_hidden_layers']-1}, "
              f"{g['num_experts']} experts")

    passthrough_keys, merged_key_map = _build_expert_merge_plan(index, moe_groups)
    print(f"Passthrough keys: {len(passthrough_keys)}, Merged groups: {len(merged_key_map)}")

    # Step 3: Plan output shards
    output_keys_with_sizes = _plan_output_keys(passthrough_keys, merged_key_map, index)
    shards = _assign_shards(output_keys_with_sizes, SHARD_SIZE_BYTES)
    print(f"Output shards: {len(shards)}")

    # Step 4: Write each shard (batch-load all needed tensors per shard in parallel)
    weight_map = {}
    total_size = 0

    for shard_idx, shard_keys in enumerate(tqdm(shards, desc="Writing shards")):
        if len(shards) == 1:
            shard_filename = "model.safetensors"
        else:
            shard_filename = f"model-{shard_idx + 1:05d}-of-{len(shards):05d}.safetensors"

        # Collect ALL source keys needed for this shard (passthrough + expert sources)
        all_source_keys = []
        for key in shard_keys:
            if key in merged_key_map:
                all_source_keys.extend(merged_key_map[key])
            else:
                all_source_keys.append(key)

        # Batch load all source tensors in parallel (grouped by source file)
        loaded = _load_keys_parallel(index, all_source_keys)

        # Convert dtype in-place first to free fp32 originals early
        loaded = {k: v.to(torch.bfloat16) for k, v in loaded.items()}
        gc.collect()

        # Assemble output shard
        shard_state_dict = OrderedDict()
        for key in shard_keys:
            if key in merged_key_map:
                expert_keys = merged_key_map[key]
                shard_state_dict[key] = torch.stack([loaded.pop(ek) for ek in expert_keys])
            else:
                shard_state_dict[key] = loaded.pop(key)

            weight_map[key] = shard_filename

        del loaded

        # Compute sizes for metadata
        for tensor in shard_state_dict.values():
            total_size += tensor.numel() * tensor.element_size()

        # Save this shard
        save_file(shard_state_dict, os.path.join(merge_hf_path, shard_filename))

        # Free memory immediately
        del shard_state_dict
        gc.collect()
        print(f"  Saved {shard_filename} ({len(shard_keys)} tensors)")

    # Step 5: Write index file if sharded
    if len(shards) > 1:
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_path = os.path.join(merge_hf_path, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)
            f.write("\n")

    # Step 6: Save config, tokenizer, processor
    config.save_pretrained(merge_hf_path)
    tokenizer.save_pretrained(merge_hf_path)
    if processor is not None:
        processor.save_pretrained(merge_hf_path)

    print(f"Done. Merged checkpoint saved to {merge_hf_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_hf_path", type=str, required=True)
    parser.add_argument("--merge_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.raw_hf_path, args.merge_hf_path)
