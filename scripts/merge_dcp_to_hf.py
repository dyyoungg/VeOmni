import argparse
import gc
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan
from veomni.utils import helper


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = helper.create_logger(__name__)


def _process_and_save_shard(
    shard_idx: int,
    shard_keys: dict,
    num_shards: int,
    checkpoint_path: str,
    output_dir: str,
    save_dtype: Optional[str],
    safe_serialization: bool,
) -> tuple:
    """Worker function: load one shard from DCP, cast dtype, and save to disk.

    Inlines the _process_shard logic so we can show a per-tensor tqdm progress
    bar inside each shard.

    Returns:
        (shard_idx, filename, list_of_hf_keys)
    """
    from torch.distributed.checkpoint import FileSystemReader, load

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    if num_shards == 1:
        filename = weights_name
    else:
        prefix, extension = weights_name.rsplit(".", maxsplit=1)
        filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

    save_path = os.path.join(output_dir, filename)
    shard_label = f"Shard {shard_idx + 1}/{num_shards}"

    # --- Phase 1: read metadata & allocate empty tensors ---
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = OrderedDict()
    dcp_keys_to_load = list(shard_keys.values())

    for dcp_key in tqdm(dcp_keys_to_load, desc=f"{shard_label} | alloc tensors", unit="tensor", leave=False):
        tensor_metadata = metadata.state_dict_metadata[dcp_key]
        if not hasattr(tensor_metadata.properties, "dtype"):
            raise ValueError(
                f"Cannot determine dtype for tensor '{dcp_key}': metadata does not contain dtype information"
            )
        state_dict[dcp_key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
        )

    # --- Phase 2: bulk load from DCP (single call, no per-tensor progress) ---
    load(
        state_dict,
        checkpoint_id=checkpoint_path,
        storage_reader=FileSystemReader(checkpoint_path),
        no_dist=True,
    )

    # --- Phase 3: cast dtype, rename, and collect ---
    processed_dict = OrderedDict()
    target_dtype = None
    if save_dtype:
        target_dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype

    for hf_key, dcp_key in tqdm(shard_keys.items(), desc=f"{shard_label} | cast & rename", unit="tensor", leave=False):
        tensor = state_dict[dcp_key]

        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        processed_dict[hf_key] = tensor.cpu().detach().clone()
        del tensor

    del state_dict
    del metadata
    del reader
    gc.collect()

    # --- Phase 4: save to disk ---
    if safe_serialization:
        save_file(processed_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(processed_dict, save_path)

    hf_keys = list(shard_keys.keys())

    del processed_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return shard_idx, filename, hf_keys


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
    num_workers: int = 1,
) -> None:
    """Convert DCP checkpoint to HuggingFace format.

    Args:
        num_workers: Number of parallel processes for shard conversion.
            When > 1, multiple shards are loaded and saved concurrently.
            Each worker holds one shard in memory, so total memory usage is
            roughly ``num_workers * shard_size``.  Default 1 (sequential).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, shard_size={shard_size}"
    )

    # Plan shards from metadata
    logger.info("Analyzing DCP metadata and planning shards...")
    shards, total_size, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype)

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    num_shards = len(shards)

    # Ensure save_dtype is a string (picklable) for multiprocessing
    if isinstance(save_dtype, torch.dtype):
        save_dtype_str = str(save_dtype).replace("torch.", "")
    else:
        save_dtype_str = save_dtype

    # ---- parallel shard processing ----
    weight_map = OrderedDict()

    if num_workers > 1 and num_shards > 1:
        actual_workers = min(num_workers, num_shards)
        logger.info(f"Using {actual_workers} parallel workers to process {num_shards} shards")

        # Collect results keyed by shard_idx so we can build weight_map in order
        results = {}
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(
                    _process_and_save_shard,
                    shard_idx,
                    shard_keys,
                    num_shards,
                    str(checkpoint_path),
                    str(output_dir),
                    save_dtype_str,
                    safe_serialization,
                ): shard_idx
                for shard_idx, shard_keys in enumerate(shards)
            }

            pbar = tqdm(total=num_shards, desc="Overall progress", unit="shard")
            for future in as_completed(futures):
                shard_idx, filename, hf_keys = future.result()
                results[shard_idx] = (filename, hf_keys)
                pbar.update(1)
                pbar.set_postfix_str(f"latest: {filename}")
            pbar.close()

        # Build weight_map in deterministic shard order
        for idx in range(num_shards):
            filename, hf_keys = results[idx]
            for hf_key in hf_keys:
                weight_map[hf_key] = filename
    else:
        # Sequential fallback (original behaviour)
        for shard_idx, shard_keys in enumerate(tqdm(shards, desc="Converting shards", unit="shard")):
            _, filename, hf_keys = _process_and_save_shard(
                shard_idx,
                shard_keys,
                num_shards,
                str(checkpoint_path),
                str(output_dir),
                save_dtype_str,
                safe_serialization,
            )
            for hf_key in hf_keys:
                weight_map[hf_key] = filename

    # Save index file for multi-shard checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(f"Saved index file to {index_file}")

    logger.info("Weight conversion complete.")

    # Save model assets (config, tokenizer, processor)
    if model_assets is not None:
        for model_asset in model_assets:
            if hasattr(model_asset, "save_pretrained"):
                model_asset.save_pretrained(output_dir)
                logger.info(f"Saved model asset: {type(model_asset).__name__}")
            else:
                logger.warning(f"Model asset {model_asset} does not implement `save_pretrained`")


def merge_to_hf_pt(
    load_dir: str,
    save_path: str,
    model_assets_dir: Optional[str] = None,
    shard_size: int = 2_000_000_000,
    num_workers: int = 1,
) -> None:
    """Main conversion function: load DCP from load_dir and save HF format to save_path."""
    model_assets = None
    if model_assets_dir is not None:
        logger.info(f"Loading model assets from {model_assets_dir}")
        model_assets = []
        try:
            config = AutoConfig.from_pretrained(model_assets_dir)
            model_assets.append(config)
        except Exception as e:
            logger.warning(f"Failed to load AutoConfig: {e}")

        try:
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            model_assets.append(processor)
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor: {e}")

        if not model_assets:
            model_assets = None

    save_model_weights(save_path, load_dir, shard_size=shard_size, model_assets=model_assets, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format (streaming optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing DCP checkpoint")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace format checkpoint (default: <load-dir>/hf_ckpt)",
    )
    parser.add_argument(
        "--model-assets-dir",
        type=str,
        default=None,
        help="Directory containing model config and processor (optional)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8_000_000_000,
        help="Maximum shard size in bytes (default: 8GB)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel processes for shard conversion (default: 1, sequential). "
        "Each worker holds one shard in memory, so total memory ≈ num_workers * shard_size.",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size
    num_workers = args.num_workers

    merge_to_hf_pt(load_dir, save_dir, model_assets_dir, shard_size=shard_size, num_workers=num_workers)


if __name__ == "__main__":
    main()
