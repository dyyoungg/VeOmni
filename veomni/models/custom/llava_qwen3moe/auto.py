from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging
import json
import os
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
import glob
from safetensors import safe_open

from veomni.models.custom.llava_qwen3moe.configuration_qwen3moe_omni import Qwen3MoeOmniConfig
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
from veomni.models.custom.llava_qwen3moe.modeling_qwen3_audio_encoder import BeeBeeQwen3AudioModelConfig
from veomni.models.custom.llava_qwen3moe.modeling_llava_qwen3moe_omni import LlavaQwen3MoeForCausalLM
from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import BeeBeeVLQwen35MoeVisionModelConfig
from veomni.models.module_utils import init_empty_weights, load_model_weights
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
from veomni.ops.fused_moe import apply_veomni_fused_moe_patch
from veomni.utils.logging import get_logger

if is_transformers_version_greater_or_equal_to("5.0.0"):
    from transformers.initialization import no_init_weights
else:
    from transformers.modeling_utils import no_init_weights

logger = get_logger(__name__)

def _set_attn_implementation_in_config(config: PretrainedConfig, attn_implementation: str, moe_implementation) -> None:
    # The custom omni wrapper forwards `config._attn_implementation` into submodules.
    setattr(config, "_attn_implementation", attn_implementation)
    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused", "fused_quack"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        logger.info_rank0(f"MoE implementation: {moe_implementation}")
        setattr(config, "_moe_implementation", moe_implementation)
        apply_veomni_fused_moe_patch(moe_implementation=moe_implementation)
      
        

def _set_foundation_dtype_in_config(config: PretrainedConfig, torch_dtype: str) -> None:
    if torch_dtype == "bfloat16":
        config.dtype = torch.bfloat16
    elif torch_dtype == "float32":
        config.dtype = torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}. Use 'bfloat16' or 'float32'.")


def _compose_vision_config(
    vision_config_path: str,
    output_size: int,
    *,
    image_downsample_size: int = 8,
    image_projector_type: str = "dynamic_avgpool",
    train_vision_projector: bool = True,
    freeze_vision_merger: bool = False,
) -> BeeBeeVLVisionModelConfig:
    raw_cfg_path = os.path.join(vision_config_path, "config.json")
    if os.path.isfile(raw_cfg_path):
        with open(raw_cfg_path, "r", encoding="utf-8") as f:
            vision_dict = json.load(f)
    else:
        # Fallback: use Transformers config object (may reshape fields into nested sub-configs).
        base_vision_cfg = AutoConfig.from_pretrained(vision_config_path, trust_remote_code=True)
        print("vision config", base_vision_cfg)
        vision_dict = base_vision_cfg.to_dict()
        # Some `Qwen2_5_VLConfig` variants store the real vision dims under `text_config`,
        # while `hidden_size/intermediate_size` at the top-level may be absent.
        if vision_dict.get("hidden_size", None) is None and getattr(base_vision_cfg, "text_config", None) is not None:
            vision_dict["hidden_size"] = getattr(base_vision_cfg.text_config, "hidden_size", None)
        if vision_dict.get("intermediate_size", None) is None and getattr(base_vision_cfg, "text_config", None) is not None:
            vision_dict["intermediate_size"] = getattr(base_vision_cfg.text_config, "intermediate_size", None)

    vision_type = vision_dict.pop("model_type", None)

    if vision_type  == "qwen2_5_vl":
        vision_cfg = BeeBeeVLVisionModelConfig(
            **vision_dict,
            output_size=output_size,
            image_downsample_size=image_downsample_size,
            image_projector_type=image_projector_type,
            return_hidden_states=False,
            train_vision_projector=train_vision_projector,
            freeze_vision_merger=freeze_vision_merger,
        )
    elif vision_type in ["qwen3_5_moe", "qwen3_omni_moe", "qwen3_8_vision"]:
        vision_cfg = BeeBeeVLQwen35MoeVisionModelConfig(
            **vision_dict,
            output_size=output_size,
            image_downsample_size=image_downsample_size,
            image_projector_type=image_projector_type,
            return_hidden_states=False,
            train_vision_projector=train_vision_projector,
            freeze_vision_merger=freeze_vision_merger,
        )
    else:
        raise NotImplementedError(f"vision_type: {vision_type} is not supported yet!")

    print(
        f"BeeBeeVLVisionModelConfig hidden_size={vision_cfg.hidden_size} "
        f"intermediate_size={vision_cfg.intermediate_size} out_hidden_size={getattr(vision_cfg, 'out_hidden_size', None)}"
    )
    return vision_cfg


def _compose_audio_config(
    audio_config_path: str,
    output_size: int,
    *,
    audio_downsample_size: int = 10,
    audio_projector_type: str = "channel_upscale",
    train_audio_projector: bool = True,
    audio_encoder_type: str = "whisper",
):
    raw_cfg_path = os.path.join(audio_config_path, "config.json")
    if os.path.isfile(raw_cfg_path):
        with open(raw_cfg_path, "r", encoding="utf-8") as f:
            audio_dict = json.load(f)
    else:
        base_audio_cfg = AutoConfig.from_pretrained(audio_config_path, trust_remote_code=True)
        audio_dict = base_audio_cfg.to_dict()

    # 处理嵌套 config: Qwen3-ASR 的 audio encoder config 在 thinker_config.audio_config 下
    if "thinker_config" in audio_dict and "audio_config" in audio_dict["thinker_config"]:
        audio_dict = audio_dict["thinker_config"]["audio_config"]
    audio_dict.pop("model_type", None)

    if audio_encoder_type == "qwen3_audio":
        return BeeBeeQwen3AudioModelConfig(
            **audio_dict,
            output_size=output_size,
            audio_downsample_size=audio_downsample_size,
            audio_projector_type=audio_projector_type,
            return_hidden_states=False,
            train_audio_projector=train_audio_projector,
        )

    return BeeBeeAudioModelConfig(
        **audio_dict,
        output_size=output_size,
        audio_downsample_size=audio_downsample_size,
        audio_projector_type=audio_projector_type,
        return_hidden_states=False,
        train_audio_projector=train_audio_projector,
    )


def _build_empty_omni_model(omni_config: Qwen3MoeOmniConfig, *, torch_dtype: str) -> LlavaQwen3MoeForCausalLM:
    # The wrapper looks at `foundation_config.dtype` to choose torch_dtype internally.
    _set_foundation_dtype_in_config(omni_config.foundation_config, torch_dtype)

    with init_empty_weights(), no_init_weights():
        model = LlavaQwen3MoeForCausalLM._from_config(omni_config)
    return model


def _freeze_all_except_projectors(model: LlavaQwen3MoeForCausalLM) -> None:
    # Keep LLM frozen by default (projector-only training).
    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)

    if model.image_encoder is not None and hasattr(model.image_encoder, "set_projector_trainable_only"):
        model.image_encoder.set_projector_trainable_only()
    elif model.image_encoder is not None:
        model.image_encoder.requires_grad_(False)

    if model.audio_encoder is not None and hasattr(model.audio_encoder, "set_projector_trainable_only"):
        model.audio_encoder.set_projector_trainable_only()
    elif model.audio_encoder is not None:
        model.audio_encoder.requires_grad_(False)


def build_qwen3moe_omni_from_pretrained(
    omni_model_path: str,
    *,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    attn_implementation: str = "veomni_flash_attention_2_with_sp",
    moe_implementation: Optional[Literal["eager", "fused", "fused_quack"]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
    freeze_except_projectors: bool = False,
    modality_aware_routing: bool = False,
    num_routing_modalities: int = 3,
) -> LlavaQwen3MoeForCausalLM:
    """
    Load a *composite* omni model directory created by `model.save_pretrained(...)`.

    This mode is what you use after training, or after you run a one-time "prepare" step that
    merges (base LLM + vision + whisper) into a single checkpoint.
    """
    omni_config = Qwen3MoeOmniConfig.from_pretrained(omni_model_path)

    parallel_state = get_parallel_state()
    global_rank = parallel_state.global_rank if parallel_state is not None else 0
    empty_init = init_device == "meta" or (init_device == "cpu" and global_rank != 0)
  
    _set_attn_implementation_in_config(omni_config.foundation_config, attn_implementation, moe_implementation)

    if getattr(omni_config.encoder_config, "image_config", None) is not None:
        _set_attn_implementation_in_config(omni_config.encoder_config.image_config, attn_implementation, None)
        omni_config.encoder_config.image_config.encoder_data_balance = encoder_data_balance
        omni_config.encoder_config.image_config.encoder_data_balance_sorting_algo = encoder_data_balance_sorting_algo
    if getattr(omni_config.encoder_config, "audio_config", None) is not None:
        _set_attn_implementation_in_config(omni_config.encoder_config.audio_config, attn_implementation, None)

    # Modality-aware routing: write into foundation_config so that every
    # Qwen3MoeTopKRouter.__init__ creates the modality_bias parameter.
    omni_config.foundation_config.modality_aware_routing = modality_aware_routing
    omni_config.foundation_config.num_routing_modalities = num_routing_modalities

    model = _build_empty_omni_model(omni_config, torch_dtype=torch_dtype)
    if not empty_init:
        load_model_weights(model, omni_model_path, init_device)

    if freeze_except_projectors:
        _freeze_all_except_projectors(model)
    return model


def build_qwen3moe_omni_from_components(
    foundation_config_path: str,
    foundation_weights_path: str,
    *,
    encoders: Dict[Literal["image", "audio"], Dict[str, str]],
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    attn_implementation: str = "veomni_flash_attention_2_with_sp",
    image_downsample_size: int = 8,
    image_projector_type: str = "dynamic_avgpool",
    audio_downsample_size: int = 10,
    audio_projector_type: str = "channel_upscale",
    audio_encoder_type: str = "whisper",
) -> LlavaQwen3MoeForCausalLM:
    """
    Build the omni wrapper and load weights *separately*:
      - foundation (Qwen3 MoE) weights
      - vision encoder weights
      - whisper encoder weights

    If projector weights are missing in the vision/audio checkpoints, projector parameters stay
    randomly initialized and can be trained.
    """
    parallel_state = get_parallel_state()
    global_rank = parallel_state.global_rank if parallel_state is not None else 0
    empty_init = init_device == "meta" or (init_device == "cpu" and global_rank != 0)

    foundation_cfg = AutoConfig.from_pretrained(foundation_config_path, trust_remote_code=True)
    if getattr(foundation_cfg, "foundation_config", None) is not None:
        foundation_cfg = foundation_cfg.foundation_config
   
    output_size = int(getattr(foundation_cfg, "hidden_size"))

    _set_foundation_dtype_in_config(foundation_cfg, torch_dtype, )
    _set_attn_implementation_in_config(foundation_cfg, attn_implementation, moe_implementation="fused")

    vision_cfg = None
    audio_cfg = None

    if "image" in encoders:
        enc = encoders["image"]
        vision_cfg = _compose_vision_config(
            enc["config_path"],
            output_size,
            image_downsample_size=image_downsample_size,
            image_projector_type=image_projector_type,
            train_vision_projector=True,
        )
        _set_attn_implementation_in_config(vision_cfg, attn_implementation, None)

      

    if "audio" in encoders:
        enc = encoders["audio"]
        audio_cfg = _compose_audio_config(
            enc["config_path"],
            output_size,
            audio_downsample_size=audio_downsample_size,
            audio_projector_type=audio_projector_type,
            train_audio_projector=True,
            audio_encoder_type=audio_encoder_type,
        )
        _set_attn_implementation_in_config(audio_cfg, attn_implementation, None)

    omni_config = Qwen3MoeOmniConfig(
        encoder_config={
            "image_config": vision_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
    )

   

    model = _build_empty_omni_model(omni_config, torch_dtype=torch_dtype)

    if not empty_init:
        # The wrapper exposes the foundation LLM as `model` + `lm_head`,
        # which matches the original foundation checkpoint key layout.
        print("loading model weight!!!")
        load_model_weights(model, foundation_weights_path, init_device)
        if model.image_encoder is not None and "image" in encoders and encoders["image"].get("model_path"):
            load_model_weights(model.image_encoder, encoders["image"]["model_path"], init_device)
        if model.audio_encoder is not None and "audio" in encoders and encoders["audio"].get("model_path"):
            load_model_weights(model.audio_encoder, encoders["audio"]["model_path"], init_device)

    _freeze_all_except_projectors(model)
    return model


def merge_component_models(
    vision_model_path,
    save_directory,
    vlm_path=None,
    load_vlm_components=("language", "vision"),
    language_model_path="/mnt/afs/share/Qwen3-30B-A3B-Instruct-2507-veomni-merge",
    audio_encoder_path="/mnt/afs/share/Qwen3-Omni-AudioTransformer",
    image_downsample_size=4,
    image_projector_type="dynamic_avgpool",
    audio_downsample_size=2,
    audio_projector_type="mlp_channel",
    audio_encoder_type="qwen3_audio",
):
    """Memory-efficient merge: streams weights from component checkpoints directly to
    output shards without materializing the full model.

    Peak memory ≈ one output shard (~8GB) + small encoder weights.
    """
    import gc
    import math
    from collections import OrderedDict, defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from transformers import AutoTokenizer, AutoProcessor
    from safetensors.torch import save_file as safetensors_save_file
    from veomni.utils.constants import DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_PAD_TOKEN

    SHARD_SIZE_BYTES = 8_000_000_000  # 8GB per output shard

    # Normalize load_vlm_components
    if load_vlm_components is None:
        load_vlm_components = set()
    else:
        load_vlm_components = set(load_vlm_components)

    os.makedirs(save_directory, exist_ok=True)

    # ---- Step 1: Build meta model to discover parameter names & shapes (zero memory) ----
    print("构建 meta 模型获取参数结构...")
    meta_model = build_qwen3moe_omni_from_components(
        foundation_config_path=language_model_path,
        foundation_weights_path=language_model_path,
        encoders={
            "image": {
                "config_path": vision_model_path,
                "model_path": vision_model_path,
            },
            "audio": {
                "config_path": audio_encoder_path,
                "model_path": audio_encoder_path,
            },
        },
        init_device="meta",
        torch_dtype="bfloat16",
        image_downsample_size=image_downsample_size,
        image_projector_type=image_projector_type,
        audio_downsample_size=audio_downsample_size,
        audio_projector_type=audio_projector_type,
        audio_encoder_type=audio_encoder_type,
    )
    omni_config = meta_model.omni_config

    # ---- Step 2: Tokenizer + vocab size ----
    processor = AutoProcessor.from_pretrained(vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_path, padding_side="right", use_fast=True)

    special_tokens_dict = [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_PAD_TOKEN]
    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_dict})
    print(f"Tokenizer 新增 {num_new_tokens} token，词表大小: {len(tokenizer)}")

    # Compute target vocab size (pad to multiple of 64)
    target_vocab_size = int(math.ceil(len(tokenizer) / 64.0)) * 64

    # Resize meta model embeddings to get correct param shapes
    meta_model.resize_token_embeddings(target_vocab_size, pad_to_multiple_of=64)

    # Collect all param names and shapes from meta model
    param_info = {}  # {name: (shape, dtype, nbytes)}
    for name, param in meta_model.named_parameters():
        shape = tuple(param.shape)
        dtype = param.dtype
        nbytes = param.numel() * param.element_size()
        param_info[name] = (shape, dtype, nbytes)

    print(f"模型共 {len(param_info)} 个参数，目标 vocab_size={target_vocab_size}")
    del meta_model
    gc.collect()

    # ---- Step 3: Build source indexes (header-only, no tensor data) ----
    def _build_source_index(model_path):
        """Build {key: filepath} from safetensors headers."""
        shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        index = {}
        for filepath in shard_files:
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    index[key] = filepath
        return index

    print("索引源 checkpoint...")
    foundation_index = _build_source_index(language_model_path)
    vision_index = _build_source_index(vision_model_path)
    audio_index = _build_source_index(audio_encoder_path)

    # VLM index (if provided)
    vlm_index = {}
    if vlm_path and load_vlm_components:
        vlm_index = _build_source_index(vlm_path)

    print(f"  Foundation: {len(foundation_index)} keys")
    print(f"  Vision: {len(vision_index)} keys")
    print(f"  Audio: {len(audio_index)} keys")
    if vlm_index:
        print(f"  VLM: {len(vlm_index)} keys")

    # ---- Step 4: Map each output param to its source ----
    # Source types: "foundation", "vision", "audio", "vlm", "init"
    param_source = {}  # {output_name: (source_type, source_key)}

    # VLM key remapping tables
    vlm_vision_prefix = "model.vision_tower.vision_tower."
    vlm_projector_mapping = {
        "model.mm_projector.mlp.0.bias": "image_encoder.mm_projector.mlp.0.bias",
        "model.mm_projector.mlp.0.weight": "image_encoder.mm_projector.mlp.0.weight",
        "model.mm_projector.mlp.2.bias": "image_encoder.mm_projector.mlp.2.bias",
        "model.mm_projector.mlp.2.weight": "image_encoder.mm_projector.mlp.2.weight",
    }
    # Reverse: output_key -> vlm_source_key
    vlm_projector_reverse = {v: k for k, v in vlm_projector_mapping.items()}

    for name in param_info:
        source_assigned = False

        # Check VLM first (higher priority if specified)
        if vlm_path and load_vlm_components:
            # Language from VLM
            if "language" in load_vlm_components and (
                name.startswith("model.") or name.startswith("lm_head.")
            ):
                if name in vlm_index:
                    param_source[name] = ("vlm", name)
                    source_assigned = True

            # Vision from VLM
            if not source_assigned and "vision" in load_vlm_components and name.startswith("image_encoder."):
                # Check projector remapping
                if name in vlm_projector_reverse and vlm_projector_reverse[name] in vlm_index:
                    param_source[name] = ("vlm", vlm_projector_reverse[name])
                    source_assigned = True
                else:
                    # Try direct key or remapped key
                    vlm_key = vlm_vision_prefix + name[len("image_encoder."):]
                    if vlm_key in vlm_index:
                        param_source[name] = ("vlm", vlm_key)
                        source_assigned = True
                    elif name in vlm_index:
                        param_source[name] = ("vlm", name)
                        source_assigned = True

            # Audio from VLM
            if not source_assigned and "audio" in load_vlm_components and name.startswith("audio_encoder."):
                if name in vlm_index:
                    param_source[name] = ("vlm", name)
                    source_assigned = True

        if source_assigned:
            continue

        # Default source mapping
        if name.startswith("image_encoder."):
            source_key = name[len("image_encoder."):]
            if source_key in vision_index:
                param_source[name] = ("vision", source_key)
            else:
                param_source[name] = ("init", name)
        elif name.startswith("audio_encoder."):
            source_key = name[len("audio_encoder."):]
            if source_key in audio_index:
                param_source[name] = ("audio", source_key)
            else:
                param_source[name] = ("init", name)
        elif name in foundation_index:
            param_source[name] = ("foundation", name)
        else:
            param_source[name] = ("init", name)

    # Count sources
    source_counts = defaultdict(int)
    for src_type, _ in param_source.values():
        source_counts[src_type] += 1
    print(f"参数来源: {dict(source_counts)}")

    # ---- Step 5: Plan output shards ----
    output_keys_ordered = list(param_info.keys())
    shards = []
    current_shard = []
    current_size = 0
    for name in output_keys_ordered:
        _, _, nbytes = param_info[name]
        if current_shard and current_size + nbytes > SHARD_SIZE_BYTES:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(name)
        current_size += nbytes
    if current_shard:
        shards.append(current_shard)
    print(f"输出 shard 数: {len(shards)}")

    # ---- Step 6: Helper to load tensors from sources ----

    def _load_batch_from_source(keys_with_source):
        """Load multiple tensors batched by file for efficiency."""
        file_to_entries = defaultdict(list)
        for output_name, source_type, source_key in keys_with_source:
            if source_type == "foundation":
                filepath = foundation_index[source_key]
            elif source_type == "vision":
                filepath = vision_index[source_key]
            elif source_type == "audio":
                filepath = audio_index[source_key]
            elif source_type == "vlm":
                filepath = vlm_index[source_key]
            else:
                continue
            file_to_entries[filepath].append((output_name, source_key))

        result = {}

        def _read_file(filepath, entries):
            tensors = {}
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for output_name, source_key in entries:
                    tensors[output_name] = f.get_tensor(source_key)
            return tensors

        if len(file_to_entries) <= 1:
            for filepath, entries in file_to_entries.items():
                result.update(_read_file(filepath, entries))
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(file_to_entries))) as executor:
                futures = {
                    executor.submit(_read_file, fp, entries): fp
                    for fp, entries in file_to_entries.items()
                }
                for future in as_completed(futures):
                    result.update(future.result())

        return result

    # ---- Step 7: Write each shard ----
    from tqdm import tqdm

    weight_map = {}
    total_size = 0

    for shard_idx, shard_keys in enumerate(tqdm(shards, desc="Writing shards")):
        if len(shards) == 1:
            shard_filename = "model.safetensors"
        else:
            shard_filename = f"model-{shard_idx + 1:05d}-of-{len(shards):05d}.safetensors"

        # Separate keys by source type
        keys_to_load = []  # (output_name, source_type, source_key)
        keys_to_init = []  # output_name

        for name in shard_keys:
            src_type, src_key = param_source[name]
            if src_type == "init":
                keys_to_init.append(name)
            else:
                keys_to_load.append((name, src_type, src_key))

        # Batch load from source files
        loaded = _load_batch_from_source(keys_to_load)

        # Assemble shard
        shard_state_dict = OrderedDict()
        for name in shard_keys:
            target_shape, target_dtype, _ = param_info[name]

            if name in loaded:
                tensor = loaded[name].to(target_dtype)
                # Handle vocab size mismatch (embed_tokens, lm_head)
                if tensor.shape != target_shape:
                    if name in ("model.embed_tokens.weight", "lm_head.weight"):
                        padded = torch.zeros(target_shape, dtype=target_dtype)
                        min_vocab = min(tensor.shape[0], target_shape[0])
                        padded[:min_vocab] = tensor[:min_vocab]
                        tensor = padded
                    else:
                        print(f"[WARNING] Shape mismatch: {name}: source {tensor.shape} vs target {target_shape}")
                shard_state_dict[name] = tensor
            else:
                # Random init for projectors etc.
                shard_state_dict[name] = torch.randn(target_shape, dtype=target_dtype) * 0.02

            weight_map[name] = shard_filename

        del loaded
        gc.collect()

        # Compute total size
        for tensor in shard_state_dict.values():
            total_size += tensor.numel() * tensor.element_size()

        # Save shard
        safetensors_save_file(shard_state_dict, os.path.join(save_directory, shard_filename))
        del shard_state_dict
        gc.collect()
        print(f"  Saved {shard_filename} ({len(shard_keys)} tensors)")

    # ---- Step 8: Write index + metadata ----
    if len(shards) > 1:
        import json as _json
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            _json.dump(index_data, f, indent=2, sort_keys=True)
            f.write("\n")

    # Save config, tokenizer, processor
    omni_config.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)

    total_params = sum(shape[0] * (shape[1] if len(shape) > 1 else 1) for shape, _, _ in param_info.values())
    print(f"Done! total_params≈{total_params:,}, saved to {save_directory}")



if __name__ == "__main__":
    vision_path = "/mnt/afs/share/Qwen35_A3B_vision_encoder"

    vlm_path = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/ckpt/0513_llavaomni_30A3B_qwen35encoder_puretext_lr1e4/checkpoints/hf_ckpt"
    save_directory = "/mnt/afs/share/llava_qwen235A22B_qwen35encoder_qwen3audio"

    # load_vlm_components 控制从 vlm_path 加载哪些部分:
    #   ("language", "vision")       - 只加载 LM + vision (默认)
    #   ("language",)                - 只加载 LM
    #   ("language", "vision", "audio") - 全部从 vlm_path 加载
    #   ()                           - 不从 vlm_path 加载任何权重
    merge_component_models(
        vision_path,
        save_directory,
        vlm_path=None,
        load_vlm_components=("language",),
        language_model_path="/mnt/afs/share/Qwen3-235B-A22B-Instruct-2507-veomni",
        audio_encoder_path="/mnt/afs/share/Qwen3-Omni-AudioTransformer",
        image_downsample_size=4,
        image_projector_type="dynamic_avgpool",
        audio_downsample_size=2,
        audio_projector_type="mlp_channel",
        audio_encoder_type="qwen3_audio",
    )
    