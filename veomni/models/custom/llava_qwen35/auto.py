from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging
import json
import os
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
import glob
from safetensors import safe_open

from veomni.models.custom.llava_qwen35.configuration_qwen35_omni import Qwen35OmniConfig
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
from veomni.models.custom.llava_qwen3moe.modeling_qwen3_audio_encoder import BeeBeeQwen3AudioModelConfig
from veomni.models.custom.llava_qwen35.modeling_llava_qwen35_omni import LlavaQwen35ForCausalLM
from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import BeeBeeVLQwen35MoeVisionModelConfig
from veomni.models.module_utils import init_empty_weights, load_model_weights
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
from veomni.utils.device import is_torch_npu_available
from veomni.utils.logging import get_logger
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import _bind_veomni_ops
from veomni.ops import apply_ops_config
from veomni.ops.kernels.moe import apply_veomni_fused_moe_patch

if is_transformers_version_greater_or_equal_to("5.0.0"):
    from transformers.initialization import no_init_weights
else:
    from transformers.modeling_utils import no_init_weights

logger = get_logger(__name__)


def _get_patched_module(omni_config: Qwen35OmniConfig):
    """Return the appropriate patched modeling module based on the foundation type."""
    if omni_config.is_moe:
        from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as _module
    else:
        from veomni.models.transformers.qwen3_5.generated import patched_modeling_qwen3_5_gpu as _module
    return _module


def _install_veomni_qwen35_ops(ops_implementation: OpsImplementationConfig, omni_config: Qwen35OmniConfig) -> None:
    """Modern replacement for the old manual patch approach.

    1. ``apply_ops_config`` installs the CE kernel into LOSS_MAPPING, binds
       GLOBAL ops (e.g. load-balancing loss), and populates the ops-config
       singleton. Idempotent — safe to call twice.
    2. ``_bind_veomni_ops`` walks the patchgen'd modeling module and binds
       every OpSlot (rms_norm / rotary_pos_emb / swiglu_mlp / cross_entropy_loss,
       and for MoE: moe_experts / load_balancing_loss).
    """
    apply_ops_config(ops_implementation)
    _patched_module = _get_patched_module(omni_config)
    _bind_veomni_ops(_patched_module, ops_implementation)
    if omni_config.is_moe:
        apply_veomni_fused_moe_patch(fused_moe_kernel=getattr(ops_implementation, "moe_implementation", None))


def _legacy_apply_fused_moe_only(moe_implementation: Optional[str]) -> None:
    """Fallback for callers that pass only the legacy ``moe_implementation``
    string (no full ``OpsImplementationConfig``). Preserves the pre-refactor
    behaviour for the merge scripts.

    Accepted values mirror ``OpsImplementationConfig.moe_implementation``:
    - ``"eager"``               → skip (no fused kernel installed)
    - ``"fused_triton"``        → ``triton``   (GPU SM70+, A100/H-series)
    - ``"fused_quack"``         → ``quack``    (GPU SM90+, Hopper/Blackwell)
    - ``"fused_npu"``           → ``npu``      (Ascend NPU)
    - ``"fused"`` (deprecated)  → hardware-resolved: NPU→``npu``, GPU→``quack``
    """
    if moe_implementation is None or moe_implementation == "eager":
        return
    if moe_implementation == "fused":
        moe_implementation = "fused_npu" if is_torch_npu_available() else "fused_quack"
        logger.warning_rank0(
            f"[llava_qwen35] moe_implementation='fused' is a deprecated alias; "
            f"resolving to '{moe_implementation}' on this host."
        )
    prefix = "fused_"
    if not moe_implementation.startswith(prefix):
        raise ValueError(
            f"Invalid moe_implementation: {moe_implementation!r}. Expected one of: "
            f"'eager', 'fused', 'fused_triton', 'fused_quack', 'fused_npu'."
        )
    fused_moe_kernel = moe_implementation[len(prefix):]  # 'triton' | 'quack' | 'npu'
    logger.info_rank0(
        f"[llava_qwen35] Legacy fused-MoE-only path: moe_implementation={moe_implementation} "
        f"(fused_moe_kernel={fused_moe_kernel}). Pass ops_implementation=OpsImplementationConfig(...) "
        f"to also bind rms_norm/rotary_pos_emb/swiglu_mlp/cross_entropy_loss kernels."
    )
    apply_veomni_fused_moe_patch(fused_moe_kernel=fused_moe_kernel)


def _set_attn_implementation_in_config(config: PretrainedConfig, attn_implementation: str) -> None:
    setattr(config, "_attn_implementation", attn_implementation)


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
        base_vision_cfg = AutoConfig.from_pretrained(vision_config_path, trust_remote_code=True)
        print("vision config", base_vision_cfg)
        vision_dict = base_vision_cfg.to_dict()
        if vision_dict.get("hidden_size", None) is None and getattr(base_vision_cfg, "text_config", None) is not None:
            vision_dict["hidden_size"] = getattr(base_vision_cfg.text_config, "hidden_size", None)
        if vision_dict.get("intermediate_size", None) is None and getattr(base_vision_cfg, "text_config", None) is not None:
            vision_dict["intermediate_size"] = getattr(base_vision_cfg.text_config, "intermediate_size", None)

    vision_type = vision_dict.pop("model_type", None)

    if vision_type == "qwen2_5_vl":
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


def _build_empty_omni_model(omni_config: Qwen35OmniConfig, *, torch_dtype: str) -> LlavaQwen35ForCausalLM:
    _set_foundation_dtype_in_config(omni_config.foundation_config, torch_dtype)

    with init_empty_weights(), no_init_weights():
        model = LlavaQwen35ForCausalLM._from_config(omni_config)
    return model


def _freeze_all_except_projectors(model: LlavaQwen35ForCausalLM) -> None:
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


def build_qwen35_omni_from_pretrained(
    omni_model_path: str,
    *,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    attn_implementation: str = "veomni_flash_attention_2_with_sp",
    moe_implementation: Optional[Literal["eager", "fused", "fused_triton", "fused_quack", "fused_npu"]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
    freeze_except_projectors: bool = False,
    ops_implementation: Optional[OpsImplementationConfig] = None,
) -> LlavaQwen35ForCausalLM:
    """
    Load a *composite* omni model directory created by `model.save_pretrained(...)`.

    This mode is what you use after training, or after you run a one-time "prepare" step that
    merges (base LLM + vision + whisper) into a single checkpoint.
    """
    omni_config = Qwen35OmniConfig.from_pretrained(omni_model_path)

    parallel_state = get_parallel_state()
    global_rank = parallel_state.global_rank if parallel_state is not None else 0
    empty_init = init_device == "meta" or (init_device == "cpu" and global_rank != 0)

    # Modern path: full OpsImplementationConfig binds every OpSlot.
    # Legacy path: just MoE (backward compat with callers that still pass moe_implementation).
    if ops_implementation is not None:
        attn_implementation = ops_implementation.attn_implementation
        _install_veomni_qwen35_ops(ops_implementation, omni_config)
    else:
        if omni_config.is_moe:
            _legacy_apply_fused_moe_only(moe_implementation)

    _set_attn_implementation_in_config(omni_config.foundation_config, attn_implementation)

    if getattr(omni_config.encoder_config, "image_config", None) is not None:
        _set_attn_implementation_in_config(omni_config.encoder_config.image_config, attn_implementation)
        omni_config.encoder_config.image_config.encoder_data_balance = encoder_data_balance
        omni_config.encoder_config.image_config.encoder_data_balance_sorting_algo = encoder_data_balance_sorting_algo
    if getattr(omni_config.encoder_config, "audio_config", None) is not None:
        _set_attn_implementation_in_config(omni_config.encoder_config.audio_config, attn_implementation)

    model = _build_empty_omni_model(omni_config, torch_dtype=torch_dtype)
    if not empty_init:
        load_model_weights(model, omni_model_path, init_device)

    if freeze_except_projectors:
        _freeze_all_except_projectors(model)
    return model


def build_qwen35_omni_from_components(
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
    ops_implementation: Optional[OpsImplementationConfig] = None,
) -> LlavaQwen35ForCausalLM:
    """
    Build the omni wrapper and load weights *separately*:
      - foundation (Qwen3.5 or Qwen3.5 MoE) weights
      - vision encoder weights
      - audio encoder weights

    If projector weights are missing in the vision/audio checkpoints, projector parameters stay
    randomly initialized and can be trained.
    """
    parallel_state = get_parallel_state()
    global_rank = parallel_state.global_rank if parallel_state is not None else 0
    empty_init = init_device == "meta" or (init_device == "cpu" and global_rank != 0)

    foundation_cfg = AutoConfig.from_pretrained(foundation_config_path, trust_remote_code=True)
    if getattr(foundation_cfg, "foundation_config", None) is not None:
        foundation_cfg = foundation_cfg.foundation_config
    # Qwen3_5Config (multimodal wrapper, model_type="qwen3_5") nests the text LLM
    # config under text_config; unwrap it so downstream code sees vocab_size etc.
    if getattr(foundation_cfg, "text_config", None) is not None and not hasattr(foundation_cfg, "vocab_size"):
        foundation_cfg = foundation_cfg.text_config

    output_size = int(getattr(foundation_cfg, "hidden_size"))

    _set_foundation_dtype_in_config(foundation_cfg, torch_dtype)
    _set_attn_implementation_in_config(foundation_cfg, attn_implementation)
   
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
        _set_attn_implementation_in_config(vision_cfg, attn_implementation)

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
        _set_attn_implementation_in_config(audio_cfg, attn_implementation)

    omni_config = Qwen35OmniConfig(
        encoder_config={
            "image_config": vision_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
    )

    # Install ops if provided (after config is composed so is_moe resolves correctly)
    if ops_implementation is not None:
        _install_veomni_qwen35_ops(ops_implementation, omni_config)

    model = _build_empty_omni_model(omni_config, torch_dtype=torch_dtype)

    if not empty_init:
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
    meta_model = build_qwen35_omni_from_components(
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

    # Compute target vocab size (pad to multiple of 64), but never shrink below the
    # original model's vocab_size to avoid truncating existing embeddings.
    original_vocab_size = omni_config.foundation_config.vocab_size
    target_vocab_size = int(math.ceil(len(tokenizer) / 64.0)) * 64
    if original_vocab_size is not None and original_vocab_size > target_vocab_size:
        target_vocab_size = int(math.ceil(original_vocab_size / 64.0)) * 64
        print(f"[INFO] 保留原始 vocab_size={original_vocab_size}，target_vocab_size 上调至 {target_vocab_size}")

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
    param_source = {}  # {output_name: (source_type, source_key)}

    # VLM key remapping tables
    vlm_vision_prefix = "model.vision_tower.vision_tower."
    vlm_projector_mapping = {
        "model.mm_projector.mlp.0.bias": "image_encoder.mm_projector.mlp.0.bias",
        "model.mm_projector.mlp.0.weight": "image_encoder.mm_projector.mlp.0.weight",
        "model.mm_projector.mlp.2.bias": "image_encoder.mm_projector.mlp.2.bias",
        "model.mm_projector.mlp.2.weight": "image_encoder.mm_projector.mlp.2.weight",
    }
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
                if name in vlm_projector_reverse and vlm_projector_reverse[name] in vlm_index:
                    param_source[name] = ("vlm", vlm_projector_reverse[name])
                    source_assigned = True
                else:
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
            # Qwen3.5 original checkpoints use "model.language_model.layers.X..."
            # while our omni model uses "model.layers.X..." — try the remapped key.
            remapped = None
            if name.startswith("model."):
                remapped = "model.language_model." + name[len("model."):]
            if remapped and remapped in foundation_index:
                param_source[name] = ("foundation", remapped)
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

    processor.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    total_params = sum(shape[0] * (shape[1] if len(shape) > 1 else 1) for shape, _, _ in param_info.values())
    print(f"Done! total_params≈{total_params:,}, saved to {save_directory}")


if __name__ == "__main__":
    vision_path = "/mnt/afs/share/Qwen38_27B_vision_encoder"

    vlm_path = None
    save_directory = "/mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base"

    merge_component_models(
        vision_path,
        save_directory,
        vlm_path=None,
        load_vlm_components=("language",),
        language_model_path="/mnt/afs/share/Qwen3.8-27B",
        audio_encoder_path="/mnt/afs/share/Qwen3-Omni-AudioTransformer",
        image_downsample_size=4,
        image_projector_type="dynamic_avgpool",
        audio_downsample_size=2,
        audio_projector_type="mlp_channel",
        audio_encoder_type="qwen3_audio",
    )
