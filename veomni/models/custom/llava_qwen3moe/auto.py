from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging
import json
import os
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
import glob
from safetensors.torch import load_file

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
    """Merge component models into a single omni checkpoint.

    Args:
        vision_model_path: Path to vision encoder config/weights and processor.
        save_directory: Output directory for the merged checkpoint.
        vlm_path: Optional path to a trained VLM checkpoint to load weights from.
        load_vlm_components: Which components to load from ``vlm_path``.
            A tuple/list/set containing any of:
              - ``"language"``: LM backbone (model.layers, embed_tokens, norm, lm_head)
              - ``"vision"``:   Vision encoder + projector weights
              - ``"audio"``:    Audio encoder + projector weights
            Defaults to ``("language", "vision")``.
            Set to ``()`` or ``None`` to skip vlm_path loading entirely.
        language_model_path: Path to the base language model.
        audio_encoder_path: Path to the audio encoder.
        image_downsample_size: Downsample ratio for image projector.
        image_projector_type: Type of image projector.
        audio_downsample_size: Downsample ratio for audio projector.
        audio_projector_type: Type of audio projector.
        audio_encoder_type: Type of audio encoder ("whisper" or "qwen3_audio").
    """
    from transformers import AutoTokenizer, AutoProcessor
    from veomni.utils.constants import DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_PAD_TOKEN

    # Normalize load_vlm_components
    if load_vlm_components is None:
        load_vlm_components = set()
    else:
        load_vlm_components = set(load_vlm_components)
    valid_components = {"language", "vision", "audio"}
    invalid = load_vlm_components - valid_components
    if invalid:
        raise ValueError(f"Invalid load_vlm_components: {invalid}. Valid values: {valid_components}")

    processor = AutoProcessor.from_pretrained(vision_model_path)

    print("正在加载语言模型的 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        language_model_path,
        padding_side="right",
        use_fast=True
    )

    # ---- Step 1: 构建模型结构，加载各 component 权重 ----
    print("正在构建 Omni 模型结构...")
    model = build_qwen3moe_omni_from_components(
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
        init_device="cuda",
        torch_dtype="bfloat16",
        image_downsample_size=image_downsample_size,
        image_projector_type=image_projector_type,
        audio_downsample_size=audio_downsample_size,
        audio_projector_type=audio_projector_type,
        audio_encoder_type=audio_encoder_type,
    )

    print(f"Built omni model: {type(model)}")
    print(f"image_encoder: {type(model.image_encoder) if model.image_encoder is not None else None}")
    print(f"audio_encoder: {type(model.audio_encoder) if model.audio_encoder is not None else None}")

    # ---- Step 2: 从已训练的 VLM checkpoint 按需加载权重 ----
    explicit_weights_to_load = {}

    if vlm_path and load_vlm_components:
        print(f"正在从 VLM checkpoint 加载 {load_vlm_components}: {vlm_path}")

        lm_weight_prefixes = (
            "model.layers.",
            "model.embed_tokens.",
            "model.norm.",
            "lm_head.",
        )
        vision_prefix = ("model.vision_tower.vision_tower.",)
        projector_key_mapping = {
            "model.mm_projector.mlp.0.bias": "image_encoder.mm_projector.mlp.0.bias",
            "model.mm_projector.mlp.0.weight": "image_encoder.mm_projector.mlp.0.weight",
            "model.mm_projector.mlp.2.bias": "image_encoder.mm_projector.mlp.2.bias",
            "model.mm_projector.mlp.2.weight": "image_encoder.mm_projector.mlp.2.weight",
        }
        audio_prefixes = ("audio_encoder.",)

        weight_files = glob.glob(os.path.join(vlm_path, "*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(os.path.join(vlm_path, "pytorch_model*.bin"))

        for w_file in weight_files:
            if w_file.endswith(".safetensors"):
                state_dict = load_file(w_file)
            else:
                state_dict = torch.load(w_file, map_location="cpu")

            for key, tensor in state_dict.items():
                # Vision: projector key remapping + vision encoder weights
                if "vision" in load_vlm_components:
                    if key in projector_key_mapping:
                        new_key = projector_key_mapping[key]
                        explicit_weights_to_load[new_key] = tensor
                        print(f"  Projector: {key} -> {new_key}")
                    elif key.startswith(vision_prefix) or key.startswith("image_encoder."):
                        new_key = key.replace("model.vision_tower.vision_tower.", "image_encoder.")
                        explicit_weights_to_load[new_key] = tensor

                # Language: LM backbone weights
                if "language" in load_vlm_components:
                    if key.startswith(lm_weight_prefixes):
                        explicit_weights_to_load[key] = tensor

                # Audio: audio encoder weights
                if "audio" in load_vlm_components:
                    if key.startswith(audio_prefixes):
                        explicit_weights_to_load[key] = tensor

        print(f"从 VLM 加载了 {len(explicit_weights_to_load)} 个权重 (components: {load_vlm_components})")

    # ---- Step 3: 添加 special tokens，resize embeddings ----
    special_tokens_dict = [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_PAD_TOKEN]
    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_dict})
    audio_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PAD_TOKEN)
    print(f"audio pad token: {audio_token_id}")
    print(f"Tokenizer 新增了 {num_new_tokens} 个 token，当前词表总大小: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    # ---- Step 4: 注入 VLM 权重，处理 vocab size 对齐 ----
    if explicit_weights_to_load:
        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            if key in explicit_weights_to_load:
                ckpt_tensor = explicit_weights_to_load[key]
                base_tensor = (model.model.embed_tokens.weight.data.clone()
                               if "embed_tokens" in key else model.lm_head.weight.data.clone())
                ckpt_vocab_size = ckpt_tensor.shape[0]
                model_vocab_size = base_tensor.shape[0]

                if ckpt_vocab_size != model_vocab_size:
                    max_vocab_size = max(ckpt_vocab_size, model_vocab_size)
                    hidden_size = base_tensor.shape[1]
                    print(f"对齐 {key}: ckpt({ckpt_vocab_size}) vs model({model_vocab_size}) -> {max_vocab_size}")
                    new_tensor = torch.zeros(max_vocab_size, hidden_size, dtype=base_tensor.dtype, device=base_tensor.device)
                    new_tensor[:model_vocab_size, :] = base_tensor
                    new_tensor[:ckpt_vocab_size, :] = ckpt_tensor
                    explicit_weights_to_load[key] = new_tensor

        # Resize if needed
        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            if key in explicit_weights_to_load:
                target_vocab_size = explicit_weights_to_load[key].shape[0]
                current_vocab_size = model.model.embed_tokens.weight.shape[0]
                if target_vocab_size != current_vocab_size:
                    print(f"Resize model embeddings: {current_vocab_size} -> {target_vocab_size}")
                    model.resize_token_embeddings(target_vocab_size)
                break

        print(f"共 {len(explicit_weights_to_load)} 个权重张量，注入模型...")
        missing_keys, unexpected_keys = model.load_state_dict(explicit_weights_to_load, strict=False)
        # Filter missing keys: only warn about components that were supposed to be loaded
        expected_missing_prefixes = []
        if "vision" not in load_vlm_components:
            expected_missing_prefixes.append("image_encoder.")
        if "audio" not in load_vlm_components:
            expected_missing_prefixes.append("audio_encoder.")
        if "language" not in load_vlm_components:
            expected_missing_prefixes.extend(["model.", "lm_head."])
        unexpected_missing = [
            k for k in missing_keys
            if not any(k.startswith(p) for p in expected_missing_prefixes)
        ]
        if unexpected_missing:
            print(f"[WARNING] unexpected missing keys: {unexpected_missing[:20]}")
        print("权重注入完成！")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params:,} trainable_params={trainable_params:,}")

    # ---- Step 5: 保存 ----
    print(f"保存至: {save_directory}")
    processor.save_pretrained(save_directory)
    model.save_pretrained(save_directory, safe_serialization=True, max_shard_size="8GB")
    tokenizer.save_pretrained(save_directory)
    print("模型保存完成！")



if __name__ == "__main__":
    vision_path = "/mnt/afs/share/Qwen38_27B_vision_encoder"

    vlm_path = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/ckpt/0513_llavaomni_30A3B_qwen35encoder_puretext_lr1e4/checkpoints/hf_ckpt"
    save_directory = "/mnt/afs/share/llava_qwen30A3B_qwen38encoder_qwen3audio_gametext"

    # load_vlm_components 控制从 vlm_path 加载哪些部分:
    #   ("language", "vision")       - 只加载 LM + vision (默认)
    #   ("language",)                - 只加载 LM
    #   ("language", "vision", "audio") - 全部从 vlm_path 加载
    #   ()                           - 不从 vlm_path 加载任何权重
    merge_component_models(
        vision_path,
        save_directory,
        vlm_path=vlm_path,
        load_vlm_components=("language",),
        language_model_path="/mnt/afs/share/Qwen3-30B-A3B-Instruct-2507-veomni-merge",
        audio_encoder_path="/mnt/afs/share/Qwen3-Omni-AudioTransformer",
        image_downsample_size=4,
        image_projector_type="dynamic_avgpool",
        audio_downsample_size=2,
        audio_projector_type="mlp_channel",
        audio_encoder_type="qwen3_audio",
    )
    