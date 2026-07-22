from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging
import json
import os
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from veomni.models.custom.llava_qwen3moe.configuration_qwen3moe_omni import Qwen3MoeOmniConfig
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
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
    elif vision_type == "qwen3_5_moe":
        vision_cfg = BeeBeeVLQwen35MoeVisionModelConfig(
            **vision_dict,
            output_size=output_size,
            image_downsample_size=image_downsample_size,
            image_projector_type=image_projector_type,
            return_hidden_states=False,
            train_vision_projector=train_vision_projector,
            freeze_vision_merger=freeze_vision_merger,
        )

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
) -> BeeBeeAudioModelConfig:
    base_audio_cfg = AutoConfig.from_pretrained(audio_config_path, trust_remote_code=True)
    audio_dict = base_audio_cfg.to_dict()
    audio_dict.pop("model_type", None)

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


def merge_component_models(vision_model_path, save_directory):
    from transformers import AutoTokenizer, AutoProcessor
    from veomni.utils.constants import DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_PAD_TOKEN
    language_model_path = "/mnt/afs/share/Qwen3-30B-A3B-Instruct-2507-veomni-merge"
    whisper_audio_encoder_path = "/mnt/afs/share/Kimi-Audio-7B-Instruct/whisper-large-v3"

    processor = AutoProcessor.from_pretrained(vision_model_path)

    print("正在加载语言模型的 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        language_model_path, 
        padding_side="right", 
        use_fast=True
    )
    print("正在构建 Omni 模型结构并加载各个 Component 权重...")
    model = build_qwen3moe_omni_from_components(
        foundation_config_path=language_model_path,
        foundation_weights_path=language_model_path,
        encoders={
            "image": {
                "config_path": vision_model_path,
                "model_path": vision_model_path,
            },
            "audio": {
                "config_path": whisper_audio_encoder_path,
                "model_path": whisper_audio_encoder_path,
            },
        },
        init_device="cuda",
        torch_dtype="bfloat16",
        image_downsample_size=4,
        image_projector_type= "dynamic_avgpool",
        audio_downsample_size = 10,
        audio_projector_type="conv_channel_upscale",
    )

    print(f"Built omni model: {type(model)}")
    print(f"image_encoder: {type(model.image_encoder) if model.image_encoder is not None else None}")
    print(f"audio_encoder: {type(model.audio_encoder) if model.audio_encoder is not None else None}")

    st1_path = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/ckpt/0513_llavaomni_30A3B_qwen35encoder_puretext_lr1e4/checkpoints/hf_ckpt"
    explicit_weights_to_load = {}
    import glob
    from safetensors.torch import load_file

        
    weight_files = glob.glob(os.path.join(st1_path, "*.safetensors"))
 
    lm_weight_prefixes = (
        "model.layers.",
        "model.embed_tokens.",
        "model.norm.",
        "lm_head."
    )
    for w_file in weight_files:
        if w_file.endswith(".safetensors"):
            state_dict = load_file(w_file)
        else:
            state_dict = torch.load(w_file, map_location="cpu")
            
        for key, tensor in state_dict.items():
            if key.startswith(lm_weight_prefixes) or key.startswith("audio_encoder."):
                explicit_weights_to_load[key] = tensor

    
    special_tokens_dict = [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_PAD_TOKEN]
    special_tokens_dict_map = {
        "additional_special_tokens": special_tokens_dict
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict_map)
    audio_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PAD_TOKEN)
    print("audio pad token", audio_token_id)

    print(f"Tokenizer 新增了 {num_new_tokens} 个 token，当前词表总大小: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    # `meta` model should still have parameter shapes; this is a quick sanity check.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params} trainable_params={trainable_params}")
    
    if explicit_weights_to_load:
        print(f"共提取到 {len(explicit_weights_to_load)} 个核心权重张量，开始注入模型...")
        missing_keys, unexpected_keys = model.load_state_dict(explicit_weights_to_load, strict=False)


    # save_directory =  "/mnt/afs/share/llava_qwen30B_A3B-qwen35encoder_veomni-down4"
    print(f"正在将组装好的模型保存至: {save_directory} ...")
   
    processor.save_pretrained(save_directory)
    
    model.save_pretrained(
        save_directory, 
        safe_serialization=True, 
        max_shard_size="8GB"     
    )
    tokenizer.save_pretrained(save_directory)
    print("模型保存完成！")



if __name__ == "__main__":
    vision_path = "/mnt/afs/share/Qwen35_A3B_vision_encoder"
    save_directory =  "/mnt/afs/share/llava_qwen30B_A3B-qwen35encoder_veomni-down4-gametext"
    merge_component_models(vision_path, save_directory)
    