from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging
import json
import os
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoProcessor
import glob
from safetensors.torch import load_file

from veomni.models.custom.llava_qwen2.configuration_llava_qwen2 import LlavaQwen2Config
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
from veomni.models.custom.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM
from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import BeeBeeVLQwen35MoeVisionModelConfig
from veomni.models.module_utils import init_empty_weights, load_model_weights
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
from veomni.models.transformers.qwen2.gpu_patch import apply_veomni_qwen2_gpu_patch
from veomni.utils.logging import get_logger

if is_transformers_version_greater_or_equal_to("5.0.0"):
    from transformers.initialization import no_init_weights
else:
    from transformers.modeling_utils import no_init_weights

logger = get_logger(__name__)

def _set_attn_implementation_in_config(config: PretrainedConfig, attn_implementation: str) -> None:
    # The custom omni wrapper forwards `config._attn_implementation` into submodules.
    setattr(config, "_attn_implementation", attn_implementation)
    apply_veomni_qwen2_gpu_patch()
    
      
        

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


def _build_empty_omni_model(omni_config: LlavaQwen2Config, *, torch_dtype: str) -> LlavaQwen2ForCausalLM:
    # The wrapper looks at `foundation_config.dtype` to choose torch_dtype internally.
    _set_foundation_dtype_in_config(omni_config.foundation_config, torch_dtype)

    with init_empty_weights(), no_init_weights():
        model = LlavaQwen2ForCausalLM._from_config(omni_config)
    return model


def _freeze_all_except_projectors(model: LlavaQwen2ForCausalLM) -> None:
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


def build_llavaqwen2_omni_from_pretrained(
    omni_model_path: str,
    *,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    attn_implementation: str = "veomni_flash_attention_2_with_sp",
    # moe_implementation: Optional[Literal["eager", "fused", "fused_quack"]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
    freeze_except_projectors: bool = True,
) -> LlavaQwen2ForCausalLM:
    """
    Load a *composite* omni model directory created by `model.save_pretrained(...)`.

    This mode is what you use after training, or after you run a one-time "prepare" step that
    merges (base LLM + vision + whisper) into a single checkpoint.
    """
    omni_config = LlavaQwen2Config.from_pretrained(omni_model_path)

    parallel_state = get_parallel_state()
    global_rank = parallel_state.global_rank if parallel_state is not None else 0
    empty_init = init_device == "meta" or (init_device == "cpu" and global_rank != 0)
  
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


def build_qwen25_omni_from_components(
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
) -> LlavaQwen2ForCausalLM:
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
        )
        _set_attn_implementation_in_config(audio_cfg, attn_implementation)

    omni_config = LlavaQwen2Config(
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


def merge_component_models(vision_model_path, save_directory, load_mm_projector=True):
   
    from veomni.utils.constants import DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_PAD_TOKEN
    language_model_path = "/mnt/afs/share/Qwen25-14B-Instruct"
    whisper_audio_encoder_path = "/mnt/afs/share/Kimi-Audio-7B-Instruct/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(vision_model_path)
    print("正在加载语言模型的 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        language_model_path, 
        padding_side="right", 
        use_fast=False
    )
    model = build_qwen25_omni_from_components(
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
        image_downsample_size= 16,
        image_projector_type= "dynamic_avgpool",
        audio_downsample_size = 10,
        audio_projector_type="conv_channel_upscale",
    )

    print(f"Built omni model: {type(model)}")
    print(f"image_encoder: {type(model.image_encoder) if model.image_encoder is not None else None}")
    print(f"audio_encoder: {type(model.audio_encoder) if model.audio_encoder is not None else None}")
    
    lm_weight_prefixes = (
        "model.layers.",
        "model.embed_tokens.",
        "model.norm.",
        "lm_head."
    )
    vision_prefix = ("model.vision_tower.vision_tower.", )
    if load_mm_projector:
        language_model_path = "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/checkpoints/omni_models/0728_llava_omni_qwen25vl_14B_16x_4k_st2_kimiwhisper_10x_unfreezeaudio_omnidata_text500w_lr2e-6"
        print("正在迁移原模型 mm_projector 权重...")

        projector_key_mapping = {
            "model.mm_projector.mlp.0.bias": "image_encoder.mm_projector.mlp.0.bias",
            "model.mm_projector.mlp.0.weight": "image_encoder.mm_projector.mlp.0.weight",
            "model.mm_projector.mlp.2.bias": "image_encoder.mm_projector.mlp.2.bias",
            "model.mm_projector.mlp.2.weight": "image_encoder.mm_projector.mlp.2.weight",
            "model.vision_tower.vision_tower.": "image_encoder."
        }
    
        explicit_weights_to_load = {}
        
        weight_files = glob.glob(os.path.join(language_model_path, "*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(os.path.join(language_model_path, "pytorch_model*.bin"))
            
        for w_file in weight_files:
            if w_file.endswith(".safetensors"):
                state_dict = load_file(w_file)
            else:
                state_dict = torch.load(w_file, map_location="cpu")
                
            for key, tensor in state_dict.items():
           
                if key in projector_key_mapping:
                    new_key = projector_key_mapping[key]
                    explicit_weights_to_load[new_key] = tensor
                    print(f"  已提取并重命名 Projector 权重: {key} -> {new_key}")
                

                elif key.startswith(lm_weight_prefixes):
                    explicit_weights_to_load[key] = tensor

                elif key.startswith(vision_prefix):
                    new_key = key.replace("model.vision_tower.vision_tower.", "image_encoder.")
                    explicit_weights_to_load[new_key] = tensor

        if explicit_weights_to_load:
            print(f"共提取到 {len(explicit_weights_to_load)} 个核心权重张量，开始注入模型...")
            missing_keys, unexpected_keys = model.load_state_dict(explicit_weights_to_load, strict=False)
            print("语言模型主干及 mm_projector 权重显式加载成功！")
        else:
            print("警告: 未从指定路径中提取到任何匹配的权重，请确认权重文件内的 Key 是否正确。")
        # ----------------------------------------------------------------------------------------
    
    special_tokens_dict_map = {
        "additional_special_tokens": [DEFAULT_AUDIO_PAD_TOKEN, DEFAULT_AUDIO_START_TOKEN,DEFAULT_AUDIO_END_TOKEN]
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict_map)
    audio_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PAD_TOKEN)
    print(f"当前所有特殊 tokens: {tokenizer.all_special_tokens}")
    print("audio pad token", audio_token_id)

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    
    print(f"Tokenizer 新增了 {num_new_tokens} 个 token，当前词表总大小: {len(tokenizer)}")
    # `meta` model should still have parameter shapes; this is a quick sanity check.
   
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params} trainable_params={trainable_params}")
    # save_directory =  "/mnt/afs/share/llava_qwen2_14B-qwen35encoder-veomni-down4"
    print(f"正在将组装好的模型保存至: {save_directory} ...")
   
    processor.save_pretrained(save_directory)
    
    
    # save_pretrained 会自动保存模型权重和配置 config.json
    model.save_pretrained(
        save_directory, 
        safe_serialization=True, # 推荐使用 safetensors 格式，加载更快且更安全
        max_shard_size="5GB"     # 因为 30B 模型很大，建议分块保存
    )
    tokenizer.save_pretrained(save_directory)
    print("模型保存完成！")



if __name__ == "__main__":
    vision_path = "/mnt/afs/share/qwen25_vl_encoder"
    save_directory =  "/mnt/afs/share/llava_qwen2_14B-qwen25encoder-st4-veomni-down16"
    merge_component_models(vision_path, save_directory)

    