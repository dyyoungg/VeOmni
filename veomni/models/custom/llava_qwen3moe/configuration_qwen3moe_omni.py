from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from transformers import AutoConfig, PretrainedConfig


def _init_config(config_dict: Optional[Dict[str, Any] | PretrainedConfig]) -> Optional["PretrainedConfig"]:
    """
    Initialize a Hugging Face PretrainedConfig from a plain dictionary using AutoConfig.
    Returns a bare PretrainedConfig if input is None or the model_type is empty.
    """
    if config_dict is None:
        return PretrainedConfig()
    if isinstance(config_dict, PretrainedConfig):
        return config_dict

    config_copy = deepcopy(config_dict)
    model_type = config_copy.pop("model_type", "")
    if model_type == "":
        return PretrainedConfig()
    return AutoConfig.for_model(model_type, **config_copy)


class Qwen3MoeOmniEncoderConfig(PretrainedConfig):
    model_type = "qwen3moe_omni_encoder"
    sub_configs = {
        "image_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        image_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        encode_input: bool = True,
        encode_output: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.image_config = _init_config(image_config)
        self.audio_config = _init_config(audio_config)
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Qwen3MoeOmniConfig(PretrainedConfig):
    """Top-level config tying together encoders and the foundation LLM.

    This omni variant intentionally omits any decoder component. It only contains:
    - image encoder
    - audio encoder
    - foundation LLM (text backbone)
    """

    model_type = "llavaqwen3moe_omni"
    sub_configs = {
        "encoder_config": AutoConfig,
        "foundation_config": AutoConfig,
    }

    def __init__(
        self,
        encoder_config: Dict[Literal["image_config", "audio_config"], Dict[str, Any]] = {},
        foundation_config: Optional[Dict[str, Any]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Compose sub-configs
        self.encoder_config = Qwen3MoeOmniEncoderConfig(**encoder_config)
        self.foundation_config = _init_config(foundation_config)
        self.initializer_range = initializer_range

        # Default architecture name communicates expected top-level class
        super().__init__(
            architectures=kwargs.pop("architectures", "LlavaQwen3MoeForCausalLM"),
            **kwargs,
        )

    def get_text_config(self) -> PretrainedConfig:
        return self.foundation_config

AutoConfig.register("llavaqwen3moe_omni", Qwen3MoeOmniConfig)


if __name__ == "__main__":
    # Example: compose the omni config from your encoders and Qwen3Moe foundation.
    from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
    from transformers import Qwen3MoeConfig

    # 1 Instantiate encoder sub-configs directly (recommended)
    image_cfg = BeeBeeVLVisionModelConfig(
        spatial_merge_size=2,
        return_hidden_states=False,
        train_vision_projector=True,
        image_downsample_size=8,
        mm_projector_type="dynamic_avgpool",
        output_size=6144,
    )

    audio_cfg = BeeBeeAudioModelConfig(
        return_hidden_states=False,
        train_audio_projector=True,
        audio_downsample_size=10,
        audio_projector_type="channel_upscale",
        output_size=6144,
    )

    # 2 Instantiate foundation LLM config (Qwen3 Moe)
    foundation_cfg = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=4,
        use_cache=True,
        tie_word_embeddings=False,
    )

    # 3 Build the top-level omni config
    omni_cfg = Qwen3MoeOmniConfig(
        encoder_config={
            "image_config": image_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
        initializer_range=0.02,
    )

    # Quick sanity prints
    print("Omni model_type:", omni_cfg.model_type)
    print("Image encoder hidden size:", getattr(omni_cfg.encoder_config.image_config, "hidden_size", None))
    print("Audio encoder output size:", getattr(omni_cfg.encoder_config.audio_config, "output_size", None))
    print("Foundation vocab size:", omni_cfg.foundation_config.vocab_size)


