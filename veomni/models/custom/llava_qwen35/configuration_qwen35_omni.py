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


class Qwen35OmniEncoderConfig(PretrainedConfig):
    model_type = "qwen35_omni_encoder"
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


class Qwen35OmniConfig(PretrainedConfig):
    """Top-level config tying together encoders and the foundation LLM.

    This omni variant supports both dense Qwen3.5 and Qwen3.5 MoE as the
    foundation LLM. It only contains:
    - image encoder
    - audio encoder
    - foundation LLM (text backbone, dense or MoE)
    """

    model_type = "llavaqwen35_omni"
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
        self.encoder_config = Qwen35OmniEncoderConfig(**encoder_config)
        self.foundation_config = _init_config(foundation_config)
        self.initializer_range = initializer_range

        # Default architecture name communicates expected top-level class.
        # transformers>=5 validates ``architectures`` as list[str] | None,
        # so wrap a bare default and normalize legacy str inputs from ckpts.
        architectures = kwargs.pop("architectures", ["LlavaQwen35ForCausalLM"])
        if isinstance(architectures, str):
            architectures = [architectures]
        super().__init__(
            architectures=architectures,
            **kwargs,
        )

    @property
    def is_moe(self) -> bool:
        """Whether the foundation LLM is a MoE model."""
        model_type = getattr(self.foundation_config, "model_type", "")
        return "moe" in model_type.lower()

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        # transformers>=5 / huggingface_hub validators call this with
        # ``decoder=``/``encoder=`` keyword args. We only host a text
        # foundation LLM (no separate encoder/decoder), so ignore them
        # and always return the foundation config.
        return self.foundation_config


AutoConfig.register("llavaqwen35_omni", Qwen35OmniConfig)


if __name__ == "__main__":
    # Example: compose the omni config from your encoders and Qwen3.5 foundation.
    from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
    from transformers import Qwen3_5MoeConfig

    # 1 Instantiate encoder sub-configs directly (recommended)
    image_cfg = BeeBeeVLVisionModelConfig(
        spatial_merge_size=2,
        return_hidden_states=False,
        train_vision_projector=True,
        image_downsample_size=8,
        image_projector_type="dynamic_avgpool",
        output_size=6144,
    )

    audio_cfg = BeeBeeAudioModelConfig(
        return_hidden_states=False,
        train_audio_projector=True,
        audio_downsample_size=10,
        audio_projector_type="channel_upscale",
        output_size=6144,
    )

    # 2 Instantiate foundation LLM config (Qwen3.5 MoE)
    foundation_cfg = Qwen3_5MoeConfig(
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
    omni_cfg = Qwen35OmniConfig(
        encoder_config={
            "image_config": image_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
        initializer_range=0.02,
    )

    # Quick sanity prints
    print("Omni model_type:", omni_cfg.model_type)
    print("Is MoE:", omni_cfg.is_moe)
    print("Image encoder hidden size:", getattr(omni_cfg.encoder_config.image_config, "hidden_size", None))
    print("Audio encoder output size:", getattr(omni_cfg.encoder_config.audio_config, "output_size", None))
    print("Foundation vocab size:", omni_cfg.foundation_config.vocab_size)
