from typing import Optional
import json

from transformers import AutoConfig, AutoModel, GenerationMixin, PreTrainedModel
import torch
import torch.nn as nn

from veomni.models.custom.llava_qwen3moe.configuration_qwen3moe_omni import Qwen3MoeOmniConfig
from veomni.models.custom.llava_qwen3moe.modeling_vision_encoder import BeeBeeVLVisionModel
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeVLAudioModel
from veomni.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from veomni.distributed.parallel_plan import ParallelPlan


class Qwen3MoeOmniPreTrainedModel(PreTrainedModel):
    config_class = Qwen3MoeOmniConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @property
    def _no_split_modules(self):
        no_split_modules = []
        for module in self.children():
            if isinstance(module, PreTrainedModel) and module._no_split_modules:
                no_split_modules.extend(module._no_split_modules)
            elif isinstance(module, nn.ModuleDict):
                for sub_module in module.children():
                    if isinstance(sub_module, PreTrainedModel) and sub_module._no_split_modules:
                        no_split_modules.extend(sub_module._no_split_modules)

        return no_split_modules

    @_no_split_modules.setter
    def _no_split_modules(self, value):
        pass

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



class LlavaQwen3MoeForCausalLM(Qwen3MoeOmniPreTrainedModel, GenerationMixin):
    """
    Qwen3 MoE causal LM augmented with image/audio encoders.

    It expects a `Qwen3MoeOmniConfig` at construction time. The underlying LLM is
    initialized from `foundation_config`. The image/audio encoders are created
    from `encoder_config`'s sub-configs.
    """
    def __init__(self, config: Qwen3MoeOmniConfig):
        # Initialize base CausalLM with foundation (text) config
        super().__init__(config.foundation_config)
        dtype = getattr(config.foundation_config, "dtype", None)
        print("dtype", dtype)
        if dtype is None or dtype==torch.bfloat16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.foundation = Qwen3MoeForCausalLM._from_config(config.foundation_config, 
                                                           attn_implementation=config.foundation_config._attn_implementation,
                                                           dtype=torch_dtype)
        # Keep a reference to the omni config
        self.omni_config = config
        # Optional multimodal encoders
        self.image_encoder: Optional[nn.Module] = None
        self.audio_encoder: Optional[nn.Module] = None

        encoder_cfg = getattr(config, "encoder_config", None)
        
       
        print("torch dtype", torch_dtype)
        if encoder_cfg is not None:
            if getattr(encoder_cfg, "image_config", None) is not None:
                self.image_encoder = BeeBeeVLVisionModel._from_config(config=encoder_cfg.image_config, 
                                                           attn_implementation=encoder_cfg.image_config._attn_implementation, 
                                                           dtype=torch_dtype)

            if getattr(encoder_cfg, "audio_config", None) is not None:
                self.audio_encoder = BeeBeeVLAudioModel._from_config(encoder_cfg.audio_config, 
                                                          attn_implementation=encoder_cfg.audio_config._attn_implementation, 
                                                          dtype=torch_dtype)
    # Convenience accessors
    def get_text_config(self):
        return self.omni_config.get_text_config()

    def set_image_encoder_trainable_only(self):
        if self.image_encoder is not None and hasattr(self.image_encoder, "set_projector_trainable_only"):
            self.image_encoder.set_projector_trainable_only()

    def set_audio_encoder_trainable_only(self):
        if self.audio_encoder is not None and hasattr(self.audio_encoder, "set_projector_trainable_only"):
            self.audio_encoder.set_projector_trainable_only()

    def get_input_embeddings(self):
        return self.foundation.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.foundation.set_input_embeddings()

    def get_output_embeddings(self):
        return self.foundation.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.foundation.set_output_embeddings(new_embeddings)

    def get_parallel_plan(self):
        parallel_plan: ParallelPlan = self.foundation.get_parallel_plan()
        parallel_plan.update_prefix("foundation")
        return parallel_plan


    def forward():
        pass
    





    


if __name__ == "__main__":
    # Minimal demo: build from config and print model structure
    from veomni.models.custom.llava_qwen3moe.modeling_vision_encoder import BeeBeeVLVisionModelConfig
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
    from veomni.models.custom.llava_qwen3moe.configuration_qwen3moe_omni import Qwen3MoeOmniConfig

    # Encoder configs
    image_cfg = BeeBeeVLVisionModelConfig(
        spatial_merge_size=2,
        return_hidden_states=False,
        train_vision_projector=True,
        image_downsample_size=8,
        image_projector_type="dynamic_avgpool",
        output_size=6144,
    )
    imag
    print(image_cfg)

    audio_cfg = BeeBeeAudioModelConfig(
        return_hidden_states=False,
        train_audio_projector=True,
        audio_downsample_size=10,
        audio_projector_type="channel_upscale",
        output_size=6144,
    )

    print(audio_cfg)
    # Foundation LLM config
    llm_config_path = "/mnt/afs/share/Qwen3-30B-A3B-Instruct-2507-veomni-merge"
    foundation_cfg = AutoConfig.from_pretrained(llm_config_path)

    # Compose omni config
    omni_cfg = Qwen3MoeOmniConfig(
        encoder_config={
            "image_config": image_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
    )

    print(omni_cfg)
    # Build full model from config and print structure
    model = LlavaQwen3MoeForCausalLM._from_config(omni_cfg)
    print(model)
    print("Has image encoder:", isinstance(model.image_encoder, nn.Module))
    print("Has audio encoder:", isinstance(model.audio_encoder, nn.Module))

    # Save and show serialized omni config
    save_dir = "./tmp_qwen3moe_omni_config"
    omni_cfg.save_pretrained(save_dir)
    print(f"Saved omni config to {save_dir}/config.json")




