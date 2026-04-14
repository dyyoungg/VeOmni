from typing import Optional
import json
import contextlib

from transformers import AutoConfig, AutoModel, GenerationMixin, PreTrainedModel
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import can_return_tuple
from transformers import Qwen2ForCausalLM

from veomni.models.custom.llava_qwen2.configuration_llava_qwen2 import LlavaQwen2Config
from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModel
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import BeeBeeVLQwen35MoeVisionModel
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeVLAudioModel
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads


class LlavaQwen2PreTrainedModel(PreTrainedModel):
    config_class = LlavaQwen2Config
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



class LlavaQwen2ForCausalLM(LlavaQwen2PreTrainedModel, GenerationMixin):

    def __init__(self, config: LlavaQwen2Config):
        # Initialize base CausalLM with foundation (text) config
        # Keep the full omni config so `save_pretrained()` can persist encoder configs too.
        super().__init__(config.foundation_config)
        self.foundation_config = config.foundation_config
        # `load_model_weights()` ties embeddings based on `model.config.tie_word_embeddings`.
        # Ensure it matches the foundation LLM setting.
        self.config.tie_word_embeddings = getattr(self.foundation_config, "tie_word_embeddings", True)

        dtype = getattr(config.foundation_config, "dtype", None)
        if dtype is None or dtype == torch.bfloat16 or dtype == "bfloat16" or dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        foundation_llm = Qwen2ForCausalLM._from_config(
            self.foundation_config,
            attn_implementation=self.foundation_config._attn_implementation,
            dtype=torch_dtype,
        )
        self.model = foundation_llm.model
        self.vocab_size = foundation_llm.vocab_size
        self.lm_head = foundation_llm.lm_head
        del foundation_llm
        # Keep a reference to the omni config
        self.omni_config = config
        # Optional multimodal encoders
        self.image_encoder: Optional[nn.Module] = None
        self.audio_encoder: Optional[nn.Module] = None

        encoder_cfg = getattr(config, "encoder_config", None)
        
        if encoder_cfg is not None:
            if getattr(encoder_cfg, "image_config", None) is not None:
                if "qwen35moe" in encoder_cfg.image_config.model_type:
                    self.image_encoder = BeeBeeVLQwen35MoeVisionModel._from_config(config=encoder_cfg.image_config, 
                                                           attn_implementation=encoder_cfg.image_config._attn_implementation, 
                                                           dtype=torch_dtype)
                    print("loading qwen35moe vision encoder!!")
                else:
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

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        """
        Persist the full omni config instead of only the foundation LLM config.

        The model runtime keeps `self.config` aligned with foundation behavior, but
        checkpoint export should serialize the top-level omni structure.
        """
        original_config = self.config
        try:
            self.config = self.omni_config
            return super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            self.config = original_config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_position_id_func(self):
        """
        Fallback position-id function for training data preprocessing.

        This omni wrapper does not implement multimodal rope delta logic like the built-in
        Qwen omni models, so we return a simple monotonically increasing position ids.
        """

        def position_id_func(input_ids: torch.LongTensor, **kwargs) -> dict[str, torch.Tensor]:
            seq_len = input_ids.shape[-1]
            pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            # Omni pipelines often expect (3, seq_len) style outputs; the forward() will normalize it.
            position_ids = torch.stack([pos, pos, pos], dim=0)
            return {"position_ids": position_ids}

        return position_id_func

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # image/video
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # audio
        audio_features: Optional[torch.Tensor] = None,
        audio_features_lens: Optional[torch.LongTensor] = None,
        # outputs
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        step_timer = kwargs.pop("step_timer", None)
        
        if input_ids is None:
            raise ValueError("forward() requires `input_ids` because image/video/audio masks are computed from input_ids.")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        parallel_state = get_parallel_state()
        sp_enabled = parallel_state is not None and parallel_state.sp_enabled and self.training
        sp_group = parallel_state.ulysses_group if sp_enabled else None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Local seq length (before SP gather/scatter).
        batch_size, seq_len_local, _ = inputs_embeds.shape

        # SP: gather inputs_embeds to full sequence and gather input_ids to compute masks.
        if sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(inputs_embeds, seq_dim=1, head_dim=2, group=sp_group) # [full_seq, h//sp]
            sp_size = parallel_state.sp_size
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=sp_group)
            input_ids = torch.cat(input_ids_list, dim=1)  # [bs, full_seq_len]

        seq_len = input_ids.shape[-1]

        # Position ids: build/check from the gathered full seq, then slice after SP restore.
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        else:
            # Best-effort normalize; we expect last dim to match gathered seq.
            if position_ids.ndim == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids[:, 0, :]
            elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
                # Some omni data pipelines output position_ids as (3, seq_len) with batch dim squeezed.
                position_ids = position_ids[0].unsqueeze(0)
            elif position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.shape[-1] != seq_len:
                raise ValueError(
                    f"Position ids shape {position_ids.shape} does not match input_ids shape {input_ids.shape}"
                )
            if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        special_image_mask = input_ids == self.omni_config.image_token_id
        special_video_mask = input_ids == self.omni_config.video_token_id
        special_audio_mask = input_ids == self.omni_config.audio_token_id

        n_image_tokens = special_image_mask.sum().item()
        n_audio_tokens = special_audio_mask.sum().item()
        n_video_tokens = special_video_mask.sum().item()

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device) # [full_seq, h//d]
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        # Merge image/video pixels and run encoder once.
        effective_n_image_tokens = n_image_tokens
        effective_n_video_tokens = n_video_tokens

        # print("image tokens", effective_n_image_tokens, "audio_token", n_audio_tokens)
      
        if effective_n_image_tokens + effective_n_video_tokens > 0:
            if self.image_encoder is None:
                raise ValueError("vision tokens present but image_encoder is None.")

            cat_pixels = []
            cat_thw = []
            if pixel_values is not None and image_grid_thw is not None:
                cat_pixels.append(pixel_values)
                cat_thw.append(image_grid_thw)
        
            
            if pixel_values_videos is not None and video_grid_thw is not None:
                cat_pixels.append(pixel_values_videos)
                cat_thw.append(video_grid_thw)
           
           
            cat_pixels = torch.cat(cat_pixels, dim=0)
            cat_thw = torch.cat(cat_thw, dim=0)
            with step_timer.measure("vit") if step_timer else contextlib.nullcontext():
                vision_features, _ = self.image_encoder.lm_encode(features=cat_pixels, grid_thw=cat_thw)
            vision_features = vision_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # print("vision feature", vision_features.shape)
            image_features = vision_features[:effective_n_image_tokens]
            video_features = vision_features[effective_n_image_tokens : effective_n_image_tokens + effective_n_video_tokens]

            if effective_n_image_tokens > 0:
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            if effective_n_video_tokens > 0:
                inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
        
        else:
            fake_embeds, _ = self.image_encoder.dummy_forward()
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds

        # Audio
        if n_audio_tokens > 0 and audio_features is not None:
            with step_timer.measure("whisper") if step_timer else contextlib.nullcontext():
                audio_features, _ = self.audio_encoder.lm_encode(
                    features=audio_features, feature_lengths=audio_features_lens
                )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            
            audio_features = audio_features[:n_audio_tokens]
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        else:
            fake_audio_embeds, fake_audio_len = self.audio_encoder.dummy_forward()
            fake_audio_embeds = fake_audio_embeds.mean() * 0.0
            fake_audio_embeds = fake_audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_audio_embeds

    
        if sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(inputs_embeds, head_dim=2, seq_dim=1, group=sp_group)
           
        with step_timer.measure("llm") if step_timer else contextlib.nullcontext():
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        loss = None
        logits = None
   
        if labels is not None and self.training:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # loss = fixed_cross_entropy(shift_logits, shift_labels, **loss_kwargs)
                loss = loss_fct(shift_logits, shift_labels)
        
      
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    



if __name__ == "__main__":
    import math
    import numpy as np

    def get_adaptive_pool_size(M, N, scale=16):
        r = 1 / math.sqrt(scale)
        Mh = max(1, int(np.round(M * r)))
        Nw = max(1, int(np.round(N * r)))
        return Mh, Nw

    def calculate_image_tokens(h, w, t, mm_downsample_ratio=16):
       
        m_h, m_w = get_adaptive_pool_size(h // 2, w // 2, mm_downsample_ratio)
        num_image_tokens = t * m_h * m_w
        return int(num_image_tokens)

    def calculate_audio_tokens(raw_audio_len, audio_frame_length=480, audio_downsample_ratio=2):
       
        audio_feature_len = math.ceil(raw_audio_len / audio_frame_length)
        actual_audio_feature_len = (audio_feature_len + audio_downsample_ratio - 1) // audio_downsample_ratio
        return int(actual_audio_feature_len)

    # Minimal demo: build from config and print model structure
    from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
    from veomni.models.custom.llava_qwen3moe.configuration_qwen3moe_omni import Qwen3MoeOmniConfig

    MM_DOWNSAMPLE_RATIO = 16    # 图像下采样率
    AUDIO_FRAME_LENGTH = 320    # 音频帧长
    AUDIO_DOWNSAMPLE_RATIO = 10  # 音频下采样率
    # Encoder configs
    image_cfg = BeeBeeVLVisionModelConfig(
        spatial_merge_size=2,
        return_hidden_states=False,
        train_vision_projector=True,
        image_downsample_size=MM_DOWNSAMPLE_RATIO,
        image_projector_type="dynamic_avgpool",
        output_size=5120,
        hidden_size=1280
    
    )
    
    # print(image_cfg)

    audio_cfg = BeeBeeAudioModelConfig(
        return_hidden_states=False,
        train_audio_projector=True,
        audio_downsample_size=AUDIO_DOWNSAMPLE_RATIO,
        audio_projector_type="channel_upscale",
        output_size=5120,
    )

    # print(audio_cfg)
    # Foundation LLM config
    # NOTE: use a tiny config for fast local sanity-check.
    # The full 30B config init is very slow because it creates all expert weights.
    from transformers import Qwen3MoeConfig

    foundation_cfg = Qwen3MoeConfig(
        vocab_size=151936,  # must cover the demo token ids below
        hidden_size=5120,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        use_cache=False,
        tie_word_embeddings=False,
        # Disable MoE for the tiny test (keep the model lightweight).
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
        output_router_logits=True,
        decoder_sparse_step=100,
    )


    # Compose omni config
    omni_cfg = Qwen3MoeOmniConfig(
        encoder_config={
            "image_config": image_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
        model_type="llavaqwen3moe_omni"
    )

    print(omni_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = LlavaQwen3MoeForCausalLM._from_config(omni_cfg).to(device).to(torch_dtype)
    # model.eval()

    print(model)
    print("Has image encoder:", isinstance(model.image_encoder, nn.Module))
    print("Has audio encoder:", isinstance(model.audio_encoder, nn.Module))

    # # Save and show serialized omni config
    # save_dir = "./tmp_qwen3moe_omni_config"
    # omni_cfg.save_pretrained(save_dir)
    # print(f"Saved omni config to {save_dir}/config.json")

    img_h, img_w, img_t = 448, 672, 1
    num_img_tokens = calculate_image_tokens(img_h // 14, img_w//14, img_t, MM_DOWNSAMPLE_RATIO)
    
    pixel_values = torch.randn(img_h *img_w // 14 // 14, 2*3*14*14, device=device, dtype=torch_dtype)
    image_grid_thw = torch.tensor([[img_t, img_h // 14, img_w // 14]], device=device) 

    raw_audio_len = 24000
    num_aud_tokens = calculate_audio_tokens(raw_audio_len, AUDIO_FRAME_LENGTH, AUDIO_DOWNSAMPLE_RATIO)
    
    pre_downsample_len = math.ceil(raw_audio_len / AUDIO_FRAME_LENGTH)
    audio_features = torch.randn(1, 128, 3000, device=device, dtype=torch_dtype)
    audio_feature_lengths = torch.tensor([pre_downsample_len], device=device)

    image_token_id = 151655 # 示例 ID
    audio_token_id = 151656 # 示例 ID
    video_token_id = 151657
    bos_token_id = 151643
    foundation_cfg.image_token_id = image_token_id
    foundation_cfg.audio_token_id = audio_token_id
    foundation_cfg.video_token_id = video_token_id
    
    # 构造序列: [BOS] + [IMAGE_TOKENS] + "Describe" + [AUDIO_TOKENS]
    new_input_ids = [bos_token_id]
    new_input_ids.extend([image_token_id] * num_img_tokens)
    new_input_ids.extend([100, 101, 102, 2738]) # 模拟 "Describe this"
    new_input_ids.extend([audio_token_id] * num_aud_tokens)
    
    input_ids = torch.tensor([new_input_ids], device=device)
    labels = torch.tensor([new_input_ids], device=device)

    print(f"--- Test Config ---")
    print(f"Image Tokens Needed: {num_img_tokens}")
    print(f"Audio Tokens Needed: {num_aud_tokens}")
    print(f"Total Input IDs Length: {input_ids.shape[1]}")

    # 5. 运行 Forward 测试
    print("\n--- Running Forward ---")
    try:
        # 这里的 model 是你定义的 LlavaQwen3MoeForCausalLM
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            audio_features=audio_features,
            audio_feature_lengths=audio_feature_lengths,
        )
        print("Forward success!")
        pass
    except Exception as e:
        import traceback
        print("\n" + "="*20 + " ERROR STACK TRACE " + "="*20)
        # 核心方法：打印详细的堆栈跟踪
        traceback.print_exc() 
        print("="*59 + "\n")
        
        print(f"Forward failed: {e}")




