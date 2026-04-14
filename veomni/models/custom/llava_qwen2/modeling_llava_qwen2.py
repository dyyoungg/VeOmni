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
    



