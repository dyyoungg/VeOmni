from typing import Optional
from functools import partial, lru_cache
import contextlib
import itertools
import math

from transformers import AutoConfig, AutoModel, GenerationMixin, PreTrainedModel
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import can_return_tuple
import numpy as np

from veomni.models.custom.llava_qwen35.configuration_qwen35_omni import Qwen35OmniConfig
from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModel
from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import BeeBeeVLQwen35MoeVisionModel
from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeVLAudioModel
from veomni.models.custom.llava_qwen3moe.modeling_qwen3_audio_encoder import BeeBeeVLQwen3AudioModel
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads


@lru_cache(maxsize=256)
def _get_adaptive_pool_size(M: int, N: int, scale: float) -> tuple[int, int]:
    """Compute compressed spatial dimensions after adaptive average pooling."""
    r = 1.0 / math.sqrt(scale)
    Mh = max(1, int(np.round(M * r)))
    Nw = max(1, int(np.round(N * r)))
    return Mh, Nw


def _get_foundation_llm_class(config: Qwen35OmniConfig):
    """Return the appropriate foundation LLM class based on config."""
    if config.is_moe:
        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
        )
        return Qwen3_5MoeForCausalLM
    else:
        from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
            Qwen3_5ForCausalLM,
        )
        return Qwen3_5ForCausalLM


def _get_load_balancing_loss_func():
    """Import load_balancing_loss_func from the MoE module."""
    from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
        load_balancing_loss_func,
    )
    return load_balancing_loss_func


def _compute_3d_rope_positions(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    image_downsample_ratios: Optional[torch.Tensor],
    video_downsample_ratios: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    default_downsample_ratio: float,
    attention_mask: Optional[torch.Tensor] = None,
    seq_lens: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """
    Compute 3D M-RoPE position IDs for packed sequences with dynamically compressed vision tokens.

    For packing (bs=1 with multiple sub-sequences), pass `seq_lens` — a list of per-batch-row
    lists of sub-sequence lengths. Each sub-sequence gets independent position IDs starting from 0.
    Without `seq_lens`, each batch row is treated as a single sequence.

    Args:
        input_ids: [batch, seq_len] token IDs
        image_grid_thw: [num_images, 3] original (t, h, w) grids from vision processor
        video_grid_thw: [num_videos, 3] original (t, h, w) grids from vision processor
        image_downsample_ratios: [num_images] per-image compression ratios
        video_downsample_ratios: [num_videos] per-video compression ratios
        image_token_id: token ID for image placeholder
        video_token_id: token ID for video placeholder
        spatial_merge_size: vision encoder spatial merge factor (typically 2)
        default_downsample_ratio: fallback ratio when per-image ratios are not provided
        attention_mask: [batch, seq_len] optional padding mask (1=valid, 0=pad)
        seq_lens: per-batch-row list of sub-sequence lengths for packing

    Returns:
        position_ids: [3, batch, seq_len] — temporal, height, width position IDs
    """
    batch_size, total_seq_len = input_ids.shape
    device = input_ids.device

    position_ids = torch.zeros(3, batch_size, total_seq_len, dtype=torch.long, device=device)

    image_idx = 0
    video_idx = 0

    for batch_idx in range(batch_size):
        row_ids = input_ids[batch_idx]

        # Determine sub-sequence boundaries for this batch row
        if seq_lens is not None and len(seq_lens) > batch_idx:
            sub_seq_lengths = seq_lens[batch_idx]
        elif attention_mask is not None:
            # Use attention_mask to determine valid length (no packing info, single seq)
            valid_len = attention_mask[batch_idx].sum().item()
            sub_seq_lengths = [int(valid_len)]
        else:
            sub_seq_lengths = [total_seq_len]

        # Process each sub-sequence independently (for packing)
        seq_offset = 0
        for sub_len in sub_seq_lengths:
            sub_ids = row_ids[seq_offset : seq_offset + sub_len].tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            current_pos = 0
            token_ptr = 0

            while token_ptr < sub_len:
                token_id = sub_ids[token_ptr]

                if token_id == image_token_id or token_id == video_token_id:
                    is_image = (token_id == image_token_id)

                    # Count contiguous vision tokens
                    vision_start = token_ptr
                    while token_ptr < sub_len and sub_ids[token_ptr] == token_id:
                        token_ptr += 1
                    num_vision_tokens = token_ptr - vision_start

                    # Get the grid and downsample ratio for this vision segment
                    if is_image:
                        if image_grid_thw is not None and image_idx < image_grid_thw.shape[0]:
                            t, h, w = image_grid_thw[image_idx].tolist()
                        else:
                            # Fallback: treat as 1x1 grid per token
                            t, h, w = 1, int(math.sqrt(num_vision_tokens * spatial_merge_size**2)), int(math.sqrt(num_vision_tokens * spatial_merge_size**2))

                        if image_downsample_ratios is not None and image_idx < len(image_downsample_ratios):
                            ratio = image_downsample_ratios[image_idx].item()
                        else:
                            ratio = default_downsample_ratio
                        image_idx += 1
                    else:
                        if video_grid_thw is not None and video_idx < video_grid_thw.shape[0]:
                            t, h, w = video_grid_thw[video_idx].tolist()
                        else:
                            t, h, w = 1, int(math.sqrt(num_vision_tokens * spatial_merge_size**2)), int(math.sqrt(num_vision_tokens * spatial_merge_size**2))

                        if video_downsample_ratios is not None and video_idx < len(video_downsample_ratios):
                            ratio = video_downsample_ratios[video_idx].item()
                        else:
                            ratio = default_downsample_ratio
                        video_idx += 1

                    # Compute compressed spatial dimensions
                    llm_h = h // spatial_merge_size
                    llm_w = w // spatial_merge_size
                    Mh, Nw = _get_adaptive_pool_size(llm_h, llm_w, ratio)

                    # Generate 3D position IDs for this vision segment
                    # Temporal: each frame gets a distinct temporal position
                    # Spatial: h/w positions form a grid per frame
                    t_ids = torch.arange(t, device=device).repeat_interleave(Mh * Nw) + current_pos
                    h_ids = torch.arange(Mh, device=device).repeat_interleave(Nw).repeat(t) + current_pos
                    w_ids = torch.arange(Nw, device=device).repeat(Mh * t) + current_pos

                    vision_pos = torch.stack([t_ids, h_ids, w_ids], dim=0)  # [3, t*Mh*Nw]

                    # Verify token count matches
                    expected_tokens = t * Mh * Nw
                    if expected_tokens != num_vision_tokens:
                        # Mismatch — truncate or pad position IDs to match actual tokens
                        if expected_tokens > num_vision_tokens:
                            vision_pos = vision_pos[:, :num_vision_tokens]
                        else:
                            # Extend with the last position value
                            pad_size = num_vision_tokens - expected_tokens
                            last_pos = vision_pos[:, -1:].expand(3, pad_size)
                            vision_pos = torch.cat([vision_pos, last_pos], dim=1)

                    llm_pos_ids_list.append(vision_pos)

                    # Advance position counter: next text starts after max spatial extent
                    current_pos += max(max(Mh, Nw), t)

                else:
                    # Text/audio token — find contiguous non-vision span
                    text_start = token_ptr
                    while token_ptr < sub_len and sub_ids[token_ptr] != image_token_id and sub_ids[token_ptr] != video_token_id:
                        token_ptr += 1
                    text_len = token_ptr - text_start

                    # All 3 dims get the same sequential position
                    text_pos = torch.arange(text_len, device=device).unsqueeze(0).expand(3, -1) + current_pos
                    llm_pos_ids_list.append(text_pos)
                    current_pos += text_len

            # Concatenate all position segments for this sub-sequence
            if llm_pos_ids_list:
                sub_positions = torch.cat(llm_pos_ids_list, dim=1)  # [3, sub_len]
                position_ids[:, batch_idx, seq_offset : seq_offset + sub_len] = sub_positions
            seq_offset += sub_len

    return position_ids


class Qwen35OmniPreTrainedModel(PreTrainedModel):
    config_class = Qwen35OmniConfig
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


class LlavaQwen35ForCausalLM(Qwen35OmniPreTrainedModel, GenerationMixin):
    """
    Qwen3.5 causal LM augmented with image/audio encoders.

    Supports both dense Qwen3.5 and Qwen3.5 MoE as foundation LLM.
    It expects a `Qwen35OmniConfig` at construction time. The underlying LLM is
    initialized from `foundation_config`. The image/audio encoders are created
    from `encoder_config`'s sub-configs.
    """

    def __init__(self, config: Qwen35OmniConfig):
        # Initialize base with foundation (text) config
        super().__init__(config.foundation_config)
        self.foundation_config = config.foundation_config
        self.config.tie_word_embeddings = getattr(self.foundation_config, "tie_word_embeddings", False)
        if not hasattr(self, "all_tied_weights_keys") and not self.config.tie_word_embeddings:
            self.all_tied_weights_keys = {}

        dtype = getattr(config.foundation_config, "dtype", None)
        if dtype is None or dtype == torch.bfloat16 or dtype == "bfloat16" or dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Instantiate the correct foundation LLM (dense or MoE)
        FoundationLLMClass = _get_foundation_llm_class(config)
        foundation_llm = FoundationLLMClass._from_config(
            self.foundation_config,
            attn_implementation=self.foundation_config._attn_implementation,
            dtype=torch_dtype,
        )
        self.model = foundation_llm.model
        self.vocab_size = foundation_llm.vocab_size
        self.lm_head = foundation_llm.lm_head

        # MoE-specific attributes
        self._is_moe = config.is_moe
        if self._is_moe:
            self.router_aux_loss_coef = foundation_llm.router_aux_loss_coef
            self.num_experts = foundation_llm.num_experts
            self.num_experts_per_tok = foundation_llm.num_experts_per_tok

        del foundation_llm

        # Keep a reference to the omni config
        self.omni_config = config

        # Optional multimodal encoders
        self.image_encoder: Optional[nn.Module] = None
        self.audio_encoder: Optional[nn.Module] = None

        encoder_cfg = getattr(config, "encoder_config", None)

        if encoder_cfg is not None:
            if getattr(encoder_cfg, "image_config", None) is not None:
                if "qwen3" in encoder_cfg.image_config.model_type:
                    self.image_encoder = BeeBeeVLQwen35MoeVisionModel._from_config(
                        config=encoder_cfg.image_config,
                        attn_implementation=encoder_cfg.image_config._attn_implementation,
                        dtype=torch_dtype,
                    )
                else:
                    self.image_encoder = BeeBeeVLVisionModel._from_config(
                        config=encoder_cfg.image_config,
                        attn_implementation=encoder_cfg.image_config._attn_implementation,
                        dtype=torch_dtype,
                    )

            if getattr(encoder_cfg, "audio_config", None) is not None:
                if "qwen3_audio" in getattr(encoder_cfg.audio_config, "model_type", ""):
                    self.audio_encoder = BeeBeeVLQwen3AudioModel._from_config(
                        encoder_cfg.audio_config,
                        attn_implementation=encoder_cfg.audio_config._attn_implementation,
                        dtype=torch_dtype,
                    )
                else:
                    self.audio_encoder = BeeBeeVLAudioModel._from_config(
                        encoder_cfg.audio_config,
                        attn_implementation=encoder_cfg.audio_config._attn_implementation,
                        dtype=torch_dtype,
                    )

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

    def get_parallel_plan(self):
        # Return the appropriate parallel plan based on foundation type.
        # Only MoE models have a dedicated parallel plan; dense models return None.
        if self._is_moe:
            from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan as _get_parallel_plan
            return _get_parallel_plan()
        return None

    def get_position_id_func(self):
        """
        Return a picklable position-id function for training data preprocessing.

        Computes 3D M-RoPE position IDs (temporal, height, width) accounting for
        dynamic image compression. Each vision token segment gets structured 3D
        positions; text/audio tokens get sequential 1D positions (same across all 3 dims).
        """
        spatial_merge_size = getattr(
            getattr(getattr(self, "image_encoder", None), "config", None),
            "spatial_merge_size", 2
        )
        default_downsample_ratio = getattr(
            getattr(getattr(self, "image_encoder", None), "mm_projector", None),
            "mm_downsample_ratio", 16
        )
        image_token_id = self.omni_config.image_token_id
        video_token_id = self.omni_config.video_token_id

        def position_id_func(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            image_downsample_ratios: Optional[torch.Tensor] = None,
            video_downsample_ratios: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> dict[str, torch.Tensor]:
            # input_ids: [1, seq_len] or [seq_len]
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            position_ids = _compute_3d_rope_positions(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_downsample_ratios=image_downsample_ratios,
                video_downsample_ratios=video_downsample_ratios,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                spatial_merge_size=spatial_merge_size,
                default_downsample_ratio=default_downsample_ratio,
            )
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
        image_downsample_ratios: Optional[torch.Tensor] = None,
        video_downsample_ratios: Optional[torch.Tensor] = None,
        # audio
        audio_features: Optional[torch.Tensor] = None,
        audio_features_lens: Optional[torch.LongTensor] = None,
        # outputs
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        step_timer = kwargs.pop("step_timer", None)

        if input_ids is None:
            raise ValueError("forward() requires `input_ids` because image/video/audio masks are computed from input_ids.")

        if self._is_moe:
            output_router_logits = (
                output_router_logits if output_router_logits is not None else self.foundation_config.output_router_logits
            )

        parallel_state = get_parallel_state()
        sp_enabled = parallel_state is not None and parallel_state.sp_enabled and self.training
        sp_group = parallel_state.sp_group if sp_enabled else None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Local seq length (before SP gather/scatter).
        batch_size, seq_len_local, _ = inputs_embeds.shape

        # SP: gather inputs_embeds to full sequence and gather input_ids to compute masks.
        if sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(inputs_embeds, seq_dim=1, head_dim=2, group=sp_group)
            sp_size = parallel_state.sp_size
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=sp_group)
            input_ids = torch.cat(input_ids_list, dim=1)  # [bs, full_seq_len]

        seq_len = input_ids.shape[-1]

        # Position ids: compute 3D M-RoPE for multimodal, or pass through pre-computed.
        has_vision = (
            (pixel_values is not None and image_grid_thw is not None) or
            (pixel_values_videos is not None and video_grid_thw is not None)
        )
        if position_ids is None:
            if has_vision:
                # Compute 3D M-RoPE position IDs from input_ids + grid info
                spatial_merge_size = getattr(
                    getattr(self.image_encoder, "config", None), "spatial_merge_size", 2
                )
                default_ratio = getattr(
                    getattr(self.image_encoder, "mm_projector", None), "mm_downsample_ratio", 16
                )
                # Derive packing boundaries from cu_seq_lens if available
                cu_seq_lens = kwargs.get("cu_seq_lens_q", None)
                packing_seq_lens = None
                if cu_seq_lens is not None:
                    # cu_seq_lens: [num_seqs + 1] cumulative lengths
                    lens = (cu_seq_lens[1:] - cu_seq_lens[:-1]).tolist()
                    packing_seq_lens = [lens] * batch_size  # bs=1 for packing

                position_ids = _compute_3d_rope_positions(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_downsample_ratios=image_downsample_ratios,
                    video_downsample_ratios=video_downsample_ratios,
                    image_token_id=self.omni_config.image_token_id,
                    video_token_id=self.omni_config.video_token_id,
                    spatial_merge_size=spatial_merge_size,
                    default_downsample_ratio=default_ratio,
                    attention_mask=attention_mask,
                    seq_lens=packing_seq_lens,
                )
            else:
                # Pure text — simple sequential positions
                past_length = 0
                if past_key_values is not None:
                    if hasattr(past_key_values, "get_seq_length"):
                        past_length = past_key_values.get_seq_length()
                    elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                        past_length = past_key_values[0][0].shape[2]

                position_ids = torch.arange(
                    past_length, past_length + seq_len,
                    dtype=torch.long, device=input_ids.device
                ).view(1, 1, -1).expand(3, batch_size, -1)
        else:
            # Pre-computed position_ids: normalize to [3, batch, seq]
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
            elif position_ids.ndim == 2:
                if position_ids.shape[0] == 3 and position_ids.shape[1] == seq_len:
                    # [3, seq_len] -> [3, batch, seq_len]
                    position_ids = position_ids.unsqueeze(1).expand(3, batch_size, -1)
                elif position_ids.shape[0] == batch_size:
                    # [batch, seq_len] -> [3, batch, seq_len]
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                else:
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            elif position_ids.ndim == 3:
                if position_ids.shape[0] == batch_size and position_ids.shape[1] == 3:
                    # [batch, 3, seq_len] -> [3, batch, seq_len]
                    position_ids = position_ids.permute(1, 0, 2)
                # else already [3, batch, seq_len]

            if position_ids.shape[-1] != seq_len:
                raise ValueError(
                    f"Position ids shape {position_ids.shape} does not match input_ids seq_len {seq_len}"
                )
            position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        special_image_mask = input_ids == self.omni_config.image_token_id
        special_video_mask = input_ids == self.omni_config.video_token_id
        special_audio_mask = input_ids == self.omni_config.audio_token_id

        n_image_tokens = special_image_mask.sum().item()
        n_audio_tokens = special_audio_mask.sum().item()
        n_video_tokens = special_video_mask.sum().item()

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        # Merge image/video pixels and run encoder once.
        effective_n_image_tokens = n_image_tokens
        effective_n_video_tokens = n_video_tokens

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
            # Concatenate downsample ratios for image + video grids
            cat_downsample_ratios = None
            if image_downsample_ratios is not None or video_downsample_ratios is not None:
                ratio_parts = []
                if pixel_values is not None and image_grid_thw is not None:
                    if image_downsample_ratios is not None:
                        ratio_parts.append(image_downsample_ratios)
                    else:
                        ratio_parts.append(torch.full((image_grid_thw.shape[0],), self.image_encoder.mm_projector.mm_downsample_ratio, device=cat_thw.device))
                if pixel_values_videos is not None and video_grid_thw is not None:
                    if video_downsample_ratios is not None:
                        ratio_parts.append(video_downsample_ratios)
                    else:
                        ratio_parts.append(torch.full((video_grid_thw.shape[0],), self.image_encoder.mm_projector.mm_downsample_ratio, device=cat_thw.device))
                if ratio_parts:
                    cat_downsample_ratios = torch.cat(ratio_parts, dim=0)
            with step_timer.measure("vit") if step_timer else contextlib.nullcontext():
                vision_features, _ = self.image_encoder.lm_encode(features=cat_pixels, grid_thw=cat_thw, downsample_ratios=cat_downsample_ratios)
            vision_features = vision_features.to(inputs_embeds.device, inputs_embeds.dtype)

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
            with step_timer.measure("audio_encoder") if step_timer else contextlib.nullcontext():
                audio_features, audio_num_tokens = self.audio_encoder.lm_encode(
                    features=audio_features, feature_lengths=audio_features_lens
                )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            actual_audio_tokens = audio_features.shape[0]
            if actual_audio_tokens != n_audio_tokens:
                print(f"[AUDIO TOKEN MISMATCH] placeholder={n_audio_tokens}, projector_output={actual_audio_tokens}, "
                      f"per_sample={audio_num_tokens}, feature_lens={audio_features_lens.tolist()}, "
                      f"mel_shape={list(audio_features.shape)}")
            audio_features = audio_features[:n_audio_tokens]
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        else:
            audio_has_trainable = not (
                getattr(self.audio_encoder, "freeze_audio_encoder", False)
                and getattr(self.audio_encoder, "freeze_audio_projector", False)
            )
            if audio_has_trainable:
                with step_timer.measure("audio_encoder") if step_timer else contextlib.nullcontext():
                    fake_audio_embeds, fake_audio_len = self.audio_encoder.dummy_forward()
                fake_audio_embeds = fake_audio_embeds.mean() * 0.0
                fake_audio_embeds = fake_audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_audio_embeds

        if sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(inputs_embeds, head_dim=2, seq_dim=1, group=sp_group)
            # Position ids must match local seq chunk after scatter; slice to
            # this SP rank's portion so RoPE encodes correct absolute positions.
            sp_rank = dist.get_rank(sp_group)
            chunk_size = position_ids.shape[-1] // parallel_state.sp_size
            position_ids = position_ids[:, :, sp_rank * chunk_size : (sp_rank + 1) * chunk_size]


        # Build model forward kwargs
        model_kwargs = dict(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if self._is_moe:
            model_kwargs["output_router_logits"] = output_router_logits

        with step_timer.measure("llm") if step_timer else contextlib.nullcontext():
            outputs = self.model(**model_kwargs)

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        loss = None
        logits = None

        if labels is not None and self.training:
            loss, logits, _ = self.loss_function(
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
                shift_logits = shift_logits.view(-1, self.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        # MoE auxiliary loss
        aux_loss = None
        if self._is_moe and output_router_logits:
            load_balancing_loss_func = _get_load_balancing_loss_func()
            mm_balance_coef = getattr(self.omni_config, "mm_balance_coef", 0.0)

            if mm_balance_coef > 0.0:
                # Split text vs multimodal, apply different balancing strength
                text_mask = (
                    (input_ids != self.omni_config.image_token_id) &
                    (input_ids != self.omni_config.video_token_id) &
                    (input_ids != self.omni_config.audio_token_id)
                ).long()

                text_aux = load_balancing_loss_func(
                    outputs.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    text_mask,
                )

                mm_mask = 1 - text_mask
                mm_aux = load_balancing_loss_func(
                    outputs.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    mm_mask,
                )
                aux_loss = text_aux + mm_balance_coef * mm_aux

            else:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )

            if labels is not None and loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        # Return appropriate output type based on foundation model
        if self._is_moe:
            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=getattr(outputs, "router_logits", None),
            )
        else:
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

    # --- Demo with Qwen3.5 MoE ---
    from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import BeeBeeVLVisionModelConfig
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import BeeBeeAudioModelConfig
    from veomni.models.custom.llava_qwen35.configuration_qwen35_omni import Qwen35OmniConfig

    MM_DOWNSAMPLE_RATIO = 16
    AUDIO_FRAME_LENGTH = 320
    AUDIO_DOWNSAMPLE_RATIO = 10

    image_cfg = BeeBeeVLVisionModelConfig(
        spatial_merge_size=2,
        return_hidden_states=False,
        train_vision_projector=True,
        image_downsample_size=MM_DOWNSAMPLE_RATIO,
        image_projector_type="dynamic_avgpool",
        output_size=5120,
        hidden_size=1280,
    )

    audio_cfg = BeeBeeAudioModelConfig(
        return_hidden_states=False,
        train_audio_projector=True,
        audio_downsample_size=AUDIO_DOWNSAMPLE_RATIO,
        audio_projector_type="channel_upscale",
        output_size=5120,
    )

    from transformers import Qwen3_5MoeConfig

    foundation_cfg = Qwen3_5MoeConfig(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        use_cache=False,
        tie_word_embeddings=False,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
        output_router_logits=True,
        decoder_sparse_step=100,
    )

    omni_cfg = Qwen35OmniConfig(
        encoder_config={
            "image_config": image_cfg,
            "audio_config": audio_cfg,
        },
        foundation_config=foundation_cfg,
        model_type="llavaqwen35_omni",
    )

    print(omni_cfg)
    print("Is MoE:", omni_cfg.is_moe)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = LlavaQwen35ForCausalLM._from_config(omni_cfg).to(device).to(torch_dtype)

    print(model)
    print("Has image encoder:", isinstance(model.image_encoder, nn.Module))
    print("Has audio encoder:", isinstance(model.audio_encoder, nn.Module))

    img_h, img_w, img_t = 448, 672, 1
    num_img_tokens = calculate_image_tokens(img_h // 14, img_w // 14, img_t, MM_DOWNSAMPLE_RATIO)

    pixel_values = torch.randn(img_h * img_w // 14 // 14, 2 * 3 * 14 * 14, device=device, dtype=torch_dtype)
    image_grid_thw = torch.tensor([[img_t, img_h // 14, img_w // 14]], device=device)

    raw_audio_len = 24000
    num_aud_tokens = calculate_audio_tokens(raw_audio_len, AUDIO_FRAME_LENGTH, AUDIO_DOWNSAMPLE_RATIO)

    pre_downsample_len = math.ceil(raw_audio_len / AUDIO_FRAME_LENGTH)
    audio_features = torch.randn(1, 128, 3000, device=device, dtype=torch_dtype)
    audio_feature_lengths = torch.tensor([pre_downsample_len], device=device)

    image_token_id = 151655
    audio_token_id = 151656
    video_token_id = 151657
    bos_token_id = 151643
    foundation_cfg.image_token_id = image_token_id
    foundation_cfg.audio_token_id = audio_token_id
    foundation_cfg.video_token_id = video_token_id

    new_input_ids = [bos_token_id]
    new_input_ids.extend([image_token_id] * num_img_tokens)
    new_input_ids.extend([100, 101, 102, 2738])
    new_input_ids.extend([audio_token_id] * num_aud_tokens)

    input_ids = torch.tensor([new_input_ids], device=device)
    labels = torch.tensor([new_input_ids], device=device)

    print(f"--- Test Config ---")
    print(f"Image Tokens Needed: {num_img_tokens}")
    print(f"Audio Tokens Needed: {num_aud_tokens}")
    print(f"Total Input IDs Length: {input_ids.shape[1]}")

    print("\n--- Running Forward ---")
    try:
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            audio_features=audio_features,
            audio_features_lens=audio_feature_lengths,
        )
        print("Forward success!")
        print(f"Loss: {outputs.loss}")
    except Exception as e:
        import traceback
        print("\n" + "=" * 20 + " ERROR STACK TRACE " + "=" * 20)
        traceback.print_exc()
        print("=" * 59 + "\n")
        print(f"Forward failed: {e}")
