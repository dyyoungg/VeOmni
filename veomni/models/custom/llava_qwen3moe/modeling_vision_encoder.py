from typing import Dict,Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from flash_attn import flash_attn_varlen_func

from veomni.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from veomni.models.custom.llava_qwen3moe.projector import build_image_projector
from veomni.models.custom.llava_qwen3moe.base import BaseEncoderModelMixin, BaseEncoderConfigMixin
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm, apply_rotary_pos_emb_vision, Qwen2_5_VLMLP
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads
from transformers import AutoConfig


def pad_tensor(x: Tensor, dim: int, padding_size: int, padding_value: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


class BeeBeeVLVisionModelConfig(BaseEncoderConfigMixin, Qwen2_5_VLVisionConfig):
    model_type = "beebee_vl_vision_model"
    def __init__(
        self,
        return_hidden_states=False,
        train_vision_projector=False,
        freeze_vision_merger: bool = False,
        image_downsample_size=8,
        image_projector_type="dynamic_avgpool",
        output_size=6144,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_hidden_states = return_hidden_states
        self.train_vision_projector = train_vision_projector
        # Used by `set_projector_trainable_only()` to decide whether to train the patch merger.
        self.freeze_vision_merger = freeze_vision_merger
        self.image_downsample_size = image_downsample_size
        self.image_projector_type = image_projector_type
        self.output_size = output_size

class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5_VLRMSNorm(context_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return x


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # ulysses sp patch: qkv projection
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        # [3, seq, heads, hidden]
        unpadded_dim_size = cu_seqlens[-1]
        if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
            qkv = gather_seq_scatter_heads(qkv, seq_dim=1, head_dim=2)
            sp_padding_size = qkv.size(1) - unpadded_dim_size
            if sp_padding_size > 0:
                qkv = unpad_tensor(qkv, dim=1, padding_size=sp_padding_size)
        q, k, v = qkv.unbind(0)
        # [seq, heads//sp, hidden]

        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings # cos sin :[seq, hidden]
        # 
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin) # [seq, head//sp, hidden]

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)

        if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
            attn_output = pad_tensor(attn_output, dim=0, padding_size=sp_padding_size)
            attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=0)
        
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "flash_attention2") -> None:
        super().__init__()
        self.norm1 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionFlashAttention2(config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    
class Qwen25ViTPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.merger = Qwen2_5_VLPatchMerger(context_dim=config.hidden_size,
                                            spatial_merge_size=config.spatial_merge_size)
        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        unpadded_dim_size = cu_seqlens[-1]
        
        if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
            # rank = get_parallel_state().global_rank
            # dp_rank = get_parallel_state().dp_rank
            # print(f"RANK:{rank}, DP rank: {dp_rank}, image shape: {hidden_states.shape[0]}")
            hidden_states = gather_seq_scatter_heads(
                hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().ulysses_group
            )
            
            sp_padding_size = hidden_states.size(0) - unpadded_dim_size
            if sp_padding_size > 0:
                hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
            if sp_padding_size > 0:
                hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            hidden_states = gather_heads_scatter_seq(
                hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().ulysses_group
            )
            
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings
                )
        
        hidden_states = self.merger(hidden_states)
        
        reverse_indices = torch.argsort(window_index)

        if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
            sp_padding_size = hidden_states.size(0) * get_parallel_state().ulysses_size - seq_len // 4
            hidden_states = gather_seq_scatter_heads(
                hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().ulysses_group
            )
            
            if sp_padding_size > 0:
                hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            
        hidden_states = hidden_states[reverse_indices, :] # [seq, hidden//sp]
        return hidden_states

class BeeBeeVLVisionModel(BaseEncoderModelMixin, Qwen25ViTPretrainedModel):
    config_class = BeeBeeVLVisionModelConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config: BeeBeeVLVisionModelConfig):
        super().__init__(config)
        self.config = config
        self.mm_projector = build_image_projector(projector_type=config.image_projector_type, 
                                                  encoder_hidden=config.hidden_size * config.spatial_merge_size**2, 
                                                  out_hidden=config.output_size, 
                                                  downsample_ratio=self.config.image_downsample_size)

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        self.mm_projector.requires_grad_(True)
        if self.config.freeze_vision_merger:
            self.merger.requires_grad_(False)
        else:
            self.merger.requires_grad_(True)
    
    def lm_encode(self, features: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        features = super().forward(features, grid_thw)
       
        features, seq_len = self.mm_projector(features, grid_thw)
        return features, seq_len

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.ones(1536, 3 * 2 * 14 * 14).to(dtype=self.dtype, device=self.device) #  [644,364]
        grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}

    def dummy_forward(self):
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.ones(1536, 3 * 2 * 14 * 14).to(dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device=self.device)
            if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
                sp_world_size = get_parallel_state().sp_size
                grid_thw = torch.tensor([[sp_world_size, 32, 48]]).to(dtype=torch.int64, device=self.device)
            self._dummy_data = {"features": pixel_values, "grid_thw": grid_thw}
        return self.lm_encode(**self._dummy_data)
    

AutoConfig.register("beebee_vl_vision_model", BeeBeeVLVisionModelConfig)