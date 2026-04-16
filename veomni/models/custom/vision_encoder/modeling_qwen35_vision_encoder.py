"""
BeeBeeVL Vision Encoder for Qwen3.5 MoE.

This file adapts the Qwen3.5MoE vision tower to the BeeBeeVL architecture pattern defined in
`modeling_vision_encoder.py`, including:
  - Simplified patch merger (LayerNorm + view only; no linear projection -- that lives in mm_projector)
  - Flash-attention-based vision attention with Ulysses sequence-parallel (SP) support
  - SP-aware vision model forward that properly handles learned positional embeddings without
    window-attention reordering (Qwen3.5MoE vision has no window attention)

Mirrors `Qwen25ViTPretrainedModel` / `BeeBeeVLVisionModel` patterns from modeling_vision_encoder.py,
adapted for the Qwen3.5MoE vision architecture differences:
  - LayerNorm (not RMSNorm) in vision blocks
  - Learned absolute position embeddings (`fast_pos_embed_interpolate`) in addition to RoPE
  - No window attention (no `fullatt_block_indexes`, no `get_window_index`)
  - Different rotary embedding API (`rot_pos_emb` returns [seq, head_dim // 2] via 2-D lookup)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_varlen_func
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads, ulysses_pad_and_slice
from veomni.models.custom.llava_qwen3moe.base import BaseEncoderConfigMixin, BaseEncoderModelMixin
from veomni.models.custom.llava_qwen3moe.projector import build_image_projector
from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionMLP,
    apply_rotary_pos_emb_vision,
)

def pad_tensor(x: Tensor, dim: int, padding_size: int, padding_value: int = 0) -> Tensor:
    """Append `padding_size` slices of `padding_value` along `dim`."""
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    """Remove the last `padding_size` slices along `dim`."""
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]

class BeeBeeVLQwen35MoeVisionModelConfig(BaseEncoderConfigMixin, Qwen3_5MoeVisionConfig):
   
    model_type = "beebee_vl_qwen35moe_vision_model"

    def __init__(
        self,
        return_hidden_states: bool = False,
        train_vision_projector: bool = False,
        freeze_vision_merger: bool = False,
        image_downsample_size: int = 8,
        image_projector_type: str = "dynamic_avgpool",
        output_size: int = 7168,   
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_hidden_states = return_hidden_states
        self.train_vision_projector = train_vision_projector
        # Decides whether the patch merger is also frozen when set_projector_trainable_only() is called.
        self.freeze_vision_merger = freeze_vision_merger
        self.image_downsample_size = image_downsample_size
        self.image_projector_type = image_projector_type
        self.output_size = output_size


class Qwen3_5MoeSimplePatchMerger(nn.Module):
    """
    Simplified patch merger — LayerNorm + reshape only, no linear projection.

    Mirrors Qwen2_5_VLPatchMerger from modeling_vision_encoder.py.
    The actual projection to the LLM hidden size is delegated to mm_projector.

    Note: Qwen3.5MoE vision blocks use LayerNorm (not RMSNorm), so we use
    nn.LayerNorm here instead of Qwen2_5_VLRMSNorm.
    """

    def __init__(self, config:BeeBeeVLQwen35MoeVisionModelConfig,  use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)

class Qwen3_5MoeVisionFlashAttention2(nn.Module):

    def __init__(self, config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1   # kept for potential eager-attention compatibility
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
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

        # QKV projection: [seq, 3*dim] → [3, seq, heads, head_dim]
        qkv = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
        )

        sp_enabled = get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training
        if sp_enabled:
            qkv = gather_seq_scatter_heads(qkv, seq_dim=1, head_dim=2)
            
        q, k, v = qkv.unbind(0)  # each [seq_full, heads_local, head_dim]

        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings  # [seq_full, head_dim]

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Flash attention (variable-length, no causal mask)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        )
       
        if sp_enabled:
            attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=0)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5MoeVisionBlockSP(nn.Module):
    """
    Vision block using Qwen3_5MoeVisionFlashAttention2 + Qwen3_5MoeVisionMLP.

    Identical structure to the original Qwen3_5MoeVisionBlock but uses our
    SP-aware attention.  LayerNorm (not RMSNorm) is kept as in the original.

    Mirrors Qwen2_5_VLVisionBlock from modeling_vision_encoder.py.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5MoeVisionFlashAttention2(config)
        self.mlp = Qwen3_5MoeVisionMLP(config)

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


class Qwen3_5MoeViTPretrainedModel(Qwen3_5MoeVisionModel):
    """
    Qwen3.5MoE vision backbone with:
      - Simplified patch merger (Qwen3_5MoeSimplePatchMerger)
      - SP-aware vision blocks (Qwen3_5MoeVisionBlockSP)
      - Ulysses-SP-compatible forward pass

    Mirrors Qwen25ViTPretrainedModel from modeling_vision_encoder.py.

    Key differences from BeeBeeVL / Qwen2.5VL:
      * Qwen3.5MoE vision has learned absolute position embeddings
        (fast_pos_embed_interpolate) added on top of RoPE.
      * There is NO window attention, so we skip the outer gather/scatter
        that exists in Qwen25ViTPretrainedModel solely for window reordering.
        Instead, learned pos-embeds are applied locally on each SP rank by
        slicing the full-sequence pos-embed tensor with the rank's index.
      * The final gather (merger output → full sequence) mirrors BeeBeeVL
        exactly so that mm_projector receives a full-sequence view.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.merger = Qwen3_5MoeSimplePatchMerger(
            config,
            use_postshuffle_norm=False
        )
    
        self.blocks = nn.ModuleList(
            [Qwen3_5MoeVisionBlockSP(config) for _ in range(config.depth)]
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states: pixel patch tokens, shape [seq_total_or_local, patch_dim].
                           In SP mode each rank receives its contiguous local shard.
            grid_thw: [num_images, 3] — temporal / height / width grid sizes.

        Returns:
            Merged patch features [seq_merged_full (or local in non-SP), hidden_merged].
            In Ulysses SP training mode the returned sequence is full (gathered) with
            the hidden dimension sharded across the SP group — exactly as
            Qwen25ViTPretrainedModel does.
        """
     
        hidden_states = self.patch_embed(hidden_states)  # [seq_local, hidden]
        sp_enabled = (
            get_parallel_state() is not None
            and get_parallel_state().sp_enabled
            and self.training
        )
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        if sp_enabled:
            pos_embeds = ulysses_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=self.spatial_merge_size ** 2)
       
        hidden_states = hidden_states + pos_embeds
      

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # FA2 needs int32; onnx tracing needs the same dtype as grid_thw.
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        unpadded_seq_len = int(cu_seqlens[-1])
        pad_seq_len = 0
        seq_len = hidden_states.size(0)
        rotary_pos_emb = rotary_pos_emb.reshape(unpadded_seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        if sp_enabled:
            sp_size = get_parallel_state().ulysses_size
            pad_seq_len = seq_len * sp_size - unpadded_seq_len
            if pad_seq_len > 0:
                emb = pad_tensor(emb, dim=0, padding_size=pad_seq_len)
                new_cumsum = cu_seqlens[-1] + pad_seq_len
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)

        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.merger(hidden_states)

        if sp_enabled:
            merge_unit = self.spatial_merge_size ** 2
       
            sp_padding_size = (
                hidden_states.size(0) * get_parallel_state().ulysses_size
                - unpadded_seq_len // merge_unit
            )
            hidden_states = gather_seq_scatter_heads(
                hidden_states,
                seq_dim=0,
                head_dim=1,
                group=get_parallel_state().ulysses_group,
            )
            if sp_padding_size > 0:
                hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)

        return hidden_states


class BeeBeeVLQwen35MoeVisionModel(BaseEncoderModelMixin, Qwen3_5MoeViTPretrainedModel):
    """
    Full BeeBeeVL vision encoder for Qwen3.5 MoE.

    Combines the SP-aware vision backbone (Qwen3_5MoeViTPretrainedModel) with
    an mm_projector that maps merged patch features to the LLM token space.

    Mirrors BeeBeeVLVisionModel from modeling_vision_encoder.py.
    """

    config_class = BeeBeeVLQwen35MoeVisionModelConfig
    _no_split_modules = ["Qwen3_5MoeVisionBlockSP"]

    def __init__(self, config: BeeBeeVLQwen35MoeVisionModelConfig):
        super().__init__(config)
        self.config = config
        self.mm_projector = build_image_projector(
            projector_type=config.image_projector_type,
            encoder_hidden=config.hidden_size * config.spatial_merge_size ** 2,
            out_hidden=config.output_size,
            downsample_ratio=config.image_downsample_size,
        )

    def set_projector_trainable_only(self):
        """Freeze the backbone; keep mm_projector (and optionally the merger) trainable."""
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
        """Dummy inputs for weight initialisation / compilation warm-up."""
        # 1536 patches of size (temporal=2, patch=14, patch=14) for a 32×48 grid
        pixel_values = torch.ones(
            1536, 3 * 2 * 14 * 14, dtype=self.dtype, device=self.device
        )
        grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}

    def dummy_forward(self):
        """Run a single forward pass with dummy data (used by SP initialisation)."""
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.ones(
                1536, 3 * 2 * 14 * 14, dtype=self.dtype, device=self.device
            )
            grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device=self.device)
            if get_parallel_state() is not None and get_parallel_state().sp_enabled and self.training:
                sp_world_size = get_parallel_state().sp_size
                # Scale T so that total tokens are divisible by sp_world_size.
                grid_thw = torch.tensor(
                    [[sp_world_size, 32, 48]], dtype=torch.int64, device=self.device
                )
            self._dummy_data = {"features": pixel_values, "grid_thw": grid_thw}
        return self.lm_encode(**self._dummy_data)

AutoConfig.register("beebee_vl_qwen35moe_vision_model", BeeBeeVLQwen35MoeVisionModelConfig)


def extract_ViT_weights():
    import os
    import json
    from safetensors.torch import load_file, save_file
    model_path = "/mnt/afs/share/Qwen3.5-35B-A3B"
    config = AutoConfig.from_pretrained(model_path)
    vision_config = config.vision_config
    print(vision_config)
    state_dicts = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            print(file)
            weights = load_file(os.path.join(model_path, file))
            for name, p in weights.items():
                if "visual." in name and "merger.linear" not in name:
                    print(name)
                    state_dicts[name.replace("model.visual.", "")] = p

    save_dir = "/mnt/afs/share/Qwen35_A3B_vision_encoder"
    os.makedirs(save_dir, exist_ok=True)
    save_file(state_dicts, os.path.join(save_dir, "model.safetensors"))
    keys = list(state_dicts.keys())
    weight_map = {k: "model.safetensors" for k in keys}
    total_size = sum(v.numel() for v in state_dicts.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    vision_config.save_pretrained(save_dir)


if __name__ == "__main__":
    extract_ViT_weights()
