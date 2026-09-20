"""Frame-level varlen cross-attention for visual token compression.

This experimental projector stays separate from ``projector.py`` so it can be
validated without changing an active training run. It keeps the token layout of
``DynamicAvgPoolProjector``: every frame emits a dynamic number of tokens, and
all frames from a packed batch are concatenated along the token dimension.

For each frame, adaptive average pooling first produces C_i base tokens. Those
tokens query all N_i dense patches from the same frame through cross-attention:

    base_i = AdaptiveAvgPool2d(dense_i)
    delta_i = CrossAttention(query=base_i, key=dense_i, value=dense_i)
    enhanced_i = base_i + zero_init_projection(delta_i)

Frames use independent varlen boundaries, so patches never leak across frames,
grids, or packed samples. Different frames may have different C_i and N_i.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def get_adaptive_pool_size(height: int, width: int, scale: float) -> tuple[int, int]:
    """Convert an area compression factor into an output spatial grid."""
    ratio = 1 / math.sqrt(scale)
    return max(1, round(height * ratio)), max(1, round(width * ratio))


@dataclass
class FrameAttentionAux:
    """Intermediate state for diagnostics and future teacher supervision.

    base_tokens/residual_tokens: [C, D] before the final projector MLP.
    cu_seqlens_q: [F + 1], frame boundaries for the C pooled queries.
    cu_seqlens_kv: [F + 1], matching boundaries for the N dense patches.
    query_lengths/key_value_lengths: Python lengths used to build the prefixes.
    """

    base_tokens: torch.Tensor
    residual_tokens: torch.Tensor
    cu_seqlens_q: torch.IntTensor
    cu_seqlens_kv: torch.IntTensor
    query_lengths: list[int]
    key_value_lengths: list[int]


class _ZeroInitOutputProjection(nn.Module):
    """Linear projection that remains zero through owning-model initialization."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.weight._veomni_fsdp_shard_dim = 1

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        self.weight._veomni_fsdp_shard_dim = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class FrameVarlenAttentionProjector(nn.Module):
    """Add frame-level dense information to adaptive-pooling tokens.

    FlashAttention receives one query sequence and one key/value sequence per
    frame, in a single varlen call. CUDA fp16/bf16 is required; no fallback is
    provided. All trainable modules execute once on the concatenated pack.

    The residual output projection is zero-initialized, so the pre-MLP tokens
    exactly match adaptive average pooling at initialization. The projection
    learns first; Q/K/V start receiving gradients after it becomes nonzero.

    Ulysses sequence parallelism is intentionally unsupported in this first
    version. Its projector input has a sharded hidden dimension, while each QK
    dot product here requires a complete attention head.
    """

    def __init__(
        self,
        encoder_hidden: int,
        out_hidden: int,
        downsample_ratio: float,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder_hidden <= 0 or out_hidden <= 0:
            raise ValueError("Projector dimensions must be positive.")
        if num_attention_heads <= 0 or encoder_hidden % num_attention_heads != 0:
            raise ValueError(
                "encoder_hidden must be divisible by num_attention_heads, got "
                f"encoder_hidden={encoder_hidden}, num_attention_heads={num_attention_heads}."
            )
        if not math.isfinite(downsample_ratio) or downsample_ratio <= 0:
            raise ValueError("downsample_ratio must be finite and positive.")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1).")
        if encoder_hidden // num_attention_heads > 256:
            raise ValueError("FlashAttention requires head_dim <= 256; increase num_attention_heads.")

        self.mm_downsample_ratio = downsample_ratio
        self.hidden_size = encoder_hidden
        self.merge_size = 2
        self.num_attention_heads = num_attention_heads
        self.head_dim = encoder_hidden // num_attention_heads
        self.attention_dropout = attention_dropout

        self.query_norm = nn.LayerNorm(encoder_hidden)
        self.key_value_norm = nn.LayerNorm(encoder_hidden)
        self.query_proj = nn.Linear(encoder_hidden, encoder_hidden, bias=False)
        self.key_proj = nn.Linear(encoder_hidden, encoder_hidden, bias=False)
        self.value_proj = nn.Linear(encoder_hidden, encoder_hidden, bias=False)

        # Query and key positions are projected separately because their grids
        # have different spatial resolutions. Coordinates are normalized T/Y/X.
        self.query_position_proj = nn.Linear(3, encoder_hidden, bias=False)
        self.key_position_proj = nn.Linear(3, encoder_hidden, bias=False)
        self.rate_proj = nn.Linear(1, encoder_hidden, bias=False)
        self.attention_out = _ZeroInitOutputProjection(encoder_hidden)

        # Keep this name and shape compatible with DynamicAvgPoolProjector.
        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, out_hidden),
        )

    def _apply(self, fn, recurse: bool = True):
        # to_empty() gives meta parameters uninitialized storage. Restore the
        # exact adaptive-pooling start after materialization.
        output_was_meta = self.attention_out.weight.is_meta
        result = super()._apply(fn, recurse=recurse)
        if output_was_meta and not self.attention_out.weight.is_meta:
            self.attention_out.reset_parameters()
        self.attention_out.weight._veomni_fsdp_shard_dim = 1
        return result

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize missing projector leaves via VeOmni's checkpoint loader.

        The loader selects the nearest parent's _init_weights for each missing
        parameter. A complete frame-attention checkpoint has no missing leaves,
        so its trained residual is preserved. Partial projector checkpoints are
        unsupported because the loader initializes a whole leaf module at once.
        """
        if isinstance(module, _ZeroInitOutputProjection):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_grid_values(
        values: Optional[torch.Tensor | Sequence[float]],
        count: int,
        default: float,
        name: str,
    ) -> list[float]:
        if values is None:
            return [default] * count
        if isinstance(values, torch.Tensor):
            flat_values = values.detach().cpu().reshape(-1).tolist()
        else:
            flat_values = list(values)
        if len(flat_values) != count:
            raise ValueError(f"{name} must contain {count} values, got {len(flat_values)}.")
        return [float(value) for value in flat_values]

    @staticmethod
    def _normalized_positions(time: int, height: int, width: int) -> torch.Tensor:
        """Return [T*H*W, 3] center coordinates normalized to [-1, 1]."""
        frame_pos = (torch.arange(time, dtype=torch.float32) + 0.5) / time
        row_pos = (torch.arange(height, dtype=torch.float32) + 0.5) / height
        col_pos = (torch.arange(width, dtype=torch.float32) + 0.5) / width
        positions = torch.stack(
            torch.meshgrid(frame_pos, row_pos, col_pos, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        return positions.mul(2).sub(1)

    @staticmethod
    def _prefix_sums(lengths: list[int], device: torch.device) -> torch.IntTensor:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, device=device, dtype=torch.int32)

    def _build_frame_metadata(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_sizes: list[float],
        downsample_ratios: list[float],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[int],
        torch.IntTensor,
        torch.IntTensor,
    ]:
        """Pool base tokens and build packed frame-level attention metadata."""
        grids = images_thw.detach().cpu().tolist()
        base_tokens = []
        query_positions = []
        key_positions = []
        query_rates = []
        query_lengths = []
        key_value_lengths = []
        feature_offset = 0

        for grid, merge_size_value, scale in zip(grids, merge_sizes, downsample_ratios):
            if any(not math.isfinite(value) or value <= 0 or int(value) != value for value in grid):
                raise ValueError(f"grid_thw values must be positive integers, got {grid}.")
            time, raw_height, raw_width = (int(value) for value in grid)
            if (
                not math.isfinite(merge_size_value)
                or merge_size_value <= 0
                or int(merge_size_value) != merge_size_value
            ):
                raise ValueError(f"merge_size must be a positive integer, got {merge_size_value}.")
            merge_size = int(merge_size_value)
            if raw_height % merge_size != 0 or raw_width % merge_size != 0:
                raise ValueError(
                    f"Grid {(time, raw_height, raw_width)} is not divisible by merge_size={merge_size}."
                )
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"downsample ratios must be finite and positive, got {scale}.")

            height = raw_height // merge_size
            width = raw_width // merge_size
            output_height, output_width = get_adaptive_pool_size(height, width, scale)
            frame_feature_count = height * width
            grid_feature_count = time * frame_feature_count
            next_feature_offset = feature_offset + grid_feature_count
            if next_feature_offset > images_feature.shape[0]:
                raise ValueError(
                    "images_feature length does not match grid_thw after spatial merging: "
                    f"expected at least {next_feature_offset}, got {images_feature.shape[0]}."
                )

            grid_features = images_feature[feature_offset:next_feature_offset].view(
                time, height, width, self.hidden_size
            )
            pooled = F.adaptive_avg_pool2d(
                grid_features.permute(0, 3, 1, 2), (output_height, output_width)
            )
            pooled = pooled.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

            queries_per_frame = output_height * output_width
            base_tokens.append(pooled)
            query_positions.append(self._normalized_positions(time, output_height, output_width))
            key_positions.append(self._normalized_positions(time, height, width))
            query_rates.append(torch.full((time * queries_per_frame, 1), math.log2(scale)))
            query_lengths.extend([queries_per_frame] * time)
            key_value_lengths.extend([frame_feature_count] * time)
            feature_offset = next_feature_offset

        if feature_offset != images_feature.shape[0]:
            raise ValueError(
                "images_feature length does not match grid_thw after spatial merging: "
                f"expected {feature_offset}, got {images_feature.shape[0]}."
            )

        device = images_feature.device
        if base_tokens:
            packed_base = torch.cat(base_tokens)
            packed_query_positions = torch.cat(query_positions).to(device=device, non_blocking=True)
            packed_key_positions = torch.cat(key_positions).to(device=device, non_blocking=True)
            packed_query_rates = torch.cat(query_rates).to(device=device, non_blocking=True)
        else:
            packed_base = images_feature.new_empty((0, self.hidden_size))
            packed_query_positions = torch.empty((0, 3), device=device, dtype=torch.float32)
            packed_key_positions = torch.empty((0, 3), device=device, dtype=torch.float32)
            packed_query_rates = torch.empty((0, 1), device=device, dtype=torch.float32)

        return (
            packed_base,
            packed_query_positions,
            packed_key_positions,
            packed_query_rates,
            query_lengths,
            key_value_lengths,
            self._prefix_sums(query_lengths, device),
            self._prefix_sums(key_value_lengths, device),
        )

    def _varlen_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.IntTensor,
        cu_seqlens_kv: torch.IntTensor,
        query_lengths: list[int],
        key_value_lengths: list[int],
    ) -> torch.Tensor:
        if query.shape[0] == 0:
            # Keep every projection in the autograd graph on ranks with no
            # visual tokens. This produces zero gradients instead of unused
            # parameters when other data-parallel ranks contain visual input.
            return query + (key.sum() + value.sum()) * 0
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(key_value_lengths),
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )

    def enhance_tokens(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_size: Optional[torch.Tensor | Sequence[float]] = None,
        downsample_ratios: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> tuple[torch.Tensor, list[int], FrameAttentionAux]:
        """Return [C, D] enhanced tokens, frame lengths, and attention metadata."""
        if images_feature.ndim != 2:
            raise ValueError(
                f"images_feature must have shape [tokens, hidden], got {tuple(images_feature.shape)}."
            )
        if images_feature.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {images_feature.shape[-1]}.")
        if images_thw.ndim != 2 or images_thw.shape[-1] != 3:
            raise ValueError(f"images_thw must have shape [grids, 3], got {tuple(images_thw.shape)}.")

        parallel_state = get_parallel_state()
        if self.training and parallel_state is not None and parallel_state.sp_enabled:
            raise NotImplementedError(
                "FrameVarlenAttentionProjector does not yet support sequence parallelism. "
                "QK dot products require complete attention heads."
            )

        grid_count = images_thw.shape[0]
        merge_sizes = self._normalize_grid_values(merge_size, grid_count, self.merge_size, "merge_size")
        ratios = self._normalize_grid_values(
            downsample_ratios, grid_count, self.mm_downsample_ratio, "downsample_ratios"
        )
        (
            base_tokens,
            query_positions,
            key_positions,
            query_rates,
            query_lengths,
            key_value_lengths,
            cu_seqlens_q,
            cu_seqlens_kv,
        ) = self._build_frame_metadata(images_feature, images_thw, merge_sizes, ratios)

        query_hidden = self.query_proj(self.query_norm(base_tokens))
        query_hidden = query_hidden + self.query_position_proj(
            query_positions.to(images_feature.dtype)
        )
        query_hidden = query_hidden + self.rate_proj(query_rates.to(images_feature.dtype))
        key_value_hidden = self.key_value_norm(images_feature)
        key_hidden = self.key_proj(key_value_hidden)
        key_hidden = key_hidden + self.key_position_proj(key_positions.to(images_feature.dtype))
        value_hidden = self.value_proj(key_value_hidden)

        query = query_hidden.view(-1, self.num_attention_heads, self.head_dim)
        key = key_hidden.view(-1, self.num_attention_heads, self.head_dim)
        value = value_hidden.view(-1, self.num_attention_heads, self.head_dim)
        if flash_attn_varlen_func is None:
            raise RuntimeError("FrameVarlenAttentionProjector requires flash-attn.")
        if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashAttention requires CUDA fp16 or bf16 query/key/value tensors.")
        attention_output = self._varlen_attention(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_kv,
            query_lengths,
            key_value_lengths,
        ).reshape(-1, self.hidden_size)
        residual_tokens = self.attention_out(attention_output)
        enhanced_tokens = base_tokens + residual_tokens

        aux = FrameAttentionAux(
            base_tokens=base_tokens,
            residual_tokens=residual_tokens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            query_lengths=query_lengths,
            key_value_lengths=key_value_lengths,
        )
        return enhanced_tokens, query_lengths, aux

    def forward(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_size: Optional[torch.Tensor | Sequence[float]] = None,
        downsample_ratios: Optional[torch.Tensor | Sequence[float]] = None,
        *,
        return_attention_aux: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], FrameAttentionAux]:
        """Enhance [C, D] pooled tokens, then map them to [C, out_hidden]."""
        enhanced_tokens, seq_len, aux = self.enhance_tokens(
            images_feature,
            images_thw,
            merge_size=merge_size,
            downsample_ratios=downsample_ratios,
        )
        hidden_states = self.mlp(enhanced_tokens)
        if return_attention_aux:
            return hidden_states, seq_len, aux
        return hidden_states, seq_len


def build_frame_attention_projector(
    encoder_hidden: int,
    out_hidden: int,
    downsample_ratio: float,
    num_attention_heads: int = 8,
    attention_dropout: float = 0.0,
) -> FrameVarlenAttentionProjector:
    """Build the standalone experimental frame-level attention projector."""
    return FrameVarlenAttentionProjector(
        encoder_hidden=encoder_hidden,
        out_hidden=out_hidden,
        downsample_ratio=downsample_ratio,
        num_attention_heads=num_attention_heads,
        attention_dropout=attention_dropout,
    )
