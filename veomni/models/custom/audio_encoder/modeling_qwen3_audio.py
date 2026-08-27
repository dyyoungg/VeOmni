"""
Qwen3 Audio Encoder — standalone implementation.

Ported from transformers Qwen3OmniMoeAudioEncoder to avoid version dependency.
Architecture: 3x Conv2d (stride 2, total 8x downsample) + Sinusoidal pos emb
              + N transformer encoder layers + LN.

Weight-compatible with:
  - Qwen3-Omni-AudioTransformer checkpoints (proj1/proj2 are ignored by wrapper)
  - Qwen3-ASR encoder checkpoints
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from flash_attn import flash_attn_varlen_func

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen3AudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        n_window: int = 50,
        output_dim: int = 2048,
        n_window_infer: int = 800,
        downsample_hidden_size: int = 480,
        max_source_positions: int = 1500,
        max_position_embeddings: int = 13,
        conv_chunksize: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.downsample_hidden_size = downsample_hidden_size
        # Support both naming conventions
        self.max_source_positions = max_source_positions
        self.max_position_embeddings = max_position_embeddings
        self.conv_chunksize = conv_chunksize
        # Alias for transformers compatibility
        self.num_hidden_layers = encoder_layers
        self.hidden_size = d_model
        self.num_attention_heads = encoder_attention_heads
        self.intermediate_size = encoder_ffn_dim


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding (not learned)."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels")
        position_embedding = self._compute_embedding()
        self.register_buffer("positional_embedding", position_embedding, persistent=False)

    def _compute_embedding(self) -> torch.Tensor:
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(self.channels // 2).float()
        )
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen3AudioAttention(nn.Module):
    """Multi-head attention using flash_attn_varlen_func."""

    def __init__(self, config: Qwen3AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.out_proj(attn_output)


class Qwen3AudioEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(self, config: Qwen3AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,)


# ---------------------------------------------------------------------------
# Audio length utilities
# ---------------------------------------------------------------------------

def _get_feat_extract_output_lengths(input_lengths, n_window=50):
    """Output length after conv stack + chunking."""
    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13


def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
    """Length after three (k=3, s=2, p=1) convolutions; zero stays zero."""
    for _ in range(3):
        lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
    return lengths


def chunk_and_pad_features(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window: int,
) -> tuple:
    """Split audio mel features into fixed-size chunks and pad.

    Args:
        input_features: (feature_dim, total_frames) — concatenated mel for all samples.
        feature_lens: (batch_size,) per-sample frame counts.
        n_window: half the target chunk size.

    Returns:
        padded_feature: (num_chunks, feature_dim, max_chunk_len)
        chunk_lengths: (num_chunks,) actual length of each chunk.
    """
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.full((chunk_num.sum().item(),), n_window * 2, dtype=torch.long, device=feature_lens.device)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    return padded_feature, chunk_lengths


def get_valid_indices(
    chunk_lengths: torch.Tensor,
    n_window: int,
) -> torch.Tensor:
    """Flat indices of valid (non-padding) positions after CNN."""
    feature_lens_after_cnn = _post_cnn_length(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=chunk_lengths.device) < feature_lens_after_cnn.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_audio_cu_seqlens(
    chunk_lengths: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window_infer: int,
    n_window: int,
) -> torch.Tensor:
    """Cumulative sequence lengths for windowed flash attention."""
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens, n_window)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths, n_window)
    max_len_after_cnn = feature_lens_after_cnn.max().item()

    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio

    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len_val = cnn_len.item() if hasattr(cnn_len, 'item') else int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len_val // window_aftercnn)
        remainder = cnn_len_val % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]

    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Qwen3AudioEncoder(PreTrainedModel):
    """
    Qwen3 Audio Encoder.

    3x Conv2d (8x temporal downsample) + sinusoidal positional embedding
    + N transformer encoder layers + LayerNorm.

    Input: (feature_dim, total_frames) packed mel + (batch,) feature_lens
           OR (batch, mel_bins, padded_len) batched mel + feature_lens
    Output: BaseModelOutput with last_hidden_state = (total_valid_tokens, d_model)
    """

    config_class = Qwen3AudioEncoderConfig
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    _no_split_modules = ["Qwen3AudioEncoderLayer"]

    def __init__(self, config: Qwen3AudioEncoderConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window

        # Positional embedding — use max_source_positions if available, else max_position_embeddings
        pos_emb_len = getattr(config, "max_source_positions", None) or config.max_position_embeddings
        self.positional_embedding = SinusoidsPositionEmbedding(pos_emb_len, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [Qwen3AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)

        # Conv2d stem: 3 layers, each stride 2 → total 8x temporal downsample
        dhs = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, dhs, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(dhs, dhs, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(dhs, dhs, 3, 2, padding=1)
        # Linear projection from flattened conv output to d_model
        freq_after_conv = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(dhs * freq_after_conv, config.d_model, bias=False)

        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = getattr(config, "conv_chunksize", 500)
        self.gradient_checkpointing = False

        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidsPositionEmbedding):
            pos_emb = module._compute_embedding()
            module.positional_embedding.copy_(pos_emb)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        """
        Args:
            input_features: Either:
                - (feature_dim, total_frames) packed mel (Qwen3 Omni style)
                - (batch, mel_bins, padded_len) batched mel (Whisper style)
            feature_lens: (batch_size,) per-sample valid frame counts.
            input_features_mask: (batch, padded_len) alternative to feature_lens.
        """
        # Normalize input format
        if input_features.ndim == 3:
            # Batched format (B, mel_bins, T) — convert to packed (mel_bins, total_frames)
            batch_size = input_features.shape[0]
            if feature_lens is None and input_features_mask is not None:
                feature_lens = input_features_mask.sum(-1).long()
            elif feature_lens is None:
                feature_lens = torch.full(
                    (batch_size,), input_features.shape[2],
                    dtype=torch.long, device=input_features.device,
                )
            # Concatenate valid frames from each sample
            parts = []
            for i in range(batch_size):
                length = feature_lens[i].item()
                parts.append(input_features[i, :, :length])
            input_features = torch.cat(parts, dim=1)  # (mel_bins, total_frames)
        elif input_features.ndim == 2:
            # Already packed (feature_dim, total_frames)
            if feature_lens is None:
                raise ValueError("feature_lens is required for packed input")
        else:
            raise ValueError(f"Unexpected input_features ndim={input_features.ndim}")

        # Chunk and pad
        padded_feature, chunk_lengths = chunk_and_pad_features(
            input_features, feature_lens, self.n_window
        )
        valid_indices = get_valid_indices(chunk_lengths, self.n_window)
        cu_seqlens = get_audio_cu_seqlens(
            chunk_lengths, feature_lens, self.n_window_infer, self.n_window
        )
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # Conv2d: (num_chunks, 1, mel_bins, chunk_len)
        padded_feature = padded_feature.unsqueeze(1).to(dtype=self.conv2d1.weight.dtype)

        # Process in chunks to avoid OOM
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embedding
        positional_embedding = (
            self.positional_embedding.positional_embedding[:padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        # Select valid (non-padding) positions into flat packed sequence
        hidden_states = torch.index_select(
            padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices
        )

        # Ulysses SP: slice hidden_states before transformer layers so each rank
        # only computes a portion. Extend cu_seqlens for the SP-padding tail.
        sp_enabled = self.training and get_parallel_state() is not None and get_parallel_state().sp_enabled
        if sp_enabled:
            unpadded_hidden_len = hidden_states.shape[0]
            hidden_states = slice_input_tensor(
                hidden_states, dim=0, group=get_parallel_state().ulysses_group, padding=True
            )
            # The slice may pad; add a cu_seqlens entry for the padded tail so
            # varlen attention treats it as an independent (zero-length) sequence.
            pad_seq_len = hidden_states.shape[0] * get_parallel_state().sp_size - unpadded_hidden_len
            if pad_seq_len > 0:
                cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)], dim=0)

        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, cu_seqlens, use_reentrant=False,
                )
            else:
                layer_outputs = layer(hidden_states, cu_seqlens, max_seqlen=max_seqlen)
            hidden_states = layer_outputs[0]

        # Ulysses SP: gather back to full sequence after transformer layers
        if sp_enabled:
            hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().ulysses_group)
            sp_padding_size = hidden_states.shape[0] - unpadded_hidden_len
            if sp_padding_size > 0:
                hidden_states = hidden_states[:unpadded_hidden_len]

        hidden_states = self.ln_post(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_output_lengths(self, feature_lens: torch.Tensor) -> torch.Tensor:
        """Compute per-sample output token counts after the conv stack."""
        return _get_feat_extract_output_lengths(feature_lens, self.n_window)
