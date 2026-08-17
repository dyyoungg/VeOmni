"""
BeeBeeVL wrapper for Qwen3 Audio Encoder.

Follows the same pattern as modeling_audio_encoder.py (Whisper wrapper):
  - Config: BeeBeeQwen3AudioModelConfig
  - Model: BeeBeeVLQwen3AudioModel
  - Unified lm_encode(features, feature_lengths) -> (packed_hidden, seq_lengths)
"""

from typing import Dict

import torch
import torch.nn as nn

from veomni.models.custom.audio_encoder.modeling_qwen3_audio import (
    Qwen3AudioEncoder,
    Qwen3AudioEncoderConfig,
    _get_feat_extract_output_lengths,
)
from veomni.models.custom.llava_qwen3moe.base import BaseEncoderModelMixin, BaseEncoderConfigMixin
from veomni.models.custom.llava_qwen3moe.projector import build_audio_projector
from transformers import AutoConfig


class BeeBeeQwen3AudioModelConfig(BaseEncoderConfigMixin, Qwen3AudioEncoderConfig):
    model_type = "beebee_qwen3_audio_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_audio_projector=False,
        audio_downsample_size=2,
        audio_projector_type="multi_conv",
        output_size=5120,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_hidden_states = return_hidden_states
        self.train_audio_projector = train_audio_projector
        self.audio_downsample_size = audio_downsample_size
        self.audio_projector_type = audio_projector_type
        self.output_size = output_size
        # Audio encoder 没有 word embeddings，禁用 tie_word_embeddings 避免 load_model_weights 报错
        self.tie_word_embeddings = False


class BeeBeeVLQwen3AudioModel(BaseEncoderModelMixin, Qwen3AudioEncoder):
    """Qwen3 audio encoder + pluggable projector with downsample support."""

    # Checkpoint key rewriting for weight loading
    _checkpoint_conversion_mapping = {
        r"^thinker\.audio_tower\.": "",
        r"^model\.encoder\.": "",
        r"^encoder\.": "",
        r"^audio_tower\.": "",
    }

    config_class = BeeBeeQwen3AudioModelConfig
    _supports_flash_attn_2 = True
    _no_split_modules = ["Qwen3AudioEncoderLayer"]

    def __init__(self, config: BeeBeeQwen3AudioModelConfig):
        super().__init__(config)
        self.config = config
        self.freeze_audio_encoder = False
        self.freeze_audio_projector = False
        self.audio_projector = build_audio_projector(
            projector_type=config.audio_projector_type,
            encoder_hidden=config.d_model,
            out_hidden=config.output_size,
            downsample_ratio=config.audio_downsample_size,
        )

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        self.audio_projector.requires_grad_(True)

    def lm_encode(
        self, features: torch.Tensor, feature_lengths: torch.Tensor, **kwargs
    ) -> tuple:
        """
        Encode audio features and project to LLM hidden dimension.

        Args:
            features: (B, 128, T) mel spectrogram (same format as Whisper pipeline).
            feature_lengths: (B,) number of valid mel frames per sample.

        Returns:
            hidden_state: (total_tokens, output_size) packed projected features.
            num_tokens: list[int] per-sample token counts after projector.
        """
        # Encoder forward
        if self.freeze_audio_encoder:
            with torch.no_grad():
                encoder_output = super().forward(
                    input_features=features, feature_lens=feature_lengths
                )
        else:
            encoder_output = super().forward(
                input_features=features, feature_lens=feature_lengths
            )

        # encoder output: (total_valid_tokens, d_model) — packed flat sequence
        hidden_states = encoder_output.last_hidden_state

        # Compute per-sample output lengths from the encoder conv stack
        per_sample_lens = _get_feat_extract_output_lengths(feature_lengths, self.config.n_window)

        # Unpack to batched (B, max_seq, d_model) for projector
        batch_size = features.shape[0]
        per_sample_lens_list = per_sample_lens.tolist()
        max_seq_len = max(per_sample_lens_list)

        batched = torch.zeros(
            batch_size, max_seq_len, hidden_states.shape[-1],
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        offset = 0
        for i, length in enumerate(per_sample_lens_list):
            length = int(length)
            batched[i, :length, :] = hidden_states[offset : offset + length]
            offset += length

        # Projector: (B, max_seq, d_model) -> packed (total_proj_tokens, output_size)
        if self.freeze_audio_projector:
            with torch.no_grad():
                hidden_state, num_tokens = self.audio_projector(batched, per_sample_lens_list)
        else:
            hidden_state, num_tokens = self.audio_projector(batched, per_sample_lens_list)

        return hidden_state, num_tokens

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        features = torch.zeros(2, 128, 3000).to(dtype=self.dtype, device=self.device)
        feature_lens = torch.tensor([50, 50], dtype=torch.int64, device=self.device)
        return {"features": features, "feature_lengths": feature_lens}

    def dummy_forward(self):
        if getattr(self, "_dummy_data", None) is None:
            features = torch.zeros(2, 128, 3000).to(dtype=self.dtype, device=self.device)
            feature_lens = torch.tensor([50, 50], dtype=torch.int64, device=self.device)
            self._dummy_data = {"features": features, "feature_lengths": feature_lens}
        return self.lm_encode(**self._dummy_data)


AutoConfig.register("beebee_qwen3_audio_model", BeeBeeQwen3AudioModelConfig)


if __name__ == "__main__":
    import math
    from safetensors.torch import load_file

    CKPT_PATH = "/mnt/afs/share/Qwen3-Omni-AudioTransformer"
    SAMPLE_RATE = 16000
    N_WINDOW = 50
    AUDIO_FRAME_LENGTH = 320  # whisper hop_length=160, 但 qwen3 audio 每帧对应约 10ms

    # ---- 理论公式：encoder conv (8x) + projector downsample ----
    def calc_encoder_output_tokens(raw_mel_len, n_window=50):
        """encoder conv stack 输出 token 数 (无 projector)"""
        return _get_feat_extract_output_lengths(
            torch.tensor([raw_mel_len]), n_window
        ).item()

    def calc_projector_output_tokens(encoder_tokens, downsample_ratio):
        """projector (multi_conv) 额外下采样后的 token 数"""
        return (encoder_tokens + downsample_ratio - 1) // downsample_ratio

    # ---- 构建模型 ----
    base_cfg = AutoConfig.from_pretrained(CKPT_PATH, trust_remote_code=True)
    cfg_dict = base_cfg.to_dict()
    cfg_dict.pop("model_type", None)

    AUDIO_DOWNSAMPLE_SIZE = 2
    AUDIO_PROJECTOR_TYPE = "multi_conv"
    OUTPUT_SIZE = 5120

    config = BeeBeeQwen3AudioModelConfig(
        **cfg_dict,
        output_size=OUTPUT_SIZE,
        audio_downsample_size=AUDIO_DOWNSAMPLE_SIZE,
        audio_projector_type=AUDIO_PROJECTOR_TYPE,
    )

    device = "cuda"
    dtype = torch.bfloat16
    model = BeeBeeVLQwen3AudioModel._from_config(config, dtype=dtype).to(device)

    # 加载 encoder 权重 (跳过 proj1/proj2)
    state_dict = load_file(f"{CKPT_PATH}/model.safetensors")
    encoder_weights = {k: v for k, v in state_dict.items() if not k.startswith("proj")}
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
    print(f"=== Weight loading ===")
    print(f"  Missing (non-projector): {[k for k in missing if 'audio_projector' not in k]}")
    print(f"  Missing (projector, expected): {[k for k in missing if 'audio_projector' in k]}")
    print(f"  Unexpected: {unexpected}")
    print(f"  d_model={config.d_model}, encoder_layers={config.encoder_layers}, "
          f"heads={config.encoder_attention_heads}")
    print(f"  projector: type={AUDIO_PROJECTOR_TYPE}, downsample={AUDIO_DOWNSAMPLE_SIZE}, "
          f"output_size={OUTPUT_SIZE}")
    print()

    # ---- 测试用例：不同音频长度 ----
    test_cases = [
        # (描述, raw_audio_samples)
        ("1s audio",   1 * SAMPLE_RATE),
        ("5s audio",   5 * SAMPLE_RATE),
        ("10s audio", 10 * SAMPLE_RATE),
        ("30s audio", 30 * SAMPLE_RATE),
        ("60s audio", 60 * SAMPLE_RATE),
    ]

    print(f"{'Case':<12} {'mel_len':>8} {'enc_theory':>11} {'enc_actual':>11} "
          f"{'proj_theory':>12} {'proj_actual':>12} {'match':>6}")
    print("-" * 80)

    for desc, raw_samples in test_cases:
        # mel 帧数 = ceil(raw_samples / hop_length)，whisper hop=160
        mel_len = math.ceil(raw_samples / 160)

        # 理论值
        enc_theory = calc_encoder_output_tokens(mel_len, N_WINDOW)
        proj_theory = calc_projector_output_tokens(enc_theory, AUDIO_DOWNSAMPLE_SIZE)

        # 构造输入: (1, 128, mel_len) — 与 whisper pipeline 格式一致
        features = torch.randn(1, 128, mel_len, device=device, dtype=dtype)
        feature_lens = torch.tensor([mel_len], device=device, dtype=torch.long)

        with torch.no_grad():
            hidden, num_tokens = model.lm_encode(features=features, feature_lengths=feature_lens)

        enc_actual = _get_feat_extract_output_lengths(feature_lens, N_WINDOW).item()
        proj_actual = num_tokens[0]
        match = "✅" if proj_actual == proj_theory else "❌"

        print(f"{desc:<12} {mel_len:>8} {enc_theory:>11} {enc_actual:>11} "
              f"{proj_theory:>12} {proj_actual:>12} {match:>6}")

    # ---- batch 测试 ----
    print(f"\n=== Batch test (2 samples, different lengths) ===")
    mel_lens = [math.ceil(3 * SAMPLE_RATE / 160), math.ceil(7 * SAMPLE_RATE / 160)]
    max_mel = max(mel_lens)
    batch_features = torch.zeros(2, 128, max_mel, device=device, dtype=dtype)
    for i, ml in enumerate(mel_lens):
        batch_features[i, :, :ml] = torch.randn(128, ml, device=device, dtype=dtype)
    batch_lens = torch.tensor(mel_lens, device=device, dtype=torch.long)

    with torch.no_grad():
        hidden, num_tokens = model.lm_encode(features=batch_features, feature_lengths=batch_lens)

    print(f"  Input mel lengths: {mel_lens}")
    print(f"  Encoder output tokens (per sample): "
          f"{_get_feat_extract_output_lengths(batch_lens, N_WINDOW).tolist()}")
    print(f"  Projector output tokens (per sample): {num_tokens}")
    print(f"  Total packed hidden shape: {hidden.shape}")
    print(f"  Expected total: {sum(num_tokens)}, actual dim0: {hidden.shape[0]}")
    assert hidden.shape[0] == sum(num_tokens), "packed token count mismatch!"
    assert hidden.shape[1] == OUTPUT_SIZE, f"output dim mismatch: {hidden.shape[1]} vs {OUTPUT_SIZE}"
    print("  ✅ All assertions passed!")
