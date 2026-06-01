from typing import Dict

import torch
import torch.nn as nn

from veomni.models.custom.audio_encoder.modeling_whisper import WhisperEncoder, WhisperConfig
from veomni.models.custom.llava_qwen3moe.base import BaseEncoderModelMixin, BaseEncoderConfigMixin
from veomni.models.custom.llava_qwen3moe.projector import build_audio_projector
from transformers import AutoConfig


class BeeBeeAudioModelConfig(BaseEncoderConfigMixin, WhisperConfig):
    model_type = "beebee_audio_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_audio_projector=False,
        audio_downsample_size=10,
        audio_projector_type="conv_channel_upscale",
        output_size=6144,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_hidden_states = return_hidden_states
        self.train_audio_projector = train_audio_projector
        self.audio_downsample_size = audio_downsample_size
        self.audio_projector_type = audio_projector_type
        self.output_size = output_size
        


class BeeBeeVLAudioModel(BaseEncoderModelMixin, WhisperEncoder):
    # Some Whisper checkpoints are saved from a full seq2seq model, where weights are prefixed with
    # `model.encoder.*`. Our encoder-only model expects `conv1/*`, `layers/*`, etc. directly.
    # `load_model_weights()` will consult this mapping (if present) to rewrite checkpoint keys.
    _checkpoint_conversion_mapping = {
        r"^model\.encoder\.": "",
        r"^encoder\.": "",
    }

    config_class = BeeBeeAudioModelConfig
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: BeeBeeAudioModelConfig):
        super().__init__(config)
        self.config = config
        self.freeze_audio_encoder = False
        self.freeze_audio_projector = False
        self.audio_projector = build_audio_projector(projector_type=config.audio_projector_type, 
                                                    encoder_hidden=config.d_model, 
                                                    out_hidden=config.output_size, 
                                                    downsample_ratio=self.config.audio_downsample_size)

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        self.audio_projector.requires_grad_(True)
       
    
    def lm_encode(self, features: torch.Tensor, feature_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.freeze_audio_encoder:
            with torch.no_grad():
                hidden_state = super().forward(input_features=features, input_seq_lens=feature_lengths).last_hidden_state # [b, max_seq, hidden]
        else:
            hidden_state = super().forward(input_features=features, input_seq_lens=feature_lengths).last_hidden_state # [b, max_seq, hidden]

        if self.freeze_audio_projector:
            with torch.no_grad():
                hidden_state, seq_len = self.audio_projector(hidden_state, feature_lengths)
        else:
            hidden_state, seq_len = self.audio_projector(hidden_state, feature_lengths)
        return hidden_state, seq_len


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
    
AutoConfig.register("beebee_audio_model", BeeBeeAudioModelConfig)
