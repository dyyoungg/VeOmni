from typing import Dict

import torch
import torch.nn as nn

from veomni.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from veomni.models.custom.llava_qwen3moe.projector import build_image_projector
from veomni.models.custom.llava_qwen3moe.base import BaseEncoderModelMixin, BaseEncoderConfigMixin
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm


class BeeBeeVLVisionModelConfig(BaseEncoderConfigMixin, Qwen2_5_VLVisionConfig):
    model_type = "beebee_vl_vision_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_origin_projector=False,
        image_downsample_size=8,
        mm_projector_type="dynamic_avgpool",
        output_size=6144,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        self.train_origin_projector = train_origin_projector
        self.image_downsample_size = image_downsample_size
        self.mm_projector_type = mm_projector_type
        self.output_size = output_size
        super().__init__(**kwargs)

class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return x
    
class Qwen25ViTPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.merger = Qwen2_5_VLPatchMerger(context_dim=config.hidden_size,
                                            spatial_merge_size=config.spatial_merge_size)


class BeeBeeVLVisionModel(BaseEncoderModelMixin, Qwen25ViTPretrainedModel):
    config_class = BeeBeeVLVisionModelConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config: BeeBeeVLVisionModelConfig):
        super().__init__(config)
        self.config = config
        self.mm_projector = build_image_projector(config.out_hidden_size, config.output_size)

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        self.mm_projector.requires_grad_(True)
        if self.config.freeze_vision_merger:
            self.merger.requires_grad_(False)
        else:
            self.merger.requires_grad_(True)
    
    def lm_encode(self, features: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(features, grid_thw)

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((1196, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device) #  [644,364]
        grid_thw = torch.tensor([[1, 26, 46]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}
