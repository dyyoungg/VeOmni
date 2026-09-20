"""Factory and real VeOmni missing-key/roundtrip checkpoint loading checks."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from veomni.models.custom.llava_qwen3moe.projector import DynamicAvgPoolProjector, build_image_projector
from veomni.models.custom.llava_qwen3moe.projector_frame_attention import FrameVarlenAttentionProjector
from veomni.models.module_utils import load_model_weights


class ProjectorHolder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.kept = torch.nn.Parameter(torch.empty(3))
        self.mm_projector = build_image_projector(
            "frame_varlen_attention", 32, 16, 4, num_attention_heads=4,
        )


def test_builder_preserves_avgpool_and_supports_checkpoint_dimensions():
    with torch.device("meta"):
        old = build_image_projector("dynamic_avgpool", 4608, 2048, 4)
        new = build_image_projector("frame_varlen_attention", 4608, 2048, 4)
    assert isinstance(old, DynamicAvgPoolProjector)
    assert isinstance(new, FrameVarlenAttentionProjector)
    assert new.num_attention_heads == 32
    assert new.head_dim == 144


def test_missing_projector_initializes_and_complete_checkpoint_preserves_trained_weights():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_file({"kept": torch.tensor([1., 2., 3.])}, str(root / "model.safetensors"), metadata={"format": "pt"})
        with torch.device("meta"):
            model = ProjectorHolder()
        load_model_weights(model, str(root), init_device="cpu")
        torch.testing.assert_close(model.kept, torch.tensor([1., 2., 3.]))
        for name, parameter in model.mm_projector.named_parameters():
            assert not parameter.is_meta and torch.isfinite(parameter).all(), name
            if name == "attention_out.weight" or name.endswith("bias"):
                assert torch.count_nonzero(parameter) == 0, name
            elif "norm.weight" in name:
                torch.testing.assert_close(parameter, torch.ones_like(parameter))
            else:
                assert parameter.std() > 0, name
        with torch.no_grad():
            model.mm_projector.attention_out.weight.fill_(0.125)
        reference = {name: value.detach().clone() for name, value in model.state_dict().items()}
        save_file(reference, str(root / "model.safetensors"), metadata={"format": "pt"})
        with torch.device("meta"):
            restored = ProjectorHolder()
        load_model_weights(restored, str(root), init_device="cpu")
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        restored.to(torch.bfloat16)
        assert torch.all(restored.mm_projector.attention_out.weight == 0.125)


def test_real_source_config_builds_new_vision_projector():
    from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import (
        BeeBeeVLQwen35MoeVisionModel,
        BeeBeeVLQwen35MoeVisionModelConfig,
    )

    config = BeeBeeVLQwen35MoeVisionModelConfig(
        hidden_size=1152, intermediate_size=4304, depth=1, num_heads=16,
        spatial_merge_size=2, output_size=2048, image_projector_type="frame_varlen_attention",
        image_projector_num_attention_heads=32, image_projector_attention_dropout=0.0,
    )
    with torch.device("meta"):
        vision = BeeBeeVLQwen35MoeVisionModel(config)
    assert isinstance(vision.mm_projector, FrameVarlenAttentionProjector)
    assert vision.mm_projector.head_dim == 144
    assert json.loads(config.to_json_string())["image_projector_num_attention_heads"] == 32
