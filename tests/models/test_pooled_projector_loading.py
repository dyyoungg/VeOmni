"""Factory and real VeOmni missing-key/roundtrip checkpoint loading checks."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from veomni.models.custom.llava_qwen3moe.projector import DynamicAvgPoolProjector, build_image_projector
from veomni.models.custom.llava_qwen3moe.projector_pooled_query_attention import (
    PooledQueryAttentionProjector,
    PoolingResidualAttentionProjector,
)
from veomni.models.module_utils import load_model_weights


class ProjectorHolder(torch.nn.Module):
    def __init__(self, projector_type="pooled_query_attention"):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.kept = torch.nn.Parameter(torch.empty(3))
        self.mm_projector = build_image_projector(
            projector_type,
            32,
            16,
            4,
            num_attention_heads=4,
        )


@pytest.mark.parametrize(
    "projector_type,projector_class",
    [
        ("pooled_query_attention", PooledQueryAttentionProjector),
        ("pooling_residual_attention", PoolingResidualAttentionProjector),
    ],
)
def test_builder_preserves_avgpool_and_supports_checkpoint_dimensions(projector_type, projector_class):
    with torch.device("meta"):
        old = build_image_projector("dynamic_avgpool", 4608, 2048, 4)
        new = build_image_projector(projector_type, 4608, 2048, 4)
    assert isinstance(old, DynamicAvgPoolProjector)
    assert type(new) is projector_class
    assert new.num_attention_heads == 32
    assert new.head_dim == 144


@pytest.mark.parametrize("projector_type", ["pooled_query_attention", "pooling_residual_attention"])
def test_missing_projector_initializes_and_complete_checkpoint_preserves_trained_weights(projector_type):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_file({"kept": torch.tensor([1.0, 2.0, 3.0])}, str(root / "model.safetensors"), metadata={"format": "pt"})
        with torch.device("meta"):
            model = ProjectorHolder(projector_type)
        load_model_weights(model, str(root), init_device="cpu")
        torch.testing.assert_close(model.kept, torch.tensor([1.0, 2.0, 3.0]))
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
            restored = ProjectorHolder(projector_type)
        load_model_weights(restored, str(root), init_device="cpu")
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        restored.to(torch.bfloat16)
        assert torch.all(restored.mm_projector.attention_out.weight == 0.125)


@pytest.mark.parametrize(
    "projector_type,projector_class",
    [
        ("pooled_query_attention", PooledQueryAttentionProjector),
        ("pooling_residual_attention", PoolingResidualAttentionProjector),
    ],
)
def test_real_source_config_builds_new_vision_projector(projector_type, projector_class):
    from veomni.models.custom.vision_encoder.modeling_qwen35_vision_encoder import (
        BeeBeeVLQwen35MoeVisionModel,
        BeeBeeVLQwen35MoeVisionModelConfig,
    )

    config = BeeBeeVLQwen35MoeVisionModelConfig(
        hidden_size=1152,
        intermediate_size=4304,
        depth=1,
        num_heads=16,
        spatial_merge_size=2,
        output_size=2048,
        image_projector_type=projector_type,
        image_projector_num_attention_heads=32,
        image_projector_attention_dropout=0.0,
    )
    with torch.device("meta"):
        vision = BeeBeeVLQwen35MoeVisionModel(config)
    assert type(vision.mm_projector) is projector_class
    assert vision.mm_projector.head_dim == 144
    assert json.loads(config.to_json_string())["image_projector_num_attention_heads"] == 32
