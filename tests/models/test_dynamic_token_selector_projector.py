import math

import torch
import torch.nn.functional as F

from veomni.models.custom.llava_qwen3moe.projector_token_selector import (
    DynamicTokenSelectorProjector,
    get_adaptive_pool_size,
)


def _adaptive_avgpool_reference(features, grids, merge_sizes, ratios):
    outputs = []
    seq_len = []
    offset = 0
    for (time, raw_height, raw_width), merge_size, ratio in zip(grids.tolist(), merge_sizes, ratios):
        height = raw_height // merge_size
        width = raw_width // merge_size
        length = time * height * width
        grid_features = features[offset : offset + length].view(time, height, width, -1)
        output_height, output_width = get_adaptive_pool_size(height, width, ratio)
        pooled = F.adaptive_avg_pool2d(
            grid_features.permute(3, 0, 1, 2), (output_height, output_width)
        )
        outputs.append(pooled.permute(1, 2, 3, 0).reshape(-1, features.shape[-1]))
        seq_len.extend([output_height * output_width] * time)
        offset += length
    return torch.cat(outputs), seq_len


def test_zero_initialized_selector_matches_dynamic_average_pooling():
    torch.manual_seed(0)
    grids = torch.tensor([[2, 10, 14], [1, 12, 8]])
    merge_sizes = [2, 2]
    ratios = [4.0, 12.0]
    merged_lengths = [
        time * (height // merge_size) * (width // merge_size)
        for (time, height, width), merge_size in zip(grids.tolist(), merge_sizes)
    ]
    features = torch.randn(sum(merged_lengths), 16)
    projector = DynamicTokenSelectorProjector(16, 16, downsample_ratio=4, selector_hidden=8)

    selected, seq_len, aux = projector.select_tokens(
        features,
        grids,
        merge_size=merge_sizes,
        downsample_ratios=ratios,
    )
    expected, expected_seq_len = _adaptive_avgpool_reference(features, grids, merge_sizes, ratios)

    torch.testing.assert_close(selected, expected)
    assert seq_len == expected_seq_len
    assert selected.shape[0] == sum(expected_seq_len)
    assert aux.cell_offsets.tolist() == [0, sum(expected_seq_len[:2]), sum(expected_seq_len)]
    for cell_index in range(selected.shape[0]):
        cell_weights = aux.weights[aux.cell_indices == cell_index]
        torch.testing.assert_close(cell_weights.sum(), torch.tensor(1.0))


def test_packed_grids_can_emit_different_token_counts():
    grids = torch.tensor([[2, 24, 24], [3, 20, 28]])
    merge_sizes = [2, 2]
    ratios = [4.0, 12.0]
    feature_count = sum(
        time * (height // merge_size) * (width // merge_size)
        for (time, height, width), merge_size in zip(grids.tolist(), merge_sizes)
    )
    projector = DynamicTokenSelectorProjector(8, 12, downsample_ratio=4, selector_hidden=4)
    output, seq_len, aux = projector(
        torch.randn(feature_count, 8),
        grids,
        merge_size=merge_sizes,
        downsample_ratios=ratios,
        return_selector_aux=True,
    )

    first_height, first_width = get_adaptive_pool_size(12, 12, 4)
    second_height, second_width = get_adaptive_pool_size(10, 14, 12)
    first_tokens = 2 * first_height * first_width
    second_tokens = 3 * second_height * second_width

    assert first_tokens != second_tokens
    assert aux.cell_offsets.tolist() == [0, first_tokens, first_tokens + second_tokens]
    assert output.shape == (first_tokens + second_tokens, 12)
    assert seq_len == [first_height * first_width] * 2 + [second_height * second_width] * 3


def test_selector_receives_gradients_from_projected_tokens():
    torch.manual_seed(1)
    grids = torch.tensor([[1, 8, 8]])
    features = torch.randn(16, 8, requires_grad=True)
    projector = DynamicTokenSelectorProjector(8, 6, downsample_ratio=4, selector_hidden=4)

    output, _ = projector(features, grids)
    coefficients = torch.arange(output.numel(), dtype=output.dtype).view_as(output)
    loss = (output * coefficients).sum()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert projector.score_proj.weight.grad is not None
    assert projector.score_proj.weight.grad.abs().sum() > 0


def test_nonuniform_selector_uses_exact_adaptive_windows():
    torch.manual_seed(2)
    grids = torch.tensor([[1, 10, 14]])
    features = torch.randn(35, 8)
    projector = DynamicTokenSelectorProjector(8, 8, downsample_ratio=4, selector_hidden=4)
    with torch.no_grad():
        projector.score_proj.weight.fill_(0.25)

    selected, _, aux = projector.select_tokens(features, grids)
    expected = torch.zeros_like(selected)
    expected.index_add_(
        0,
        aux.cell_indices,
        features.index_select(0, aux.candidate_indices) * aux.weights.unsqueeze(-1),
    )

    torch.testing.assert_close(selected, expected)
    assert aux.candidate_indices.numel() > features.shape[0]
    assert aux.candidate_indices.unique().numel() == features.shape[0]


def test_meta_materialization_restores_zero_init_and_shard_dimension():
    with torch.device("meta"):
        projector = DynamicTokenSelectorProjector(8, 8, downsample_ratio=4, selector_hidden=4)

    assert projector.score_proj.weight.is_meta
    projector.to_empty(device="cpu")

    assert torch.count_nonzero(projector.score_proj.weight) == 0
    assert projector.score_proj.weight._veomni_fsdp_shard_dim == 1

    with torch.no_grad():
        projector.score_proj.weight.fill_(1)
    projector.score_proj.reset_parameters()
    assert torch.count_nonzero(projector.score_proj.weight) == 0
    assert projector.score_proj.weight._veomni_fsdp_shard_dim == 1


def test_owning_model_linear_initialization_preserves_uniform_selector_start():
    with torch.device("meta"):
        projector = DynamicTokenSelectorProjector(8, 8, downsample_ratio=4, selector_hidden=4)

    projector.to_empty(device="cpu")

    # Mirrors HF initialization dispatch after VeOmni resets every module's
    # _is_hf_initialized flag on the random-init path.
    for module in projector.modules():
        module._is_hf_initialized = False
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)
        module._is_hf_initialized = True

    assert torch.count_nonzero(projector.key_proj.weight) > 0
    assert torch.count_nonzero(projector.score_proj.weight) == 0
    assert projector.score_proj.weight._veomni_fsdp_shard_dim == 1


def test_nonempty_features_with_empty_grid_are_rejected():
    projector = DynamicTokenSelectorProjector(8, 8, downsample_ratio=4, selector_hidden=4)

    try:
        projector(torch.randn(1, 8), torch.empty((0, 3), dtype=torch.long))
    except ValueError as error:
        assert "must be empty" in str(error)
    else:
        raise AssertionError("Expected an empty-grid validation error.")


def test_invalid_feature_length_is_rejected():
    projector = DynamicTokenSelectorProjector(8, 8, downsample_ratio=4, selector_hidden=4)
    grids = torch.tensor([[1, 8, 8]])
    features = torch.randn(math.prod((1, 4, 4)) - 1, 8)

    try:
        projector(features, grids)
    except ValueError as error:
        assert "images_feature length" in str(error)
    else:
        raise AssertionError("Expected a feature-length validation error.")
