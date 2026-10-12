"""Experimental content-aware visual token compression.

This module is intentionally separate from ``projector.py`` so experiments can
load and validate the selector without changing an active training run.

Read ``select_tokens`` first for the main computation, then the metadata helpers.
Data flow: packed features -> adaptive cells -> scores -> cell-local softmax
-> weighted feature sums -> projector MLP -> packed LLM visual tokens.

Here "selector" means differentiable weighted pooling, not hard top-k selection.
The supplied compression ratio determines how many cells (output tokens) a grid
has; the learned scorer determines what information each cell retains. It does
not learn the token budget itself, and it does not pool across time steps.

Shape notation used below:
    G: visual grids (one text sample may contain several grids).
    N: input tokens after the vision encoder's spatial merger.
    C: total output cells/tokens across all grids.
    E: candidate memberships across cells; overlapping cells can make E > N.
    D: encoder_hidden; S: selector_hidden; L: out_hidden.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state


def get_adaptive_pool_size(height: int, width: int, scale: float) -> tuple[int, int]:
    """Convert an area compression factor into a spatial output grid.

    Both axes shrink by sqrt(scale); time is unchanged. For example, a 14x14
    merged grid with scale=4 emits 7x7 tokens per time step. Rounding and the
    minimum size of 1 mean the realized token ratio can differ from scale.
    """
    ratio = 1 / math.sqrt(scale)
    return max(1, round(height * ratio)), max(1, round(width * ratio))


@dataclass
class TokenSelectorAux:
    """Ragged selector state for diagnostics or future teacher supervision.

    logits, weights: [E], one score/weight per candidate membership.
    candidate_indices: [E], source row in packed images_feature for each entry.
    cell_indices: [E], destination output cell for each entry.
    cell_offsets: [G + 1], prefix sums delimiting each grid's output cells.
    cell_counts: [C], number of candidate entries contributing to each cell.

    Example: candidate_indices=[0, 1, 1, 2], cell_indices=[0, 0, 1, 1]
    represents two cells, {x0, x1} and {x1, x2}. The shared x1 receives a
    separate weight in each cell. No teacher loss is computed by this module.
    """

    logits: torch.Tensor
    weights: torch.Tensor
    candidate_indices: torch.LongTensor
    cell_indices: torch.LongTensor
    cell_offsets: torch.LongTensor
    cell_counts: torch.LongTensor


class _ZeroInitScoreProjection(nn.Module):
    """Scalar projection whose zero initialization survives owning-model init."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, hidden_size))
        self.weight._veomni_fsdp_shard_dim = 1

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        self.weight._veomni_fsdp_shard_dim = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class DynamicTokenSelectorProjector(nn.Module):
    """Compress visual features with learned, cell-local weighted pooling.

    Each adaptive pooling cell emits one token. Cells from all packed visual
    grids are flattened into one ragged collection, so trainable selector
    modules are invoked a fixed number of times regardless of how many samples
    or grids are packed on a data-parallel rank with a nonempty visual input.
    The empty-grid path skips the selector; distributed callers still need to
    arrange consistent module participation (for example, with dummy inputs).

    For grid i, C_i = T_i * output_height_i * output_width_i. Each grid may
    have a different C_i; concatenation produces [sum(C_i), D] without padding
    samples to a shared K. Parameters are shared across all compression ratios.

    The selector score projection is zero-initialized. At initialization all
    candidates within a cell receive equal weight, making the selected features
    equivalent to adaptive average pooling before the projector MLP.

    Sequence parallelism is not supported yet. The current vision encoder gives
    the projector a full sequence with a sharded hidden dimension under Ulysses;
    content-dependent selector scores must be synchronized across those hidden
    shards before this module can be used in that mode.
    """

    def __init__(
        self,
        encoder_hidden: int,
        out_hidden: int,
        downsample_ratio: float,
        selector_hidden: int = 128,
        selector_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if encoder_hidden <= 0 or out_hidden <= 0 or selector_hidden <= 0:
            raise ValueError("Projector dimensions must be positive.")
        if downsample_ratio <= 0:
            raise ValueError("downsample_ratio must be positive.")
        if selector_temperature <= 0:
            raise ValueError("selector_temperature must be positive.")

        self.mm_downsample_ratio = downsample_ratio
        self.hidden_size = encoder_hidden
        self.merge_size = 2
        self.selector_temperature = selector_temperature

        # Additive attention in a small S-dimensional scoring space:
        # key = Wk LN(x); context = Wq LN(cell_mean) + Wr log2(rate) + Wp position.
        # Values remain in the original D-dimensional feature space.
        self.selector_norm = nn.LayerNorm(encoder_hidden)
        self.key_proj = nn.Linear(encoder_hidden, selector_hidden, bias=False)
        self.query_proj = nn.Linear(encoder_hidden, selector_hidden, bias=False)
        self.rate_proj = nn.Linear(1, selector_hidden, bias=False)
        self.position_proj = nn.Linear(3, selector_hidden, bias=False)
        # This is intentionally not nn.Linear: an owning PreTrainedModel's
        # generic Linear initializer must not overwrite the uniform start.
        self.score_proj = _ZeroInitScoreProjection(selector_hidden)
        # Zero scores yield uniform softmax weights (the avgpool starting point).
        # Initially only score_proj can get a nonzero selector gradient; the
        # other scoring layers start learning once score_proj moves away from 0.

        # Keep this name and layout compatible with DynamicAvgPoolProjector.
        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, out_hidden),
        )

    def _apply(self, fn, recurse: bool = True):
        # to_empty() allocates uninitialized storage for meta parameters. Restore
        # the uniform start then; ordinary dtype/device moves keep learned scores.
        score_was_meta = self.score_proj.weight.is_meta
        result = super()._apply(fn, recurse=recurse)
        if score_was_meta and not self.score_proj.weight.is_meta:
            self.score_proj.reset_parameters()
        self.score_proj.weight._veomni_fsdp_shard_dim = 1
        return result

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
    def _grid_candidate_metadata(
        *,
        time: int,
        height: int,
        width: int,
        output_height: int,
        output_width: int,
        feature_offset: int,
        cell_offset: int,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        """Build candidate memberships for one merged T x H x W feature grid.

        Output order is time, output row, output column. Feature and cell offsets
        turn local indices into global indices in the concatenated pack.
        """
        # Adaptive pooling uses [floor(i*H/Ho), ceil((i+1)*H/Ho)) on each axis.
        # Windows can overlap: H=5, Ho=2 gives [0,3) and [2,5). Keeping those
        # memberships is necessary to reproduce avgpool at zero initialization.
        output_y = torch.arange(output_height, dtype=torch.long)
        output_x = torch.arange(output_width, dtype=torch.long)
        y_start = torch.div(output_y * height, output_height, rounding_mode="floor")
        y_end = torch.div((output_y + 1) * height + output_height - 1, output_height, rounding_mode="floor")
        x_start = torch.div(output_x * width, output_width, rounding_mode="floor")
        x_end = torch.div((output_x + 1) * width + output_width - 1, output_width, rounding_mode="floor")

        # Temporarily pad windows to their maximum height/width to enumerate
        # them together. valid_y/valid_x remove padding before feature gathering.
        max_y = max(
            ((index + 1) * height + output_height - 1) // output_height
            - index * height // output_height
            for index in range(output_height)
        )
        max_x = max(
            ((index + 1) * width + output_width - 1) // output_width
            - index * width // output_width
            for index in range(output_width)
        )
        y_delta = torch.arange(max_y, dtype=torch.long)
        x_delta = torch.arange(max_x, dtype=torch.long)
        candidate_y = y_start[:, None] + y_delta[None, :]
        candidate_x = x_start[:, None] + x_delta[None, :]
        valid_y = candidate_y < y_end[:, None]
        valid_x = candidate_x < x_end[:, None]

        # [Ho, Wo, max_y, max_x]: every output cell's candidate source positions.
        # Repeat this spatial layout independently for each time step.
        spatial_indices = candidate_y[:, None, :, None] * width + candidate_x[None, :, None, :]
        spatial_valid = valid_y[:, None, :, None] & valid_x[None, :, None, :]
        candidates_per_frame = spatial_indices.numel()
        frame_offsets = torch.arange(time, dtype=torch.long) * height * width
        candidate_indices = spatial_indices.reshape(1, -1) + frame_offsets[:, None] + feature_offset
        candidate_valid = spatial_valid.reshape(1, -1).expand(time, candidates_per_frame)

        # Build a parallel destination index for each candidate, then strip the
        # padding from both arrays. These paired [E] arrays define ragged cells.
        candidates_per_cell = max_y * max_x
        local_cells = torch.arange(time * output_height * output_width, dtype=torch.long).reshape(-1, 1)
        cell_indices = local_cells.expand(-1, candidates_per_cell).reshape(time, -1) + cell_offset
        candidate_indices = candidate_indices[candidate_valid]
        cell_indices = cell_indices[candidate_valid]

        # [C_i, 3] normalized cell-center coordinates, shared by all candidates
        # in a cell. These are grid-relative positions, not video timestamps or
        # candidate-relative offsets. They condition scores without mixing time.
        cell_y = output_y.to(torch.float32)
        cell_x = output_x.to(torch.float32)
        frame_pos = (torch.arange(time, dtype=torch.float32) + 0.5) / time
        row_pos = (cell_y + 0.5) / output_height
        col_pos = (cell_x + 0.5) / output_width
        positions = torch.stack(
            torch.meshgrid(frame_pos, row_pos, col_pos, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        positions = positions.mul(2).sub(1)
        return candidate_indices, cell_indices, positions

    def _build_ragged_cells(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_sizes: list[float],
        downsample_ratios: list[float],
    ) -> tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
        list[int],
        int,
    ]:
        """Concatenate per-grid metadata; no trainable layer runs in this loop."""
        device = images_feature.device
        grids = images_thw.detach().cpu().tolist()
        all_candidate_indices = []
        all_cell_indices = []
        all_positions = []
        all_rates = []
        cell_offsets = [0]
        seq_len = []
        feature_offset = 0
        cell_offset = 0

        for grid, merge_size_value, scale in zip(grids, merge_sizes, downsample_ratios):
            time, raw_height, raw_width = (int(value) for value in grid)
            merge_size = int(merge_size_value)
            if time <= 0 or raw_height <= 0 or raw_width <= 0:
                raise ValueError(f"grid_thw values must be positive, got {grid}.")
            if merge_size <= 0 or merge_size != merge_size_value:
                raise ValueError(f"merge_size must be a positive integer, got {merge_size_value}.")
            if raw_height % merge_size != 0 or raw_width % merge_size != 0:
                raise ValueError(
                    f"Grid {(time, raw_height, raw_width)} is not divisible by merge_size={merge_size}."
                )
            if scale <= 0:
                raise ValueError(f"downsample ratios must be positive, got {scale}.")

            # grid_thw describes the pre-merger patch grid, while images_feature
            # already contains merged features: [T * (H/merge) * (W/merge), D].
            # Dynamic pooling applies a second spatial reduction after merging.
            height = raw_height // merge_size
            width = raw_width // merge_size
            output_height, output_width = get_adaptive_pool_size(height, width, scale)
            grid_cells = time * output_height * output_width
            grid_feature_count = time * height * width
            next_feature_offset = feature_offset + grid_feature_count
            if next_feature_offset > images_feature.shape[0]:
                raise ValueError(
                    "images_feature length does not match grid_thw after spatial merging: "
                    f"expected at least {next_feature_offset}, got {images_feature.shape[0]}."
                )

            candidate_indices, cell_indices, positions = self._grid_candidate_metadata(
                time=time,
                height=height,
                width=width,
                output_height=output_height,
                output_width=output_width,
                feature_offset=feature_offset,
                cell_offset=cell_offset,
            )
            all_candidate_indices.append(candidate_indices)
            all_cell_indices.append(cell_indices)
            all_positions.append(positions)
            all_rates.append(torch.full((grid_cells, 1), math.log2(scale)))

            feature_offset = next_feature_offset
            cell_offset += grid_cells
            cell_offsets.append(cell_offset)
            # Compatibility with the original projector: one length per temporal
            # grid step, not one length per video or packed text sample.
            seq_len.extend([output_height * output_width] * time)

        if feature_offset != images_feature.shape[0]:
            raise ValueError(
                "images_feature length does not match grid_thw after spatial merging: "
                f"expected {feature_offset}, got {images_feature.shape[0]}."
            )

        # Assemble variable-size grids before moving metadata to the feature
        # device. num_cells stays a Python int, avoiding a later device .item().
        return (
            torch.cat(all_candidate_indices).to(device=device, non_blocking=True),
            torch.cat(all_cell_indices).to(device=device, non_blocking=True),
            torch.cat(all_positions).to(device=device, non_blocking=True),
            torch.cat(all_rates).to(device=device, non_blocking=True),
            torch.tensor(cell_offsets, device=device, dtype=torch.long),
            seq_len,
            cell_offset,
        )

    @staticmethod
    def _segment_softmax(
        logits: torch.Tensor,
        cell_indices: torch.LongTensor,
        num_cells: int,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Normalize [E] scores independently within each of C cells.

        scatter_reduce/index_add replace a padded [C, max_candidates] softmax.
        Candidates in different cells, grids, or samples never compete.
        """
        logits_fp32 = logits.float()
        maxima = torch.full(
            (num_cells,), -torch.inf, device=logits.device, dtype=torch.float32
        )
        # Subtract the per-cell maximum for stability. Detaching this shift
        # preserves the softmax gradient while avoiding a max-reduction backward.
        maxima.scatter_reduce_(
            0, cell_indices, logits_fp32.detach(), reduce="amax", include_self=True
        )
        unnormalized = torch.exp(logits_fp32 - maxima.index_select(0, cell_indices))
        denominators = torch.zeros(num_cells, device=logits.device, dtype=torch.float32)
        denominators.index_add_(0, cell_indices, unnormalized)
        weights = unnormalized / denominators.index_select(0, cell_indices)
        counts = torch.bincount(cell_indices, minlength=num_cells)
        return weights, counts

    def select_tokens(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_size: Optional[torch.Tensor | Sequence[float]] = None,
        downsample_ratios: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> tuple[torch.Tensor, list[int], TokenSelectorAux]:
        """Return [C, D] pooled tokens, per-time-step lengths, and diagnostics.

        images_feature: [N, D], concatenated merged grids in time/row/column order.
        images_thw: [G, 3], pre-merger patch grids (T, H, W).
        merge_size/downsample_ratios: optional [G] metadata, one value per grid.

        For candidate j in cell c:
            e_cj = Ws tanh(Wk LN(x_j) + Wq LN(mean_c)
                           + Wr log2(rate_c) + Wp position_c) / temperature
            a_cj = softmax_j(e_cj)
            selected_c = sum_j(a_cj * x_j)

        The query is derived from local visual content, not the user's text.
        """
        if images_feature.ndim != 2:
            raise ValueError(
                f"images_feature must have shape [tokens, hidden], got {tuple(images_feature.shape)}."
            )
        if images_feature.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {images_feature.shape[-1]}."
            )
        if images_thw.ndim != 2 or images_thw.shape[-1] != 3:
            raise ValueError(f"images_thw must have shape [grids, 3], got {tuple(images_thw.shape)}.")

        parallel_state = get_parallel_state()
        if self.training and parallel_state is not None and parallel_state.sp_enabled:
            raise NotImplementedError(
                "DynamicTokenSelectorProjector does not yet support sequence parallelism. "
                "Selector scores must first be synchronized across hidden-dimension shards."
            )

        grid_count = images_thw.shape[0]
        if grid_count == 0:
            if images_feature.shape[0] != 0:
                raise ValueError(
                    "images_feature must be empty when images_thw contains no grids, "
                    f"got {images_feature.shape[0]} tokens."
                )
            empty_long = torch.empty(0, device=images_feature.device, dtype=torch.long)
            empty_float = torch.empty(0, device=images_feature.device, dtype=torch.float32)
            aux = TokenSelectorAux(
                logits=empty_float,
                weights=empty_float,
                candidate_indices=empty_long,
                cell_indices=empty_long,
                cell_offsets=torch.zeros(1, device=images_feature.device, dtype=torch.long),
                cell_counts=empty_long,
            )
            return images_feature, [], aux

        merge_sizes = self._normalize_grid_values(merge_size, grid_count, self.merge_size, "merge_size")
        ratios = self._normalize_grid_values(
            downsample_ratios, grid_count, self.mm_downsample_ratio, "downsample_ratios"
        )
        (
            candidate_indices,
            cell_indices,
            cell_positions,
            cell_rates,
            cell_offsets,
            seq_len,
            num_cells,
        ) = self._build_ragged_cells(images_feature, images_thw, merge_sizes, ratios)

        # Shapes: images_feature=[N, D], candidate_indices=[E], cell_indices=[E].
        # E is the total number of cell memberships, which can exceed N when
        # adaptive-pooling windows overlap. C=num_cells is the output-token count.
        # 1. Gather candidate values: candidate_features=[E, D]. The same input
        # token may appear more than once here, once for each cell containing it.
        candidate_features = images_feature.index_select(0, candidate_indices)

        # 2. Count memberships per output cell: cell_counts=[C]. Each count is
        # used to normalize that cell's unweighted context mean below.
        cell_counts = torch.bincount(cell_indices, minlength=num_cells)

        # 3. Segment-sum candidate_features by destination cell:
        # cell_sums=[C, D]. index_add_ accumulates row e of [E, D] into row
        # cell_indices[e] of [C, D], without constructing a padded [C, E, D].
        cell_sums = images_feature.new_zeros((num_cells, self.hidden_size))
        cell_sums.index_add_(0, cell_indices, candidate_features)

        # 4. Broadcast counts from [C] to [C, 1] and divide row-wise:
        # cell_means=[C, D], one local context vector for every output cell.
        cell_means = cell_sums / cell_counts.to(images_feature.dtype).unsqueeze(-1)

        # 2. Project keys once over the N unique input tokens, then gather [E, S].
        # Context is [C, S]; index_select broadcasts it to each cell's candidates.
        # All grids share these batched calls, regardless of their output sizes.
        normalized_features = self.selector_norm(images_feature)
        candidate_keys = self.key_proj(normalized_features).index_select(0, candidate_indices)
        cell_context = self.query_proj(self.selector_norm(cell_means))
        cell_context = cell_context + self.rate_proj(cell_rates.to(images_feature.dtype))
        cell_context = cell_context + self.position_proj(cell_positions.to(images_feature.dtype))
        selector_hidden = torch.tanh(candidate_keys + cell_context.index_select(0, cell_indices))
        logits = self.score_proj(selector_hidden).squeeze(-1) / self.selector_temperature
        # 3. FP32 segmented softmax: weights sum to 1 within each cell. A cell
        # with one candidate (e.g. ratio=1) always has weight 1 and cannot learn
        # to choose among alternatives, though the scoring path still executes.
        weights, cell_counts = self._segment_softmax(logits, cell_indices, num_cells)

        # 4. Weighted pooling returns one D-dimensional token per cell. Every
        # candidate contributes continuously; there is no discrete top-k/gather
        # decision that would block gradients to the scorer.
        selected = images_feature.new_zeros((num_cells, self.hidden_size))
        selected.index_add_(
            0,
            cell_indices,
            candidate_features * weights.to(images_feature.dtype).unsqueeze(-1),
        )
        aux = TokenSelectorAux(
            logits=logits,
            weights=weights,
            candidate_indices=candidate_indices,
            cell_indices=cell_indices,
            cell_offsets=cell_offsets,
            cell_counts=cell_counts,
        )
        return selected, seq_len, aux

    def forward(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
        merge_size: Optional[torch.Tensor | Sequence[float]] = None,
        downsample_ratios: Optional[torch.Tensor | Sequence[float]] = None,
        *,
        return_selector_aux: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], TokenSelectorAux]:
        """Compress in encoder space, then map [C, D] to LLM features [C, L]."""
        selected, seq_len, aux = self.select_tokens(
            images_feature,
            images_thw,
            merge_size=merge_size,
            downsample_ratios=downsample_ratios,
        )
        hidden_states = self.mlp(selected)
        if return_selector_aux:
            return hidden_states, seq_len, aux
        return hidden_states, seq_len


def build_token_selector_projector(
    encoder_hidden: int,
    out_hidden: int,
    downsample_ratio: float,
    selector_hidden: int = 128,
    selector_temperature: float = 1.0,
) -> DynamicTokenSelectorProjector:
    """Build the standalone experimental visual token selector."""
    return DynamicTokenSelectorProjector(
        encoder_hidden=encoder_hidden,
        out_hidden=out_hidden,
        downsample_ratio=downsample_ratio,
        selector_hidden=selector_hidden,
        selector_temperature=selector_temperature,
    )
