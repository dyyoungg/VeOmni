"""Unit tests for checkpoint callback _last_saved_step correctness.

Validates that _last_saved_step is only updated AFTER the save operation
succeeds, so that a failed save does not suppress future retry attempts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import (
    CheckpointerCallback,
    HuggingfaceCkptCallback,
    VeOmniModelRuntime,
)


def _make_mock_trainer(save_path="/tmp/test_ckpt", save_async=False):
    """Build a minimal mock trainer for CheckpointerCallback tests."""
    checkpoint_cfg = SimpleNamespace(
        save_path=save_path,
        save_steps=5,
        save_epochs=1,
        save_async=save_async,
        load_path=None,
        manager="dcp",
        dcp_save_to_lowest_rank=False,
        save_hf_weights=True,
        hf_save_steps=5,
        hf_save_epochs=1,
        model_assets_dir="/tmp/assets",
        output_dir="/tmp/output",
    )
    fsdp_config = SimpleNamespace(fsdp_mode="fsdp2")
    accelerator = SimpleNamespace(fsdp_config=fsdp_config)
    train_cfg = SimpleNamespace(
        checkpoint=checkpoint_cfg,
        accelerator=accelerator,
        global_rank=0,
    )
    model_cfg = SimpleNamespace(fqn_to_index_mapping={})
    args = SimpleNamespace(train=train_cfg, model=model_cfg)

    trainer = MagicMock()
    trainer.args = args
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.environ_meter = MagicMock()
    trainer.channel_loss_callback = MagicMock()
    trainer.channel_loss_callback.state_dict.return_value = {}
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.save_future = None
    trainer.model_assets = []

    return trainer


@patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer")
@patch("veomni.trainer.callbacks.checkpoint_callback.dist")
@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestCheckpointerCallbackLastSavedStep:
    """Tests for CheckpointerCallback._last_saved_step placement."""

    def test_last_saved_step_updated_after_successful_save(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_save_failure(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        trainer.checkpointer.save.side_effect = RuntimeError("disk full")
        cb = CheckpointerCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="disk full"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_save_includes_channel_loss_callback_state(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        trainer.channel_loss_callback.state_dict.return_value = {
            "source_registry": [(1, "train/a")],
        }
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        cb._save_checkpoint(TrainerState(global_step=10))

        checkpoint_state = trainer.checkpointer.save.call_args.args[1]
        assert checkpoint_state["extra_state"]["channel_loss_callback"] == {"source_registry": [(1, "train/a")]}

    def test_load_restores_channel_loss_callback_state(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/tmp/test_ckpt/global_step_7"
        trainer.args.train_steps = 100
        trainer.state = TrainerState()
        mock_build_ckpt.return_value = trainer.checkpointer
        callback_state = {"source_registry": [(1, "train/a")]}

        def load_checkpoint(path, state, **kwargs):
            state["extra_state"] = {
                "global_step": 7,
                "start_epoch": 0,
                "start_step": 7,
                "lr_scheduler": {},
                "train_dataloader": None,
                "environ_meter": {},
                "channel_loss_callback": callback_state,
                "torch_rng_state": torch.get_rng_state(),
            }

        trainer.checkpointer.load.side_effect = load_checkpoint
        cb = CheckpointerCallback(trainer)

        cb._load_checkpoint()

        trainer.channel_loss_callback.load_state_dict.assert_called_once_with(callback_state)

    def test_epoch_end_retries_after_failed_save(self, mock_helper, mock_dist, mock_build_ckpt):
        """If save fails at step_end, epoch_end should still attempt to save (not skip)."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        # Simulate save failure at step_end
        trainer.checkpointer.save.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the disk is available again
        trainer.checkpointer.save.side_effect = None
        trainer.checkpointer.save.reset_mock()

        # epoch_end should NOT skip because _last_saved_step was not updated
        cb.on_epoch_end(state)
        assert trainer.checkpointer.save.call_count == 1
        assert cb._last_saved_step == 5

    def test_epoch_end_preserves_unconditional_legacy_save(self, mock_helper, mock_dist, mock_build_ckpt):
        """The pre-merge local trainer saves at epoch end even after a step save."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        trainer.checkpointer.save.reset_mock()
        cb.on_epoch_end(state)
        trainer.checkpointer.save.assert_called_once()


@patch("veomni.trainer.callbacks.checkpoint_callback.save_hf_safetensor")
@patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer")
@patch("veomni.trainer.callbacks.checkpoint_callback.dist")
@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
@patch("os.path.exists", return_value=True)
class TestHuggingfaceCkptCallbackLastSavedStep:
    """Tests for HuggingfaceCkptCallback._last_saved_step placement."""

    @pytest.mark.xfail(strict=True, reason="Pre-merge HF export does not record its step when the DCP already exists.")
    def test_last_saved_step_updated_after_successful_hf_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_hf_save_failure(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        mock_save_hf.side_effect = RuntimeError("conversion failed")
        cb = HuggingfaceCkptCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="conversion failed"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_train_end_retries_after_failed_hf_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        """If HF save fails at step_end, train_end should still attempt to save."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        # Simulate HF save failure at step_end
        mock_save_hf.side_effect = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the save works
        mock_save_hf.side_effect = None
        mock_save_hf.reset_mock()

        # train_end should NOT skip because _last_saved_step was not updated
        cb.on_train_end(state)
        assert mock_save_hf.call_count == 1

    @pytest.mark.xfail(strict=True, reason="Pre-merge HF export can repeat at train end; retain that behavior in rollback.")
    def test_train_end_skips_after_successful_step_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        """If HF save succeeds at step_end, train_end should skip."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        mock_save_hf.reset_mock()
        cb.on_train_end(state)
        mock_save_hf.assert_not_called()


@pytest.mark.parametrize("use_runtime", [False, True])
@pytest.mark.parametrize("has_dataloader", [False, True])
def test_legacy_checkpoint_payload_roundtrip(tmp_path, use_runtime, has_dataloader):
    """Both trainer interfaces keep the pre-merge payload, paths and resume cursor."""
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.current_epoch = 2
    trainer.current_step = 3
    trainer.args.train_steps = 10
    trainer.args.model.lora_config = None
    trainer.lr_scheduler.state_dict.return_value = {"last_epoch": 23}
    trainer.train_dataloader.state_dict.return_value = {"position": 23}
    trainer.environ_meter.state_dict.return_value = {"tokens": 230}
    trainer.channel_loss_callback.state_dict.return_value = {"channel": 1}
    if not has_dataloader:
        trainer.train_dataloader = None

    model = trainer.model
    optimizer = trainer.optimizer
    scheduler = trainer.lr_scheduler
    if use_runtime:
        runtime = MagicMock(spec=VeOmniModelRuntime)
        runtime.model = model
        runtime.optimizer = optimizer
        runtime.lr_scheduler = scheduler
        runtime.model_assets = trainer.model_assets
        runtime.parallel_state = SimpleNamespace(global_rank=0)
        trainer.model = runtime
        trainer.args.model.accelerator = trainer.args.train.accelerator
        del trainer.args.train.accelerator
        del trainer.optimizer
        del trainer.lr_scheduler
        del trainer.data_iterator

    with (
        patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer", return_value=trainer.checkpointer),
        patch("veomni.trainer.callbacks.checkpoint_callback.dist"),
        patch("veomni.trainer.callbacks.checkpoint_callback.helper"),
    ):
        cb = CheckpointerCallback(trainer)
        cb._save_checkpoint(TrainerState(global_step=23))
        path, payload = trainer.checkpointer.save.call_args.args
        assert path == str(tmp_path / "global_step_23")
        assert set(payload) == {"model", "optimizer", "extra_state"}
        assert payload["model"] is model
        assert payload["optimizer"] is optimizer
        assert payload["extra_state"]["start_epoch"] == 2
        assert payload["extra_state"]["start_step"] == 3
        assert payload["extra_state"]["lr_scheduler"] == {"last_epoch": 23}
        assert payload["extra_state"]["train_dataloader"] == ({"position": 23} if has_dataloader else None)
        assert "stage_dir" not in trainer.checkpointer.save.call_args.kwargs
        if use_runtime:
            assert trainer.checkpointer.save.call_args.kwargs["parallel_state"] is runtime.parallel_state

        trainer.args.train.checkpoint.load_path = path

        def load_checkpoint(load_path, target, **kwargs):
            assert load_path == path
            assert target["model"] is model
            assert target["optimizer"] is optimizer
            target["extra_state"] = payload["extra_state"]

        trainer.checkpointer.load.side_effect = load_checkpoint
        cb._load_checkpoint()
        assert trainer.state.global_step == 23
        assert trainer.state.start_epoch == 2
        assert trainer.state.start_step == 3
        if use_runtime:
            assert trainer.start_epoch == 2
            assert trainer.start_step == 3
        scheduler.load_state_dict.assert_called_once_with({"last_epoch": 23})
        if has_dataloader:
            trainer.train_dataloader.load_state_dict.assert_called_once_with({"position": 23})
