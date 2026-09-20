"""Legacy job config and model-local Runtime views must select the same settings."""

import sys
from dataclasses import asdict

import pytest
import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig, TrainingArguments, VeOmniArguments
from veomni.arguments.parser import _instantiate_recursive, parse_args, save_args


@pytest.fixture(autouse=True)
def config_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    # Operator availability is orthogonal to config ownership and needs hardware.
    monkeypatch.setattr(OpsImplementationConfig, "__post_init__", lambda self: None)


def legacy_config():
    return {
        "model": {"model_path": "test-model"},
        "data": {"chat_template": "chatml"},
        "train": {
            "global_batch_size": 16,
            "micro_batch_size": 2,
            "accelerator": {"ulysses_size": 2, "fsdp_config": {"offload_pin_memory": False}},
            "optimizer": {"lr": 0.0002, "max_grad_norm": 0.5},
            "gradient_checkpointing": {"enable": False},
            "torch_compile": {"enable": False},
            "init_device": "meta",
            "broadcast_model_weights_from_rank0": False,
            "ep_sharded_stream_load": True,
            "local_rank": 0,
            "global_rank": 0,
            "world_size": 128,
        },
    }


def model_local_config():
    config = legacy_config()
    train, model = config["train"], config["model"]
    for name in ("accelerator", "optimizer", "broadcast_model_weights_from_rank0", "ep_sharded_stream_load"):
        model[name] = train.pop(name)
    for name in ("init_device", "gradient_checkpointing", "torch_compile"):
        model["accelerator"][name] = train.pop(name)
    model["chat_template"] = config["data"].pop("chat_template")
    return config


def test_legacy_config_binds_runtime_and_preserves_topology():
    config = legacy_config()
    args = _instantiate_recursive(VeOmniArguments, config)
    assert "optimizer" in config["train"], "parsing must not mutate caller input"
    assert args.model.accelerator is args.train.accelerator
    assert args.model.optimizer is args.train.optimizer
    assert args.train.gradient_checkpointing is args.model.accelerator.gradient_checkpointing
    assert args.train.torch_compile is args.model.accelerator.torch_compile
    assert args.train.accelerator.dp_size == 4
    assert args.train.gradient_accumulation_steps == 2
    assert args.train.optimizer.lr == 0.0002
    assert args.train.optimizer.max_grad_norm == 0.5
    assert args.train.gradient_checkpointing.enable is False
    assert args.train.accelerator.fsdp_config.offload_pin_memory is False
    assert args.model.broadcast_model_weights_from_rank0 is False
    assert args.model.ep_sharded_stream_load is True
    assert args.model.chat_template == args.data.chat_template == "chatml"
    assert (args.train.local_rank, args.train.global_rank, args.train.world_size) == (3, 3, 8)


def test_model_local_yaml_selects_same_settings():
    old = _instantiate_recursive(VeOmniArguments, legacy_config())
    new = _instantiate_recursive(VeOmniArguments, model_local_config())
    assert asdict(old) == asdict(new)


def test_saved_yaml_uses_legacy_layout_and_roundtrips(tmp_path):
    args = _instantiate_recursive(VeOmniArguments, legacy_config())
    save_args(args, str(tmp_path))
    saved = yaml.safe_load((tmp_path / "veomni_cli.yaml").read_text())
    assert "optimizer" not in saved["model"]
    assert "accelerator" not in saved["model"]
    assert "chat_template" not in saved["model"]
    assert "gradient_checkpointing" not in saved["train"]["accelerator"]
    assert saved["train"]["gradient_checkpointing"]["enable"] is False
    reloaded = _instantiate_recursive(VeOmniArguments, saved)
    # YAML represents optimizer tuples as lists; compare serialized values.
    assert yaml.safe_load(yaml.safe_dump(asdict(reloaded))) == yaml.safe_load(yaml.safe_dump(asdict(args)))


@pytest.mark.parametrize("config_factory", [legacy_config, model_local_config])
@pytest.mark.parametrize("cli_key", [None, "--train.optimizer.lr", "--model.optimizer.lr"])
def test_cli_overrides_preserve_other_yaml_settings(tmp_path, monkeypatch, config_factory, cli_key):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_factory()))
    argv = ["train", str(path)]
    if cli_key:
        argv.extend([cli_key, "0.0003"])
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args(VeOmniArguments)
    assert args.model.model_path == "test-model"
    assert args.train.optimizer.lr == (0.0003 if cli_key else 0.0002)
    assert args.train.optimizer.max_grad_norm == 0.5
    assert args.train.accelerator.ulysses_size == 2


def test_conflicting_config_does_not_silently_ignore_old_values():
    config = legacy_config()
    config["model"]["optimizer"] = {"lr": 0.01}
    with pytest.raises(ValueError, match=r"Conflicting.*train.optimizer.lr"):
        _instantiate_recursive(VeOmniArguments, config)


def test_unknown_keys_still_fail():
    config = legacy_config()
    config["train"]["optimzer"] = {"lr": 0.01}
    with pytest.raises(ValueError, match=r"train.optimzer"):
        _instantiate_recursive(VeOmniArguments, config)


def test_standalone_training_arguments_have_concrete_legacy_defaults():
    args = TrainingArguments()
    assert args.init_device == "meta"
    assert args.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.gradient_checkpointing is args.accelerator.gradient_checkpointing
    assert args.optimizer.lr > 0
