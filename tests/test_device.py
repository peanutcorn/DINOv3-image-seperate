import pytest
import torch
from src.utils import resolve_device, select_device


def test_cpu_selection_does_not_require_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert select_device("prompt") == "cpu"


def test_gpu_selection_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert select_device("prompt") == "cuda"


def test_unavailable_gpu_requires_explicit_cpu_choice(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    choices = iter(["invalid", "2", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(choices))
    assert select_device("prompt") == "cpu"
    assert "NVIDIA GPU is unavailable" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="NVIDIA GPU is unavailable"):
        resolve_device("cuda")


@pytest.mark.parametrize("available, expected", [(False, "cpu"), (True, "cuda")])
def test_auto_and_explicit_config_skip_menu(monkeypatch, available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert select_device("auto") == expected
    assert select_device("cpu") == "cpu"


def test_missing_input_explains_cli_option(monkeypatch):
    def no_input(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    with pytest.raises(RuntimeError, match="--device cpu"):
        select_device("prompt")
