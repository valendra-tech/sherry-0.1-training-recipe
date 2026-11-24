import json
import subprocess
import sys
from pathlib import Path

import pytest

from reasoning_levels import launch_grpo_after_sft
from reasoning_levels.launch_grpo_after_sft import (
    parse_args,
    read_exit_status,
    run_chain,
    validate_sft_artifacts,
)


def test_read_exit_status_requires_a_single_integer(tmp_path):
    status = tmp_path / "status"
    status.write_text("0\n")
    assert read_exit_status(status) == 0
    status.write_text("137\n")
    assert read_exit_status(status) == 137


def test_launcher_direct_script_supports_help():
    script = Path(__file__).parents[1] / "launch_grpo_after_sft.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Launch GRPO after a successful SFT run" in result.stdout


def test_read_exit_status_rejects_missing_or_malformed_status(tmp_path):
    with pytest.raises(RuntimeError, match="status"):
        read_exit_status(tmp_path / "missing")
    status = tmp_path / "status"
    status.write_text("running\n")
    with pytest.raises(RuntimeError, match="status"):
        read_exit_status(status)


def test_validate_sft_artifacts_requires_metadata_and_adapter(tmp_path):
    output = tmp_path / "sft"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"adapter")
    (output / "adapter_config.json").write_text("{}")
    (output / "sft_run.json").write_text(
        json.dumps(
            {
                "thinking_truncated_train_examples": 1,
                "effort_token_ids": [248077, 248078, 248079],
            }
        )
    )
    validate_sft_artifacts(output)


def test_validate_sft_artifacts_rejects_missing_sft_metadata(tmp_path):
    output = tmp_path / "sft"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(RuntimeError, match="sft_run"):
        validate_sft_artifacts(output)


def test_nonzero_sft_status_never_runs_grpo(tmp_path):
    sft_status = tmp_path / "training.status"
    sft_status.write_text("137\n")
    calls = []
    args = parse_args(
        [
            "--sft-status",
            str(sft_status),
            "--sft-output",
            str(tmp_path / "sft"),
            "--grpo-output",
            str(tmp_path / "grpo"),
            "--chain-status",
            str(tmp_path / "chain.status"),
            "--chain-log",
            str(tmp_path / "chain.log"),
            "--lock",
            str(tmp_path / "chain.lock"),
        ]
    )

    result = run_chain(
        args,
        run=lambda command, log_path: calls.append((command, log_path)) or 0,
        session_exists=lambda session: False,
        sleep=lambda seconds: None,
    )

    assert result == 1
    assert calls == []
    assert (tmp_path / "chain.status").read_text() == "1\n"
    assert not (tmp_path / "chain.lock").exists()


def test_chain_uses_the_global_grpo_generation_budget(monkeypatch, tmp_path):
    calls = []
    args = parse_args(
        [
            "--sft-status",
            str(tmp_path / "training.status"),
            "--sft-output",
            str(tmp_path / "sft"),
            "--grpo-output",
            str(tmp_path / "grpo"),
            "--grpo-status",
            str(tmp_path / "grpo.status"),
            "--chain-status",
            str(tmp_path / "chain.status"),
            "--chain-log",
            str(tmp_path / "chain.log"),
            "--grpo-log",
            str(tmp_path / "grpo.log"),
            "--lock",
            str(tmp_path / "chain.lock"),
        ]
    )
    monkeypatch.setattr(launch_grpo_after_sft, "_wait_for_sft", lambda *a, **kw: 0)
    monkeypatch.setattr(launch_grpo_after_sft, "validate_sft_artifacts", lambda path: None)
    monkeypatch.setattr(launch_grpo_after_sft, "_ensure_mergekit", lambda path: None)

    result = run_chain(
        args,
        run=lambda command, log_path: calls.append(command) or 0,
    )

    assert result == 0
    full_command = calls[-1]
    token_flag = full_command.index("--max-new-tokens")
    assert full_command[token_flag + 1] == "65536"
