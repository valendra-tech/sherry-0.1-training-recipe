from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


GLOBAL_GRPO_MAX_NEW_TOKENS = 65536


def read_exit_status(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"status file is missing: {path}")
    values = path.read_text(encoding="utf-8").split()
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError(f"status file is malformed: {path}")
    return int(values[0])


def validate_sft_artifacts(path: str | Path) -> None:
    path = Path(path)
    required = [path / "adapter_config.json", path / "sft_run.json"]
    if not any(
        (path / name).exists()
        for name in ["adapter_model.safetensors", "adapter_model.bin"]
    ):
        raise RuntimeError(f"SFT adapter weights are missing: {path}")
    missing = [item.name for item in required if not item.exists()]
    if missing:
        raise RuntimeError(f"SFT artifacts are missing {missing}")
    try:
        metadata = json.loads((path / "sft_run.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("sft_run.json is not valid JSON") from error
    if not isinstance(metadata.get("thinking_truncated_train_examples"), int):
        raise RuntimeError("sft_run.json lacks corrected truncation metadata")
    if metadata.get("effort_token_ids") != [248077, 248078, 248079]:
        raise RuntimeError("SFT effort token IDs do not match the verified tokenizer")


def _tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}\n")


def _write_status(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


def _wait_for_sft(
    session: str,
    status_path: Path,
    poll_seconds: int,
    log_path: Path,
    session_exists: Callable[[str], bool] = _tmux_session_exists,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    while session_exists(session):
        if status_path.exists():
            _log(log_path, "SFT status exists; waiting for the session shell to exit")
            sleep(1)
            continue
        _log(log_path, f"waiting for SFT session {session}")
        sleep(poll_seconds)
    if not status_path.exists():
        raise RuntimeError(f"SFT session ended without status file: {status_path}")
    return read_exit_status(status_path)


def _run(command: list[str], log_path: Path) -> int:
    _log(log_path, "running: " + " ".join(command))
    with log_path.open("a", encoding="utf-8") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def _ensure_mergekit(log_path: Path) -> None:
    if importlib.util.find_spec("mergekit") is not None:
        _log(log_path, "mergekit import is available")
        return
    return_code = _run([sys.executable, "-m", "pip", "install", "mergekit"], log_path)
    if return_code != 0 or importlib.util.find_spec("mergekit") is None:
        raise RuntimeError("mergekit installation or import failed")
    _log(log_path, "mergekit installed and import is available")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch GRPO after a successful SFT run")
    parser.add_argument("--sft-session", default="reasoning-sft-v2")
    parser.add_argument("--sft-status", default="data/sft/training-v2.status")
    parser.add_argument("--sft-output", default="checkpoints/qwen36-reasoning-effort-sft-0.2")
    parser.add_argument("--grpo-output", default="checkpoints/qwen36-reasoning-effort-grpo-0.1")
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--sft-eval", default="data/sft/eval.jsonl")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--grpo-status", default="data/sft/grpo.status")
    parser.add_argument("--chain-status", default="data/sft/grpo-chain.status")
    parser.add_argument("--chain-log", default="data/sft/grpo-chain.log")
    parser.add_argument("--grpo-log", default="data/sft/grpo.log")
    parser.add_argument("--lock", default="data/sft/grpo-chain.lock")
    return parser.parse_args(argv)


def run_chain(
    args: argparse.Namespace,
    run: Callable[[list[str], Path], int] = _run,
    session_exists: Callable[[str], bool] = _tmux_session_exists,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    chain_log = Path(args.chain_log)
    chain_status = Path(args.chain_status)
    lock_path = Path(args.lock)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"GRPO chain lock already exists: {lock_path}") from error

    try:
        sft_status = _wait_for_sft(
            args.sft_session,
            Path(args.sft_status),
            args.poll_seconds,
            chain_log,
            session_exists=session_exists,
            sleep=sleep,
        )
        if sft_status != 0:
            raise RuntimeError(f"SFT failed with exit status {sft_status}")
        validate_sft_artifacts(args.sft_output)

        dry_run_output = Path(args.grpo_output).parent / ".grpo-dry-run"
        dry_run_command = [
            sys.executable,
            "-u",
            "-m",
            "reasoning_levels.grpo_train",
            "--adapter",
            args.sft_output,
            "--train",
            args.train,
            "--sft-eval",
            args.sft_eval,
            "--problems-output",
            "data/sft/grpo-dry-run-problems.jsonl",
            "--problems-manifest",
            "data/sft/grpo-dry-run-problems-manifest.json",
            "--output",
            str(dry_run_output),
            "--dry-run",
        ]
        _ensure_mergekit(chain_log)
        if run(dry_run_command, chain_log) != 0:
            raise RuntimeError("GRPO dry-run failed")

        full_command = [
            sys.executable,
            "-u",
            "-m",
            "reasoning_levels.grpo_train",
            "--adapter",
            args.sft_output,
            "--train",
            args.train,
            "--sft-eval",
            args.sft_eval,
            "--problems-output",
            "data/sft/grpo-problems.jsonl",
            "--problems-manifest",
            "data/sft/grpo-problems-manifest.json",
            "--output",
            args.grpo_output,
            "--max-steps",
            "200",
            "--max-new-tokens",
            str(GLOBAL_GRPO_MAX_NEW_TOKENS),
            "--seed",
            "42",
        ]
        grpo_status = run(full_command, Path(args.grpo_log))
        _write_status(Path(args.grpo_status), grpo_status)
        if grpo_status != 0:
            raise RuntimeError(f"GRPO failed with exit status {grpo_status}")
        _write_status(chain_status, 0)
        _log(chain_log, "GRPO chain completed successfully")
        return 0
    except Exception as error:
        _write_status(chain_status, 1)
        _log(chain_log, f"GRPO chain stopped: {error}")
        return 1
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    return run_chain(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
