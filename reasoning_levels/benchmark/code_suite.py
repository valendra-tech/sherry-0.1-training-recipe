"""Local basic Python coding suite: loading, extraction, and grading.

The suite is authored in-repo (`code_tasks/basic_python.json`) so it needs no
external evaluation package. Generated code is graded by running the hidden
asserts in an isolated subprocess (`python -I`) with CPU and address-space
rlimits and a hard timeout. This is a process-level sandbox, not a security
boundary: there is no network isolation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SUITE_PATH = Path(__file__).resolve().parent / "code_tasks" / "basic_python.json"
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_ADDRESS_SPACE_LIMIT = 1024 * 1024 * 1024
_CPU_LIMIT_SECONDS = 5

_BOOTSTRAP = (
    "import resource\n"
    "def _limit(kind, value):\n"
    "    try:\n"
    "        resource.setrlimit(kind, (value, value))\n"
    "    except (ValueError, OSError):\n"
    "        pass\n"
    "_limit(resource.RLIMIT_CPU, {cpu})\n"
    "_limit(resource.RLIMIT_AS, {mem})\n"
    "code = {program!r}\n"
    "exec(compile(code, '<submission>', 'exec'))\n"
)


def load_tasks(limit: int = 0) -> list[dict]:
    """Load the local suite, optionally truncated to the first ``limit`` tasks."""
    tasks = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    if limit:
        tasks = tasks[:limit]
    return tasks


def suite_revision() -> str:
    """Return the sha256 of the suite file, used as the dataset revision."""
    return hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()


def extract_code(content: str, function_name: str) -> str | None:
    """Extract the model's Python answer for ``function_name`` from content.

    Preference order: the last fenced code block that defines the function, then
    the last fenced block, then the text from the first ``def function_name``
    line to the end of the content. Returns ``None`` when no definition is
    found.
    """
    blocks = _CODE_BLOCK.findall(content or "")
    defining = [block for block in blocks if f"def {function_name}" in block]
    if defining:
        return defining[-1].strip()
    if blocks:
        return blocks[-1].strip()
    marker = f"def {function_name}"
    index = (content or "").find(marker)
    if index >= 0:
        return content[index:].strip()
    return None


def run_tests(code: str, tests: list[str], timeout: int = 6) -> tuple[bool, str]:
    """Run the hidden asserts against ``code``; return (passed, detail)."""
    program = code + "\n\n" + "\n".join(tests) + "\n"
    bootstrap = _BOOTSTRAP.format(
        cpu=_CPU_LIMIT_SECONDS, mem=_ADDRESS_SPACE_LIMIT, program=program
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode == 0:
        return True, "passed"
    stderr = result.stderr.decode("utf-8", "replace").strip().splitlines()
    return False, stderr[-1] if stderr else f"exit code {result.returncode}"


def grade(row: dict, content: str) -> dict:
    """Grade one completion, returning the standard scoring dict."""
    code = extract_code(content, row["function_name"])
    if code is None:
        return {
            "answer_extracted": None,
            "expected": None,
            "exact_match": False,
            "math_verify": None,
        }
    passed, _detail = run_tests(code, row["tests"])
    return {
        "answer_extracted": code,
        "expected": None,
        "exact_match": passed,
        "math_verify": None,
    }
