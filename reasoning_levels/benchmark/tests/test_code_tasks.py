import json
import subprocess
import sys
from pathlib import Path

SUITE_PATH = Path(__file__).resolve().parents[1] / "code_tasks" / "basic_python.json"


def _load():
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def test_suite_shape():
    tasks = _load()
    assert len(tasks) == 40
    assert [t["tier"] for t in tasks].count("easy") == 15
    assert [t["tier"] for t in tasks].count("medium") == 15
    assert [t["tier"] for t in tasks].count("tricky") == 10
    assert len({t["problem_id"] for t in tasks}) == 40
    assert len({t["function_name"] for t in tasks}) == 40
    for task in tasks:
        assert set(task) == {"problem_id", "tier", "prompt", "function_name", "tests", "reference"}
        assert 5 <= len(task["tests"]) <= 12
        assert task["prompt"].startswith("Complete the following Python function.")
        assert task["function_name"] in task["prompt"]
        assert task["function_name"] in task["reference"]


def test_reference_solutions_pass_their_tests():
    tasks = _load()
    failures = []
    for task in tasks:
        program = task["reference"] + "\n\n" + "\n".join(task["tests"]) + "\n"
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{task['problem_id']}: timeout")
            continue
        if result.returncode != 0:
            failures.append(
                f"{task['problem_id']}: {result.stderr.decode('utf-8', 'replace').strip().splitlines()[-1:]}"
            )
    assert failures == []
