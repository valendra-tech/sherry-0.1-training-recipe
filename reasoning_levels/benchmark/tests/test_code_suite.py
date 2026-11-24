from reasoning_levels.benchmark import code_suite
from reasoning_levels.benchmark.datasets import TASKS, load_task
from reasoning_levels.benchmark.prompts import STOPS, doc_to_text
from reasoning_levels.benchmark.run import score_for_task


def test_load_tasks_and_limit():
    tasks = code_suite.load_tasks()
    assert len(tasks) == 40
    assert len(code_suite.load_tasks(limit=3)) == 3
    assert code_suite.suite_revision() == TASKS["code_basic"]["revision"]


def test_extract_code_prefers_defining_block():
    content = "```python\ndef other():\n    pass\n```\n```python\ndef add(a, b):\n    return a + b\n```"
    assert code_suite.extract_code(content, "add") == "def add(a, b):\n    return a + b"


def test_extract_code_falls_back_to_bare_def():
    content = "Here you go:\ndef add(a, b):\n    return a + b"
    assert code_suite.extract_code(content, "add") == "def add(a, b):\n    return a + b"


def test_extract_code_returns_none_without_definition():
    assert code_suite.extract_code("no code here", "add") is None


def test_grade_passes_a_correct_solution():
    row = {"function_name": "add", "tests": ["assert add(1, 2) == 3"]}
    result = code_suite.grade(row, "```python\ndef add(a, b):\n    return a + b\n```")
    assert result["exact_match"] is True
    assert result["answer_extracted"].startswith("def add")


def test_grade_fails_a_wrong_solution():
    row = {"function_name": "add", "tests": ["assert add(1, 2) == 3"]}
    result = code_suite.grade(row, "```python\ndef add(a, b):\n    return a - b\n```")
    assert result["exact_match"] is False


def test_code_basic_is_wired_into_prompts_and_scoring():
    assert STOPS["code_basic"] == ["<|im_end|>"]
    row = load_task("code_basic", limit=1)[0]
    assert doc_to_text("code_basic", row) == row["prompt"]
    result = score_for_task("code_basic", row, "no code")
    assert result["exact_match"] is False
