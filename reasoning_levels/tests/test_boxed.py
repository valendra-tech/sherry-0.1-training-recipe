from reasoning_levels.boxed import extract_boxed, extract_answer, answers_match


def test_extract_boxed_simple():
    assert extract_boxed("The answer is \\boxed{42}.") == "42"


def test_extract_boxed_with_latex():
    assert extract_boxed("So we get \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"


def test_extract_boxed_nested_braces():
    assert extract_boxed("Hence \\boxed{x = \\sqrt{2}}.") == "x = \\sqrt{2}"


def test_extract_boxed_no_match():
    assert extract_boxed("No boxed answer here.") is None


def test_extract_boxed_multiple_takes_last():
    text = "\\boxed{1} and then \\boxed{2}"
    assert extract_boxed(text) == "2"


def test_extract_answer_strips_latex():
    assert extract_answer("\\boxed{42}") == "42"
    assert extract_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_answer("42") == "42"


def test_extract_answer_from_gsm8k():
    assert extract_answer("Some reasoning.\n#### 72") == "72"
    assert extract_answer("#### 100") == "100"
    assert extract_answer("no number") is None


def test_answers_match_exact():
    assert answers_match("42", "42") is True
    assert answers_match("42", "43") is False
    assert answers_match(None, "42") is False


def test_answers_match_numeric():
    assert answers_match("42.0", "42") is True
    assert answers_match("42", "42.00") is True


def test_answers_match_latex_normalized():
    assert answers_match("\\frac{1}{2}", "\\frac{1}{2}") is True


def test_answers_match_fraction_against_rounded_decimal():
    """NuminaMath may store a rounded decimal while the model returns a fraction."""
    assert answers_match(r"\frac{128}{823,543}", "0.00016") is True


def test_answers_match_rejects_different_numeric_value():
    assert answers_match(r"\frac{128}{823,543}", "0.1") is False
