import math
import re


def extract_boxed(text: str) -> str | None:
    """Extract the content of the LAST \\boxed{...} in text.
    Handles nested braces by counting brace depth."""
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1]


def extract_answer(text: str) -> str | None:
    """Extract the final answer from text. Tries \\boxed{} first, then #### format,
    then plain number passthrough."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if "####" in text:
        after = text.split("####")[-1].strip()
        num = ""
        for ch in after:
            if ch.isdigit():
                num += ch
            elif num:
                break
        return num if num else None
    # Fallback: if the whole text is a number, return it
    stripped = text.strip()
    if stripped:
        try:
            float(stripped)
            return stripped
        except ValueError:
            pass
    return None


def _numeric_value(value: str) -> float | None:
    """Evaluate the small set of numeric formats used by the math datasets."""
    normalized = value.strip().replace(",", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")

    try:
        return float(normalized)
    except ValueError:
        pass

    fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", normalized)
    if fraction:
        numerator = _numeric_value(fraction.group(1))
        denominator = _numeric_value(fraction.group(2))
        if numerator is not None and denominator not in (None, 0):
            return numerator / denominator

    slash_fraction = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d+)?))\s*/\s*([+-]?(?:\d+(?:\.\d+)?))",
        normalized,
    )
    if slash_fraction:
        denominator = float(slash_fraction.group(2))
        if denominator != 0:
            return float(slash_fraction.group(1)) / denominator

    power = re.fullmatch(r"\((.+)\)\s*\^\s*\{?([+-]?\d+)\}?", normalized)
    if power:
        base = _numeric_value(power.group(1))
        if base is not None:
            return base ** int(power.group(2))

    return None


def _decimal_rounding_tolerance(value: str) -> float:
    """Allow a decimal ground truth to represent a rounded exact answer."""
    normalized = value.strip().replace(",", "")
    decimal = re.fullmatch(r"[+-]?\d+\.(\d+)", normalized)
    if not decimal:
        return 0.0
    return 0.5 * 10 ** -len(decimal.group(1))


def answers_match(predicted: str | None, ground_truth: str | None) -> bool:
    """Compare exact answers and simple numeric/LaTeX equivalents."""
    if predicted is None or ground_truth is None:
        return False
    p = predicted.strip()
    g = ground_truth.strip()
    if p == g:
        return True

    predicted_value = _numeric_value(p)
    ground_truth_value = _numeric_value(g)
    if predicted_value is None or ground_truth_value is None:
        return False

    tolerance = _decimal_rounding_tolerance(g)
    return math.isclose(
        predicted_value,
        ground_truth_value,
        rel_tol=1e-9,
        abs_tol=tolerance + 1e-12,
    )
