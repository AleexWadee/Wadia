"""The calculator tool evaluates maths safely, without Python's eval()."""
import pytest

from tools import calculate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 2", "4"),
        ("(17.5 * 3) / 4", "13.125"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("10 % 3", "1"),
        ("sqrt(16)", "4.0"),
        ("round(pi, 2)", "3.14"),
        ("factorial(5)", "120"),
    ],
)
def test_calculates_correctly(expression, expected):
    assert calculate(expression) == expected


def test_rounds_long_decimals():
    assert calculate("1 / 3") == "0.3333333333"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",  # code injection
        "open('secrets.txt').read()",              # file access
        "x + 1",                                   # unknown names
        "2 ** 100000",                             # huge exponent
    ],
)
def test_rejects_unsafe_or_invalid_input(expression):
    assert calculate(expression).startswith("Could not calculate")


def test_division_by_zero_is_reported_not_raised():
    assert calculate("1 / 0").startswith("Could not calculate")
