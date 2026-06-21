import pytest
from prelabel import parse_label


@pytest.mark.parametrize("raw,expected", [
    # Clean exact matches
    ("analysis", "analysis"),
    ("hot_take", "hot_take"),
    ("reaction", "reaction"),
    # Whitespace + casing + trailing punctuation
    ("  Analysis. ", "analysis"),
    ("  HOT_TAKE", "hot_take"),
    ("REACTION!", "reaction"),
    # Hyphen normalization
    ("hot-take", "hot_take"),
    # Substring / embedded label
    ("the label is hot_take", "hot_take"),
    ("Label: reaction", "reaction"),
    ("I think it's analysis maybe", "analysis"),
])
def test_parse_label_normalizes(raw, expected):
    assert parse_label(raw) == expected


@pytest.mark.parametrize("raw", [
    # Genuine garbage — no valid label present
    "I don't know",
    "",
    "42",
    # Ambiguous — two valid labels, must reject rather than guess
    "ANALYSIS_HOT_TAKE",
    "analysis or reaction",
])
def test_parse_label_rejects_garbage(raw):
    assert parse_label(raw) is None
