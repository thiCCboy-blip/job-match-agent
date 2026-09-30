"""Tests for recovering JSON from constrained model output.

Grammar-constrained decoding injects whitespace and newline tokens between
every character, and a nested `}` can appear before the intended object ends.
These cases pin the recovery logic, which is what turns truncated output into a
usable extraction instead of silently dropping requirements.
"""

from __future__ import annotations

import pytest

from jobmatch.extract import QwenExtractor, _verified_min_years
from jobmatch.schemas import Evidence

parse = QwenExtractor._parse


def test_parses_plain_object():
    assert parse('{"a": 1}') == {"a": 1}


def test_recovers_nested_object():
    assert parse('{ "a": { "b": 1 } }') == {"a": {"b": 1}}


def test_ignores_leading_whitespace_and_newlines():
    assert parse('\n\n\n{ "a": 1 }\n') == {"a": 1}


def test_trailing_fragment_does_not_truncate_the_object():
    # A `}` belonging to a nested array must not be mistaken for the end of
    # the document, or every requirement after the first would be lost.
    raw = """
    {
      "requirements": [
        {"name": "SQL"},
        {"name": "Excel"}
      ] ,
    "title": "Data Analyst"
    }
    """
    parsed = parse(raw)
    assert [r["name"] for r in parsed["requirements"]] == ["SQL", "Excel"]
    assert parsed["title"] == "Data Analyst"


def test_brace_inside_a_string_is_not_a_delimiter():
    assert parse('{ "text": "a } brace" }') == {"text": "a } brace"}


def test_escaped_quote_inside_a_string_is_handled():
    assert parse(r'{ "text": "he said \"hi } there\"" }') == {
        "text": 'he said "hi } there"'
    }


def test_missing_object_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        parse("no json here")


def test_unterminated_object_raises():
    with pytest.raises(ValueError, match="unterminated"):
        parse('{"a": 1')


# --- min_years repair ----------------------------------------------------
#
# Qwen3-1.7B copies the evidence line's own number into min_years for lines
# that state no duration, which silently tightens scoring. The quoted text is
# the only authority.


def evidence_for(text: str | None) -> Evidence | None:
    return Evidence(text=text, line=1) if text else None


@pytest.mark.parametrize(
    ("model_value", "evidence_text", "expected"),
    [
        # Hallucinated from the line number: no duration is stated.
        (4.0, "Bachelor degree in Engineering required", None),
        (None, "- SQL required", None),
        # Agrees with the text.
        (3.0, "- 3+ years of SQL experience required", 3.0),
        ("4", "- 4+ years Python", 4.0),
        # Disagrees: the text wins.
        (2.0, "- at least 5 years of experience", 5.0),
        (9.9, "- at least 5 years", 5.0),
        (True, "- 2 years", 2.0),
        # Unparseable model output still yields the stated duration.
        ("abc", "- 3 years", 3.0),
        # Spoken numbers.
        (2.0, "eight plus years of SQL", 8.0),
    ],
)
def test_min_years_comes_from_evidence(model_value, evidence_text, expected):
    assert _verified_min_years(model_value, evidence_for(evidence_text)) == expected


@pytest.mark.parametrize("model_value", [4.0, None, "4", True])
def test_min_years_is_dropped_without_evidence(model_value):
    assert _verified_min_years(model_value, evidence_for(None)) is None
    assert _verified_min_years(model_value, evidence_for("")) is None
