"""Tests for reducing model-written requirement names to canonical skills.

Qwen3-1.7B names requirements the way a posting reads, not the way a taxonomy
lists them. Without this reduction, gap detection reports skills the candidate
does have as missing, because "K8s administration" is not "Kubernetes".
"""

from __future__ import annotations

import pytest

from jobmatch.extract import _canonical_model_name as canonical


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        # A bare canonical name is already correct.
        ("SQL", "SQL"),
        ("Excel", "Excel"),
        ("Kubernetes", "Kubernetes"),
        # Usage wording that names a known skill.
        ("K8s administration", "Kubernetes"),
        ("Node.js services", "Node.js"),
        ("Postgres administration", "PostgreSQL"),
        ("CI/CD pipelines", "CI/CD"),
        ("SQL experience", "SQL"),
        ("dbt experience", "dbt"),
        ("Excel skills", "Excel"),
        # Adjectives qualify a skill without naming a different one.
        ("Advanced SQL", "SQL"),
        # The skill is named at the end of a duration clause.
        ("4+ years building dashboards in Power BI", "Power BI"),
        # Soft skills.
        ("Stakeholder management", "Communication"),
        # A compound of two skills resolves to one of them; which one is not
        # guaranteed, but it must be one of the skills actually named.
        ("Experience with Excel and Tableau", "Tableau"),
    ],
)
def test_model_names_reduce_to_canonical(model_name, expected):
    assert canonical(model_name) == expected


@pytest.mark.parametrize(
    "model_name",
    [
        # Not a skill. Keeping the wording preserves the evidence and lets the
        # education matcher handle it.
        "Bachelor degree in Engineering",
        "managing construction site operations",
        # Genuinely unknown, so it must survive untouched.
        "Some Exotic Unmapped Thing",
    ],
)
def test_unknown_names_are_left_alone(model_name):
    assert canonical(model_name) == model_name
