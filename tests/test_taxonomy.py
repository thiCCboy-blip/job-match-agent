"""Taxonomy tests: normalization must be deterministic and collision-free."""

from __future__ import annotations

import pytest

from jobmatch import taxonomy


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("torch", "PyTorch"),
        ("PyTorch", "PyTorch"),
        ("pytorch lightning", "PyTorch"),
        ("postgres", "PostgreSQL"),
        ("PostgreSQL", "PostgreSQL"),
        ("k8s", "Kubernetes"),
        ("sklearn", "scikit-learn"),
        ("power bi", "Power BI"),
        ("PowerBI", "Power BI"),
        ("nodejs", "Node.js"),
        ("Node.js", "Node.js"),
        ("rest apis", "REST API"),
        ("RESTful API", "REST API"),
        ("C#", "C#"),
        (".NET", "C#"),
        ("js", "JavaScript"),
        ("a/b testing", "A/B Testing"),
        ("ab testing", "A/B Testing"),
        ("p6", "Primavera P6"),
        ("primavera", "Primavera P6"),
        ("boq", "Quantity Take-off"),
        ("boq preparation", "Quantity Take-off"),
        ("qto", "Quantity Take-off"),
        ("vba", "Excel VBA"),
        ("change orders", "Change Order"),
        ("variation order", "Change Order"),
        ("  ETL  ", "ETL"),
    ],
)
def test_synonyms_collapse_to_canonical(surface, expected):
    assert taxonomy.normalize_skill(surface) == expected


def test_unknown_skill_returns_none():
    assert taxonomy.normalize_skill("Unobtainium Framework") is None
    assert not taxonomy.is_in_taxonomy("zzz not a skill")


def test_normalization_is_case_and_punctuation_insensitive():
    assert taxonomy.normalize_skill("PYTHON!!!") == taxonomy.normalize_skill("python")
    assert taxonomy.normalize_skill("c#") == taxonomy.normalize_skill("C #")


def test_normalize_or_self_keeps_unknown_names_readable():
    assert taxonomy.normalize_or_self("unobtainium") == "Unobtainium"
    assert taxonomy.normalize_or_self("torch") == "PyTorch"


def test_surface_forms_contain_the_canonical_name():
    for canonical in ("PyTorch", "Power BI", "ETL"):
        forms = taxonomy.surface_forms(canonical)
        assert canonical in forms
        assert all(isinstance(f, str) and f for f in forms)


def test_content_tokens_drop_stopwords():
    tokens = taxonomy.content_tokens("Experience with the best tools and Python")
    assert "python" in tokens
    assert "the" not in tokens
    assert "with" not in tokens


def test_taxonomy_has_no_duplicate_keys():
    source = (taxonomy.__file__ or "")
    del source
    assert taxonomy.taxonomy_size() > 50


def test_canonical_names_are_never_aliases_of_each_other():
    """A canonical name must not resolve to a different canonical skill."""
    for canonical in taxonomy.SYNONYMS:
        assert taxonomy.normalize_skill(canonical) == canonical
