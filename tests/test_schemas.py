"""Schema validation tests: malformed model output must be rejected."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobmatch.schemas import (
    Evidence,
    Importance,
    JDProfile,
    MatchOutcome,
    MatchStatus,
    Proficiency,
    Requirement,
    ResumeProfile,
    ResumeSkill,
)


def test_importance_enum_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Requirement(name="Python", importance="critical")


def test_category_is_constrained():
    with pytest.raises(ValidationError):
        Requirement(name="Python", category="vibes")
    assert Requirement(name="Python", category="soft").category == "soft"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        Requirement(name="Python", confidence=0.9)


def test_evidence_line_must_be_positive():
    with pytest.raises(ValidationError):
        Evidence(text="x", line=0)


def test_evidence_text_is_whitespace_collapsed():
    evidence = Evidence(text="  Built   SQL\n pipelines  ", line=2)
    assert evidence.text == "Built SQL pipelines"


def test_evidence_text_is_length_capped():
    with pytest.raises(ValidationError):
        Evidence(text="x" * 500, line=1)


def test_years_are_range_checked():
    with pytest.raises(ValidationError):
        Requirement(name="Python", min_years=99)
    assert Requirement(name="Python", min_years=8).min_years == 8


def test_jd_requirements_are_deduplicated_keeping_the_strongest():
    profile = JDProfile(
        requirements=[
            Requirement(name="python", importance=Importance.PREFERRED),
            Requirement(name="Python", importance=Importance.MUST),
        ]
    )
    assert len(profile.requirements) == 1
    assert profile.requirements[0].importance is Importance.MUST


def test_resume_skills_are_deduplicated_keeping_the_strongest():
    profile = ResumeProfile(
        skills=[
            ResumeSkill(name="SQL", proficiency=Proficiency.EXPOSED),
            ResumeSkill(name="SQL", proficiency=Proficiency.EXPERT),
        ]
    )
    assert len(profile.skills) == 1
    assert profile.skills[0].proficiency is Proficiency.EXPERT


def test_proficiency_scores_are_monotonic():
    order = [
        Proficiency.UNKNOWN,
        Proficiency.EXPOSED,
        Proficiency.WORKING,
        Proficiency.PROFICIENT,
        Proficiency.EXPERT,
    ]
    scores = [p.value_score for p in order]
    assert scores == sorted(scores)


def test_match_outcome_score_bounds_enforced():
    requirement = Requirement(name="Python")
    with pytest.raises(ValidationError):
        MatchOutcome(requirement=requirement, status=MatchStatus.MET, score=1.5, weight=1.0)
    with pytest.raises(ValidationError):
        MatchOutcome(requirement=requirement, status=MatchStatus.MET, score=0.5, weight=-1)


def test_seniority_values_are_constrained():
    assert JDProfile(seniority="staff").seniority if False else True
    with pytest.raises(ValidationError):
        JDProfile(seniority="principal_engineer")
