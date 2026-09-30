"""Matching and scoring tests.

These pin the behaviour of the deterministic layer, which is where every
number in a report comes from. Embeddings are disabled here on purpose: a test
that depends on a 400 MB model download is a test nobody runs.
"""

from __future__ import annotations

import pytest

from jobmatch import matcher
from jobmatch.schemas import (
    Evidence,
    Importance,
    JDProfile,
    Proficiency,
    Requirement,
    ResumeProfile,
    ResumeSkill,
)


def jd(*requirements: Requirement) -> JDProfile:
    return JDProfile(title="Test Role", requirements=list(requirements))


def resume(*skills: ResumeSkill, years: float | None = None) -> ResumeProfile:
    return ResumeProfile(skills=list(skills), total_years_experience=years)


def skill(name: str, proficiency=Proficiency.PROFICIENT, **kwargs) -> ResumeSkill:
    return ResumeSkill(name=name, proficiency=proficiency, **kwargs)


def config(**overrides) -> matcher.MatchConfig:
    base = {"use_embeddings": False}
    base.update(overrides)
    return matcher.MatchConfig(**base)


def test_exact_skill_is_met():
    report = matcher.analyze(
        jd(Requirement(name="Python", importance=Importance.MUST)),
        resume(skill("Python")),
        config(),
    )
    assert report.overall_score == 100.0
    assert report.verdict == "strong"
    assert report.outcomes[0].status.value == "met"


def test_synonym_variant_still_matches():
    report = matcher.analyze(
        jd(Requirement(name="PyTorch", importance=Importance.MUST)),
        resume(skill("torch")),
        config(),
    )
    assert report.verdict == "strong"
    assert report.outcomes[0].matched_as == "torch"


def test_missing_must_have_caps_the_verdict():
    report = matcher.analyze(
        jd(
            Requirement(name="Kubernetes", importance=Importance.MUST),
            Requirement(name="Python", importance=Importance.PREFERRED),
            Requirement(name="SQL", importance=Importance.PREFERRED),
            Requirement(name="Docker", importance=Importance.PREFERRED),
        ),
        resume(skill("Python"), skill("SQL"), skill("Docker")),
        config(),
    )
    # Three of four requirements match, but the only mandatory one is absent,
    # so the verdict is capped even though the weighted mean is mid-range.
    assert 50.0 < report.overall_score < 65.0
    assert report.verdict == "weak"
    assert report.must_have_coverage == 0.0
    assert [b.requirement.name for b in report.blockers] == ["Kubernetes"]


def test_must_have_gate_can_be_disabled():
    report = matcher.analyze(
        jd(
            Requirement(name="Kubernetes", importance=Importance.MUST),
            Requirement(name="Python", importance=Importance.PREFERRED),
            Requirement(name="SQL", importance=Importance.PREFERRED),
        ),
        resume(skill("Python"), skill("SQL")),
        config(must_have_gate=False),
    )
    assert report.verdict != "weak"


def test_preferred_weighs_less_than_must():
    must_weight = matcher.analyze(
        jd(Requirement(name="Absent", importance=Importance.MUST)),
        resume(skill("Python")),
        config(),
    ).outcomes[0].weight
    pref_weight = matcher.analyze(
        jd(Requirement(name="Absent", importance=Importance.PREFERRED)),
        resume(skill("Python")),
        config(),
    ).outcomes[0].weight
    assert must_weight > pref_weight
    assert pref_weight == pytest.approx(0.45)


def test_deeper_proficiency_scores_higher():
    shallow = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.MUST)),
        resume(skill("SQL", Proficiency.EXPOSED)),
        config(),
    ).overall_score
    deep = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.MUST)),
        resume(skill("SQL", Proficiency.EXPERT)),
        config(),
    ).overall_score
    assert deep > shallow


def test_years_shortfall_reduces_but_does_not_zero_the_score():
    met = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.MUST, min_years=3)),
        resume(skill("SQL", years=5)),
        config(),
    )
    short = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.MUST, min_years=8)),
        resume(skill("SQL", years=1)),
        config(),
    )
    assert met.overall_score > short.overall_score
    assert short.overall_score > 0
    assert "year minimum" in short.outcomes[0].explanation


def test_years_sensitivity_knob_is_wired():
    requirement = jd(Requirement(name="SQL", importance=Importance.MUST, min_years=8))
    profile = resume(skill("SQL", years=2))
    gentle = matcher.analyze(requirement, profile, config(min_years_sensitivity=0.3))
    strict = matcher.analyze(requirement, profile, config(min_years_sensitivity=2.0))
    assert strict.overall_score < gentle.overall_score


def test_overlapping_roles_use_merged_experience_for_years():
    report = matcher.analyze(
        jd(Requirement(name="Project Management", importance=Importance.MUST, min_years=3)),
        resume(skill("Project Management"), years=3.5),
        config(),
    )
    assert report.outcomes[0].status.value in {"met", "partial"}


def test_evidence_appears_in_the_report():
    evidence = Evidence(text="Built SQL pipelines", line=4, section="experience")
    report = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.MUST)),
        resume(ResumeSkill(name="SQL", proficiency=Proficiency.EXPERT, evidence=evidence)),
        config(),
    )
    assert report.outcomes[0].resume_evidence is not None
    assert report.outcomes[0].resume_evidence.line == 4


def test_boost_does_not_promote_unrelated_missing_skills():
    report = matcher.analyze(
        jd(
            Requirement(name="Kubernetes", importance=Importance.PREFERRED),
            Requirement(name="Power BI", importance=Importance.PREFERRED),
        ),
        resume(ResumeSkill(name="Python", evidence=Evidence(text="wrote python code", line=1))),
        config(),
    )
    assert all(o.status.value == "missing" for o in report.outcomes)


def test_boost_recovers_a_skill_mentioned_only_inside_another_evidence_line():
    report = matcher.analyze(
        jd(Requirement(name="SQL", importance=Importance.PREFERRED)),
        resume(
            ResumeSkill(
                name="Python",
                evidence=Evidence(text="Built SQL and Python pipelines daily", line=3),
            )
        ),
        config(),
    )
    outcome = report.outcomes[0]
    assert outcome.status.value == "partial"
    assert "corroborated" in outcome.explanation


def test_weak_resume_scores_low():
    report = matcher.analyze(
        jd(
            Requirement(name="Kubernetes", importance=Importance.MUST),
            Requirement(name="Rust", importance=Importance.MUST),
            Requirement(name="Go", importance=Importance.MUST),
            Requirement(name="Scala", importance=Importance.MUST),
        ),
        resume(skill("Excel")),
        config(),
    )
    assert report.overall_score < 20
    assert report.verdict == "weak"
    assert report.must_have_coverage == 0.0


def test_empty_jd_does_not_crash():
    report = matcher.analyze(jd(), resume(skill("Python")), config())
    assert report.overall_score == 0.0
    assert report.verdict == "weak"


def test_must_have_coverage_is_one_when_no_musts_exist():
    report = matcher.analyze(
        jd(Requirement(name="Python", importance=Importance.PREFERRED)),
        resume(skill("Python")),
        config(),
    )
    assert report.must_have_coverage == 1.0


def test_suggestions_name_the_specific_gap():
    report = matcher.analyze(
        jd(Requirement(name="Kubernetes", importance=Importance.MUST)),
        resume(skill("Python")),
        config(),
    )
    assert any("Kubernetes" in s for s in report.suggestions)


def test_scoring_config_is_recorded_for_reproducibility():
    report = matcher.analyze(
        jd(Requirement(name="Python", importance=Importance.MUST)),
        resume(skill("Python")),
        config(),
    )
    assert report.scoring_config["use_embeddings"] == "False"


def test_duplicate_resume_skills_collapse_to_the_strongest():
    index = matcher.ResumeIndex.build(
        resume(skill("SQL", Proficiency.EXPOSED), skill("SQL", Proficiency.EXPERT))
    )
    assert index.by_canonical["SQL"].proficiency is Proficiency.EXPERT


def test_outcomes_are_sorted_by_weight_then_score():
    report = matcher.analyze(
        jd(
            Requirement(name="AbsentPreferred", importance=Importance.PREFERRED),
            Requirement(name="Python", importance=Importance.MUST),
        ),
        resume(skill("Python")),
        config(),
    )
    assert report.outcomes[0].requirement.name == "Python"


# --- degree requirements -------------------------------------------------
#
# A qualification is a level, not a keyword. These pin the cases where treating
# it as a keyword match gives the candidate the wrong answer.


def education_resume(qualification: str) -> ResumeProfile:
    return ResumeProfile(
        skills=[],
        highest_education=qualification,
        education_evidence=Evidence(text=qualification, line=9, section="education"),
    )


def test_bachelors_requirement_is_met_by_btech():
    report = matcher.analyze(
        jd(Requirement(name="Bachelor's degree in Engineering", category="education")),
        education_resume("B.Tech Computer Engineering"),
        config(),
    )
    assert report.outcomes[0].status is matcher.MatchStatus.MET


def test_masters_requirement_is_not_met_by_btech():
    report = matcher.analyze(
        jd(Requirement(name="MBA", category="education", importance=Importance.PREFERRED)),
        education_resume("B.Tech Civil Engineering"),
        config(),
    )
    assert report.outcomes[0].status is matcher.MatchStatus.MISSING


def test_higher_degree_satisfies_lower_requirement():
    report = matcher.analyze(
        jd(Requirement(name="Bachelor's degree", category="education")),
        education_resume("Master of Business Administration"),
        config(),
    )
    assert report.outcomes[0].status is matcher.MatchStatus.MET


def test_education_evidence_cites_the_real_line():
    report = matcher.analyze(
        jd(Requirement(name="Bachelor's degree", category="education")),
        education_resume("B.Tech Computer Engineering"),
        config(),
    )
    assert report.outcomes[0].resume_evidence.line == 9


def test_no_education_line_means_degree_is_unmet():
    report = matcher.analyze(
        jd(Requirement(name="Bachelor's degree", category="education")),
        ResumeProfile(skills=[]),
        config(),
    )
    assert report.outcomes[0].status is matcher.MatchStatus.MISSING


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PhD in Civil Engineering", 5),
        ("Master of Business", 4),
        ("MBA", 4),
        ("B.Tech Civil Engineering", 3),
        ("Bachelor of Science", 3),
        ("Diploma in Civil", 2),
        ("No qualification here", 0),
    ],
)
def test_degree_levels(text, expected):
    assert matcher._degree_level(text) == expected


# --- compound requirements ----------------------------------------------
#
# Postings bundle a duration with the domain it applies to. The duration is
# scored separately, so the domain phrase is what has to resolve.


def test_duration_plus_domain_matches_the_domain_skill():
    report = matcher.analyze(
        jd(
            Requirement(
                name="8+ years managing construction site operations",
                min_years=8.0,
                importance=Importance.MUST,
            )
        ),
        resume(skill("Site Supervision"), years=5.3),
        config(),
    )
    outcome = report.outcomes[0]
    assert outcome.matched_as == "Site Supervision"
    # Found the skill, but short of the stated years.
    assert outcome.status is not matcher.MatchStatus.MET


def test_unrelated_compound_requirement_does_not_match():
    report = matcher.analyze(
        jd(Requirement(name="5+ years managing hospital pharmacy operations")),
        resume(skill("Site Supervision")),
        config(),
    )
    assert report.outcomes[0].status is matcher.MatchStatus.MISSING
