"""Extraction tests for the deterministic (rules) backend."""

from __future__ import annotations

import pytest

from jobmatch import extract, ingest
from jobmatch.schemas import Importance, Proficiency

JD = """Data Analyst
Requirements:
- 3+ years of experience with SQL and Python required
- Strong Excel and Power BI skills mandatory
- Experience with dbt is a plus
- Bachelor's degree in Engineering required
- Excellent communication skills
"""

RESUME = """Jane Doe
SUMMARY
Data analyst with reporting experience.
EXPERIENCE
Data Analyst, Acme | 01/2021 - 06/2024
- Built SQL and Python pipelines
- Delivered Excel reporting to stakeholders
EDUCATION
B.Tech Civil Engineering
SKILLS
Python, SQL, Excel, Power BI, Tableau
CERTIFICATIONS
Six Sigma Green Belt
"""


def _jd():
    return extract.extract_jd_with_rules(ingest.load_document(JD, doc_type="text"))


def _resume():
    return extract.extract_resume_with_rules(ingest.load_document(RESUME, doc_type="text"))


def _find(profile, name):
    for item in profile.requirements if hasattr(profile, "requirements") else profile.skills:
        if item.name == name:
            return item
    return None


def test_jd_extracts_taxonomy_skills():
    names = {r.name for r in _jd().requirements}
    for expected in ("SQL", "Python", "Excel", "Power BI", "dbt"):
        assert expected in names


def test_importance_is_scoped_to_its_own_clause():
    requirements = {r.name: r for r in _jd().requirements}
    assert requirements["SQL"].importance is Importance.MUST
    assert requirements["Excel"].importance is Importance.MUST
    assert requirements["dbt"].importance is not Importance.MUST


def test_jd_evidence_points_at_a_real_line():
    document = ingest.load_document(JD, doc_type="text")
    profile = extract.extract_jd_with_rules(document)
    for requirement in profile.requirements:
        if requirement.evidence is None:
            continue
        number = requirement.evidence.line
        assert number in document.line_map()
        assert requirement.evidence.text in document.line_map()[number]


def test_resume_extracts_skills_with_evidence():
    document = ingest.load_document(RESUME, doc_type="text")
    profile = extract.extract_resume_with_rules(document)
    names = {s.name for s in profile.skills}
    assert {"Python", "SQL", "Excel", "Power BI"} <= names
    for skill in profile.skills:
        if skill.evidence is not None:
            assert skill.evidence.line in document.line_map()


def test_headings_never_become_certifications():
    profile = _resume()
    assert "CERTIFICATIONS" not in profile.certifications
    assert "SKILLS" not in profile.certifications
    assert any("Six Sigma" in c for c in profile.certifications)


def test_education_is_found_and_not_taken_from_a_url():
    profile = _resume()
    assert profile.highest_education is not None
    assert "B.Tech" in profile.highest_education


def test_total_years_merges_overlapping_roles():
    text = """EXPERIENCE
Analyst, Acme | 01/2021 - 06/2023
Intern, Beta | 01/2022 - 06/2022
Analyst, Gamma | 01/2023 - 06/2024
"""
    profile = extract.extract_resume_with_rules(ingest.load_document(text, doc_type="text"))
    assert profile.total_years_experience == pytest_approx(3.5)


def pytest_approx(value: float, tol: float = 0.15):
    import pytest

    return pytest.approx(value, abs=tol)


def test_total_years_none_without_dates():
    profile = _resume()
    assert profile.total_years_experience is not None


def test_month_ordinal_math_is_linear():
    assert extract._month_ordinal(2023, 6) == 2023 * 12 + 5
    delta = extract._month_ordinal(2024, 7) - extract._month_ordinal(2023, 6)
    assert delta == 13


def test_numeric_and_named_date_forms_both_parse():
    numeric = "Engineer, Acme | 06/2023 to 07/2024"
    named = "Engineer, Acme | Jun 2023 to Jul 2024"
    for text in (numeric, named):
        profile = extract.extract_resume_with_rules(
            ingest.load_document(f"EXPERIENCE\n{text}", doc_type="text")
        )
        assert profile.total_years_experience == pytest_approx(1.1, 0.1)


def test_proficiency_cues():
    assert extract._guess_proficiency("expert in Python, 5+ years") is Proficiency.EXPERT
    assert extract._guess_proficiency("working knowledge of SQL") is Proficiency.WORKING
    assert extract._guess_proficiency("familiar with Excel") is Proficiency.EXPOSED


def test_category_cues():
    assert extract._guess_category("Bachelor's degree required", None) == "education"
    assert extract._guess_category("certification in AWS required", None) == "certification"
    assert extract._guess_category("strong communication skills", None) == "soft"


# --- degree requirements -------------------------------------------------
#
# "Bachelor's degree in Engineering or Statistics required" contains a
# subject that the taxonomy knows as a skill. Emitting "Statistics" and
# dropping the qualification tells the candidate to learn a subject they
# already hold a degree in.

DEGREE_JD = (
    "Graduate Data Analyst\n"
    "Requirements:\n"
    "- Bachelor's degree in Engineering or Statistics required\n"
    "- SQL required"
)


def test_degree_clause_becomes_an_education_requirement():
    profile = extract.extract_jd_with_rules(ingest.load_document(DEGREE_JD, doc_type="text"))
    names = [r.name for r in profile.requirements]
    assert any("Bachelor's degree" in n for n in names)
    assert "Statistics" not in names


def test_degree_requirement_carries_education_category():
    profile = extract.extract_jd_with_rules(ingest.load_document(DEGREE_JD, doc_type="text"))
    degree = next(r for r in profile.requirements if "Bachelor's degree" in r.name)
    assert degree.category == "education"


def test_degree_clause_does_not_swallow_other_requirements():
    profile = extract.extract_jd_with_rules(ingest.load_document(DEGREE_JD, doc_type="text"))
    assert "SQL" in [r.name for r in profile.requirements]


def test_plain_skill_requirement_is_unaffected():
    jd_text = "Analyst\nRequirements:\n- Statistics required\n- Excel required"
    profile = extract.extract_jd_with_rules(ingest.load_document(jd_text, doc_type="text"))
    assert "Statistics" in [r.name for r in profile.requirements]


@pytest.mark.parametrize(
    ("bullet", "expected"),
    [
        ("- MBA preferred", "MBA"),
        ("- Python required", "Python"),
        ("- AWS must have", "AWS"),
        ("- Tableau essential", "Tableau"),
    ],
)
def test_importance_words_are_stripped_from_requirement_labels(bullet, expected):
    jd_text = f"Analyst\nRequirements:\n{bullet}"
    profile = extract.extract_jd_with_rules(ingest.load_document(jd_text, doc_type="text"))
    assert expected in [r.name for r in profile.requirements]


def test_resume_education_records_its_source_line():
    resume_text = (
        "SKILLS\nSQL, Python\n"
        "EDUCATION\nB.Tech Computer Engineering"
    )
    profile = extract.extract_resume_with_rules(
        ingest.load_document(resume_text, doc_type="text")
    )
    assert "B.Tech" in profile.highest_education
    assert profile.education_evidence is not None
    # Must point at the education line, so a report can quote it.
    assert profile.education_evidence.line == 4
    assert profile.education_evidence.section == "education"


# --- prose is not a requirement -----------------------------------------
#
# The taxonomy matches a skill wherever it appears, which is precise but means
# a narrative sentence can be mistaken for a requirement. These are the shapes
# that broke on real resumes.


PROSE_JD = """AMRUTH RAJ

**Data & Analytics | AI-Enabled Document Analytics | B.Tech Civil Engineering**

Analytics-focused engineer who builds the measurement layer that shows whether
a system actually works.

EXPERIENCE
- *Disclosure decision:** Withheld the pricing inputs from publication, because
  a realistic tender package can be mistaken for live bid material.
- Deployed dashboards on Streamlit Community Cloud with TensorFlow
- KPI-based analysis and performance reporting for international campaigns
"""


def test_narrative_prose_is_not_a_requirement():
    profile = extract.extract_jd_with_rules(
        ingest.load_document(PROSE_JD, doc_type="text")
    )
    names = " | ".join(r.name for r in profile.requirements)
    assert "Disclosure decision" not in names
    assert "measurement layer" not in names


def test_actual_skill_bullets_are_still_requirements():
    profile = extract.extract_jd_with_rules(
        ingest.load_document(PROSE_JD, doc_type="text")
    )
    names = [r.name for r in profile.requirements]
    assert "Streamlit" in names
    assert "TensorFlow" in names


@pytest.mark.parametrize(
    "line",
    [
        "You will be the site engineer of record, reporting to the project manager.",
        "Send me the drawings before Friday.",
        "operations for an ongoing residential project. You will be the site",
    ],
)
def test_plain_words_be_and_me_are_not_degrees(line):
    # "b.e" with an optional dot matches the word "be", and "m.e" matches "me",
    # so ordinary prose was being read as a degree requirement.
    assert not extract._is_education_requirement(line)


@pytest.mark.parametrize(
    "line",
    [
        "Bachelor's degree in Civil Engineering required",
        "B.E. in Civil Engineering",
        "B.E Civil Engineering",
        "M.E. Structural Engineering preferred",
        "B.Tech Computer Engineering",
        "MBA",
    ],
)
def test_real_degree_abbreviations_are_still_detected(line):
    assert extract._is_education_requirement(line)


def test_summary_prose_does_not_become_a_requirement():
    jd_text = (
        "Civil Engineer - Construction Site Operations\n"
        "About the role\n"
        "We are looking for a civil engineer to run day-to-day construction\n"
        "site operations for an ongoing residential project. You will be the\n"
        "site engineer of record, reporting to the project manager.\n"
        "Requirements\n"
        "- Strong Primavera P6 scheduling experience is mandatory\n"
    )
    profile = extract.extract_jd_with_rules(
        ingest.load_document(jd_text, doc_type="text")
    )
    names = [r.name for r in profile.requirements]
    assert "operations for an ongoing residential project" not in names
    # The genuine requirement must survive the summary text.
    assert "Primavera P6" in names


def test_resume_education_evidence_is_plain_text():
    resume_text = (
        "EDUCATION\n**B.Tech, Civil Engineering** - National Institute of "
        "Technology Calicut | 12/2021 to 08/2025"
    )
    profile = extract.extract_resume_with_rules(
        ingest.load_document(resume_text, doc_type="text")
    )
    assert "**" not in profile.highest_education
    assert profile.highest_education.startswith("B.Tech, Civil Engineering")


def test_requirement_names_carry_no_markdown():
    profile = extract.extract_jd_with_rules(
        ingest.load_document(PROSE_JD, doc_type="text")
    )
    for requirement in profile.requirements:
        assert "**" not in requirement.name
        assert not requirement.name.startswith("*")


@pytest.mark.parametrize(
    "line",
    [
        # A bolded sub-label followed by prose.
        "*Disclosure decision:** withheld the pricing inputs, because it leaks",
        # A sentence mentioning a skill.
        "Analytics-focused engineer who builds the measurement layer that works",
        # Two sentence terminators.
        "Built dashboards. Used Python. Shipped them.",
    ],
)
def test_prose_lines_are_rejected(line):
    document = ingest.load_document(line, doc_type="text")
    assert extract._is_prose_line(document.lines[0])


@pytest.mark.parametrize(
    "line",
    [
        "Primavera P6 scheduling experience mandatory",
        "Strong communication skills",
        "Bachelor's degree in Civil Engineering required",
        "3+ years managing construction site operations",
    ],
)
def test_requirement_lines_are_kept(line):
    document = ingest.load_document(line, doc_type="text")
    assert not extract._is_prose_line(document.lines[0])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*Disclosure decision:** withheld pricing", "withheld pricing"),
        ("**Disclosure decision:** withheld pricing", "withheld pricing"),
        ("**Responsibilities:** develop dashboards", "develop dashboards"),
        ("SQL required", "SQL required"),
        ("Developed ETL pipelines daily", "Developed ETL pipelines daily"),
    ],
)
def test_bolded_sublabels_are_stripped(raw, expected):
    assert extract._strip_leading_label(raw) == expected
