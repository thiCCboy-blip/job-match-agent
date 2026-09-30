"""Pydantic contracts shared by extraction, matching, and reporting.

Every LLM-produced structure lands here. Anything that fails validation is
rejected before it can influence a match score, so a malformed extraction
degrades to "unknown" rather than to a wrong number.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Importance(str, Enum):
    """How strongly a requirement gates the match."""

    MUST = "must"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


IMPORTANCE_WEIGHTS: dict[Importance, float] = {
    Importance.MUST: 1.0,
    Importance.PREFERRED: 0.45,
    Importance.NICE_TO_HAVE: 0.15,
}


class Proficiency(str, Enum):
    """Depth claimed by a resume for a skill, ordered weakest to strongest."""

    UNKNOWN = "unknown"
    EXPOSED = "exposed"
    WORKING = "working"
    PROFICIENT = "proficient"
    EXPERT = "expert"

    @property
    def value_score(self) -> float:
        """Multiplier applied to a matched requirement.

        `working` and above all score 1.0 on purpose. A job description lists
        capabilities it needs, not a mastery level, so penalising a candidate
        for how confidently a resume phrased a skill list measures writing
        rather than capability. Only evidence weaker than applied use is
        discounted, which is what "exposure with nothing to show" deserves.
        """
        return {
            Proficiency.UNKNOWN: 0.0,
            Proficiency.EXPOSED: 0.7,
            Proficiency.WORKING: 1.0,
            Proficiency.PROFICIENT: 1.0,
            Proficiency.EXPERT: 1.0,
        }[self]


class Evidence(BaseModel):
    """A verbatim span from the source document backing a claim.

    `text` must appear in the source. `line` is 1-indexed and refers to the
    line numbers produced by the ingest layer, so a report claim can be
    checked against the original file by a human.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=400)
    line: int = Field(ge=1)
    section: str | None = Field(default=None, max_length=80)

    @field_validator("text")
    @classmethod
    def _collapse(cls, v: str) -> str:
        return " ".join(v.split())


class Requirement(BaseModel):
    """A single capability the job description asks for."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    category: Literal[
        "skill",
        "experience",
        "education",
        "certification",
        "soft",
        "domain",
        "language",
        "other",
    ] = "skill"
    importance: Importance = Importance.PREFERRED
    min_years: float | None = Field(default=None, ge=0, le=50)
    evidence: Evidence | None = None


class JDProfile(BaseModel):
    """Structured view of a job description."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=160)
    company: str | None = Field(default=None, max_length=160)
    seniority: Literal["entry", "mid", "senior", "lead", "principal", "unknown"] = "unknown"
    requirements: list[Requirement] = Field(default_factory=list)
    extractor: str = "unknown"
    degraded: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_validator("requirements")
    @classmethod
    def _dedupe(cls, reqs: list[Requirement]) -> list[Requirement]:
        seen: dict[str, Requirement] = {}
        for r in reqs:
            key = r.name.strip().lower()
            if key not in seen:
                seen[key] = r
            elif r.importance is Importance.MUST:
                seen[key] = r
        return list(seen.values())


class ResumeSkill(BaseModel):
    """A capability the resume claims, with how deep the claim goes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    category: Literal[
        "skill",
        "experience",
        "education",
        "certification",
        "soft",
        "domain",
        "language",
        "other",
    ] = "skill"
    proficiency: Proficiency = Proficiency.WORKING
    years: float | None = Field(default=None, ge=0, le=50)
    evidence: Evidence | None = None


class ResumeProfile(BaseModel):
    """Structured view of a resume."""

    model_config = ConfigDict(extra="forbid")

    skills: list[ResumeSkill] = Field(default_factory=list)
    total_years_experience: float | None = Field(default=None, ge=0, le=60)
    highest_education: str | None = Field(default=None, max_length=160)
    education_evidence: Evidence | None = None
    certifications: list[str] = Field(default_factory=list)
    extractor: str = "unknown"
    degraded: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def _dedupe(cls, skills: list[ResumeSkill]) -> list[ResumeSkill]:
        best: dict[str, ResumeSkill] = {}
        for s in skills:
            key = s.name.strip().lower()
            if key not in best or s.proficiency.value_score > best[key].proficiency.value_score:
                best[key] = s
        return list(best.values())


class MatchStatus(str, Enum):
    MET = "met"
    PARTIAL = "partial"
    MISSING = "missing"
    CONTESTED = "contested"


class MatchOutcome(BaseModel):
    """Per-requirement verdict with the reasoning that produced it."""

    model_config = ConfigDict(extra="forbid")

    requirement: Requirement
    status: MatchStatus
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    matched_as: str | None = None
    matched_proficiency: Proficiency | None = None
    resume_evidence: Evidence | None = None
    method: Literal["exact", "synonym", "taxonomy", "embedding", "none"] = "none"
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str = ""


class MatchReport(BaseModel):
    """Full result of comparing a resume against a job description."""

    model_config = ConfigDict(extra="forbid")

    overall_score: float = Field(ge=0.0, le=100.0)
    verdict: Literal["strong", "plausible", "stretch", "weak"]
    must_have_coverage: float = Field(ge=0.0, le=1.0)
    outcomes: list[MatchOutcome] = Field(default_factory=list)
    blockers: list[MatchOutcome] = Field(default_factory=list)
    strengths: list[MatchOutcome] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    jd: JDProfile
    resume: ResumeProfile
    scoring_config: dict[str, str] = Field(default_factory=dict)
