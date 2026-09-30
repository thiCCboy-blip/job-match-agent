"""Deterministic matching and scoring.

The model never sees this file. Extraction produces candidates; every score,
gap, and verdict below is computed by rules that can be read, tested, and
argued with. That split is the whole design: a wrong number here is a bug you
can fix, whereas a wrong number from a model is a shrug.

Scoring model
-------------
Each requirement carries a weight from its importance (must 1.0, preferred
0.45, nice-to-have 0.15). A requirement's own score is a product of

    coverage  x  proficiency  x  recency-independent seniority factor

where `coverage` is how well the resume addresses it at all, `proficiency` is
how deep the claim goes, and the seniority factor encodes the usual
"3 years means 3 years" expectation when a job states a duration.

The overall score is the weighted mean, then gated: a missing must-have
requirement caps the verdict regardless of how much else matches, because a
knockout requirement should not be averaged away.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from jobmatch import taxonomy
from jobmatch.schemas import (
    IMPORTANCE_WEIGHTS,
    Evidence,
    Importance,
    JDProfile,
    MatchOutcome,
    MatchReport,
    MatchStatus,
    Proficiency,
    Requirement,
    ResumeProfile,
    ResumeSkill,
)

EMBEDDING_MATCH_THRESHOLD = 0.62
EMBEDDING_PARTIAL_THRESHOLD = 0.48
MIN_YEAR_PARTIAL_CREDIT = 0.4
# Credit for matching a named practice through one of its implementing tools.
IMPLEMENTATION_CREDIT = 0.55

MatchMethod = Literal["exact", "synonym", "taxonomy", "embedding", "none"]

VERDICT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "strong": (78.0, 1.0),
    "plausible": (60.0, 0.85),
    "stretch": (42.0, 0.5),
}


@dataclass
class MatchConfig:
    """Tunable scoring knobs, recorded in the report so runs are reproducible."""

    use_embeddings: bool = True
    use_recency: bool = False
    partial_credit: float = 0.5
    min_years_sensitivity: float = 1.0
    embedding_weight: float = 0.35
    evidence_credit: float = 1.0
    must_have_gate: bool = True
    embedder: object | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, str]:
        return {
            "use_embeddings": str(self.use_embeddings),
            "partial_credit": f"{self.partial_credit}",
            "min_years_sensitivity": f"{self.min_years_sensitivity}",
            "embedding_weight": f"{self.embedding_weight}",
            "must_have_gate": str(self.must_have_gate),
        }


# --------------------------------------------------------------------------
# resume indexing
# --------------------------------------------------------------------------


@dataclass
class ResumeIndex:
    """Skills keyed by canonical name, plus a list for embedding fallback."""

    by_canonical: dict[str, ResumeSkill]
    all_skills: list[ResumeSkill]
    unindexed: list[ResumeSkill]
    # (embedder, pool, vectors); populated lazily on first embedding lookup.
    _vectors: tuple[object, list[ResumeSkill], list[list[float]]] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def build(cls, resume: ResumeProfile) -> "ResumeIndex":
        by_canonical: dict[str, ResumeSkill] = {}
        unindexed: list[ResumeSkill] = []
        for skill in resume.skills:
            canonical = taxonomy.normalize_skill(skill.name)
            if canonical is None:
                unindexed.append(skill)
                continue
            existing = by_canonical.get(canonical)
            if existing is None or skill.proficiency.value_score > existing.proficiency.value_score:
                by_canonical[canonical] = skill
        return cls(by_canonical=by_canonical, all_skills=resume.skills, unindexed=unindexed)

    def surface_index(self) -> dict[str, str]:
        """Normalized surface form -> canonical, for substring-level matching."""
        index: dict[str, str] = {}
        for canonical, skill in self.by_canonical.items():
            for form in taxonomy.surface_forms(canonical):
                index[taxonomy._norm_key(form)] = canonical
            index.setdefault(taxonomy._norm_key(skill.name), canonical)
        for skill in self.unindexed:
            index.setdefault(taxonomy._norm_key(skill.name), skill.name)
        return index


# --------------------------------------------------------------------------
# individual match logic
# --------------------------------------------------------------------------


def _score_from_years(
    required_years: float, available_years: float | None, sensitivity: float
) -> tuple[float, str]:
    """Compare claimed years against a stated minimum, with a partial-credit band.

    The penalty is bounded by `MIN_YEAR_PARTIAL_CREDIT` so meeting a stated
    minimum is never the only route to a pass. Someone with one year of SQL
    against an eight-year posting is under-qualified, but they are not a
    non-match, and a score of exactly zero would read as "no evidence at all".
    """
    if available_years is None:
        return 0.85, f"years not stated; credited at 0.85 against a {required_years:g}-year minimum"
    ratio = available_years / required_years if required_years else 1.0
    if ratio >= 1.0:
        return 1.0, f"{available_years:g} years stated against a {required_years:g}-year minimum"
    shortfall = 1.0 - ratio
    penalty = min(1.0 - MIN_YEAR_PARTIAL_CREDIT, shortfall * 1.6 * sensitivity)
    score = max(0.0, 1.0 - penalty)
    return score, f"{available_years:g} years stated against a {required_years:g}-year minimum"


def _requirement_years(skill: ResumeSkill, requirement: Requirement, resume: ResumeProfile) -> float | None:
    if skill.years:
        return skill.years
    if requirement.min_years and resume.total_years_experience:
        return resume.total_years_experience
    return None


def _text_overlap(requirement: Requirement, skill: ResumeSkill) -> float:
    """Token overlap between the requirement wording and the evidence that backs it."""
    req_tokens = taxonomy.content_tokens(requirement.name)
    if not req_tokens:
        return 0.0
    evidence_text = skill.evidence.text if skill.evidence else skill.name
    evidence_tokens = taxonomy.content_tokens(evidence_text)
    if not evidence_tokens:
        return 0.0
    return len(req_tokens & evidence_tokens) / len(req_tokens)


# A tool that implements a broader practice. A posting asking for "CI/CD" and a
# resume naming "GitHub Actions" are a real match that no alias table should
# assert, because the reverse does not hold: a Jenkins user has not necessarily
# used GitHub Actions. These pairs are only consulted through the embedding
# fallback, and the match is reported as `contested` rather than `met`, so a
# human still decides.
IMPLEMENTS: dict[str, tuple[str, ...]] = {
    "CI/CD": ("GitHub Actions", "Jenkins", "GitLab CI", "Azure Pipelines", "CircleCI"),
    "Data Analysis": ("Power BI", "Tableau", "Excel", "Statistics"),
    "Data Engineering": ("ETL", "ETL Tools"),
    "REST API": ("FastAPI", "Flask", "Django", "gRPC"),
    "NLP": ("LLM", "RAG"),
    "Docker": ("Kubernetes",),
    "Project Management": ("Primavera P6", "Scheduling"),
    "Version Control": ("Git",),
    "Data Analytics": ("Power BI", "Tableau", "Excel"),
}


def _implementation_candidates(name: str) -> list[str]:
    """Reverse the IMPLEMENTS map: a tool's postings may name the practice."""
    canonical = taxonomy.normalize_skill(name)
    normalized = taxonomy._norm_key(name)
    out: list[str] = []
    for practice, tools in IMPLEMENTS.items():
        practice_canonical = taxonomy.normalize_skill(practice)
        if practice_canonical == canonical or taxonomy._norm_key(practice) == normalized:
            out.extend(tools)
    return out


# Ordered by level. A B.Tech satisfies a bachelor's requirement but must not
# satisfy an MBA one, so the level has to be compared, not just the presence of
# any degree token.
_DEGREE_LEVELS: list[tuple[str, int]] = [
    (r"\b(ph\.?d|doctorate|doctoral)\b", 5),
    (r"\b(master|m\.?tech|m\.?e\b|m\.?sc|m\.?com|mba|m\.?ba)\b", 4),
    (r"\b(bachelor|b\.?tech|b\.?e\b|b\.?sc|b\.?com|b\.?ba|degree)\b", 3),
    (r"\b(diploma|polytechnic|foundation)\b", 2),
]


def _degree_level(text: str) -> int:
    """Highest degree level stated in the text, 0 if none."""
    for pattern, level in _DEGREE_LEVELS:
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return 0


def _match_education(requirement: Requirement, resume: ResumeProfile) -> ResumeSkill | None:
    """Satisfy a degree requirement from the resume's education line.

    A posting asking for "Bachelor's degree in Engineering or Statistics"
    is met by a B.Tech, even though no single word is shared. The comparison is
    on degree *level* plus any stated subject, so a candidate is not blocked by
    the wording of the qualification, but an MBA posting is not met by a
    bachelor's.
    """
    if not resume.highest_education:
        return None
    text = resume.highest_education
    required_level = _degree_level(requirement.name)
    held_level = _degree_level(text)
    if required_level == 0 or held_level < required_level:
        return None

    have_subjects = taxonomy.content_tokens(text)
    req_subjects = taxonomy.content_tokens(requirement.name) & have_subjects
    # A subject stated on both sides is a direct hit; otherwise the degree
    # level alone carries it, at working rather than expert proficiency.
    level_exact = held_level == required_level

    evidence = resume.education_evidence or Evidence(
        text=text[:400], line=1, section="education"
    )
    return ResumeSkill(
        name=resume.highest_education[:120],
        category="education",
        proficiency=(
            Proficiency.EXPERT
            if (req_subjects or level_exact)
            else Proficiency.WORKING
        ),
        evidence=evidence.model_copy(update={"text": text[:400]}),
    )


# A requirement like "8+ years managing construction site operations" is a
# duration plus a domain phrase. The duration is checked separately, so the
# domain phrase is what identifies the skill.
_DURATION_NOISE = re.compile(
    r"\b\d+\+?\s*(?:years?|yrs?)\b|\byears? of\b|\bof experience\b|"
    r"\bexperience (?:in|with|of)\b|\bmanaging\b|\bmanagement of\b|\bresponsible for\b",
    re.IGNORECASE,
)

# Split on the separators directly. Wrapping `/` in `\b` cannot match it,
# since `/` is not a word character, and the resulting pattern backtracks
# catastrophically on any name containing a slash. Hyphenated and slashed
# skills are protected first, because "CI/CD" must not become "CI" and "CD".
_COMPOUND_SPLIT = re.compile(r"\s+(?:and|or)\s+|,", re.IGNORECASE)
_SLASH_SPLIT = re.compile(r"/")

# Skills that contain a slash or hyphen and must survive splitting intact.
_SLASHED_SKILL = re.compile(r"\b(?:[A-Za-z0-9+#.]+[/-])+[A-Za-z0-9+#.]+\b")

_SPLIT_SENTINEL = "\x00"
_SENTINEL_RE = re.compile(r"\x00(\d+)\x00")


def _split_compound(text: str) -> list[str]:
    """Split a compound requirement into candidate skill phrases."""
    stashed: list[str] = []

    def hide(match: re.Match[str]) -> str:
        stashed.append(match.group(0))
        return f"{_SPLIT_SENTINEL}{len(stashed) - 1}{_SPLIT_SENTINEL}"

    def reveal(match: re.Match[str]) -> str:
        return stashed[int(match.group(1))]

    # Protect "CI/CD" and "full-stack" before any slash or comma is treated
    # as a separator.
    protected = _SLASHED_SKILL.sub(hide, text)
    parts: list[str] = []
    for chunk in _COMPOUND_SPLIT.split(protected):
        for part in _SLASH_SPLIT.split(chunk):
            if part.strip():
                parts.append(part)

    return [
        _SENTINEL_RE.sub(reveal, part).strip(" .;:-")
        for part in parts
        if part.strip()
    ]


def _match_compound_requirement(
    requirement: Requirement, index: ResumeIndex
) -> tuple[ResumeSkill | None, str]:
    """Match a multi-clause requirement on the skill phrase inside it.

    Postings bundle experience with the thing being experienced ("5+ years
    managing construction sites"). The taxonomy resolves neither the whole
    string nor the duration, so each candidate phrase is tried separately and
    the requirement matches if any one of them names a skill the resume has.
    """
    text = _DURATION_NOISE.sub(" ", requirement.name)
    candidates = [part for part in _split_compound(text) if len(part) > 2]

    # Also slide a window over the phrase: "construction site operations"
    # needs no split, it just needs the sub-phrase "construction site" tried.
    words = text.split()
    for start in range(len(words)):
        for width in range(4, 0, -1):
            chunk = " ".join(words[start : start + width])
            canonical = taxonomy.normalize_skill(chunk)
            if canonical and canonical in index.by_canonical:
                return index.by_canonical[canonical], "taxonomy"

    for candidate in candidates:
        canonical = taxonomy.normalize_skill(candidate)
        if canonical and canonical in index.by_canonical:
            return index.by_canonical[canonical], "taxonomy"
    return None, "none"


def _match_one(
    requirement: Requirement,
    index: ResumeIndex,
    resume: ResumeProfile,
    config: MatchConfig,
    embedder: object | None,
) -> MatchOutcome:
    weight = IMPORTANCE_WEIGHTS[requirement.importance]
    canonical = taxonomy.normalize_skill(requirement.name)

    hit = None
    method: MatchMethod = "none"
    similarity: float | None = None

    if canonical and canonical in index.by_canonical:
        hit, method = index.by_canonical[canonical], "taxonomy"
    else:
        key = taxonomy._norm_key(requirement.name)
        if key in index.surface_index():
            matched_name = index.surface_index()[key]
            hit = index.by_canonical.get(matched_name) or _find_by_name(index.all_skills, matched_name)
            method = "synonym" if hit else "none"

    if hit is None and requirement.category == "education":
        hit, method = _match_education(requirement, resume), "taxonomy"

    if hit is None:
        compound_hit, compound_method = _match_compound_requirement(requirement, index)
        if compound_hit is not None:
            hit, method = compound_hit, compound_method

    if hit is None:
        # A tool that implements the named practice is a real but weaker
        # signal than the practice itself, so it lands as `contested`.
        for tool in _implementation_candidates(requirement.name):
            if tool in index.by_canonical:
                hit = index.by_canonical[tool]
                method = "embedding"
                similarity = IMPLEMENTATION_CREDIT
                break

    if hit is None and embedder is not None and not taxonomy.is_in_taxonomy(requirement.name):
        # Only for vocabulary the taxonomy does not know. When the taxonomy
        # knows a skill and the resume omits it, that absence is confident and
        # a fuzzy vector score must not override it.
        if embedder.is_available():
            hit, similarity = _embedding_best(
                requirement, index, embedder._loaded, config
            )
            method = "embedding" if hit else "none"

    if hit is None:
        return MatchOutcome(
            requirement=requirement,
            status=MatchStatus.MISSING,
            score=0.0,
            weight=weight,
            method="none",
            explanation="No mention of this requirement in the resume.",
        )

    overlap = _text_overlap(requirement, hit)
    if method == "taxonomy":
        exactness = 1.0
    elif method == "synonym":
        exactness = 0.95
    else:
        exactness = similarity if similarity is not None else 0.7

    proficiency_score = hit.proficiency.value_score
    coverage = min(1.0, exactness + 0.1 * overlap)

    if requirement.min_years:
        years_score, years_note = _score_from_years(
            requirement.min_years,
            _requirement_years(hit, requirement, resume),
            config.min_years_sensitivity,
        )
    else:
        years_score, years_note = 1.0, ""

    score = round(min(1.0, coverage * proficiency_score * years_score), 4)

    if score >= 0.7:
        status = MatchStatus.MET
    elif score >= 0.3:
        status = MatchStatus.PARTIAL
    else:
        status = MatchStatus.CONTESTED

    if similarity is not None and similarity < EMBEDDING_MATCH_THRESHOLD:
        status = MatchStatus.CONTESTED
        score = round(min(score, similarity), 4)

    parts = [
        f"matched via {method} as '{hit.name}' ({hit.proficiency.value})",
    ]
    if requirement.min_years:
        parts.append(years_note)
    if overlap > 0 and method == "embedding":
        parts.append(f"wording overlap {overlap:.0%}")

    return MatchOutcome(
        requirement=requirement,
        status=status,
        score=score,
        weight=weight,
        matched_as=hit.name,
        matched_proficiency=hit.proficiency,
        resume_evidence=hit.evidence,
        method=method,
        similarity=similarity,
        explanation="; ".join(parts),
    )


def _find_by_name(skills: list[ResumeSkill], name: str) -> ResumeSkill | None:
    target = taxonomy._norm_key(name)
    for skill in skills:
        if taxonomy._norm_key(skill.name) == target:
            return skill
    return None


def _pool_vectors(
    index: ResumeIndex, embedder: object
) -> tuple[list[ResumeSkill], list[list[float]]] | None:
    """Embed the resume's skill pool once per report.

    The pool is the same for every requirement, so encoding it inside the
    per-requirement loop is what turns a sub-second run into a multi-second
    one. Cached on the index, which is built fresh per `analyze` call.
    """
    cached = getattr(index, "_vectors", None)
    if cached is not None and cached[0] is embedder:
        return cached[1], cached[2]

    from jobmatch import embed

    pool = [s for s in index.all_skills if s.name and not s.name.startswith("[")]
    if not pool:
        return None
    vectors = embed.embed_texts([_skill_text(s) for s in pool], embedder)
    index._vectors = (embedder, pool, vectors)  # type: ignore[attr-defined]
    return pool, vectors


def _embedding_best(
    requirement: Requirement,
    index: ResumeIndex,
    embedder: object,
    config: MatchConfig,
) -> tuple[ResumeSkill | None, float | None]:
    """Nearest resume skill by embedding similarity.

    Only reached for requirements the taxonomy does not recognise. When the
    taxonomy knows a skill and the resume does not mention it, that is a
    confident absence, and letting a fuzzy vector score override it produces
    false matches: on the eval fixture, "dbt" scored 0.647 against a resume
    containing only SQL, while the genuine "Communication" match scored 0.583.
    The two distributions overlap, so no threshold separates them, and the
    embedding arm is a net loss on known skills. Restricting it to unknown
    vocabulary keeps the precision it does add without the false positives.

    Any failure returns no match: an enhancement must never break a report.
    """
    try:
        prepared = _pool_vectors(index, embedder)
        if prepared is None:
            return None, None
        pool, pool_vectors = prepared
        from jobmatch import embed

        req_vec = embed.embed_texts([_requirement_text(requirement)], embedder)[0]
        scores = embed.cosine_matrix([req_vec], pool_vectors)[0]
    except Exception:  # noqa: BLE001
        return None, None

    best_index = max(range(len(scores)), key=lambda i: scores[i])
    best_score = float(scores[best_index])
    if best_score < EMBEDDING_MATCH_THRESHOLD:
        return None, round(best_score, 4)
    return pool[best_index], round(best_score, 4)


def _requirement_text(requirement: Requirement) -> str:
    if requirement.evidence:
        return f"{requirement.name}. {requirement.evidence.text}"
    return requirement.name


def _skill_text(skill: ResumeSkill) -> str:
    if skill.evidence:
        return f"{skill.name}. {skill.evidence.text}"
    return skill.name


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return 0.0
    return sum(v * w for v, w in values) / total_weight


def _overall_score(outcomes: list[MatchOutcome]) -> float:
    if not outcomes:
        return 0.0
    scored = [(o.score, o.weight) for o in outcomes]
    base = _weighted_mean(scored) * 100.0
    return round(max(0.0, min(100.0, base)), 1)


def _must_have_coverage(outcomes: list[MatchOutcome]) -> float:
    musts = [o for o in outcomes if o.requirement.importance is Importance.MUST]
    if not musts:
        return 1.0
    met = sum(1 for o in musts if o.status is MatchStatus.MET)
    partial = sum(0.5 for o in musts if o.status is MatchStatus.PARTIAL)
    return round(min(1.0, (met + partial) / len(musts)), 4)


def _verdict(
    score: float, must_coverage: float, outcomes: list[MatchOutcome], gate: bool
) -> str:
    if gate and config_gate(must_coverage, outcomes):
        return "weak"
    for label, (threshold, coverage_floor) in VERDICT_THRESHOLDS.items():
        # The coverage floor is part of the knockout rule, so it only applies
        # when that rule is switched on. Otherwise disabling the gate would
        # still cap the verdict through the back door.
        if score >= threshold and (must_coverage >= coverage_floor or not gate):
            return label
    return "weak"


def config_gate(must_coverage: float, outcomes: list[MatchOutcome]) -> bool:
    """True when a knockout requirement is absent or only weakly evidenced."""
    return any(
        o.requirement.importance is Importance.MUST
        and o.status in (MatchStatus.MISSING, MatchStatus.CONTESTED)
        for o in outcomes
    )


BLOCKER_STATUSES = (MatchStatus.MISSING, MatchStatus.CONTESTED)


def _extraction_looks_thin(jd: JDProfile) -> str | None:
    """Warn when too few requirements came out of too much text.

    Returns the warning, or None when the parse looks plausible. Two signals,
    because either alone misfires: a very long document that yielded almost
    nothing usually means the line structure was lost, and a single requirement
    whose name is nearly the whole document means one clause swallowed
    everything.
    """
    if not jd.requirements:
        return (
            "No requirements could be read from the job description, so this "
            "verdict is meaningless. Check that the posting was pasted with its "
            "line breaks intact."
        )

    longest_name = max((len(r.name) for r in jd.requirements), default=0)

    # One requirement, and that requirement is most of the document.
    if len(jd.requirements) == 1 and longest_name >= 80:
        return (
            f"Only one requirement was read, and its name is {longest_name} "
            "characters long, which means the job description lost its line "
            "breaks and the whole posting collapsed into a single clause. "
            "Re-paste it with the original formatting; this score is not "
            "trustworthy."
        )

    return None


def _suggestions(
    outcomes: list[MatchOutcome], jd: JDProfile, resume: ResumeProfile
) -> list[str]:
    """Actionable, ranked next steps. Each one names the specific gap."""
    suggestions: list[str] = []

    # A verdict computed from one or two requirements is not evidence of a
    # strong match, it is evidence of a failed parse. Pasting a posting whose
    # line breaks were lost produces exactly this: the whole document collapses
    # into one clause, one requirement gets extracted, and it matches, and the
    # weighted mean of a single 1.0 is a confident 100/100. Say so plainly
    # instead of reporting a number the input cannot support.
    thin = _extraction_looks_thin(jd)
    if thin is not None:
        suggestions.insert(0, thin)

    blockers = [o for o in outcomes if o.requirement.importance is Importance.MUST and o.status in BLOCKER_STATUSES]
    for outcome in sorted(blockers, key=lambda o: o.weight, reverse=True):
        suggestions.append(
            f"Blocker: '{outcome.requirement.name}' is required but not evidenced. "
            f"Add a concrete line showing it, or drop it from the resume if you have no example."
        )

    partials = [o for o in outcomes if o.status is MatchStatus.PARTIAL and o.requirement.importance is Importance.MUST]
    for outcome in partials[:3]:
        suggestions.append(
            f"Strengthen '{outcome.requirement.name}': currently {outcome.matched_proficiency.value if outcome.matched_proficiency else 'weak'}. "
            f"One quantified result would move it."
        )

    contested = [o for o in outcomes if o.status is MatchStatus.CONTESTED and o.requirement.importance is not Importance.MUST]
    for outcome in contested[:3]:
        suggestions.append(
            f"Check '{outcome.requirement.name}': the closest resume evidence scored "
            f"{outcome.score:.2f}. Confirm whether you actually have this experience."
        )

    if resume.degraded:
        suggestions.append(
            "Extraction ran in degraded (rule-based) mode, so recall on unusual phrasing is lower. "
            "Re-run with the model backend for a fuller read."
        )

    if not resume.total_years_experience and jd.requirements:
        suggestions.append(
            "No parseable employment dates were found. Add MM/YYYY ranges so total experience is computable."
        )

    missing_preferred = [
        o for o in outcomes
        if o.status is MatchStatus.MISSING and o.requirement.importance is Importance.PREFERRED
    ]
    for outcome in missing_preferred[:4]:
        suggestions.append(
            f"Preferred gap: '{outcome.requirement.name}'. Worth one mention if you have any exposure."
        )

    return suggestions[:12]


def _boost_with_context(outcomes: list[MatchOutcome], resume: ResumeProfile) -> None:
    """Recover requirements the resume mentions but extraction did not list.

    Extraction is a lossy step: a skill can appear inside a sentence the
    extractor summarised without becoming its own entry. A requirement that
    is nevertheless visible in the resume text is a partial match, not a gap,
    and reporting it as a gap would push a candidate to add something they
    have already written down.
    """
    evidence_text = " ".join(
        e.text.lower() for s in resume.skills if s.evidence for e in [s.evidence]
    )
    padded = f" {_norm_words(evidence_text)} "

    for outcome in outcomes:
        if outcome.status not in (MatchStatus.MISSING, MatchStatus.CONTESTED):
            continue
        if outcome.requirement.importance is Importance.MUST:
            continue

        name = taxonomy._norm_key(outcome.requirement.name)
        if len(name) < 3:
            continue

        # Word-boundary matching, so "sap" does not fire inside "sapphire".
        # Presence is tested in the evidence text, never the reverse:
        # `name in form` is trivially true for the canonical form of every
        # skill and would promote every gap to a partial match.
        forms = {
            taxonomy._norm_key(f) for f in taxonomy.surface_forms(outcome.requirement.name)
        }
        forms.add(name)
        present = any(
            len(form) >= 3 and f" {form} " in padded for form in forms
        )
        if not present:
            continue

        outcome.status = MatchStatus.PARTIAL
        outcome.score = max(outcome.score, 0.45)
        outcome.method = "taxonomy"
        outcome.explanation += "; corroborated by resume text outside the extracted skill list"


def _norm_words(text: str) -> str:
    """Normalize a span for word-boundary substring search."""
    return " ".join(taxonomy._norm_key(text).split())


class _LazyEmbedder:
    """Defers loading the embedding model until a lookup actually needs it.

    The fallback is consulted for requirements the taxonomy does not recognise.
    On a typical resume most requirements are recognised, so eagerly loading
    the model cost about twenty seconds to never use it. `is_available` keeps
    the "should we even try" decision in the matcher, where the requirement
    text is, and loading is left to `embed`.
    """

    def __init__(self, *, enabled: bool, provided: object | None) -> None:
        self._enabled = enabled
        self._provided = provided
        self._loaded: object | None = provided
        self.attempted = False

    def is_available(self) -> bool:
        if not self._enabled or self.attempted:
            return self._loaded is not None
        self.attempted = True
        if self._provided is not None:
            return True
        try:
            from jobmatch import embed

            self._loaded = embed.load_embedder()
        except Exception:  # noqa: BLE001 - an enhancement, never fatal
            self._loaded = None
        return self._loaded is not None

    def __bool__(self) -> bool:
        return self._loaded is not None


def analyze(
    jd: JDProfile,
    resume: ResumeProfile,
    config: MatchConfig | None = None,
) -> MatchReport:
    """Compare an extracted JD against an extracted resume."""
    config = config or MatchConfig()
    index = ResumeIndex.build(resume)

    # A lazy handle instead of a loaded model. Most requirements resolve from
    # the taxonomy, and loading a 400 MB embedder to then never use it added
    # ~20s to a run that finished in milliseconds. The model is fetched on
    # first use, and only if some requirement is actually unrecognised.
    lazy = _LazyEmbedder(enabled=config.use_embeddings, provided=config.embedder)

    outcomes = [
        _match_one(requirement, index, resume, config, lazy)
        for requirement in jd.requirements
    ]
    _boost_with_context(outcomes, resume)

    must_coverage = _must_have_coverage(outcomes)
    score = _overall_score(outcomes)
    verdict = _verdict(score, must_coverage, outcomes, config.must_have_gate)

    blockers = [
        o for o in outcomes
        if o.requirement.importance is Importance.MUST and o.status in BLOCKER_STATUSES
    ] if config.must_have_gate else []
    strengths = sorted(
        (o for o in outcomes if o.status is MatchStatus.MET),
        key=lambda o: (o.weight, o.score),
        reverse=True,
    )

    return MatchReport(
        overall_score=score,
        verdict=verdict,
        must_have_coverage=must_coverage,
        outcomes=sorted(outcomes, key=lambda o: (o.weight, o.score), reverse=True),
        blockers=blockers,
        strengths=strengths[:8],
        suggestions=_suggestions(outcomes, jd, resume),
        jd=jd,
        resume=resume,
        scoring_config=config.as_dict(),
    )
