"""Local open-weights extraction with grammar-constrained JSON decoding.

Two backends, same output contract:

- `llm`  : Qwen3-1.7B-Instruct under `lm-format-enforcer`, so the decoder
           physically cannot emit a token that breaks the JSON schema. This
           replaces retry-and-hope prompt engineering with a hard constraint.
- `rules`: a deterministic pattern extractor that needs no model and no GPU.

Both are wrapped so the caller cannot tell which one produced a profile, and
both attach real line-numbered evidence. If the model path fails for any
reason, `rules` takes over and the profile is flagged `degraded`, so a
downstream report can say the extraction was not model-derived.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jobmatch import ingest, taxonomy
from jobmatch.schemas import (
    Evidence,
    Importance,
    JDProfile,
    Proficiency,
    Requirement,
    ResumeProfile,
    ResumeSkill,
)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MAX_PROMPT_CHARS = 12000
MAX_LINES_PER_CHUNK = 90
CHUNK_OVERLAP_LINES = 2
MAX_NEW_TOKENS = 1400
RULES_MAX_ITEMS = 60

_IMPORTANCE_HINTS: tuple[tuple[str, Importance], ...] = (
    (r"\bmust\b|\brequired\b|\bessential\b|\bmandatory\b|\bminimum\b", Importance.MUST),
    (r"\bpreferred\b|\bdesired\b|\bplus\b|\bbonus\b|\bnice to have\b|\bnice-to-have\b", Importance.PREFERRED),
    (r"\bgood to have\b|\badvantage\b|\boptional\b", Importance.NICE_TO_HAVE),
)

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_YEARS_RE = re.compile(r"(\d{1,2})(?:\s*\+)?\s*(?:\+\s*)?(?:years?|yrs?)\b", re.IGNORECASE)
_YEARS_WORD_RE = re.compile(
    # "eight years", "five plus years" and "eight+ years" all state a duration.
    r"\b("
    + "|".join(_WORD_NUMBERS)
    + r")(?:\s*\+|\s+plus)?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

_PROFICIENCY_CUES: tuple[tuple[str, Proficiency], ...] = (
    (r"\bexpert\b|\bmaster\b|\bdeep\b|\badvanced\b|\b5\+?\s*years\b", Proficiency.EXPERT),
    (r"\bproficient\b|\bstrong\b|\bsolid\b|\bextensive\b|\b4\+?\s*years\b", Proficiency.PROFICIENT),
    (r"\bworking\b|\bintermediate\b|\b3\+?\s*years\b|\b2\+?\s*years\b", Proficiency.WORKING),
    (r"\bexposure\b|\bbasic\b|\bfamiliar\b|\b1\+?\s*years\b|\bworked with\b", Proficiency.EXPOSED),
)

_CATEGORY_CUES: tuple[tuple[str, str], ...] = (
    (r"\bbachelor|\bmaster|\bphd|\bb\.?tech|\bm\.?tech|\bdegree\b|\bgraduat", "education"),
    (r"\bcertif|\blicen[cs]e\b|\baccredit|\bcertification\b", "certification"),
    (r"\bcommunication\b|\bleadership\b|\bteamwork\b|\bstakeholder\b|\bownership\b|\bpresent", "soft"),
    (r"\bspeaking\b|\bfluent\b|\blanguage\b|\benglish\b|\bhindi\b|\bmalayalam\b|\btamil\b|\barabic\b", "language"),
)

_SECTION_CATEGORY = {
    "education": "education",
    "certifications": "certification",
    "experience": "experience",
    "projects": "experience",
    "skills": "skill",
    "summary": "other",
    "contact": "other",
    "projects ": "experience",
}

_SPLIT_RE = re.compile(r"[,\n;•|]|(?<=\w)\s+and\s+(?=\w)|(?<=\w)\s+/\s+(?=\w)")
_KNOWN_SECTION_WORDS = {
    "summary", "experience", "education", "skills", "certifications", "projects",
    "contact", "profile", "objective", "responsibilities", "requirements",
    "qualifications", "about", "employment", "history", "technologies",
}


# --------------------------------------------------------------------------
# JSON schemas for constrained decoding
# --------------------------------------------------------------------------

_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "line": {"type": "integer", "minimum": 1},
    },
    "required": ["text", "line"],
    "additionalProperties": False,
}

_JD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "company": {"type": ["string", "null"]},
        "seniority": {
            "type": "string",
            "enum": ["entry", "mid", "senior", "lead", "principal", "unknown"],
        },
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "skill", "experience", "education", "certification",
                            "soft", "domain", "language", "other",
                        ],
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["must", "preferred", "nice_to_have"],
                    },
                    "min_years": {"type": ["number", "null"]},
                    "evidence": _EVIDENCE_SCHEMA,
                },
                "required": ["name", "category", "importance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "seniority", "requirements"],
    "additionalProperties": False,
}

_RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "total_years_experience": {"type": ["number", "null"]},
        "highest_education": {"type": ["string", "null"]},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "skill", "experience", "education", "certification",
                            "soft", "domain", "language", "other",
                        ],
                    },
                    "proficiency": {
                        "type": "string",
                        "enum": ["exposed", "working", "proficient", "expert"],
                    },
                    "years": {"type": ["number", "null"]},
                    "evidence": _EVIDENCE_SCHEMA,
                },
                "required": ["name", "category", "proficiency"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["total_years_experience", "skills"],
    "additionalProperties": False,
}

JD_JSON_SCHEMA = _JD_SCHEMA
RESUME_JSON_SCHEMA = _RESUME_SCHEMA


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _guess_importance(text: str, default: Importance) -> Importance:
    """Read importance from the requirement's own clause only.

    Scope matters here. Postings put one requirement per bullet, so a marker
    three lines away belongs to a different requirement; reading a window
    around each hit is how "dbt is a plus" ends up classified as mandatory.
    A negative marker anywhere in the clause also wins over a positive one,
    because "not required, but familiarity with X is a plus" is a plus.
    """
    lowered = text.lower()
    signals = [
        importance
        for pattern, importance in _IMPORTANCE_HINTS
        if re.search(pattern, lowered)
    ]
    if not signals:
        return default
    for downgrade in (Importance.NICE_TO_HAVE, Importance.PREFERRED):
        if downgrade in signals:
            return downgrade
    return Importance.MUST


def _guess_proficiency(text: str) -> Proficiency:
    lowered = text.lower()
    for pattern, proficiency in _PROFICIENCY_CUES:
        if re.search(pattern, lowered):
            return proficiency
    return Proficiency.WORKING


def _is_heading(text: str) -> bool:
    """True for a section heading rather than content.

    Markdown hashes are the common case, but an all-caps or colon-terminated
    short line is treated as a heading too, since a heading must never become a
    certification or a skill.
    """
    stripped = text.strip()
    if stripped.startswith(("#", "*", "-")) and len(stripped.split()) <= 5:
        return stripped.startswith("#")
    words = stripped.split()
    if not words or len(words) > 6:
        return False
    if stripped.endswith(":"):
        return True
    return all(word.isupper() for word in words if word.isalpha())


def _guess_category(text: str, section: str | None) -> str:
    lowered = text.lower()
    for pattern, category in _CATEGORY_CUES:
        if re.search(pattern, lowered):
            return category
    if section in _SECTION_CATEGORY:
        return _SECTION_CATEGORY[section]
    return "skill"


def _find_years(text: str) -> float | None:
    """Years stated in a phrase, from digits or a spelled-out number.

    Both forms occur in resumes: a posting says "6+ years" while the matching
    bullet says "three years of daily SQL". Reading only digits silently
    discards the candidate's own statement of their experience.
    """
    match = _YEARS_RE.search(text)
    if match:
        value = int(match.group(1))
        return float(value) if 0 <= value <= 50 else None
    match = _YEARS_WORD_RE.search(text)
    if match:
        return float(_WORD_NUMBERS[match.group(1).lower()])
    return None


def _resolve_evidence(
    quote: str | None,
    line_hint: int | None,
    document: ingest.Document,
    *,
    search_window: int = 400,
) -> Evidence | None:
    """Turn a model-supplied quote into a real, line-numbered citation.

    A quote that does not appear in the document is discarded rather than
    trusted. If the quote is missing we fall back to locating the requirement
    name itself, so evidence is never fabricated.
    """
    line_map = document.line_map()
    if not line_map:
        return None

    if quote:
        needle = " ".join(quote.split()).lower()
        for number, text in line_map.items():
            haystack = text.lower()
            if needle and (needle in haystack or (len(needle) > 12 and needle[:60] in haystack)):
                return Evidence(text=text, line=number, section=_section_of(document, number))

    if line_hint:
        nearest = min(line_map, key=lambda n: abs(n - line_hint))
        return Evidence(
            text=line_map[nearest], line=nearest, section=_section_of(document, nearest)
        )

    return None


def _section_of(document: ingest.Document, number: int) -> str | None:
    for line in document.lines:
        if line.number == number:
            return line.section
    return None


def _evidence_from_line(line: ingest.Line) -> Evidence:
    """Build evidence from a source line, truncating to the schema's cap.

    Long prose lines are common in resumes, and a citation is still useful
    when it carries only the first 400 characters of the line it came from.
    """
    text = line.text[:400]
    return Evidence(text=text, line=line.number, section=line.section)


def _line_window_text(document: ingest.Document, number: int, window: int) -> str:
    """Text of the line itself, plus any non-bullet continuation lines.

    A window of surrounding bullet lines is deliberately excluded: a
    requirement's importance and years are properties of its own clause, and
    borrowing a neighbouring bullet's "required" turns optional skills into
    blockers. Continuation lines that are not new bullets belong to the same
    clause, so they are included.
    """
    lines = document.lines
    index = next((i for i, line in enumerate(lines) if line.number == number), None)
    if index is None:
        return ""

    parts = [lines[index].text]
    for offset in range(1, window + 1):
        forward = index + offset
        if forward >= len(lines) or lines[forward].is_bullet:
            break
        if lines[forward].section != lines[index].section:
            break
        parts.append(lines[forward].text)

        backward = index - offset
        if backward < 0 or lines[backward].is_bullet:
            break
        if lines[backward].section != lines[index].section:
            break
        parts.insert(0, lines[backward].text)

    return " ".join(parts)


def _chunk_lines(document: ingest.Document, size: int, overlap: int) -> list[list[ingest.Line]]:
    lines = document.lines
    if not lines:
        return []
    chunks: list[list[ingest.Line]] = []
    start = 0
    while start < len(lines):
        chunk = lines[start : start + size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(lines):
            break
        start += max(1, size - overlap)
    return chunks


# --------------------------------------------------------------------------
# deterministic fallback extractor
# --------------------------------------------------------------------------


def _candidate_fragments(
    document: ingest.Document, *, section: str | None = None
) -> list[tuple[str, ingest.Line]]:
    """Split lines into skill-like fragments paired with their source line."""
    pairs: list[tuple[str, ingest.Line]] = []
    for line in document.lines:
        if section and line.section != section:
            continue
        text = line.text
        stripped = re.sub(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s*", "", text)
        if not stripped:
            continue
        parts = [p.strip(" .;:-") for p in _SPLIT_RE.split(stripped)]
        for part in parts:
            if not (2 <= len(part) <= 70):
                continue
            if part.lower() in _KNOWN_SECTION_WORDS:
                continue
            if re.fullmatch(r"[\d\s.,%/()-]+", part):
                continue
            if EMAIL.search(part) or ingest.URL.search(part):
                continue
            pairs.append((part, line))
    return pairs


def _taxonomy_fragments(
    document: ingest.Document, *, only_sections: set[str] | None = None
) -> list[tuple[str, ingest.Line]]:
    """Find every taxonomy skill mentioned anywhere, with its source line.

    This is the highest-precision signal available: a skill is present only if
    a known surface form appears literally in the text.
    """
    found: list[tuple[str, ingest.Line]] = []
    seen: set[tuple[str, int]] = set()
    for line in document.lines:
        if only_sections is not None and line.section not in only_sections:
            continue
        padded = f" {taxonomy._norm_key(line.text)} "
        for canonical in sorted(taxonomy.SYNONYMS, key=len, reverse=True):
            for form in taxonomy.surface_forms(canonical):
                token = taxonomy._norm_key(form)
                if not token:
                    continue
                if f" {token} " in padded:
                    key = (canonical, line.number)
                    if key not in seen:
                        seen.add(key)
                        found.append((canonical, line))
                    break
    return found


# Prose that merely mentions a skill is not a requirement. A requirement is a
# short statement naming what the role needs; these are the shapes that mean
# "this line is not one of them".
_PROSE_MARKERS = re.compile(
    r"\b(because|which|that|while|where|although|however|therefore|so that|"
    r"in order to|as well as|which is|which was|it is|they are|we are)\b",
    re.IGNORECASE,
)


def _is_prose_line(line: ingest.Line) -> bool:
    """True when a line reads as a sentence rather than a requirement.

    The taxonomy matches a skill anywhere in the text, which is what makes it
    precise, but a long narrative line is not a list of requirements. Naming a
    requirement after "*Disclosure decision:** withheld the pricing inputs,
    because a realistic tender package can be mistaken for live bid material"
    produces a requirement that is a sentence, so such lines are excluded from
    requirement extraction while still counting as resume evidence.
    """
    text = _strip_leading_label(line.text)
    stripped = _clean_markdown(text).strip()
    words = stripped.split()
    if len(words) > 16:
        return True
    if _PROSE_MARKERS.search(stripped):
        return True
    # Two or more sentence terminators means prose, not a requirement.
    return stripped.count(".") >= 2


_EDUCATION_REQUIREMENT_RE = re.compile(
    r"\b(bachelor(?:'?s)?|master(?:'?s)?|mba|ph\.?d|b\.?\.?tech|m\.?\.?tech|b\.?\.?sc|m\.?\.?sc|"
    r"b\.?\.?e\b|m\.?\.?e\b|b\.?\.?com|diploma|polytechnic|degree|diplomas?)\b",
    re.IGNORECASE,
)


def _is_education_requirement(text: str) -> bool:
    """True when a clause asks for a qualification rather than a skill.

    "Bachelor's degree in Engineering or Statistics required" is a degree
    requirement that happens to contain a subject name. Emitting "Statistics"
    as a skill and dropping the degree is how a candidate ends up told to
    learn statistics when they already hold the qualification.
    """
    return bool(_EDUCATION_REQUIREMENT_RE.search(text))


def extract_jd_with_rules(document: ingest.Document) -> JDProfile:
    """Pattern-based job description extraction. No model, fully deterministic."""
    requirements: list[Requirement] = []
    seen: set[str] = set()

    # A clause that asks for a qualification is kept whole, and any taxonomy
    # skill scraped out of its subject list is suppressed.
    education_lines: set[int] = set()
    for line in document.lines:
        if _is_education_requirement(line.text):
            education_lines.add(line.number)

    taxonomy_hits = [
        (canonical, line)
        for canonical, line in _taxonomy_fragments(document)
        if not _is_prose_line(line)
    ]
    for canonical, line in taxonomy_hits:
        context = _line_window_text(document, line.number, 3)
        if _is_education_requirement(context):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        requirements.append(
            Requirement(
                name=canonical,
                category=_guess_category(context, line.section),
                importance=_guess_importance(context, Importance.PREFERRED),
                min_years=_find_years(context),
                evidence=_evidence_from_line(line),
            )
        )

    for line in document.lines:
        if line.number not in education_lines:
            continue
        phrase = _requirement_phrase(
            _clean_markdown(
                re.sub(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s*", "", line.text)
            )
        )
        if not phrase:
            continue
        key = taxonomy._norm_key(phrase)
        if key in seen:
            continue
        seen.add(key)
        requirements.append(
            Requirement(
                name=phrase,
                category="education",
                importance=_guess_importance(line.text, Importance.PREFERRED),
                evidence=_evidence_from_line(line),
            )
        )

    # Lines already represented by a taxonomy hit are covered. Re-emitting a
    # verb phrase for the same line would double-count one bullet's weight in
    # the weighted mean, so those lines are skipped here.
    covered_lines = {line.number for _, line in taxonomy_hits} | education_lines

    # Only real requirement lines. `":" in line.text` used to be enough, but
    # that admits any prose containing a colon, so a summary line like
    # "*Disclosure decision:** withheld the pricing inputs" became a
    # requirement. A bullet is a requirement line, and a short "Label: text"
    # heading is one; a long sentence with a colon inside it is prose.
    bullet_lines = [
        line
        for line in document.lines
        if line.is_bullet or _is_labeled_heading(line.text)
    ]
    for line in bullet_lines:
        if line.number in covered_lines:
            continue
        text = re.sub(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s*", "", line.text)
        if len(text) < 12:
            continue
        has_verb = re.search(
            r"\b(develop|build|manage|managing|lead|leading|design|create|maintain|deploy|analyz|analys|"
            r"implement|own|drive|support|deliver|ensure|review|collaborat|coordinate|handling|"
            r"administer|oversee|supervis|superviz|prepare|producing|run|operate|maintaining|"
            r"work with|partner|reporting on|exposure to|experience in|experience with|familiarity with|"
            r"working knowledge of|strong|expert|advanced|proficient)\w*\b",
            text,
            re.IGNORECASE,
        )
        if not has_verb:
            continue
        # A bullet can open with a bolded sub-label, e.g. a resume's
        # "- **Disclosure decision:** withheld the pricing inputs". Starting the
        # requirement at the label yields "*Disclosure decision:** withheld..."
        # as a requirement name, which is prose, not a requirement. Where the
        # verb sits after the colon, the label is not part of the requirement.
        text = _strip_leading_label(text)
        if len(text) < 12:
            continue
        phrase = _requirement_phrase(_clean_markdown(text))
        if not phrase:
            continue
        key = taxonomy._norm_key(phrase)
        if key in seen:
            continue
        seen.add(key)
        requirements.append(
            Requirement(
                name=phrase,
                category=_guess_category(text, line.section),
                importance=_guess_importance(text, Importance.PREFERRED),
                min_years=_find_years(text),
                evidence=_evidence_from_line(line),
            )
        )

    first = document.lines[0].text if document.lines else ""
    title = None
    match = re.search(r"(?:position|role|job title|title)\s*[:\-]\s*(.+)", first, re.IGNORECASE)
    if match:
        title = match.group(1).strip()[:160]
    else:
        title = _first_title_line(document)

    return JDProfile(
        title=title,
        seniority=_guess_seniority(document),
        requirements=requirements[:RULES_MAX_ITEMS],
        extractor="rules",
        degraded=True,
        notes=["Model extraction unavailable or disabled; used pattern-based extraction."],
    )


def _requirement_phrase(text: str) -> str | None:
    """Pull a noun phrase out of a responsibility bullet."""
    text = re.sub(
        r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s*", "", text
    ).strip()
    if len(text) < 12:
        return None
    clauses = re.split(r"(?<=[.!?])\s+|;\s+", text)
    clause = clauses[0].strip()
    if len(clause) < 12:
        return None
    words = clause.split()
    if len(words) > 22:
        clause = " ".join(words[:22])
    clause = clause.rstrip(" .;:-")[:120]
    # Importance and modality words are captured separately by
    # `_guess_importance`; leaving them on the label makes "MBA preferred"
    # and "MBA" two different requirements.
    return re.sub(
        r"\s+(?:preferred|required|must have|must|essential|mandatory|desirable|"
        r"nice to have|nice-to-have|optional|good to have)\s*$",
        "",
        clause,
        flags=re.IGNORECASE,
    ).rstrip(" .;:-") or clause


# "Label: text" where the label is short, e.g. "Requirements: SQL". Markdown
# emphasis is stripped first so "**Disclosure decision:** prose" is judged on
# its words rather than its bold markers. The label must not read as a
# sentence, which is what keeps a prose line containing a colon out.
_LABELLED_HEADING = re.compile(
    r"^\**\s*[A-Z][A-Za-z /&]{0,28}?\**\s*:\s*\S"
)
_LABELLED_PROSE = re.compile(
    r"^\**\s*[A-Z][A-Za-z /&]{0,28}?\**\s*:\s+\S.*\b(?:is|are|was|were|it|that|which|because)\b",
    re.IGNORECASE,
)
# A section label is a short noun phrase: "Requirements", "Nice to have",
# "Key Skills". "Disclosure decision" is three words of prose and is not one.
_SECTION_LABEL = re.compile(
    r"^\**\s*(?:"
    r"requirements?|responsibilit(?:y|ies)|qualifications?|skills?|must haves?|"
    r"preferred|nice to have|nice-to-have|desired|essential|core|technical|"
    r"key \w+|about the role|what you.ll do|benefits?"
    r")\s*\**\s*:",
    re.IGNORECASE,
)


# A bolded sub-label at the start of a bullet. Resumes write these both ways:
# "**Disclosure decision:** text" and "*Disclosure decision:** text", so the
# opening and closing markers are matched independently.
_LEADING_LABEL = re.compile(r"^\**\s*[A-Za-z][A-Za-z /&-]{0,32}?\**\s*:\s*\**\s*(?=\S)")


def _strip_leading_label(text: str) -> str:
    """Drop a bolded sub-label so the requirement starts at the real content.

    Resumes commonly write bullets as "- **Disclosure decision:** withheld the
    pricing inputs". The label is the author's annotation, not the content, and
    keeping it turns a sentence into a nonsense requirement name. Unbalanced
    emphasis markers left behind by the removal are cleaned up too, so the
    result is plain text rather than a fragment of markdown.
    """
    stripped = _LEADING_LABEL.sub("", text, count=1).strip()
    # The label removal leaves the opening "**" of the label's markup behind
    # when the closing pair was inside it.
    stripped = re.sub(r"^\*+\s*", "", stripped)
    return stripped.strip()


def _clean_markdown(text: str) -> str:
    """Remove emphasis markers left dangling by markdown-heavy source text."""
    return re.sub(r"\s*\*+", " ", text).strip()


def _is_labeled_heading(text: str) -> bool:
    stripped = re.sub(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s*", "", text).strip()
    if _SECTION_LABEL.match(stripped):
        return True
    if not _LABELLED_HEADING.match(stripped):
        return False
    if _LABELLED_PROSE.match(stripped):
        return False
    # What follows the colon must look like a requirement list, not a
    # sentence. This is the decisive test: a section puts a short,
    # skill-like enumeration after the colon, whereas "*Disclosure
    # decision:** withheld the pricing inputs, because ..." does not.
    tail = stripped.split(":", 1)[1].strip()
    if not tail:
        return False
    if len(tail.split()) > 12:
        return False
    return not re.search(r"[.!?]\s|\b(?:because|which|that|while|where)\b", tail, re.IGNORECASE)


def _guess_seniority(document: ingest.Document) -> str:
    lowered = document.text.lower()
    table = (
        ("principal", ("principal", "distinguished", "fellow")),
        ("lead", ("lead ", "leadership", "team lead", "manager", "head of")),
        ("senior", ("senior", "sr.", "staff", "expert")),
        ("entry", ("junior", "jr.", "entry", "graduate", "intern", "trainee", "fresher")),
        ("mid", ("mid", "intermediate", "2-4 years", "3-5 years")),
    )
    for level, markers in table:
        if any(marker in lowered for marker in markers):
            return level
    return "unknown"


def extract_resume_with_rules(document: ingest.Document) -> ResumeProfile:
    """Pattern-based resume extraction. No model, fully deterministic."""
    skills: list[ResumeSkill] = []
    seen: set[str] = set()

    for canonical, line in _taxonomy_fragments(document):
        key = taxonomy._norm_key(canonical)
        if key in seen:
            continue
        seen.add(key)
        context = _line_window_text(document, line.number, 2)
        skills.append(
            ResumeSkill(
                name=canonical,
                category=_guess_category(context, line.section),
                proficiency=_guess_proficiency(context),
                years=_find_years(context),
                evidence=_evidence_from_line(line),
            )
        )

    for line in document.lines:
        if line.section != "certifications":
            continue
        if _is_heading(line.text):
            continue
        for part in _SPLIT_RE.split(re.sub(r"^[^:]{0,40}:\s*", "", line.text)):
            cleaned = part.strip(" .-;")
            if 3 <= len(cleaned) <= 90 and not re.search(r"\d{4}\s*[-–]\s*$", cleaned):
                skills.append(
                    ResumeSkill(
                        name=cleaned,
                        category="certification",
                        proficiency=Proficiency.PROFICIENT,
                        evidence=_evidence_from_line(line),
                    )
                )

    total_years = _estimate_total_years(document)
    highest_education, education_evidence = _extract_education(document)
    certifications = sorted(
        {s.name for s in skills if s.category == "certification"}
    )

    return ResumeProfile(
        skills=skills[:RULES_MAX_ITEMS],
        total_years_experience=total_years,
        highest_education=highest_education,
        education_evidence=education_evidence,
        certifications=certifications,
        extractor="rules",
        degraded=True,
        notes=["Model extraction unavailable or disabled; used pattern-based extraction."],
    )


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Two date notations occur in practice: "Jun 2023 - Present" and "06/2023 to 07/2024".
_RANGE_RE = re.compile(
    r"(?:(?P<sm>[A-Za-z]{3,9})\.?\s*(?P<sy>(?:19|20)\d{2})|(?P<nm>\d{1,2})/(?P<ny>(?:19|20)\d{2}))"
    r"\s*(?:[-–—]{1,2}|to|through|until)\s*"
    r"(?:(?P<em>[A-Za-z]{3,9}|present|now|current|ongoing)\.?\s*(?P<ey>(?:19|20)\d{2})?"
    r"|(?P<nem>\d{1,2})/(?P<ney>(?:19|20)\d{2}))",
    re.IGNORECASE,
)

_CURRENT_TOKENS = {"present", "now", "current", "ongoing", "todate", "date"}


def _is_current_token(token: str) -> bool:
    """True for the "still going" markers a resume uses instead of a date."""
    key = token.strip().lower().rstrip(".")
    return (
        key in _CURRENT_TOKENS
        or key.startswith("present")
        or key.startswith("curr")
        or key.startswith("to ")
    )


def _parse_month(token: str | None) -> int | None:
    if not token:
        return None
    key = token.strip().lower()[:4].rstrip(".")
    if _is_current_token(key):
        return 12
    if key in _MONTHS:
        return _MONTHS[key]
    if key in ("sept", "sep"):
        return 9
    return None


def _valid_year(year: int | None) -> bool:
    return year is not None and 1970 <= year <= 2100


def _default_as_of() -> tuple[int, int]:
    """Today, as (year, month). An open-ended role runs to the present."""
    today = datetime.date.today()
    return today.year, today.month


def _month_ordinal(year: int, month: int) -> int:
    """Absolute month index, so ranges can be compared and differenced.

    Dates are compared as `year * 12 + month` rather than as `year * 100 +
    month`, because the latter does not subtract: 202407 - 202306 is 101, not
    the 14 months between the two dates.
    """
    return year * 12 + (month - 1)


def _estimate_total_years(
    document: ingest.Document, *, as_of: tuple[int, int] | None = None
) -> float | None:
    """Total non-overlapping employment months, in years.

    Overlapping roles are merged rather than summed, so a concurrent
    internship and a full-time role do not inflate the total. A range still
    open at the end (Present) is closed at `as_of`, which callers pass as
    today's month so the number stays reproducible across runs.
    """
    ranges: list[tuple[int, int]] = []
    as_of = as_of or _default_as_of()
    for line in document.lines:
        if line.section not in (None, "experience"):
            continue
        for match in _RANGE_RE.finditer(line.text):
            if match.group("nm"):
                start_month, start_year = int(match.group("nm")), int(match.group("ny"))
            else:
                start_month = _parse_month(match.group("sm"))
                start_year = int(match.group("sy")) if _valid_year(int(match.group("sy"))) else None
            if start_month is None or not _valid_year(start_year) or not 1 <= start_month <= 12:
                continue

            if match.group("nem"):
                end_month, end_year = int(match.group("nem")), int(match.group("ney"))
            elif match.group("em"):
                end_token = match.group("em").lower()
                if _is_current_token(end_token):
                    # "Present" with no year means the role is still running,
                    # so it ends in the current month rather than in December
                    # of the year the role started. as_of is (year, month),
                    # the reverse of the end_month, end_year order below.
                    end_year, end_month = as_of
                else:
                    end_month = _parse_month(end_token)
                    end_year = int(match.group("ey")) if match.group("ey") else start_year
            else:
                end_month, end_year = 12, start_year

            if not _valid_year(end_year):
                end_year = as_of[1]
            if not end_month or not 1 <= end_month <= 12:
                end_month = 12

            start = _month_ordinal(start_year, start_month)
            end = _month_ordinal(end_year, end_month)
            if end < start:
                continue
            ranges.append((start, end))

    if not ranges:
        return None

    ranges.sort()
    merged: list[list[int]] = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    months = sum(end - start + 1 for start, end in merged)
    years = months / 12.0
    return round(min(years, 60.0), 1) if years > 0 else None


def _extract_education(document: ingest.Document) -> tuple[str | None, Evidence | None]:
    """Highest qualification found, with the line that states it."""
    degree_re = re.compile(
        r"\b(bachelor(?:'?s)?(?:\s+of)?(?:\s+(?:science|arts|engineering|technology|commerce|business))?|"
        r"master(?:'?s)?(?:\s+of)?(?:\s+(?:science|arts|engineering|technology|commerce|business))?|"
        r"b\.?\.?tech|m\.?\.?tech|b\.?\.?e\b|m\.?\.?e\b|ph\.?\.?d|diploma|polytechnic|"
        r"b\.?\.?sc\b|m\.?\.?sc\b|b\.?\.?com\b|b\.?\.?ba\b|mba)\b",
        re.IGNORECASE,
    )
    for line in document.lines:
        if line.section == "education" and degree_re.search(line.text):
            return line.text[:160], _evidence_from_line(line)
    for line in document.lines:
        if line.section is not None:
            continue
        if degree_re.search(line.text):
            return line.text[:160], _evidence_from_line(line)
    return None, None


# --------------------------------------------------------------------------
# model-backed extractor
# --------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    payload: dict[str, Any]
    raw: str
    chunks_used: int
    token_count: int


class QwenExtractor:
    """Qwen3-Instruct under grammar-constrained decoding.

    Loaded lazily so that importing the package, running the rule-based path,
    and running the test suite never pay for a model download.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._parser_cache: dict[str, Any] = {}

    # -- loading ---------------------------------------------------------
    def _torch(self):
        import torch

        return torch

    def load(self) -> None:
        if self._model is not None:
            return
        torch = self._torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "auto": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch_dtype
        ).to(self.device)
        self._model.eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- decoding --------------------------------------------------------
    def _parser(self, schema: dict[str, Any], key: str):
        if key not in self._parser_cache:
            from lmformatenforcer import JsonSchemaParser

            self._parser_cache[key] = JsonSchemaParser(schema)
        return self._parser_cache[key]

    def _build_prefix_fn(self, tokenizer, schema: dict[str, Any], key: str):
        """Build the token-mask function for grammar-constrained decoding.

        `lmformatenforcer.integrations.transformers` cannot be imported under
        transformers 5.x: its module-level guard still imports
        `PreTrainedTokenizerBase` from `transformers.tokenization_utils`, which
        moved, so it raises ImportError even with transformers installed. That
        leaves the public `build_transformers_prefix_allowed_tokens_fn`
        unreachable too, since it lives in the same module.

        The remaining pieces are still importable, so the small amount of glue
        that module provides is rebuilt here against the public `TokenEnforcer`
        API rather than depending on a version that does not work.
        """
        try:
            from lmformatenforcer import build_transformers_prefix_allowed_tokens_fn

            return build_transformers_prefix_allowed_tokens_fn(
                tokenizer, self._parser(schema, key)
            )
        except ImportError:
            pass

        import functools

        from lmformatenforcer import TokenEnforcer, TokenEnforcerTokenizerData

        # Token strings are derived by decoding each id, plus by decoding it
        # after token 0 to learn whether it starts a word (that leading space
        # matters when a token can be glued onto the previous one).
        token_0 = 0
        regular_tokens = []
        for token_idx in range(len(tokenizer)):
            if token_idx in tokenizer.all_special_ids:
                continue
            decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
            decoded_regular = tokenizer.decode([token_idx])
            regular_tokens.append(
                (token_idx, decoded_after_0, len(decoded_after_0) > len(decoded_regular))
            )

        def decode_fn(tokens):
            return tokenizer.decode(tokens).rstrip("\ufffd")

        tokenizer_data = TokenEnforcerTokenizerData(
            regular_tokens,
            functools.partial(decode_fn),
            tokenizer.eos_token_id,
            False,
            len(tokenizer),
        )
        token_enforcer = TokenEnforcer(tokenizer_data, self._parser(schema, key))

        def prefix_allowed_tokens_fn(batch_id, sent):
            return token_enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

        return prefix_allowed_tokens_fn

    def _generate(self, system: str, user: str, schema: dict[str, Any], key: str) -> str:
        self.load()
        torch = self._torch()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        prefix_allowed_tokens = self._build_prefix_fn(self._tokenizer, schema, key)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        """Recover the JSON object from constrained output.

        Grammar-constrained decoding interleaves whitespace and newline tokens
        between every character, and a nested `}` can appear before the object
        actually ends. Taking the text between the first `{` and the *matching*
        brace, rather than the last one, keeps a trailing fragment from
        truncating a valid result.
        """
        start = raw.find("{")
        if start == -1:
            raise ValueError("model output contained no JSON object")
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(raw)):
            char = raw[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start : position + 1])
        raise ValueError("model output had an unterminated JSON object")


JD_SYSTEM_PROMPT = (
    "You extract structured hiring requirements from job descriptions. "
    "Rules: one entry per distinct requirement, not one per repeated mention. "
    "A requirement is must if the posting marks it required/essential/mandatory, "
    "preferred if it says preferred/desired/plus, otherwise nice_to_have. "
    "min_years is a number only when the posting states a duration, else null. "
    "category is one of skill, experience, education, certification, soft, domain, "
    "language, other. evidence.text must be copied verbatim from the input and "
    "evidence.line must be that line's 1-indexed number. Never invent requirements."
)

RESUME_SYSTEM_PROMPT = (
    "You extract candidate capabilities from resumes. "
    "Rules: one entry per distinct skill, using the strongest evidence available. "
    "Only include what the resume states or clearly demonstrates. Do not infer "
    "skills that are absent. proficiency is exposed for a mention, working for "
    "applied use, proficient for strong independent use, expert for deep or "
    "lead-level use. years is a number only when the resume states a duration. "
    "category is one of skill, experience, education, certification, soft, domain, "
    "language, other. evidence.text must be copied verbatim and evidence.line must "
    "be that line's 1-indexed number."
)


def _number_lines(chunk: list[ingest.Line]) -> str:
    return "\n".join(f"{line.number}: {line.text}" for line in chunk)


# A 1.7B model reliably closes the array after the first entry unless the
# instruction to work through the whole input is in the user turn next to the
# text. Verified on Qwen3-1.7B: with this line it extracts every bullet, without
# it the first one and nothing else.
_EXHAUST_INPUT = (
    "\n\nWork through the input from top to bottom. Every requirement, "
    "skill, or qualification line gets its own array entry. Do not stop "
    "after the first entry."
)


def _user_prompt_text(chunk: list[ingest.Line], what: str) -> str:
    return f"{what} with line numbers:\n\n{_number_lines(chunk)}{_EXHAUST_INPUT}"


# Words that describe how a skill is used rather than naming the skill. The
# model writes "K8s administration" where the taxonomy knows "Kubernetes".
_USE_SUFFIXES = re.compile(
    r"\b(usage|use|using|application|applications|development|developing|management|"
    r"managing|administration|administering|administration|operations|operation|"
    r"pipelines?|services?|experience|experienced|expertise|proficiency|"
    r"knowledge|skills?|tools?|technologies|tech|stack|work|working|workflows?|"
    r"practices?|processes|projects?|solutions?|systems?|platforms?)\b",
    re.IGNORECASE,
)


# Adjectives and duration lead-ins that qualify a skill without naming a
# different one: "Advanced SQL" is SQL, "4+ years building dashboards in
# Power BI" is Power BI.
_QUALIFIER_WORDS = re.compile(
    r"\b(advanced|intermediate|beginner|basic|expert|expertise|expertise|strong|"
    r"solid|proficient|working|extensive|proven|expert-level|fluent|deep|"
    r"hands-on|handson|applied|relevant|good|excellent|strong|expert)\b",
    re.IGNORECASE,
)

# A leading duration clause such as "4+ years building dashboards in".
# The gap is bounded and non-greedy tokens are avoided deliberately: a
# character-class repetition followed by an optional tail backtracks
# catastrophically on these strings.
_DURATION_CLAUSE = re.compile(
    r"^\s*\d+\s*\+?\s*(?:years?|yrs?)\b(?:\s+\w+){0,4}?\s+(?:in|of|with|using)\s+",
    re.IGNORECASE,
)


def _canonical_model_name(name: str) -> str:
    """Reduce a model-written requirement name to a taxonomy canonical name.

    The model names things the way a posting reads, e.g. "K8s administration",
    "Advanced SQL", or "4+ years building dashboards in Power BI". Those are
    all real requirements, but they never equal a canonical skill name, so gap
    detection compares them as unknowns and reports skills the candidate has as
    missing. Reducing the phrase to the skill itself gives the taxonomy and the
    matcher something they can resolve. Returns the original name when nothing
    canonical is found, so an unusual requirement keeps its wording and its
    evidence.
    """
    direct = taxonomy.normalize_skill(name)
    if direct:
        return direct

    stripped = _USE_SUFFIXES.sub(" ", name)
    stripped = _QUALIFIER_WORDS.sub(" ", stripped)
    # A leading "4+ years of X in Y" leaves the Y as the skill.
    trailing = _DURATION_CLAUSE.sub(" ", stripped)

    # Split on the separators, not with `\b(?:and|or|/|,)\b`: `/` is not a word
    # character, so the boundary never matches it and the pattern backtracks
    # catastrophically on names containing a slash such as "CI/CD pipelines".
    variants: list[str] = []
    for variant in (stripped, trailing):
        cleaned = re.sub(r"[^A-Za-z0-9+#.\-/ ]+", " ", variant)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        if not cleaned:
            continue
        variants.append(cleaned)
        # In a long phrase the skill usually comes last ("... in Power BI").
        for part in re.split(r"\s+(?:and|or)\s+|/|,", cleaned, flags=re.IGNORECASE):
            part = part.strip()
            if part:
                variants.append(part)

    # Try exact matches on the whole phrase before any fragment of it, so
    # "Node.js services" is not reduced to something unrelated first.
    seen: set[str] = set()
    ordered = list(variants)
    ordered.sort(key=lambda v: (len(v.split()) == 1, -len(v)))
    for candidate in ordered:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        canonical = taxonomy.normalize_skill(candidate)
        if canonical:
            return canonical
    return name


def _verified_min_years(value: object, evidence: Evidence | None) -> float | None:
    """Derive a requirement's minimum years from its own quoted evidence.

    The quoted line is the only authority. Qwen3-1.7B has been observed
    copying the evidence line's own line number into `min_years`, e.g. reading
    "Bachelor degree in Engineering required" on line 4 as 4.0 years, so the
    model's number is checked against the text and replaced rather than
    trusted. A duration that is not in the evidence is a hallucination and is
    dropped; a stated duration wins even when the model reported a different
    one, and a line like "eight plus years" is still read correctly.
    """
    if evidence is None or not evidence.text:
        return None
    return _find_years(evidence.text)


def extract_jd_with_llm(
    document: ingest.Document, extractor: QwenExtractor
) -> JDProfile:
    """Model extraction, merged across chunks, then repaired against the source."""
    chunks = _chunk_lines(document, MAX_LINES_PER_CHUNK, CHUNK_OVERLAP_LINES)
    merged: list[Requirement] = []
    seen: set[str] = set()
    degraded = False
    model_title: str | None = None
    model_company: str | None = None

    for chunk in chunks:
        try:
            raw = extractor._generate(
                JD_SYSTEM_PROMPT,
                _user_prompt_text(chunk, "Job description"),
                _JD_SCHEMA,
                "jd",
            )
            payload = extractor._parse(raw)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the pipeline
            degraded = True
            merged.append(
                Requirement(
                    name=f"[extraction failed on lines {chunk[0].number}-{chunk[-1].number}: {type(exc).__name__}]",
                    category="other",
                    importance=Importance.NICE_TO_HAVE,
                )
            )
            continue
        if model_title is None:
            candidate = payload.get("title")
            model_title = str(candidate).strip()[:160] if isinstance(candidate, str) and candidate.strip() else None
        if model_company is None:
            candidate = payload.get("company")
            model_company = str(candidate).strip()[:160] if isinstance(candidate, str) and candidate.strip() else None
        for item in _as_items(payload.get("requirements")):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = taxonomy.normalize_or_self(name).lower()
            if key in seen:
                continue
            seen.add(key)
            line_hint = (item.get("evidence") or {}).get("line")
            evidence = _resolve_evidence(
                (item.get("evidence") or {}).get("text"),
                line_hint if isinstance(line_hint, int) else None,
                document,
            )
            merged.append(
                Requirement(
                    name=_canonical_model_name(name)[:120],
                    category=item.get("category", "skill"),
                    importance=item.get("importance", "preferred"),
                    min_years=_verified_min_years(item.get("min_years"), evidence),
                    evidence=evidence,
                )
            )

    if not merged or all(r.name.startswith("[extraction failed") for r in merged):
        raise ExtractionError("model produced no usable job description structure")

    if model_title is None:
        model_title = _first_title_line(document)

    return JDProfile(
        title=model_title,
        company=model_company,
        seniority=_guess_seniority(document),
        requirements=merged[:RULES_MAX_ITEMS],
        extractor=f"qwen:{extractor.model_id}",
        degraded=degraded,
        notes=(["Some chunks failed constrained decoding."] if degraded else []),
    )


def _first_title_line(document: ingest.Document) -> str | None:
    for line in document.lines[:3]:
        if not ingest.EMAIL.search(line.text) and not ingest.URL.search(line.text) and len(line.text) <= 80:
            return line.text[:160]
    return None


def extract_resume_with_llm(
    document: ingest.Document, extractor: QwenExtractor
) -> ResumeProfile:
    """Model extraction over the resume, then a taxonomy cross-check pass.

    The cross-check matters: a model can miss a skill that is literally
    present in the text. Anything the taxonomy finds and the model missed is
    added, because a false negative in extraction is a false gap in the report.
    """
    chunks = _chunk_lines(document, MAX_LINES_PER_CHUNK, CHUNK_OVERLAP_LINES)
    merged: list[ResumeSkill] = []
    seen: set[str] = set()
    degraded = False

    for chunk in chunks:
        try:
            raw = extractor._generate(
                RESUME_SYSTEM_PROMPT,
                _user_prompt_text(chunk, "Resume"),
                _RESUME_SCHEMA,
                "resume",
            )
            payload = extractor._parse(raw)
        except Exception as exc:  # noqa: BLE001
            degraded = True
            merged.append(
                ResumeSkill(
                    name=f"[extraction failed: {type(exc).__name__}]",
                    category="other",
                )
            )
            continue
        for item in _as_items(payload.get("skills")):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = taxonomy.normalize_or_self(name).lower()
            if key in seen:
                continue
            seen.add(key)
            line_hint = (item.get("evidence") or {}).get("line")
            merged.append(
                ResumeSkill(
                    name=name[:120],
                    category=item.get("category", "skill"),
                    proficiency=item.get("proficiency", "working"),
                    years=item.get("years"),
                    evidence=_resolve_evidence(
                        (item.get("evidence") or {}).get("text"),
                        line_hint if isinstance(line_hint, int) else None,
                        document,
                    ),
                )
            )

    usable = [s for s in merged if not s.name.startswith("[extraction failed")]
    if not usable:
        raise ExtractionError("model produced no usable resume structure")

    for canonical, line in _taxonomy_fragments(document):
        key = taxonomy.normalize_or_self(canonical).lower()
        if key in seen:
            continue
        seen.add(key)
        context = _line_window_text(document, line.number, 2)
        usable.append(
            ResumeSkill(
                name=canonical,
                category=_guess_category(context, line.section),
                proficiency=_guess_proficiency(context),
                years=_find_years(context),
                evidence=_evidence_from_line(line),
            )
        )

    certifications = sorted(
        {s.name for s in usable if s.category == "certification"}
    )
    return ResumeProfile(
        skills=usable[:RULES_MAX_ITEMS],
        total_years_experience=_estimate_total_years(document),
        highest_education=_extract_education(document),
        certifications=certifications,
        extractor=f"qwen:{extractor.model_id}",
        degraded=degraded,
        notes=(["Some chunks failed constrained decoding."] if degraded else []),
    )


def _as_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _first(value: Any, default: Any) -> Any:
    return value if value else default


class ExtractionError(Exception):
    """Raised when model extraction yields nothing usable."""


def _cuda_available() -> bool:
    if os.environ.get("JOBMATCH_FORCE_CPU") == "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


Backend = Literal["llm", "rules", "auto"]


def build_extractor(backend: Backend) -> QwenExtractor | None:
    """Construct the model extractor, or None when unavailable."""
    if backend == "rules":
        return None
    try:
        return QwenExtractor(
            device=os.environ.get("JOBMATCH_DEVICE") or None,
        )
    except Exception:  # noqa: BLE001
        return None


def model_cache_dir() -> Path:
    return Path(
        os.environ.get("HF_HOME")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
