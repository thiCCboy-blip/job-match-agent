"""End-to-end pipeline: ingest, extract, match, report.

`analyze_files` and `analyze_text` are the library entrypoints. The CLI in
`jobmatch.cli` is a thin wrapper over them, so anything the command line can do
is reachable from Python and vice versa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jobmatch import extract, ingest, matcher, report as report_mod, taxonomy
from jobmatch.schemas import JDProfile, MatchReport, ResumeProfile

Backend = Literal["auto", "llm", "rules"]


class PipelineError(Exception):
    """Raised when a run cannot produce a report at all."""


@dataclass
class Timing:
    ingest_ms: float
    extract_ms: float
    match_ms: float
    total_ms: float


@dataclass
class RunResult:
    report: MatchReport
    timing: Timing
    jd_document: ingest.Document
    resume_document: ingest.Document


def _time() -> float:
    return time.perf_counter()


def _elapsed_ms(start: float) -> float:
    return round((_time() - start) * 1000, 1)


def _merge_requirements(
    model: JDProfile, rules: JDProfile, document: ingest.Document
) -> JDProfile:
    """Union the model's requirements with the rules pass.

    A 1.7B model reads a three-bullet posting and returns one requirement, then
    closes the array. Constrained decoding guarantees the output is valid, not
    that it is complete, so the model cannot be the only source. The rules pass
    is exhaustive by construction, so it supplies completeness and the model
    supplies the nuance it gets right: importance, categories, and wording the
    patterns miss. Neither is trusted alone.
    """
    merged: list = []
    seen: set[str] = set()

    for requirement in [*model.requirements, *rules.requirements]:
        canonical = taxonomy.normalize_or_self(requirement.name)
        key = taxonomy._norm_key(canonical)
        if key in seen:
            continue
        seen.add(key)
        merged.append(requirement)

    notes = list(model.notes) + list(rules.notes)
    if model.requirements and len(rules.requirements) > len(model.requirements):
        notes.append(
            f"Model returned {len(model.requirements)} requirement(s); "
            f"merged with {len(rules.requirements)} from pattern rules."
        )
    return model.model_copy(update={"requirements": merged, "notes": notes})


def _extract_jd(
    document: ingest.Document, backend: Backend, extractor: extract.QwenExtractor | None
) -> JDProfile:
    rules = extract.extract_jd_with_rules(document)
    if extractor is None:
        return rules
    try:
        model = extract.extract_jd_with_llm(document, extractor)
    except Exception as exc:  # noqa: BLE001 - fall back, record why
        rules.notes.append(f"Model extraction failed ({type(exc).__name__}), used rules.")
        return rules
    return _merge_requirements(model, rules, document)


def _extract_resume(
    document: ingest.Document, backend: Backend, extractor: extract.QwenExtractor | None
) -> ResumeProfile:
    if extractor is not None:
        try:
            return extract.extract_resume_with_llm(document, extractor)
        except Exception as exc:  # noqa: BLE001
            fallback = extract.extract_resume_with_rules(document)
            fallback.notes.append(f"Model extraction failed ({type(exc).__name__}), used rules.")
            return fallback
    return extract.extract_resume_with_rules(document)


def analyze_text(
    job_description: str,
    resume_text: str,
    *,
    backend: Backend = "auto",
    config: matcher.MatchConfig | None = None,
) -> RunResult:
    """Match a pasted job description against a pasted resume."""
    return analyze_texts_with_documents(
        ingest.load_document(job_description, doc_type="text"),
        ingest.load_document(resume_text, doc_type="text"),
        backend=backend,
        config=config,
    )


def analyze_texts_with_documents(
    jd_document: ingest.Document,
    resume_document: ingest.Document,
    *,
    backend: Backend = "auto",
    config: matcher.MatchConfig | None = None,
) -> RunResult:
    """Match two already-ingested documents."""
    started = _time()

    extractor: extract.QwenExtractor | None
    if backend == "rules":
        extractor = None
    else:
        extractor = extract.build_extractor("llm" if backend == "llm" else "auto")

    extract_start = _time()
    jd_profile = _extract_jd(jd_document, backend, extractor)
    resume_profile = _extract_resume(resume_document, backend, extractor)
    extract_ms = _elapsed_ms(extract_start)

    match_start = _time()
    match_report = matcher.analyze(jd_profile, resume_profile, config)
    match_ms = _elapsed_ms(match_start)

    return RunResult(
        report=match_report,
        timing=Timing(
            ingest_ms=0.0,
            extract_ms=extract_ms,
            match_ms=match_ms,
            total_ms=_elapsed_ms(started),
        ),
        jd_document=jd_document,
        resume_document=resume_document,
    )


def analyze_files(
    jd_path: str | Path,
    resume_path: str | Path,
    *,
    backend: Backend = "auto",
    config: matcher.MatchConfig | None = None,
) -> RunResult:
    """Match a job description file against a resume file."""
    started = _time()
    try:
        jd_document = ingest.load_file(jd_path)
        resume_document = ingest.load_file(resume_path)
    except ingest.IngestError as exc:
        raise PipelineError(str(exc)) from exc

    ingest_ms = _elapsed_ms(started)
    result = analyze_texts_with_documents(
        jd_document, resume_document, backend=backend, config=config
    )
    result.timing.ingest_ms = ingest_ms
    return result


def render(result: RunResult, *, style: Literal["console", "text"] = "console", top: int = 30) -> str:
    if style == "text":
        return report_mod.render_text(result.report, top=top)
    return report_mod.render_console(result.report, top=top)
