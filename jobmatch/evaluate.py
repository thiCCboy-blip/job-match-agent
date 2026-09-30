"""Evaluation harness over a labelled fixture.

Reports four numbers, because a single accuracy figure hides the failure mode
that matters here:

- requirement precision / recall / F1 over the labelled matched set
- verdict agreement against the human label
- per-requirement evidence accuracy: does the cited line actually support it
- gap accuracy: were the labelled gaps actually reported as gaps

    python -m jobmatch.evaluate data/eval_cases.json --backend rules
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from jobmatch import extract, ingest, matcher, pipeline, taxonomy
from jobmatch.matcher import MatchConfig
from jobmatch.schemas import MatchStatus

VERDICT_ORDER = {"weak": 0, "stretch": 1, "plausible": 2, "strong": 3}


@dataclass
class CaseSpec:
    id: str
    note: str
    jd_text: str
    resume_text: str
    expected_verdict: str
    expected_matched: list[str]
    expected_missing: list[str]


@dataclass
class CaseResult:
    spec: CaseSpec
    predicted_verdict: str
    overall_score: float
    matched: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    evidence_ok: bool = True
    bad_evidence: list[str] = field(default_factory=list)
    unmatched_expected: list[str] = field(default_factory=list)
    unexpected_matched: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None


def load_cases(path: str | Path) -> list[CaseSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    specs: list[CaseSpec] = []
    for case in cases:
        specs.append(
            CaseSpec(
                id=case["id"],
                note=case.get("note", ""),
                jd_text=case["jd_text"],
                resume_text=case["resume_text"],
                expected_verdict=case["expected_verdict"],
                expected_matched=[str(s) for s in case.get("expected_matched", [])],
                expected_missing=[str(s) for s in case.get("expected_missing", [])],
            )
        )
    if not specs:
        raise ValueError("fixture contains no cases")
    return specs


def _case_result(spec: CaseSpec, backend: str, config: MatchConfig) -> CaseResult:
    started = time.perf_counter()
    try:
        result = pipeline.analyze_text(spec.jd_text, spec.resume_text, backend=backend, config=config)
    except Exception as exc:  # noqa: BLE001 - a failing case is a data point
        return CaseResult(
            spec=spec,
            predicted_verdict="error",
            overall_score=0.0,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    report = result.report
    resume_document = ingest.load_document(spec.resume_text, doc_type="text")
    line_map = resume_document.line_map()

    matched: set[str] = set()
    missing: set[str] = set()
    bad_evidence: list[str] = []

    for outcome in report.outcomes:
        name = outcome.requirement.name
        if outcome.status is MatchStatus.MISSING:
            missing.add(name)
            continue
        matched.add(name)
        if outcome.resume_evidence is not None:
            line = outcome.resume_evidence.line
            if line not in line_map or outcome.resume_evidence.text not in line_map[line]:
                bad_evidence.append(f"{name} -> L{line}")

    return CaseResult(
        spec=spec,
        predicted_verdict=report.verdict,
        overall_score=report.overall_score,
        matched=matched,
        missing=missing,
        evidence_ok=not bad_evidence,
        bad_evidence=bad_evidence,
        unmatched_expected=[
            m for m in spec.expected_matched if not any(_labels_match(m, g) for g in matched)
        ],
        unexpected_matched=sorted(
            g
            for g in matched
            if not any(_labels_match(g, m) for m in spec.expected_matched)
            and not any(_labels_match(g, m) for m in spec.expected_missing)
        ),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _label_key(name: str) -> str:
    """Compare labels by meaning, not by the exact string the backend chose.

    The rules backend says "Kubernetes" where the model says "K8s
    administration", and both name the same skill. Scoring those as different
    would measure wording rather than correctness, and would make the two
    backends incomparable. `extract._canonical_model_name` applies the same
    reduction the pipeline uses, so a label is compared after the reduction
    both paths share. Falls back to the normalized text when nothing canonical
    is found, so an unusual requirement still matches itself.
    """
    canonical = taxonomy.normalize_skill(extract._canonical_model_name(name))
    if canonical:
        return canonical.casefold()
    return taxonomy._norm_key(name)


def _labels_match(left: str, right: str) -> bool:
    return _label_key(left) == _label_key(right)


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if r.error is None]
    tp = fp = fn = 0
    gap_tp = gap_fn = 0
    for result in scored:
        for name in result.spec.expected_matched:
            if any(_labels_match(name, g) for g in result.matched):
                tp += 1
            else:
                fn += 1
        for name in result.unexpected_matched:
            fp += 1
        for name in result.spec.expected_missing:
            if any(_labels_match(name, g) for g in result.missing):
                gap_tp += 1
            else:
                gap_fn += 1

    verdict_exact = sum(1 for r in scored if r.predicted_verdict == r.spec.expected_verdict)
    verdict_adjacent = sum(
        1
        for r in scored
        if abs(VERDICT_ORDER.get(r.predicted_verdict, -1) - VERDICT_ORDER.get(r.spec.expected_verdict, -1)) <= 1
    )
    evidence_ok = sum(1 for r in scored if r.evidence_ok)

    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "requirement_prf": _prf(tp, fp, fn),
        "verdict": {
            "exact": round(verdict_exact / len(scored), 4) if scored else 0.0,
            "within_one_grade": round(verdict_adjacent / len(scored), 4) if scored else 0.0,
        },
        "gap_detection": _prf(gap_tp, 0, gap_fn),
        "evidence_accuracy": round(evidence_ok / len(scored), 4) if scored else 0.0,
        "mean_score": round(sum(r.overall_score for r in scored) / len(scored), 2) if scored else 0.0,
        "total_ms": round(sum(r.elapsed_ms for r in results), 1),
    }


def run(cases: Iterable[CaseSpec], *, backend: str = "rules", use_embeddings: bool = False) -> list[CaseResult]:
    config = MatchConfig(use_embeddings=use_embeddings)
    return [_case_result(spec, backend, config) for spec in cases]


def format_results(results: list[CaseResult], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"{'case':32} {'expected':10} {'got':10} {'score':>6}  detail")
    lines.append("-" * 100)
    for r in results:
        detail = ""
        if r.error:
            detail = r.error
        else:
            if r.unmatched_expected:
                detail += f"missed={r.unmatched_expected} "
            if r.unexpected_matched:
                detail += f"extra={r.unexpected_matched} "
            if not r.evidence_ok:
                detail += f"BAD_EVIDENCE={r.bad_evidence}"
            if not detail:
                detail = "ok"
        lines.append(
            f"{r.spec.id:32} {r.spec.expected_verdict:10} {r.predicted_verdict:10} "
            f"{r.overall_score:6.1f}  {detail}"
        )
    lines.append("-" * 100)
    prf = summary["requirement_prf"]
    verdict = summary["verdict"]
    gap = summary["gap_detection"]
    lines.append(
        f"cases={summary['cases']} errors={summary['errors']} "
        f"mean_score={summary['mean_score']} time={summary['total_ms']}ms"
    )
    lines.append(
        f"requirement  P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f}"
    )
    lines.append(
        f"verdict      exact={verdict['exact']:.1%} within_one={verdict['within_one_grade']:.1%}"
    )
    lines.append(
        f"gap detection P={gap['precision']:.3f} R={gap['recall']:.3f} F1={gap['f1']:.3f}"
    )
    lines.append(f"evidence accuracy={summary['evidence_accuracy']:.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobmatch.evaluate", description="Evaluate the matcher on a labelled fixture.")
    parser.add_argument("fixture", nargs="?", default="data/eval_cases.json")
    parser.add_argument("--backend", choices=("auto", "llm", "rules"), default="rules")
    parser.add_argument("--embeddings", action="store_true", help="enable the embedding fallback")
    parser.add_argument("--json", metavar="PATH", help="write results as JSON")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.fixture)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: could not load fixture: {exc}", file=sys.stderr)
        return 2

    results = run(cases, backend=args.backend, use_embeddings=args.embeddings)
    summary = summarize(results)
    print(format_results(results, summary))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "cases": [
                        {
                            "id": r.spec.id,
                            "note": r.spec.note,
                            "expected_verdict": r.spec.expected_verdict,
                            "predicted_verdict": r.predicted_verdict,
                            "overall_score": r.overall_score,
                            "matched": sorted(r.matched),
                            "missing": sorted(r.missing),
                            "unmatched_expected": r.unmatched_expected,
                            "unexpected_matched": r.unexpected_matched,
                            "evidence_ok": r.evidence_ok,
                            "bad_evidence": r.bad_evidence,
                            "elapsed_ms": r.elapsed_ms,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
