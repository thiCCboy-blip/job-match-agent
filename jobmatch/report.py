"""Human-facing rendering of a match report.

Two renderers: a terminal report via `rich`, and a plain-text one for logs and
files. Both cite the resume line behind every claim, because a score a
candidate cannot audit is a score they will not trust.
"""

from __future__ import annotations

import json
from typing import Any

from jobmatch.schemas import IMPORTANCE_WEIGHTS, Importance, MatchOutcome, MatchReport, MatchStatus

VERDICT_LABEL = {
    "strong": "Strong match",
    "plausible": "Plausible match",
    "stretch": "Stretch",
    "weak": "Weak match",
}

STATUS_ICON = {
    MatchStatus.MET: "[green]MET    [/]",
    MatchStatus.PARTIAL: "[yellow]PARTIAL[/]",
    MatchStatus.CONTESTED: "[magenta]CONTEST[/]",
    MatchStatus.MISSING: "[red]MISSING[/]",
}


def to_dict(report: MatchReport) -> dict[str, Any]:
    return report.model_dump(mode="json")


def to_json(report: MatchReport, *, indent: int = 2) -> str:
    return json.dumps(to_dict(report), indent=indent, ensure_ascii=False)


def _importance_tag(importance: Importance) -> str:
    return {Importance.MUST: "MUST", Importance.PREFERRED: "PREF", Importance.NICE_TO_HAVE: "NICE"}[importance]


def render_text(report: MatchReport, *, top: int = 40) -> str:
    lines: list[str] = []
    title = report.jd.title or "Untitled role"
    company = f" at {report.jd.company}" if report.jd.company else ""
    lines.append(f"JOB MATCH REPORT - {title}{company}")
    lines.append("=" * 72)
    lines.append(
        f"overall {report.overall_score:5.1f}/100   verdict: {VERDICT_LABEL[report.verdict]}"
    )
    musts = sum(1 for r in report.jd.requirements if r.importance is Importance.MUST)
    lines.append(
        f"must-have coverage {report.must_have_coverage:.0%} ({musts} required items)   "
        f"requirements assessed: {len(report.jd.requirements)}"
    )
    lines.append(
        f"extraction: jd={report.jd.extractor} resume={report.resume.extractor}"
    )
    lines.append("")
    lines.append("REQUIREMENT BREAKDOWN")
    lines.append("-" * 72)
    for outcome in report.outcomes[:top]:
        weight = IMPORTANCE_WEIGHTS[outcome.requirement.importance]
        detail = ""
        if outcome.resume_evidence:
            detail = f" [resume L{outcome.resume_evidence.line}]"
        lines.append(
            f"{_importance_tag(outcome.requirement.importance):5} "
            f"{outcome.status.value:9} {outcome.score:4.2f} w{weight:.2f}  "
            f"{outcome.requirement.name}{detail}"
        )
        if outcome.matched_as and outcome.matched_as.lower() != outcome.requirement.name.lower():
            lines.append(f"        matched as: {outcome.matched_as} via {outcome.method}")
    if len(report.outcomes) > top:
        lines.append(f"... and {len(report.outcomes) - top} more")

    if report.blockers:
        lines.append("")
        lines.append("BLOCKERS (required and not evidenced)")
        lines.append("-" * 72)
        for outcome in report.blockers:
            lines.append(f"  - {outcome.requirement.name}: {outcome.explanation}")

    lines.append("")
    lines.append("STRONGEST MATCHES")
    lines.append("-" * 72)
    for outcome in report.strengths[:6]:
        where = f" (resume L{outcome.resume_evidence.line})" if outcome.resume_evidence else ""
        lines.append(f"  + {outcome.requirement.name}{where}")

    if report.suggestions:
        lines.append("")
        lines.append("WHAT TO DO NEXT")
        lines.append("-" * 72)
        for index, suggestion in enumerate(report.suggestions, start=1):
            lines.append(f"  {index}. {suggestion}")

    lines.append("")
    lines.append(f"scoring config: {report.scoring_config}")
    return "\n".join(lines)


def render_console(report: MatchReport, *, top: int = 30) -> str:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    title = report.jd.title or "Untitled role"
    header = f"[bold]{title}[/]"
    if report.jd.company:
        header += f" - {report.jd.company}"
    if report.jd.seniority != "unknown":
        header += f" ({report.jd.seniority} level)"

    verdict_color = {
        "strong": "green", "plausible": "cyan", "stretch": "yellow", "weak": "red",
    }[report.verdict]
    body = (
        f"[bold]{report.overall_score:.1f}[/bold]/100   "
        f"[{verdict_color}]{VERDICT_LABEL[report.verdict]}[/{verdict_color}]\n"
        f"must-have coverage [bold]{report.must_have_coverage:.0%}[/bold]   "
        f"requirements assessed: {len(report.jd.requirements)}"
    )

    table = Table(show_lines=False, box=None, pad_edge=False)
    table.add_column("", width=9)
    table.add_column("req", width=5)
    table.add_column("requirement", ratio=3)
    table.add_column("score", width=6, justify="right")
    table.add_column("via", width=11)
    table.add_column("resume", width=7, justify="right")

    for outcome in report.outcomes[:top]:
        table.add_row(
            STATUS_ICON[outcome.status],
            _importance_tag(outcome.requirement.importance),
            outcome.requirement.name,
            f"{outcome.score:.2f}",
            outcome.method,
            f"L{outcome.resume_evidence.line}" if outcome.resume_evidence else "-",
        )
    if len(report.outcomes) > top:
        table.caption = f"... and {len(report.outcomes) - top} more requirements"

    sections = [Panel(body, title=header, border_style=verdict_color), table]

    if report.blockers:
        blockers = "\n".join(
            f"  [red]x[/red] {o.requirement.name} - {o.explanation}" for o in report.blockers
        )
        sections.append(Panel(blockers, title="blockers", border_style="red"))

    if report.suggestions:
        numbered = "\n".join(
            f"  [bold]{i}.[/bold] {s}" for i, s in enumerate(report.suggestions, start=1)
        )
        sections.append(Panel(numbered, title="what to do next", border_style="cyan"))

    console = Console(record=True, width=100)
    for section in sections:
        console.print(section)
    return console.export_text()
