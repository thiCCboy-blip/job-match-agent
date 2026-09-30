"""Pipeline, report rendering, and CLI tests. No model, no network."""

from __future__ import annotations

import json

import pytest

from jobmatch import cli, pipeline, report as report_mod
from jobmatch.schemas import Importance, MatchStatus

JD = """Data Analyst
Requirements:
- 3+ years of SQL required
- Excel mandatory
- dbt is a plus
"""

RESUME = """Jane Doe
EXPERIENCE
Data Analyst, Acme | 01/2021 - 06/2024
- Built SQL pipelines
EDUCATION
B.Tech Civil Engineering
SKILLS
Python, SQL, Excel
"""


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    monkeypatch.setenv("JOBMATCH_NO_EMBED", "1")
    monkeypatch.setattr(pipeline.extract, "build_extractor", lambda backend: None)


def test_analyze_text_runs_end_to_end():
    result = pipeline.analyze_text(JD, RESUME, backend="rules")
    assert result.report.overall_score > 0
    assert result.report.jd.degraded is True
    assert result.timing.total_ms >= 0


def test_analyze_files_roundtrip(tmp_path):
    jd_path = tmp_path / "jd.md"
    resume_path = tmp_path / "resume.md"
    jd_path.write_text(JD, encoding="utf-8")
    resume_path.write_text(RESUME, encoding="utf-8")
    result = pipeline.analyze_files(jd_path, resume_path, backend="rules")
    assert result.jd_document.path == jd_path
    assert result.report.jd.title


def test_missing_file_raises_pipeline_error(tmp_path):
    with pytest.raises(pipeline.PipelineError):
        pipeline.analyze_files(tmp_path / "nope.txt", tmp_path / "also-nope.txt")


def test_json_report_roundtrips():
    result = pipeline.analyze_text(JD, RESUME, backend="rules")
    payload = json.loads(report_mod.to_json(result.report))
    assert payload["verdict"] in {"strong", "plausible", "stretch", "weak"}
    assert isinstance(payload["outcomes"], list)


def test_text_report_mentions_scores_and_evidence():
    result = pipeline.analyze_text(JD, RESUME, backend="rules")
    text = report_mod.render_text(result.report)
    assert "JOB MATCH REPORT" in text
    assert "overall" in text
    assert "REQUIREMENT BREAKDOWN" in text
    assert "SQL" in text


def test_console_renderer_runs():
    result = pipeline.analyze_text(JD, RESUME, backend="rules")
    rendered = report_mod.render_console(result.report)
    assert "overall" in rendered.lower() or "match" in rendered.lower()


def test_cli_returns_blocked_exit_code(tmp_path, capsys):
    jd = tmp_path / "jd.md"
    resume = tmp_path / "resume.md"
    jd.write_text("Role\nRequirements:\n- Kubernetes required\n", encoding="utf-8")
    resume.write_text(RESUME, encoding="utf-8")
    code = cli.main(["--jd", str(jd), "--resume", str(resume), "--backend", "rules", "--style", "text"])
    assert code == cli.EXIT_BLOCKED


def test_cli_success_exit_code(tmp_path):
    jd = tmp_path / "jd.md"
    resume = tmp_path / "resume.md"
    jd.write_text("Role\nRequirements:\n- SQL required\n", encoding="utf-8")
    resume.write_text(RESUME, encoding="utf-8")
    code = cli.main(["--jd", str(jd), "--resume", str(resume), "--backend", "rules", "--style", "text", "--quiet"])
    assert code == cli.EXIT_OK


def test_cli_writes_json(tmp_path):
    jd = tmp_path / "jd.md"
    resume = tmp_path / "resume.md"
    out = tmp_path / "out" / "report.json"
    jd.write_text(JD, encoding="utf-8")
    resume.write_text(RESUME, encoding="utf-8")
    code = cli.main(
        [
            "--jd", str(jd), "--resume", str(resume),
            "--backend", "rules", "--style", "text", "--quiet", "--json", str(out),
        ]
    )
    assert code in (cli.EXIT_OK, cli.EXIT_BLOCKED)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "overall_score" in payload


def test_cli_rejects_missing_resume(capsys):
    assert cli.main(["--jd-text", "Role", "--resume", "definitely-missing.md"]) == cli.EXIT_USAGE


def test_cli_rejects_bad_partial_credit(capsys):
    code = cli.main(["--jd-text", "Role", "--resume", __file__, "--partial-credit", "5"])
    assert code == cli.EXIT_USAGE


def test_render_dispatches_to_both_styles():
    result = pipeline.analyze_text(JD, RESUME, backend="rules")
    assert isinstance(pipeline.render(result, style="text"), str)
    assert isinstance(pipeline.render(result, style="console"), str)
