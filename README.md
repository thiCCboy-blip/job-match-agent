# job-match-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Matches a resume against a job description and explains the result: which
requirements are met, which are genuinely missing, and why a call went the way
it did. Runs entirely on your machine with open-weights models. Nothing is
uploaded.

## Why it exists

Keyword overlap is a poor proxy for fit. A candidate who supervised a
construction site matches "managing construction site operations" perfectly, and
a candidate with a B.Tech matches "Bachelor's degree in Engineering" perfectly,
even though neither shares the exact phrase a naive matcher looks for. It also
gets those cases backwards: someone who *mentions* dbt next to SQL does not
thereby have dbt.

The design here is that extraction and matching are separate jobs. Extraction
finds what the documents say. Matching decides what that means, deterministically
and with a written reason attached. Every score can be traced to a line number.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The extra index matters on Windows: the plain PyPI `torch` wheel is CPU-only.

## Use

```powershell
# Readable report
python -m jobmatch --jd posting.txt --resume resume.md

# Job description as a literal string
python -m jobmatch --jd-text "Data Analyst. Requirements: SQL, Python" --resume resume.pdf

# Machine-readable output
python -m jobmatch --jd posting.txt --resume resume.md --json report.json

# No model, no download: pattern rules only
python -m jobmatch --jd posting.txt --resume resume.md --backend rules
```

Exit codes: `0` matched, `1` error, `2` bad usage, `3` blocked by a missing
must-have. Useful in CI to fail a pipeline on a known gap.

Supported inputs: PDF, DOCX, Markdown, plain text.

### Web interface

```powershell
python -m jobmatch.web          # then open http://127.0.0.1:8765
```

Paste a job description and a resume, press Compare. The page is plain HTML
with no build step and no CDN, so it works offline and starts in about a
second. It is bound to `127.0.0.1` and uses the `rules` backend, which makes a
match take roughly 100ms.

`--host` and `--port` are available, but binding to anything other than
localhost puts resume data on the network with no authentication. The page
itself is safe against cross-site scripting: documents travel as JSON and are
inserted with `textContent`, never as HTML.

The browser accepts pasted text only. For PDF or DOCX, use the command line.

As a library:

```python
from jobmatch import pipeline
from jobmatch.schemas import MatchStatus

result = pipeline.analyze_files("posting.pdf", "resume.docx")
print(result.report.verdict, result.report.overall_score)

for outcome in result.report.outcomes:
    if outcome.status is MatchStatus.MISSING:
        print(outcome.requirement.name, "-", outcome.explanation)
```

## How it works

**Ingest.** Files become numbered lines with section tags, so any claim in a
report can cite a line. PII is masked in evidence.

**Extract.** Two sources, merged:
- Qwen3-1.7B under grammar-constrained decoding, so the output is valid JSON by
  construction.
- Pattern rules, which are exhaustive by construction.

Neither is trusted alone. Constrained decoding guarantees *valid*, not
*complete*: the 1.7B model reads a three-bullet posting and returns one
requirement. The rules pass supplies everything the model skipped, and the model
supplies the importance and category nuance the patterns miss. The merged
requirement list is what gets scored.

**Match.** Deterministic, in order: canonical name, synonym, degree level,
compound phrase, then a tool that implements a named practice (`CI/CD` satisfied
by GitHub Actions, at contested rather than full credit). Everything is
weighted by importance, must-have 1.0, preferred 0.45, nice-to-have 0.15.

**Report.** Console or text, with a JSON option. Gaps carry a reason string.

## Measured behaviour

208 unit tests, plus an 8-case labelled evaluation set
(`data/eval_cases.json`).

| Backend | Requirement P/R/F1 | Verdict exact | Gap detection F1 | Evidence |
|---|---|---|---|---|
| `rules` | 1.000 / 1.000 / 1.000 | 87.5% | 1.000 | 100% |
| `rules` + embeddings | 1.000 / 1.000 / 1.000 | 87.5% | 1.000 | 100% |
| `llm` (merged with rules) | 0.966 / 1.000 / 0.983 | 75.0% | 1.000 | 100% |

Verdicts are graded `strong` / `stretch` / `weak`. Within one grade, all three
backends score 100%.

Reproduce:

```powershell
python -m jobmatch.evaluate data/eval_cases.json --backend rules
```

The `rules` backend runs the whole set in about 140ms. The `llm` backend takes
around 50s per case on an 8GB GPU, almost all of it generation.

Note on embeddings: the fallback is restricted to requirements the taxonomy does
not recognise, and the model is loaded lazily. Two measured reasons. Letting it
second-guess a skill the taxonomy confidently marks absent is a net loss,
because the genuine and false similarities overlap: "Communication" against a
resume with no such skill scored 0.583, while the false "dbt" against a resume
with only SQL scored 0.647, and no threshold separates those. And loading eagerly
added ~20s to runs that never consulted it, so a run that resolves entirely from
the taxonomy now finishes in ~25ms either way.

## Notes and limits

- Extraction is English-only. Matching rules are English-only too.
- The taxonomy is deliberately finite. Unknown skills are reported as
  unrecognised rather than guessed at.
- `Qwen3-1.7B` is small. It is good at classification and bad at exhaustive
  enumeration, which is why the merge exists rather than a bigger model.
- Scores are a structured opinion, not a hiring decision.
- **A verdict is only as good as the parse.** Pasting a posting whose line
  breaks were lost collapses it into one clause, yields one requirement, and
  produces a confident and meaningless 100/100. The report detects that
  specific failure and says so in its first line, but it is worth knowing that
  "strong match" on a one-requirement input means the input was not read, not
  that you are a strong match.
- The `llm` backend scores *worse* than `rules` on verdict accuracy (75% vs
  87.5%) while taking 50s per case. It earns its place on unusual phrasing,
  not on the benchmark set. `rules` is the default the web interface uses.
