"""Command line interface.

    python -m jobmatch --jd posting.txt --resume resume.pdf
    python -m jobmatch --jd posting.txt --resume resume.md --json out.json
    python -m jobmatch --jd posting.txt --resume resume.md --backend rules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobmatch import embed, pipeline, report as report_mod
from jobmatch.matcher import MatchConfig

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ERROR = 1
EXIT_BLOCKED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobmatch",
        description="Match a job description against a resume, locally, with open-weights models.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd", metavar="PATH", help="job description file (txt, md, pdf, docx)")
    source.add_argument("--jd-text", metavar="TEXT", help="job description as a literal string")

    parser.add_argument(
        "--resume", metavar="PATH", required=True, help="resume file (txt, md, pdf, docx)"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "llm", "rules"),
        default="auto",
        help=(
            "auto or llm: use the local model, merged with pattern rules "
            "(recommended); rules: no model at all"
        ),
    )
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON to this path")
    parser.add_argument(
        "--style", choices=("console", "text"), default="console", help="output renderer"
    )
    parser.add_argument("--top", type=int, default=30, help="how many requirements to list")
    parser.add_argument(
        "--no-embeddings", action="store_true", help="disable the embedding fallback entirely"
    )
    parser.add_argument(
        "--must-have-gate/--no-must-have-gate",
        dest="must_have_gate",
        default=True,
        help="cap the verdict when a required item is missing (default: on)",
    )
    parser.add_argument(
        "--partial-credit",
        type=float,
        default=0.5,
        help="credit given for a partially evidenced requirement (0 to 1)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the human-readable report")
    return parser


def _resolve_jd(args: argparse.Namespace) -> str:
    if args.jd_text:
        return args.jd_text
    return args.jd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (0.0 <= args.partial_credit <= 1.0):
        print("error: --partial-credit must be between 0 and 1", file=sys.stderr)
        return EXIT_USAGE

    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"error: resume not found: {resume_path}", file=sys.stderr)
        return EXIT_USAGE

    config = MatchConfig(
        use_embeddings=not args.no_embeddings,
        partial_credit=args.partial_credit,
        must_have_gate=args.must_have_gate,
    )

    try:
        if args.jd_text:
            result = pipeline.analyze_text(args.jd_text, str(resume_path), backend=args.backend, config=config)
        else:
            result = pipeline.analyze_files(args.jd, resume_path, backend=args.backend, config=config)
    except pipeline.PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_mod.to_json(result.report), encoding="utf-8")
        if not args.quiet:
            print(f"wrote {out_path}")

    if not args.quiet:
        print(pipeline.render(result, style=args.style, top=args.top))

    if not args.quiet and args.style == "console":
        timing = result.timing
        print(
            f"jd={result.report.jd.extractor} resume={result.report.resume.extractor} "
            f"extract={timing.extract_ms}ms match={timing.match_ms}ms total={timing.total_ms}ms"
        )
        if not embed.is_available():
            print("note: sentence-transformers not installed, embedding fallback disabled")

    if result.report.blockers:
        return EXIT_BLOCKED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
