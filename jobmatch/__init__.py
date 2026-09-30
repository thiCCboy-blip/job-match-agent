"""package init: public library surface for job-match-agent."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["analyze", "MatchConfig"]


def __getattr__(name: str):  # lazy so importing the package stays cheap
    if name in {"analyze", "MatchConfig"}:
        from jobmatch import matcher

        return getattr(matcher, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
