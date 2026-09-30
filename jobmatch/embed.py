"""Optional sentence-embedding fallback for skills outside the taxonomy.

Only two backends are supported and both are open-weights:

- `BAAI/bge-small-en-v1.5` via sentence-transformers, if installed
- `sentence-transformers/all-MiniLM-L6-v2` as the lighter fallback

Every function here degrades to "no embeddings" rather than raising. The
taxonomy plus rules are the primary matcher; embeddings only rescue requirements
whose vocabulary the taxonomy does not know, and a report must never depend on
a network download succeeding.
"""

from __future__ import annotations

import os
import threading
from typing import Sequence

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


class EmbeddingUnavailable(RuntimeError):
    """Raised only when a caller explicitly demands embeddings."""


def _backend() -> str | None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    return os.environ.get("JOBMATCH_EMBED_MODEL") or "sentence-transformers"


def is_available() -> bool:
    return _backend() is not None


def load_embedder(model_name: str | None = None) -> object | None:
    """Load the sentence-transformers model, cached per process.

    Returns None when the backend is missing, the model cannot be fetched, or
    the environment sets `JOBMATCH_NO_EMBED=1`.
    """
    if os.environ.get("JOBMATCH_NO_EMBED") == "1":
        return None
    backend = _backend()
    if backend is None:
        return None

    key = model_name or DEFAULT_MODEL
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        from sentence_transformers import SentenceTransformer

        candidates = [key]
        if key == DEFAULT_MODEL:
            candidates.append(FALLBACK_MODEL)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                model = SentenceTransformer(candidate)
                _CACHE[key] = model
                return model
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise EmbeddingUnavailable(str(last_error)) from last_error


def embed_texts(texts: Sequence[str], model: object | None = None) -> list[list[float]]:
    """Embed a batch of strings to unit-normalized vectors."""
    active = model or load_embedder()
    if active is None:
        raise EmbeddingUnavailable("sentence-transformers is not installed")
    vectors = active.encode(  # type: ignore[attr-defined]
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [list(map(float, row)) for row in vectors]


def cosine_matrix(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Cosine similarity between two sets of vectors, computed without numpy."""
    if not left or not right:
        return [[] for _ in left]
    out: list[list[float]] = []
    for a in left:
        row: list[float] = []
        for b in right:
            dot = sum(x * y for x, y in zip(a, b))
            row.append(max(-1.0, min(1.0, dot)))
        out.append(row)
    return out
