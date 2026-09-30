"""Document ingestion with line-level provenance.

Every document becomes a list of `Line` records. Downstream code never sees a
bare string, because every claim the matcher makes has to be traceable back to
a line a human can open and read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_LINES = 4000
MAX_LINE_CHARS = 2000

RESUME_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("summary", r"^(professional\s+)?(summary|profile|objective|about)\b"),
    ("experience", r"^(work\s+|professional\s+|relevant\s+)?(experience|employment|work history|history)\b"),
    ("education", r"^education\b"),
    ("skills", r"^(technical\s+|core\s+|key\s+)?(skills|competencies|technologies|tool ?stack)\b"),
    ("certifications", r"^(certifications?|licenses?|credentials?)\b"),
    ("projects", r"^projects?\b"),
    ("contact", r"^(contact|contact info|personal details?)\b"),
]

NUMBERED_ITEM = re.compile(r"^\s*(?:[-*\u2022\u2013\u2014\u25cf]|\d+[.)])\s+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:github|linkedin)\.com/\S+", re.IGNORECASE)
# Covers grouped western numbers (555 123 4567), compact ones (5551234567), and
# country-prefixed forms with a variable-length group such as +91 82812 12657.
PHONE = re.compile(
    r"(?<![\w.])"
    r"(?:\+?\d{1,3}[\s.-]?|\(\d{3}\)[\s.-]?)?"
    r"(?:\d[\s.-]?){7,14}\d"
    r"(?!\d)"
)


class IngestError(Exception):
    """Raised when a document cannot be read or exceeds the size limits."""


@dataclass(frozen=True)
class Line:
    """One logical line of a document, tagged with its section."""

    number: int
    text: str
    section: str | None = None
    is_bullet: bool = False

    def evidence(self) -> dict[str, object]:
        return {"text": self.text, "line": self.number, "section": self.section}


@dataclass
class Document:
    """A parsed document: ordered lines plus the metadata we care about."""

    path: Path | None
    lines: list[Line] = field(default_factory=list)
    raw_text: str = ""
    doc_type: str = "text"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def line_map(self) -> dict[int, str]:
        return {line.number: line.text for line in self.lines}

    def __bool__(self) -> bool:
        return bool(self.raw_text.strip())


def _guess_doc_type(path: Path | None) -> str:
    if path is None:
        return "text"
    suffix = path.suffix.lower()
    return {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".rtf": "text",
    }.get(suffix, "text")


def _clean_lines(raw: str) -> list[tuple[str, bool]]:
    """Split raw text, drop empties, flag bullets, and cap pathological input."""
    out: list[tuple[str, bool]] = []
    for raw_line in raw.splitlines():
        line = raw_line.replace("\u00a0", " ").replace("\xad", "").rstrip()
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS]
        if not line.strip():
            continue
        out.append((line.strip(), bool(NUMBERED_ITEM.match(line))))
        if len(out) >= MAX_LINES:
            break
    return out


def _tag_sections(cleaned: list[tuple[str, bool]]) -> list[Line]:
    """Label each line with the resume section it sits under.

    Heading detection is deliberately conservative: a heading is a short line
    with no bullet marker and no trailing sentence punctuation. Mislabeling a
    line is a display issue only, never a scoring one.
    """
    lines: list[Line] = []
    current: str | None = None
    for number, (text, is_bullet) in enumerate(cleaned, start=1):
        if not is_bullet and len(text) <= 60 and not EMAIL.search(text) and not URL.search(text):
            lowered = text.lower().strip(" :#*")
            for section, pattern in RESUME_SECTION_PATTERNS:
                if re.search(pattern, lowered):
                    current = section
                    break
        lines.append(Line(number=number, text=text, section=current, is_bullet=is_bullet))
    return lines


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError("pypdf is required to read PDF files") from exc
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise IngestError(f"could not parse PDF: {exc}") from exc


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError("python-docx is required to read DOCX files") from exc
    try:
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as exc:
        raise IngestError(f"could not parse DOCX: {exc}") from exc


def load_document(source: str | Path, *, doc_type: str | None = None) -> Document:
    """Read a file or literal string into a `Document`.

    Passing a string that looks like a path and exists is treated as a path;
    anything else is treated as literal text, so callers can feed either a
    file or a pasted job description without branching.
    """
    if isinstance(source, Path) or ("\n" not in str(source) and len(str(source)) < 260):
        candidate = Path(source)
        if candidate.exists():
            return load_file(candidate, doc_type=doc_type)

    text = str(source)
    if not text.strip():
        raise IngestError("input document is empty")
    return Document(
        path=None,
        lines=_tag_sections(_clean_lines(text)),
        raw_text=text,
        doc_type=doc_type or "text",
    )


def load_file(path: str | Path, *, doc_type: str | None = None) -> Document:
    """Read a file from disk into a `Document`, enforcing size limits."""
    file_path = Path(path)
    if not file_path.exists():
        raise IngestError(f"file not found: {file_path}")
    if not file_path.is_file():
        raise IngestError(f"not a file: {file_path}")

    size = file_path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise IngestError(
            f"file is {size / 1e6:.1f} MB, over the {MAX_FILE_BYTES / 1e6:.0f} MB limit"
        )
    if size == 0:
        raise IngestError(f"file is empty: {file_path}")

    kind = doc_type or _guess_doc_type(file_path)
    if kind == "pdf":
        raw = _read_pdf(file_path)
    elif kind == "docx":
        raw = _read_docx(file_path)
    else:
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                raw = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise IngestError(f"could not decode {file_path.name} as text")

    if not raw.strip():
        raise IngestError(
            f"no text extracted from {file_path.name}; scanned images are not supported"
        )

    return Document(
        path=file_path,
        lines=_tag_sections(_clean_lines(raw)),
        raw_text=raw,
        doc_type=kind,
    )


def mask_pii(text: str) -> str:
    """Redact contact details so documents can be logged safely."""
    text = EMAIL.sub("[email]", text)
    text = URL.sub("[url]", text)
    return PHONE.sub("[phone]", text)


def looks_like_jd(text: str) -> bool:
    """Heuristic check that a pasted block reads like a job description."""
    lowered = text.lower()
    signals = (
        "responsibilit",
        "requirement",
        "qualification",
        "we are looking",
        "you have",
        "experience with",
        "skills",
        "role",
        "about the job",
    )
    hits = sum(1 for s in signals if s in lowered)
    return hits >= 2
