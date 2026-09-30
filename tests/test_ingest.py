"""Ingest tests: line numbering, section tagging, format handling, limits."""

from __future__ import annotations

import pytest

from jobmatch import ingest


def test_line_numbers_are_sequential_and_1_indexed():
    doc = ingest.load_document("alpha\n\nbeta\ngamma\n", doc_type="text")
    assert [line.number for line in doc.lines] == [1, 2, 3]
    assert doc.text == "alpha\nbeta\ngamma"


def test_section_headings_are_tagged():
    text = """Jane Doe
EXPERIENCE
Engineer at Acme | 01/2020 - 01/2022
EDUCATION
B.Tech Civil Engineering
SKILLS
Python, SQL
"""
    doc = ingest.load_document(text, doc_type="text")
    sections = {line.section for line in doc.lines}
    assert "experience" in sections
    assert "education" in sections
    assert "skills" in sections

    # Blank lines are dropped, so the numbering is over retained lines.
    assert doc.line_map() == {
        1: "Jane Doe",
        2: "EXPERIENCE",
        3: "Engineer at Acme | 01/2020 - 01/2022",
        4: "EDUCATION",
        5: "B.Tech Civil Engineering",
        6: "SKILLS",
        7: "Python, SQL",
    }
    assert doc.lines[2].section == "experience"
    assert doc.lines[4].section == "education"
    assert doc.lines[6].section == "skills"


def test_bullets_are_flagged():
    doc = ingest.load_document("- one\n* two\n1. three\nplain\n", doc_type="text")
    flags = [line.is_bullet for line in doc.lines]
    assert flags == [True, True, True, False]


def test_long_lines_are_capped():
    doc = ingest.load_document("x" * 5000, doc_type="text")
    assert len(doc.lines[0].text) <= ingest.MAX_LINE_CHARS


def test_empty_input_raises():
    with pytest.raises(ingest.IngestError):
        ingest.load_document("   \n  \n", doc_type="text")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ingest.IngestError):
        ingest.load_file(tmp_path / "nope.pdf")


def test_oversized_file_rejected(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("a" * (ingest.MAX_FILE_BYTES + 1), encoding="utf-8")
    with pytest.raises(ingest.IngestError, match="over the"):
        ingest.load_file(path)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(ingest.IngestError):
        ingest.load_file(path)


def test_docx_roundtrip(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "resume.docx"
    document = docx.Document()
    document.add_paragraph("SKILLS")
    document.add_paragraph("- Python and SQL")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Company"
    table.rows[0].cells[1].text = "Role"
    document.save(str(path))

    loaded = ingest.load_file(path)
    assert "Python and SQL" in loaded.text
    assert "Company" in loaded.text
    assert loaded.doc_type == "docx"


def test_mask_pii_hides_contact_details():
    text = "reach me at a.b@example.com or +91 82812 12657, site https://github.com/x"
    masked = ingest.mask_pii(text)
    assert "a.b@example.com" not in masked
    assert "82812" not in masked
    assert "github.com/x" not in masked


def test_looks_like_jd():
    assert ingest.looks_like_jd("Responsibilities: you will build pipelines. Requirements: 3 yrs SQL")
    assert not ingest.looks_like_jd("just a short note")
