"""Tests for the local web front end.

The server is started on an ephemeral port in a thread, so these exercise real
HTTP rather than calling handler methods directly.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from jobmatch import web


@pytest.fixture(scope="module")
def base_url():
    """A running server, shared by the module's tests."""
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(base_url: str, path: str, payload: dict, timeout: float = 60.0):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(base_url: str, path: str):
    try:
        with urllib.request.urlopen(base_url + path, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


JD = """Civil Engineer - Site Operations
Requirements
- 3+ years managing construction site operations is required
- Strong Primavera P6 scheduling experience is mandatory
- AutoCAD for site layouts required
- Bachelor's degree in Civil Engineering required
"""

RESUME = """SKILLS
Primavera P6, AutoCAD
EDUCATION
B.Tech, Civil Engineering - National Institute of Technology Calicut
"""


def test_index_page_is_served(base_url):
    status, body, _ = _get(base_url, "/")
    assert status == 200
    html = body.decode()
    assert "<textarea id=\"jd\"" in html
    assert "<textarea id=\"resume\"" in html
    # The page must not pull code from a third party; it runs offline.
    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "https://" not in html.replace("https://", "", 0)


def test_index_sets_no_store_and_nosniff(base_url):
    _, _, headers = _get(base_url, "/")
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_health(base_url):
    status, body, _ = _get(base_url, "/api/health")
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_match_returns_a_renderable_report(base_url):
    status, data = _post(base_url, "/api/match",
                         {"jd": JD, "resume": RESUME, "top": 5})
    assert status == 200
    report = data["report"]

    assert 0.0 <= report["overall_score"] <= 100.0
    assert report["verdict"] in {"strong", "plausible", "stretch", "weak"}
    assert 0.0 <= report["must_have_coverage"] <= 1.0

    # Every field the browser table reads must be present.
    for outcome in report["outcomes"]:
        assert set(outcome) >= {
            "status", "score", "method", "requirement", "explanation",
        }
        assert outcome["requirement"]["name"]
        assert 0.0 <= outcome["score"] <= 1.0

    assert data["text"].startswith("JOB MATCH REPORT")
    assert data["elapsed_ms"] > 0


def test_text_report_has_no_rich_console_markup(base_url):
    # render_text is a plain renderer, but rich tags would show up as literal
    # "[green]" in the browser's <pre> block.
    _, data = _post(base_url, "/api/match", {"jd": JD, "resume": RESUME, "top": 5})
    for tag in ("[green]", "[yellow]", "[red]", "[magenta]", "[/]"):
        assert tag not in data["text"]


def test_hostile_documents_are_returned_as_data_not_executed(base_url):
    """A resume can contain anything; it must arrive as inert JSON."""
    evil = "<img src=x onerror=alert(1)>\nSKILLS\nPython"
    status, data = _post(base_url, "/api/match", {"jd": JD, "resume": evil})
    assert status == 200
    # The payload is JSON, so the browser's esc() renders it literally. This
    # test guards the server side of that contract: it stays data, and the
    # response is JSON rather than HTML.
    assert isinstance(data, dict)
    json.dumps(data)  # must stay serialisable
    assert "onerror" not in json.dumps(data["report"]).replace("\\u003c", "<")


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"jd": JD, "resume": "   "}, "resume"),
        ({"jd": "", "resume": RESUME}, "job description"),
        ({"jd": "  ", "resume": "  "}, "job description"),
        ({"jd": JD}, "'resume'"),
    ],
)
def test_incomplete_input_is_rejected_with_a_useful_message(base_url, payload, fragment):
    status, data = _post(base_url, "/api/match", payload)
    assert status == 400
    assert fragment in data["error"]


def test_bad_json_body(base_url):
    request = urllib.request.Request(
        base_url + "/api/match", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=30)
    assert exc.value.code == 400


def test_non_numeric_top_is_rejected(base_url):
    status, data = _post(base_url, "/api/match",
                         {"jd": JD, "resume": RESUME, "top": "abc"})
    assert status == 400
    assert "error" in data


def test_unknown_post_route_is_404(base_url):
    request = urllib.request.Request(
        base_url + "/api/nope", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=30)
    assert exc.value.code == 404


@pytest.mark.parametrize("path", [
    "/../jobmatch/web.py",
    "/../../requirements.txt",
    "/static/../extract.py",
])
def test_no_path_traversal(base_url, path):
    status, body, _ = _get(base_url, path)
    assert status == 404
    assert b"class Handler" not in body
    assert b"def main" not in body


def test_oversize_body_is_refused(base_url):
    huge = "x" * (web.MAX_BODY + 1024)
    status, data = _post(base_url, "/api/match", {"jd": JD, "resume": huge})
    assert status == 413
    assert "error" in data


def test_concurrent_requests(base_url):
    """ThreadingHTTPServer is used so one slow match cannot block the page."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(
            lambda _: _post(base_url, "/api/match",
                            {"jd": JD, "resume": RESUME, "top": 3})[0],
            range(6),
        ))
    assert statuses == [200] * 6
