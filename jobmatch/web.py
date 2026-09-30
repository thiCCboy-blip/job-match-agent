"""A local web front end for jobmatch.

Deliberately dependency-free and bound to localhost. A resume is personal data,
so the page holds the documents in the browser, calls a server on the same
machine, and never sends anything to a third party. That constraint is the
reason this is ``http.server`` and not FastAPI: the alternative is three extra
dependencies in exchange for a service that should not be reachable anyway.

    python -m jobmatch.web

Then open http://127.0.0.1:8765.
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jobmatch import pipeline, report

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>job-match-agent</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #16181d;
    --muted: #5d6470;
    --line: #dfe3ea;
    --card: #f7f8fa;
    --accent: #2f5bd0;
    --met: #1c7a45;
    --partial: #8a6100;
    --missing: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #e7e9ee; --muted: #9aa2b1; --line: #2b2f37;
      --card: #1b1e24; --accent: #7ea2ff;
      --met: #58d68d; --partial: #e0b64a; --missing: #ff8a80;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    padding: 20px 24px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: -0.01em; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { max-width: 1180px; margin: 0 auto; padding: 20px 24px 60px; }
  .inputs { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 860px) { .inputs { grid-template-columns: 1fr; } }
  label { display: block; font-weight: 600; font-size: 13px; margin-bottom: 6px; }
  textarea {
    width: 100%; min-height: 190px; resize: vertical; padding: 10px 12px;
    border: 1px solid var(--line); border-radius: 8px; background: var(--card);
    color: var(--fg); font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  textarea:focus, button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  .bar { display: flex; gap: 12px; align-items: center; margin: 18px 0; flex-wrap: wrap; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 8px;
    padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button[disabled] { opacity: 0.6; cursor: progress; }
  .bar label { margin: 0; font-weight: 400; color: var(--muted); }
  .bar input[type=number] {
    width: 70px; padding: 7px 8px; border: 1px solid var(--line);
    border-radius: 6px; background: var(--card); color: var(--fg);
  }
  .note { color: var(--muted); font-size: 12px; margin: 4px 0 0; }
  .summary { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin: 8px 0 16px; }
  .score { font-size: 30px; font-weight: 700; letter-spacing: -0.02em; }
  .verdict {
    padding: 3px 11px; border-radius: 999px; font-size: 13px; font-weight: 600;
    background: var(--card); border: 1px solid var(--line);
  }
  .meta { color: var(--muted); font-size: 13px; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  td.status { white-space: nowrap; font-weight: 600; }
  .s-met { color: var(--met); } .s-partial { color: var(--partial); }
  .s-missing { color: var(--missing); } .s-contested { color: var(--partial); }
  .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .req { font-weight: 600; }
  .why { color: var(--muted); font-size: 12.5px; }
  code {
    font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--card); border: 1px solid var(--line);
    border-radius: 5px; padding: 1px 5px; word-break: break-word;
  }
  .why code { display: inline-block; margin-top: 3px; }
  h2 { font-size: 14px; margin: 26px 0 8px; text-transform: uppercase;
       letter-spacing: 0.05em; color: var(--muted); }
  ul { margin: 0; padding-left: 20px; font-size: 13.5px; }
  li { margin: 3px 0; }
  .empty { color: var(--muted); font-size: 14px; padding: 40px 0; text-align: center; }
  .err {
    border: 1px solid var(--missing); border-left-width: 4px; border-radius: 8px;
    padding: 12px 14px; color: var(--missing); background: var(--card);
  }
  details { margin-top: 26px; }
  summary { cursor: pointer; font-size: 13px; color: var(--muted); }
  pre {
    background: var(--card); border: 1px solid var(--line); border-radius: 8px;
    padding: 14px; overflow: auto; font-size: 12.5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
</style>
</head>
<body>
<header>
  <h1>job-match-agent</h1>
  <p>Runs entirely on this machine. Nothing is uploaded.</p>
</header>
<main>
  <div class="inputs">
    <div>
      <label for="jd">Job description</label>
      <textarea id="jd" spellcheck="false"
        placeholder="Paste the posting, requirements and all."></textarea>
    </div>
    <div>
      <label for="resume">Resume</label>
      <textarea id="resume" spellcheck="false"
        placeholder="Paste your resume text."></textarea>
    </div>
  </div>
  <div class="bar">
    <button id="go" type="button">Compare</button>
    <label for="top">show top</label>
    <input id="top" type="number" value="25" min="1" max="200">
    <span id="status" class="note"></span>
  </div>
  <p class="note">
    Plain text only in the browser. For PDF or DOCX, use the command line:
    <code>python -m jobmatch --jd posting.pdf --resume resume.docx</code>
  </p>

  <div id="out"><p class="empty">Paste both documents and press Compare.</p></div>

  <details id="raw-wrap" hidden>
    <summary>Plain-text report</summary>
    <pre id="raw"></pre>
  </details>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function statusClass(status) {
  return { met: "s-met", partial: "s-partial", contested: "s-contested",
           missing: "s-missing" }[status] || "";
}

function row(o) {
  const detail = [];
  if (o.explanation) detail.push(esc(o.explanation));
  if (o.matched_as && o.matched_as !== o.requirement.name)
    detail.push("matched as <code>" + esc(o.matched_as) + "</code>");
  if (o.resume_evidence)
    detail.push("<code>L" + esc(o.resume_evidence.line) + ": " +
                esc(o.resume_evidence.text) + "</code>");
  const req = esc(o.requirement.name);
  const imp = o.requirement.importance === "must" ? "MUST" : "PREF";
  return `<tr>
    <td class="status ${statusClass(o.status)}">${esc(o.status.toUpperCase())}</td>
    <td>${imp}</td>
    <td><span class="req">${req}</span>
      ${detail.length ? `<div class="why">${detail.join("<br>")}</div>` : ""}</td>
    <td class="num">${o.score.toFixed(2)}</td>
    <td class="why">${esc(o.method)}</td>
  </tr>`;
}

function list(title, items) {
  if (!items || !items.length) return "";
  const inner = items.map((x) => {
    if (typeof x === "string") return "<li>" + esc(x) + "</li>";
    return `<li><span class="req">${esc(x.requirement.name)}</span>
      <span class="why">${esc(x.explanation || "")}</span></li>`;
  }).join("");
  return `<h2>${esc(title)}</h2><ul>${inner}</ul>`;
}

function render(d, text) {
  const r = d;
  const parts = [];
  parts.push(`<div class="summary">
    <span class="score">${r.overall_score.toFixed(1)}</span>
    <span class="verdict">${esc(r.verdict)}</span>
    <span class="meta">must-have coverage
      ${Math.round(r.must_have_coverage * 100)}% &middot;
      ${r.outcomes.length} requirements assessed &middot;
      jd=${esc(r.jd.extractor)} resume=${esc(r.resume.extractor)}</span>
  </div>`);
  parts.push(`<table>
    <thead><tr><th>Status</th><th>Need</th><th>Requirement</th>
      <th class="num">Score</th><th>Method</th></tr></thead>
    <tbody>${r.outcomes.map(row).join("")}</tbody></table>`);
  parts.push(list("Blockers", r.blockers));
  parts.push(list("Strongest matches", r.strengths));
  parts.push(list("What to do next", r.suggestions));
  $("out").innerHTML = parts.join("");
  $("raw").textContent = text;
  $("raw-wrap").hidden = false;
}

async function compare() {
  const jd = $("jd").value.trim();
  const resume = $("resume").value.trim();
  if (!jd || !resume) {
    $("out").innerHTML = '<p class="err">Paste both a job description and a resume.</p>';
    return;
  }
  const btn = $("go");
  btn.disabled = true;
  $("status").textContent = "Comparing...";
  try {
    const res = await fetch("api/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jd: jd,
        resume: resume,
        top: Number($("top").value) || 25,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    render(data.report, data.text);
    $("status").textContent = data.elapsed_ms + " ms";
  } catch (err) {
    $("out").innerHTML = '<p class="err">' + esc(err.message) + "</p>";
    $("status").textContent = "";
  } finally {
    btn.disabled = false;
  }
}

$("go").addEventListener("click", compare);
for (const id of ["jd", "resume"]) {
  $(id).addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") compare();
  });
}
</script>
</body>
</html>
"""

MAX_BODY = 4 * 1024 * 1024  # 4 MiB of pasted text is far more than any CV.

# An over-limit body is discarded rather than parsed, up to this much. Beyond it
# the connection is closed instead: there is no reason to read gigabytes from a
# client that is already past the limit.
DRAIN_LIMIT = 64 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "jobmatch-web"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, kind: str, *, close: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        # A page that reads a resume should not be cached or framed.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if close:
            # When a request is refused before its body is read, the unread
            # bytes would be parsed as the next request on this connection.
            # Closing keeps one bad request from corrupting a keep-alive stream.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _drain(self, length: int, chunk: int = 65536) -> bool:
        """Read and discard a request body so the socket ends up clean.

        Returns False when the body was too large to bother with, in which case
        the caller closes the connection instead of trying to reply.
        """
        if length > DRAIN_LIMIT:
            return False
        remaining = length
        while remaining > 0:
            data = self.rfile.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
        return True

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/api/match":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, b'{"error": "bad Content-Length"}', "application/json")
            return
        if length <= 0:
            self._send(400, b'{"error": "empty request"}', "application/json")
            return
        if length > MAX_BODY:
            # Drain the request body before answering. Replying while the client
            # is still writing leaves unread bytes in the socket; closing then
            # aborts its pending write, and the caller sees a connection reset
            # instead of the 413 explaining the problem. Reading and discarding
            # costs nothing at these sizes and keeps the error legible.
            drained = self._drain(length)
            self._send(
                413,
                b'{"error": "too much text; paste a shorter pair"}',
                "application/json",
                close=not drained,
            )
            return

        try:
            payload = json.loads(self.rfile.read(length))
            jd = str(payload["jd"]).strip()
            resume = str(payload["resume"]).strip()
            top = max(1, min(200, int(payload.get("top") or 25)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send(400, json.dumps({"error": f"bad request: {exc}"}).encode(),
                       "application/json")
            return

        # Both documents must be present, otherwise the report would read as a
        # confident verdict on an empty input.
        missing = [n for n, v in (("job description", jd), ("resume", resume)) if not v]
        if missing:
            self._send(
                400,
                json.dumps({"error": "paste the " + " and the ".join(missing)}).encode(),
                "application/json",
            )
            return

        try:
            result = pipeline.analyze_text(jd, resume, backend="rules")
        except Exception as exc:  # noqa: BLE001 - a bad document must not kill the server
            self._send(
                500,
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(),
                "application/json",
            )
            return

        body = json.dumps(
            {
                "report": report.to_dict(result.report),
                "text": report.render_text(result.report, top=top),
                "elapsed_ms": round(result.timing.total_ms, 1),
            }
        ).encode("utf-8")
        self._send(200, body, "application/json")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"job-match-agent web UI on {url}")
    print("Everything runs locally. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m jobmatch.web",
        description="Local web interface for jobmatch.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument("--open", action="store_true", help="open a browser window")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "warning: this serves resume data over the network without "
            "authentication. Only do that on a network you trust.",
            file=sys.stderr,
        )
    serve(args.host, args.port, args.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
