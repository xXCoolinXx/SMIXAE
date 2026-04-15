"""Local HTTP server for camera-ready figure collection.

Workflow:
1. Start the server: ``smixae latex save-server --output-dir results/camera_ready/``
2. Open any experts.html in your browser.  Each page detects the server and shows
   a floating badge in the corner with the current queue count.
3. Browse experts, rotate the 3D view, click "Save scatter" or "Save means" on
   figures you like.  Each click queues the figure server-side — no dialog.
4. The badge updates as you accumulate figures.  Click it to open the gallery at
   http://127.0.0.1:7788 where you can review thumbnails, remove unwanted ones,
   and click "Save All" to write every queued PNG to disk at once.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import typer

app = typer.Typer()

_DEFAULT_PORT = 7788


def _autocrop_png(data: bytes) -> bytes:
    """Crop white borders from a PNG by finding the bounding box of non-white pixels."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Invert: white (255) becomes black (0), anything else becomes non-zero.
    # getbbox() returns the bounding box of non-zero pixels = non-white content.
    bbox = ImageOps.invert(img).getbbox()
    if bbox is None:
        return data

    # Small padding so the plot doesn't touch the edge
    pad = 4
    bbox = (
        max(bbox[0] - pad, 0),
        max(bbox[1] - pad, 0),
        min(bbox[2] + pad, img.width),
        min(bbox[3] + pad, img.height),
    )
    cropped = img.crop(bbox)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _ssh_forward_hint(port: int) -> str | None:
    """Return a port-forward hint string if running inside an SSH session, else None."""
    if not (os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
        return None
    hostname = socket.gethostname()
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    return (
        f"\n  \033[33mSSH session detected.\033[0m  Forward the port with:\n"
        f"    ssh -L {port}:localhost:{port} {user}@{hostname}\n"
        f"  Then open  http://127.0.0.1:{port}/  in your local browser.\n"
    )

# ── In-memory queue ────────────────────────────────────────────────────────────

# filename (str) → PNG bytes
_queue: dict[str, bytes] = {}


# ── Gallery HTML ───────────────────────────────────────────────────────────────

def _ssh_banner_html(port: int) -> str:
    """Return an HTML banner block if running inside SSH, else empty string."""
    if not (os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
        return ""
    hostname = socket.gethostname()
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    cmd = f"ssh -L {port}:localhost:{port} {user}@{hostname}"
    return (
        f'<div style="background:#fff3cd;border:1px solid #ffc107;padding:10px 20px;'
        f'font-size:13px;font-family:monospace;color:#555;">'
        f'<b style="color:#856404">SSH session detected</b> &mdash; '
        f'forward the port in a new terminal to view this page locally:<br>'
        f'<code style="background:#f8f8f8;padding:2px 6px;border-radius:3px">{cmd}</code>'
        f'</div>'
    )


def _scan_html_files(probe_dir: Path | None) -> list[dict]:
    """Return sorted list of {rel, label} dicts for browsable HTML files under probe_dir.

    Only includes ``experts.html`` (probing) and ``top_experts.html`` (newline).
    """
    if probe_dir is None or not probe_dir.exists():
        return []
    entries = []
    for p in sorted(probe_dir.rglob("*.html")):
        if p.name not in ("experts.html", "top_experts.html"):
            continue
        rel = str(p.relative_to(probe_dir))
        # Exclude old/ directories
        if "/old/" in rel or rel.startswith("old/"):
            continue
        # Derive a display label:
        #   experts.html  → parent folder (e.g. "hours", "colors")
        #   top_experts.html → first "newline_*" ancestor (e.g. "newline_150")
        if p.name == "top_experts.html":
            label = None
            for part in p.relative_to(probe_dir).parts:
                if part.startswith("newline"):
                    label = part
                    break
            if label is None:
                label = p.parent.name
        else:
            label = p.parent.name
        entries.append({"rel": rel, "label": label})
    return entries


def _gallery_html(output_dir: Path, port: int, probe_dir: Path | None = None) -> str:
    ssh_banner = _ssh_banner_html(port)
    has_nav = probe_dir is not None and probe_dir.exists()
    layout_style = "display:flex;flex-direction:row;height:calc(100vh - 56px);overflow:hidden;" if has_nav else ""
    nav_style = "" if has_nav else "display:none"
    main_style = "flex:1;overflow-y:auto;" if has_nav else ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Camera-Ready Queue</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5; color: #333; height: 100vh; display: flex;
            flex-direction: column; }}
    header {{ background: #1a1a2e; color: #eee; padding: 0 24px; height: 56px;
              display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
    header h1 {{ font-size: 18px; font-weight: 600; flex: 1; }}
    .dir-label {{ font-size: 12px; color: #aaa; }}
    #save-all-btn {{ padding: 8px 20px; background: #4caf50; color: #fff;
                     border: none; border-radius: 4px; font-size: 14px;
                     cursor: pointer; font-weight: 600; }}
    #save-all-btn:disabled {{ background: #999; cursor: default; }}
    #save-all-btn:hover:not(:disabled) {{ background: #388e3c; }}
    #clear-btn {{ padding: 8px 16px; background: transparent; color: #ccc;
                  border: 1px solid #555; border-radius: 4px; font-size: 13px;
                  cursor: pointer; }}
    #clear-btn:hover {{ border-color: #e57373; color: #e57373; }}
    #body-wrap {{ {layout_style} flex: 1; overflow: hidden; }}
    #nav {{ width: 240px; background: #fff; border-right: 1px solid #e0e0e0;
            overflow-y: auto; flex-shrink: 0; padding: 12px 0; {nav_style} }}
    #nav-search {{ width: calc(100% - 16px); margin: 0 8px 10px;
                   padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px;
                   font-size: 12px; }}
    .nav-group {{ font-size: 10px; font-weight: 700; color: #999; letter-spacing: .06em;
                  text-transform: uppercase; padding: 8px 12px 2px; }}
    .nav-item {{ display: block; padding: 6px 12px; font-size: 12px; color: #333;
                 text-decoration: none; border-left: 3px solid transparent;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .nav-item:hover {{ background: #f0f0f0; border-left-color: #1a1a2e; }}
    .nav-item.active {{ background: #eef2ff; border-left-color: #4c6ef5;
                        color: #4c6ef5; font-weight: 600; }}
    #main {{ {main_style} }}
    #status-bar {{ padding: 8px 24px; font-size: 13px; color: #555;
                   background: #fff; border-bottom: 1px solid #e0e0e0; }}
    #grid {{ display: grid;
             grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 16px; padding: 24px; }}
    .card {{ background: #fff; border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 4px rgba(0,0,0,0.12);
             display: flex; flex-direction: column; }}
    .card img {{ width: 100%; aspect-ratio: 1.3; object-fit: contain;
                 background: #fafafa; border-bottom: 1px solid #eee; }}
    .card-body {{ padding: 10px 12px; flex: 1; }}
    .card-title {{ font-size: 11px; word-break: break-all; color: #444;
                   line-height: 1.4; margin-bottom: 6px; }}
    .meta {{ font-size: 11px; color: #888; }}
    .card-footer {{ padding: 8px 12px; display: flex; justify-content: flex-end;
                    border-top: 1px solid #f0f0f0; }}
    .remove-btn {{ font-size: 12px; color: #c0392b; cursor: pointer;
                   background: none; border: none; padding: 2px 6px; }}
    .remove-btn:hover {{ text-decoration: underline; }}
    #empty-msg {{ text-align: center; padding: 60px 24px; color: #999;
                  font-size: 15px; }}
    .toast {{ position: fixed; bottom: 24px; right: 24px; background: #323232;
              color: #fff; padding: 10px 18px; border-radius: 4px; font-size: 13px;
              opacity: 0; transition: opacity 0.3s; pointer-events: none; }}
    .toast.show {{ opacity: 1; }}
  </style>
</head>
<body>
  <header>
    <h1>Camera-Ready Queue</h1>
    <div class="dir-label">→ {output_dir.resolve()}</div>
    <button id="clear-btn" onclick="clearQueue()">Clear all</button>
    <button id="save-all-btn" onclick="saveAll()">Save All (0)</button>
  </header>
  {ssh_banner}
  <div id="body-wrap">
    <nav id="nav">
      <input id="nav-search" type="search" placeholder="Filter…" oninput="filterNav(this.value)">
      <div id="nav-list"></div>
    </nav>
    <div id="main">
      <div id="status-bar">Loading…</div>
      <div id="grid"></div>
      <div id="empty-msg" style="display:none">
        No figures queued yet.<br>
        Open an <b>experts.html</b> page from the left panel and click
        <b>Save scatter</b> or <b>Save means</b>.
      </div>
    </div>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const PORT = {port};
    const BASE = 'http://127.0.0.1:' + PORT;

    // ── Queue / gallery ────────────────────────────────────────────────────
    async function refresh() {{
      let data;
      try {{
        const r = await fetch(BASE + '/queue');
        data = await r.json();
      }} catch {{ return; }}
      renderGrid(data.items);
      document.getElementById('save-all-btn').textContent = 'Save All (' + data.items.length + ')';
      document.getElementById('save-all-btn').disabled = data.items.length === 0;
      document.getElementById('status-bar').textContent =
        data.items.length + ' figure(s) queued';
      document.getElementById('empty-msg').style.display = data.items.length ? 'none' : 'block';
      document.getElementById('grid').style.display = data.items.length ? 'grid' : 'none';
    }}

    function parseFilename(fn) {{
      // {{experiment_id}}__{{task}}__E{{id}}__{{hyp}}__{{score_type}}__{{score}}__{{figure_type}}.png
      const m = fn.match(/^(.+?)__(.+?)__E(\\d+)__(.+?)__([^_]+)__([0-9.]+)__(scatter|means)\\.png$/);
      if (!m) return null;
      return {{ exp: m[1], task: m[2], expert: m[3], hyp: m[4],
                scoreType: m[5], score: m[6], type: m[7] }};
    }}

    function renderGrid(items) {{
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      items.forEach(fn => {{
        const p = parseFilename(fn);
        const meta = p
          ? `<div class="meta">${{p.task}} · E${{p.expert}} · ${{p.hyp}} · ${{p.scoreType}}=${{p.score}}</div>`
          : '';
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <img src="${{BASE}}/thumbnail/${{encodeURIComponent(fn)}}" loading="lazy">
          <div class="card-body">
            <div class="card-title">${{fn}}</div>
            ${{meta}}
          </div>
          <div class="card-footer">
            <button class="remove-btn" onclick="remove('${{fn}}')">Remove</button>
          </div>`;
        grid.appendChild(card);
      }});
    }}

    async function remove(fn) {{
      await fetch(BASE + '/queue/' + encodeURIComponent(fn), {{method: 'DELETE'}});
      refresh();
    }}

    async function clearQueue() {{
      if (!confirm('Clear all queued figures?')) return;
      await fetch(BASE + '/queue', {{method: 'DELETE'}});
      refresh();
    }}

    async function saveAll() {{
      const btn = document.getElementById('save-all-btn');
      btn.disabled = true;
      btn.textContent = 'Saving…';
      const r = await fetch(BASE + '/save-all', {{method: 'POST'}});
      const data = await r.json();
      showToast('Saved ' + data.saved + ' file(s) to disk');
      refresh();
    }}

    // ── Navigation panel ───────────────────────────────────────────────────
    let _navItems = [];

    async function loadNav() {{
      try {{
        const r = await fetch(BASE + '/html-files');
        const data = await r.json();
        _navItems = data.files || [];
      }} catch {{ _navItems = []; }}
      renderNav(_navItems);
    }}

    function renderNav(items) {{
      const list = document.getElementById('nav-list');
      if (!items.length) {{
        list.innerHTML = '<div style="padding:8px 12px;font-size:12px;color:#999">No experts.html files found</div>';
        return;
      }}
      // Group by first path component (experiment), then by second (probe / newline_*)
      const groups = {{}};
      items.forEach(item => {{
        const parts = item.rel.split('/');
        const grp = parts.length > 1 ? parts[0] : '';
        const sub = parts.length > 2 ? parts[1] : '';
        const key = sub ? grp + ' / ' + sub : grp;
        if (!groups[key]) groups[key] = [];
        groups[key].push(item);
      }});
      let html = '';
      for (const [grp, grpItems] of Object.entries(groups)) {{
        html += `<div class="nav-group">${{grp}}</div>`;
        grpItems.forEach(item => {{
          html += `<a class="nav-item" href="${{BASE}}/view?p=${{encodeURIComponent(item.rel)}}"
                      target="expert-frame" title="${{item.rel}}">${{item.label}}</a>`;
        }});
      }}
      list.innerHTML = html;
    }}

    function filterNav(q) {{
      const filtered = q
        ? _navItems.filter(i => i.label.toLowerCase().includes(q.toLowerCase()) ||
                                i.rel.toLowerCase().includes(q.toLowerCase()))
        : _navItems;
      renderNav(filtered);
    }}

    function showToast(msg) {{
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 3000);
    }}

    loadNav();
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""


# ── Request handler ────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    server: "_SaveServer"  # type: ignore[assignment]

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, unquote, urlparse
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "":
            body = _gallery_html(
                self.server.output_dir, self.server.port, self.server.probe_dir
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif path == "/ping":
            self._json(200, {"status": "ok", "queued": len(_queue)})

        elif path == "/queue":
            self._json(200, {"items": list(_queue.keys()), "count": len(_queue)})

        elif path == "/html-files":
            files = _scan_html_files(self.server.probe_dir)
            self._json(200, {"files": files})

        elif path == "/view":
            probe_dir = self.server.probe_dir
            rel = unquote(qs.get("p", [""])[0])
            if not rel or probe_dir is None:
                self._json(400, {"error": "missing p parameter or no probe-dir configured"})
                return
            target = (probe_dir / rel).resolve()
            # Safety: only serve files under probe_dir
            if not str(target).startswith(str(probe_dir.resolve())):
                self._json(403, {"error": "path escapes probe-dir"})
                return
            if not target.exists() or target.suffix != ".html":
                self._json(404, {"error": "file not found"})
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif path.startswith("/thumbnail/"):
            fn = path[len("/thumbnail/"):]
            if fn in _queue:
                data = _queue[fn]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not in queue"})

        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/queue":
            _queue.clear()
            self._json(200, {"cleared": True})
        elif path.startswith("/queue/"):
            fn = path[len("/queue/"):]
            removed = _queue.pop(fn, None) is not None
            self._json(200, {"removed": removed})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        body_bytes = self.rfile.read(int(self.headers.get("Content-Length", 0)))

        if path == "/queue":
            try:
                body = json.loads(body_bytes)
                fn = Path(body["filename"]).name
                if not fn.endswith(".png"):
                    fn += ".png"
                data = base64.b64decode(body["data"])
                data = _autocrop_png(data)
                _queue[fn] = data
                typer.echo(f"  Queued  [{len(_queue):3d}]  {fn}")
                self._json(200, {"queued": fn, "total": len(_queue)})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        elif path == "/save-all":
            saved: list[str] = []
            for fn, data in list(_queue.items()):
                out = self.server.output_dir / fn
                out.write_bytes(data)
                saved.append(str(out))
                typer.echo(f"  Saved   {out}")
            _queue.clear()
            typer.echo(f"  → {len(saved)} file(s) written to {self.server.output_dir}")
            self._json(200, {"saved": len(saved), "paths": saved})

        else:
            self._json(404, {"error": "not found"})

    # ── Helpers ──────────────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        pass  # suppress per-request noise


class _SaveServer(HTTPServer):
    def __init__(self, output_dir: Path, port: int, results_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.port = port
        self.probe_dir = results_dir  # used by handler as the root to scan/serve HTML from
        super().__init__(("127.0.0.1", port), _Handler)


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def save_server(
    output_dir: Path = typer.Option(..., help="Directory where PNGs will be written on Save All"),
    results_dir: Path = typer.Option(None, help="Root results directory to scan for experts.html files (shown in left nav; scans recursively)"),
    port: int = typer.Option(_DEFAULT_PORT, help="Local port to listen on (default 7788)"),
) -> None:
    """Start the camera-ready save server.

    1. Run this command, optionally passing --results-dir to enable the file browser.
    2. Open http://127.0.0.1:{port} — the left panel lists all experts.html pages found
       anywhere under --results-dir.
    3. Click a page in the nav to open it; click Save scatter / Save means to queue figures.
    4. The floating badge on each HTML page shows the current queue count.
    5. Click Save All in the gallery to write every queued PNG to disk at once.

    Stop with Ctrl-C when done.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    server = _SaveServer(output_dir, port, results_dir=results_dir)
    typer.echo(f"Save server  →  http://127.0.0.1:{port}/")
    typer.echo(f"Output dir   →  {output_dir.resolve()}")
    if results_dir:
        n = len(_scan_html_files(results_dir))
        typer.echo(f"Results dir  →  {results_dir.resolve()}  ({n} experts.html found)")
    hint = _ssh_forward_hint(port)
    if hint:
        typer.echo(hint)
    typer.echo("Waiting for figures…  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        typer.echo(f"\nServer stopped.  {len(_queue)} unsaved figure(s) in queue.")
