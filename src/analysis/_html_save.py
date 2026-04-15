"""Client-side JavaScript for the camera-ready save workflow.

Injected into every experts.html via the ``__SAVEJS__`` placeholder in
``utils.py``.  Must be a plain string (no Python format escaping) so it can
be spliced in with a simple ``.replace()``.
"""

#: Inline ``<script>`` block injected into every experts.html.
#: Provides the queue-based save workflow:
#:   - Polls ``localhost:SAVE_PORT/ping`` on page load and every 3 s.
#:   - Shows a floating badge with the current queue count.
#:   - ``_queueFigure(divId, filename)`` — POSTs to the save server; falls back
#:     to a browser download if the server is unreachable.
SAVE_CLIENT_JS: str = """\
<script id="save-server-js">
  const SAVE_PORT = 7788;
  let _queueCount = 0;

  (function () {
    // Floating badge linking to the gallery
    const badge = document.createElement('a');
    badge.id = 'queue-badge';
    badge.href = 'http://127.0.0.1:' + SAVE_PORT + '/';
    badge.target = '_blank';
    badge.style.cssText =
      'position:fixed;top:12px;right:12px;z-index:9999;' +
      'background:#1a1a2e;color:#eee;padding:6px 14px;border-radius:20px;' +
      'font-size:12px;font-family:sans-serif;text-decoration:none;display:none;' +
      'box-shadow:0 2px 8px rgba(0,0,0,.35);white-space:nowrap;';
    document.body.appendChild(badge);

    // Toast notification
    const toast = document.createElement('div');
    toast.id = 'save-toast';
    toast.style.cssText =
      'position:fixed;bottom:20px;right:20px;z-index:9999;' +
      'background:#323232;color:#fff;padding:9px 16px;border-radius:4px;' +
      'font-size:13px;font-family:sans-serif;opacity:0;transition:opacity 0.3s;' +
      'pointer-events:none;max-width:360px;';
    document.body.appendChild(toast);

    async function pollBadge() {
      try {
        const r = await fetch(
          'http://127.0.0.1:' + SAVE_PORT + '/ping',
          { signal: AbortSignal.timeout(800) }
        );
        if (r.ok) {
          const d = await r.json();
          _queueCount = d.queued || 0;
          badge.textContent = _queueCount + ' queued \u2192 Gallery';
          badge.style.display = 'block';
        }
      } catch (_) {}
    }
    pollBadge();
    setInterval(pollBadge, 3000);
  })();

  function _showToast(msg) {
    const t = document.getElementById('save-toast');
    t.textContent = msg;
    t.style.opacity = '1';
    setTimeout(() => { t.style.opacity = '0'; }, 3000);
  }

  async function _queueFigure(divId, filename) {
    const gd = document.getElementById(divId);
    // Snapshot current per-trace showlegend / showscale so we can restore them
    const traceState = gd.data.map(t => ({
      showlegend: t.showlegend,
      marker: { showscale: t.marker ? t.marker.showscale : undefined },
    }));
    const layout = gd._fullLayout || {};
    const savedTitle = layout.title ? (layout.title.text !== undefined ? layout.title.text : layout.title) : '';
    const savedMargin = Object.assign({}, layout.margin || {});

    // Hide everything that isn't the 3-D scatter
    Plotly.relayout(divId, {
      showlegend: false,
      title: { text: '' },
      margin: { t: 0, b: 0, l: 0, r: 0 },
    });
    gd.data.forEach(t => {
      t.showlegend = false;
      if (t.marker) t.marker.showscale = false;
    });
    Plotly.redraw(divId);

    let imgData;
    try {
      imgData = await Plotly.toImage(
        divId, { format: 'png', width: 1100, height: 850, scale: 2 }
      );
    } finally {
      // Restore
      gd.data.forEach((t, i) => {
        t.showlegend = traceState[i].showlegend;
        if (t.marker) t.marker.showscale = traceState[i].marker.showscale;
      });
      Plotly.relayout(divId, {
        showlegend: true,
        title: { text: savedTitle },
        margin: savedMargin,
      });
    }
    const base64 = imgData.split(',')[1];
    try {
      const r = await fetch('http://127.0.0.1:' + SAVE_PORT + '/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename + '.png', data: base64 })
      });
      const d = await r.json();
      _queueCount = d.total || _queueCount + 1;
      const badge = document.getElementById('queue-badge');
      badge.textContent = _queueCount + ' queued \u2192 Gallery';
      badge.style.display = 'block';
      _showToast('\u2713 Queued (' + _queueCount + ' total)');
    } catch (_) {
      // Server not running — fall back to a browser download
      const link = document.createElement('a');
      link.href = imgData;
      link.download = filename + '.png';
      link.click();
      _showToast('Downloaded (save-server not running)');
    }
  }
</script>"""
