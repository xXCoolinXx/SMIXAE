/**
 * save_client.js — PNG queue logic for the camera-ready save workflow.
 *
 * Mirrors the old _html_save.py SAVE_CLIENT_JS but works as a standalone JS file.
 * Polls the save server, shows the queue badge, and handles "Save scatter/means"
 * buttons by capturing Plotly figures as PNG and POSTing them to /queue.
 */

const SAVE_PORT = 7788;
let _queueCount = 0;

(function initBadge() {
  const badge = document.getElementById('queue-badge');
  const link = document.getElementById('badge-link');
  if (!badge || !link) return;
  link.href = `http://127.0.0.1:${SAVE_PORT}/`;

  async function poll() {
    try {
      const r = await fetch(`http://127.0.0.1:${SAVE_PORT}/ping`,
                            { signal: AbortSignal.timeout(800) });
      if (r.ok) {
        const d = await r.json();
        _queueCount = d.queued || 0;
        link.textContent = `${_queueCount} queued → Gallery`;
        badge.style.display = 'inline';
      }
    } catch (_) { /* server not running */ }
  }
  poll();
  setInterval(poll, 3000);
})();

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

async function saveFigurePNG(divId, suffix) {
  const gd = document.getElementById(divId);
  if (!gd || !gd.data) return;

  // Snapshot current legend/scale state so we can restore
  const traceState = gd.data.map(t => ({
    showlegend: t.showlegend,
    marker: { showscale: t.marker ? t.marker.showscale : undefined },
  }));
  const layout = gd._fullLayout || {};
  const savedTitle = layout.title ? (layout.title.text !== undefined ? layout.title.text : layout.title) : '';
  const savedMargin = Object.assign({}, layout.margin || {});

  // Hide everything that isn't the 3D scatter
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

  // Build filename using the current expert's metadata
  const m = window._currentExpertMeta;
  if (!m) return;
  const scoreLabel = m.score_type || 'score';
  const score = m.hyp_score != null ? m.hyp_score.toFixed(4) : 'NA';
  const filename = [
    window._experimentId || '',
    window._datasetName || '',
    'E' + m.expert_id,
    m.hyp_name || '',
    scoreLabel,
    score,
    suffix,
  ].join('__') + '.png';

  try {
    const r = await fetch(`http://127.0.0.1:${SAVE_PORT}/queue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, data: base64 }),
    });
    const d = await r.json();
    _queueCount = d.total || _queueCount + 1;
    const link = document.getElementById('badge-link');
    if (link) link.textContent = `${_queueCount} queued → Gallery`;
    document.getElementById('queue-badge').style.display = 'inline';
    showToast(`✓ Queued (${_queueCount} total)`);
  } catch (_) {
    // Server not running — fall back to browser download
    const link = document.createElement('a');
    link.href = imgData;
    link.download = filename;
    link.click();
    showToast('Downloaded (save-server not running)');
  }
}
