/**
 * save_client.js — PNG queue logic for the camera-ready save workflow.
 *
 * Captures the Plotly figure by rendering a cloned figure object off-screen
 * (never modifying the live div), so the graph never shifts.  The clone has
 * the colorbar and legend stripped but preserves the current margin and camera
 * so the 3D projection matches exactly what the user sees.
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

  const SCALE = 2;

  // Capture at the element's actual display dimensions so the 3D projection
  // matches what the user sees and annotations land at the correct positions.
  const divRect = gd.getBoundingClientRect();
  const captureW = Math.round(divRect.width)  || 1100;
  const captureH = Math.round(divRect.height) || 850;

  // Snapshot annotation positions before anything else (scatter only)
  const annSnapshots = [];
  if (divId === 'active-scatter') {
    const wrapper = document.getElementById('scatter-wrapper');
    const overlay = document.getElementById('annotation-overlay');
    if (wrapper && overlay) {
      const wr = wrapper.getBoundingClientRect();
      for (const ann of overlay.querySelectorAll('.annotation-label')) {
        const textEl = ann.querySelector('.annotation-text');
        const text = textEl ? textEl.textContent.trim() : '';
        if (!text) continue;
        const ar = ann.getBoundingClientRect();
        annSnapshots.push({
          x: (ar.left - wr.left) * SCALE,
          y: (ar.top  - wr.top)  * SCALE,
          w: ar.width  * SCALE,
          h: ar.height * SCALE,
          text,
        });
      }
    }
  }

  // Build a clean figure object: strip colorbar/legend but keep the current
  // margin and camera so the off-screen render matches the live view exactly.
  // Passing a figure object (not the div) means the DOM is never touched.
  const fl = gd._fullLayout || {};
  const cleanData = gd.data.map(t => {
    const tc = Object.assign({}, t);
    tc.showlegend = false;
    if (tc.marker) tc.marker = Object.assign({}, tc.marker, { showscale: false });
    return tc;
  });
  const captureLayout = Object.assign({}, gd.layout || {}, {
    showlegend: false,
    title: { text: '' },
    // Preserve the resolved margin so the scene geometry stays identical
    margin: fl.margin,
    // Preserve the current camera angle (updated as the user rotates)
    scene: Object.assign({}, (gd.layout || {}).scene || {}, {
      camera: (fl.scene || {}).camera,
    }),
  });

  const imgData = await Plotly.toImage(
    { data: cleanData, layout: captureLayout },
    { format: 'png', width: captureW, height: captureH, scale: SCALE }
  );

  // Composite annotation boxes onto the PNG at exact scaled positions
  let base64;
  if (annSnapshots.length > 0) {
    const IW = captureW * SCALE;
    const IH = captureH * SCALE;
    const canvas = document.createElement('canvas');
    canvas.width = IW;
    canvas.height = IH;
    const ctx = canvas.getContext('2d');

    const img = new Image();
    await new Promise(res => { img.onload = res; img.src = imgData; });
    ctx.drawImage(img, 0, 0);

    const handleH = 16 * SCALE;
    for (const { x, y, w, h, text } of annSnapshots) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 2 * SCALE;
      ctx.strokeRect(x, y, w, h);
      const fs = Math.round(14 * SCALE);
      ctx.fillStyle = '#000000';
      ctx.font = `${fs}px sans-serif`;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x + 9 * SCALE, y + handleH + (h - handleH) / 2);
    }
    base64 = canvas.toDataURL('image/png').split(',')[1];
  } else {
    base64 = imgData.split(',')[1];
  }

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
    link.href = 'data:image/png;base64,' + base64;
    link.download = filename;
    link.click();
    showToast('Downloaded (save-server not running)');
  }
}
