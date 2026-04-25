/**
 * viewer.js — Main controller for the SMIXAE Expert Viewer.
 *
 * Loads the task index, builds the navigation and tab strips, fetches expert
 * data on demand, and renders Plotly figures via scatter.js.
 */

const Viewer = (() => {
  let taskIndex = null;
  let taskRelPath = '';
  let currentView = null;
  let currentExpertId = null;

  // ── Task loading ───────────────────────────────────────────────────────
  async function loadTask(relPath) {
    taskRelPath = relPath;
    const r = await fetch(`/task?p=${encodeURIComponent(relPath)}`);
    taskIndex = await r.json();

    document.getElementById('task-title').textContent =
      taskIndex.title || taskIndex.dataset_name || relPath;

    // Expose experiment metadata for save_client.js filename generation
    window._experimentId = taskIndex.experiment_id || '';
    window._datasetName = taskIndex.dataset_name || '';

    buildNav();
  }

  // ── Navigation (one section per view in experts_by_view) ──────────────
  function buildNav() {
    const nav = document.getElementById('view-list');
    nav.innerHTML = '';
    const views = taskIndex.experts_by_view || {};
    const firstView = Object.keys(views)[0];

    for (const viewName in views) {
      const rankings = views[viewName];
      const group = document.createElement('div');
      group.className = 'view-group';
      group.textContent = viewName;
      nav.appendChild(group);

      rankings.forEach((r, i) => {
        const btn = document.createElement('button');
        btn.className = 'view-btn';
        btn.textContent = `#${r.rank} E${r.expert_id}${r.score != null ? ` (${r.score.toFixed(3)})` : ''}`;
        btn.onclick = () => selectExpert(viewName, r.expert_id, i, rankings.length);
        if (viewName === firstView && i === 0) {
          btn.classList.add('active');
        }
        nav.appendChild(btn);
      });
    }

    // Auto-select first expert of first view
    if (firstView && views[firstView].length > 0) {
      selectExpert(firstView, views[firstView][0].expert_id, 0, views[firstView].length);
    }
  }

  // ── Expert selection ──────────────────────────────────────────────────
  async function selectExpert(viewName, expertId, rank, total) {
    currentView = viewName;
    currentExpertId = expertId;

    // Update active states in nav
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    // Find and activate the right button
    document.querySelectorAll('.view-btn').forEach(b => {
      if (b.textContent.includes(`E${expertId}`) && b.textContent.startsWith(`#${rank + 1}`)) {
        b.classList.add('active');
      }
    });

    // Fetch expert metadata
    const metaR = await fetch(`/expert?p=${encodeURIComponent(taskRelPath)}&id=${expertId}`);
    const meta = await metaR.json();

    // Build tab strip (single-tab for now; could extend to per-hypothesis)
    const strip = document.getElementById('tab-strip');
    strip.innerHTML = '';
    const rankings = taskIndex.experts_by_view[viewName] || [];
    rankings.forEach((r, i) => {
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (r.expert_id === expertId ? ' active' : '');
      btn.textContent = `E${r.expert_id}`;
      btn.onclick = () => selectExpert(viewName, r.expert_id, i, rankings.length);
      strip.appendChild(btn);
    });

    // Expert header
    const parts = [`<b>Expert ${expertId}</b>`];
    const ranking = rankings.find(r => r.expert_id === expertId);
    if (ranking && ranking.score != null) parts.push(`${viewName} = ${ranking.score.toFixed(3)}`);
    if (meta.metrics.fisher_score != null) parts.push(`Fisher = ${meta.metrics.fisher_score.toFixed(2)}`);
    if (meta.metrics.mean_continuity != null) parts.push(`Cont = ${meta.metrics.mean_continuity.toFixed(3)}`);
    parts.push(`${meta.n_points} points`);
    document.getElementById('expert-header').innerHTML = parts.join(' &nbsp;|&nbsp; ');

    // Show save buttons
    document.getElementById('save-bar').style.display = 'flex';
    document.getElementById('save-means-btn').style.display =
      meta.tensor_keys.includes('labels') ? 'inline-block' : 'none';

    // Set metadata for save_client.js
    const scoreType = (taskIndex.hypotheses && taskIndex.hypotheses.length > 0)
      ? (taskIndex.hypotheses.find(h => h.name === viewName)?.score_type || 'score')
      : (viewName === 'random' ? 'cont' : 'score');
    window._currentExpertMeta = {
      expert_id: expertId,
      hyp_name: viewName,
      hyp_score: ranking?.score ?? null,
      score_type: scoreType,
    };

    // Regression scores table
    const regDiv = document.getElementById('reg-scores-container');
    const regScores = meta.metrics.regression_scores;
    if (regScores && Object.keys(regScores).length > 0) {
      const rows = Object.entries(regScores)
        .sort((a, b) => (b[1].mean || 0) - (a[1].mean || 0))
        .map(([k, v]) =>
          `<tr><td style="padding:2px 8px;border:1px solid #ccc">${k}</td>` +
          `<td style="padding:2px 8px;border:1px solid #ccc">${(v.mean != null ? v.mean.toFixed(4) : 'nan')}</td></tr>`
        ).join('');
      regDiv.innerHTML =
        `<details style="margin-bottom:8px;font-size:12px">` +
        `<summary style="cursor:pointer">All regression scores</summary>` +
        `<table style="border-collapse:collapse;margin-top:4px">` +
        `<thead><tr><th style="padding:2px 8px;border:1px solid #ccc">Hypothesis</th>` +
        `<th style="padding:2px 8px;border:1px solid #ccc">Score</th></tr></thead>` +
        `<tbody>${rows}</tbody></table></details>`;
    } else {
      regDiv.innerHTML = '';
    }

    // Fetch tensors and render
    const tensorR = await fetch(`/expert-tensors?p=${encodeURIComponent(taskRelPath)}&id=${expertId}`);
    const tensorBuf = await tensorR.arrayBuffer();

    const nPts = meta.n_points;
    const floatView = new Float32Array(tensorBuf, 0, nPts * 3);
    let labels = null;
    let continuity = null;
    let offset = nPts * 3 * 4;

    if (meta.tensor_keys.includes('labels')) {
      labels = new Int32Array(tensorBuf, offset, nPts);
      offset += nPts * 4;
    }
    if (meta.tensor_keys.includes('continuity')) {
      continuity = new Float32Array(tensorBuf, offset, nPts);
    }

    // hover_text is per-point; pass null if absent or empty.
    const hoverText = (meta.hover_text && meta.hover_text.length > 0)
      ? meta.hover_text : null;

    const { data, layout } = await Scatter.buildScatter(
      floatView, labels, continuity,
      taskIndex.color, taskIndex.label_names,
      {
        title: `Expert ${expertId}`,
        scatterSize: taskIndex.scatter_size || 1,
        hoverText,
      }
    );

    Plotly.newPlot('active-scatter', data, layout, { responsive: true });

    // Means plot — class means only (no per-point scatter).
    // Shown whenever labels are present (discrete OR continuous-with-labels).
    if (labels !== null) {
      const { data: mData, layout: mLayout } = await Scatter.buildScatter(
        floatView, labels, null,
        taskIndex.color, taskIndex.label_names,
        {
          title: `Expert ${expertId} [class means]`,
          scatterSize: taskIndex.scatter_size || 1,
          meansOnly: true,
          // hover_text omitted for means view — class name is the hover text
        }
      );
      Plotly.newPlot('active-mean', mData, mLayout, { responsive: true });
      document.getElementById('active-mean').style.display = 'block';
    } else {
      document.getElementById('active-mean').style.display = 'none';
    }
  }

  // ── Init from URL params ──────────────────────────────────────────────
  async function init() {
    const params = new URLSearchParams(window.location.search);
    const rel = params.get('task');
    if (rel) {
      await loadTask(rel);
    }
  }

  return { loadTask, init };
})();

document.addEventListener('DOMContentLoaded', Viewer.init);
