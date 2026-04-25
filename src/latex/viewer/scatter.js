/**
 * scatter.js — Plotly 3D scatter trace builder for the SMIXAE viewer.
 *
 * Consumes per-expert data (points + labels) and a color spec from the task
 * index, and builds Plotly traces that mirror the Python scatter3d logic.
 */

const Scatter = (() => {

  // ── Colorscale sampling ────────────────────────────────────────────────
  // Cache of {name}_{n} → [rgb_string, ...] fetched from /colorscale endpoint.
  const _colorCache = {};

  async function fetchColorscale(name, n, skipEndpoints = true) {
    const key = `${name}_${n}_${skipEndpoints}`;
    if (_colorCache[key]) return _colorCache[key];
    const r = await fetch(
      `/colorscale?name=${encodeURIComponent(name)}&n=${n}&skip_endpoints=${skipEndpoints}`
    );
    const data = await r.json();
    _colorCache[key] = data.colors;
    return data.colors;
  }

  // ── Class mean computation ─────────────────────────────────────────────
  function classMeans(points, labels) {
    const byClass = {};
    for (let i = 0; i < labels.length; i++) {
      const c = labels[i];
      if (!byClass[c]) byClass[c] = { xs: [], ys: [], zs: [], n: 0 };
      byClass[c].xs.push(points[i * 3]);
      byClass[c].ys.push(points[i * 3 + 1]);
      byClass[c].zs.push(points[i * 3 + 2]);
      byClass[c].n++;
    }
    const means = {};
    for (const c in byClass) {
      const b = byClass[c];
      means[c] = [
        b.xs.reduce((a, v) => a + v, 0) / b.n,
        b.ys.reduce((a, v) => a + v, 0) / b.n,
        b.zs.reduce((a, v) => a + v, 0) / b.n,
      ];
    }
    return means;
  }

  // ── Build traces ───────────────────────────────────────────────────────

  /**
   * Build Plotly traces + layout for a single expert's scatter plot.
   *
   * @param {Float32Array} points - Flat (n*3) bottleneck coordinates.
   * @param {Int32Array|null} labels - Per-point class ids, or null for unlabeled.
   * @param {Float32Array|null} continuity - Per-point continuity scores, or null.
   * @param {object} colorSpec - {mode, scale, color_map, continuous_label} from task index.
   * @param {object|null} labelNames - {id: name} mapping or null.
   * @param {object} opts - {title, scatterSize, showMeans}
   * @returns {Promise<{data: Array, layout: object}>}
   */
  async function buildScatter(points, labels, continuity, colorSpec, labelNames, opts = {}) {
    const n = points.length / 3;
    const x = new Float32Array(points.buffer, 0, n * 4);
    // Actually points is already flat — just slice.
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < n; i++) {
      xs.push(points[i * 3]);
      ys.push(points[i * 3 + 1]);
      zs.push(points[i * 3 + 2]);
    }

    const traces = [];
    const size = opts.scatterSize || 1;
    const isDiscrete = colorSpec.mode === 'discrete' && labels;

    if (isDiscrete) {
      // Discrete color: one trace per class, using fetched colorscale colors.
      const uniqueClasses = [...new Set(Array.from(labels))];
      const nClasses = uniqueClasses.length;
      let colorMap = colorSpec.color_map || null;

      if (!colorMap) {
        // Fetch colors from server
        const rgbs = await fetchColorscale(colorSpec.scale || 'Plasma', nClasses);
        colorMap = {};
        uniqueClasses.forEach((c, i) => {
          const name = labelNames ? labelNames[c] : String(c);
          colorMap[name] = rgbs[i];
        });
      }

      const legendNames = labelNames || {};
      for (const c of uniqueClasses) {
        const cx = [], cy = [], cz = [];
        for (let i = 0; i < n; i++) {
          if (labels[i] === c) { cx.push(xs[i]); cy.push(ys[i]); cz.push(zs[i]); }
        }
        const displayName = legendNames[c] || String(c);
        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: cx, y: cy, z: cz,
          marker: { size: size, color: colorMap[displayName] || '#888' },
          name: displayName,
        });
      }

      // Class mean spheres
      if (opts.showMeans !== false) {
        const means = classMeans(xs.map((_, i) => [xs[i], ys[i], zs[i]]).flat(),
                                 Array.from(labels));
        // Recompute as flat arrays
        const flatPts = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          flatPts[i*3] = xs[i]; flatPts[i*3+1] = ys[i]; flatPts[i*3+2] = zs[i];
        }
        const m2 = classMeans(flatPts, labels);
        const mx = [], my = [], mz = [], mc = [];
        for (const c in m2) {
          mx.push(m2[c][0]); my.push(m2[c][1]); mz.push(m2[c][2]);
          const name = legendNames[c] || String(c);
          mc.push(colorMap[name] || '#888');
        }
        traces.push({
          type: 'scatter3d', mode: 'markers+text',
          x: mx, y: my, z: mz,
          marker: { size: 8, color: mc, opacity: 0.9 },
          text: mx.map((_, i) => legendNames[Object.keys(m2)[i]] || ''),
          textposition: 'top center',
          showlegend: false,
        });
      }

    } else {
      // Continuous color: colorbar from labels or continuity.
      let colorArr;
      let cbarTitle = colorSpec.continuous_label || '';
      if (labels) {
        colorArr = Array.from(labels);
      } else if (continuity) {
        colorArr = Array.from(continuity);
      } else {
        // Distance from origin
        colorArr = [];
        for (let i = 0; i < n; i++) {
          colorArr.push(Math.sqrt(xs[i]*xs[i] + ys[i]*ys[i] + zs[i]*zs[i]));
        }
        cbarTitle = 'Distance from origin';
      }

      traces.push({
        type: 'scatter3d', mode: 'markers',
        x: xs, y: ys, z: zs,
        marker: {
          size: size,
          color: colorArr,
          colorscale: colorSpec.scale || 'Viridis',
          colorbar: { title: cbarTitle },
          opacity: 0.8,
        },
        showlegend: false,
      });
    }

    const layout = {
      title: opts.title || '',
      margin: { l: 0, r: 0, t: 30, b: 0 },
      scene: {
        xaxis: { showticklabels: false, title: '' },
        yaxis: { showticklabels: false, title: '' },
        zaxis: { showticklabels: false, title: '' },
      },
    };

    return { data: traces, layout };
  }

  return { buildScatter, classMeans, fetchColorscale };
})();
