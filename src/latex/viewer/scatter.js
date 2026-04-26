/**
 * scatter.js — Plotly 3D scatter trace builder for the SMIXAE viewer.
 *
 * Two independent decisions drive rendering:
 *
 *   isDiscrete = colorSpec.mode === 'discrete' && labelNames !== null && labels !== null
 *     true  → per-class coloring, swatch legend   (body_parts, weekdays, …)
 *     false → continuous colorscale, colorbar      (newline, temperature, continuity)
 *
 *   hasLabels = labels !== null
 *     true  → compute and render class means (regardless of isDiscrete)
 *     false → no means
 *
 * buildScatter opts
 * -----------------
 *   scatterSize    (default 1)      marker size for scatter points
 *   meansOnly      (default false)  skip scatter; render only means + origin
 *   connectMeans   (default false)  draw gradient line between adjacent means
 *   hoverText      (default null)   string[] of per-point hover text (scatter only)
 *
 * Colorscale handling
 * -------------------
 * Named colorscales are always resolved server-side (Python's plotly) and sent
 * as [[t, 'rgb(...)'], ...] arrays so the browser viewer matches the paper output
 * exactly, regardless of which scales Plotly.js recognises natively.
 */

const Scatter = (() => {

  // ── Colorscale cache ──────────────────────────────────────────────────
  const _colorCache = {};

  const _SERVER = window.location.hostname === '127.0.0.1' && window.location.port === '7788';

  // Fetch N discrete 'rgb(...)' strings from the server or a pre-generated static file.
  async function fetchColorscale(name, n, skipEndpoints = true) {
    const key = `${name}_${n}_${skipEndpoints ? 'True' : 'False'}`;
    if (_colorCache[key]) return _colorCache[key];
    const url = _SERVER
      ? `/colorscale?name=${encodeURIComponent(name)}&n=${n}&skip_endpoints=${skipEndpoints}`
      : `/browser/colorscales/${encodeURIComponent(name)}_${n}_${skipEndpoints ? 'True' : 'False'}.json`;
    const r = await fetch(url);
    const data = await r.json();
    _colorCache[key] = data.colors;
    return data.colors;
  }

  // Fetch a smooth [[t, 'rgb(...)'], ...] Plotly colorscale array.
  // skip_endpoints=false gives full 0→1 coverage for continuous colorbars.
  async function fetchColorscaleArray(name, n = 64) {
    const key = `_arr_${name}_${n}`;
    if (_colorCache[key]) return _colorCache[key];
    const url = _SERVER
      ? `/colorscale?name=${encodeURIComponent(name)}&n=${n}&skip_endpoints=false`
      : `/browser/colorscales/${encodeURIComponent(name)}_${n}_False.json`;
    const r = await fetch(url);
    const data = await r.json();
    const arr = data.colors.map((c, i) => [i / (n - 1), c]);
    _colorCache[key] = arr;
    return arr;
  }

  // ── Axis helper ────────────────────────────────────────────────────────
  function _axis() {
    return {
      title: '',
      showticklabels: false,
      showbackground: true,
      backgroundcolor: 'rgb(235,235,240)',
      showgrid: true,
      gridcolor: 'rgb(200,200,200)',
      gridwidth: 2,
      zeroline: true,
      zerolinecolor: 'rgb(150,150,150)',
      zerolinewidth: 2,
      showspikes: false,
    };
  }

  // ── Colorbar spec ──────────────────────────────────────────────────────
  function _colorbar(title) {
    return {
      title: { text: title || '', side: 'right', font: { size: 14 } },
      thickness: 30,
      len: 0.9,
      x: 1.02,
      outlinewidth: 1,
      outlinecolor: 'black',
      ticks: 'outside',
      ticklen: 6,
      tickwidth: 1,
      tickcolor: 'black',
      tickfont: { size: 13 },
    };
  }

  // ── Class mean computation ─────────────────────────────────────────────
  function classMeans(points, labels) {
    const byClass = {};
    for (let i = 0; i < labels.length; i++) {
      const c = labels[i];
      if (!byClass[c]) byClass[c] = { sx: 0, sy: 0, sz: 0, n: 0 };
      byClass[c].sx += points[i * 3];
      byClass[c].sy += points[i * 3 + 1];
      byClass[c].sz += points[i * 3 + 2];
      byClass[c].n++;
    }
    const means = {};
    for (const c in byClass) {
      const b = byClass[c];
      means[c] = [b.sx / b.n, b.sy / b.n, b.sz / b.n];
    }
    return means;
  }

  // ── Origin cross ──────────────────────────────────────────────────────
  function _originTrace() {
    return {
      type: 'scatter3d', mode: 'markers',
      x: [0], y: [0], z: [0],
      marker: {
        symbol: 'cross',
        size: 5,
        color: 'rgba(30,30,30,0.7)',
        line: { width: 1, color: 'rgba(30,30,30,0.7)' },
      },
      hoverinfo: 'skip',
      showlegend: false,
      name: 'origin',
    };
  }

  // ── Build traces ───────────────────────────────────────────────────────

  /**
   * @param {Float32Array} points          Flat (n*3) bottleneck coords.
   * @param {Int32Array|null} labels       Per-point class ids, or null.
   * @param {Float32Array|null} continuity Per-point continuity scores, or null.
   * @param {object} colorSpec             {mode, scale, color_map,
   *                                        continuous_label, skip_endpoints}
   * @param {object|null} labelNames       {id: name} or null.
   * @param {object} opts                  {title, scatterSize, meansOnly,
   *                                        connectMeans, hoverText}
   * @returns {Promise<{data: Array, layout: object}>}
   */
  async function buildScatter(points, labels, continuity, colorSpec, labelNames, opts = {}) {
    const n            = points.length / 3;
    const scatterSize  = opts.scatterSize  || 1;
    const meansOnly    = opts.meansOnly    || false;
    const connectMeans = opts.connectMeans || false;
    const hoverText    = (opts.hoverText && opts.hoverText.length === n) ? opts.hoverText : null;

    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const zs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = points[i * 3];
      ys[i] = points[i * 3 + 1];
      zs[i] = points[i * 3 + 2];
    }

    const traces = [];
    const isDiscrete = colorSpec.mode === 'discrete' && labelNames !== null && labels !== null;
    const hasLabels  = labels !== null;

    if (isDiscrete) {
      // ── Discrete: named classes, swatch legend ────────────────────────

      const uniqueClasses = [...new Set(Array.from(labels))].sort((a, b) => a - b);
      const nClasses = uniqueClasses.length;
      const skipEp   = colorSpec.skip_endpoints !== false;

      // colorMap keys: either display name or string class id.
      let colorMap = {};
      if (colorSpec.color_map) {
        colorMap = colorSpec.color_map;
      } else {
        const rgbs = await fetchColorscale(colorSpec.scale || 'Plasma', nClasses, skipEp);
        uniqueClasses.forEach((c, i) => { colorMap[String(c)] = rgbs[i]; });
      }

      function classColor(c) {
        const name = labelNames[c] || String(c);
        return colorMap[name] || colorMap[String(c)] || colorMap[c] || '#888';
      }

      // Per-class scatter (skipped in meansOnly mode).
      if (!meansOnly) {
        uniqueClasses.forEach(c => {
          const cx = [], cy = [], cz = [], ch = [];
          for (let i = 0; i < n; i++) {
            if (labels[i] === c) {
              cx.push(xs[i]); cy.push(ys[i]); cz.push(zs[i]);
              if (hoverText) ch.push(hoverText[i]);
            }
          }
          traces.push({
            type: 'scatter3d', mode: 'markers',
            x: cx, y: cy, z: cz,
            marker: { size: scatterSize, color: classColor(c), opacity: 0.5, line: { width: 0 } },
            name: labelNames[c] || String(c),
            showlegend: true,
            hovertext:  hoverText ? ch      : undefined,
            hoverinfo:  hoverText ? 'text'  : 'skip',
          });
        });
      }

      // Class means — no outline, no text labels on the graph.
      const means = classMeans(points, labels);
      const mx = [], my = [], mz = [], mc = [], mhover = [];
      for (const c in means) {
        mx.push(means[c][0]); my.push(means[c][1]); mz.push(means[c][2]);
        mc.push(classColor(c));
        mhover.push(labelNames[c] || String(c));
      }
      traces.push({
        type: 'scatter3d', mode: 'markers',
        x: mx, y: my, z: mz,
        marker: { size: 5, color: mc, opacity: 1.0 },
        hovertext: mhover,
        hoverinfo: 'text',
        showlegend: false,
        name: 'means',
      });

    } else {
      // ── Continuous: colorscale gradient, colorbar ─────────────────────

      let colorArr;
      let cbarTitle = colorSpec.continuous_label || '';
      if (hasLabels) {
        // Newline / temperature: integer bucket ids drive the colorscale.
        colorArr = Array.from(labels);
      } else {
        // Unlabeled (continuity, random): color by distance from origin.
        // The continuity tensor is metrics-only and must not affect the color.
        colorArr = [];
        for (let i = 0; i < n; i++) {
          colorArr.push(Math.sqrt(xs[i]*xs[i] + ys[i]*ys[i] + zs[i]*zs[i]));
        }
        if (!cbarTitle) cbarTitle = 'Distance from origin';
      }

      const scale    = colorSpec.scale || 'Viridis';
      // Always fetch the colorscale array from the server so that custom/cmocean
      // scales (e.g. 'thermal', 'phase', 'mygbm') render identically to Python.
      const csArr    = await fetchColorscaleArray(scale);
      const cbarSpec = _colorbar(cbarTitle);

      // Scatter trace — skipped in meansOnly mode.
      // When means are also shown, they carry the colorbar; suppress it here.
      if (!meansOnly) {
        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: Array.from(xs), y: Array.from(ys), z: Array.from(zs),
          marker: {
            size: scatterSize,
            color: colorArr,
            colorscale: csArr,
            opacity: 0.8,
            showscale: !hasLabels,
            colorbar: cbarSpec,
            line: { width: 0 },
          },
          hovertext:  hoverText ? hoverText : undefined,
          hoverinfo:  hoverText ? 'text' : 'skip',
          showlegend: false,
          name: 'scatter',
        });
      }

      // Continuous class means — whenever labels are present.
      // Means carry the sole colorbar (scatter's is suppressed when both present).
      if (hasLabels) {
        const means = classMeans(points, labels);
        const sortedClasses = Object.keys(means).map(Number).sort((a, b) => a - b);
        const mx = [], my = [], mz = [], mc = [];
        for (const c of sortedClasses) {
          mx.push(means[c][0]); my.push(means[c][1]); mz.push(means[c][2]);
          mc.push(c);
        }
        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: mx, y: my, z: mz,
          marker: {
            size: 5,
            color: mc,
            colorscale: csArr,
            opacity: 1.0,
            showscale: true,
            colorbar: cbarSpec,
          },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'means',
        });

        // Gradient line connecting adjacent means — driven by connect_means in index.json.
        if (connectMeans && sortedClasses.length > 1) {
          traces.push({
            type: 'scatter3d', mode: 'lines',
            x: mx, y: my, z: mz,
            line: {
              color: mc,         // integer class ids → same scale as means dots
              colorscale: csArr,
              width: 4,
              cauto: true,
            },
            hoverinfo: 'skip',
            showlegend: false,
            name: 'mean_line',
          });
        }
      }
    }

    // Origin cross (all modes).
    traces.push(_originTrace());

    const layout = {
      title: opts.title || '',
      margin: { l: 0, r: 100, t: 30, b: 0 },
      scene: {
        xaxis: _axis(),
        yaxis: _axis(),
        zaxis: _axis(),
        aspectmode: 'data',
      },
      showlegend: isDiscrete && !meansOnly,
    };

    return { data: traces, layout };
  }

  return { buildScatter, classMeans, fetchColorscale, fetchColorscaleArray };
})();
