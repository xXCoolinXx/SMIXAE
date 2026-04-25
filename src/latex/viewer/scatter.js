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
 *   scatterSize  (default 1)      marker size for scatter points
 *   meansOnly    (default false)  skip per-class/continuous scatter;
 *                                 render only means + origin
 *   hoverText    (default null)   string[] of per-point hover text;
 *                                 attached to scatter traces only (not means)
 */

const Scatter = (() => {

  // ── Colorscale sampling ────────────────────────────────────────────────
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

  // ── Axis helper ────────────────────────────────────────────────────────
  // Mirrors Python scatter3d._make_axis(): grey panes, grid, zeroline,
  // no tick labels, no spikes.
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
      title: { text: title || '', side: 'right' },
      thickness: 30,
      len: 0.9,
      x: 1.02,
      outlinewidth: 1,
      outlinecolor: 'black',
      ticks: 'outside',
      ticklen: 6,
      tickwidth: 1,
      tickcolor: 'black',
    };
  }

  // ── Class mean computation ─────────────────────────────────────────────
  // points: flat Float32Array [x0,y0,z0, x1,y1,z1, ...]
  // labels: Int32Array
  // returns: { classId(string): [mx, my, mz], ... }
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

  // ── Origin cross marker ────────────────────────────────────────────────
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
   * @param {object} opts                  {title, scatterSize, meansOnly, hoverText}
   * @returns {Promise<{data: Array, layout: object}>}
   */
  async function buildScatter(points, labels, continuity, colorSpec, labelNames, opts = {}) {
    const n           = points.length / 3;
    const scatterSize = opts.scatterSize || 1;
    const meansOnly   = opts.meansOnly   || false;
    const hoverText   = (opts.hoverText && opts.hoverText.length === n) ? opts.hoverText : null;

    // Unpack flat buffer.
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const zs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = points[i * 3];
      ys[i] = points[i * 3 + 1];
      zs[i] = points[i * 3 + 2];
    }

    const traces = [];

    // mode === 'discrete' AND named labels → discrete swatch legend.
    // Otherwise → continuous colorscale + colorbar.
    // hasLabels drives means rendering independently.
    const isDiscrete = colorSpec.mode === 'discrete' && labelNames !== null && labels !== null;
    const hasLabels  = labels !== null;

    if (isDiscrete) {
      // ── Discrete ────────────────────────────────────────────────────
      const uniqueClasses = [...new Set(Array.from(labels))].sort((a, b) => a - b);
      const nClasses = uniqueClasses.length;
      const skipEp   = colorSpec.skip_endpoints !== false;

      // Build colorMap: explicit map wins, else fetch sampled colors.
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

      // Per-class scatter — skipped in meansOnly mode.
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
            hovertext:  hoverText ? ch   : undefined,
            hoverinfo:  hoverText ? 'text' : 'skip',
          });
        });
      }

      // Class means — mode:'markers' only, no text on the graph.
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
        marker: { size: 5, color: mc, opacity: 1.0, line: { width: 1, color: 'black' } },
        hovertext: mhover,
        hoverinfo: 'text',
        showlegend: false,
        name: 'means',
      });

    } else {
      // ── Continuous ──────────────────────────────────────────────────
      let colorArr;
      let cbarTitle = colorSpec.continuous_label || '';

      if (hasLabels) {
        colorArr = Array.from(labels);
      } else if (continuity !== null) {
        colorArr = Array.from(continuity);
      } else {
        colorArr = [];
        for (let i = 0; i < n; i++) {
          colorArr.push(Math.sqrt(xs[i]*xs[i] + ys[i]*ys[i] + zs[i]*zs[i]));
        }
        cbarTitle = 'Distance from origin';
      }

      const scale = colorSpec.scale || 'Viridis';

      // Scatter trace — skipped in meansOnly mode.
      // When labels are present the means trace carries the colorbar,
      // so showscale is false here to avoid duplication.
      if (!meansOnly) {
        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: Array.from(xs), y: Array.from(ys), z: Array.from(zs),
          marker: {
            size: scatterSize,
            color: colorArr,
            colorscale: scale,
            opacity: 0.8,
            showscale: !hasLabels,
            colorbar: _colorbar(cbarTitle),
            line: { width: 0 },
          },
          hovertext:  hoverText ? hoverText : undefined,
          hoverinfo:  hoverText ? 'text' : 'skip',
          showlegend: false,
          name: 'scatter',
        });
      }

      // Continuous class means — rendered whenever labels are present.
      // Colored by integer label index through the same colorscale,
      // giving the helix/spiral structure for newline/temperature tasks.
      // This trace always carries the colorbar (single bar, correct range).
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
            colorscale: scale,
            opacity: 1.0,
            showscale: true,
            colorbar: _colorbar(cbarTitle),
            line: { width: 1, color: 'rgba(0,0,0,0.4)' },
          },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'means',
        });
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

  return { buildScatter, classMeans, fetchColorscale };
})();
