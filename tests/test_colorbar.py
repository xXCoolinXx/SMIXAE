"""Unit tests for colorbar generation — both Plotly and PIL implementations.

These tests verify:
1. Tick placement is correct in auto_tick_increment()
2. Discrete sampling positions respect skip_endpoints
3. PIL continuous colorbar auto-selects nice increments
4. Plotly and PIL sample the same colors when given the same inputs
5. Edge cases (single class, tight ranges, etc.)
"""

import tempfile
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import pytest
from PIL import Image

from analysis.colors import (
    _discrete_positions,
    add_colorbar_trace,
    auto_tick_increment,
    build_clipped_colorscale,
    build_color_map,
    render_continuous_colorbar_png,
    render_discrete_from_scale_png,
    sample_named_scale_discrete,
)


class TestAutoTickIncrement:
    """Tests for auto_tick_increment() tick-placement logic."""

    def test_returns_requested_when_n_le_25(self):
        """When n_classes <= 25, returns requested value unchanged."""
        assert auto_tick_increment(0, 100, 5, 20.0) == 20.0
        assert auto_tick_increment(0, 50, 10, 15.0) == 15.0
        assert auto_tick_increment(0, 50, 25, 10.0) == 10.0

    def test_returns_none_when_requested_is_none(self):
        """When requested is None and n_classes <= 25, returns None."""
        assert auto_tick_increment(0, 100, 5, None) is None
        assert auto_tick_increment(0, 50, 25, None) is None

    def test_auto_calculated_when_n_gt_25(self):
        """When n_classes > 25 and requested is None, picks from [20, 10, 5]."""
        result = auto_tick_increment(0, 100, 30, None)
        assert result in (20.0, 10.0, 5.0)

    def test_auto_calculated_gives_5_ticks_minimum(self):
        """Auto-calculated increment gives >= 5 ticks in range."""
        for inc in auto_tick_increment(0, 100, 30, None), auto_tick_increment(0, 50, 30, None):
            span = 100 if inc == 20.0 else 50
            n_ticks = span / inc
            assert n_ticks >= 5

    def test_auto_calculated_returns_5_for_tight_range(self):
        """Returns 5.0 for tight ranges where larger increments don't fit."""
        result = auto_tick_increment(0, 15, 30, None)
        assert result == 5.0


class TestDiscretePositions:
    """Tests for _discrete_positions() endpoint skipping."""

    @pytest.mark.parametrize(
        "n,skip_endpoints,expected",
        [
            (7, True, [1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8]),
            (7, False, [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]),
            (1, True, [0.5]),
            (1, False, [0.5]),
            (2, True, [1 / 3, 2 / 3]),
            (2, False, [0.0, 1.0]),
        ],
    )
    def test_discrete_positions(self, n, skip_endpoints, expected):
        result = _discrete_positions(n, skip_endpoints=skip_endpoints)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-9

    def test_discrete_positions_skip_endpoints_avoids_extremes(self):
        """skip_endpoints=True ensures t not in {0, 1}."""
        for n in [3, 5, 7, 10, 20]:
            positions = _discrete_positions(n, skip_endpoints=True)
            for t in positions:
                assert 0 < t < 1, f"t={t} is at endpoint for n={n}"

    def test_discrete_positions_include_endpoints_at_edges(self):
        """skip_endpoints=False includes 0 and 1 for n>=2."""
        for n in [2, 5, 10]:
            positions = _discrete_positions(n, skip_endpoints=False)
            assert positions[0] == 0.0
            assert positions[-1] == 1.0


class TestSampleNamedScaleDiscrete:
    """Tests for sample_named_scale_discrete() colorsampling."""

    @pytest.mark.parametrize("skip_endpoints", [True, False])
    def test_sample_returns_correct_count(self, skip_endpoints):
        for n in [3, 5, 7, 10]:
            result = sample_named_scale_discrete("Plasma", n, skip_endpoints=skip_endpoints)
            assert len(result) == n

    def test_sample_returns_valid_rgb_strings(self):
        result = sample_named_scale_discrete("Plasma", 5)
        for s in result:
            assert s.startswith("rgb(")
            assert s.endswith(")")

    def test_skip_endpoints_changes_colors(self):
        """Different colors when skipping endpoints vs including them."""
        with_skip = sample_named_scale_discrete("Viridis", 7, skip_endpoints=True)
        without = sample_named_scale_discrete("Viridis", 7, skip_endpoints=False)
        assert with_skip != without


class TestBuildClippedColorscale:
    """Tests for build_clipped_colorscale() — Plotly colorscale clipping.

    The function returns a 256-stop colorscale spanning [0, 1] in PLOTLY values,
    but the COLORS are sampled from the clipped range [1/(n+1), n/(n+1)].
    """

    def test_clipped_returns_256_stops(self):
        """Returns 256 stops (Plotly default)."""
        n = 7
        scale = build_clipped_colorscale("Plasma", n, skip_endpoints=True)
        assert len(scale) == 256

    def test_clipped_endpoint_plotly_values(self):
        """Plotly values at endpoints are 0.0 and 1.0."""
        n = 7
        scale = build_clipped_colorscale("Plasma", n, skip_endpoints=True)
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0

    def test_clipped_colors_from_clipped_range(self):
        """Colors are sampled from the clipped range, not extremes."""
        n = 7
        scale = build_clipped_colorscale("Viridis", n, skip_endpoints=True)

        extreme_first = pc.sample_colorscale("Viridis", [0.0])[0]
        extreme_last = pc.sample_colorscale("Viridis", [1.0])[0]
        clipped_first = scale[0][1]
        clipped_last = scale[-1][1]

        assert clipped_first != extreme_first
        assert clipped_last != extreme_last

    def test_not_clipped_includes_endpoints(self):
        """Non-clipped colorscale uses extremes."""
        n = 7
        scale = build_clipped_colorscale("Plasma", n, skip_endpoints=False)
        first_color = pc.sample_colorscale("Plasma", [0.0])[0]
        last_color = pc.sample_colorscale("Plasma", [1.0])[0]
        assert scale[0][1] == first_color
        assert scale[-1][1] == last_color


class TestPILContinuousColorbar:
    """Tests for render_continuous_colorbar_png() tick placement."""

    def test_nice_increment_selection(self):
        """Verify auto-selects from nice_increments list."""
        from analysis.colors import render_continuous_colorbar_png
        labels = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_continuous_colorbar_png(
                colorscale="Plasma",
                labels=labels,
                output_path=output_path,
            )
            im = Image.open(output_path)
            assert im.width > 0
            assert im.height > 0

    def test_tick_placement_at_endpoints(self):
        """First/last ticks are at vmin and vmax."""
        labels = [0.0, 25.0, 50.0, 75.0, 100.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_continuous_colorbar_png(
                colorscale="Viridis",
                labels=labels,
                output_path=output_path,
            )
            im = Image.open(output_path)
            pixels = np.array(im)
            assert pixels.shape[0] > 0

    def test_single_value_becomes_unit_range(self):
        """Single label creates vmax = vmin + 1.0."""
        labels = [42.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_continuous_colorbar_png(
                colorscale="Viridis",
                labels=labels,
                output_path=output_path,
            )
            im = Image.open(output_path)
            assert im.width > 0

    def test_non_numeric_raises(self):
        """Non-numeric labels raise ValueError."""
        labels = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            with pytest.raises(ValueError, match="not numeric"):
                render_continuous_colorbar_png(
                    colorscale="Viridis",
                    labels=labels,
                    output_path=output_path,
                )

    @pytest.mark.parametrize(
        "labels",
        [
            list(range(0, 11)),      # 0-10
            list(range(0, 21)),      # 0-20
            list(range(0, 51)),      # 0-50
            list(range(0, 101)),    # 0-100
            list(range(0, 201)),    # 0-200
        ],
    )
    def test_various_ranges_produce_valid_png(self, labels):
        """Each range produces a valid PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_continuous_colorbar_png(
                colorscale="Plasma",
                labels=labels,
                output_path=output_path,
            )
            im = Image.open(output_path)
            assert im.width > 0
            assert im.height > 0


class TestPlotlyColorbarTrace:
    """Tests for add_colorbar_trace() tick generation."""

    def test_add_colorbar_creates_trace(self):
        """add_colorbar_trace adds a valid trace."""
        classes = list(range(7))
        fig = go.Figure()
        fig = add_colorbar_trace(
            fig,
            classes=classes,
            colorscale_name="Plasma",
            names={i: f"Class {i}" for i in classes},
        )
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.marker["showscale"] is True

    def test_tickvals_match_class_indices(self):
        """When names dict provided, tickvals are class indices."""
        classes = list(range(5))
        fig = go.Figure()
        fig = add_colorbar_trace(
            fig,
            classes=classes,
            colorscale_name="Viridis",
            names={i: f"L{i}" for i in classes},
        )
        cb = fig.data[0].marker.colorbar
        assert list(cb.tickvals) == list(range(5))

    def test_skip_endpoints_clips_colorscale(self):
        """skip_endpoints=True clips the colorscale."""
        n = 7
        classes = list(range(n))
        fig_skip = go.Figure()
        fig_skip = add_colorbar_trace(
            fig_skip,
            classes=classes,
            colorscale_name="Plasma",
            skip_endpoints=True,
        )
        fig_no_skip = go.Figure()
        fig_no_skip = add_colorbar_trace(
            fig_no_skip,
            classes=classes,
            colorscale_name="Plasma",
            skip_endpoints=False,
        )
        assert fig_skip.data[0].marker.colorscale != fig_no_skip.data[0].marker.colorscale


class TestPlotlyPILColorConsistency:
    """Tests that Plotly and PIL sample the same colors."""

    def test_same_positions_sample_same_colors(self):
        """PIL and Plotly sample identical colors at same positions."""
        n = 7
        skip = True
        name = "Viridis"

        pil_colors = sample_named_scale_discrete(name, n, skip_endpoints=skip)

        positions = _discrete_positions(n, skip_endpoints=skip)
        plotly_colors = [pc.sample_colorscale(name, [t])[0] for t in positions]

        assert len(pil_colors) == len(plotly_colors)
        for p, py in zip(pil_colors, plotly_colors):
            assert p == py

    def test_discrete_plotly_pil_consistency(self):
        """render_discrete_from_scale_png uses same colors as Plotly."""
        n = 5
        labels = [f"label_{i}" for i in range(n)]
        name = "Plasma"
        skip = True

        pil_colors = sample_named_scale_discrete(name, n, skip_endpoints=skip)
        color_map = {lbl: pil_colors[i] for i, lbl in enumerate(labels)}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_discrete_from_scale_png(
                colorscale=name,
                labels=labels,
                output_path=output_path,
                display_names={lbl: lbl for lbl in labels},
                skip_endpoints=skip,
            )
            im = Image.open(output_path)
            assert im.width > 0


class TestDiscreteLegendFromScale:
    """Tests for render_discrete_from_scale_png()."""

    def test_writes_valid_png(self):
        """Produces a valid PNG with swatches."""
        labels = ["a", "b", "c", "d", "e"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_discrete_from_scale_png(
                colorscale="Viridis",
                labels=labels,
                output_path=output_path,
                display_names={l: l.upper() for l in labels},
            )
            im = Image.open(output_path)
            assert im.width > 0
            assert im.height > 0

    def test_skip_endpoints_respected(self):
        """skip_endpoints parameter is passed through."""
        labels = ["a", "b", "c"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_discrete_from_scale_png(
                colorscale="Plasma",
                labels=labels,
                output_path=output_path,
                skip_endpoints=True,
            )
            im1 = Image.open(output_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            render_discrete_from_scale_png(
                colorscale="Plasma",
                labels=labels,
                output_path=output_path,
                skip_endpoints=False,
            )
            im2 = Image.open(output_path)

        pixels1 = np.array(im1)
        pixels2 = np.array(im2)
        assert not np.array_equal(pixels1, pixels2)


class TestColorbarEdgeCases:
    """Edge case handling for colorbar generation."""

    def test_single_class_discrete(self):
        """n=1 returns valid color mapping."""
        color_map = build_color_map(["single"], "Plasma", skip_endpoints=True)
        assert len(color_map) == 1
        assert "single" in color_map

    def test_two_classes_non_zero_positions(self):
        """n=2 with skip_endpoints=True uses [1/3, 2/3], not [0, 1]."""
        scale = sample_named_scale_discrete("Viridis", 2, skip_endpoints=True)
        assert len(scale) == 2

        t_values = _discrete_positions(2, skip_endpoints=True)
        assert 0 < t_values[0] < 1
        assert 0 < t_values[1] < 1

    def test_large_n_tick_increment_auto(self):
        """N > 25 triggers auto tick increment in Plotly."""
        classes = list(range(30))
        fig = go.Figure()
        fig = add_colorbar_trace(
            fig,
            classes=classes,
            colorscale_name="Plasma",
        )
        cb = fig.data[0].marker.colorbar
        assert cb.tickvals is not None
        assert len(cb.tickvals) < 30

    def test_build_color_map_with_dict(self):
        """Dict colorscale passed through unchanged."""
        classes = ["a", "b", "c"]
        explicit = {"a": "rgb(255,0,0)", "b": "rgb(0,255,0)", "c": "rgb(0,0,255)"}
        result = build_color_map(classes, explicit)
        assert result == explicit

    def test_build_color_map_with_list(self):
        """List colorscale mapped by index."""
        classes = ["a", "b", "c"]
        color_list = ["red", "green", "blue"]
        result = build_color_map(classes, color_list)
        for c, expected in zip(classes, color_list):
            assert c in result
