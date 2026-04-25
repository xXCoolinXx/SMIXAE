"""Unit tests for the probing I/O format: write/read round-trips.

These tests validate that probing_io.write_probing_task produces artifacts
that can be read back correctly by the reader functions.
"""


import pytest
import torch

from analysis import probing_io


def _make_labeled_task_index() -> probing_io.TaskIndex:
    """Create a TaskIndex for a labeled probing task with hypotheses."""
    return probing_io.TaskIndex(
        task_type="labeled_probe",
        experiment_id="test_exp",
        dataset_name="test_dataset",
        title="Test Dataset — Expert Analysis",
        model_name="test/model",
        hook_name="model.layers.0",
        d_bottleneck=3,
        n_experts_total=2048,
        color=probing_io.ColorSpec(
            mode="discrete",
            scale="Plasma",
            color_map={"Monday": "rgb(188,72,221)", "Tuesday": "rgb(85,36,163)"},
        ),
        label_names={0: "Monday", 1: "Tuesday"},
        hypotheses=[
            probing_io.HypothesisSpec(
                name="linear_7d",
                description="7-Day Linear",
                regression_type="linear",
                score_type="r2",
            )
        ],
        experts_by_view={
            "fisher": [
                probing_io.ExpertRanking(rank=1, expert_id=3, score=0.845),
                probing_io.ExpertRanking(rank=2, expert_id=1, score=0.712),
            ],
            "linear_7d": [
                probing_io.ExpertRanking(
                    rank=1, expert_id=1, score=0.876, score_std=0.012
                ),
            ],
        },
        scatter_size=1.0,
    )


def _make_unlabeled_task_index() -> probing_io.TaskIndex:
    """Create a TaskIndex for an unlabeled continuity task."""
    return probing_io.TaskIndex(
        task_type="unlabeled_probe",
        experiment_id="test_exp",
        dataset_name="continuity",
        title="Continuity — Expert Analysis",
        model_name="test/model",
        hook_name="model.layers.0",
        d_bottleneck=3,
        n_experts_total=2048,
        color=probing_io.ColorSpec(
            mode="continuous",
            scale="Viridis",
            continuous_label="Distance from origin",
        ),
        label_names=None,
        hypotheses=[],
        experts_by_view={
            "continuity": [
                probing_io.ExpertRanking(rank=1, expert_id=5, score=0.567),
            ],
        },
        scatter_size=5.0,
    )


def _make_newline_task_index() -> probing_io.TaskIndex:
    """Create a TaskIndex for a newline probing task."""
    return probing_io.TaskIndex(
        task_type="newline",
        experiment_id="test_exp",
        dataset_name="newline_80",
        title="Newline 80 — Expert Analysis",
        model_name="test/model",
        hook_name="model.layers.0",
        d_bottleneck=3,
        n_experts_total=2048,
        color=probing_io.ColorSpec(
            mode="continuous",
            scale="Viridis",
            continuous_label="chars since newline",
        ),
        label_names=None,
        hypotheses=[],
        experts_by_view={
            "periodic_gain": [
                probing_io.ExpertRanking(rank=1, expert_id=7, score=0.312),
                probing_io.ExpertRanking(rank=2, expert_id=2, score=0.245),
            ],
        },
        scatter_size=1.0,
    )


def _make_expert_record(
    expert_id: int,
    n_points: int,
    has_labels: bool = True,
    has_continuity: bool = True,
    extra_metrics: dict | None = None,
) -> probing_io.ExpertRecord:
    """Create a synthetic ExpertRecord with known tensor values."""
    torch.manual_seed(expert_id * 42)
    points = torch.randn(n_points, 3)

    if has_labels:
        labels = torch.randint(0, 3, (n_points,), dtype=torch.int64)
    else:
        labels = None

    if has_continuity:
        continuity = torch.rand(n_points)
    else:
        continuity = None

    metrics = {
        "n_points": n_points,
        "mean_latent_l0": 2.3,
        "fisher_score": 0.7 + expert_id * 0.05,
        "adjusted_fisher_score": 0.65 + expert_id * 0.04,
        "mean_continuity": 0.5 + expert_id * 0.03,
    }
    if has_labels:
        metrics["n_unique_labels"] = 3

    if extra_metrics:
        metrics.update(extra_metrics)

    hover_text = [f"[tok {i}]" for i in range(n_points)]

    return probing_io.ExpertRecord(
        expert_id=expert_id,
        metrics=metrics,
        hover_text=hover_text,
        points=points,
        labels=labels,
        continuity=continuity,
    )


class TestProbingIORoundTrip:
    """Test write_probing_task produces correct artifacts."""

    @pytest.fixture
    def tmp_task_dir(self, tmp_path):
        """Create a fresh temp directory for each test."""
        return tmp_path / "task"

    def test_labeled_task_round_trip(self, tmp_task_dir):
        """Test labeled task write/read round-trip."""
        index = _make_labeled_task_index()
        experts = [
            _make_expert_record(3, n_points=100, has_labels=True, has_continuity=False),
            _make_expert_record(1, n_points=100, has_labels=True, has_continuity=False),
        ]

        probing_io.write_probing_task(tmp_task_dir, index=index, experts=experts)

        read_idx = probing_io.read_task_index(tmp_task_dir)
        assert read_idx["task_type"] == "labeled_probe"
        assert read_idx["experiment_id"] == "test_exp"
        assert read_idx["dataset_name"] == "test_dataset"
        assert read_idx["d_bottleneck"] == 3

        assert read_idx["hypotheses"][0]["name"] == "linear_7d"
        assert read_idx["experts_by_view"]["fisher"][0]["expert_id"] == 3
        assert read_idx["experts_by_view"]["fisher"][0]["score"] == 0.845

        meta = probing_io.read_expert_meta(tmp_task_dir, 3)
        assert meta["expert_id"] == 3
        assert meta["metrics"]["fisher_score"] == pytest.approx(0.85, rel=0.01)
        assert "points" in meta["tensor_keys"]

        tensors = probing_io.read_expert_tensors(tmp_task_dir, 3)
        assert tensors["points"].shape == (100, 3)
        assert tensors["labels"] is not None
        assert tensors["labels"].shape == (100,)

    def test_unlabeled_task_round_trip(self, tmp_task_dir):
        """Test unlabeled (continuity) task round-trip."""
        index = _make_unlabeled_task_index()
        experts = [
            _make_expert_record(5, n_points=100, has_labels=False, has_continuity=True),
        ]

        probing_io.write_probing_task(tmp_task_dir, index=index, experts=experts)

        read_idx = probing_io.read_task_index(tmp_task_dir)
        assert read_idx["task_type"] == "unlabeled_probe"
        assert read_idx["color"]["mode"] == "continuous"
        assert read_idx["experts_by_view"]["continuity"][0]["score"] == pytest.approx(0.567)

        meta = probing_io.read_expert_meta(tmp_task_dir, 5)
        assert meta["n_points"] == 100

        tensors = probing_io.read_expert_tensors(tmp_task_dir, 5)
        assert tensors["points"].shape == (100, 3)
        assert tensors["continuity"] is not None

    def test_newline_task_round_trip(self, tmp_task_dir):
        """Test newline task round-trip with periodic_gain metrics."""
        index = _make_newline_task_index()
        experts = [
            _make_expert_record(
                7,
                n_points=100,
                has_labels=False,
                has_continuity=False,
                extra_metrics={
                    "decode_r2": 0.423,
                    "encode_linear_r2": 0.312,
                    "encode_periodic_r2": 0.456,
                    "periodic_gain": 0.144,
                    "dim0_corr": 0.567,
                    "dim0_linear_r2": 0.234,
                    "dim0_periodic_r2": 0.345,
                },
            ),
            _make_expert_record(
                2,
                n_points=100,
                has_labels=False,
                has_continuity=False,
                extra_metrics={
                    "decode_r2": 0.312,
                    "encode_linear_r2": 0.198,
                    "encode_periodic_r2": 0.267,
                    "periodic_gain": 0.069,
                    "dim1_corr": 0.412,
                },
            ),
        ]

        probing_io.write_probing_task(tmp_task_dir, index=index, experts=experts)

        read_idx = probing_io.read_task_index(tmp_task_dir)
        assert read_idx["task_type"] == "newline"
        assert read_idx["experts_by_view"]["periodic_gain"][0]["score"] == pytest.approx(0.312)

        meta = probing_io.read_expert_meta(tmp_task_dir, 7)
        assert meta["metrics"]["periodic_gain"] == pytest.approx(0.144)
        assert meta["metrics"]["encode_periodic_r2"] == pytest.approx(0.456)

        tensors = probing_io.read_expert_tensors(tmp_task_dir, 7)
        assert tensors["points"].shape == (100, 3)
        assert "labels" not in tensors
        assert "continuity" not in tensors

    def test_tensor_values_preserved(self, tmp_task_dir):
        """Test that tensor values match exactly after round-trip."""
        torch.manual_seed(123)
        expected_points = torch.randn(50, 3)
        expected_labels = torch.randint(0, 2, (50,), dtype=torch.int64)

        record = probing_io.ExpertRecord(
            expert_id=10,
            metrics={"n_points": 50},
            hover_text=[],
            points=expected_points,
            labels=expected_labels,
            continuity=None,
        )

        index = probing_io.TaskIndex(
            task_type="labeled_probe",
            experiment_id="test",
            dataset_name="test",
            title="Test",
            model_name="test",
            hook_name="layer.0",
            d_bottleneck=3,
            n_experts_total=8,
            color=probing_io.ColorSpec(mode="discrete", scale="Plasma"),
            label_names=None,
            hypotheses=[],
            experts_by_view={"fisher": []},
        )

        probing_io.write_probing_task(tmp_task_dir, index=index, experts=[record])

        tensors = probing_io.read_expert_tensors(tmp_task_dir, 10)
        torch.testing.assert_close(tensors["points"], expected_points, atol=1e-5, rtol=0)
        torch.testing.assert_close(tensors["labels"], expected_labels)

    def test_no_html_files_written(self, tmp_task_dir):
        """Assert no experts.html or top_experts_html are written."""
        index = _make_labeled_task_index()
        experts = [_make_expert_record(1, n_points=100, has_labels=True)]

        probing_io.write_probing_task(tmp_task_dir, index=index, experts=experts)

        assert not (tmp_task_dir / "experts.html").exists()
        assert not (tmp_task_dir / "top_experts.html").exists()

    def test_stale_expert_cleanup(self, tmp_task_dir):
        """Test that old expert files are cleaned up on re-write."""
        index1 = _make_labeled_task_index()
        experts1 = [
            _make_expert_record(1, n_points=100),
            _make_expert_record(2, n_points=100),
            _make_expert_record(3, n_points=100),
        ]
        probing_io.write_probing_task(tmp_task_dir, index=index1, experts=experts1)

        assert (tmp_task_dir / "experts" / "E1.json").exists()
        assert (tmp_task_dir / "experts" / "E3.json").exists()

        experts2 = [
            _make_expert_record(1, n_points=100),
            _make_expert_record(4, n_points=100),
        ]
        probing_io.write_probing_task(tmp_task_dir, index=index1, experts=experts2)

        assert (tmp_task_dir / "experts" / "E1.json").exists()
        assert not (tmp_task_dir / "experts" / "E2.json").exists()
        assert not (tmp_task_dir / "experts" / "E2.pth").exists()
        assert not (tmp_task_dir / "experts" / "E3.json").exists()


class TestProbingIOHelpers:
    """Test helper functions."""

    def test_is_task_dir(self, tmp_path):
        """Test is_task_dir correctly identifies task directories."""
        assert not probing_io.is_task_dir(tmp_path)

        fake_dir = tmp_path / "not_a_task"
        fake_dir.mkdir()
        assert not probing_io.is_task_dir(fake_dir)

        task_dir = tmp_path / "valid_task"
        task_dir.mkdir()
        index = _make_labeled_task_index()
        experts = [_make_expert_record(1, n_points=100)]
        probing_io.write_probing_task(task_dir, index=index, experts=experts)

        assert probing_io.is_task_dir(task_dir)

    def test_iter_task_dirs(self, tmp_path):
        """Test iter_task_dirs finds all task directories."""
        task1 = tmp_path / "task1"
        task2 = tmp_path / "task2"
        other = tmp_path / "other"

        task1.mkdir(parents=True)
        task2.mkdir(parents=True)
        other.mkdir()

        index = _make_labeled_task_index()
        experts = [_make_expert_record(1, n_points=100)]

        probing_io.write_probing_task(task1, index=index, experts=experts)
        probing_io.write_probing_task(task2, index=index, experts=experts)

        non_task_dir = other / "index.json"
        non_task_dir.write_text('{"invalid": true}')

        found = list(probing_io.iter_task_dirs(tmp_path))
        assert len(found) == 2
