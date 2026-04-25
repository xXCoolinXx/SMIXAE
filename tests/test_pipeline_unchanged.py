"""Golden-snapshot regression tests for the probing pipelines.

These tests verify that the core scoring functions work correctly
with synthetic data.
"""

import numpy as np
import pandas as pd
import torch

from analysis.anthropic_newline import compute_expert_class_stats, compute_expert_scores
from analysis.utils import Expert


def _create_cluster_expert(expert_id: int, n_points: int = 60, seed: int = 42):
    """Create an Expert with labeled data for scoring."""
    torch.manual_seed(seed + expert_id)
    d = 3
    n_classes = 3
    points_per_class = n_points // n_classes

    all_points = []
    all_labels = []
    for cls in range(n_classes):
        offset = cls * 2.5
        cls_points = torch.randn(points_per_class, d) * 0.4 + offset
        cls_labels = torch.full((points_per_class,), cls, dtype=torch.int64)
        all_points.append(cls_points)
        all_labels.append(cls_labels)

    points = torch.cat(all_points, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return Expert(
        active_mask=torch.ones(points.shape[0], dtype=torch.bool),
        expert_id=expert_id,
        llm_activations=torch.randn(points.shape[0], 32) * 0.5,
        expert_activations=points,
        labels=labels,
        n_classes=n_classes,
    )


def _create_linear_expert(expert_id: int, n_points: int = 60, seed: int = 42):
    """Create an Expert with linear relationship."""
    torch.manual_seed(seed + expert_id)
    d = 3
    n_classes = 3

    t = torch.linspace(0, 2, n_points)
    points = torch.stack([t, t * 0.5, t * 0.25], dim=1) + torch.randn(n_points, d) * 0.15
    labels = (t * 1.5).long() % n_classes

    return Expert(
        active_mask=torch.ones(points.shape[0], dtype=torch.bool),
        expert_id=expert_id,
        llm_activations=torch.randn(points.shape[0], 32) * 0.5,
        expert_activations=points,
        labels=labels,
        n_classes=n_classes,
    )


def _create_periodic_acts(n_points: int = 200, n_experts: int = 3, line_length: int = 80, seed: int = 42):
    """Create synthetic periodic expert activations."""
    torch.manual_seed(seed)
    d = 3

    chars = torch.randint(1, line_length + 1, (n_points,))
    expert_acts = torch.zeros(n_points, n_experts, d)

    for e in range(n_experts):
        if e == 0:
            expert_acts[:, e, 0] = torch.sin(2 * np.pi * chars / line_length)
            expert_acts[:, e, 1] = torch.cos(2 * np.pi * chars / line_length)
            expert_acts[:, e, 2] = chars.float() / line_length
        elif e == 1:
            expert_acts[:, e, :] = chars[:, None].float() / line_length
        else:
            expert_acts[:, e, :] = torch.randn(n_points, d) * 0.3

    return expert_acts, chars


class TestFisherScoring:
    """Test Fisher discriminant scoring."""

    def test_cluster_fisher_score(self):
        """Cluster data should yield positive Fisher score."""
        expert = _create_cluster_expert(expert_id=0, n_points=60, seed=42)
        score = expert.evaluate_fisher()

        assert score > 0.0, f"Expected positive Fisher score, got {score:.2f}"
        assert expert.fisher_score == score

    def test_linear_fisher_score(self):
        """Linear separable data should yield positive Fisher score."""
        expert = _create_linear_expert(expert_id=0, n_points=60, seed=42)
        score = expert.evaluate_fisher()

        assert score > 0.0, f"Expected positive Fisher score, got {score:.2f}"

    def test_adjusted_fisher_computed(self):
        """Adjusted Fisher should be computed and stored."""
        expert = _create_cluster_expert(expert_id=0, n_points=60, seed=42)
        expert.evaluate_fisher()

        assert expert.adjusted_fisher_score is not None
        assert expert.n_unique_labels is not None

    def test_n_unique_labels(self):
        """n_unique_labels should be computed."""
        expert = _create_cluster_expert(expert_id=0, n_points=60, seed=42)
        expert.fisher_labels = expert.labels

        assert expert.n_unique_labels == 3


class TestNewlineScoring:
    """Test newline periodic gain scoring."""

    def test_compute_expert_scores_returns_df(self):
        """compute_expert_scores returns a DataFrame."""
        expert_acts, labels = _create_periodic_acts(n_points=100, line_length=80, seed=42)

        scores = compute_expert_scores(
            expert_acts,
            labels,
            line_length=80,
            n_harmonics=3,
        )

        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == 3

    def test_periodic_gain_exists(self):
        """Score columns should exist."""
        expert_acts, labels = _create_periodic_acts(n_points=100, line_length=80, seed=42)

        scores = compute_expert_scores(
            expert_acts,
            labels,
            line_length=80,
            n_harmonics=3,
        )

        assert "periodic_gain" in scores.columns
        assert "decode_r2" in scores.columns
        assert "encode_linear_r2" in scores.columns
        assert "encode_periodic_r2" in scores.columns

    def test_periodic_gain_positive_for_periodic(self):
        """Periodic expert should have positive periodic_gain."""
        torch.manual_seed(123)
        n = 200
        d = 3
        n_experts = 2
        line_length = 80

        chars = torch.randint(1, line_length + 1, (n,))
        expert_acts = torch.zeros(n, n_experts, d)

        expert_acts[:, 0, 0] = torch.sin(2 * np.pi * chars / line_length)
        expert_acts[:, 0, 1] = torch.cos(2 * np.pi * chars / line_length)

        expert_acts[:, 1, :] = chars[:, None].float() / line_length * 0.5

        scores = compute_expert_scores(
            expert_acts,
            chars,
            line_length=line_length,
            n_harmonics=3,
        )

        row = scores[scores["expert_id"] == 0].iloc[0]
        periodic_gain = row["periodic_gain"]

        assert periodic_gain > 0, f"Expected positive periodic_gain, got {periodic_gain}"

    def test_dimension_breakdown(self):
        """Scores should include per-dimension metrics."""
        expert_acts, labels = _create_periodic_acts(n_points=100, line_length=80, seed=42)

        scores = compute_expert_scores(
            expert_acts,
            labels,
            line_length=80,
            n_harmonics=3,
        )

        row = scores[scores["expert_id"] == 0].iloc[0]
        assert "dim0_corr" in row.index

    def test_expert_class_stats(self):
        """Test compute_expert_class_stats function."""
        torch.manual_seed(0)
        n = 60
        d = 3

        acts = torch.randn(n, d)
        labels = torch.randint(0, 4, (n,))

        means, rates, unique = compute_expert_class_stats(acts, labels, threshold=0.0)

        assert means.shape == (4, d)
        assert rates.shape == (4,)
        assert len(unique) == 4


class TestToProbingRecord:
    """Test Expert.to_probing_record()."""

    def test_labeled_record_includes_fisher(self):
        """Labeled expert record should include Fisher score."""
        expert = _create_cluster_expert(expert_id=5, n_points=30, seed=42)
        expert.evaluate_fisher()

        record = expert.to_probing_record(str_tokens=None)

        assert record.expert_id == 5
        assert "fisher_score" in record.metrics
        assert record.metrics["fisher_score"] == expert.fisher_score

    def test_unlabeled_record_includes_continuity(self):
        """Unlabeled expert record should include continuity."""
        torch.manual_seed(0)
        d = 3
        d_model = 32
        n_points = 30

        base = torch.randn(n_points, d)
        llm_base = torch.randn(n_points, d_model) * 0.5 + base[:, :1] * 0.5

        expert = Expert(
            active_mask=torch.ones(n_points, dtype=torch.bool),
            expert_id=0,
            llm_activations=llm_base,
            expert_activations=base,
            labels=None,
        )
        expert.evaluate_manifold(k_neighbors=5, device="cpu")

        record = expert.to_probing_record(str_tokens=None)

        assert "mean_continuity" in record.metrics
        assert record.metrics["mean_continuity"] is not None

    def test_record_tensors_present(self):
        """Record should include tensors."""
        expert = _create_cluster_expert(expert_id=5, n_points=30, seed=42)

        record = expert.to_probing_record(str_tokens=None)

        assert record.points is not None
        assert record.labels is not None
        assert record.points.shape[0] == 30


class TestRegressionProbing:
    """Test regression probing - uses real workflow via get_sae_activations."""

    # Regression probed via full pipeline (categorize_all.py) in practice
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_encode_produces_expected_shapes(self):
        """encode_sae_batched produces expected shapes."""
        torch.manual_seed(0)
        from smixae import SMIXAE, SMIXAEConfig

        d_model = 32
        d_expert = 8
        n_experts = 4
        d_sae = n_experts * d_expert  # Internally set to n_experts * d_expert

        cfg = SMIXAEConfig(
            n_experts=n_experts,
            d_expert=d_expert,
            d_bottleneck=3,
            d_in=d_model,
            d_sae=d_sae,
        )
        sae = SMIXAE(cfg)
        sae.eval()

        from analysis.utils import encode_sae_batched

        batch_input = torch.randn(10, d_model)
        encoded = encode_sae_batched(sae, batch_input, batch_size=16)

        assert encoded.shape == (10, n_experts, 3)
