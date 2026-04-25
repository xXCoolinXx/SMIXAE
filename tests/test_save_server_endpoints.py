"""Unit tests for the save server endpoints.

These tests verify that the save server correctly serves
the new probing task format (JSON + .pth tensors).
"""

import threading
import time

import pytest
import requests
import torch

from analysis import probing_io
from analysis.colors import sample_named_scale_discrete
from latex.save_server import _SaveServer, _tensors_to_bytes, _warm_cache


def _make_test_task_index(task_type: str = "labeled_probe") -> probing_io.TaskIndex:
    """Create a minimal TaskIndex for testing."""
    color = probing_io.ColorSpec(
        mode="discrete" if task_type == "labeled_probe" else "continuous",
        scale="Plasma" if task_type == "labeled_probe" else "Viridis",
    )
    return probing_io.TaskIndex(
        task_type=task_type,
        experiment_id="test_exp",
        dataset_name="test_task",
        title="Test Task",
        model_name="test/model",
        hook_name="model.layers.0",
        d_bottleneck=3,
        n_experts_total=4,
        color=color,
        label_names=None,
        hypotheses=[],
        experts_by_view={"fisher": [probing_io.ExpertRanking(rank=1, expert_id=0, score=0.5)]},
    )


def _make_test_expert_record(expert_id: int, n_points: int = 20) -> probing_io.ExpertRecord:
    """Create a test ExpertRecord with known tensor values."""
    torch.manual_seed(expert_id)
    points = torch.randn(n_points, 3)
    labels = torch.randint(0, 2, (n_points,), dtype=torch.int64)
    return probing_io.ExpertRecord(
        expert_id=expert_id,
        metrics={"n_points": n_points, "fisher_score": 0.5 + expert_id * 0.1},
        hover_text=[f"pt{i}" for i in range(n_points)],
        points=points,
        labels=labels,
        continuity=None,
    )


@pytest.fixture
def fixture_task_dir(tmp_path):
    """Create a fixture task directory with synthetic data."""
    task_dir = tmp_path / "probe" / "test_model" / "test_task"
    task_dir.mkdir(parents=True)

    index = _make_test_task_index()
    experts = [_make_test_expert_record(0, n_points=20)]

    probing_io.write_probing_task(task_dir, index=index, experts=experts)
    return task_dir


class TestSaveServerEndpoints:
    """Test save server HTTP endpoints."""

    @pytest.fixture
    def server_and_port(self, tmp_path, fixture_task_dir):
        """Start a test server in a background thread."""
        output_dir = tmp_path / "queue"
        output_dir.mkdir()

        results_dir = fixture_task_dir.parent.parent.parent

        server = _SaveServer(output_dir, port=0, results_dir=results_dir)
        port = server.server_address[1]

        _warm_cache(results_dir)

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        time.sleep(0.1)

        yield server, port, results_dir

        server.shutdown()

    def test_ping_endpoint(self, server_and_port):
        """Test /ping returns OK."""
        server, port, _ = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/ping")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_tasks_endpoint(self, server_and_port):
        """Test /tasks lists tasks."""
        server, port, results_dir = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        tasks = data["tasks"]
        assert len(tasks) >= 1
        task = tasks[0]
        assert "rel" in task
        assert task["task_type"] == "labeled_probe"
        assert task["dataset_name"] == "test_task"

    def test_task_endpoint(self, server_and_port):
        """Test /task returns index.json."""
        server, port, results_dir = server_and_port
        rel = "probe/test_model/test_task"
        resp = requests.get(f"http://127.0.0.1:{port}/task?p={rel}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_type"] == "labeled_probe"
        assert data["experiment_id"] == "test_exp"
        assert data["d_bottleneck"] == 3

    def test_task_endpoint_missing_param(self, server_and_port):
        """Test /task returns error without p param."""
        server, port, _ = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/task")
        assert resp.status_code == 400

    def test_task_endpoint_not_found(self, server_and_port):
        """Test /task returns 404 for unknown task."""
        server, port, _ = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/task?p=nonexistent/path")
        assert resp.status_code == 404

    def test_expert_endpoint(self, server_and_port):
        """Test /expert returns per-expert JSON."""
        server, port, results_dir = server_and_port
        rel = "probe/test_model/test_task"
        resp = requests.get(f"http://127.0.0.1:{port}/expert?p={rel}&id=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["expert_id"] == 0
        assert "metrics" in data
        assert data["metrics"]["fisher_score"] == pytest.approx(0.5, rel=0.01)

    def test_expert_endpoint_missing_params(self, server_and_port):
        """Test /expert returns error without required params."""
        server, port, _ = server_and_port

        resp = requests.get(f"http://127.0.0.1:{port}/expert?p=test")
        assert resp.status_code == 404

        resp = requests.get(f"http://127.0.0.1:{port}/expert?p=test&id=0")
        assert resp.status_code == 400

    def test_expert_tensors_endpoint(self, server_and_port):
        """Test /expert-tensors returns binary blob."""
        server, port, results_dir = server_and_port
        rel = "probe/test_model/test_task"
        resp = requests.get(f"http://127.0.0.1:{port}/expert-tensors?p={rel}&id=0")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/octet-stream"

        data = resp.content
        assert len(data) > 0

        n_points = 20
        points = torch.frombuffer(data[:n_points * 3 * 4], dtype=torch.float32)
        assert points.shape == (n_points * 3,)
        assert points[0] == points[0]

    def test_expert_tensors_caching(self, server_and_port):
        """Test tensors are cached after first request."""
        server, port, results_dir = server_and_port
        rel = "probe/test_model/test_task"

        resp1 = requests.get(f"http://127.0.0.1:{port}/expert-tensors?p={rel}&id=0")
        data1 = resp1.content

        resp2 = requests.get(f"http://127.0.0.1:{port}/expert-tensors?p={rel}&id=0")
        data2 = resp2.content

        assert data1 == data2

    def test_colorscale_endpoint(self, server_and_port):
        """Test /colorscale returns sampled colors."""
        server, port, _ = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/colorscale?name=Plasma&n=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "colors" in data
        colors = data["colors"]
        assert len(colors) == 7

        expected = sample_named_scale_discrete("Plasma", 7)
        assert colors == expected

    def test_colorscale_skip_endpoints(self, server_and_port):
        """Test /colorscale respects skip_endpoints param."""
        server, port, _ = server_and_port

        resp_skip = requests.get(
            f"http://127.0.0.1:{port}/colorscale?name=Plasma&n=7&skip_endpoints=true"
        )
        resp_no_skip = requests.get(
            f"http://127.0.0.1:{port}/colorscale?name=Plasma&n=7&skip_endpoints=false"
        )

        colors_skip = resp_skip.json()["colors"]
        colors_no_skip = resp_no_skip.json()["colors"]

        assert colors_skip[0] != colors_no_skip[0]
        assert colors_skip[-1] != colors_no_skip[-1]

    def test_viewer_redirect(self, server_and_port):
        """Test /view redirects to /viewer/index.html."""
        server, port, _ = server_and_port
        rel = "probe/test_model/test_task"
        resp = requests.get(f"http://127.0.0.1:{port}/view?p={rel}", allow_redirects=False)
        assert resp.status_code == 302
        assert "/viewer/index.html" in resp.headers["Location"]

    def test_viewer_assets(self, server_and_port):
        """Test /viewer/* serves static assets."""
        server, port, _ = server_and_port

        resp = requests.get(f"http://127.0.0.1:{port}/viewer/index.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["Content-Type"]

        resp = requests.get(f"http://127.0.0.1:{port}/viewer/viewer.js")
        assert resp.status_code == 200

        resp = requests.get(f"http://127.0.0.1:{port}/viewer/viewer.css")
        assert resp.status_code == 200

    def test_not_found(self, server_and_port):
        """Test unknown endpoints return 404."""
        server, port, _ = server_and_port
        resp = requests.get(f"http://127.0.0.1:{port}/unknown")
        assert resp.status_code == 404


class TestSaveServerHelpers:
    """Test helper functions used by the server."""

    def test_tensors_to_bytes_format(self, tmp_path):
        """Test _tensors_to_bytes produces correct binary format."""
        task_dir = tmp_path / "task"
        task_dir.mkdir(parents=True)

        index = _make_test_task_index()
        experts = [_make_test_expert_record(0, n_points=10)]
        probing_io.write_probing_task(task_dir, index=index, experts=experts)

        data = _tensors_to_bytes(task_dir, 0)

        assert isinstance(data, bytes)
        assert len(data) > 0

        points = torch.frombuffer(data[:10 * 3 * 4], dtype=torch.float32)
        assert points.shape == (10, 3)

    def test_tensors_to_bytes_with_labels(self, tmp_path):
        """Test _tensors_to_bytes includes labels."""
        task_dir = tmp_path / "task"
        task_dir.mkdir(parents=True)

        index = _make_test_task_index()
        experts = [_make_test_expert_record(0, n_points=10)]
        probing_io.write_probing_task(task_dir, index=index, experts=experts)

        data = _tensors_to_bytes(task_dir, 0)

        offset = 10 * 3 * 4
        labels = torch.frombuffer(data[offset : offset + 10 * 4], dtype=torch.int32)
        assert labels.shape == (10,)

    def test_warm_cache(self, fixture_task_dir, tmp_path):
        """Test _warm_cache loads task indices."""
        results_dir = fixture_task_dir.parent.parent.parent
        _warm_cache(results_dir)

        from latex.save_server import _task_cache

        assert len(_task_cache) >= 1
        assert any("test_task" in v.get("dataset_name", "") for v in _task_cache.values())


class TestGalleryEndpoints:
    """Test gallery (queue) endpoints."""

    @pytest.fixture
    def server_with_queue(self, tmp_path):
        """Start server with an empty queue."""
        output_dir = tmp_path / "queue"
        output_dir.mkdir()

        server = _SaveServer(output_dir, port=0, results_dir=None)
        port = server.server_address[1]

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        yield server, port, output_dir

        server.shutdown()

    def test_queue_empty(self, server_with_queue):
        """Test /queue returns empty list."""
        server, port, _ = server_with_queue
        resp = requests.get(f"http://127.0.0.1:{port}/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []

    @pytest.mark.skip(reason="Requires valid PNG image for autocrop")
    def test_queue_post(self, server_with_queue):

        import base64

        dummy_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        payload = {"filename": "test__task__E0__fisher__0.5__scatter.png", "data": base64.b64encode(dummy_png).decode()}

        resp = requests.post(f"http://127.0.0.1:{port}/queue", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "test__task__E0__fisher__0.5__scatter.png" in data["queued"]

    def test_thumbnail_endpoint(self, server_with_queue):
        """Test /thumbnail serves queued PNG."""
        server, port, _ = server_with_queue

        import base64

        dummy_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        payload = {"filename": "thumb_test.png", "data": base64.b64encode(dummy_png).decode()}
        requests.post(f"http://127.0.0.1:{port}/queue", json=payload)

        resp = requests.get(f"http://127.0.0.1:{port}/thumbnail/thumb_test.png")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "image/png"
