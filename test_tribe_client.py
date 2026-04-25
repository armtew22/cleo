"""Tests for the remote TRIBE backend client.

No network — every HTTP call is mocked via httpx.MockTransport so the tests run
deterministically and offline. Run: ``python -m unittest test_tribe_client``
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from cleo.tribe_client import TribeBackendError, TribeClient


def _build_mock_client(handler) -> TribeClient:
    """Wire an httpx.MockTransport into TribeClient by patching httpx.Client."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    client = TribeClient("http://test-host:8004")
    # Patch is applied per-test via context manager around individual calls.
    client._patch_factory = _factory  # type: ignore[attr-defined]
    return client


class TestHealth(unittest.TestCase):
    def test_health_returns_payload(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.url.path, "/v1/health")
            return httpx.Response(200, json={"status": "ok", "inference": "gpu"})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            out = client.health()
        self.assertEqual(out["inference"], "gpu")


class TestSubmit(unittest.TestCase):
    def setUp(self) -> None:
        # Make a tiny fake video file
        self.video = Path("/tmp/cleo_test_video.mp4")
        self.video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def tearDown(self) -> None:
        self.video.unlink(missing_ok=True)

    def test_submit_returns_job_id(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/v1/runs")
            captured["body"] = req.content
            return httpx.Response(202, json={"job_id": "abc123", "status": "queued"})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            jid = client.submit(self.video, "a busy room")
        self.assertEqual(jid, "abc123")
        # multipart body should contain caption and filename
        self.assertIn(b"a busy room", captured["body"])
        self.assertIn(b"cleo_test_video.mp4", captured["body"])

    def test_submit_retries_on_503(self) -> None:
        attempts = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, headers={"Retry-After": "0"},
                                      json={"error": {"code": "QUEUE_FULL", "message": "full"}})
            return httpx.Response(202, json={"job_id": "retry-ok", "status": "queued"})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            jid = client.submit(self.video, "x")
        self.assertEqual(jid, "retry-ok")
        self.assertEqual(attempts["n"], 2)

    def test_submit_raises_on_non_2xx(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"error": {"code": "VALIDATION", "message": "bad"}})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with self.assertRaises(TribeBackendError) as ctx:
                client.submit(self.video, "x")
        self.assertIn("VALIDATION", str(ctx.exception))

    def test_submit_raises_when_unreachable(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with self.assertRaises(TribeBackendError) as ctx:
                client.submit(self.video, "x")
        self.assertIn("could not reach", str(ctx.exception))


class TestWait(unittest.TestCase):
    def test_wait_returns_done_payload(self) -> None:
        # Sequence of GET responses: queued → running → done
        states = iter([
            {"status": "queued"},
            {"status": "running"},
            {
                "status": "done",
                "report": {"top_regions": [{"name": "L_V1_ROI", "z_score": 2.1}]},
                "mesh": "/v1/runs/abc/mesh",
            },
        ])

        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.url.path, "/v1/runs/abc")
            return httpx.Response(200, json=next(states))

        client = _build_mock_client(handler)
        seen: list[str] = []
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with patch("cleo.tribe_client.time.sleep"):  # don't actually sleep
                job = client.wait("abc", poll_interval=0.0, on_status=lambda s, _: seen.append(s))
        self.assertEqual(job["status"], "done")
        self.assertEqual(seen, ["queued", "running", "done"])
        self.assertEqual(job["mesh"], "/v1/runs/abc/mesh")

    def test_wait_raises_on_failed(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "status": "failed",
                "error": {"code": "INFERENCE_FAILURE", "message": "CUDA OOM"},
            })

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with self.assertRaises(TribeBackendError) as ctx:
                client.wait("abc", poll_interval=0.0)
        self.assertIn("INFERENCE_FAILURE", str(ctx.exception))
        self.assertIn("CUDA OOM", str(ctx.exception))

    def test_wait_times_out(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "running"})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with patch("cleo.tribe_client.time.sleep"):
                with self.assertRaises(TribeBackendError) as ctx:
                    client.wait("abc", poll_interval=0.0, timeout=0.0)
        self.assertIn("did not complete", str(ctx.exception))

    def test_get_job_404_raises(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"code": "JOB_NOT_FOUND"}})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            with self.assertRaises(TribeBackendError):
                client.get_job("nope")


class TestCancel(unittest.TestCase):
    def test_cancel_returns_true_on_204(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.method, "DELETE")
            return httpx.Response(204)

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            self.assertTrue(client.cancel("abc"))

    def test_cancel_returns_false_on_409(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"error": {"code": "JOB_NOT_CANCELLABLE"}})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            self.assertFalse(client.cancel("abc"))


class TestMesh(unittest.TestCase):
    def test_fetch_mesh_manifest(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.url.path, "/v1/runs/abc/mesh")
            return httpx.Response(200, json={
                "job_id": "abc",
                "artifacts": {
                    "meta":     "/v1/runs/abc/mesh/meta",
                    "colors":   "/v1/runs/abc/mesh/colors",
                    "vertices": "/v1/runs/abc/mesh/vertices",
                    "faces":    "/v1/runs/abc/mesh/faces",
                },
            })

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            man = client.fetch_mesh_manifest("abc")
        self.assertIn("artifacts", man)
        self.assertEqual(man["artifacts"]["colors"], "/v1/runs/abc/mesh/colors")

    def test_fetch_mesh_artifact_returns_bytes(self) -> None:
        payload = b"\x00" * 81936  # 20484 * 4

        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.url.path, "/v1/runs/abc/mesh/colors")
            return httpx.Response(200, content=payload,
                                  headers={"Content-Type": "application/octet-stream"})

        client = _build_mock_client(handler)
        with patch("cleo.tribe_client.httpx.Client", side_effect=client._patch_factory):
            data = client.fetch_mesh_artifact("abc", "colors")
        self.assertEqual(len(data), 81936)

    def test_fetch_mesh_artifact_bad_kind(self) -> None:
        client = _build_mock_client(lambda req: httpx.Response(200))
        with self.assertRaises(ValueError):
            client.fetch_mesh_artifact("abc", "bogus")


if __name__ == "__main__":
    unittest.main()
