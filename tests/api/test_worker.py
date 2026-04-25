"""Worker tests — drive the JobStore + a stub ControlUnit through transitions.

The real ControlUnit.process_window is sync and CPU/GPU-bound; the worker runs
it in a thread (asyncio.to_thread). We use stub callables here so the tests
don't need a fixture mp4.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from tribe_backend.api.jobs import JobStore
from tribe_backend.api.worker import run_worker, WorkerStop


class _StubInferenceFailure(Exception):
    pass


def _run(coro):
    return asyncio.run(coro)


def test_worker_processes_one_job_to_done():
    async def go():
        store = JobStore(max_queue=4)

        def fake_process(**payload):
            return {"top_regions": [], "text": payload["text"], "method": "z", "z_threshold": 1.0}

        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, fake_process, stop))

        jid = await store.submit({"text": "hello", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})

        # Wait for the worker to finish the job.
        for _ in range(50):
            await asyncio.sleep(0.05)
            job = await store.get(jid)
            if job.status in ("done", "failed"):
                break
        assert job.status == "done"
        assert job.report["text"] == "hello"
        assert job.started_at is not None
        assert job.finished_at is not None

        stop.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
    _run(go())


def test_worker_marks_failed_on_inference_failure():
    async def go():
        store = JobStore(max_queue=4)

        def boom(**_payload):
            from tribe_backend.control.factory import InferenceFailure
            raise InferenceFailure("nope")

        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, boom, stop))

        jid = await store.submit({"text": "x", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})

        for _ in range(50):
            await asyncio.sleep(0.05)
            job = await store.get(jid)
            if job.status in ("done", "failed"):
                break
        assert job.status == "failed"
        assert job.error["code"] == "INFERENCE_FAILURE"
        assert "nope" in job.error["message"]

        stop.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
    _run(go())


def test_worker_marks_failed_on_unhandled_exception():
    async def go():
        store = JobStore(max_queue=4)

        def crash(**_payload):
            raise RuntimeError("kaboom")

        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, crash, stop))

        jid = await store.submit({"text": "x", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})

        for _ in range(50):
            await asyncio.sleep(0.05)
            job = await store.get(jid)
            if job.status == "failed":
                break
        assert job.status == "failed"
        assert job.error["code"] == "internal_error"

        stop.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
    _run(go())


def test_worker_skips_cancelled_jobs():
    async def go():
        store = JobStore(max_queue=4)

        invocations: list[dict] = []

        def fake_process(**payload):
            invocations.append(payload)
            return {"top_regions": [], "text": "x", "method": "z", "z_threshold": 1.0}

        stop = asyncio.Event()

        # Submit + cancel immediately, BEFORE starting the worker — so the
        # worker dequeues an already-cancelled job and must skip it.
        jid = await store.submit({"text": "x", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})
        await store.cancel(jid)

        worker_task = asyncio.create_task(run_worker(store, fake_process, stop))

        # Give the worker time to dequeue + skip.
        await asyncio.sleep(0.2)

        job = await store.get(jid)
        assert job.status == "cancelled"
        assert invocations == []

        stop.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
    _run(go())


def test_worker_processes_jobs_serially():
    """With one worker task, jobs run one at a time — never overlapping."""
    async def go():
        store = JobStore(max_queue=10)
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        def slow_process(**payload):
            nonlocal active, max_active
            # We're inside asyncio.to_thread — use a sync sleep to simulate work.
            time.sleep(0.05)
            return {"top_regions": [], "text": payload["text"], "method": "z", "z_threshold": 1.0}

        # Sentinel: increment "active" via an event-loop-scheduled hook just
        # before each process call. We do this by wrapping run_worker behavior
        # via a side-channel: here we simply rely on the fact that jobs are
        # processed serially because there's only one worker task.
        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, slow_process, stop))

        ids = []
        for i in range(3):
            ids.append(await store.submit({"text": f"job{i}", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0}))

        # Wait for all to complete.
        for _ in range(100):
            await asyncio.sleep(0.05)
            statuses = [(await store.get(jid)).status for jid in ids]
            if all(s == "done" for s in statuses):
                break
        for jid in ids:
            j = await store.get(jid)
            assert j.status == "done"

        # Verify ordering: started_at is monotonic (next started after prev finished).
        jobs = [await store.get(jid) for jid in ids]
        for prev, curr in zip(jobs, jobs[1:]):
            assert prev.finished_at <= curr.started_at, (
                f"jobs ran in parallel: {prev.finished_at} > {curr.started_at}"
            )

        stop.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
    _run(go())


def test_worker_stops_on_stop_event_with_drain_timeout():
    """When stop is signaled, the worker drains within drain_timeout and exits cleanly."""
    async def go():
        store = JobStore(max_queue=4)

        def fast(**payload):
            return {"top_regions": [], "text": "x", "method": "z", "z_threshold": 1.0}

        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, fast, stop, drain_timeout=1.0))

        # Submit a couple of jobs then stop.
        await store.submit({"text": "a", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})
        await store.submit({"text": "b", "video": None, "audio": None, "audio_sr": 16000, "video_fps": 25.0})
        await asyncio.sleep(0.1)
        stop.set()

        # Worker should exit without raising.
        await asyncio.wait_for(worker_task, timeout=3.0)
    _run(go())
