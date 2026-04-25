"""JobStore unit tests.

The JobStore is an in-memory async-safe registry of submitted jobs. It is
agnostic to the worker — submit() enqueues a payload (whatever shape) onto an
asyncio.Queue and assigns a job_id; transitions are driven externally by the
worker via mark_running / mark_done / mark_failed / mark_cancelled.

Persistence upgrade path is documented in jobs.py — for v1 it's a dict.
"""
from __future__ import annotations

import asyncio

import pytest

from tribe_backend.api.jobs import (
    Job,
    JobStore,
    JobStatus,
    QueueFull,
    UnknownJob,
    NotCancellable,
)


@pytest.fixture
def store():
    return JobStore(max_queue=4)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_submit_assigns_id_and_returns_queued():
    async def go():
        store = JobStore(max_queue=4)
        job_id = await store.submit({"text": "hi"})
        job = await store.get(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.status == "queued"
        assert job.submitted_at is not None
        assert job.started_at is None
        assert job.finished_at is None
        assert job.report is None
        assert job.error is None
    _run(go())


def test_unknown_job_raises():
    async def go():
        store = JobStore(max_queue=4)
        with pytest.raises(UnknownJob):
            await store.get("nope")
    _run(go())


def test_submit_full_queue_raises_queue_full():
    async def go():
        store = JobStore(max_queue=2)
        await store.submit({"i": 0})
        await store.submit({"i": 1})
        with pytest.raises(QueueFull):
            await store.submit({"i": 2})
    _run(go())


def test_dequeue_returns_payload_and_id_and_marks_running():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({"text": "hello"})
        # Worker drains the queue:
        got_id, payload = await store.dequeue()
        assert got_id == jid
        assert payload == {"text": "hello"}
        # Job is still queued until mark_running is called:
        job = await store.get(jid)
        assert job.status == "queued"
        await store.mark_running(jid)
        job = await store.get(jid)
        assert job.status == "running"
        assert job.started_at is not None
    _run(go())


def test_mark_done_records_report():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({})
        await store.dequeue()
        await store.mark_running(jid)
        await store.mark_done(jid, report={"top_regions": [], "text": "x", "method": "z", "z_threshold": 1.0})
        job = await store.get(jid)
        assert job.status == "done"
        assert job.finished_at is not None
        assert job.report == {"top_regions": [], "text": "x", "method": "z", "z_threshold": 1.0}
    _run(go())


def test_mark_failed_records_error_envelope():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({})
        await store.dequeue()
        await store.mark_running(jid)
        await store.mark_failed(jid, code="INFERENCE_FAILURE", message="boom")
        job = await store.get(jid)
        assert job.status == "failed"
        assert job.error == {"code": "INFERENCE_FAILURE", "message": "boom"}
        assert job.finished_at is not None
    _run(go())


def test_cancel_queued_job_succeeds():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({})
        ok = await store.cancel(jid)
        assert ok is True
        job = await store.get(jid)
        assert job.status == "cancelled"
        # The queued payload is still in the queue but the worker should skip it.
        got_id, _ = await store.dequeue()
        assert got_id == jid
        # Worker calls mark_running which detects cancellation and refuses:
        with pytest.raises(NotCancellable):
            await store.mark_running(jid)
    _run(go())


def test_cancel_running_job_raises_not_cancellable():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({})
        await store.dequeue()
        await store.mark_running(jid)
        with pytest.raises(NotCancellable):
            await store.cancel(jid)
    _run(go())


def test_cancel_done_job_raises_not_cancellable():
    async def go():
        store = JobStore(max_queue=4)
        jid = await store.submit({})
        await store.dequeue()
        await store.mark_running(jid)
        await store.mark_done(jid, report={"x": 1})
        with pytest.raises(NotCancellable):
            await store.cancel(jid)
    _run(go())


def test_cancel_unknown_raises_unknown():
    async def go():
        store = JobStore(max_queue=4)
        with pytest.raises(UnknownJob):
            await store.cancel("nope")
    _run(go())


def test_job_id_is_unique():
    async def go():
        store = JobStore(max_queue=10)
        ids = set()
        for _ in range(5):
            ids.add(await store.submit({}))
        assert len(ids) == 5
    _run(go())
