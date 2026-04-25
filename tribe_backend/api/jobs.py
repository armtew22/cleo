"""In-memory async-safe JobStore for the Phase B job queue.

Lifecycle states (per plan §3):

    queued ──▶ running ──▶ done
       │           │
       │           └──▶ failed
       └──▶ cancelled  (only from queued; running is uninterruptible in v1)

Persistence upgrade path (NOT implemented in v1):
    - v1 (now): in-process dict, lost on restart.
    - v2: SQLite at out_dir/jobs.db — one file, restartable, single-node.
    - v3: Postgres + Redis queue if/when horizontal scale is needed.

Concurrency model: a single asyncio.Lock guards the dict; an asyncio.Queue
holds (job_id, payload) tuples for the worker to drain. The worker is a
single asyncio task — there is no multi-consumer contention to worry about.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Tuple

__all__ = [
    "Job",
    "JobStatus",
    "JobStore",
    "QueueFull",
    "UnknownJob",
    "NotCancellable",
]


JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class JobStoreError(Exception):
    """Base class for JobStore exceptions."""


class UnknownJob(JobStoreError):
    """Raised when a job_id is not present in the store."""


class QueueFull(JobStoreError):
    """Raised when submit() is called and the queue is at maxsize."""


class NotCancellable(JobStoreError):
    """Raised when cancel() is called on a job that is no longer queued.

    Also raised by mark_running() when the job has been cancelled — the worker
    catches this and skips the job.
    """


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    job_id: str
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    report: Optional[dict[str, Any]] = None
    error: Optional[dict[str, str]] = None
    # `payload` is intentionally NOT exposed via get(); it's pulled out of the
    # queue by dequeue() and consumed by the worker.


@dataclass
class _Entry:
    job: Job
    payload: Any


class JobStore:
    """Thread-/task-safe in-memory job registry + queue.

    Parameters
    ----------
    max_queue
        Maximum number of in-flight (queued + running) jobs. Once reached,
        further submit() calls raise QueueFull and the route returns 503 with
        a Retry-After header. Default 4 — single-GPU constraint: only one
        model fits in VRAM, so queueing more than ~4 isn't useful.
    """

    def __init__(self, max_queue: int = 4) -> None:
        self._max = int(max_queue)
        self._jobs: dict[str, _Entry] = {}
        # The queue holds job ids only; payloads live in self._jobs to keep
        # cancellation cheap (we don't have to scan the queue).
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=0)
        # Tracks the count of "live" (queued + running) jobs to enforce max_queue.
        self._live = 0
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------- submit
    async def submit(self, payload: Any) -> str:
        """Enqueue a payload, return the assigned job_id.

        Raises QueueFull if the in-flight job count would exceed max_queue.
        """
        async with self._lock:
            if self._live >= self._max:
                raise QueueFull(
                    f"job queue is full ({self._live}/{self._max} in flight)"
                )
            job_id = uuid.uuid4().hex
            job = Job(job_id=job_id, status="queued", submitted_at=_now())
            self._jobs[job_id] = _Entry(job=job, payload=payload)
            self._live += 1
        # Queue.put is unbounded (we enforce capacity via _live above), but
        # we still await it to remain coroutine-friendly.
        await self._queue.put(job_id)
        return job_id

    # -------------------------------------------------------------- get
    async def get(self, job_id: str) -> Job:
        async with self._lock:
            entry = self._jobs.get(job_id)
        if entry is None:
            raise UnknownJob(job_id)
        return entry.job

    # -------------------------------------------------------------- dequeue
    async def dequeue(self) -> Tuple[str, Any]:
        """Worker entrypoint: await the next job_id, return (id, payload).

        The job's status is NOT changed here — the worker should call
        mark_running() once it actually begins. This separation lets the
        worker check for cancellation between dequeue and start.

        Returns (None, None) sentinel when the queue is closed via close().
        """
        job_id = await self._queue.get()
        if job_id is None:  # close sentinel
            return None, None  # type: ignore[return-value]
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:  # pragma: no cover — submit always inserts
                raise UnknownJob(job_id)
            payload = entry.payload
        return job_id, payload

    async def close(self) -> None:
        """Push a None sentinel so a blocked dequeue() returns immediately.

        Used by the worker shutdown path to unblock its `await dequeue()`
        without losing in-flight items.
        """
        await self._queue.put(None)  # type: ignore[arg-type]

    # -------------------------------------------------------------- transitions
    async def mark_running(self, job_id: str) -> None:
        """Worker calls this when it actually starts processing a job.

        Refuses (raises NotCancellable) if the job was cancelled between
        dequeue and start — the worker should catch and skip.
        """
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJob(job_id)
            if entry.job.status == "cancelled":
                raise NotCancellable(
                    f"job {job_id} was cancelled before it started"
                )
            if entry.job.status != "queued":
                raise NotCancellable(
                    f"job {job_id} is {entry.job.status}, cannot mark running"
                )
            entry.job.status = "running"
            entry.job.started_at = _now()

    async def mark_done(self, job_id: str, *, report: dict[str, Any]) -> None:
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJob(job_id)
            entry.job.status = "done"
            entry.job.finished_at = _now()
            entry.job.report = report
            # Job is no longer in flight.
            self._live = max(0, self._live - 1)

    async def mark_failed(self, job_id: str, *, code: str, message: str) -> None:
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJob(job_id)
            entry.job.status = "failed"
            entry.job.finished_at = _now()
            entry.job.error = {"code": code, "message": message}
            self._live = max(0, self._live - 1)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Returns True on success.

        Raises UnknownJob if id is unknown, NotCancellable if the job has
        already moved past 'queued'.
        """
        async with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJob(job_id)
            if entry.job.status != "queued":
                raise NotCancellable(
                    f"job {job_id} is {entry.job.status}, cannot cancel"
                )
            entry.job.status = "cancelled"
            entry.job.finished_at = _now()
            self._live = max(0, self._live - 1)
        return True

    # -------------------------------------------------------------- introspection
    @property
    def queue_depth(self) -> int:
        """Approximate number of jobs waiting to be picked up by the worker."""
        return self._queue.qsize()

    @property
    def live_count(self) -> int:
        """In-flight job count (queued + running)."""
        return self._live
