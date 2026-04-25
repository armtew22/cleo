"""Single-consumer asyncio worker that drains the JobStore queue.

Runs as one task spawned from the FastAPI lifespan. For each dequeued job:

    1. mark_running (skipping cancelled jobs cleanly)
    2. await asyncio.to_thread(process_window, **payload)
       — process_window is sync + GPU-bound, so we offload to a thread to
         keep the event loop responsive for status polls / cancel requests.
    3. mark_done(report) on success, mark_failed(envelope) on InferenceFailure
       or any other exception (no traceback leaks to the client).

Single-GPU constraint: there is exactly ONE worker task per process and
exactly one model in VRAM. Running multiple workers would either OOM the
GPU or force expensive weight reloads. The queue maxsize is small (default
4) for the same reason — anything beyond that would just queue up multi-
minute waits.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from tribe_backend.api.jobs import (
    JobStore,
    NotCancellable,
    UnknownJob,
)
from tribe_backend.control.factory import InferenceFailure

__all__ = ["run_worker", "WorkerStop"]

logger = logging.getLogger(__name__)


class WorkerStop(Exception):
    """Internal sentinel raised when the worker should exit its loop."""


async def run_worker(
    store: JobStore,
    process_window: Callable[..., dict[str, Any]],
    stop: asyncio.Event,
    *,
    drain_timeout: float = 5.0,
) -> None:
    """Drain `store` indefinitely, calling `process_window(**payload)` per job.

    Parameters
    ----------
    store
        The JobStore to drain.
    process_window
        Sync callable. Receives the payload kwargs that were passed to
        store.submit(...). Must return a JSON-serializable dict (the report).
        In production this is a thin wrapper around ControlUnit.process_window
        that turns the Report dataclass into its to_json() dict shape.
    stop
        asyncio.Event — when set, the worker drains in-flight work and exits.
    drain_timeout
        Seconds to keep processing after stop is set. Beyond this, the worker
        abandons remaining queued jobs.
    """
    logger.info("worker started")

    async def _stop_watcher() -> None:
        """When stop fires, push the sentinel immediately so the worker stops
        accepting NEW jobs. drain_timeout is enforced separately around the
        in-flight to_thread call below."""
        await stop.wait()
        await store.close()

    watcher = asyncio.create_task(_stop_watcher())
    try:
        while True:
            try:
                job_id, payload = await store.dequeue()
            except asyncio.CancelledError:
                logger.info("worker cancelled while waiting for job")
                raise

            if job_id is None:
                # Sentinel: stop drain expired, exit cleanly.
                logger.info("worker received stop sentinel, exiting")
                return

            # Cheap shortcut: if stop has already fired AND the queue still
            # has items, the drain timer is in charge. We continue processing
            # until the sentinel arrives.

            try:
                await store.mark_running(job_id)
            except NotCancellable:
                # Job was cancelled before it started; skip it cleanly.
                logger.info("worker skipping cancelled job %s", job_id)
                continue
            except UnknownJob:  # pragma: no cover — defensive
                logger.warning("worker dequeued unknown job %s", job_id)
                continue

            try:
                report = await asyncio.to_thread(process_window, **payload)
            except InferenceFailure as e:
                logger.warning("inference failure for job %s: %s", job_id, e)
                await store.mark_failed(
                    job_id,
                    code="INFERENCE_FAILURE",
                    message=str(e) or "inference failed",
                )
                continue
            except Exception:  # noqa: BLE001 — last-resort isolation
                logger.exception("worker crash on job %s", job_id)
                await store.mark_failed(
                    job_id,
                    code="internal_error",
                    message="internal worker error",
                )
                continue

            await store.mark_done(job_id, report=report)
    except asyncio.CancelledError:
        logger.info("worker task cancelled")
        raise
    finally:
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, BaseException):
            pass
        logger.info("worker exiting")
