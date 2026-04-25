"""FastAPI app factory.

`get_app()` is the uvicorn `--factory` entrypoint. On startup it constructs
ONE ControlUnit (via `default_control_unit`) and a single JobStore + worker
task that drains submitted jobs serially. The CORS middleware is mounted
from settings, and the §9 error envelope handlers are installed.

Single-GPU constraint: exactly one worker per process. Running multiple
uvicorn workers would each try to load weights into the same GPU and OOM —
the worker is process-local and the JobStore is in-memory.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tribe_backend.api import worker as worker_mod
from tribe_backend.api.errors import install_handlers
from tribe_backend.api.jobs import JobStore
from tribe_backend.api.mesh_routes import mesh_router
from tribe_backend.api.routes import router
from tribe_backend.api.settings import Settings
from tribe_backend.control.factory import build_inference, default_control_unit

logger = logging.getLogger(__name__)


def _build_fake_fixture():
    """Synthetic TribeOutput for the dev/fake inference path.

    Imported from tests.conftest because the synthetic factory already lives
    there and re-implementing it here would drift. The api/ package only
    pulls this in when TRIBE_INFERENCE=fake — never in production.
    """
    from tests.conftest import make_synthetic
    return make_synthetic()


def _make_process_fn(cu):
    """Wrap ControlUnit.process_window into the worker's expected shape:
    a sync callable returning a JSON-ready dict (Report.to_json())."""
    def process(**payload):
        report = cu.process_window(**payload)
        if report is None:
            from tribe_backend.api.errors import APIError  # local import; avoid cycle
            # Surface as InferenceFailure-equivalent in the worker.
            from tribe_backend.control.factory import InferenceFailure
            raise InferenceFailure("parcellation produced no report")
        return _json.loads(report.to_json())
    return process


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    fixture = _build_fake_fixture() if settings.inference == "fake" else None
    inference = build_inference(settings.inference, fake_fixture=fixture)
    cu = default_control_unit(out_dir=settings.out_dir, inference=inference)
    app.state.control_unit = cu

    jobs = JobStore(max_queue=settings.queue_depth)
    app.state.jobs = jobs

    stop_event = asyncio.Event()
    app.state.stop_event = stop_event
    process_fn = _make_process_fn(cu)
    worker_task = asyncio.create_task(
        worker_mod.run_worker(jobs, process_fn, stop_event, drain_timeout=settings.drain_timeout_s)
    )
    app.state.worker_task = worker_task

    logger.info(
        "API ready: inference=%s out_dir=%s queue_depth=%d",
        settings.inference, settings.out_dir, settings.queue_depth,
    )
    try:
        yield
    finally:
        stop_event.set()
        # Wait for worker to drain and exit; cap at drain_timeout + slack.
        try:
            await asyncio.wait_for(worker_task, timeout=settings.drain_timeout_s + 2.0)
        except asyncio.TimeoutError:
            logger.warning("worker did not exit within drain timeout; cancelling")
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, BaseException):
                pass
        app.state.control_unit = None


def get_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title="tribe-backend",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_handlers(app)
    app.include_router(router)
    app.include_router(mesh_router)
    return app
