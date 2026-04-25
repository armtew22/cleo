"""FastAPI app factory.

`get_app()` is the uvicorn `--factory` entrypoint. On startup it constructs
ONE ControlUnit (via `default_control_unit`) and stashes it on `app.state`,
along with the resolved Settings. The CORS middleware is mounted from
settings, and the §9 error envelope handlers are installed.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tribe_backend.api.errors import install_handlers
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    fixture = _build_fake_fixture() if settings.inference == "fake" else None
    inference = build_inference(settings.inference, fake_fixture=fixture)
    cu = default_control_unit(out_dir=settings.out_dir, inference=inference)
    app.state.control_unit = cu
    logger.info(
        "API ready: inference=%s out_dir=%s", settings.inference, settings.out_dir
    )
    try:
        yield
    finally:
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
    return app
