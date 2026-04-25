"""CLI entrypoint: ``python -m tribe_backend.api``.

Wraps `uvicorn.run` with the only configuration that's safe for this
project's single-GPU constraint:

* ``--workers 1`` is mandatory. Multiple uvicorn worker processes would
  each try to load the tribev2 weights into the same GPU and OOM. The
  in-memory JobStore is process-local for the same reason — multi-worker
  would split jobs across stores and break status polling.
* Host defaults to ``0.0.0.0`` so a Slurm allocation's ``NodeAddr`` is
  reachable from the head node / SSH tunnel; override with ``HOST`` env.
* Port defaults to ``8000``; override with ``PORT`` env.

Usage:

    TRIBE_INFERENCE=gpu CUDA_VISIBLE_DEVICES=0 \\
        python -m tribe_backend.api

For an A6000 box the load takes ~30-60s; the first ``/v1/health`` request
should be issued only after uvicorn logs ``Uvicorn running on ...``.
"""
from __future__ import annotations

import os
import sys


def _parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise SystemExit(f"{name} must be an integer; got {raw!r}") from e


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover — declared in pyproject deps
        print(f"uvicorn not importable: {e}", file=sys.stderr)
        raise SystemExit(2)

    host = os.environ.get("HOST", "0.0.0.0")
    port = _parse_int("PORT", 8000)
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    # ``--factory`` semantics: uvicorn calls get_app() once per worker. We
    # enforce workers=1 explicitly so an operator who tries to scale this
    # process gets the failure at startup, not at first OOM.
    uvicorn.run(
        "tribe_backend.api.app:get_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
        log_level=log_level,
        # No reload — model load is too expensive to bounce on file changes.
        reload=False,
    )


if __name__ == "__main__":
    main()
