"""
FastAPI application.

Thin bootstrap: build the container, mount the router, install CORS and a
lifespan hook. Everything else lives in `app.api.routes`.

Run with:
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import contextlib
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import get_container, set_container
from app.api.routes import router
from app.config import get_settings
from app.logging_config import configure_logging, get_logger

logger = get_logger("app.api")

DESCRIPTION = """
Multi-agent coding orchestration platform.

Give it a software engineering request. It inspects the repository, decomposes
the work into a dependency graph, dispatches specialised coding agents into
isolated Git worktrees, merges their branches (resolving conflicts with a merge
agent), runs the test suite, and reports the result.

Orchestrations run in the background. Subscribe to
`GET /orchestrations/{id}/stream` for live task and agent events.
"""


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = get_container()
    logger.info(
        "API starting",
        extra={
            "provider": container.provider.name,
            "model": container.provider.model,
            "agents": len(container.registry),
        },
    )
    # Ask the provider whether it can serve its model while the operator is
    # still watching the log. A wrong model name or a rejected key must not
    # wait for a run to surface it, one wasted retry per task later.
    probe = await container.probe_provider()
    if probe is not None:
        if probe.get("reachable") and probe.get("model_available"):
            logger.info(
                "Provider ready",
                extra={
                    "provider": probe.get("provider"),
                    "model": probe.get("model"),
                    "base_url": probe.get("base_url"),
                    "models": len(probe.get("models") or []),
                },
            )
        else:
            logger.error(
                "Provider is not ready: %s",
                probe.get("error") or "the configured model could not be confirmed",
                extra={"provider": probe.get("provider"), "model": probe.get("model")},
            )
    try:
        yield
    finally:
        logger.info("API shutting down")
        with contextlib.suppress(Exception):
            await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    application = FastAPI(
        title="Multi-Agent Coding Task System",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Accel-Buffering"],
    )

    application.include_router(router)

    @application.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover
        logger.exception("Unhandled API error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": f"internal error: {type(exc).__name__}"},
        )

    @application.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "Multi-Agent Coding Task System",
            "docs": "/docs",
            "health": "/health",
            "endpoints": [
                "POST   /projects",
                "GET    /projects",
                "GET    /projects/{id}",
                "POST   /orchestrations",
                "GET    /orchestrations",
                "GET    /orchestrations/{id}",
                "GET    /orchestrations/{id}/dag",
                "GET    /orchestrations/{id}/agents",
                "GET    /orchestrations/{id}/logs",
                "GET    /orchestrations/{id}/stream",
                "GET    /orchestrations/{id}/changes",
                "GET    /orchestrations/{id}/diff",
                "GET    /orchestrations/{id}/test-results",
                "GET    /orchestrations/{id}/report",
                "POST   /orchestrations/{id}/pause",
                "POST   /orchestrations/{id}/resume",
                "POST   /orchestrations/{id}/cancel",
                "POST   /orchestrations/{id}/tasks",
                "GET    /tasks/{id}",
                "POST   /agents/{id}/retry",
                "GET    /agents",
                "GET    /system/info",
                "GET    /system/models",
            ],
        }

    return application


app = create_app()


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    import uvicorn

    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="app.api.main", description="Run the multi-agent orchestration API."
    )
    # `Settings` remains the source of truth; these only exist so that passing
    # --port is not silently ignored, which reads as "the flag is broken".
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument(
        "--reload", action="store_true", help="Restart on source changes (development)."
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
