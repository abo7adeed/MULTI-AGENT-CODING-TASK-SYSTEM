"""API layer: FastAPI application, schemas, dependency wiring and routes."""

from app.api.deps import Container, build_container, get_container, set_container
from app.api.main import app, create_app
from app.api.routes import router

__all__ = [
    "Container",
    "app",
    "build_container",
    "create_app",
    "get_container",
    "router",
    "set_container",
]
