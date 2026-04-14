"""FastAPI gateway exposing the geo-alpha quant + ML capabilities."""

from .app import create_app

__all__ = ["create_app"]
