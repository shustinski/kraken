"""FastAPI application factory; adapter selection stays in the composition root."""

from kraken_server.interface.http.factory import create_app
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal

__all__ = ["API_PREFIX", "SessionPrincipal", "create_app"]
