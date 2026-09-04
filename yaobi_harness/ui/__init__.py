"""Local operator console (stdlib HTTP server + single-file page)."""

from .server import ConsoleService, create_server, serve

__all__ = ["ConsoleService", "create_server", "serve"]
