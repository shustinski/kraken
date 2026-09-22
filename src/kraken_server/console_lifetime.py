"""Server entry kept for callers that already import the console lifetime hook."""

from kraken_core.process_lifetime import bind_child_process_lifetime as bind_console_lifetime

__all__ = ["bind_console_lifetime"]
