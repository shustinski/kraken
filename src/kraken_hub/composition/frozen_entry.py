"""Route the bundled Desktop's private Agent entry without invoking a Python CLI."""

from __future__ import annotations

import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--kraken-stitch-worker"]:
        if getattr(sys, "frozen", False):
            from kraken_manager.infrastructure.compute_runtime import initialize_packaged_compute_runtime
            initialize_packaged_compute_runtime()
        from kraken_manager.infrastructure.stitch_worker import main as stitch_main
        return stitch_main()
    if arguments[:1] == ["--kraken-agent"]:
        from kraken_agent.__main__ import main as agent_main

        sys.argv = [sys.argv[0], *arguments[1:]]
        return agent_main()
    if getattr(sys, 'frozen', False):
        from kraken_manager.infrastructure.compute_runtime import initialize_packaged_compute_runtime
        initialize_packaged_compute_runtime()
    from kraken_hub.composition.app import main as desktop_main

    return desktop_main(arguments)
