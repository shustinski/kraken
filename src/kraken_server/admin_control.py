"""Private inherited stdin control; no network shutdown endpoint."""

from __future__ import annotations

import asyncio
import sys
import threading


def run_controlled(config: object) -> None:
    import uvicorn

    server = uvicorn.Server(config)

    def control() -> None:
        # EOF also means that the owning Admin process has gone away.
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        server.should_exit = True

    async def run() -> None:
        async def announce() -> None:
            while not server.started and not server.should_exit:
                await asyncio.sleep(0.05)
            if server.started:
                print("KRAKEN_ADMIN_READY", flush=True)

        ready = asyncio.create_task(announce())
        try:
            await server.serve()
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    threading.Thread(target=control, name="admin-control", daemon=True).start()
    asyncio.run(run())
