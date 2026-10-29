"""Project event websocket."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal


def register(ctx: HttpContext) -> None:
    app = ctx.app
    account_store = ctx.account_store
    development = ctx.development
    hub = ctx.hub
    session_resolver = ctx.session_resolver
    WebSocketDisconnect = ctx.WebSocketDisconnect
    run_in_threadpool = ctx.run_in_threadpool
    has_project_access = ctx.has_project_access

    @app.websocket(f"{API_PREFIX}/ws")
    async def websocket(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""

        def resolve_session() -> SessionPrincipal | None:
            if session_resolver is not None:
                return session_resolver(token)
            if account_store is not None:
                account = account_store.resolve_session(token)
                if account is None:
                    return None
                return SessionPrincipal(str(account.account_id), "local", token)
            if development and token:
                return SessionPrincipal(token, "development", token)
            return None

        session = await run_in_threadpool(resolve_session)
        if session is None:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        hub.register(websocket)
        await websocket.send_json({"type": "connected", "api_version": "v1"})
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
                except TimeoutError:
                    still_valid = await run_in_threadpool(resolve_session) is not None
                    if not still_valid:
                        await websocket.close(code=4401)
                        return
                    continue
                kind = str(message.get("type", ""))
                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                elif kind == "subscribe":
                    project_id = message.get("project_id")
                    catalog = bool(message.get("catalog"))
                    if project_id is not None and not await run_in_threadpool(
                        has_project_access, str(project_id), session
                    ):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": "authorization.denied",
                                "project_id": str(project_id),
                            }
                        )
                        continue
                    hub.subscribe(
                        websocket,
                        project_id=None if project_id is None else str(project_id),
                        catalog=catalog,
                        catalog_visible=(lambda project_id, subject=session: has_project_access(project_id, subject))
                        if catalog
                        else None,
                    )
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "project_id": None if project_id is None else str(project_id),
                            "catalog": bool(message.get("catalog")),
                        }
                    )
                elif kind == "unsubscribe":
                    project_id = message.get("project_id")
                    hub.unsubscribe(
                        websocket,
                        project_id=None if project_id is None else str(project_id),
                        catalog=bool(message.get("catalog")),
                    )
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "project_id": None if project_id is None else str(project_id),
                            "catalog": bool(message.get("catalog")),
                        }
                    )
        except WebSocketDisconnect:
            hub.unregister(websocket)
            return
        finally:
            hub.unregister(websocket)
