"""cli: list projects."""
from __future__ import annotations
import argparse
import json

from kraken_admin.cli_parts.api_request import (
    _api_request,
    _server_session,
)

def _list_projects(args: argparse.Namespace) -> None:
    server, token = _server_session(args)
    response = _api_request(server, "GET", "/api/v1/projects", token=token)
    items = response.get("items", []) if isinstance(response, dict) else []
    if args.json:
        print(json.dumps(items, ensure_ascii=False))
        return
    if not items:
        print("Доступных проектов нет.")
        return
    for project in items:
        print(
            f"{project.get('project_id', '')}  {project.get('name', '')}  "
            f"{project.get('width', '?')}x{project.get('height', '?')}"
        )
