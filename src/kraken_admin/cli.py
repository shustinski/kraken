"""Packaged Kraken Server setup, recovery, and machine-token commands."""

from __future__ import annotations
import argparse
import getpass
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4
from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import (
    Argon2PasswordHasher,
    LocalAccountStore,
    ScryptPasswordHasher,
)
from kraken_server.configuration import (
    ServerConfig,
    default_config_path,
    default_local_config_path,
    run_migrations,
    write_config,
)
from kraken_server.provisioning import (
    new_postgres_provisioning,
    provision_postgres,
    remove_provisioned_postgres,
)

from .cli_parts.install_service import (
    _HelpFormatter,
    _ArgumentParser,
    _secret,
    _text_value,
    _integer_value,
    _password,
    _postgres_store,
    _database_url,
    _bootstrap,
    _data_root,
    _default_blob_root,
    _default_local_blob_root,
    _default_server_executable,
    _default_blob_gateway_executable,
    _blob_gateway_public_url,
    _install_service,
)
from .cli_parts.initialize_local_server import _initialize_local_server
from .cli_parts.doctor import _doctor
from .cli_parts.api_request import (
    _record_migration_check,
    _record_health_check,
    _record_tls_check,
    _record_service_check,
    _service_status,
    _uninstall_service,
    _api_request,
    _server_session,
    _create_project,
)
from .cli_parts.list_projects import _list_projects
from .cli_parts.parser import _parser
from .cli_parts.run_local_admin import _run_local_admin
from .cli_parts.execute import _execute
from .cli_parts.main import main

if __name__ == "__main__":
    raise SystemExit(main())

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _HelpFormatter,
    _ArgumentParser,
    _secret,
    _text_value,
    _integer_value,
    _password,
    _postgres_store,
    _database_url,
    _bootstrap,
    _data_root,
    _default_blob_root,
    _default_local_blob_root,
    _default_server_executable,
    _default_blob_gateway_executable,
    _blob_gateway_public_url,
    _install_service,
    _initialize_local_server,
    _doctor,
    _record_migration_check,
    _record_health_check,
    _record_tls_check,
    _record_service_check,
    _service_status,
    _uninstall_service,
    _api_request,
    _server_session,
    _create_project,
    _list_projects,
    _parser,
    _run_local_admin,
    _execute,
    main,
)
