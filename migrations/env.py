from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool


config = context.config
# fileConfig() replaces every root-logger handler (and, by default, disables
# existing loggers). That is fine for the alembic CLI but wipes the caller's
# logging when alembic is embedded programmatically (e.g. the pytest
# `database` fixture or kraken_server.run_migrations). Programmatic callers
# opt out with Config(..., attributes={"configure_logger": False}); the CLI
# keeps the historic behaviour.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

database_url = os.environ.get("KRAKEN_DATABASE_URL")
if not database_url:
    raise RuntimeError("KRAKEN_DATABASE_URL is required")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

