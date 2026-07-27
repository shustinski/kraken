PostgreSQL schema migrations for Kraken Server. Set `KRAKEN_DATABASE_URL` and
run `alembic upgrade head`. Event payload evolution is handled by application
upcasters and is intentionally separate from these schema migrations.

