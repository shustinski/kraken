"""Upgrade the exact live-server schema gap on an isolated test schema."""
import json
from uuid import uuid4

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
import sqlalchemy as sa

from kraken_server.agent_gateway import PostgresAgentGateway

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_execution_state_upgrade_preserves_existing_job_and_unblocks_gateway(database):
    services, _, _, _ = database
    engine = services.engine
    command.downgrade(Config("alembic.ini", attributes={"configure_logger": False}), "20261025_0010")
    job_id, project_id = str(uuid4()), str(uuid4())
    with engine.begin() as connection:
        connection.execute(sa.text("""
            INSERT INTO background_jobs (job_id,project_id,state,payload,created_at,updated_at,telemetry)
            VALUES (:job,:project,'queued',CAST(:payload AS json),CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CAST(:telemetry AS json))
        """), {"job": job_id, "project": project_id, "payload": json.dumps({"capability": "karakal.analyze.v1"}),
                 "telemetry": json.dumps({"generation": 1, "cpu": 2})})
        before = dict(connection.execute(sa.text("SELECT * FROM background_jobs WHERE job_id=:job"),
                                         {"job": job_id}).mappings().one())
    gateway = PostgresAgentGateway(engine, tokens=None, blobs=None)
    with pytest.raises(sa.exc.ProgrammingError, match="execution_state"):
        gateway.project_execution_states(project_id)
    command.upgrade(Config("alembic.ini", attributes={"configure_logger": False}), "head")
    with engine.connect() as connection:
        after = dict(connection.execute(sa.text("SELECT * FROM background_jobs WHERE job_id=:job"),
                                        {"job": job_id}).mappings().one())
        # The chain must land on the current head, whatever revision that is;
        # a hardcoded id here silently rotted once 20261110_0012 landed.
        head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
        assert connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one() == head
    assert after.pop("execution_state") is None
    assert after == before
    assert gateway.project_execution_states(project_id) == {job_id: {}}
