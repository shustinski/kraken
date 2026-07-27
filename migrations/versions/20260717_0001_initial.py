"""initial event, projection, identity and job schema

Revision ID: 20260717_0001
Revises:
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260717_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "event_streams",
        sa.Column("stream_id", sa.Text(), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.create_index("ix_event_streams_project_id", "event_streams", ["project_id"])
    op.create_table(
        "domain_events",
        sa.Column("position", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False, unique=True),
        sa.Column("stream_id", sa.Text(), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True)),
        sa.Column("actor", sa.JSON(), nullable=False),
        sa.Column("performer_id", sa.Uuid(as_uuid=False)),
        sa.Column("program", sa.JSON()),
        sa.Column("correlation_id", sa.Uuid(as_uuid=False)),
        sa.Column("causation_id", sa.Uuid(as_uuid=False)),
        sa.Column("idempotency_key", sa.Text()),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.UniqueConstraint("stream_id", "revision", name="uq_domain_events_stream_revision"),
    )
    op.create_index("ix_domain_events_project_time", "domain_events", ["project_id", "recorded_at"])
    op.create_index(
        "ix_domain_events_project_type_time", "domain_events", ["project_id", "event_type", "recorded_at"]
    )
    op.create_index(
        "ix_domain_events_project_idempotency", "domain_events", ["project_id", "idempotency_key"]
    )
    op.create_table(
        "command_idempotency",
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("idempotency_key", sa.Text(), primary_key=True),
        sa.Column("event_ids", sa.JSON(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "transactional_outbox",
        sa.Column("outbox_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False, unique=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "projections_current",
        sa.Column("kind", sa.Text(), primary_key=True),
        sa.Column("entity_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False)),
        sa.Column("frame_id", sa.Uuid(as_uuid=False)),
        sa.Column("parent_id", sa.Uuid(as_uuid=False)),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_projections_current_viewport", "projections_current", ["project_id", "layer_id", "frame_id"]
    )
    op.create_table(
        "projections_temporal",
        sa.Column("sequence", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False)),
        sa.Column("frame_id", sa.Uuid(as_uuid=False)),
        sa.Column("parent_id", sa.Uuid(as_uuid=False)),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_projections_temporal_asof", "projections_temporal", ["project_id", "kind", "valid_from", "valid_to"]
    )
    op.create_index(
        "ix_projections_temporal_entity", "projections_temporal", ["kind", "entity_id", "valid_from"]
    )
    op.create_table(
        "activity_events",
        sa.Column("position", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False, unique=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False)),
        sa.Column("frame_id", sa.Uuid(as_uuid=False)),
        sa.Column("actor_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("performer_id", sa.Uuid(as_uuid=False)),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attributes", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_activity_report", "activity_events", ["project_id", "recorded_at", "event_type", "performer_id"]
    )
    op.create_table(
        "principals",
        sa.Column("principal_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("external_key", sa.Text(), nullable=False, unique=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("issuer", sa.Text()),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text()),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("system_roles", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.create_table(
        "server_accounts",
        sa.Column("account_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("username_key", sa.Text(), nullable=False, unique=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("blocked_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "server_sessions",
        sa.Column("token_hash", sa.Text(), primary_key=True),
        sa.Column(
            "account_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("server_accounts.account_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_server_sessions_account", "server_sessions", ["account_id"])
    op.create_index("ix_server_sessions_expiry", "server_sessions", ["expires_at"])
    op.create_table(
        "server_global_roles",
        sa.Column(
            "account_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("server_accounts.account_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", sa.Text(), primary_key=True),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "project_acl",
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("principal_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("role", sa.Text(), primary_key=True),
        sa.Column("granted_by", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "performers",
        sa.Column("performer_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column(
            "principal_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("principals.principal_id", ondelete="RESTRICT"),
            nullable=True,
            unique=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_performers_active_name", "performers", ["active", "name", "performer_id"]
    )
    op.create_table(
        "background_jobs",
        sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("lease_owner", sa.Text()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_background_jobs_state", "background_jobs", ["state", "created_at"])
    op.create_table(
        "federated_sessions",
        sa.Column("token_hash", sa.Text(), primary_key=True),
        sa.Column(
            "principal_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("principals.principal_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_federated_sessions_expiry", "federated_sessions", ["expires_at"])


def downgrade() -> None:
    for table in (
        "federated_sessions",
        "background_jobs",
        "performers",
        "project_acl",
        "server_global_roles",
        "server_sessions",
        "server_accounts",
        "principals",
        "activity_events",
        "projections_temporal",
        "projections_current",
        "transactional_outbox",
        "command_idempotency",
        "domain_events",
        "event_streams",
    ):
        op.drop_table(table)
