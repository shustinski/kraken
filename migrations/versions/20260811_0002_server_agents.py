"""server agent identities and idempotent publications

Revision ID: 20260811_0002
Revises: 20260717_0001
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260811_0002"
down_revision = "20260717_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "server_agent_tokens",
        sa.Column("token_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_server_agent_tokens_active",
        "server_agent_tokens",
        ["revoked_at", "token_id"],
    )
    op.create_table(
        "agent_job_publications",
        sa.Column("publication_id", sa.Text(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("background_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_agent_job_publications_job",
        "agent_job_publications",
        ["job_id", "created_at"],
    )
    op.add_column(
        "background_jobs",
        sa.Column("lease_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("background_jobs", sa.Column("last_error", sa.Text()))
    op.create_table(
        "agent_job_outputs",
        sa.Column(
            "job_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("background_jobs.job_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("output_id", sa.Text(), primary_key=True),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_background_jobs_lease",
        "background_jobs",
        ["state", "lease_until", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_background_jobs_lease", table_name="background_jobs")
    op.drop_column("background_jobs", "last_error")
    op.drop_column("background_jobs", "lease_attempts")
    op.drop_table("agent_job_outputs")
    op.drop_index("ix_agent_job_publications_job", table_name="agent_job_publications")
    op.drop_table("agent_job_publications")
    op.drop_index("ix_server_agent_tokens_active", table_name="server_agent_tokens")
    op.drop_table("server_agent_tokens")
