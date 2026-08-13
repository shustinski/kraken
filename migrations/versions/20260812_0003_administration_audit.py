"""administrative audit log

Revision ID: 20260812_0003
Revises: 20260811_0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260812_0003"
down_revision = "20260811_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "administration_audit",
        sa.Column("audit_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("actor_id", sa.Uuid(as_uuid=False)),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("target_account_id", sa.Uuid(as_uuid=False)),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_administration_audit_recorded_at",
        "administration_audit",
        ["recorded_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_administration_audit_recorded_at",
        table_name="administration_audit",
    )
    op.drop_table("administration_audit")
