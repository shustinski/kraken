"""Indexes for viewport parent lookup and project ACL by principal.

Revision ID: 20260924_0007
Revises: 20260924_0006
"""

from __future__ import annotations

from alembic import op

revision = "20260924_0007"
down_revision = "20260924_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_projections_current_parent_active",
        "projections_current",
        ["kind", "parent_id"],
        postgresql_where="active",
    )
    op.create_index(
        "ix_projections_current_kind_layer",
        "projections_current",
        ["kind", "layer_id"],
    )
    op.create_index(
        "ix_project_acl_principal_active",
        "project_acl",
        ["principal_id"],
        postgresql_where="revoked_at IS NULL",
    )


def downgrade() -> None:
    op.drop_index("ix_project_acl_principal_active", table_name="project_acl")
    op.drop_index("ix_projections_current_kind_layer", table_name="projections_current")
    op.drop_index("ix_projections_current_parent_active", table_name="projections_current")
