"""Optional deletion reason and the administrator auto-confirm setting."""

import sqlalchemy as sa
from alembic import op

from kraken_admin.deletion_confirmation import settings_table

revision = "20260925_0009"
down_revision = "20260925_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("project_deletion_requests")}
    if "reason" not in columns:
        op.add_column(
            "project_deletion_requests",
            sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        )
    if "admin_runtime_settings" not in inspector.get_table_names():
        settings_table().create(bind)


def downgrade() -> None:
    op.drop_table("admin_runtime_settings")
    op.drop_column("project_deletion_requests", "reason")
