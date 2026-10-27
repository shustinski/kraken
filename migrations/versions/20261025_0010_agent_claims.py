"""Fence reclaimed jobs by a unique token for every lease acquisition."""

import sqlalchemy as sa
from alembic import op

revision = "20261025_0010"
down_revision = "20260925_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("background_jobs", sa.Column("lease_claim", sa.Text(), nullable=True))
    op.add_column("background_jobs", sa.Column("telemetry", sa.JSON(), nullable=True))
    # Pre-upgrade processes have no generation credential. They must reacquire.
    op.execute(sa.text("UPDATE background_jobs SET state='queued', lease_owner=NULL, lease_until=NULL WHERE state='leased'"))


def downgrade() -> None:
    op.drop_column("background_jobs", "telemetry")
    op.drop_column("background_jobs", "lease_claim")
