"""Persist portable Agent progress and checkpoint references on the server."""
import sqlalchemy as sa
from alembic import op

revision = "20261029_0011"
down_revision = "20261025_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("background_jobs", sa.Column("execution_state", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("background_jobs", "execution_state")
