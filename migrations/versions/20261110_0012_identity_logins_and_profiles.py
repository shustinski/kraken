"""Add principal logins/profiles and server account auth_provider."""
import sqlalchemy as sa
from alembic import op

revision = "20261110_0012"
down_revision = "20261029_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("principals", sa.Column("login", sa.Text(), nullable=True))
    op.add_column("principals", sa.Column("login_key", sa.Text(), nullable=True))
    op.add_column("principals", sa.Column("avatar_url", sa.Text(), nullable=True))
    op.add_column("principals", sa.Column("avatar_source", sa.Text(), nullable=True))
    op.add_column("principals", sa.Column("status_text", sa.Text(), nullable=True))
    op.add_column("principals", sa.Column("profile_updated_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "ux_principals_login_key",
        "principals",
        ["login_key"],
        unique=True,
        postgresql_where=sa.text("login_key IS NOT NULL"),
    )
    op.add_column("server_accounts", sa.Column("auth_provider", sa.Text(), nullable=False, server_default="local"))
    op.execute(
        sa.text(
            """
            UPDATE principals p
               SET login = a.username, login_key = a.username_key
             FROM server_accounts a
            WHERE p.provider = 'local'
              AND p.external_key = 'local:' || a.username
              AND p.login_key IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM principals q
                   WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id)
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO administration_audit (actor_id, action, target_account_id, details, recorded_at)
            SELECT NULL, 'login_claim_conflict_backfill', p.principal_id,
                   jsonb_build_object('login', a.username), now()
              FROM server_accounts a
              JOIN principals p ON p.external_key = 'local:' || a.username
             WHERE p.login_key IS NULL
               AND EXISTS (SELECT 1 FROM principals q
                            WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id)
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ux_principals_login_key", table_name="principals")
    op.drop_column("principals", "profile_updated_at")
    op.drop_column("principals", "status_text")
    op.drop_column("principals", "avatar_source")
    op.drop_column("principals", "avatar_url")
    op.drop_column("principals", "login_key")
    op.drop_column("principals", "login")
    op.drop_column("server_accounts", "auth_provider")
