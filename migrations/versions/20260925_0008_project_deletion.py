"""Durable admin-approved project deletion and database write barriers."""

from alembic import op
import sqlalchemy as sa

from kraken_server.project_deletion import request_table

revision = "20260925_0008"
down_revision = "20260924_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table = request_table()
    table.create(op.get_bind())
    op.create_table("project_transfer_blobs",
                    sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
                    sa.Column("sha256", sa.Text(), primary_key=True))
    op.create_index("uq_project_deletion_open", table.name, ["project_id"], unique=True,
                    postgresql_where=sa.text("state IN ('pending', 'deleting', 'failed')"))
    op.execute("""
        CREATE FUNCTION kraken_project_write_guard() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM pg_advisory_xact_lock_shared(hashtextextended('kraken:storage', 0));
          PERFORM pg_advisory_xact_lock_shared(hashtextextended('project:' || NEW.project_id::text, 0));
          IF EXISTS (SELECT 1 FROM project_deletion_requests
                     WHERE project_id::text = NEW.project_id::text AND state IN ('deleting','failed','deleted')) THEN
            RAISE EXCEPTION 'Project is being deleted' USING ERRCODE = '55000';
          END IF;
          RETURN NEW;
        END $$;
    """)
    inspector = sa.inspect(op.get_bind())
    for name in inspector.get_table_names():
        if name == table.name:
            continue
        if any(column["name"] == "project_id" for column in inspector.get_columns(name)):
            op.execute(sa.text(f'CREATE TRIGGER kraken_deletion_guard BEFORE INSERT OR UPDATE ON "{name}" '
                               'FOR EACH ROW EXECUTE FUNCTION kraken_project_write_guard()'))


def downgrade() -> None:
    op.execute("DROP FUNCTION kraken_project_write_guard() CASCADE")
    op.drop_table("project_transfer_blobs")
    op.drop_table("project_deletion_requests")
