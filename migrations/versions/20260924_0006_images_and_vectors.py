"""Keep image files and vector files in separate catalog tables.

Revision ID: 20260924_0006
Revises: 20260924_0005
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260924_0006"
down_revision = "20260924_0005"
branch_labels = None
depends_on = None


def _file_columns() -> list[sa.Column]:
    return [
        sa.Column("project_name", sa.Text(), nullable=False),
        sa.Column("layer_name", sa.Text(), nullable=False),
        sa.Column("representation_name", sa.Text(), nullable=False),
        sa.Column("x", sa.Integer()),
        sa.Column("y", sa.Integer()),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("file_uri", sa.Text()),
        sa.Column("media_type", sa.Text()),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("series_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("representation_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("frame_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.PrimaryKeyConstraint("representation_id", "frame_id"),
    ]


def upgrade() -> None:
    op.create_table("images", *_file_columns())
    op.create_index("ix_images_project_name", "images", ["project_name", "y", "x"])
    op.create_table("vectors", *_file_columns())
    op.create_index("ix_vectors_project_name", "vectors", ["project_name", "y", "x"])
    op.execute(
        """
        INSERT INTO images (
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            media_type, size_bytes, series_id, project_id, layer_id, representation_id, frame_id
        )
        SELECT
            frames.project_name, frames.layer_name, frames.representation_name,
            frames.x, frames.y, frames.filename,
            COALESCE(versions.file_uri, frames.file_uri),
            versions.media_type, versions.size_bytes,
            frames.series_id, frames.project_id, frames.layer_id,
            frames.representation_id, frames.frame_id
        FROM frames
        JOIN representations ON representations.representation_id = frames.representation_id
        LEFT JOIN LATERAL (
            SELECT media_type, size_bytes, file_uri
            FROM artifact_versions
            WHERE series_id = frames.series_id AND active
            LIMIT 1
        ) AS versions ON true
        WHERE representations.kind = 'image'
        """
    )
    op.execute(
        """
        INSERT INTO vectors (
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            media_type, size_bytes, series_id, project_id, layer_id, representation_id, frame_id
        )
        SELECT
            frames.project_name, frames.layer_name, frames.representation_name,
            frames.x, frames.y, frames.filename,
            COALESCE(versions.file_uri, frames.file_uri),
            versions.media_type, versions.size_bytes,
            frames.series_id, frames.project_id, frames.layer_id,
            frames.representation_id, frames.frame_id
        FROM frames
        JOIN representations ON representations.representation_id = frames.representation_id
        LEFT JOIN LATERAL (
            SELECT media_type, size_bytes, file_uri
            FROM artifact_versions
            WHERE series_id = frames.series_id AND active
            LIMIT 1
        ) AS versions ON true
        WHERE representations.kind = 'vector'
        """
    )
    op.drop_table("artifact_versions")
    op.drop_index("ix_frames_project_name", table_name="frames")
    op.drop_table("frames")


def downgrade() -> None:
    op.create_table(
        "frames",
        sa.Column("project_name", sa.Text(), nullable=False),
        sa.Column("layer_name", sa.Text(), nullable=False),
        sa.Column("representation_name", sa.Text(), nullable=False),
        sa.Column("x", sa.Integer()),
        sa.Column("y", sa.Integer()),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("file_uri", sa.Text()),
        sa.Column("series_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("representation_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("frame_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.PrimaryKeyConstraint("representation_id", "frame_id"),
    )
    op.create_index("ix_frames_project_name", "frames", ["project_name", "representation_name", "y", "x"])
    op.execute(
        """
        INSERT INTO frames (
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            series_id, project_id, layer_id, representation_id, frame_id
        )
        SELECT
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            series_id, project_id, layer_id, representation_id, frame_id
        FROM images
        UNION ALL
        SELECT
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            series_id, project_id, layer_id, representation_id, frame_id
        FROM vectors
        """
    )
    op.create_table(
        "artifact_versions",
        sa.Column("project_name", sa.Text(), nullable=False),
        sa.Column("layer_name", sa.Text(), nullable=False),
        sa.Column("representation_name", sa.Text(), nullable=False),
        sa.Column("x", sa.Integer()),
        sa.Column("y", sa.Integer()),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("file_uri", sa.Text()),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("series_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("version_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    op.drop_index("ix_vectors_project_name", table_name="vectors")
    op.drop_table("vectors")
    op.drop_index("ix_images_project_name", table_name="images")
    op.drop_table("images")
