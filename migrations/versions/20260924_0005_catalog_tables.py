"""Readable catalog tables beside the JSON projection.

Revision ID: 20260924_0005
Revises: 20260922_0004
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260924_0005"
down_revision = "20260922_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("orientation", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("source_directory", sa.Text()),
        sa.Column("derived_directory", sa.Text()),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    op.create_index("ix_projects_name", "projects", ["name"])
    op.create_table(
        "layers",
        sa.Column("project_name", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("layer_type", sa.Text(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("image_directory", sa.Text()),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    op.create_index("ix_layers_project_name", "layers", ["project_name", "sort_order"])
    op.create_table(
        "representations",
        sa.Column("project_name", sa.Text(), nullable=False),
        sa.Column("layer_name", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("source", sa.Text()),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("representation_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
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
    op.execute(
        """
        INSERT INTO projects (
            name, width, height, orientation, state, source_directory, derived_directory,
            revision, created_at, updated_at, project_id
        )
        SELECT
            payload->>'name',
            (payload->>'width')::integer,
            (payload->>'height')::integer,
            payload->>'orientation',
            payload->>'state',
            NULL,
            NULL,
            revision,
            (payload->>'created_at')::timestamptz,
            updated_at,
            entity_id
        FROM projections_current
        WHERE kind = 'project'
        """
    )
    op.execute(
        """
        INSERT INTO layers (
            project_name, name, layer_type, sort_order, state, image_directory,
            revision, created_at, project_id, layer_id
        )
        SELECT
            COALESCE(projects.name, ''),
            layer.payload->>'name',
            layer.payload->>'type',
            layer.sort_order,
            layer.payload->>'state',
            NULL,
            layer.revision,
            (layer.payload->>'created_at')::timestamptz,
            layer.project_id,
            layer.entity_id
        FROM projections_current AS layer
        LEFT JOIN projects ON projects.project_id = layer.project_id
        WHERE layer.kind = 'layer'
        """
    )
    op.execute(
        """
        INSERT INTO representations (
            project_name, layer_name, name, kind, purpose, state, source, active,
            revision, created_at, project_id, layer_id, representation_id
        )
        SELECT
            COALESCE(layers.project_name, ''),
            COALESCE(layers.name, ''),
            item.payload->>'name',
            item.payload->>'kind',
            item.payload->>'purpose',
            item.payload->>'state',
            NULLIF(item.payload->>'source', ''),
            item.active,
            item.revision,
            (item.payload->>'created_at')::timestamptz,
            item.project_id,
            item.layer_id,
            item.entity_id
        FROM projections_current AS item
        LEFT JOIN layers ON layers.layer_id = item.layer_id
        WHERE item.kind = 'representation'
        """
    )
    op.execute(
        """
        UPDATE layers
        SET image_directory = source.source
        FROM representations AS source
        WHERE source.layer_id = layers.layer_id
          AND source.purpose = 'source'
          AND source.source IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE projects
        SET source_directory = regexp_replace(layers.image_directory, '[\\\\/]img[\\\\/][^\\\\/]+$', '')
        FROM layers
        WHERE layers.project_id = projects.project_id
          AND layers.image_directory IS NOT NULL
          AND projects.source_directory IS NULL
        """
    )
    op.execute(
        """
        INSERT INTO frames (
            project_name, layer_name, representation_name, x, y, filename, file_uri,
            series_id, project_id, layer_id, representation_id, frame_id
        )
        SELECT
            representations.project_name,
            representations.layer_name,
            representations.name,
            (number.frame_index % projects.width) + 1,
            (number.frame_index / projects.width) + 1,
            series.payload->>'name',
            NULL,
            series.entity_id,
            series.project_id,
            series.layer_id,
            representations.representation_id,
            (series.payload->>'frame_id')::uuid
        FROM projections_current AS series
        JOIN representations
          ON representations.representation_id = (series.payload->>'representation_id')::uuid
        JOIN projects ON projects.project_id = series.project_id
        JOIN LATERAL (
            SELECT substring(
                regexp_replace(series.payload->>'name', '\\.[^.]*$', ''),
                '([0-9]+)$'
            )::integer AS frame_index
        ) AS number ON number.frame_index IS NOT NULL
        WHERE series.kind = 'artifact_series'
          AND series.payload->>'scope' = 'frame_representation'
        """
    )
    op.execute(
        """
        INSERT INTO artifact_versions (
            project_name, layer_name, representation_name, x, y, filename, media_type,
            size_bytes, file_uri, active, created_at, project_id, series_id, version_id
        )
        SELECT
            COALESCE(frames.project_name, ''),
            COALESCE(frames.layer_name, ''),
            COALESCE(frames.representation_name, ''),
            frames.x,
            frames.y,
            item.payload->>'filename',
            item.payload->>'media_type',
            (item.payload->>'size_bytes')::bigint,
            item.payload->'external'->>'uri',
            item.active,
            (item.payload->>'created_at')::timestamptz,
            COALESCE(frames.project_id, item.project_id),
            item.parent_id,
            item.entity_id
        FROM projections_current AS item
        LEFT JOIN frames ON frames.series_id = item.parent_id
        WHERE item.kind = 'artifact_version'
          AND item.parent_id IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE frames
        SET file_uri = versions.file_uri
        FROM artifact_versions AS versions
        WHERE versions.series_id = frames.series_id
          AND versions.active
          AND versions.file_uri IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE VIEW jobs AS
        SELECT
            projects.name AS project_name,
            layers.name AS layer_name,
            plugin.payload->>'capability' AS capability,
            COALESCE(agent.state, plugin.payload->>'state') AS state,
            plugin.payload->>'error' AS error,
            plugin.updated_at
        FROM projections_current AS plugin
        JOIN projects ON projects.project_id = plugin.project_id
        LEFT JOIN layers ON layers.layer_id = plugin.layer_id
        LEFT JOIN background_jobs AS agent ON agent.job_id = plugin.entity_id
        WHERE plugin.kind = 'plugin_job'
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS jobs")
    op.drop_table("artifact_versions")
    op.drop_index("ix_frames_project_name", table_name="frames")
    op.drop_table("frames")
    op.drop_table("representations")
    op.drop_index("ix_layers_project_name", table_name="layers")
    op.drop_table("layers")
    op.drop_index("ix_projects_name", table_name="projects")
    op.drop_table("projects")
