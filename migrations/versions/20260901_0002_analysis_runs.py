"""partitioned Karakal analysis projections

Revision ID: 20260901_0002
Revises: 20260717_0001
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260901_0002"
down_revision = "20260717_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("total_frames", sa.BigInteger(), nullable=False),
        sa.Column("completed_frames", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("failed_frames", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_analysis_runs_project_created", "analysis_runs", ["project_id", "created_at"])
    op.create_index("ix_analysis_runs_fingerprint", "analysis_runs", ["fingerprint"])
    op.create_table(
        "analysis_sources",
        sa.Column("run_id", sa.Text(), sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("binding_key", sa.Text(), primary_key=True),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_version", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
    )
    op.create_table(
        "analysis_partitions",
        sa.Column("partition_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("partition_index", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=False, unique=True),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bundle_sha256", sa.String(length=64)),
        sa.Column("result", sa.JSON()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "partition_index", name="uq_analysis_partition_run_index"),
    )
    op.create_index("ix_analysis_partitions_run_state", "analysis_partitions", ["run_id", "state"])
    op.create_table(
        "analysis_frame_results",
        sa.Column("run_id", sa.Text(), sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("frame_id", sa.Text(), primary_key=True),
        sa.Column(
            "partition_id",
            sa.Text(),
            sa.ForeignKey("analysis_partitions.partition_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("x", sa.Integer(), nullable=False),
        sa.Column("y", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
    )
    op.create_table(
        "analysis_metric_values",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("frame_id", sa.Text(), primary_key=True),
        sa.Column("metric_key", sa.Text(), primary_key=True),
        sa.Column("raw_value", sa.Float(), nullable=False),
        sa.Column("goodness", sa.Float(), nullable=False),
        sa.Column("percentile", sa.Float()),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("higher_is_better", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "frame_id"],
            ["analysis_frame_results.run_id", "analysis_frame_results.frame_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_analysis_metric_run_key_goodness",
        "analysis_metric_values",
        ["run_id", "metric_key", "goodness"],
    )
    op.create_table(
        "analysis_metric_scales",
        sa.Column("run_id", sa.Text(), sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("metric_key", sa.Text(), primary_key=True),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("p05", sa.Float()),
        sa.Column("p50", sa.Float()),
        sa.Column("p95", sa.Float()),
        sa.Column("clipped_low", sa.BigInteger(), nullable=False),
        sa.Column("clipped_high", sa.BigInteger(), nullable=False),
    )
    op.create_table(
        "analysis_artifacts",
        sa.Column("artifact_key", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.Text(), sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("recipe_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("blob_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "analysis_artifacts",
        "analysis_metric_scales",
        "analysis_metric_values",
        "analysis_frame_results",
        "analysis_partitions",
        "analysis_sources",
        "analysis_runs",
    ):
        op.drop_table(table)
