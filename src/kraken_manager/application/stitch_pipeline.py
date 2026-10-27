"""A stitch request belongs to an attempt of the existing layer workflow."""
from .errors import ConflictError
from .pipeline_execution import PipelineExecutionService
from .quality import QualityService


def check_stitch_owner(events, projections, manifest, *, dispatch=False):
    owner = manifest.parameters.get("_pipeline")
    if owner is None:
        return True
    state = PipelineExecutionService(events).load(manifest.layer_id)
    node = state["nodes"].get(owner["node_id"])
    if (state["project_id"] != manifest.project_id or state["run_id"] != owner["run_id"]
            or state["definition_fingerprint"] != owner["definition_fingerprint"] or node is None
            or not node["attempts"] or node["attempts"][-1]["job_id"] != manifest.job_id
            or node["status"] not in {"dispatching", "running"}
            or state["status"] not in {"running", "paused"}):
        raise ConflictError("Pipeline stitch attempt was cancelled or superseded")
    attempt = node["attempts"][-1]
    if attempt["request"]["parameters"]["_pipeline"] != owner:
        raise ConflictError("Stitch pipeline ownership metadata differs from its canonical attempt")
    for series_id, version_id in attempt["request"]["input_heads"].items():
        current = projections.get_active_artifact_version(series_id)
        if current is None or str(current.id) != version_id:
            raise ConflictError("Pipeline stitch input head changed after binding")
    if owner["quality_gate"]:
        QualityService.require_clear_inputs(events, manifest.project_id, tuple(a.asset_id for a in manifest.inputs), projections=projections)
    return not (dispatch and state["status"] == "paused")
