"""HTTP client for versioned physical registration and contact review."""
from __future__ import annotations


class MultilayerMixin:
    stitch_execution_remote = True

    def get_multilayer_model(self, project_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/multilayer")

    def stitch_vector_header(self, project_id, job_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/stitch-renders/{job_id}/vectors")

    def stitch_vector_block(self, project_id, job_id, column, row):
        return self._call("GET", f"/api/v1/projects/{project_id}/stitch-renders/{job_id}/vector-blocks/{column}/{row}")

    def export_stitch_vectors(self, *, project_id, job_id, destination, cancelled=lambda: False):
        from kraken_manager.infrastructure.vector_exchange import write_vector_exchange
        header = self.stitch_vector_header(project_id, job_id)
        return write_vector_exchange(destination, header,
            lambda column, row: self.stitch_vector_block(project_id, job_id, column, row), cancelled=cancelled)

    def mutate_multilayer_model(self, *, project_id, principal, expected_revision, operation, arguments, idempotency_key):
        return self._call(
            "POST", f"/api/v1/projects/{project_id}/multilayer/commands",
            payload={"operation": operation, "arguments": arguments},
            idempotency_key=idempotency_key, if_match=expected_revision,
        )

    def remote_stitch_block(self, project_id, job_id, column, row, *, coverage=False):
        import base64
        payload = self._call("GET", f"/api/v1/projects/{project_id}/stitch-renders/{job_id}/blocks/{column}/{row}"
                            + ("?coverage=true" if coverage else ""))
        data = base64.b64decode(payload["image_base64"], validate=True)
        if len(data) > 8 * 1024**2:
            raise ValueError("Remote mosaic block exceeds its size limit")
        return data, payload["metadata"], payload["x"], payload["y"], payload["columns"], payload["rows"]

    def import_stitch_observations(self, *, project_id, layer_id, principal, source_path, idempotency_key):
        import hashlib
        import json
        from uuid import UUID, uuid5
        from kraken_core.safe_files import open_regular_read
        from kraken_manager.domain.artifacts import ArtifactScope

        with open_regular_read(source_path) as stream:
            data = stream.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError("Stitch observations exceed the JSON size limit")
        arguments = json.loads(data.decode("utf-8-sig"))
        if (not isinstance(arguments, dict) or str(arguments.get("layer_id", layer_id)) != str(layer_id)
                or set(arguments) - {"layer_id", "reference_frame_id", "tiles", "max_error_m", "max_condition"}):
            raise ValueError("Invalid stitch observations or layer identity")
        series = self.create_artifact_series(principal=principal, project_id=project_id, layer_id=layer_id,
            scope=ArtifactScope.LAYER_ATTACHMENT, name="Stitch observations",
            series_id=str(uuid5(UUID(str(project_id)), "stitch-observations:" + idempotency_key)),
            idempotency_key=idempotency_key + ":series")
        version = self.add_managed_artifact_version(principal=principal, project_id=project_id, series_id=series.id,
            source=source_path, idempotency_key=idempotency_key + ":evidence")
        if version.sha256 != hashlib.sha256(data).hexdigest():
            raise ValueError("Uploaded observation document differs from the submitted geometry")
        return self.mutate_multilayer_model(project_id=project_id, principal=principal,
            expected_revision=self.get_multilayer_model(project_id)["model"]["revision"], operation="plan_layer_stitch",
            arguments={**arguments, "layer_id": str(layer_id), "plan_version_id": str(version.id)},
            idempotency_key=idempotency_key + ":geometry")
