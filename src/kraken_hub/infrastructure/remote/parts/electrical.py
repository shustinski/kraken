"""Remote conductor review, explicit binding and partial netlist export."""
from __future__ import annotations


class ElectricalMixin:
    def get_electrical_model(self, project_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/electrical")

    def export_electrical_model(self, project_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/electrical/export")

    def mutate_electrical_model(self, *, project_id, principal, expected_revision, expected_physical_revision,
                                operation, arguments, idempotency_key):
        return self._call("POST", f"/api/v1/projects/{project_id}/electrical/commands",
            payload={"operation": operation, "arguments": arguments, "expected_physical_revision": expected_physical_revision},
            idempotency_key=idempotency_key, if_match=expected_revision)
