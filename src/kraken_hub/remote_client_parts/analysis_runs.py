"""Remote AnalysisRun control and measurements; the server resolves all inputs."""
from urllib.parse import quote


class ServerAnalysisMixin:
    def request_server_analysis_map(self, project_id, run_id, frame_id, *, expected_revision, idempotency_key):
        return self._call("POST", f"/api/v1/projects/{project_id}/analysis-runs/{quote(str(run_id), safe='')}/frames/"
                          f"{quote(str(frame_id), safe='')}/map", idempotency_key=idempotency_key, if_match=expected_revision)

    def server_analysis_map_bytes(self, project_id, run_id, frame_id):
        path = (f"/api/v1/projects/{project_id}/analysis-runs/{quote(str(run_id), safe='')}/frames/"
                f"{quote(str(frame_id), safe='')}/map")
        content = bytearray()
        for chunk in self._http.iter_bytes(path, token=self.auth.access_token):
            content.extend(chunk)
            if len(content) > 256 * 1024 * 1024:
                raise ValueError("Rendered map exceeds its size limit")
        return bytes(content)

    def server_analysis_runs(self, project_id):
        return tuple(self._call("GET", f"/api/v1/projects/{project_id}/analysis-runs")["items"])

    def server_analysis_run(self, project_id, run_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/analysis-runs/{quote(str(run_id), safe='')}")

    def create_server_analysis(self, project_id, payload, *, idempotency_key):
        return self._call("POST", f"/api/v1/projects/{project_id}/analysis-runs", payload=payload,
                          idempotency_key=idempotency_key)

    def command_server_analysis(self, project_id, run_id, operation, *, expected_revision, idempotency_key):
        return self._call("POST", f"/api/v1/projects/{project_id}/analysis-runs/{quote(str(run_id), safe='')}/commands",
                          payload={"operation": operation}, idempotency_key=idempotency_key, if_match=expected_revision)

    def server_analysis_frames(self, project_id, run_id, metric_key):
        return tuple(self._call("GET", f"/api/v1/projects/{project_id}/analysis-runs/{quote(str(run_id), safe='')}/frames"
                                f"?metric_key={quote(str(metric_key), safe='')}")["items"])
