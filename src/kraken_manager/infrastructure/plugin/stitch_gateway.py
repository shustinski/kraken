"""Stitch V2 staging uses the same authenticated Agent control transport."""
from kraken_agent.jobs import StagingWorkspace
from kraken_core.plugin_protocol import PluginResultPublicationV2
from kraken_core.safe_files import open_regular_read
from .analysis_gateway import AgentAnalysisGateway


class AgentStitchGateway(AgentAnalysisGateway):
    def submit(self, manifest, source_paths):
        workspace = StagingWorkspace(self.staging_root, manifest.job_id)
        workspace.create()
        for asset in manifest.inputs:
            destination = workspace.resolve(asset.relative_path)
            if destination.is_file() and workspace.digest(asset.relative_path) == asset.sha256:
                continue
            workspace.stage_file(source_paths[asset.asset_id], asset.relative_path, expected_sha256=asset.sha256)
        workspace.write_manifest(manifest)
        return self._job(self._request("POST", "/api/v1/jobs", manifest.to_dict()))

    def result(self, manifest):
        workspace = StagingWorkspace(self.staging_root, manifest.job_id)
        with open_regular_read(workspace.resolve("result.json"), root=workspace.path) as stream:
            parsed = PluginResultPublicationV2.from_json(stream.read(16 * 1024**2 + 1).decode())
        if parsed.job_id != manifest.job_id:
            raise ValueError("Mosaic publication belongs to another Agent job")
        workspace.verify_result(parsed, manifest)
        return parsed, None if not parsed.outputs else workspace.resolve(parsed.outputs[0].relative_path)
