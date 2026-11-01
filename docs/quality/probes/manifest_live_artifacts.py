import json
from live_server_context import RUN, connect

remote = connect()
projects = remote.list_projects(include_archived=True)
assert len(projects) == 1, "Select a project explicitly if the catalog changes"
project = projects[0]
receipt = {"project_id": str(project.id), "project_revision": project.revision, "representations": [], "artifacts": []}
for layer in remote.list_layers(project.id):
    for representation in remote.list_representations(project.id, layer.id):
        cells = remote.frame_cells(project.id, layer.id, representation.id)
        expected = {(x, y) for y in range(1, project.height + 1) for x in range(1, project.width + 1)}
        actual = {(cell.x, cell.y) for cell in cells if cell.sha256}
        receipt["representations"].append({"id": str(representation.id), "kind": representation.kind.value,
                                           "populated": len(actual), "complete_grid": actual == expected})
        for cell in cells:
            if not cell.sha256:
                continue
            version = remote.artifact_version(project.id, cell.artifact_version_id)
            assert version is not None and version.blob is not None
            assert version.sha256 == cell.sha256
            receipt["artifacts"].append({"id": str(version.id), "representation_id": str(representation.id),
                "kind": representation.kind.value, "x": cell.x, "y": cell.y,
                "sha256": version.sha256, "size_bytes": version.blob.size_bytes, "media_type": version.media_type})
receipt["artifact_count"] = len(receipt["artifacts"])
receipt["total_bytes"] = sum(item["size_bytes"] for item in receipt["artifacts"])
receipt["largest_bytes"] = max((item["size_bytes"] for item in receipt["artifacts"]), default=0)
(RUN / "artifact-manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
print(json.dumps({key: value for key, value in receipt.items() if key != "artifacts"}))
