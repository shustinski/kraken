import hashlib
import json
from urllib.parse import urlsplit

from PIL import Image
from contour.serializers import clear_cif_parse_cache, load_polygons_cif
from live_server_context import RUN, connect

remote = connect()
manifest = json.loads((RUN / "artifact-manifest.json").read_text(encoding="utf-8"))
pid = manifest["project_id"]
assert manifest["total_bytes"] <= 256 * 1024 * 1024, "Reassess before unbounded downloading"
history_before = [str(event.event_id) for event in remote.history(pid)]
transfers = []
download = remote._http.download_url


def observed_download(url, *, token, destination):
    parsed = urlsplit(url)
    transfers.append(parsed.scheme + "://" + parsed.netloc)
    return download(url, token=token, destination=destination)


remote._http.download_url = observed_download
receipt = {"project_id": pid, "status": "running", "checked": [], "failures": [], "total_bytes": 0}
dimensions = {}
for number, item in enumerate(manifest["artifacts"], 1):
    try:
        path = remote.managed_artifact_path(pid, item["id"])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == item["sha256"], "Downloaded SHA-256 differs from registered version"
        assert path.stat().st_size == item["size_bytes"], "Downloaded size differs from registered version"
        check = {"id": item["id"], "kind": item["kind"], "x": item["x"], "y": item["y"], "sha256": digest}
        if item["kind"] == "image":
            with Image.open(path) as image:
                check["dimensions"] = image.size
                dimensions[(item["x"], item["y"])] = image.size
                image.verify()
        elif item["kind"] == "vector":
            clear_cif_parse_cache()
            _image_name, declared_dimensions, polygons = load_polygons_cif(path)
            check["polygons"] = len(polygons)
            check["holes"] = sum(bool(p.is_hole) for p in polygons)
            check["declared_dimensions"] = declared_dimensions
            if declared_dimensions and all(declared_dimensions):
                assert tuple(declared_dimensions) == dimensions[(item["x"], item["y"])], "CIF/source dimensions differ"
        receipt["checked"].append(check)
        receipt["total_bytes"] += path.stat().st_size
    except Exception as error:
        receipt["failures"].append({"id": item["id"], "error_type": type(error).__name__, "error": str(error)})
    if number % 20 == 0 or receipt["failures"]:
        (RUN / "artifact-verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
        print(f"checked {number}/{manifest['artifact_count']}, failures={len(receipt['failures'])}", flush=True)
fresh = connect()
history_after = [str(event.event_id) for event in fresh.history(pid)]
receipt["revision_before"] = manifest["project_revision"]
receipt["revision_after"] = fresh.get_project(pid).revision
receipt["project_history_unchanged"] = history_before == history_after
receipt["gateway_downloads"] = len(transfers)
receipt["gateway_origins"] = sorted(set(transfers))
receipt["quality_assessed"] = False
receipt["status"] = "passed" if not receipt["failures"] and receipt["project_history_unchanged"] else "failed"
(RUN / "artifact-verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
print(json.dumps({key: value for key, value in receipt.items() if key not in {"checked", "failures"}}), flush=True)
assert receipt["status"] == "passed", receipt["failures"][:3]
