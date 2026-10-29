"""Nonempty managed project fixture and actual Qt matrix deployment checks."""
from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path
import time

from kraken_manager.application.imports import ImportMappingMode, ImportPlanner, ImportSource
from kraken_manager.domain.project import RepresentationKind


def probe_records(run: Path, reference_manifest: str | Path | None = None) -> list[dict]:
    if reference_manifest is not None:
        manifest = Path(reference_manifest).resolve(strict=True)
        with manifest.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in islice(stream, 16)]
        if not records:
            raise ValueError("Reference deployment manifest is empty")
        for index, record in enumerate(records):
            record["x"], record["y"] = index % 4 + 1, index // 4 + 1
            for role in ("image", "vector"):
                source = Path(record[role]).resolve(strict=True)
                if hashlib.sha256(source.read_bytes()).hexdigest() != record[role + "_sha256"]:
                    raise ValueError("Reference deployment source SHA mismatch")
                record[role] = str(source)
        return records
    from PIL import Image

    source = run / "diagnostic-source"
    source.mkdir()
    records = []
    for index, (x, y, color) in enumerate(((1, 1, "red"), (2, 2, "blue"))):
        image, vector = source / f"{x}_{y}.png", source / f"{x}_{y}.cif"
        Image.new("RGB", (64, 64), color).save(image)
        vector.write_text("DS 1 1 1;\nL M1;\nP 0 0 64 0 64 64 0 64;\nDF;\nE;\n", encoding="ascii")
        records.append({"key": str(index), "x": x, "y": y, "diagnostic_color": color,
                        "image": str(image), "vector": str(vector),
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "vector_sha256": hashlib.sha256(vector.read_bytes()).hexdigest()})
    return records


def prepare_project(service, session, project, records: list[dict]) -> dict:
    layer = service.list_layers(project.id)[0]
    images = service.create_representation(
        principal=session.principal, project=service.get_project(project.id), layer=layer,
        name="Deployment microscopy", kind=RepresentationKind.IMAGE, source="managed-import",
        active=True, idempotency_key="deployment-images",
    )
    vectors = service.create_representation(
        principal=session.principal, project=service.get_project(project.id), layer=service.list_layers(project.id)[0],
        name="Deployment CIF", kind=RepresentationKind.VECTOR, source="managed-import",
        source_image_representation_id=images.id, active=True, idempotency_key="deployment-vectors",
    )
    expected = []
    for index, record in enumerate(records):
        x, y = record.get("x", index % project.width + 1), record.get("y", index // project.width + 1)
        item = {"x": x, "y": y, "image_sha256": record["image_sha256"],
                "vector_sha256": record["vector_sha256"], "diagnostic_color": record.get("diagnostic_color")}
        for role, representation in (("image", images), ("vector", vectors)):
            source = Path(record[role])
            plan = ImportPlanner().plan(width=project.width, height=project.height,
                sources=(ImportSource(str(source), source.name, source.stat().st_size, record[role + "_sha256"]),),
                mode=ImportMappingMode.EXPLICIT, explicit={str(source): (x, y)})
            imported = service.commit_managed_import(
                principal=session.principal, project=service.get_project(project.id), layer=layer,
                representation=representation, plan=plan, idempotency_key=f"deployment:{index}:{role}",
            )
            item[role + "_version_id"] = str(imported.versions[0].id)
        expected.append(item)
    receipt = {"project_id": str(project.id), "layer_id": str(layer.id),
               "image_representation_id": str(images.id), "vector_representation_id": str(vectors.id),
               "expected": expected, "coordinate_semantics": "diagnostic/catalog positions; physical adjacency unknown"}
    verify_project(service, receipt)  # Prime the spatial cache before changing its head.
    advance_head(service, session, receipt, records[0]["vector"], "deployment-new-head")
    verify_project(service, receipt)
    return receipt


def advance_head(service, session, receipt: dict, source, idempotency_key: str) -> None:
    first = receipt["expected"][0]
    parent = service.artifact_version(receipt["project_id"], first["vector_version_id"])
    # Identical bytes are intentional: the immutable version/revision must change
    # even when the content-addressed blob is reused. Coordinates live in its parent.
    version = service.add_managed_artifact_version(
        principal=session.principal, project_id=receipt["project_id"], series_id=parent.series_id,
        source=source, parent_version_id=parent.id, idempotency_key=idempotency_key,
    )
    if version.id == parent.id or version.parameters.get("x") is not None:
        raise RuntimeError("Deployment parent-coordinate fixture was not created")
    receipt.setdefault("previous_vector_version_ids", []).append(first["vector_version_id"])
    first["vector_version_id"] = str(version.id)


def verify_project(service, receipt: dict) -> None:
    for item in receipt["expected"]:
        view = service.matrix_viewport(receipt["project_id"], layer_id=receipt["layer_id"],
            representation_ids=(receipt["image_representation_id"], receipt["vector_representation_id"]),
            x1=item["x"], y1=item["y"], x2=item["x"], y2=item["y"])
        if len(view["cells"]) != 1:
            raise RuntimeError("Managed deployment viewport lost a frame")
        cell = view["cells"][0]
        if (cell["asset_sha256"] != item["image_sha256"] or cell["sha256"] != item["vector_sha256"]
                or cell["artifact_version_id"] != item["vector_version_id"]):
            raise RuntimeError("Managed deployment viewport returned a stale or different artifact")


def verify_workspace(app, controller, receipt: dict, screenshot: Path) -> None:
    workspace = controller._workspace
    item = workspace.layer_model.layer_by_id(receipt["layer_id"])
    if item is None:
        raise RuntimeError("Deployment layer is absent from the actual workspace")
    controller._select_layer(workspace, receipt["project_id"], item)
    matrix = workspace.matrix_view
    errors = []
    matrix.errorOccurred.connect(errors.append)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        if errors:
            raise RuntimeError(f"Managed matrix background/thumbnail failure: {errors}")
        ready = True
        for expected in receipt["expected"]:
            cell = matrix.cell_data(expected["x"], expected["y"])
            payload = cell.payload or {}
            if (payload.get("artifact_version_id") != expected["vector_version_id"]
                    or payload.get("asset_sha256") != expected["image_sha256"]
                    or cell.thumbnail is None or cell.thumbnail.isNull()):
                ready = False
                break
        if ready and not matrix._loading:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("Actual workspace failed to show managed heads and raster thumbnails")
    for expected in receipt["expected"]:
        if expected["diagnostic_color"]:
            from PyQt6.QtGui import QColor

            image = matrix.cell_data(expected["x"], expected["y"]).thumbnail.toImage()
            if image.pixelColor(image.width() // 2, image.height() // 2) != QColor(expected["diagnostic_color"]):
                raise RuntimeError("Managed thumbnail pixels differ from the diagnostic source")
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    legend = workspace.matrix_legend
    legend.layout().activate()
    sections = [w for w in legend.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
                if w.isVisible()]
    if len(sections) != 4 or any(
        left.geometry().intersects(right.geometry())
        for index, left in enumerate(sections) for right in sections[index+1:]
    ):
        raise RuntimeError("Matrix legend contains overlapping or stale visible sections")
    if not controller.shell.grab().save(str(screenshot)):
        raise RuntimeError("Could not save the nonempty workspace screenshot")
