"""Opt-in single-project functional lifecycle, with bounded CPU training.

Representative fixture only: this never certifies real-device geometry or quality.
Every phase records durable IDs; a failed run leaves its partial receipt and logs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4
import zipfile

import pytest


@pytest.mark.skipif(os.environ.get("KRAKEN_RUN_END_TO_END_FUNCTIONAL") != "1", reason="Explicit bounded single-project E2E")
def test_single_project_twenty_stage_functional_lifecycle(tmp_path):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1" if os.name == "nt" else "",
        QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NEURALIMAGE_TORCH_COMPILE="0", KRAKEN_PROJECT_ACCESS_MODE="acl")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (tmp_path / "gate.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
            env=environment, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], timeout=10,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise
    assert code == 0, (tmp_path / "gate.log").read_text(encoding="utf-8", errors="replace")[-14000:]
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "passed" and set(receipt["stages"]) == {str(i) for i in range(1, 21)}


def run_gate(root):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from PIL import Image, ImageDraw
    from contour.serializers import load_polygons_cif, save_polygons_cif
    from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
    from kraken_hub.agent_runtime import LocalAgentRuntime
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
    from kraken_manager.application.errors import AuthorizationError
    from kraken_manager.application.use_cases import AddArtifactVersionHandler
    from kraken_manager.domain.artifacts import ArtifactScope, deterministic_frame_series_id
    from kraken_manager.domain.identity import Performer, Principal, ProjectRole
    from kraken_manager.domain.multilayer import MultilayerModel
    from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation, PipelineModelInput
    from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation
    from kraken_manager.domain.project import GridOrientation, LayerType, ProjectState, RepresentationKind, RepresentationPurpose
    from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    from kraken_manager.infrastructure.mosaic import MosaicReader
    from kraken_manager.infrastructure.plugin import AgentPluginGateway
    from kraken_manager.infrastructure.reports.work_plans import write_csv
    from tests.test_work_orders import _service
    from tests.test_quality_analysis import real_run

    started = time.monotonic()
    receipt = {"status": "running", "fixture": "representative 3x128x128, controlled synthetic stack",
        "training": "one CPU epoch, 2 train/1 validation, 90-second phase deadline",
        "quality_assessed": False, "real_device_electrical_acceptance": False, "stages": {}}

    def record(number, **facts):
        receipt["stages"][str(number)] = {"seconds": round(time.monotonic()-started, 3), **facts}
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print("stage", number, json.dumps(facts), flush=True)

    local = _service(root)
    actor = local.create_initial_account("owner", "Functional owner", "").principal
    project = local.create_project(principal=actor, name="Unified representative functional lifecycle", width=3, height=1,
        orientation=GridOrientation.Y_DOWN, idempotency_key="project")
    receipt["project_id"] = str(project.id)
    record(1, project_id=str(project.id))
    layers = []
    for i, (name, kind) in enumerate((("M1", LayerType.METAL), ("V1", LayerType.CONTACT), ("M2", LayerType.METAL))):
        layers.append(local.create_layer(principal=actor, project=local.get_project(project.id), name=name,
            layer_type=kind, order=i, idempotency_key=name))
    layer = layers[0]
    record(2, layer_ids=[str(v.id) for v in layers])
    performer = local.performers.create(Performer.create(name="Functional corrector", color="#336699", principal_id=actor.id))
    reviewer = Principal.local(subject="unified-reviewer", display_name="Independent functional reviewer")
    local.identities.save(reviewer)
    local.assign_project_role(principal=actor, project=local.get_project(project.id), target_principal_id=reviewer.id,
        role=ProjectRole.MAINTAINER, expected_revision=0, idempotency_key="reviewer")
    plan = local.create_work_plan(project_id=project.id, title="Representative full lifecycle", principal=actor, idempotency_key="work-plan")
    plan = local.mutate_work_plan(project_id=project.id, plan_id=plan.id, principal=actor, expected_revision=plan.revision,
        operation="add_item", arguments={"item": {"title": "Execute and independently review lifecycle",
            "scope": {"layer_ids": [str(v.id) for v in layers]}, "assignment": {"performer_id": performer.id},
            "criteria": [{"description": "Versioned computed result independently reviewed"}]}}, idempotency_key="work-item")

    def planning(operation, arguments, principal=actor):
        nonlocal plan
        plan = local.mutate_work_plan(project_id=project.id, plan_id=plan.id, principal=principal,
            expected_revision=plan.revision, operation=operation, arguments=arguments, idempotency_key=str(uuid4()))

    planning("transition", {"state": "active"})
    planning("move_item", {"item_id": plan.items[0].id, "state": "running"})
    record(3, plan_id=plan.id, revision=plan.revision)

    def representation(name, kind=RepresentationKind.IMAGE, purpose=None, source=None):
        options = {} if purpose is None else {"purpose": purpose}
        if source is not None:
            options["source_image_representation_id"] = source.id
        return local.create_representation(principal=actor, project=local.get_project(project.id),
            layer=local.list_layers(project.id)[0], name=name, kind=kind, active=False, idempotency_key=name, **options)

    images = representation("Images")
    vectors = representation("Contour", RepresentationKind.VECTOR, source=images)
    masks = representation("Predicted binary", purpose=RepresentationPurpose.BINARY)
    source = root / "source"
    source.mkdir()
    for x in range(1, 4):
        image = Image.new("RGB", (128, 128), (20, 20, 20))
        ImageDraw.Draw(image).rectangle((20+x, 30, 100, 90), fill=(220, 220, 220))
        image.save(source / f"{x}_1.png")
    original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    imported = local.commit_managed_import(principal=actor, project=local.get_project(project.id), layer=local.list_layers(project.id)[0],
        representation=images, plan=local.plan_import_directory(project=project, directory=source), idempotency_key="images")
    image_versions = {local._projection(project.id).get_artifact_series(v.series_id).frame_id: v for v in imported.versions}
    record(4, image_versions=[str(v.id) for v in imported.versions])
    coordinates = {str(project.frame_id_at(x, 1)): (x, 1) for x in range(1, 4)}
    plugins = {name: load_plugin_metadata_from_directory(repository / "plugins" / name) for name in ("contour", "neuralimage")}
    runtime = LocalAgentRuntime(root / "agent", tuple(PluginInventoryItem(v, installed=False) for v in plugins.values()), builtin_stitching=True)
    try:
        capabilities = runtime.ensure_started()
        gateway = AgentPluginGateway(base_url=runtime.base_url, token=runtime.token, staging_root=runtime.data_dir / "staging",
            capabilities=capabilities, source_for_version=lambda v: local.artifact_source_path(project.id, v),
            coordinate_for_frame=lambda f: coordinates[str(f)], media_type_for_version=lambda v: local.get_artifact_version(project.id, v).media_type)

        definition_version = 0

        def execute(operations, key, limit=80):
            nonlocal definition_version
            definition_version += 1
            common = dict(project_id=project.id, layer_id=layer.id, principal=actor)
            revision = FilesystemEventStore(local.catalog_root, project.id).current_revision(f"layer-pipeline:{layer.id}")
            definition = LayerPipelineDefinition(str(project.id), str(layer.id), tuple(operations), version=definition_version)
            state = local.define_layer_pipeline(**common, definition=definition.to_dict(), expected_revision=revision, idempotency_key=key)
            local.command_layer_pipeline(**common, operation="start", expected_revision=state["revision"], idempotency_key=key+"-start")
            deadline = time.monotonic()+limit
            while time.monotonic() < deadline:
                state = local.tick_layer_pipeline(**common, gateway=gateway)
                (root / (key+"-state.json")).write_text(json.dumps(state, indent=2), encoding="utf-8")
                for job in local.synchronize_plugin_jobs(principal=actor, gateway=gateway):
                    if job.state.value == "importing":
                        result = local.import_agent_result(principal=actor, result_payload=gateway.get_result(job.id), staging_root=runtime.data_dir / "staging")
                        assert not result.requires_partial_confirmation and not result.awaiting_authorization
                        gateway.complete_import(job.id)
                if state["status"] in {"completed", "failed"}:
                    break
                time.sleep(.05)
            if state["status"] != "completed":
                local.command_layer_pipeline(**common, operation="cancel", expected_revision=state["revision"], idempotency_key=key+"-deadline-cancel")
                raise AssertionError(json.dumps(state))
            return state

        # Freeze selected immutable input IDs before any algorithm runs.
        record(5, selected_versions=[str(v.id) for v in imported.versions], preparation="immutable managed input selection; dataset publication follows Contour")
        vectorize = PipelineOperation(str(uuid4()), "Actual vectorization", "frames.vectorize.v1", plugins["contour"].version,
            str(images.id), str(vectors.id), "image", "vector", tuple(coordinates.values()), parameters={"execution_mode": "headless"})
        state = execute([vectorize], "contour")
        vector_ids = state["nodes"][vectorize.id]["attempts"][-1]["output_version_ids"]
        assert len(vector_ids) == 3
        record(6, vector_versions=vector_ids)
        dataset = root / "dataset"
        dataset_inputs = []
        for x in range(1, 4):
            split = dataset if x < 3 else dataset / "validation"
            for name in ("images", "cif"):
                (split / name).mkdir(parents=True, exist_ok=True)
            frame = project.frame_id_at(x, 1)
            vector = local.active_artifact_version(project.id, deterministic_frame_series_id(vectors.id, frame))
            _, size, polygons = load_polygons_cif(local.artifact_source_path(project.id, vector.id))
            assert size == (128, 128) and polygons
            image_path = split / "images" / f"{x}.png"
            image_path.write_bytes(local.artifact_source_path(project.id, image_versions[frame].id).read_bytes())
            save_polygons_cif(split / "cif" / f"{x}.cif", str(image_path), polygons, size)
            dataset_inputs.extend((str(vector.id), str(image_versions[frame].id)))
        archive_path = root / "dataset.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(dataset).as_posix())
        series = local.create_artifact_series(principal=actor, project_id=project.id, layer_id=layer.id,
            scope=ArtifactScope.LAYER_ATTACHMENT, name="Dataset from actual Contour outputs", idempotency_key="dataset-series")
        dataset_version = AddArtifactVersionHandler(local._uow(str(project.id)), local.profiles, local.clock)(
            AddArtifactVersionCommand(context=CommandContext(actor=actor, idempotency_key="dataset-version"),
                project_id=project.id, series_id=series.id, filename="dataset.zip", media_type="application/zip",
                expected_series_revision=local.artifact_stream_revision(project.id, series.id),
                parameters={"artifact_role": "dataset"}, input_version_ids=tuple(dataset_inputs)), (archive_path.read_bytes(),))
        settings = dict(model="S 660k", sample_size=[128, 128], train_patch_size=[128, 128], step=128,
            batch_size=2, train_batch_size=2, vertical_rotation=False, horizontal_rotation=False,
            use_validation=True, validation_source="external", dataloader_num_workers=0, early_stopping_enabled=True)
        train = PipelineOperation(str(uuid4()), "One CPU epoch", "dataset.model.train.v1", plugins["neuralimage"].version,
            str(representation("Dataset port").id), str(representation("Model port").id), "dataset", "model", ((1, 1),),
            parameters={"dataset_version_id": dataset_version.id, "training": {"epochs": 1, "seed": 20261027,
                "validation_relative_root": "validation", "settings": settings}})
        state = execute([train], "training", limit=90)
        model_id, = state["nodes"][train.id]["attempts"][-1]["output_version_ids"]
        model = local.get_artifact_version(project.id, model_id)
        measured = model.parameters["managed_execution"]
        assert measured["epoch_upper_bound"] == 1 and not measured["cuda_available"]
        assert measured["train_images"] == 2 and measured["validation_images"] == 1
        assert str(dataset_version.id) in model.input_version_ids
        record(7, dataset_version=str(dataset_version.id), model_version=model_id, model_sha256=model.sha256, execution=dict(measured))
        inference = PipelineOperation(str(uuid4()), "Infer published CPU model", "frames.binary-segment.v2", plugins["neuralimage"].version,
            str(images.id), str(masks.id), "image", "binary_mask", tuple(coordinates.values()),
            model_input=PipelineModelInput(version_id=model_id), parameters={"execution_mode": "headless", "patch_size": 128})
        state = execute([inference], "inference")
        mask_ids = state["nodes"][inference.id]["attempts"][-1]["output_version_ids"]
        assert len(mask_ids) == 3
        record(8, mask_versions=mask_ids, model_version=model_id)

        def attachment(name, path, target_layer=layer):
            s = local.create_artifact_series(principal=actor, project_id=project.id, layer_id=target_layer.id,
                scope=ArtifactScope.LAYER_ATTACHMENT, name=name, idempotency_key=name+"-series")
            return local.add_managed_artifact_version(principal=actor, project_id=project.id, series_id=s.id, source=path, idempotency_key=name)

        reference_path = root / "reference-input" / "1_1.png"
        reference_path.parent.mkdir()
        reference_image = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(reference_image).rectangle((21, 30, 100, 90), fill=255)
        reference_image.save(reference_path)
        reference_rep = representation("Independent known synthetic reference", purpose=RepresentationPurpose.BINARY)
        reference_import = local.commit_managed_import(principal=actor, project=local.get_project(project.id),
            layer=local.list_layers(project.id)[0], representation=reference_rep,
            plan=local.plan_import_directory(project=project, directory=reference_path.parent), idempotency_key="reference-import")
        reference, = reference_import.versions
        original_mask = local.active_artifact_version(project.id, deterministic_frame_series_id(masks.id, project.frame_id_at(1, 1)))
        fault = root / "controlled-empty-mask.png"
        Image.new("L", (128, 128), 0).save(fault)
        faulty = local.add_managed_artifact_version(principal=actor, project_id=project.id, series_id=original_mask.series_id,
            parent_version_id=original_mask.id, source=fault, idempotency_key="injected-empty-mask")
        q = SimpleNamespace(service=local, project=project)
        store = FilesystemAnalysisStore(local.catalog_root, str(project.id))
        analysis, _ = real_run(q, root / "analysis", store, faulty, reference)
        rule = {"metric_key": "iou", "value_field": "raw_value", "comparison": "lt", "threshold": .9, "target_binding": "A"}
        issue, = local.create_quality_from_analysis(project_id=project.id, principal=actor, run_id=analysis.run_id,
            rule=rule, severity="major", idempotency_key="analysis-findings")
        record(9, analysis_run_id=analysis.run_id, issue_id=issue.id, fault_injection="new empty-mask child version; model quality is not evaluated")
        issue = local.mutate_quality_issue(project_id=project.id, principal=actor, issue_id=issue.id, expected_revision=issue.revision,
            operation="assign", arguments={"assignee_id": performer.id, "note": "Restore the known functional fixture mask"}, idempotency_key="assign")
        record(10, issue_id=issue.id, assignee_id=performer.id)
        corrected = local.add_managed_artifact_version(principal=actor, project_id=project.id, series_id=faulty.series_id,
            parent_version_id=faulty.id, source=reference_path, idempotency_key="correction")
        issue = local.mutate_quality_issue(project_id=project.id, principal=actor, issue_id=issue.id, expected_revision=issue.revision,
            operation="correct", arguments={"evidence_version_ids": [corrected.id], "note": "Known synthetic fixture restored"}, idempotency_key="correct")
        checked, _ = real_run(q, root / "recheck", store, corrected, reference)
        try:
            local.recheck_quality_from_analysis(project_id=project.id, principal=actor, issue_id=issue.id,
                run_id=checked.run_id, expected_revision=issue.revision, idempotency_key="self-recheck", note="Must refuse self-review")
        except AuthorizationError:
            pass
        else:
            raise AssertionError("Self-review accepted")
        record(11, correction_version=corrected.id, parent_version=faulty.id, recheck_run_id=checked.run_id)
        issue = local.recheck_quality_from_analysis(project_id=project.id, principal=reviewer, issue_id=issue.id,
            run_id=checked.run_id, expected_revision=issue.revision, idempotency_key="independent-recheck", note="Measured fixture restoration, no real-device claim")
        issue = local.mutate_quality_issue(project_id=project.id, principal=reviewer, issue_id=issue.id, expected_revision=issue.revision,
            operation="accept", arguments={"note": "Controlled fault repaired and independently checked"}, idempotency_key="accept")
        assert issue.state.value == "accepted"
        record(12, issue_id=issue.id, reviewer_id=str(reviewer.id), state=issue.state.value)

        def physical(operation, arguments):
            return local.mutate_multilayer_model(project_id=project.id, principal=actor,
                expected_revision=local.get_multilayer_model(project.id)["model"]["revision"],
                operation=operation, arguments=arguments, idempotency_key=str(uuid4()))

        physical("set_stack", {"layers": [{"layer_id": v.id, "layer_type": v.type, "z_m": i*1e-6,
            "thickness_m": 1e-7, "conductive": v.type == LayerType.METAL} for i, v in enumerate(layers)]})
        pose_path = root / "declared-fixture-poses.json"
        pose_path.write_text('{"scope":"controlled synthetic poses, not catalog-derived or measured real geometry"}')
        pose = attachment("Declared fixture poses", pose_path)
        tiles = []
        for x in range(1, 4):
            # Manual corrections have no automatic grid-preserving producer.
            # Explicitly establish geometry on the current corrected versions;
            # do not assert that a parent link certifies unchanged pixel grids.
            current_mask = local.active_artifact_version(project.id, deterministic_frame_series_id(masks.id, project.frame_id_at(x, 1)))
            tile = {"frame_id": str(project.frame_id_at(x, 1)), "artifact_version_id": str(current_mask.id),
                "width": 128, "height": 128, "pixel_size_m": [1e-6, 1e-6], "y_axis": "down", "catalog_xy": [x, 1]}
            if x > 1:
                tile.update(declared_affine=[1, 0, (x-1)*128e-6, 0, 1, 0], note="Explicit controlled fixture pose")
            tiles.append(tile)
        physical_plan = MultilayerModel.from_dict(physical("plan_layer_stitch", {"layer_id": layer.id, "plan_version_id": pose.id,
            "reference_frame_id": tiles[0]["frame_id"], "tiles": tiles, "max_error_m": 1e-12})).stitches[-1]
        stitch = PipelineLayerOperation(str(uuid4()), "Stitch corrected current masks", "layer.stitch.v1", "1.0.0", physical_plan.id,
            physical_plan.revision, content=PipelineLayerContent("binary_mask", representation_id=str(masks.id)), parameters={"block_size": 64})
        state = execute([stitch], "stitch")
        mosaic_id, = state["nodes"][stitch.id]["attempts"][-1]["output_version_ids"]
        mosaic = local.get_artifact_version(project.id, mosaic_id)
        assert str(corrected.id) in mosaic.input_version_ids
        with MosaicReader(local.artifact_source_path(project.id, mosaic_id)) as reader:
            reader.validate(expected_plan=physical_plan)
        record(13, mosaic_version=mosaic_id, input_versions=list(mosaic.input_version_ids), stitch_id=physical_plan.id)

        upper_mask = attachment("Controlled upper conductor mask", reference_path, layers[2])
        upper_image = attachment("Independent upper material drawing", source / "1_1.png", layers[2])
        contact_path = root / "contact-evidence.png"
        contact_image = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(contact_image).ellipse((48, 48, 52, 52), fill=255)
        contact_image.save(contact_path)
        contact_version = attachment("Independent synthetic contact drawing", contact_path, layers[1])
        for target, version in ((layer, corrected), (layers[2], upper_mask)):
            physical("calibrate", {"layer_id": target.id, "artifact_version_id": version.id,
                "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9})
        registered = physical("register_affine", {"source_layer_id": layer.id, "target_layer_id": layers[2].id,
            "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9} for p in ((0, 0), (100, 0), (0, 100), (100, 100))], "max_error_m": 1e-9})
        registration = registered["model"]["registrations"][-1]
        assert registration["fit"]["status"] == "fitted"
        proposed = physical("propose_contact", {"source_layer_id": layer.id, "target_layer_id": layers[2].id,
            "contact_layer_id": layers[1].id, "registration_id": registration["id"], "registration_revision": registration["revision"],
            "position_target_m": [50e-6, 50e-6], "radius_m": 1e-8})
        contact = proposed["model"]["contacts"][-1]
        physical("review_contact", {"contact_id": contact["id"], "decision": "confirmed",
            "evidence_version_ids": [contact_version.id], "review_note": "Controlled synthetic stack evidence only"})
        record(14, registration_id=registration["id"], contact_id=contact["id"], scope="synthetic stack only")

        def electrical(operation, arguments):
            current = local.get_electrical_model(project.id)
            return local.mutate_electrical_model(project_id=project.id, principal=actor,
                expected_revision=current["electrical"]["revision"], expected_physical_revision=current["physical_model_revision"],
                operation=operation, arguments=arguments, idempotency_key=str(uuid4()))

        components = []
        for target, mask, evidence, port, direction in ((layer, corrected, image_versions[project.frame_id_at(1, 1)], "A", "input"),
                (layers[2], upper_mask, upper_image, "Y", "output")):
            extracted = electrical("extract_conductors", {"layer_id": target.id, "mask_version_id": mask.id, "connectivity": 4, "foreground_value": 255})
            values = extracted["electrical"]["extractions"][-1]["components"]
            assert len(values) == 1
            component = values[0]["id"]
            components.append(component)
            electrical("review_components", {"component_ids": [component], "decision": "confirmed",
                "evidence_version_ids": [evidence.id], "note": "Independent known synthetic drawing", "declared_net_name": "SIGNAL"})
            electrical("bind_terminal", {"component_id": component, "port": port, "direction": direction,
                "evidence_version_ids": [evidence.id], "note": "Explicit synthetic pad observation"})
        connected = electrical("bind_contact", {"contact_id": contact["id"], "source_component_id": components[0],
            "target_component_id": components[1], "note": "Explicit reviewed synthetic contact binding"})
        assert len(connected["graph"]["nets"]) == 1
        record(15, components=components, nets=len(connected["graph"]["nets"]), scope="reviewed synthetic signal path only")
        exports = root / "export"
        exports.mkdir()
        exported = local.export_electrical_model(project.id)
        edif = exports / "signal.edf"
        edif.write_text(exported["content"], encoding="utf-8")
        from krona.bootstrap import logic_functions_to_dict
        logic = logic_functions_to_dict(edif)
        assert [r["value"] for r in logic["outputs"]["Y"]["truth_table"]] == [0, 1]
        for identifier in [*vector_ids, mosaic_id, model_id]:
            version = local.get_artifact_version(project.id, identifier)
            local.export_artifact_version(project.id, version, exports / (str(identifier)+"-"+version.filename))
        record(16, edif_sha256=exported["sha256"], exported_versions=[*vector_ids, mosaic_id, model_id], krona_truth_table=[0, 1])
        planning("record_progress", {"item_id": plan.items[0].id, "fraction": 1, "actual_effort_hours": 0})
        planning("move_item", {"item_id": plan.items[0].id, "state": "in_review"})
        planning("satisfy_criterion", {"item_id": plan.items[0].id, "criterion_id": plan.items[0].criteria[0].id,
            "evidence_version_ids": [mosaic_id, corrected.id]}, reviewer)
        planning("move_item", {"item_id": plan.items[0].id, "state": "accepted"}, reviewer)
        planning("transition", {"state": "completed"}, reviewer)
        report = local.work_plan_analytics(project.id, principal=actor)
        write_csv(exports / "plan-report.csv", report)
        assert report["source_plan_revisions"][plan.id] == plan.revision
        assert report["summary"]["readiness_fraction"] == 1 and report["summary"]["completed_plans"] == 1
        record(17, report_sha256=hashlib.sha256((exports / "plan-report.csv").read_bytes()).hexdigest(), plan_revisions=report["source_plan_revisions"])
        physical_snapshot, electrical_snapshot = local.get_multilayer_model(project.id), local.get_electrical_model(project.id)
        local.archive_project(principal=actor, project=local.get_project(project.id), idempotency_key="archive")
        record(18, state=local.get_project(project.id).state.value)
        reopened = EmbeddedProjectService(local.data_dir, secret_store=local.secrets)
        assert reopened.get_project(project.id).state is ProjectState.ARCHIVED
        assert reopened.get_quality_issue(project.id, issue.id) == issue
        assert reopened.get_multilayer_model(project.id) == physical_snapshot
        assert reopened.get_electrical_model(project.id) == electrical_snapshot
        record(19, state=reopened.get_project(project.id).state.value, quality_revision=issue.revision)
        local.export_backup(project.id, root / "backup")
        restored = _service(root / "restored")
        restored_project = restored.import_backup(root / "backup")
        assert restored_project.state is ProjectState.ARCHIVED and restored.scan_integrity().valid
        assert restored.get_work_plan(project.id, plan.id) == plan
        assert restored.get_quality_issue(project.id, issue.id) == issue
        assert restored.get_multilayer_model(project.id) == physical_snapshot
        assert restored.get_electrical_model(project.id) == electrical_snapshot
        restored_analysis = FilesystemAnalysisStore(restored.catalog_root, str(project.id))
        for run in (analysis, checked):
            assert restored_analysis.get_run(run.run_id) == store.get_run(run.run_id)
            assert restored_analysis.frame_results(run.run_id, "iou") == store.frame_results(run.run_id, "iou")
        all_versions = [v for s in local.list_artifact_series(project.id) for v in local.artifact_versions(project.id, s.id)]
        for v in all_versions:
            assert restored.get_artifact_version(project.id, v.id) == v
            assert hashlib.sha256(restored.artifact_source_path(project.id, v.id).read_bytes()).hexdigest() == v.sha256
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == original_hashes
        receipt["status"] = "passed"
        record(20, verified_versions=len(all_versions), original_inputs_unchanged=True, backup_integrity=True)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    destination = Path(sys.argv[1]).resolve()
    try:
        run_gate(destination)
    except BaseException as error:
        receipt_path = destination / "receipt.json"
        partial = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"stages": {}}
        partial.update(status="failed", error=f"{type(error).__name__}: {error}")
        receipt_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        raise
