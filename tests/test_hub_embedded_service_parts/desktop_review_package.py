"""test_hub_embedded_service: test desktop review package."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


class DesktopReviewPackageMixin:
    def test_desktop_review_package_round_trip_uses_manifest_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(
                root / "data",
                secret_store=self._memory_secret_store(),
            )
            owner = service.create_initial_account("owner", "Owner", "")
            performer = service.create_manual_performer(
                name="Reviewer",
                color="#336699",
            )
            project = service.create_project(
                principal=owner.principal,
                name="Review",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=owner.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=0,
                idempotency_key="layer",
            )
            images = service.create_representation(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                name="Images",
                kind=RepresentationKind.IMAGE,
                active=True,
                idempotency_key="images",
            )
            layer = service.list_layers(project.id)[0]
            vectors = service.create_representation(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                name="CIF",
                kind=RepresentationKind.VECTOR,
                source_image_representation_id=images.id,
                active=True,
                idempotency_key="vectors",
            )
            source_images = root / "images"
            source_vectors = root / "vectors"
            source_images.mkdir()
            source_vectors.mkdir()
            (source_images / "1_1.png").write_bytes(b"image")
            (source_vectors / "1_1.cif").write_text("DS 1 1 1;\nDF;\nE\n", encoding="utf-8")
            service.commit_managed_import(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                representation=images,
                plan=service.plan_import_directory(
                    project=project,
                    directory=source_images,
                ),
                idempotency_key="import-images",
            )
            service.commit_managed_import(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                representation=vectors,
                plan=service.plan_import_directory(
                    project=project,
                    directory=source_vectors,
                ),
                idempotency_key="import-vectors",
            )
            batch = service.create_review_batch(
                principal=owner.principal,
                project_id=project.id,
                layer_id=layer.id,
                image_representation_id=images.id,
                vector_representation_id=vectors.id,
                coordinates=((1, 1),),
                assignee_id=performer.id,
                instructions="Check",
                due_at=None,
                idempotency_key="review",
            )
            package = root / "review-package"
            issued = service.export_review_batch(
                principal=owner.principal,
                batch=batch,
                destination=package,
                idempotency_key="export",
            )
            returned_cif = next(package.rglob("*.cif"))
            returned_cif.write_text("DS 1 1 1;\nP 0 0 1 0 1 1;\nDF;\nE\n", encoding="utf-8")

            matched_batch, preflight = service.review_return_preflight(
                principal=owner.principal,
                source=package,
                idempotency_key="preflight",
            )
            self.assertEqual(issued.id, matched_batch.id)
            self.assertTrue(preflight.report.can_commit)
            committed = service.commit_review_return(
                principal=owner.principal,
                batch=matched_batch,
                source=package,
                idempotency_key="commit",
            )
            self.assertEqual(1, len(committed.candidate_versions))
            accepted = service.accept_review(
                principal=owner.principal,
                batch=committed.batch,
                candidate_version_ids=tuple(
                    version.id for version in committed.candidate_versions
                ),
                idempotency_key="accept",
            )
            self.assertEqual("completed", accepted.state.value)
