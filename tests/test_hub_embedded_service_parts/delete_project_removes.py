"""test_hub_embedded_service: test delete project removes."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_core.frame_matrix import StoreNamespace
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation


class DeleteProjectRemovesMixin:
    def test_delete_project_removes_kraken_cache_but_preserves_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("operator", "Operator", "")
            (root / "source").mkdir()
            (root / "derived").mkdir()
            project = service.create_project(
                principal=session.principal,
                name="Preserved project",
                width=2,
                height=2,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="create-project",
                source_root=root / "source",
                derived_root=root / "derived",
            )
            binding = service.project_workspace(project.id)
            self.assertIsNotNone(binding)
            assert binding is not None
            source_file = Path(binding.source_project_dir) / "img" / "0.jpg"
            result_file = Path(binding.derived_project_dir) / "result" / "model.bin"
            vector_file = Path(binding.derived_project_dir) / "vector" / "0.cif"
            source_file.write_bytes(b"image")
            result_file.write_bytes(b"model")
            vector_file.write_bytes(b"vector")

            project_cache = service.catalog_root / "projects" / str(project.id)
            (project_cache / "staging" / "temporary.bin").write_bytes(b"cache")
            thumbnail_root = service.data_dir / "cache" / "frame-thumbnails"
            thumbnail_namespace = thumbnail_root / StoreNamespace(
                plugin="matrix",
                project=str(project.id),
                generation="v1",
            ).digest()
            thumbnail_namespace.mkdir(parents=True)
            (thumbnail_namespace / "thumb.sqlite3").write_bytes(b"cache")
            project_staging = service.data_dir / "agent-staging" / f"{project.id}-job"
            project_staging.mkdir(parents=True)
            (project_staging / "job.json").write_text(
                f'{{"project_id":"{project.id}"}}',
                encoding="utf-8",
            )
            other_staging = service.data_dir / "agent-staging" / "other-job"
            other_staging.mkdir()
            (other_staging / "job.json").write_text(
                '{"project_id":"other-project"}',
                encoding="utf-8",
            )

            result = service.delete_project(
                principal=session.principal,
                project=project,
                confirmation_name=project.name,
            )

            self.assertTrue(result.catalog_cache_removed)
            self.assertTrue(result.thumbnail_cache_removed)
            self.assertEqual(1, result.staging_directories_removed)
            self.assertFalse(project_cache.exists())
            self.assertFalse(thumbnail_namespace.exists())
            self.assertFalse(project_staging.exists())
            self.assertTrue(other_staging.exists())
            self.assertTrue(source_file.is_file())
            self.assertTrue(result_file.is_file())
            self.assertTrue(vector_file.is_file())
            self.assertEqual(
                (),
                tuple(item for item in service.list_projects(include_archived=True) if item.id == project.id),
            )
            self.assertEqual(
                frozenset(),
                service.project_roles(project.id, session.principal.id),
            )
