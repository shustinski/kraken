from __future__ import annotations
import hashlib
import tempfile
import unittest
from pathlib import Path
from kraken_core.frame_matrix import StoreNamespace
from kraken_core.plugin_protocol import PluginAsset, PluginResultPublicationV2
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.artifacts import ArtifactScope, deterministic_frame_series_id
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem import LocalProjectUnitOfWorkFactory, SQLiteProjectionStore

from test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin
from test_hub_embedded_service_parts.export_artifact_version import ExportArtifactVersionMixin
from test_hub_embedded_service_parts.notes_artifact_lifecycle import NotesArtifactLifecycleMixin
from test_hub_embedded_service_parts.delete_project_removes import DeleteProjectRemovesMixin
from test_hub_embedded_service_parts.legacy_directory_representation import LegacyDirectoryRepresentationMixin
from test_hub_embedded_service_parts.image_vector_representations import ImageVectorRepresentationsMixin
from test_hub_embedded_service_parts.project_layer_lifecycle import ProjectLayerLifecycleMixin
from test_hub_embedded_service_parts.directory_import_preflighted import DirectoryImportPreflightedMixin
from test_hub_embedded_service_parts.sparse_managed_vector import SparseManagedVectorMixin
from test_hub_embedded_service_parts.managed_vector_keeps import ManagedVectorKeepsMixin
from test_hub_embedded_service_parts.project_acl_changes import ProjectAclChangesMixin
from test_hub_embedded_service_parts.desktop_review_package import DesktopReviewPackageMixin
from test_hub_embedded_service_parts.backup_restore_requires import BackupRestoreRequiresMixin
from test_hub_embedded_service_parts.v2_publications_are import V2PublicationsAreMixin

class EmbeddedProjectServiceTests(
    MemorySecretStoreMixin,
    ExportArtifactVersionMixin,
    NotesArtifactLifecycleMixin,
    DeleteProjectRemovesMixin,
    LegacyDirectoryRepresentationMixin,
    ImageVectorRepresentationsMixin,
    ProjectLayerLifecycleMixin,
    DirectoryImportPreflightedMixin,
    SparseManagedVectorMixin,
    ManagedVectorKeepsMixin,
    ProjectAclChangesMixin,
    DesktopReviewPackageMixin,
    BackupRestoreRequiresMixin,
    V2PublicationsAreMixin,
    unittest.TestCase,
):
    pass

if __name__ == "__main__":
    unittest.main()
