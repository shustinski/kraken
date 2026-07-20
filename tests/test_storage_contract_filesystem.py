from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_manager.infrastructure.filesystem import FilesystemEventStore

from storage_contract_suite import BlobStoreContract, EventStoreContract, PROJECT_ID


class FilesystemSemanticContractTests(unittest.TestCase, EventStoreContract, BlobStoreContract):
    def test_semantic_ports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.event_store = FilesystemEventStore(root, PROJECT_ID)
            self.blob_store = FilesystemBlobStore.for_project(root, PROJECT_ID)
            self.assert_event_store_contract()
            self.assert_blob_store_contract()


if __name__ == "__main__":
    unittest.main()

