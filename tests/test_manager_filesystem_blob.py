from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from kraken_manager.application.ports import BlobStore
from kraken_manager.infrastructure.blob import BlobIntegrityError, FilesystemBlobStore


class FilesystemBlobStoreTests(unittest.TestCase):
    def test_streaming_content_addressing_and_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemBlobStore(Path(directory) / "objects")
            self.assertIsInstance(store, BlobStore)
            chunks = [b"hello", b" ", b"world"]
            digest = hashlib.sha256(b"hello world").hexdigest()

            first = store.put(iter(chunks), expected_sha256=digest)
            self.assertFalse(first.already_existed)
            self.assertEqual(first.blob.sha256, digest)
            self.assertEqual(first.blob.size_bytes, 11)
            object_path = store.root / digest[:2] / digest[2:4] / digest
            self.assertEqual(object_path.read_bytes(), b"hello world")

            second = store.put((b"hello world",), expected_sha256=digest)
            self.assertTrue(second.already_existed)
            self.assertEqual(b"".join(store.iter_bytes(first.blob, chunk_size=2)), b"hello world")
            self.assertEqual(list(store.iter_refs()), [first.blob])

    def test_mismatch_and_corrupt_existing_blob_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemBlobStore(Path(directory) / "objects")
            digest = hashlib.sha256(b"original").hexdigest()
            with self.assertRaises(BlobIntegrityError):
                store.put((b"other",), expected_sha256=digest)
            self.assertFalse(store.exists(digest))

            reference = store.put((b"original",)).blob
            object_path = store.root / digest[:2] / digest[2:4] / digest
            object_path.write_bytes(b"tampered")
            with self.assertRaises(BlobIntegrityError):
                store.put((b"original",))
            self.assertEqual(object_path.read_bytes(), b"tampered")
            with self.assertRaises(BlobIntegrityError):
                store.verify(reference)


if __name__ == "__main__":
    unittest.main()
