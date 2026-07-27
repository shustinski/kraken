from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from kraken_manager.domain.workflows import ReviewPackageFileV1, ReviewPackageManifestV1
from kraken_manager.infrastructure.review.package import ReturnCategory, ReviewPackageReader


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


FRAME_1 = "00000000-0000-0000-0000-000000000001"
FRAME_2 = "00000000-0000-0000-0000-000000000002"
VERSION_1 = "10000000-0000-0000-0000-000000000001"
VERSION_2 = "10000000-0000-0000-0000-000000000002"


def package_manifest() -> ReviewPackageManifestV1:
    return ReviewPackageManifestV1(
        package_id="20000000-0000-0000-0000-000000000001",
        batch_id="20000000-0000-0000-0000-000000000002",
        project_id="30000000-0000-0000-0000-000000000001",
        layer_id="40000000-0000-0000-0000-000000000001",
        performer_id="50000000-0000-0000-0000-000000000001",
        issued_by="60000000-0000-0000-0000-000000000001",
        issued_at=datetime.now(UTC),
        files=(
            ReviewPackageFileV1(FRAME_1, VERSION_1, sha(b"original-a"), "vectors/1_1.cif", 1, 1),
            ReviewPackageFileV1(FRAME_2, VERSION_2, sha(b"original-b"), "vectors/2_1.cif", 2, 1),
        ),
    )


class ReviewReturnTests(unittest.TestCase):
    def test_strict_byte_categories_and_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vectors").mkdir()
            (root / "vectors" / "1_1.cif").write_bytes(b"original-a")
            (root / "vectors" / "2_1.cif").write_bytes(b"changed-b")
            (root / "unexpected.cif").write_bytes(b"extra")
            inspected = ReviewPackageReader(b"unused").inspect_return(root, package_manifest())
            categories = {item.relative_path: item.category for item in inspected}
            self.assertEqual(ReturnCategory.UNCHANGED, categories["vectors/1_1.cif"])
            self.assertEqual(ReturnCategory.CHANGED, categories["vectors/2_1.cif"])
            self.assertEqual(ReturnCategory.EXTRA, categories["unexpected.cif"])

    def test_stale_base_is_a_conflict_even_for_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vectors").mkdir()
            (root / "vectors" / "1_1.cif").write_bytes(b"original-a")
            inspected = ReviewPackageReader(b"unused").inspect_return(
                root,
                package_manifest(),
                active_vector_versions={FRAME_1: "70000000-0000-0000-0000-000000000001", FRAME_2: VERSION_2},
            )
            by_frame = {item.frame_id: item.category for item in inspected if item.frame_id}
            self.assertEqual(ReturnCategory.STALE_BASE_CONFLICT, by_frame[FRAME_1])
            self.assertEqual(ReturnCategory.MISSING, by_frame[FRAME_2])


if __name__ == "__main__":
    unittest.main()
