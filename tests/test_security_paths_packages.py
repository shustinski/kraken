from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import unittest
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kraken_agent.jobs import StagingWorkspace
from kraken_core.plugin_protocol import safe_relative_path
from kraken_core.safe_files import is_link_or_reparse
from kraken_manager.infrastructure.review.package import (
    ReviewPackageLimits,
    ReviewPackageReader,
    UnsafeReviewPackage,
    ReviewPackageWriter,
)
from kraken_manager.infrastructure.review.crypto import (
    Ed25519KeyPair,
    decrypt_archive,
    encrypt_archive,
)
from kraken_manager.domain.workflows import ReviewPackageFileV1, ReviewPackageManifestV1


def _zip(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return buffer.getvalue()


def _signed_manifest(content: bytes = b"CIF bytes") -> ReviewPackageManifestV1:
    return ReviewPackageManifestV1(
        package_id="20000000-0000-0000-0000-000000000001",
        batch_id="20000000-0000-0000-0000-000000000002",
        project_id="30000000-0000-0000-0000-000000000001",
        layer_id="40000000-0000-0000-0000-000000000001",
        performer_id="50000000-0000-0000-0000-000000000001",
        issued_by="60000000-0000-0000-0000-000000000001",
        issued_at=datetime.now(UTC),
        files=(
            ReviewPackageFileV1(
                "00000000-0000-0000-0000-000000000001",
                "10000000-0000-0000-0000-000000000001",
                hashlib.sha256(content).hexdigest(),
                "vectors/1_1.cif",
                1,
                1,
            ),
        ),
    )


class PathSecurityTests(unittest.TestCase):
    def test_windows_reparse_attribute_is_treated_as_a_link(self) -> None:
        fake_stat = SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
        )
        with patch("kraken_core.safe_files.os.lstat", return_value=fake_stat):
            self.assertTrue(is_link_or_reparse("junction"))

    def test_transport_paths_reject_ambiguous_windows_and_posix_names(self) -> None:
        invalid = (
            "a//b",
            "a/./b",
            "folder/file:stream",
            "CON/file.cif",
            "vectors/NUL.txt",
            "vectors/trailing. ",
            "vectors/control\x01.cif",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative_path(value)

    def test_staging_rejects_a_symlinked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"content")
            link = root / "source-link.bin"
            try:
                os.symlink(source, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            workspace = StagingWorkspace(root / "staging", "job-1")
            workspace.create()
            with self.assertRaises(ValueError):
                workspace.stage_file(
                    link,
                    "inputs/source.bin",
                    expected_sha256=hashlib.sha256(b"content").hexdigest(),
                )

    def test_staging_rejects_a_symlinked_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            workspace = StagingWorkspace(root / "staging", "job-1")
            workspace.create()
            workspace_output = workspace.path / "outputs"
            workspace_output.rmdir()
            try:
                os.symlink(outside, workspace_output, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                workspace_output.mkdir()
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            with self.assertRaises(ValueError):
                workspace.resolve("outputs/result.cif")

    def test_folder_reader_does_not_follow_symlinked_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.cif"
            outside.write_bytes(b"secret")
            package = root / "package"
            package.mkdir()
            try:
                os.symlink(outside, package / "linked.cif")
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaises(UnsafeReviewPackage):
                ReviewPackageReader(b"unused").list_relative_paths(str(package))


class ZipPackageSecurityTests(unittest.TestCase):
    def test_secure_encrypted_package_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_bytes(b"CIF bytes")
            relative = "vectors/1_1.cif"
            manifest = _signed_manifest()
            keys = Ed25519KeyPair.generate()
            package = ReviewPackageWriter(keys.private_key).write_encrypted(
                root / "review",
                manifest,
                {relative: source},
                password="correct horse battery staple",
            )
            destination = ReviewPackageReader(keys.public_key).decrypt_to(
                package,
                root / "unpacked",
                password="correct horse battery staple",
            )
            self.assertEqual(b"CIF bytes", (destination / relative).read_bytes())

    def test_unsigned_archive_extra_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_bytes(b"CIF bytes")
            keys = Ed25519KeyPair.generate()
            password = "correct horse battery staple"
            package = ReviewPackageWriter(keys.private_key).write_encrypted(
                root / "review",
                _signed_manifest(),
                {"vectors/1_1.cif": source},
                password=password,
            )
            archive_payload = decrypt_archive(package.read_bytes(), password)
            source_archive = zipfile.ZipFile(io.BytesIO(archive_payload))
            modified_buffer = io.BytesIO()
            with source_archive, zipfile.ZipFile(
                modified_buffer,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as modified:
                for info in source_archive.infolist():
                    modified.writestr(info.filename, source_archive.read(info))
                modified.writestr("unsigned-extra.cif", b"not signed")
            package.write_bytes(encrypt_archive(modified_buffer.getvalue(), password))
            with self.assertRaisesRegex(UnsafeReviewPackage, "signed manifest"):
                ReviewPackageReader(keys.public_key).decrypt_to(
                    package,
                    root / "unpacked",
                    password=password,
                )

    def _decrypt(self, payload: bytes, *, limits: ReviewPackageLimits) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encrypted = root / "input.kraken-review"
            encrypted.write_bytes(b"encrypted")
            target = root / "output"
            reader = ReviewPackageReader(b"unused", limits=limits)
            with patch(
                "kraken_manager.infrastructure.review.package.decrypt_archive",
                return_value=payload,
            ):
                reader.decrypt_to(encrypted, target, password="irrelevant-password")

    def test_zip_path_traversal_is_rejected_before_extraction(self) -> None:
        payload = _zip(
            [
                ("kraken-review.json", b"{}"),
                ("kraken-review.sig", b"sig"),
                ("../outside.cif", b"escape"),
            ]
        )
        with self.assertRaises(ValueError):
            self._decrypt(payload, limits=ReviewPackageLimits())

    def test_zip_entry_count_is_bounded(self) -> None:
        payload = _zip(
            [
                ("kraken-review.json", b"{}"),
                ("kraken-review.sig", b"sig"),
                ("vectors/a.cif", b"a"),
            ]
        )
        with self.assertRaisesRegex(UnsafeReviewPackage, "too many entries"):
            self._decrypt(payload, limits=ReviewPackageLimits(max_entries=2))

    def test_zip_compression_ratio_is_bounded(self) -> None:
        payload = _zip(
            [
                ("kraken-review.json", b"{}"),
                ("kraken-review.sig", b"sig"),
                ("vectors/bomb.cif", b"0" * 100_000),
            ]
        )
        with self.assertRaisesRegex(UnsafeReviewPackage, "compression ratio"):
            self._decrypt(
                payload,
                limits=ReviewPackageLimits(
                    max_compression_ratio=2,
                    ratio_check_min_bytes=1,
                ),
            )

    def test_zip_symlink_entry_is_rejected(self) -> None:
        symlink = zipfile.ZipInfo("vectors/link.cif")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        payload = _zip(
            [
                ("kraken-review.json", b"{}"),
                ("kraken-review.sig", b"sig"),
                (symlink, b"../../outside"),
            ]
        )
        with self.assertRaisesRegex(UnsafeReviewPackage, "link or special"):
            self._decrypt(payload, limits=ReviewPackageLimits())

    def test_encrypted_envelope_size_is_checked_before_decryption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "large.kraken-review"
            package.write_bytes(b"12345")
            reader = ReviewPackageReader(
                b"unused",
                limits=ReviewPackageLimits(max_encrypted_bytes=4),
            )
            with patch(
                "kraken_manager.infrastructure.review.package.decrypt_archive"
            ) as decrypt, self.assertRaisesRegex(UnsafeReviewPackage, "Encrypted review package"):
                reader.decrypt_to(package, root / "output", password="irrelevant-password")
            decrypt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
