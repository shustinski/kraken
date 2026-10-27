"""Offline unit checks for the Linux gate's cache and isolation mechanics."""

import argparse
import ctypes
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import verify_linux_distribution as gate


class LinuxGateToolTests(unittest.TestCase):
    def test_posix_lifetime_import_never_uses_windows_callback_factory(self):
        path = gate.ROOT / "src/kraken_core/process_lifetime.py"
        namespace = {"__name__": "linux_lifetime_import_probe"}
        with patch.object(gate.os, "name", "posix"), patch.object(
            ctypes, "WINFUNCTYPE", create=True, side_effect=AssertionError("Windows API used on POSIX"),
        ):
            exec(compile(path.read_bytes(), str(path), "exec"), namespace)
            namespace["bind_child_process_lifetime"]()
            self.assertEqual(namespace["detached_creationflags"](123), 123)

    def test_exports_complete_wheel_and_records_original_body_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            body = cache / "opaque.body"
            with zipfile.ZipFile(body, "w") as archive:
                archive.writestr("example-1.0.dist-info/METADATA", "Name: example\nVersion: 1.0\n")
                archive.writestr("example-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
            (cache / "partial.body").write_bytes(b"incomplete ZIP download")
            output = root / "wheelhouse"
            manifest = gate.export_cached_wheels([cache], output)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["sha256"], hashlib.sha256(body.read_bytes()).hexdigest())
            self.assertEqual((output / "example-1.0-py3-none-any.whl").read_bytes(), body.read_bytes())

    def test_conflicting_cached_wheels_fail_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            for index in range(2):
                with zipfile.ZipFile(cache / f"{index}.body", "w") as archive:
                    archive.writestr("example-1.0.dist-info/METADATA", "Name: example\nVersion: 1.0\n")
                    archive.writestr("example-1.0.dist-info/WHEEL", "Tag: py3-none-any\n")
                    archive.writestr("different.txt", str(index))
            with self.assertRaisesRegex(ValueError, "Conflicting cached wheel"):
                gate.export_cached_wheels([cache], root / "output")

    def test_offline_host_uses_new_owned_container_and_network_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "uv.lock").write_text("package=[]\n")
            (root / "pyproject.toml").write_text("[project]\nname='example'\n")
            (root / "README.md").write_text("test")
            wheel = root / "example.whl"
            wheel.write_bytes(b"test fixture")
            arguments = argparse.Namespace(report=root / "report", wheel=wheel, pip_cache=[], wheelhouse=None,
                offline=True, profile="core", installation_timeout=30)
            with patch.object(gate, "ROOT", root), patch.object(gate, "run_logged", return_value=0) as execute:
                self.assertEqual(gate.host(arguments), 0)
            command = execute.call_args.args[0]
            self.assertEqual(command[:2], ["docker", "run"])
            self.assertIn("--network=none", command)
            self.assertIn("--offline", command)
            self.assertTrue(command[command.index("--name") + 1].startswith("kraken-linux-audit-"))
            self.assertNotIn("desktop", gate.PROFILES["core"]["extras"])
            self.assertNotIn("registration", gate.PROFILES["core"]["extras"])


if __name__ == "__main__":
    unittest.main()
