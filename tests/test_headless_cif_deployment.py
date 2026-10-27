"""The deployed stitch codec must not initialize a plugin GUI or its settings."""
import json
import os
import subprocess
import sys


def test_strict_cif_hole_and_rotated_box_without_contour_application_or_qt(tmp_path):
    source = tmp_path / "profile.cif"
    source.write_text("DS 1 1 1;\nL NM;\n( S 12 12 );\n"
        "P 1 11 9 11 7 9 3 9 3 5 7 5 7 9 9 11 9 3 1 3 1 11;\n"
        "B 2 2 10 2 1 1;\nDF;\nE\n", encoding="utf-8")
    script = '''
import importlib.abc, sys, pathlib, hashlib, json
class BlockGUI(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(('contour.application', 'contour.widget', 'contour.presentation', 'PyQt6', 'cv2', 'torch', 'matplotlib')):
   raise ModuleNotFoundError('GUI dependency is forbidden: '+fullname)
sys.meta_path.insert(0, BlockGUI())
from contour.infrastructure.stitch_cif import load_stitch_cif
p=pathlib.Path(sys.argv[1])
values=load_stitch_cif(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), image_size=(12,12))
assert not any(name.startswith(('contour.application','contour.widget','PyQt6','cv2','torch','matplotlib')) for name in sys.modules)
print(json.dumps([(category, polygon.area, len(polygon.interiors)) for category, polygon in values]))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", script, str(source)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["conductor", 48.0, 1], ["via", 2.0, 0]]
