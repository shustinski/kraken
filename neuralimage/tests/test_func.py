from pathlib import Path
from uuid import uuid4

from conftest import safe_import_or_skip

safe_import_or_skip('torch')

from torch import nn

from lib.func import get_input_channels, get_names_of_files, compare_filenames_in_folders


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3)


def test_get_input_channels_reads_first_conv():
    assert get_input_channels(_Model()) == 3


def test_get_names_of_files_returns_stems():
    files = ['/tmp/a.jpg', '/tmp/b.bmp']
    assert get_names_of_files(files) == ['a', 'b']


def _make_workspace_test_dir(test_name: str) -> Path:
    root = Path('.test_runtime') / f'{test_name}_{uuid4().hex}'
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_compare_filenames_in_folders_reports_missing_images():
    tmp_path = _make_workspace_test_dir('compare_missing')
    vector_dir = tmp_path / 'vectors'
    image_dir = tmp_path / 'images'
    vector_dir.mkdir()
    image_dir.mkdir()
    (vector_dir / 'a.cif').write_text('x', encoding='utf-8')

    result = compare_filenames_in_folders(image_dir, vector_dir)

    assert isinstance(result, tuple)
    assert result[0] == 0


def test_compare_filenames_in_folders_success():
    tmp_path = _make_workspace_test_dir('compare_success')
    vector_dir = tmp_path / 'vectors'
    image_dir = tmp_path / 'images'
    vector_dir.mkdir()
    image_dir.mkdir()
    (vector_dir / 'a.cif').write_text('x', encoding='utf-8')
    (image_dir / 'a.jpg').write_text('x', encoding='utf-8')

    result = compare_filenames_in_folders(image_dir, vector_dir)

    assert result == 1

