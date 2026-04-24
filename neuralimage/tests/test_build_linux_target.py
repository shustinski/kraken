from pathlib import Path

import pytest

import tools.build_linux as build_linux


def test_build_linux_pyinstaller_args_uses_linux_dist_and_work_paths(tmp_path):
    spec_path = tmp_path / 'NeuralImage.spec'
    dist_path = tmp_path / 'dist' / 'linux'
    work_path = tmp_path / 'build' / 'linux'

    args = build_linux.build_linux_pyinstaller_args(
        spec_path=spec_path,
        dist_path=dist_path,
        work_path=work_path,
    )

    assert args == [
        '--noconfirm',
        '--clean',
        '--distpath',
        str(dist_path),
        '--workpath',
        str(work_path),
        str(spec_path),
    ]


def test_default_linux_paths_are_scoped_under_project_root(tmp_path):
    assert build_linux.default_dist_path(tmp_path) == Path(tmp_path) / 'dist' / 'linux'
    assert build_linux.default_work_path(tmp_path) == Path(tmp_path) / 'build' / 'linux'


def test_validate_linux_host_rejects_non_linux_without_override(monkeypatch):
    monkeypatch.setattr(build_linux.sys, 'platform', 'win32')
    monkeypatch.delenv('NEURALIMAGE_ALLOW_CROSS_BUILD', raising=False)

    with pytest.raises(SystemExit):
        build_linux._validate_linux_host()


def test_validate_linux_host_allows_override(monkeypatch):
    monkeypatch.setattr(build_linux.sys, 'platform', 'win32')
    monkeypatch.setenv('NEURALIMAGE_ALLOW_CROSS_BUILD', '1')

    build_linux._validate_linux_host()
