from pathlib import Path

from neuralimage.lib.shared_styles import _rewrite_relative_urls


def test_rewrite_relative_urls_uses_absolute_filesystem_paths_for_qt() -> None:
    base_dir = Path(r'D:\PyCharm\neuralimage-feature-no_cut_dataset\_internal\resources')
    content = 'QCheckBox::indicator:checked { image: url(icons/check_light.svg); }'

    rewritten = _rewrite_relative_urls(content, base_dir)

    assert 'file:///' not in rewritten
    assert 'url("D:/PyCharm/neuralimage-feature-no_cut_dataset/_internal/resources/icons/check_light.svg")' in rewritten


def test_rewrite_relative_urls_preserves_existing_file_uri() -> None:
    base_dir = Path(r'D:\PyCharm\neuralimage-feature-no_cut_dataset\_internal\resources')
    original = 'QCheckBox::indicator:checked { image: url("file:///D:/icons/check_light.svg"); }'

    rewritten = _rewrite_relative_urls(original, base_dir)

    assert rewritten == original
