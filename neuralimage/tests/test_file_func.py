from lib.file_func import filter_files, filter_images
from tests.helpers import make_test_dir


def test_filter_files_filters_extensions_case_insensitive():
    tmp_path = make_test_dir("file_func_case")
    (tmp_path / 'a.jpg').write_text('x', encoding='utf-8')
    (tmp_path / 'b.BMP').write_text('x', encoding='utf-8')
    (tmp_path / 'c.txt').write_text('x', encoding='utf-8')
    (tmp_path / 'folder.jpg').mkdir()

    result = filter_files(tmp_path, ('.jpg', '.bmp'))

    assert sorted(p.name for p in result) == ['a.jpg', 'b.BMP']


def test_filter_images_uses_supported_extensions():
    tmp_path = make_test_dir("file_images_case")
    (tmp_path / 'a.jpg').write_text('x', encoding='utf-8')
    (tmp_path / 'b.bmp').write_text('x', encoding='utf-8')
    (tmp_path / 'c.png').write_text('x', encoding='utf-8')

    result = filter_images(tmp_path)

    assert sorted(p.name for p in result) == ['a.jpg', 'b.bmp', 'c.png']

