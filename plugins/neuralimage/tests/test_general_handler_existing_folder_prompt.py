from neuralimage.model.general_neural_handler import GeneralNeuralHandler


def _build_handler(question):
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.question = question
    return handler


class _FolderStub:
    def __init__(self, name: str, exists: bool = True):
        self.name = name
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


def test_existing_folder_question_uses_no_as_default_with_timeout():
    asked: dict[str, object] = {}
    folder = _FolderStub('input_dir')

    def _question(text: str, header: str, default_answer: bool, timeout_seconds: int):
        asked['text'] = text
        asked['header'] = header
        asked['default_answer'] = default_answer
        asked['timeout_seconds'] = timeout_seconds
        return True

    handler = _build_handler(_question)
    answer = GeneralNeuralHandler._check_folder_existance(handler, folder)

    assert answer is True
    assert asked == {
        'text': 'Папка input_dir существует, использовать данные из неё?',
        'header': 'Папка существует',
        'default_answer': False,
        'timeout_seconds': 15,
    }


def test_existing_folder_is_deleted_when_answer_is_no(monkeypatch):
    import neuralimage.model.general_neural_handler as target
    removed: list[tuple[object, bool]] = []
    folder = _FolderStub('label_dir')

    def _fake_rmtree(path, ignore_errors=False):
        removed.append((path, ignore_errors))

    monkeypatch.setattr(target.shutil, 'rmtree', _fake_rmtree)

    handler = _build_handler(lambda *args, **kwargs: False)
    answer = GeneralNeuralHandler._check_folder_existance(handler, folder)

    assert answer is False
    assert removed == [(folder, False)]
