from __future__ import annotations

import hashlib

import cv2
import numpy as np

from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisRecipe,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
)
from karakal.standalone_analysis import StandaloneAnalysisService
from karakal.storage import AnalysisHistoryStore
from karakal.ui.history_dialog import StandaloneHistoryDialog


def _contracts(tmp_path):
    artifacts = []
    for key, values in {"A": [[1, 0]], "B": [[0, 1]]}.items():
        path = tmp_path / "inputs" / f"{key}.png"
        path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path), np.asarray(values, dtype=np.uint8) * 255)
        artifacts.append(
            AnalysisArtifactInput(
                key,
                AnalysisSourceRole.MODEL_OUTPUT,
                f"artifact-{key}",
                f"version-{key}",
                f"inputs/{key}.png",
                "image/png",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    )
    run = AnalysisRunManifest(
        "run-1",
        "standalone",
        ("frame-1",),
        (AnalysisSourceBinding("A", "model-a", "v1"), AnalysisSourceBinding("B", "model-b", "v1")),
        recipe,
        AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "test"),
    )
    partition = AnalysisPartitionJobManifest(
        "job-1",
        run.run_id,
        "part-0",
        run.project_id,
        0,
        1,
        run.fingerprint,
        recipe,
        (AnalysisFrameInput("frame-1", 1, 1, tuple(artifacts)),),
    )
    return run, partition


def test_standalone_service_persists_and_resumes_only_incomplete_partitions(tmp_path) -> None:
    run, partition = _contracts(tmp_path)
    history = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    service = StandaloneAnalysisService(history)

    completed = service.start(
        run,
        (partition,),
        workspace=tmp_path,
        output_dir=tmp_path / "outputs",
    )
    repeated = service.resume(run.run_id, workspace=tmp_path, output_dir=tmp_path / "outputs")

    assert completed.state == "completed"
    assert repeated == completed
    assert history.incomplete_partitions(run.run_id) == ()


def test_standalone_history_dialog_shows_saved_run_and_can_repeat(qtbot, tmp_path) -> None:
    run, _partition = _contracts(tmp_path)
    history = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    history.create_run(run)
    dialog = StandaloneHistoryDialog(history)
    qtbot.addWidget(dialog)
    dialog.table.selectRow(0)

    with qtbot.waitSignal(dialog.repeatRequested) as signal:
        dialog.repeat_button.click()

    assert signal.args == [run]
    assert dialog.table.rowCount() == 1
