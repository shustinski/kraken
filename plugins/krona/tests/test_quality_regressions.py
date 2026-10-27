from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from krona.application.logic_functions import ExtractLogicFunctions, _Transistor
from krona.bootstrap import logic_functions_to_dict
from krona.infrastructure.edif_repository import EdifRepository


def _extractor():
    return ExtractLogicFunctions(repository=None)


def test_gate_net_is_not_replaced_by_similarly_numbered_input():
    assert _extractor()._select_gate_net(["N00726"], {"N00727"}, {}, "M1") == "N00726"


def test_unknown_gate_does_not_prove_output_level():
    tx = _Transistor("M1", "nmos", "UNKNOWN", "GND", "OUT")
    assert _extractor()._evaluate_output_state("OUT", {}, {}, {"VDD"}, {"GND"}, [tx]) is None


def test_conflict_on_output_overrides_previously_known_input_level():
    tx = _Transistor("M1", "nmos", "VDD", "GND", "OUT")
    assert _extractor()._evaluate_output_state(
        "OUT", {"IN1": 1}, {"IN1": "OUT"}, {"VDD"}, {"GND"}, [tx]
    ) is None


def test_valid_inverter_remains_deterministic():
    txs = [
        _Transistor("P1", "pmos", "IN", "VDD", "OUT"),
        _Transistor("N1", "nmos", "IN", "GND", "OUT"),
    ]
    for bit in (0, 1):
        assert _extractor()._evaluate_output_state(
            "OUT", {"IN1": bit}, {"IN1": "IN"}, {"VDD"}, {"GND"}, txs
        ) == 1 - bit


def test_repository_reloads_rewritten_artifact_with_identical_size_and_mtime(tmp_path):
    path = tmp_path / "revision.edf"
    path.write_text("(edif FIRST)", encoding="utf-8")
    repository = EdifRepository()
    initial = repository._parser_for(path)
    assert repository._parser_for(path) is initial
    stat = path.stat()
    path.write_text("(edif OTHER)", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    revised = repository._parser_for(path)
    assert revised is not initial
    assert revised.file_text == "(edif OTHER)"


def test_extracted_schematic_result_is_not_electrical_confirmation():
    path = Path(__file__).resolve().parents[1] / "resources/test_edifs/e129_model_logic.EDF"
    result = logic_functions_to_dict(path)
    assert result["meta"]["electrical_connectivity"] == "unverified"
    assert result["meta"]["analysis_basis"] == "explicit_edif_joined"
    assert result["meta"]["source_artifact"] == str(path.resolve())
    assert result["meta"]["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert result["meta"]["net_aliases"]


def test_source_revision_change_during_analysis_is_rejected(tmp_path):
    import pytest

    source = Path(__file__).resolve().parents[1] / "resources/test_edifs/e129_model_logic.EDF"
    path = tmp_path / "source.edf"
    path.write_bytes(source.read_bytes())

    class ChangingRepository(EdifRepository):
        def read_view_index(self, artifact):
            result = super().read_view_index(artifact)
            artifact.write_bytes(artifact.read_bytes() + b"\n")
            return result

    with pytest.raises(ValueError, match="source changed"):
        ExtractLogicFunctions(ChangingRepository()).execute(path)
