from __future__ import annotations

import numpy as np

from karakal.core.attention_issues import (
    ATTENTION_COMPUTE_LIGHTWEIGHT,
    ATTENTION_COMPUTE_STANDARD,
    ATTENTION_ISSUE_ARTIFACT,
    ATTENTION_ISSUE_BREAK,
    ATTENTION_ISSUE_MERGE,
    ATTENTION_ISSUE_UNCERTAIN_BOUNDARY,
    ATTENTION_ISSUE_UNCERTAIN_FILL,
    AttentionIssue,
    build_attention_issues,
    normalize_attention_compute_mode,
)
from karakal.infra.services import KarakalSettingsService
from PyQt6.QtCore import QSettings


def test_normalize_attention_compute_mode_defaults_to_lightweight() -> None:
    assert normalize_attention_compute_mode(None) == ATTENTION_COMPUTE_LIGHTWEIGHT
    assert normalize_attention_compute_mode("standard") == ATTENTION_COMPUTE_STANDARD
    assert normalize_attention_compute_mode("weird") == ATTENTION_COMPUTE_LIGHTWEIGHT


def test_settings_roundtrip_attention_compute_mode(tmp_path) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    service = KarakalSettingsService(settings)
    assert service.load_attention_compute_mode() == ATTENTION_COMPUTE_LIGHTWEIGHT
    service.save_attention_compute_mode(ATTENTION_COMPUTE_STANDARD)
    service.sync()
    restored = KarakalSettingsService(QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    assert restored.load_attention_compute_mode() == ATTENTION_COMPUTE_STANDARD


def test_build_attention_issues_marks_tiny_blob_as_artifact() -> None:
    values = np.full((128, 128), 0.05, dtype=np.float32)
    values[60:63, 60:63] = 0.5
    mask = values >= 0.45
    issues = build_attention_issues(values, mask, source="output", limit=4)
    assert issues
    assert any(issue.issue_type == ATTENTION_ISSUE_ARTIFACT for issue in issues)


def test_build_attention_issues_prefers_boundary_for_ring() -> None:
    values = np.full((80, 80), 0.05, dtype=np.float32)
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 20:60] = True
    mask[28:52, 28:52] = False
    values[mask] = 1.0
    values[20:60, 20] = 0.5
    values[20:60, 59] = 0.5
    values[20, 20:60] = 0.5
    values[59, 20:60] = 0.5
    issues = build_attention_issues(values, mask, source="output", limit=8)
    assert issues
    assert any(issue.issue_type == ATTENTION_ISSUE_UNCERTAIN_BOUNDARY for issue in issues)


def test_build_attention_issues_detects_merge_bridge() -> None:
    values = np.full((80, 80), 0.0, dtype=np.float32)
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:40, 10:30] = True
    mask[20:40, 50:70] = True
    mask[28:32, 30:50] = True  # thin bridge
    values[mask] = 1.0
    values[28:32, 30:50] = 0.5  # uncertain bridge
    issues = build_attention_issues(
        values, mask, source="output", limit=8, compute_mode=ATTENTION_COMPUTE_STANDARD
    )
    assert issues
    assert any(issue.issue_type in {ATTENTION_ISSUE_MERGE, ATTENTION_ISSUE_UNCERTAIN_FILL} for issue in issues)


def test_build_attention_issues_detects_break_gap() -> None:
    values = np.full((80, 80), 0.0, dtype=np.float32)
    mask = np.zeros((80, 80), dtype=bool)
    mask[30:50, 10:35] = True
    mask[30:50, 45:70] = True
    values[mask] = 1.0
    values[30:50, 35:45] = 0.5  # uncertain gap between lobes
    issues = build_attention_issues(
        values, mask, source="output", limit=8, compute_mode=ATTENTION_COMPUTE_STANDARD
    )
    assert issues
    assert any(issue.issue_type in {ATTENTION_ISSUE_BREAK, ATTENTION_ISSUE_UNCERTAIN_FILL} for issue in issues)


def test_lightweight_skips_merge_break_and_original_mismatch(monkeypatch) -> None:
    values = np.full((80, 80), 0.0, dtype=np.float32)
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:40, 10:30] = True
    mask[20:40, 50:70] = True
    mask[28:32, 30:50] = True
    values[mask] = 1.0
    values[28:32, 30:50] = 0.5

    calls = {"merge": 0, "break": 0, "edge_support": 0}

    def _spy_merge(*_args, **_kwargs):
        calls["merge"] += 1
        return True

    def _spy_break(*_args, **_kwargs):
        calls["break"] += 1
        return True

    def _spy_edge_support(*_args, **_kwargs):
        calls["edge_support"] += 1
        return np.zeros_like(values, dtype=bool)

    monkeypatch.setattr("karakal.core.attention_issues._looks_like_merge", _spy_merge)
    monkeypatch.setattr("karakal.core.attention_issues._looks_like_break", _spy_break)
    monkeypatch.setattr("karakal.core.attention_issues._original_edge_support", _spy_edge_support)

    issues = build_attention_issues(
        values,
        mask,
        original=np.zeros_like(values),
        source="output",
        limit=8,
        compute_mode=ATTENTION_COMPUTE_LIGHTWEIGHT,
    )
    assert issues
    assert calls == {"merge": 0, "break": 0, "edge_support": 0}
    assert all(
        issue.issue_type
        in {
            ATTENTION_ISSUE_UNCERTAIN_BOUNDARY,
            ATTENTION_ISSUE_UNCERTAIN_FILL,
            ATTENTION_ISSUE_ARTIFACT,
        }
        for issue in issues
    )


def test_build_attention_issues_works_on_binary_masks() -> None:
    values = np.zeros((96, 96), dtype=np.float32)
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:70, 20:70] = True
    mask[35:55, 35:55] = False  # hole / ring-like topology
    values[mask] = 1.0
    issues = build_attention_issues(values, mask, source="output", limit=8)
    assert issues, "binary masks must produce attention markers via distance proxy"
    assert all(0.0 <= issue.score <= 1.0 for issue in issues)


def test_build_attention_issues_accepts_precomputed_probability_proxy() -> None:
    values = np.zeros((64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    values[mask] = 1.0
    proxy = np.full((64, 64), 0.1, dtype=np.float32)
    proxy[mask] = 0.9
    proxy[28:36, 28:36] = 0.5
    issues = build_attention_issues(
        values,
        mask,
        source="output",
        limit=4,
        probability_proxy=proxy,
        compute_mode=ATTENTION_COMPUTE_LIGHTWEIGHT,
    )
    assert issues


def test_attention_issue_serialization_roundtrip() -> None:
    issue = AttentionIssue(
        issue_id="m1:break:1",
        issue_type=ATTENTION_ISSUE_BREAK,
        severity="high",
        score=0.7,
        bbox=(0.1, 0.2, 0.3, 0.4),
        label_key="attention.issue.break",
        detail="mean_uncertainty=0.500",
        model_id="m1",
    )
    restored = AttentionIssue.from_dict(issue.to_dict())
    assert restored is not None
    assert restored.issue_type == ATTENTION_ISSUE_BREAK
    assert restored.bbox == (0.1, 0.2, 0.3, 0.4)
