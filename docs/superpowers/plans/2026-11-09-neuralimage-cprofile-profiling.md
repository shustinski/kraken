# NeuralImage cProfile Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add contour-style cProfile profiling (env + ini, off by default) that captures every Python function in each training step and each recognized frame, so the progressive slowdown can be localized.

**Architecture:** A central `neuralimage/infrastructure/profiling.py` (mirroring `contour/src/contour/infrastructure/profiling.py`) exposes enable switches, a rotating-log report writer, cProfile enable/disable with conflict handling, and a reusable `cprofile_region(kind, label, extra)` context manager. Training wraps each batch in `run_train_epoch.py`; recognition wraps each frame in the single-thread path and each worker-item in the multiprocessing cut/predict/sew workers. An ini file (`resources/neuralimage.ini`) + `NEURALIMAGE_PROFILE*` env vars control enable/top-lines/stride, loaded via a new `runtime_config.py`.

**Tech Stack:** Python 3.14, stdlib `cProfile` / `pstats` / `configparser` (no new dependencies), pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-neuralimage-cprofile-profiling-design.md`

## Global Constraints

- Python `>=3.14` (per `plugins/neuralimage/pyproject.toml`).
- **No new third-party dependencies** — `cProfile`, `pstats`, `configparser`, `logging`, `pathlib` are stdlib.
- Env var prefix is `NEURALIMAGE_PROFILE*`; ini section is `[profiling]` in `resources/neuralimage.ini` (plugin root, next to the existing `resources/*.json` / `resources/*.md`).
- Profiling is **off by default** (`enabled = false` in the ini).
- Profiling is **observation-only**: zero overhead and zero behavior change when disabled; enabling must not change training results or recognition output.
- The ini path is resolved with the existing `neuralimage.lib.runtime_paths.resolve_resource_path("neuralimage.ini")` — do not reinvent contour's self-contained path logic.
- The ini must be packaged: add `"resources/*.ini"` to the `[tool.setuptools.data-files]` `"resources"` list in `plugins/neuralimage/pyproject.toml`.
- cProfile allows **one active profiler per thread** — always go through `try_enable_profiler` / `try_disable_profiler` and fall back to wall-clock-only on conflict.
- Report file names are pid + timestamp suffixed and pruned to the newest 500 (so multiprocessing workers do not collide).
- Tests live flat in `plugins/neuralimage/tests/` (per `testpaths = ["tests"]`), import from `src` (per `pythonpath = ["src"]`).

## Review Focus

- **Another cProfile already active on the thread** (debugger / nested profiler) when a region starts → must fall back to a wall-clock-only report with `cprofile_skipped=yes`, never crash. (Task 2 `test_cprofile_region_busy_falls_back_wallclock`.)
- **An exception raised inside a profiled region** (e.g. `TrainingInstabilityError`, a pause) → the region's `finally` must still disable the profiler; no leaked active profiler, no crash in `__exit__`. (Task 3 `test_train_step_region_disables_on_exception`.)
- **Profiling disabled (default) in a real run** → no cProfile allocated, no log file created, no measurable behavior change. (Tasks 2–5 "disabled" tests; Task 3 no-regression run.)
- **Long recognition (many frames) with profiling on, across many worker processes** → log rotation keeps disk bounded (≤500 files) and pid-suffixed names do not collide. (Task 2 `test_prune_profile_logs`.)
- **Env var set but ini not readable from a child process** → env takes priority over ini, so workers resolve enable correctly even if the ini path differs per process. (Task 2 `test_profiling_enabled_env_beats_ini`.)

---

### Task 1: ini file + `runtime_config.py`

**Files:**
- Create: `plugins/neuralimage/resources/neuralimage.ini`
- Create: `plugins/neuralimage/src/neuralimage/infrastructure/runtime_config.py`
- Modify: `plugins/neuralimage/pyproject.toml` (data-files)
- Test: `plugins/neuralimage/tests/test_neuralimage_runtime_config.py`

**Interfaces:**
- Produces (used by Task 2):
  - `bundled_config_path() -> Path` — equals `resolve_resource_path("neuralimage.ini")`.
  - `runtime_config_path() -> Path` — `NEURALIMAGE_CONFIG` env override if set (non-empty), else `bundled_config_path()`.
  - `load_runtime_config() -> configparser.ConfigParser` — cached (`lru_cache(maxsize=1)`), reads `runtime_config_path()` as UTF-8, `interpolation=None`.
  - `clear_runtime_config_cache() -> None`.
  - `config_string(section: str, option: str, default: str) -> str`.
  - `config_bool(section: str, option: str, default: bool) -> bool`.
  - `config_int(section: str, option: str, default: int, *, minimum: int | None = None) -> int`.
  - `config_float(section: str, option: str, default: float, *, minimum: float | None = None) -> float`.
  - Module constants `CONFIG_ENV = "NEURALIMAGE_CONFIG"`, `CONFIG_FILENAME = "neuralimage.ini"`.

- [ ] **Step 1: Write the failing tests**

Create `plugins/neuralimage/tests/test_neuralimage_runtime_config.py`:

```python
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clear_config_cache():
    from neuralimage.infrastructure import runtime_config
    runtime_config.clear_runtime_config_cache()
    yield
    runtime_config.clear_runtime_config_cache()


def test_bundled_path_uses_resource_path():
    from neuralimage.infrastructure import runtime_config
    from neuralimage.lib.runtime_paths import resolve_resource_path
    assert runtime_config.bundled_config_path() == resolve_resource_path("neuralimage.ini")


def test_env_override_path(monkeypatch, tmp_path):
    from neuralimage.infrastructure import runtime_config
    custom = tmp_path / "custom.ini"
    custom.write_text("[profiling]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(custom))
    assert runtime_config.runtime_config_path() == custom
    assert runtime_config.config_bool("profiling", "enabled", False) is True


def test_config_bool_default_when_missing():
    from neuralimage.infrastructure import runtime_config
    assert runtime_config.config_bool("profiling", "nope", True) is True


def test_config_int_minimum_clamps(monkeypatch, tmp_path):
    from neuralimage.infrastructure import runtime_config
    custom = tmp_path / "c.ini"
    custom.write_text("[profiling]\ntop_lines = 0\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(custom))
    assert runtime_config.config_int("profiling", "top_lines", 40, minimum=1) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest plugins/neuralimage/tests/test_neuralimage_runtime_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuralimage.infrastructure.runtime_config'`.

- [ ] **Step 3: Create the ini file**

Create `plugins/neuralimage/resources/neuralimage.ini`:

```ini
; NeuralImage runtime configuration. Restart the application after editing.

[profiling]
; Master switch. A per-profiler switch below has priority.
enabled = false
log_file = profiling.log
; A new timestamped file is created for every launch; the newest 500 are retained.
top_lines = 40

; Leave a value empty to inherit "enabled"; set true/false for an override.
training =
recognition =

training_top_lines = 40
recognition_top_lines = 40
training_step_stride = 1
```

- [ ] **Step 4: Implement `runtime_config.py`**

Create `plugins/neuralimage/src/neuralimage/infrastructure/runtime_config.py`. Model it on `contour/src/contour/infrastructure/runtime_config.py`, but resolve the bundled path via `resolve_resource_path` (not a self-contained frozen-app path chain). Key points:
- `from neuralimage.lib.runtime_paths import resolve_resource_path`.
- `bundled_config_path()` returns `resolve_resource_path(CONFIG_FILENAME)`.
- `runtime_config_path()` returns `Path(os.environ[CONFIG_ENV]).expanduser()` when the env var is a non-empty string, else `bundled_config_path()`.
- `load_runtime_config()` is `@lru_cache(maxsize=1)`, builds a `configparser.ConfigParser(interpolation=None)` and `parser.read(runtime_config_path(), encoding="utf-8")`.
- `config_bool/int/float/string` mirror contour's (return `default` on `ValueError` / missing option; `config_int`/`config_float` clamp to `minimum` when provided).

- [ ] **Step 5: Package the ini**

Modify `plugins/neuralimage/pyproject.toml` `[tool.setuptools.data-files]`:

```toml
"resources" = [
    "resources/*.json",
    "resources/*.md",
    "resources/*.ini",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest plugins/neuralimage/tests/test_neuralimage_runtime_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add plugins/neuralimage/resources/neuralimage.ini \
        plugins/neuralimage/src/neuralimage/infrastructure/runtime_config.py \
        plugins/neuralimage/pyproject.toml \
        plugins/neuralimage/tests/test_neuralimage_runtime_config.py
git commit -m "feat(neuralimage): add profiling ini + runtime_config loader"
```

---

### Task 2: central `profiling.py`

**Files:**
- Create: `plugins/neuralimage/src/neuralimage/infrastructure/profiling.py`
- Test: `plugins/neuralimage/tests/test_neuralimage_profiling.py`

**Interfaces:**
- Consumes (Task 1): `config_bool`, `config_int` from `neuralimage.infrastructure.runtime_config`.
- Produces (used by Tasks 3–5):
  - `profiling_enabled(kind: str, *, default: bool = False, legacy_env: tuple[str, ...] = ()) -> bool`
    Resolution order: env `NEURALIMAGE_PROFILE_<KIND.upper()>` → each `legacy_env` → master env (`NEURALIMAGE_PROFILE`, `NEURALIMAGE_PROFILING`, `NEURALIMAGE_PROFILE_ALL`) → ini `[profiling] <kind>` (only if present) → ini `[profiling] enabled` (only if present) → `default`.
  - `profiling_top_lines(kind: str, default: int) -> int` — env `NEURALIMAGE_PROFILE_<KIND.upper()>_TOP` → `NEURALIMAGE_PROFILE_TOP` → ini `<kind>_top_lines` → ini `top_lines` → `default`.
  - `training_profiling_enabled() -> bool`, `recognition_profiling_enabled() -> bool`.
  - `training_top_lines() -> int` (default 40), `recognition_top_lines() -> int` (default 40).
  - `training_profile_stride() -> int` — env `NEURALIMAGE_PROFILE_TRAINING_STEP_STRIDE` → ini `training_step_stride` → `1` (minimum 1).
  - `set_profile_base_dir(path: Path | None) -> None` — sets the process's base dir for the log.
  - `profiling_log_path() -> Path` — `<base_dir or cwd>/<log_file>` where `log_file` is ini `[profiling] log_file` (default `profiling.log`); an absolute `log_file` is used as-is.
  - `write_profile_report(*messages: object) -> None` — appends to a rotating UTF-8 log (one module-level `FileHandler` per process, lazy).
  - `reset_profile_output() -> None` — closes the handler so a new file is created next write.
  - `try_enable_profiler(profiler: cProfile.Profile) -> bool` — `profiler.enable()`; on `ValueError` whose message contains `"Another profiling tool is already active"` return `False`, else re-raise.
  - `try_disable_profiler(profiler: cProfile.Profile) -> None` — `profiler.disable()` swallowing `ValueError`.
  - `@contextmanager cprofile_region(kind: str, *, label: str, extra: str = "") -> Iterator[None]` — see Step 3.
  - Module constants: `PROFILE_ENV_TRUE`, `PROFILE_ENV_FALSE`, `MAX_PROFILE_LOG_FILES = 500`, `DEFAULT_TOP_LINES = 40`.

- [ ] **Step 1: Write the failing tests**

Create `plugins/neuralimage/tests/test_neuralimage_profiling.py`:

```python
from __future__ import annotations

import cProfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    for var in (
        "NEURALIMAGE_PROFILE", "NEURALIMAGE_PROFILING", "NEURALIMAGE_PROFILE_ALL",
        "NEURALIMAGE_PROFILE_TRAINING", "NEURALIMAGE_PROFILE_RECOGNITION",
        "NEURALIMAGE_PROFILE_TOP", "NEURALIMAGE_PROFILE_TRAINING_TOP",
        "NEURALIMAGE_PROFILE_RECOGNITION_TOP", "NEURALIMAGE_PROFILE_TRAINING_STEP_STRIDE",
        "NEURALIMAGE_CONFIG",
    ):
        monkeypatch.delenv(var, raising=False)
    runtime_config.clear_runtime_config_cache()
    profiling.reset_profile_output()
    profiling.set_profile_base_dir(tmp_path)
    yield
    profiling.reset_profile_output()
    profiling.set_profile_base_dir(None)
    runtime_config.clear_runtime_config_cache()


def test_enabled_default_false():
    from neuralimage.infrastructure import profiling
    assert profiling.training_profiling_enabled() is False
    assert profiling.recognition_profiling_enabled() is False


def test_enabled_env_per_kind(monkeypatch):
    from neuralimage.infrastructure import profiling
    monkeypatch.setenv("NEURALIMAGE_PROFILE_TRAINING", "1")
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is False


def test_enabled_env_master(monkeypatch):
    from neuralimage.infrastructure import profiling
    monkeypatch.setenv("NEURALIMAGE_PROFILE", "on")
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is True


def test_enabled_env_beats_ini(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    ini = tmp_path / "n.ini"
    ini.write_text("[profiling]\nenabled = true\ntraining = false\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(ini))
    runtime_config.clear_runtime_config_cache()
    monkeypatch.setenv("NEURALIMAGE_PROFILE_TRAINING", "1")
    # env per-kind must win even though ini says training=false
    assert profiling.training_profiling_enabled() is True


def test_enabled_inherits_ini_master(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    ini = tmp_path / "n.ini"
    ini.write_text("[profiling]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(ini))
    runtime_config.clear_runtime_config_cache()
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is True


def test_try_enable_conflict_returns_false():
    from neuralimage.infrastructure import profiling
    profiler = cProfile.Profile()
    try:
        profiler.enable()
    except Exception:
        pass
    # force a conflict by leaving it enabled
    try:
        assert profiling.try_enable_profiler(cProfile.Profile()) is False
    finally:
        profiler.disable()


def test_write_report_creates_pid_suffixed_file(tmp_path):
    from neuralimage.infrastructure import profiling
    profiling.set_profile_base_dir(tmp_path)
    profiling.reset_profile_output()
    profiling.write_profile_report("hello-report")
    files = list(tmp_path.glob("profiling.log-*"))
    assert files, "expected a pid+timestamp suffixed log file"
    assert "hello-report" in files[0].read_text(encoding="utf-8")


def test_prune_profile_logs(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling
    monkeypatch.setattr(profiling, "MAX_PROFILE_LOG_FILES", 3)
    configured = tmp_path / "profiling.log"
    for i in range(5):
        p = tmp_path / f"profiling.log-{i}-p1.log"
        p.write_text("x", encoding="utf-8")
        p.touch()
    profiling._prune_profile_logs(configured)
    remaining = list(tmp_path.glob("profiling.log-*"))
    assert len(remaining) == 3  # newest 3 retained, oldest 2 removed


def test_cprofile_region_disabled_noop(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with profiling.cprofile_region("recognition", label="recognition frame", extra="x"):
        pass
    assert calls == []


def test_cprofile_region_enabled_writes_report(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(profiling, "profiling_enabled", lambda *a, **k: True)
    with profiling.cprofile_region("recognition", label="recognition frame", extra="frame=1/1 name=a.png"):
        sum(range(1000))
    assert calls
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition frame]" in output
    assert "wall_ms=" in output
    assert "[neuralimage recognition frame stats]" in output


def test_cprofile_region_busy_falls_back_wallclock(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(profiling, "profiling_enabled", lambda *a, **k: True)
    monkeypatch.setattr(profiling, "try_enable_profiler", lambda p: False)
    with profiling.cprofile_region("recognition", label="recognition frame", extra="x"):
        pass
    output = "\n".join(str(m) for c in calls for m in c)
    assert "wall_ms=" in output
    assert "cprofile_skipped=yes" in output


def test_training_profile_stride_default(monkeypatch):
    from neuralimage.infrastructure import profiling
    assert profiling.training_profile_stride() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest plugins/neuralimage/tests/test_neuralimage_profiling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuralimage.infrastructure.profiling'`.

- [ ] **Step 3: Implement `profiling.py`**

Create `plugins/neuralimage/src/neuralimage/infrastructure/profiling.py`. Mirror `contour/src/contour/infrastructure/profiling.py` for the env/ini resolution, the rotating-log writer (`write_profile_report`, `_new_profile_log_path`, `_profile_log_files`, `_prune_profile_logs`, `reset_profile_output`), and `try_enable_profiler` / `try_disable_profiler`. Adaptations for neuralimage:
- Env prefix is `NEURALIMAGE_PROFILE*`; master env names are `NEURALIMAGE_PROFILE`, `NEURALIMAGE_PROFILING`, `NEURALIMAGE_PROFILE_ALL`.
- `set_profile_base_dir(path)` stores a module-level `_PROFILE_BASE_DIR: Path | None`; `profiling_log_path()` returns `_PROFILE_BASE_DIR / log_file` when set (and `log_file` is relative), else resolves `log_file` (ini `[profiling] log_file`, default `profiling.log`) against `Path.cwd()`. An absolute ini `log_file` is returned as-is.
- Add `training_profiling_enabled()`, `recognition_profiling_enabled()`, `training_top_lines()`, `recognition_top_lines()`, `training_profile_stride()`.
- Add the reusable region:

```python
@contextmanager
def cprofile_region(kind: str, *, label: str, extra: str = "") -> Iterator[None]:
    if not profiling_enabled(kind, default=False):
        yield
        return
    profiler = cProfile.Profile()
    active = try_enable_profiler(profiler)
    started = perf_counter()
    try:
        yield
    finally:
        wall_ms = (perf_counter() - started) * 1000.0
        if active:
            try_disable_profiler(profiler)
        write_profile_report(*_format_cprofile_report(kind, label=label, extra=extra, wall_ms=wall_ms, profiler=profiler if active else None))
```

`_format_cprofile_report(kind, *, label, extra, wall_ms, profiler) -> list[str]` builds:
- summary line `f"[neuralimage {label}] {extra} wall_ms={wall_ms:.3f}".rstrip()` (append ` cprofile_skipped=yes` when `profiler is None`);
- when `profiler is not None`, a second line `f"[neuralimage {label} stats] top={n}\n{...}"` where `n = profiling_top_lines(kind, DEFAULT_TOP_LINES)` and the stats block is `pstats.Stats(profiler, stream=io.StringIO()).sort_stats("cumtime").print_stats(n)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest plugins/neuralimage/tests/test_neuralimage_profiling.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add plugins/neuralimage/src/neuralimage/infrastructure/profiling.py \
        plugins/neuralimage/tests/test_neuralimage_profiling.py
git commit -m "feat(neuralimage): add central cProfile profiling module"
```

---

### Task 3: training per-batch cProfile

**Files:**
- Create: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/train_step_profile.py`
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/run_train_epoch.py:68-160`
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/run_impl.py:10-16`
- Test: `plugins/neuralimage/tests/test_train_step_profile.py`

**Interfaces:**
- Consumes (Task 2): `cprofile_region`, `training_profiling_enabled`, `training_profile_stride`, `set_profile_base_dir`, `profiling_log_path`.
- Produces:
  - `train_step_profile_context(*, epoch: int, batch_index: int, global_batch: int) -> ContextManager` — returns `cprofile_region("training", label="train step", extra=f"epoch={epoch+1} batch={batch_index+1} global={global_batch}")` when `training_profiling_enabled()` and `global_batch % max(1, training_profile_stride()) == 0`, else `contextlib.nullcontext()`.
  - `train_profiling_notice_line() -> str | None` — `None` when `training_profiling_enabled()` is false, else `f"cProfile profiling enabled, log={profiling_log_path()}"`.

- [ ] **Step 1: Write the failing tests**

Create `plugins/neuralimage/tests/test_train_step_profile.py`:

```python
from __future__ import annotations

import pytest


def test_disabled_returns_nullcontext(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: False)
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    ctx = tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0)
    with ctx:
        sum(range(100))
    assert calls == []  # disabled -> no profiler, no report


def test_enabled_writes_train_step_report(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    calls = []
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 1)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0):
        sum(range(1000))
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage train step]" in output
    assert "epoch=1 batch=1 global=0" in output


def test_stride_skips_non_matching_global_batch(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    calls = []
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 2)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=1):
        pass
    assert calls == []  # 1 % 2 != 0 -> no report


def test_notice_line(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    profiling.set_profile_base_dir(tmp_path)
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: False)
    assert tsp.train_profiling_notice_line() is None
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    line = tsp.train_profiling_notice_line()
    assert line is not None
    assert "cProfile profiling enabled" in line
    assert "log=" in line


def test_region_disables_on_exception(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 1)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: None)
    state = {"enabled": 0, "disabled": 0}

    class FakeProfiler:
        def enable(self):
            state["enabled"] += 1
        def disable(self):
            state["disabled"] += 1

    class FakeCProfile:
        @staticmethod
        def Profile():
            return FakeProfiler()

    monkeypatch.setattr(profiling, "cProfile", FakeCProfile)
    with pytest.raises(RuntimeError):
        with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0):
            raise RuntimeError("boom")
    assert state["enabled"] == 1
    assert state["disabled"] == 1  # disabled in finally even when the body raises
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest plugins/neuralimage/tests/test_train_step_profile.py -v`
Expected: FAIL — `ModuleNotFoundError` for `train_step_profile`.

- [ ] **Step 3: Implement `train_step_profile.py`**

Create `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/train_step_profile.py`:

```python
"""model_train_and_recognition: per-batch cProfile context."""
from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

from neuralimage.infrastructure.profiling import (
    cprofile_region,
    profiling_log_path,
    training_profile_stride,
    training_profiling_enabled,
)


def train_step_profile_context(*, epoch: int, batch_index: int, global_batch: int) -> ContextManager:
    if training_profiling_enabled() and (global_batch % max(1, training_profile_stride()) == 0):
        return cprofile_region(
            "training",
            label="train step",
            extra=f"epoch={epoch + 1} batch={batch_index + 1} global={global_batch}",
        )
    return nullcontext()


def train_profiling_notice_line() -> str | None:
    if not training_profiling_enabled():
        return None
    return f'cProfile profiling enabled, log={profiling_log_path()}'
```

- [ ] **Step 4: Wrap the training loop body in `run_train_epoch.py`**

In `run_train_epoch.py`, import `train_step_profile_context`. In `_run_train_epoch`, the `while True:` body currently starts at the line `data_wait_started_at = time.perf_counter()` and ends at the `pause_event` check (`raise _TrainingPauseRequested(...)`). Wrap that entire body (the block from `data_wait_started_at = ...` through the final `if pause_event.is_set(): raise ...`) in:

```python
with train_step_profile_context(epoch=epoch, batch_index=batch_index, global_batch=self._global_batch):
```

This adds one indentation level to the whole body. All existing `break`, `continue`, `raise`, and the inner `with self._profile_region(...)` blocks stay inside; Python runs the context `__exit__` on `break`/`continue`/`raise`, so every iteration (including skipped and early-stop ones) is reported when enabled, and the disabled path is a `nullcontext` (zero overhead).

- [ ] **Step 5: Set the profiling base dir + bus notice at training start in `run_impl.py`**

In `run_impl.py`, import `set_profile_base_dir` from `neuralimage.infrastructure.profiling` and `train_profiling_notice_line` from `...model_train_and_recognition_parts.train_step_profile`. At the top of `_run_impl` (before the epoch loop), add:

```python
set_profile_base_dir(self._save_path.parent)
notice = train_profiling_notice_line()
if notice is not None:
    self._bus.put(['logging', notice])
```

This mirrors the existing torch profiler's `self._bus.put(['logging', ...])` start notice (spec §6).

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `pytest plugins/neuralimage/tests/test_train_step_profile.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 7: No-regression run (profiling disabled by default)**

Run: `pytest plugins/neuralimage/tests/test_automatic_early_stopping.py -q`
Expected: PASS — confirms wrapping the loop body in a `nullcontext` (profiling off) does not change training behavior. If this test file does not exist, run the closest existing training-loop test in `plugins/neuralimage/tests/` that exercises `_run_train_epoch` / epoch stepping, and confirm it still passes.

- [ ] **Step 8: Commit**

```bash
git add plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/train_step_profile.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/run_train_epoch.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/model_train_and_recognition_parts/run_impl.py \
        plugins/neuralimage/tests/test_train_step_profile.py
git commit -m "feat(neuralimage): cProfile each training step"
```

---

### Task 4: recognition single-thread per-frame cProfile

**Files:**
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run_single_thread_recognition.py:63-192`
- Test: `plugins/neuralimage/tests/test_recognition_single_thread_profile.py`

**Interfaces:**
- Consumes (Task 2): `cprofile_region`, `set_profile_base_dir`.
- No new produced symbols (the wrap is inline).

- [ ] **Step 1: Write the failing test**

Create `plugins/neuralimage/tests/test_recognition_single_thread_profile.py`. It runs `run_single_thread_recognition` with **one** source frame and all heavy dependencies patched, asserting a per-frame report is written when enabled and none when disabled.

```python
from __future__ import annotations

import torch
from PIL import Image
from multiprocessing import Event

import pytest


def _one_frame(tmp_path) -> str:
    p = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (10, 200, 10)).save(p)
    return str(p)


def _patched(monkeypatch, *, enabled: bool, calls: list):
    from neuralimage.model.NeuralNetwork import recognition_pipeline as rp
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts import run_single_thread_recognition as rst
    from neuralimage.infrastructure import profiling
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    # cut/predict/sew are re-imported from recognition_pipeline inside the function body
    monkeypatch.setattr(rp, "cut_image_prepare", lambda *a, **k: {"cutted_image": "x", "name": "a.png", "source_path": "a.png"})
    monkeypatch.setattr(rp, "gpu_predict", lambda *a, **k: {"predicted_image": "y", "name": "a.png"})
    monkeypatch.setattr(rp, "sew", lambda *a, **k: "out.png")
    # preview/memory helpers are module-level names in run_single_thread_recognition
    monkeypatch.setattr(rst, "_publish_recognition_preview", lambda **k: None)
    monkeypatch.setattr(rst, "_publish_memory_metrics", lambda **k: None)


class _FakeModel:
    def eval(self):
        return self
    def to(self, *a, **k):
        return self


def test_single_thread_writes_frame_report_when_enabled(monkeypatch, tmp_path):
    from neuralimage.model.NeuralNetwork.recognition_pipeline import run_single_thread_recognition
    calls = []
    _patched(monkeypatch, enabled=True, calls=calls)
    run_single_thread_recognition(
        source_files=[tmp_path / "a.png"],
        result_folder=tmp_path,
        part_size=(8, 8),
        overlap=0,
        batch_size=1,
        colors=3,
        model=_FakeModel(),
        device=torch.device("cpu"),
        stop_event=Event(),
        publish=lambda *a, **k: None,
        collect_memory_metrics=lambda: None,
    )
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition frame]" in output
    assert "name=a.png" in output


def test_single_thread_no_report_when_disabled(monkeypatch, tmp_path):
    from neuralimage.model.NeuralNetwork.recognition_pipeline import run_single_thread_recognition
    calls = []
    _patched(monkeypatch, enabled=False, calls=calls)
    run_single_thread_recognition(
        source_files=[tmp_path / "a.png"],
        result_folder=tmp_path,
        part_size=(8, 8),
        overlap=0,
        batch_size=1,
        colors=3,
        model=_FakeModel(),
        device=torch.device("cpu"),
        stop_event=Event(),
        publish=lambda *a, **k: None,
        collect_memory_metrics=lambda: None,
    )
    assert calls == []
```

> Implementer note: `run_single_thread_recognition` **re-imports** `cut_image_prepare`, `gpu_predict`, `sew` from `neuralimage.model.NeuralNetwork.recognition_pipeline` inside the function body, so patch those on `recognition_pipeline` (`rp`). But `_publish_recognition_preview` and `_publish_memory_metrics` are **module-level** names in `run_single_thread_recognition` (imported at the top of that file), so patch them on the `run_single_thread_recognition` module (`rst`) — as `_patched` does.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest plugins/neuralimage/tests/test_recognition_single_thread_profile.py -v`
Expected: FAIL — `test_single_thread_writes_frame_report_when_enabled` gets `calls == []` (no profiling yet).

- [ ] **Step 3: Implement the wrap in `run_single_thread_recognition.py`**

In `run_single_thread_recognition.py`:
- Import `cprofile_region` and `set_profile_base_dir` from `neuralimage.infrastructure.profiling`.
- Near the top (after `model.eval(); model.to(device); publish('logging', 'Recognition started in single-thread mode.')`), add `set_profile_base_dir(Path(result_folder))`.
- In the `for index, image_path in enumerate(source_files, start=1):` loop, after the `if stop_event.is_set(): break` guard, wrap the entire per-frame body (from `prepared = cut_image_prepare(...)` through the trailing `_publish_memory_metrics(...)`) in:

```python
with cprofile_region("recognition", label="recognition frame", extra=f"frame={index}/{len(source_files)} name={Path(image_path).name}"):
```

This adds one indentation level to the per-frame body (including the compile-fallback block, `sew`, preview, and `reporter.publish_frame`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest plugins/neuralimage/tests/test_recognition_single_thread_profile.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run_single_thread_recognition.py \
        plugins/neuralimage/tests/test_recognition_single_thread_profile.py
git commit -m "feat(neuralimage): cProfile each frame in single-thread recognition"
```

---

### Task 5: recognition multiprocessing per-item cProfile (cut / predict / sew)

**Files:**
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run.py:96-175` (3 `mp.Process` arg tuples)
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run_cut_worker.py:26-124`
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/cut_image_process.py:48-72`
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/imgpredict.py:78-109`
- Modify: `plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/imgsew.py:39-65`
- Test: `plugins/neuralimage/tests/test_recognition_worker_profile.py`

**Interfaces:**
- Consumes (Task 2): `cprofile_region`, `set_profile_base_dir`.
- Produces:
  - Each of `_run_cut_worker`, `_run_predict_worker`, `_run_sew_worker` gains a trailing keyword param `profile_base_dir: Path | None = None`; when not `None`, each calls `set_profile_base_dir(Path(profile_base_dir))` before invoking the inner worker function.
  - `run.py` passes `self._config.workload.result_folder` as that trailing arg for all three worker types.
  - The inner workers `cut_image_process`, `imgpredict`, `imgsew` wrap their per-item body in `cprofile_region("recognition", label="recognition cut|predict|sew", extra=...)`.

- [ ] **Step 1: Write the failing tests**

Create `plugins/neuralimage/tests/test_recognition_worker_profile.py`. Each worker's inner function is driven with a **plain `queue.Queue`** (one item + a stop token) and patched heavy deps; assert a per-item report when enabled, none when disabled. Plus a test that `run.py` passes `result_folder` to the workers.

```python
from __future__ import annotations

import queue as queue_module
from pathlib import Path

import pytest


def _run_imgsew(monkeypatch, tmp_path, *, enabled: bool):
    # imgsew re-imports `sew` from recognition_pipeline INSIDE the function body,
    # so patch it there (the re-import source), not on the worker module.
    import multiprocessing
    import neuralimage.model.NeuralNetwork.recognition_pipeline as rp
    import neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgsew as mod
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(rp, "sew", lambda *a, **k: str(tmp_path / "o.png"))
    ev = multiprocessing.Event()  # never set
    q = queue_module.Queue()
    q.put({"name": "a.png", "predicted_image": "y", "confidence_image": "c"})
    q.put("__STOP__")
    mod.imgsew(tmp_path, q, queue_module.Queue(), 95, ev, stop_token="__STOP__")
    return calls


def test_imgsew_writes_sew_report_when_enabled(monkeypatch, tmp_path):
    calls = _run_imgsew(monkeypatch, tmp_path, enabled=True)
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition sew]" in output
    assert "name=a.png" in output


def test_imgsew_no_report_when_disabled(monkeypatch, tmp_path):
    calls = _run_imgsew(monkeypatch, tmp_path, enabled=False)
    assert calls == []


def test_run_passes_result_folder_to_workers(monkeypatch, tmp_path):
    import multiprocessing as mp
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.run import RunMixin
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import (
        RecognitionWorkload, WorkerCounts, RuntimeWorkerConfig,
    )
    import torch
    captured = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            captured["args"] = kwargs.get("args")
        def start(self):
            return None
    monkeypatch.setattr(mp, "Process", FakeProcess)

    workload = RecognitionWorkload(
        source_files=[], result_folder=tmp_path, part_size=(8, 8), overlap=0,
        batch_size=1, colors=3, jpeg_quality=95, binarize_output=True, threshold=None,
        postprocess_enabled=False, postprocess_kernel_size=0, recognition_tta_enabled=False,
        confidence_tta_enabled=False, confidence_save_mode="off", devices=[], model_source="m",
    )
    runner = RunMixin(
        config=RuntimeWorkerConfig(workload=workload, worker_counts=WorkerCounts(cut=1, predict=1, sew=1), stop_token="__STOP__"),
        stop_event=mp.Event(), publish=lambda *a, **k: None,
    )
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import PipelineQueues
    def run_start(which):
        captured.clear()
        q = PipelineQueues(cut=mp.Queue(), predict=mp.Queue(), sew=mp.Queue(), sewed=mp.Queue(), errors=mp.Queue())
        getattr(runner, which)(q)
        return captured["args"]
    for which in ("_start_cut_workers", "_start_predict_workers", "_start_sew_workers"):
        args = run_start(which)
        assert tmp_path in list(args), f"{which} must receive result_folder"
```

> Implementer notes:
> - **Patch targets:** `imgsew`, `cut_image_process`, and `imgpredict` each **re-import their heavy dependency from `neuralimage.model.NeuralNetwork.recognition_pipeline` inside the function body** (e.g. `from ...recognition_pipeline import _try_get_queue_item, sew`). So patch the dependency on **`neuralimage.model.NeuralNetwork.recognition_pipeline`** (the re-import source), not on the worker module. Exception: `imgpredict.load_model_artifact` is a module-level import, so patch it on the `imgpredict` module.
> - `_try_get_queue_item` works with a plain `queue.Queue` (no `_reader` attr → falls back to `get(timeout)`). Use `multiprocessing.Event()` for `stop_event` (never set). `_restore_payload_array_from_multiprocessing` copies non-tensor values as-is, so plain-string payload values are safe.
> - Add an `imgpredict` test analogous to the `imgsew` one: patch `recognition_pipeline.gpu_predict` (return `{"predicted_image": "y", "name": "a.png"}`) and `imgpredict.load_model_artifact` (return a fake model with `eval()`/`to()`), feed an item `{"name": "a.png", "cutted_image": "x"}`, pass `torch.device("cpu")` as `gpu`, and assert `[neuralimage recognition predict]` + `name=a.png` (enabled) / no report (disabled).
> - Add a `cut_image_process` test: patch `recognition_pipeline.cut_image_prepare` (return `{"name": "a.png", "cutted_image": "x"}`), feed an item that is a `Path` (the cut queue carries image paths), and assert `[neuralimage recognition cut]` (enabled) / no report (disabled).
> - The `run_passes_result_folder` test relies on the new trailing arg being appended to each `mp.Process(..., args=(...))`; it fails until Step 3/4 are done.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest plugins/neuralimage/tests/test_recognition_worker_profile.py -v`
Expected: FAIL — the report assertions get empty `calls` (workers not wrapped yet) and `test_run_passes_result_folder_to_workers` fails (no `result_folder` in args).

- [ ] **Step 3: Wrap the per-item body in the three inner workers**

In each of the three inner worker functions, import `cprofile_region` from `neuralimage.infrastructure.profiling` and wrap the per-item body (the block between the `stop_token` check and the queue `put`) in a `with cprofile_region(...)`:
- `cut_image_process.py` — wrap from `image_payload = cut_image_prepare(...)` through `cutted_queue.put(image_payload)`; `label="recognition cut"`, `extra=f"name={Path(image_path).name}"`.
- `imgpredict.py` — wrap from `restored = _restore_payload_array_from_multiprocessing(item, 'cutted_image')` through `predicted_queue.put(predicted)`; `label="recognition predict"`, `extra=f"name={restored.get('name', '')}"`.
- `imgsew.py` — wrap from `restored_item = _restore_payload_array_from_multiprocessing(item, 'predicted_image')` through the `sewed_queue.put({...})`; `label="recognition sew"`, `extra=f"name={restored_item.get('name', '')}"`.

- [ ] **Step 4: Plumb `profile_base_dir` through the worker entry points**

- In `run_cut_worker.py`: import `set_profile_base_dir` from `neuralimage.infrastructure.profiling`. Add `profile_base_dir: Path | None = None` as the **last** parameter of `_run_cut_worker`, `_run_predict_worker`, and `_run_sew_worker`. At the top of each (before the inner `cut_image_process` / `imgpredict` / `imgsew` call), add:

```python
if profile_base_dir is not None:
    set_profile_base_dir(Path(profile_base_dir))
```

- In `run.py`: append `self._config.workload.result_folder` as the **last** element of the `args` tuple in `_start_cut_workers`, `_start_predict_workers`, and `_start_sew_workers` (matching the new trailing param).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest plugins/neuralimage/tests/test_recognition_worker_profile.py -v`
Expected: PASS (all tests, including the cut/predict/sew report tests and the `result_folder` plumbing test).

- [ ] **Step 6: Commit**

```bash
git add plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/run_cut_worker.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/cut_image_process.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/imgpredict.py \
        plugins/neuralimage/src/neuralimage/model/NeuralNetwork/recognition_pipeline_parts/imgsew.py \
        plugins/neuralimage/tests/test_recognition_worker_profile.py
git commit -m "feat(neuralimage): cProfile each item in multiprocessing recognition workers"
```

---

## Final Validation (after all tasks)

- Run the whole plugin suite with profiling **disabled** (default):
  Run: `pytest plugins/neuralimage/tests -q`
  Expected: PASS — confirms no behavior change with profiling off.
- Run the four new profiling test files together:
  Run: `pytest plugins/neuralimage/tests/test_neuralimage_runtime_config.py plugins/neuralimage/tests/test_neuralimage_profiling.py plugins/neuralimage/tests/test_train_step_profile.py plugins/neuralimage/tests/test_recognition_single_thread_profile.py plugins/neuralimage/tests/test_recognition_worker_profile.py -v`
  Expected: PASS.
- Manual smoke (not automated): set `NEURALIMAGE_PROFILE=1` in the environment, run a short training and a small recognition, and confirm the log files appear (next to the model for training, in `result_folder` for recognition) with per-step / per-frame / per-item lines and pstats top-N, and that the wall-clock per step is visible as a trend.

## Notes / deferred

- **End-of-epoch tail** (`_handle_validation`, `_checkpoint_manager.save_best` / `_save_epoch_artifacts`, `_save_metric_charts`) is **not** profiled in this plan, per the user ("хвост пока не профилируем"). If the per-batch wall-clock trend comes out flat but the run still slows down, add a `train epoch-end` profiled unit in `run_single_training_epoch.py` (same `training` switch, `cprofile_region("training", label="train epoch-end", ...)`) as follow-up.
- cProfile captures in-process Python functions only: DataLoader prefetch (separate subprocesses) and CUDA kernels appear in wall-clock, not in the per-function breakdown.
