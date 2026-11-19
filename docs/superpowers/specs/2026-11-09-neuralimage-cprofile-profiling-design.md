# NeuralImage cProfile Profiling — Design

Date: 2026-11-09
Status: approved (2026-11-09)

## Intent / Why

Training and frame recognition in the `neuralimage` plugin **progressively get
slower over time** ("постепенно становится всё медленнее и медленнее"). The goal
is to find what is causing it.

The user chose:
- **cProfile** (exhaustive Python function coverage; profiling overhead is
  explicitly *not* a concern).
- **Enabled exactly like the `contour` plugin**: env variables + an ini file with
  defaults + a central profiling module (not a UI checkbox, not a bare code
  constant).
- Scope:
  - **Every training step (batch)** — all operations in the step, no function missed.
  - **Every recognized frame** — all operations (cut / predict / sew), no function missed.

The output must make the progressive slowdown visible: a **per-step / per-frame
wall-clock time series** (to see the trend) plus a **per-function cumulative-time
breakdown (top-N)** so a growing function/region can be named.

## Goals

- cProfile-based and **exhaustive**: capture every Python function executed in the
  process during each training step / recognized frame (that is what "no function
  missed" means — global capture, not hand-picked regions).
- **Per training batch** and **per recognized frame**; in multiprocessing
  recognition, **per worker-item** in each of the cut / predict / sew workers.
- Enable mechanism **exactly like contour**: `NEURALIMAGE_PROFILE*` env vars + a
  `[profiling]` section in `neuralimage.ini` + a central `profiling.py` module.
- **Off by default**; zero (or negligible) overhead when disabled.
- Output to a **discoverable rotating log** (like contour) and, where a message bus
  exists (training), a one-line "profiling enabled, log=<path>" notice.
- Reuse contour's `try_enable_profiler` / `try_disable_profiler` conflict handling
  (cProfile allows only one active profiler per thread).

## Non-goals / Anti-goals

- **Not** a UI toggle (the user wants contour-style env + ini).
- **Not** GPU-kernel-level tracing (that is `torch.profiler`; cProfile is Python
  level). CUDA time shows up in wall-clock, not per-kernel.
- **Not** profiling DataLoader worker subprocesses (separate OS processes; their
  time appears in the wall-clock total only, not in the function breakdown).
- **Not** changing training/recognition behavior. Profiling is observation-only:
  zero behavior change when disabled; must not alter results when enabled.
- **Not** a real-time dashboard.
- **Not** profiling the end-of-epoch tail of `_run_single_training_epoch`
  (`_handle_validation`, `_checkpoint_manager.save_best` / `_save_epoch_artifacts`,
  `_save_metric_charts`). Explicitly deferred per the user ("хвост пока не
  профилируем"). These run once per epoch, *between* profiled batches, and some grow
  over time (chart re-render from accumulated data, validation export cache). If the
  per-batch wall-clock trend comes out flat but the overall run still slows down,
  add a `train epoch-end` profiled unit (same `training` switch) as follow-up.

## Constraints

- **cProfile is one active profiler per thread.** Must use
  `try_enable_profiler` / `try_disable_profiler` and handle the
  "Another profiling tool is already active" conflict by falling back to
  wall-clock-only for that unit.
- **Recognition runs in multiple `mp.Process` workers** (cut / predict / sew). Each
  is a separate process → each can hold its own cProfile on its own thread. Env
  vars are **inherited by child processes**, which is exactly why the env + ini
  mechanism works cleanly here (no config plumbing into workers needed).
- **The training loop runs in `mp.Process` (`TrainerProcess`), single main thread**
  → one cProfile instance.
- An **existing** `torch.profiler`-based training profiler already exists
  (`TRAINING_PROFILER_ENABLED`, `start_training_profiler.py`). It is a separate
  feature and stays untouched. It can coexist with the new cProfile profiler
  (independent profilers, no conflict).
- Log writes must be **safe across multiprocessing workers** (pid-suffixed file
  names, like contour).
- The disabled path must be **zero-overhead**: a single boolean check, no
  profiler allocation, no I/O.

## Approaches considered

1. **torch-level profiler (extend the existing one).** Captures CUDA kernels +
   CPU ops, lighter, but it is not "every Python function" and recognition has
   none at all. *Rejected* — the user wants exhaustive Python function coverage.
2. **cProfile with manual region instrumentation.** *Rejected* — "no function
   missed" means not hand-picking regions; cProfile's global capture is the point.
3. **cProfile, exactly like contour** (env + ini central module + per-unit cProfile
   + rotating log + pstats top-N). ***Chosen*** — matches "точно как в contour",
   is exhaustive, and works across the recognition multiprocessing workers via env
   inheritance.

## Design

### 1. New module: `neuralimage/infrastructure/profiling.py`

Mirror of `contour/src/contour/infrastructure/profiling.py`. Provides:

- `PROFILE_ENV_TRUE` / `PROFILE_ENV_FALSE` sets.
- `profiling_log_path(base_dir: Path) -> Path` — resolves the log path for a base
  directory: file name from ini `[profiling] log_file` (default `profiling.log`);
  an absolute `log_file` is used as-is.
- `_new_profile_log_path`, `_profile_log_files`, `_prune_profile_logs` — rotating,
  newest 500 retained (contour behavior).
- `write_profile_report(*messages)` — appends to a rotating UTF-8 log; one
  module-level handler per process; file name is pid + timestamp suffixed.
- `set_profile_base_dir(path: Path)` — sets the process's base directory used by
  `profiling_log_path`; called once at training start and at each recognition
  worker start. *(Builder: finalize exact plumbing.)*
- `try_enable_profiler(profiler) -> bool` and `try_disable_profiler(profiler)` —
  handle the "Another profiling tool is already active" conflict.
- `profiling_enabled(kind, default, legacy_env)` — resolution order:
  env `NEURALIMAGE_PROFILE_<KIND>` → master env
  (`NEURALIMAGE_PROFILE` / `NEURALIMAGE_PROFILING` / `NEURALIMAGE_PROFILE_ALL`) →
  ini `[profiling] <kind>` → ini `[profiling] enabled` → `default`.
- `profiling_top_lines(kind, default)` — resolution order:
  env `NEURALIMAGE_PROFILE_<KIND>_TOP` → `NEURALIMAGE_PROFILE_TOP` →
  ini `<kind>_top_lines` → ini `top_lines` → `default`.
- Kinds + accessors:
  - `training_profiling_enabled()`, `training_top_lines()` (default 40)
  - `recognition_profiling_enabled()`, `recognition_top_lines()` (default 40)
- `reset_profile_output()` for tests.

### 2. New module: `neuralimage/infrastructure/runtime_config.py`

Mirror of `contour/src/contour/infrastructure/runtime_config.py`, adapted to
neuralimage/kraken:

- `CONFIG_ENV = "NEURALIMAGE_CONFIG"`, `CONFIG_FILENAME = "neuralimage.ini"`.
- `bundled_config_path()` → `neuralimage/resources/neuralimage.ini`.
- `runtime_config_path()` → `NEURALIMAGE_CONFIG` override, else bundled.
  (Contour copies the bundled ini to a per-user dir on first run for frozen apps;
  replicate that only if kraken has a standard user-config directory, otherwise
  read the bundled ini directly.) *(Builder: confirm the kraken app-dir /
  user-config convention; fall back to bundled read-only.)*
- `load_runtime_config()` (`lru_cache`), `config_string/bool/int/float`,
  `clear_runtime_config_cache()`.

### 3. New file: `neuralimage/resources/neuralimage.ini`

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
```

> Note: contour ships `enabled = true`. I chose `enabled = false` because this is
> an enable-able diagnostic tool with real overhead — off by default, turned on
> while diagnosing. **Flagged as my call**; easy to flip if you want it on by default.

### 4. Training profiler (per-batch cProfile)

**Location:** `model/NeuralNetwork/model_train_and_recognition_parts/run_train_epoch.py`,
the per-batch `while` loop in `_run_train_epoch`.

**Mechanism:** a context manager (new mixin or module helper), e.g.
`train_step_profile(epoch, batch_index, global_batch)`:

- `__enter__`: if `training_profiling_enabled()` → create `cProfile.Profile()`,
  `try_enable_profiler(...)`, record `start = perf_counter()`. Else no-op.
- `__exit__`: if active → `try_disable_profiler`, compute `wall_ms`, build and
  `write_profile_report`:
  - summary line: `[neuralimage train step] epoch=.. batch=.. global=.. wall_ms=..`
  - function table: `pstats.Stats(profiler).sort_stats("cumtime").print_stats(training_top_lines())`

**Wrap the entire per-batch loop body** — from
`data_wait_started_at = time.perf_counter()` (which includes
`batch = next(train_iterator)`, i.e. data wait) through `_prepare_train_batch`,
`_run_train_step`, and the post-processing (`_update_loss_aware_sampling`,
`_publish_train_batch_runtime`, `_record_random_patch_profile_batch`), including the
`continue` paths. `continue`/`break` inside a `with` still run `__exit__`, so
skipped batches are reported too (with their wall time). This guarantees no
operation in a step is missed.

**Base dir:** at training start (e.g. in `_run_impl` or construction), call
`set_profile_base_dir(self._save_path.parent)` so the log lands next to the model
(matches the existing `training_log.txt` convention in `build_distributed_loader.py`).

**Optional stride knob:** `training_step_stride` (env `NEURALIMAGE_PROFILE_TRAINING_STEP_STRIDE`
/ ini), default **1 = every step**, to cap log size on very long runs. The
per-step wall-clock trend is the primary signal; the knob only reduces frequency.
*(Builder: include as a knob, default 1.)*

### 5. Recognition profiler (per-frame cProfile)

Two execution modes.

**(a) Single-thread** — `recognition_pipeline_parts/run_single_thread_recognition.py`.
Wrap each frame's per-frame loop body — `cut_image_prepare → gpu_predict → sew`
(+ the optional compile-fallback re-predict) — in a cProfile context manager gated
by `recognition_profiling_enabled()`. Report:
`[neuralimage recognition frame] frame=i/N name=.. wall_ms=.. <top-N>`.
Base dir = `result_folder`.

**(b) Multiprocessing** — each worker wraps its **per-item** body:

- **cut worker** — `cut_image_process()` loop (`recognition_pipeline_parts/cut_image_process.py`):
  wrap `cut_image_prepare(...) → _store_payload... → cutted_queue.put(...)`.
  Report tagged `recognition cut`, include the source frame name.
- **predict worker** — `imgpredict()` loop (`recognition_pipeline_parts/imgpredict.py`):
  wrap `_restore_payload... → gpu_predict(...) → _store_payload... → predicted_queue.put(...)`.
  Report tagged `recognition predict`.
- **sew worker** — `imgsew()` loop (`recognition_pipeline_parts/imgsew.py`):
  wrap the per-item body. Report tagged `recognition sew`.

Each worker calls `set_profile_base_dir(result_folder)` so all recognition logs
land in one discoverable place. Env vars are inherited by the `mp.Process` workers,
so `recognition_profiling_enabled()` works in each child with no extra config
plumbing. Each child gets its own pid-suffixed log file (contour's
`_new_profile_log_path` already appends `-p<pid>`), so workers do not collide.
*(Builder: the cut worker currently receives `source_root`, not `result_folder`;
pass `result_folder` into it (or derive the log base) so its logs are discoverable.)*

### 6. Report output & discoverability

- Rotating UTF-8 log, pid + timestamp suffixed, pruned to newest 500 (contour behavior).
- **Training** → next to the model: `<artifact_dir>/profiling.log-<ts>-p<pid>`.
- **Recognition** → in `result_folder`.
- Where a message bus exists (training), emit one line at start:
  `profiling enabled, log=<path>` (mirrors the existing profiler's
  `self._bus.put(['logging', ...])`).

## What's left to the builder

- Exact kraken app-dir / user-config resolution for the ini (mirror contour's
  frozen-app copy behavior; fall back to bundled read-only if no standard dir).
- Exact plumbing of `set_profile_base_dir` and the per-context base dirs, including
  passing `result_folder` to the cut worker.
- The precise per-batch / per-item wrap boundaries (must not change behavior; a
  `with` around the existing body is the intended shape).
- The `training_step_stride` knob (default 1).
- **Unit tests mirroring contour's** (`tests/unit/test_*profiler.py`):
  enable/disable resolution (env / ini / master / per-kind), `try_enable` conflict
  handling, report writing to a temp dir, and that the disabled path activates no
  cProfile.
- **No-behavior-change verification:** run the existing training/recognition tests
  with profiling disabled, and a short enabled smoke run on a tiny dataset.

## Validation

- Existing tests pass with profiling **disabled** (zero-overhead path).
- New unit tests for the profiling module (analogous to contour's
  `tests/unit/test_startup_profiler.py` and `test_frame_switch_profiler.py`).
- Manual: enable via `NEURALIMAGE_PROFILE=1` (or set the ini), run a short training
  and a small recognition, confirm the log contains per-step / per-frame lines with
  pstats top-N, and that the wall-clock trend is visible.

## Known cProfile limitations (accepted)

- DataLoader prefetch runs in separate subprocesses → its time is in the
  wall-clock total, not in the per-function breakdown.
- CUDA kernels are not individual Python functions → forward/backward show as the
  top-level torch calls; GPU time is visible via wall-clock, not per-kernel.
