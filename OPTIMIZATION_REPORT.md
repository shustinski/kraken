# Karakal audit, profiling, and optimization report

Original audit date: 2026-08-04; refactoring validation updated: 2026-08-26  
Platform used for measurements: Windows, Python 3.13, 8 logical CPUs. OpenCV was limited to one thread per process worker. The resource sampler was unavailable in this environment, so process CPU and peak RAM fields are exported as `null`; no RAM improvement is claimed.

## Scope and preservation policy

The pre-existing dirty Karakal worktree was treated as the candidate implementation and was not reset. A `git archive HEAD` copy under the system temporary directory was used for clean-source behavior and timing baselines. Root `kraken_manager` code and Karakal's folder manager remain in scope as supported functionality; only the removed Karakal Manager Mode was cleaned up.

No dependency, metric formula, threshold, classification, public plugin API, or result/export schema was intentionally changed by this work. Internal grid cache keys were advanced to `grid_damage_v64_compact_cache` after changing the private pickle representation.

## Current architecture

The application flow is:

1. `KarakalWidget` owns application controls, the modeless profiling dialog, mode stack, and persisted settings.
2. `KarakalPresenter` captures a request/generation token and starts a Qt worker (`FrameIndexWorker`, `AnalyticsWorker`, grid workers, or detail workers).
3. Workers call the focused owning modules under `core`, `core.grid_anomaly`, and the `comparison` package. Process grid workers receive paths plus immutable configuration and return compact analysis dictionaries and `WorkerProfilePacket` aggregates; NumPy image arrays are not sent through the process queue.
4. Comparison artifacts and RAM/disk caches feed `BuildResult`/`FrameRecord` values.
5. `MatrixListWidget` renders a virtualized scene, loads visible tiles asynchronously, and rejects tile results whose generation is stale.
6. Detail views and exporters reuse the same repository/result model.

The repository implementation is now split into focused image I/O, cache, metric, confidence, analytics, detail, and export modules. `core.repository` remains a backwards-compatible re-export facade. The presenter remains intact for a separate UI-focused refactoring stage.

## Manager Mode cleanup

- Removed Manager Mode source modules, pipeline/selector imports, presenter branches, state, controls, translations, and packaging references already present in the WIP.
- Removed the final unused `MGMT_ROLE_*` constants.
- Kept folder-manager and acquisition/confidence/grid features that still have active consumers.
- Removed Manager Mode wording from the README and verified the rebuilt wheel, sdist, `PKG-INFO`, and `SOURCES.txt` contain no Manager Mode, primary-labeling, deleted pipeline, or deleted package entries.
- Added a one-time QSettings migration. It removes only `ui/management_settings` and known Manager-specific fields, changes persisted `manager`/`management` modes to `validation`, preserves unrelated folder/build/detail settings, emits one warning, and does not start analysis.
- Offscreen startup regression verifies exactly two application modes (`validation` and `grid_inspection`), two stack pages, and no hidden Manager action.

The stale ignored `plugins/karakal/build` and `src/karakal.egg-info` directories were moved to a recoverable temporary backup before a clean package rebuild.

## Performance configuration and profiler

`PerformanceConfig` centralizes CPU/I/O workers, batch size, OpenCV threads, byte limits, progress throttling, and profiler settings. Precedence is environment override, QSettings, then validated safe defaults. Existing `KARAKAL_GRID_INSPECTION_WORKERS`, `KARAKAL_GRID_INSPECTION_CHUNK_SIZE`, `KARAKAL_GRID_INSPECTION_OPENCV_THREADS`, and execution-mode variables remain compatible aliases.

Defaults are profiling `off`, trace and process memory collection disabled, UI refresh 300 ms, trace limit 100 frames, history 10 runs, and one OpenCV thread per process worker.

The profiler uses `perf_counter_ns()` and `process_time_ns()`, thread-local nested stacks, inclusive/self time, Welford statistics, deterministic bounded samples, p50/p90/p95/p99, bounded slow/error/cancel information, counters, and worker aggregate merging. The disabled path records nothing. Chrome/Perfetto trace data is emitted separately and is capped.

Instrumented areas include:

- indexing/matching and analytics;
- image read/decode/preprocessing;
- mask, confidence, pairwise, and ensemble stages;
- grid thresholding, contours, feature extraction, reference profile, classification, clustering, cache, worker pool, queue, and IPC aggregate merge;
- analysis/detail/tile cache reads and writes;
- matrix refresh, tile queue preparation, tile result handling, stale/discard/error counts, and detail preparation.

JSON, CSV, Markdown, and trace files use temp-file plus atomic replace. The last ten run groups are retained. Missing RAM/CPU utilization values are explicit `null`, never fabricated.

The modeless Diagnostics panel shows mode, current stage, elapsed time, throughput, ETA, workers, queue depth, cache hit rate, RAM availability, and a sortable calls/total/self/average/median/p95/p99/share/error table.

## Cache and task orchestration

- RAM image, resized-image, detail/preview, comparison, subpixel, and grid-preview caches now have byte ceilings in addition to existing item safety caps.
- Disk analysis, detail, grid, and tile caches are trimmed oldest-first by bytes and optional file count.
- Cache identity includes source path, size, mtime, algorithm/format version, and relevant analysis parameters.
- Pickle writes serialize to a unique temporary file, flush/fsync, verify readability, then atomically replace the destination. Incomplete temporary files are cleaned up.
- Process-pool submission/future failures are logged with context. Per-frame grid failures are grouped, counted, exposed through `analysisErrors`, and no longer disappear silently.
- Pending futures are cancelled during teardown. Existing presenter, detail, grid, and tile generation checks reject stale payloads before state/cache updates.

## Correctness verification

Clean `HEAD` ran 45 original Karakal tests using the root pytest configuration. The refactored candidate runs 96 functional Karakal tests plus 2 performance-harness and 6 shared analysis-protocol tests.

Golden checks:

| Area | Baseline comparison | Result |
|---|---|---|
| Grid, 64 deterministic 384x512 frames | score, defect count, status, and order | Exact, 0 differences |
| Pair comparison (`polygon`, `mixed`, `line_network`) and Ensemble | canonical serialized payload excluding runtime timing telemetry | Exact SHA-256 `9f902817d0ca4b8be1d79294b18d05f8973b86b22bbb0e15e7242da560b8594b` |
| Exported comparison-difference JPEG | file bytes | Exact SHA-256 `8f66b3a479b53ef2d3ccae4a20a9217741e43ce9156da629ebc1a4db0b783f75` |
| Export manifest | semantic fields and relative output | Exact; absolute temporary roots intentionally differ |
| Sequential vs process grid workers | deterministic result digests | Exact |
| Profiling `off` vs `summary` | complete grid result objects | Exact |
| Legacy vs compact grid-cache pickle | complete result objects | Exact |

Regression tests cover empty/missing/corrupt inputs, folder/frame/confidence matching, Pair/Model-vs-Model/Ensemble comparisons, polygon/mask/point/confidence behavior, grid analysis and linked grid layers, detail/export paths, cache corruption/concurrent writes, cancellation, stale request generations, offscreen startup, profiler aggregation/worker merge/trace bounds/export retention, and cache byte eviction.

The comparison policy remains exact for deterministic classes, scores after normal application rounding, ordering, and export bytes. Numeric array assertions retain `rtol=1e-7, atol=1e-9` for float64 and `rtol=1e-5, atol=1e-6` for float32 where exact representation is not appropriate.

## Measurements

All reported scenarios use a warm-up and at least three measurements; the table reports the median. Synthetic inputs are deterministic and live only in ignored build/temporary directories. `--limit` can exercise 1k, 10k, and 40k logical frames without committing binaries.

| Scenario | Clean HEAD | Candidate | Delta | Notes |
|---|---:|---:|---:|---|
| Grid sequential, 64 x 384x512, cache off | 1.069 s | 1.422 s | +32.96% | Regression; no speedup claim |
| Grid process, 256 x 384x512, 4 workers, batch 8, cache off | 1.795 s | 1.931 s | +7.54% | Below 10% warning threshold, but not an improvement |
| Grid warm cache, 128 frames, before compact cache | 0.164 s | 0.239 s | +45.62% | Triggered cache deserialization investigation |
| Grid warm cache, after compact cache | 0.164 s | 0.218 s | +32.68% | Candidate improved, but remains slower than HEAD |

The repository-refactoring verification reran the fixed 64-frame sequential scenario on the current environment at a 1.157 s median with zero failures. Against the pre-refactoring dirty-worktree result of 1.422 s recorded above, this is an 18.6% improvement; it does not cross the plan's 10% regression gate. Because the older clean-HEAD number was collected under Python 3.13, it is retained as historical evidence rather than presented as a same-environment comparison with the Python 3.14+ refactoring run.

The current detailed grid profile identified these highest self-time stages on 64 frames:

| Stage | Self time | Run share |
|---|---:|---:|
| `validation.grid.contours.features` | 730.620 ms | 60.41% |
| `validation.grid.cells.classify` | 187.990 ms | 15.54% |
| `validation.grid.contours.find` | 91.305 ms | 7.55% |

`summary` profiling added 5.76% median runtime over `off` in the same 64-frame scenario while producing identical results.

Small comparison cases (3-run medians) remained within 10% for all cold calculations: Pair polygon +3.38%, Pair line +5.53%, Ensemble polygon -0.41%, Ensemble line (8 models) +9.58%. Warm cache lookups changed by 0.007-0.017 ms in absolute time; their large percentages are timer-resolution dominated and are not treated as meaningful speedups or regressions.

## Accepted and rejected optimizations

Accepted: compact private grid-cache pickle representation.

- Target: pickle deserialization inside warm-cache reads, selected from measured self time.
- Legacy median: 217.561 ms for four passes over 64 results.
- Compact median: 165.736 ms.
- Target-stage improvement: 23.82%.
- Serialized bytes: 921,030 to 883,926 (-4.03%).
- Full-object equality: exact.
- End-to-end current warm scenario: 0.239 s to 0.218 s (-8.89%).
- Peak RAM: unavailable, exported as `null`; the smaller serialized payload is not substituted for a peak-RAM measurement.

Rejected: replacing `np.any` with direct `logical_or.reduce` in contour outline features.

- Micro-loop appeared faster, but the required full-stage median changed from 1,000.015 ms to 1,313.837 ms (+31.38%).
- Results were exact, but the optimization failed the performance gate and was reverted.

The existing WIP vectorization, cache-first partitioning, process batching, and OpenCV thread limiting were retained because they are part of the user's starting point and pass correctness checks. They are not presented as newly proven speedups: the clean-HEAD comparisons above do not satisfy the acceptance threshold.

## Commands and CI gates

```text
uv run --package karakal --extra dev ruff check plugins/karakal/src plugins/karakal/tests plugins/karakal/benchmarks
uv run --package karakal --extra dev pytest plugins/karakal/tests -q
uv run --package karakal --extra dev pytest plugins/karakal/tests -q -m "not performance"
uv run --package karakal --extra dev pytest plugins/karakal/tests/performance -q
uv run --package karakal --extra build python -m build plugins/karakal --outdir plugins/karakal/build/package
```

The dedicated Karakal workflow runs Ruff, functional tests without the performance marker, then the hardware-sensitive performance harness separately. Regression comparison emits a warning above 10% and fails above 25%.

## Remaining risks

- Peak process/worker RAM and sampled CPU were unavailable on this machine. They remain `null`, so the 5% peak-RAM acceptance gate requires a run on a host with the optional resource sampler available.
- 1k/10k/40k logical-frame runs and interactive zoom/scroll/mode-switch scenarios are supported by the harness/instrumentation but were not executed during this audit. No large-scale speedup is claimed.
- The current WIP result objects are larger than clean HEAD because of already-added cell/cluster data. Compact serialization improves that bottleneck but does not erase the warm-cache regression relative to HEAD without changing the result format.
- Contour feature extraction remains the dominant cold-grid bottleneck. Further work should isolate hull/moments/outline costs and must pass the same exact-result and ≥10% stage/≥5% scenario gates.
- The repository-wide `uv run pytest -q` stops during collection with 44 errors outside Karakal: missing optional `torch`, `PIL`, and `django`, plus pre-existing Krona/NeuralImage test-package import-path collisions (`tests.*`, `tools`, and `main`). It also reports 28 optional-dependency skips. Karakal's scoped gates are green; these collection failures are not attributed to this change.
