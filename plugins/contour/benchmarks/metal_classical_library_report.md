# Classical metal-segmentation library: implementation and validation report

Date: 2026-08-25

## Scope and implementation

The Contour plugin now exposes six additional, explicitly selected classical
metal-segmentation strategies without changing the existing `Auto`,
`Threshold`, `Watershed`, or `Gradient watershed` behaviour:

- **OWT-UCM**: a Windows-compatible Python port of the supplied Berkeley
  Segmentation Repository `contours2ucm.m` / `ucm_mean_pb.cpp` path. It uses
  eight SEM-derived oriented boundary channels, the finest watershed
  partition, BSR dynamic mean-boundary agglomeration and a user-controlled
  ultrametric cut.
- **Graph Multi-Separator**: the supplied upstream C++ GSS/GSG algorithms,
  compiled as a pybind11 extension. SEM evidence is converted to the upstream
  vertex/interaction costs; GSG is the Standard solver and GSS remains an
  explicit alternative. Separator nodes with strong conductor-core evidence
  are projected back into material regions before polygon extraction; only
  weak separators may merge two already-classified metal regions.
- **GASP**: signed attraction/repulsion graph with selectable average,
  mutex-absolute-maximum, or sum linkage.
- **Mutex Watershed**: weight-ordered attractive edges plus explicit mutex
  constraints.
- **Multicut**: signed log-odds or linear edge costs with deterministic greedy
  additive contraction.
- **Lifted Multicut**: Multicut plus bounded sparse non-local relations; no
  dense all-pairs construction.

`IC-SEM expert` is registered as a hidden extension point only. Selecting it
programmatically raises a clear unavailable-backend error; it never silently
falls back to another algorithm.

All strategies share cached full-resolution SEM structural features and a
deterministic region-material classifier. They return one canonical contract:
binary material mask, per-pixel labels, optional boundary/separator map, debug
maps, timings, and metadata. Detector contour extraction is label-aware, so
touching labelled regions remain distinct. No neural network, image resize,
random solver, or hidden strategy fallback is used.

The UI is generated from a declarative strategy registry. Each new method has
nested, serializable parameters, bounds, Basic/Advanced grouping, tooltips,
and a Standard preset. Selecting a new method opens its parameter panel.
Legacy flat configuration and snapshots are migrated into the nested model on
read, while existing strategy identifiers retain their old behaviour.

The exact relevant BSR sources and licence are vendored for auditability. The
Multi-Separator include tree and Python binding source are vendored from
commit `437c651ddf1452452cca4cbc3c0eed2065308486`; only MSVC compatibility,
GUI-console suppression, GIL release and module namespace patches were made.
`pybind11>=3.1` is now a build-only dependency. BSR is AGPL-3.0-or-later, so
the Contour package metadata and distribution obligations were updated
accordingly. The supplied Multi-Separator checkout has no licence file; its
use in this private project was explicitly authorised by the project owner.

## Benchmark protocol

- Source: bundled `plugins/contour/tests/test_metal/images` and matching CIF
  annotations.
- Ground truth: outer polygons minus polygons marked as holes.
- Evaluation: UI-output stage with the existing 50-pixel border crop.
- Hard set: `0175`, `0580`, `3242`.
- Full set: all 23 paired frames (21 positive, 2 empty).
- Metrics: pixel IoU, boundary F1, component F1, false merges/splits, missed and
  false components, exact-topology frames, and wall/stage timings.

## Hard-set results

| Strategy | IoU | Boundary F1 | Component F1 | False merges / splits | Mean time, ms |
|---|---:|---:|---:|---:|---:|
| Existing Auto | 0.884 | 0.922 | 0.984 | 18 / 8 | 11020 |
| OWT-UCM | 0.615 | 0.440 | 0.739 | 319 / 19 | 7405 |
| Graph Multi-Separator | 0.809 | 0.874 | 0.879 | 8 / 40 | 40452 |
| GASP | 0.272 | 0.417 | 0.675 | 56 / 11 | 12045 |
| Mutex Watershed | 0.279 | 0.420 | 0.670 | 57 / 11 | 4725 |
| Multicut | 0.297 | 0.436 | 0.684 | 64 / 13 | 4901 |
| Lifted Multicut | 0.290 | 0.429 | 0.691 | 65 / 15 | 6875 |

The BSR OWT-UCM port is strong on `0175` (IoU 0.958), but degrades on `0580`
(0.474) and `3242` (0.412). Corrected Multi-Separator reaches 0.983, 0.717 and
0.728 on the same frames. Its hard-set IoU rose from 0.320 to 0.809 and missed
components fell from 711 to 308; the remaining loss is concentrated in the
densest `3242` topology rather than a global threshold mismatch.

## Full-set results

| Strategy | IoU | Boundary F1 | Component F1 | False components | Missed components | Merges / splits | Exact topology (all / positive) | Mean time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing Auto | 0.934 | 0.966 | 0.990 | 44 | 35 | 27 / 14 | 15 / 13 | 8111 |
| OWT-UCM | 0.521 | 0.523 | 0.735 | 17 | 1420 | 865 / 49 | 2 / 0 | 6032 |
| Graph Multi-Separator | 0.784 | 0.888 | 0.909 | 5 | 418 | 9 / 144 | 11 / 9 | 33025 |
| GASP | 0.537 | 0.631 | 0.818 | 34 | 1005 | 163 / 43 | 2 / 0 | 11845 |
| Mutex Watershed | 0.540 | 0.634 | 0.820 | 38 | 995 | 174 / 45 | 2 / 0 | 4859 |
| Multicut | 0.541 | 0.635 | 0.821 | 37 | 993 | 179 / 46 | 2 / 0 | 4060 |
| Lifted Multicut | 0.550 | 0.640 | 0.827 | 36 | 973 | 184 / 50 | 2 / 0 | 5854 |

All seven strategies produce zero false-metal area on both empty frames. The
new classical backends are fully integrated and deterministic, but their
current generic Standard presets are **not replacements for Auto** on this
dataset. Corrected Multi-Separator is now the strongest new method on both the
hard and full sets, while Lifted Multicut remains the strongest signed-graph
variant. Multi-Separator still trails Auto on dense scenes, but its full-set
misses fell from 1987 to 418 while retaining 0.935 mean precision.

Frame `0004` exposed the adapter defect directly. The native partition and
material classifier retained most conductor pixels, but the adapter converted
every upstream separator node to background. That left open or sub-4-pixel
instance contours, so the shared polygon extractor accepted only 25 of 144
cropped conductors (IoU 0.113). Material-aware separator projection restores
all 144 with no false, missed, merged or split components and IoU 0.968; the
shared width and geometry filters remain unchanged.

## Sensitivity on the hard set

Three values were evaluated for three important parameters of every new
strategy (54 configurations, 162 frame runs). The table shows the value with
the highest mean positive-frame IoU; it is diagnostic, not an automatically
applied preset change.

| Strategy | Parameter | Best tested value | IoU | Boundary F1 | Merges / splits |
|---|---|---:|---:|---:|---:|
| OWT-UCM | hierarchy level | 0.20 | 0.615 | 0.440 | 319 / 19 |
| OWT-UCM | minimum contour strength | 0.24 | 0.714 | 0.631 | 408 / 7 |
| OWT-UCM | watershed minima suppression | 0.15 | 0.620 | 0.375 | 221 / 15 |
| Graph Multi-Separator | projection core evidence | 0.25 | 0.809 | 0.874 | 8 / 40 |
| Graph Multi-Separator | projection core margin | 0.25 | 0.809 | 0.874 | 8 / 40 |
| Graph Multi-Separator | metal-merge separator ceiling | 0.70 | 0.809 | 0.874 | 8 / 40 |
| GASP | minimum merge affinity | 0.20 | 0.280 | 0.415 | 56 / 10 |
| GASP | maximum repulsive conflict | 0.70 | 0.286 | 0.433 | 60 / 13 |
| GASP | linkage criterion | sum | 0.285 | 0.422 | 60 / 12 |
| Mutex Watershed | attractive scale | 0.60 | 0.280 | 0.419 | 64 / 12 |
| Mutex Watershed | mutex scale | 1.00 | 0.279 | 0.420 | 57 / 11 |
| Mutex Watershed | minimum mutex confidence | 0.70 | 0.280 | 0.424 | 62 / 7 |
| Multicut | affinity bias | 0.35 | 0.536 | 0.469 | 63 / 33 |
| Multicut | repulsion scale | 1.00 | 0.297 | 0.436 | 64 / 13 |
| Multicut | atomic-region scale | 24 | 0.331 | 0.462 | 116 / 11 |
| Lifted Multicut | maximum lifted distance | 12 | 0.299 | 0.441 | 65 / 15 |
| Lifted Multicut | lifted repulsion weight | 0.35 | 0.290 | 0.429 | 65 / 15 |
| Lifted Multicut | lifted confidence threshold | 0.45 | 0.318 | 0.450 | 67 / 15 |

Important observations:

- OWT-UCM has a non-monotonic hierarchy response: levels 0.08, 0.20 and 0.45
  yield IoU 0.311, 0.615 and 0.582. Minimum contour strength is more decisive;
  0.24 reaches 0.714 on the hard set.
- Multi-Separator projection is stable for minimum core evidence 0.15--0.25
  (both IoU 0.809); 0.40 is too strict (0.784). A core-over-substrate margin of
  0.25 is best among 0.00, 0.10 and 0.25. A 0.70 merge ceiling yields higher
  IoU and one fewer merge than 0.85, so the Standard values use the safer end
  of each tested interval.
- Multicut affinity bias is similarly sensitive: 0.35 yields 0.536 versus
  0.297 at the Standard value 0.50.
- Lifted repulsion weight produced identical aggregate masks at the three
  tested values. The parameter participates in lifted edge costs, but no edge
  changed the final greedy decision on these three frames; distance and
  confidence threshold did change the output.

These results deliberately do not promote the per-parameter maxima into a new
combined preset: a one-at-a-time hard-set sweep is insufficient evidence for
joint tuning or full-set generalization.

## Performance and safety notes

Feature maps and the signed atomic graph are cached by image/config identity.
The GASP run in the combined artifact built the shared signed graph once;
subsequent Mutex Watershed, Multicut, and Lifted Multicut runs reused it. The
final Multi-Separator replacement was benchmarked separately, so its timing
includes its own feature-cache context and should not be used to compare only
solver cost with the original combined run. The signed-graph builder has an
explicit one-million-pixel guard and reports a controlled configuration error
instead of attempting unbounded work.

The upstream Multi-Separator solvers operate at original pixel resolution and
do not resize the SEM image. A single global 2000x2000 GSS solve was not
operationally viable on Windows, and a global GSG run reached roughly 7.4 GB
when combined with shared features. The production adapter therefore exposes
explicit overlapping full-resolution tiles (Standard: 384 px with 16 px
overlap), runs the exact upstream solver in each tile and stitches only tile
cores. Peak memory in the full benchmark stayed below 2 GB. Setting tile size
to zero remains an explicit global-solve option; there is no hidden fallback.

The JSON artifacts contain all per-frame metrics, configurations, debug seed
diagnostics, and stage timings. CSV files provide flat review/export data.

## Conclusion

The requested production-facing library surface is complete: six selectable
classical backends, explicit configuration and serialization, canonical
results, debug/timing data, deterministic execution, per-label contours,
benchmarks, and an IC-SEM extension boundary. Validation also establishes an
important limitation: generic graph costs/classification do not yet match the
dataset-specific existing Auto pipeline. Any future attempt to promote a new
method to a default should first tune on a training split and confirm the
choice on a held-out set, with topology metrics treated as first-class gates.
