# Metal segmentation benchmark

Run from the repository root:

```powershell
$env:PYTHONPATH = "plugins/contour/src"
.\.venv\Scripts\python.exe plugins/contour/scripts/benchmark_metal_segmentation.py
```

Use `--output <path>` to retain the full per-case JSON report. The checked-in
`metal_segmentation_before.json` and `metal_segmentation_after.json` reports use
the same deterministic scenes and random seeds.

Run the 23-frame, full-resolution JPG/CIF benchmark with:

```powershell
$env:PYTHONPATH = "plugins/contour/src"
.\.venv\Scripts\python.exe plugins/contour/scripts/benchmark_metal_segmentation.py `
  --suite real --strategies auto `
  --evaluation-stage ui `
  --output plugins/contour/benchmarks/metal_real_after.json `
  --csv plugins/contour/benchmarks/metal_real_after.csv `
  --diagnostics-dir tmp/contour-metal-after
```

`--skip-presence-check` reproduces the pre-change watershed baseline. The CSV
contains one row per frame/strategy, including foreground, FP/FN, topology,
seed, histogram, gradient, local-contrast, and metal-presence diagnostics.

## Method

The 16 ground-truth scenes cover large and 4 px conductors, 1–3 px elevated
gaps, dark conductor centres, bright rims, weak contrast, one/multiple FOV
borders, parallel bundles, a junction, noise, illumination drift, blur, and a
combined stress case. Rendering adds smooth illumination, Gaussian read noise,
scaled Poisson shot noise, and Gaussian edge blur.

The default `ui` evaluation stage runs the same path as interactive recognition:
SEM preprocessing, segmentation, topology repair, component filters, contour
extraction, source-contrast filtering, and polygon rasterization. Use
`--evaluation-stage segmentation` only for diagnosing the raw segmentation
mask.

The report records both the preprocessing and recovery configurations. The
standard real-frame run uses background subtraction, background scale 0.05,
CLAHE 2.0 / grid 8, low denoising, watershed seed speckle radius 4, and minimum
object-to-local-background source contrast 12.

Metrics include:

- pixel IoU, precision, and recall;
- boundary F1 with a two-pixel tolerance;
- expected/predicted object counts and absolute count error;
- component precision, recall, and F1;
- false-positive predicted objects and missed expected objects;
- false merges (one predicted component materially overlaps multiple reference
  objects);
- false splits (one reference object materially overlaps multiple predicted
  components);
- elapsed solver time.

The resolution probe independently measures nearest-neighbour round-trip loss
for 1–8 px hard markers when a 2000 px frame is reduced to the former 640 px
solver width.

## Result summary

| strategy | mean IoU before | mean IoU after | boundary F1 before | boundary F1 after | merges before/after | splits before/after |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Watershed | 0.660 | 0.847 | 0.805 | 0.905 | 5 / 5 | 1 / 1 |
| Random Walker | 0.697 | 0.828 | 0.841 | 0.897 | 5 / 5 | 2 / 1 |
| Graph Cut | 0.619 | 0.764 | 0.291 | 0.847 | 5 / 5 | 1 / 1 |
| Reconstruction | 0.661 | 0.765 | 0.806 | 0.853 | 5 / 5 | 1 / 1 |

## Current real JPG/CIF UI-path result

The real suite has 21 positive frames and two empty frames. Current metrics are
for automatic topology control and the final UI polygons; macro metrics are
averaged over positive frames. Reference rasterization preserves independent
conductors inside another conductor's hole instead of clearing them globally.

| metric | current |
|---|---:|
| Macro IoU | 0.856 |
| Median IoU | 0.882 |
| Precision | 0.927 |
| Recall | 0.925 |
| Boundary F1 | 0.927 |
| Component precision | 0.986 |
| Component recall | 0.968 |
| Component F1 | 0.973 |
| False-positive objects | 46 |
| Missed expected objects | 53 |
| Component count absolute error | 101 |
| False merges | 68 |
| False splits | 54 |
| Empty false-metal fraction, mean | 0.0000 |
| Total runtime, mean ms/frame | 4542.1 |

The remaining error is concentrated in topology-heavy frames: `3242` has 32
missed expected objects and eight splits, while `5101` has 25 splits. These
counts stay explicit so a high pixel IoU cannot hide incorrect connectivity.

## Seed heuristic audit

- `narrow_valley_seeds()` assumes a separator is a thin, deep local valley. It
  preserves the resolved grey-seam and 1-4 px gap regressions, but cannot infer
  a gap whose source intensities contain no valley.
- `keep_sandwiched_valley_seeds()` requires brighter support on opposite sides.
  It prevents dark texture inside a trace from becoming a separator; a weak
  seam without two-sided support intentionally remains uncertain.
- `keep_rim_lined_seeds()` assumes a dark substrate component has adjacent
  upper-class rim evidence. Its former unconditional border rule broke dark
  conductors leaving the FOV; border components are now retained without rim
  evidence only when they are geometrically thin.
- `keep_thin_valley_components()` rejects valley blobs wider than a plausible
  separator and protects wide dark conductor centres. Wide ordinary substrate
  is still represented by the primary low-intensity seed path.
- `_cores_covering_fill()` is not present in the current pipeline; the relevant
  core builders are `_local_metal_core_seeds()` and
  `_isolated_weak_core_seeds()`. Their high-confidence thresholds were not
  globally lowered because the real FN audit showed the lost wide interiors
  were predominantly unlabeled, not missing from a permissive global class.

## Diagnosis and rejected experiments

The empty frames had ordinary texture that forced two-class Otsu/seed
generation to produce both classes, so watershed necessarily returned roughly
one third foreground. They lack both a broad robust histogram and a spatially
persistent local-contrast component. `0008` retains the latter evidence despite
its low true metal fraction.

Across `0175`, `0501`, `0673`, and `1170`, 94-98% of false-negative pixels were
unlabeled before watershed. Only 1.7-6.1% were hard groove seeds. The loss
therefore occurred when watershed assigned low-texture uncertain interiors to
background, not in contour extraction or topology filtering.

Rejected recovery variants were kept out of production:

- unrestricted barrier-region growth raised selected target IoUs but reduced
  the 15 protected-frame mean from 0.783 to 0.684-0.743 and flooded `0008`;
- ridge/trench polarity without coherent-line filtering produced only
  0.580-0.633 positive Macro IoU;
- multi-scale closed-boundary masks produced 0.473-0.530 Macro IoU and were
  inconsistent on the four wide-conductor frames;
- filling holes in closed core seeds left Macro IoU near 0.674 and did not
  recover the wide interiors;
- allowing refinement on sub-1% regions created 83 new merges on `3242`; the
  local-ROI size gate removed all of those merges without sacrificing any of
  the four target recoveries.

The topology totals deliberately remain visible: the benchmark's blurred,
elevated 1–3 px gaps have no resolved intensity valley, so the current classical
evidence cannot justify a split. A tested multi-scale valley detector did not
reduce those merges and introduced six additional false splits; it was rejected.

Before marker-preserving downsampling, a one-pixel feature vanished in 8/10
sampling phases and a two-pixel feature in 4/10. The iterative solver path now
uses conservative marker reduction and reapplies native-resolution hard markers
after upsampling. This preserves confirmed narrow evidence but cannot invent a
separator absent from the source image.
