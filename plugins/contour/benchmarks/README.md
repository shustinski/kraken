# Metal segmentation benchmark

Run from the repository root:

```powershell
$env:PYTHONPATH = "plugins/contour/src"
.\.venv\Scripts\python.exe plugins/contour/scripts/benchmark_metal_segmentation.py
```

Use `--output <path>` to retain the full per-case JSON report. The checked-in
`metal_segmentation_before.json` and `metal_segmentation_after.json` reports use
the same deterministic scenes and random seeds.

## Method

The 16 ground-truth scenes cover large and 4 px conductors, 1–3 px elevated
gaps, dark conductor centres, bright rims, weak contrast, one/multiple FOV
borders, parallel bundles, a junction, noise, illumination drift, blur, and a
combined stress case. Rendering adds smooth illumination, Gaussian read noise,
scaled Poisson shot noise, and Gaussian edge blur.

Metrics are computed on the raw segmentation mask, before contour geometry
filters:

- pixel IoU, precision, and recall;
- boundary F1 with a two-pixel tolerance;
- connected-component count;
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
| Gradient Watershed | 0.660 | 0.799 | 0.805 | 0.887 | 5 / 5 | 1 / 1 |
| Random Walker | 0.697 | 0.781 | 0.841 | 0.878 | 5 / 5 | 2 / 1 |
| Graph Cut | 0.619 | 0.721 | 0.291 | 0.827 | 5 / 5 | 1 / 1 |
| Reconstruction | 0.661 | 0.721 | 0.806 | 0.834 | 5 / 5 | 1 / 1 |

On the two checked-in 2000×2000 SEM+CIF pairs, Gradient Watershed raw-mask IoU
changed from `0.566` to `0.786` (MSP430) and from `0.611` to `0.885` (OCTA1).
The downstream golden CIF tests remain the authority for polygon output.

The topology totals deliberately remain visible: the benchmark's blurred,
elevated 1–3 px gaps have no resolved intensity valley, so the current classical
evidence cannot justify a split. A tested multi-scale valley detector did not
reduce those merges and introduced six additional false splits; it was rejected.

Before marker-preserving downsampling, a one-pixel feature vanished in 8/10
sampling phases and a two-pixel feature in 4/10. The iterative solver path now
uses conservative marker reduction and reapplies native-resolution hard markers
after upsampling. This preserves confirmed narrow evidence but cannot invent a
separator absent from the source image.
