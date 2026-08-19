# NeuralImage: topology-first SEM segmentation

This guide describes the experimental neural-network features in `plugins/neuralimage` for integrated-circuit SEM images. It does not change contour extraction, contour postprocessing, or the binary polygon-mask label format. Auxiliary targets exist only during training; recognition still exports the mask head.

No real SEM dataset was available while this implementation was built. Consequently, this document makes no claim of improved real-world quality and `sem_topology_recommended_v1` remains unavailable until the three-seed acceptance experiment succeeds.

## Contents

1. Configuration and compatibility
2. Multi-target supervision
3. Multi-head models
4. Composite and topology-aware losses
5. Shared SEM preprocessing
6. SEM augmentation
7. Hard-example mining
8. Context branch audit
9. Confidence and uncertainty
10. Active Learning export
11. Full-frame validation
12. Module boundaries
13. Experiment protocol
14. Geometry-aware supervision

## 1. Configuration and compatibility

`SemSegmentationConfig` is versioned and contains `preprocessing`, `augmentation`, `targets`, `heads`, `losses`, `hard_mining`, `context`, `uncertainty`, `active_learning`, `validation`, and `experiment`. Invalid ranges and cross-field dependencies fail before training. Enabled targets and auxiliary heads must match.

Available presets:

- `legacy_v1`: new quality-affecting behavior is disabled and old workflows retain their behavior.
- `sem_topology_experimental_v1`: basic heads, shared preprocessing, SEM v2 augmentation, topology validation and confidence infrastructure are enabled for ablation.
- `sem_topology_recommended_v1`: unavailable until the acceptance gate passes on real data.

Qt and Web expose the complete configuration as typed sections with checkboxes, numeric inputs, choices and path fields. JSON is not shown or edited in the user interface. The internal dictionary contract remains versioned so old workflow snapshots still load and round-trip. Model artifacts contain model/head kwargs plus the preprocessing config and hash; the run manifest contains the complete training configuration.

CLI configurations now use `training_parameters` and `recognition_parameters`. Historical `tranining_parameters` and `recogniton_parameters` spellings remain accepted as read aliases.

```json
{
  "training_parameters": {
    "image_path": "D:/sem/train/images",
    "label_path": "D:/sem/train/masks",
    "sem_segmentation_config": {
      "version": 1,
      "preset": "custom",
      "preprocessing": {
        "percentile_normalization": true,
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "scan_line_suppression": true,
        "scan_axis": "rows"
      },
      "augmentation": {"enabled": true, "plan": "sem_v2"},
      "targets": {
        "basic": {"boundary": true, "skeleton": true, "sdf": true},
        "auxiliary_head_weights": {"boundary": 0.2, "skeleton": 0.3, "sdf": 0.15},
        "distance_boundary_weight": 0.1
      },
      "heads": {"enabled": ["boundary", "skeleton", "sdf"]},
      "losses": {"weighting_strategy": "static", "mask_weight_floor": 0.4},
      "hard_mining": {"mode": "online", "exploration_floor": 0.15},
      "context": {"enabled": false},
      "uncertainty": {"enabled": true, "method": "confidence_head"},
      "active_learning": {"enabled": false},
      "validation": {"enabled": true, "full_frame": true, "boundary_tolerance": 2},
      "experiment": {"seeds": [17, 29, 43], "topology_first": true}
    }
  }
}
```

## 2. Multi-target supervision

Targets are generated from the augmented binary mask after crop selection. The batch contract is `dict[str, Tensor]`; every auxiliary target can have a `<name>__valid` mask. Mapping-aware transfer, filtering, MixUp and validation preserve this contract.

- Boundary: inner morphological boundary.
- Skeleton: `skimage.morphology.skeletonize(..., method="zhang")` performs Zhang-Suen topology-preserving thinning to convergence. A positive `skeleton_iterations` uses the bounded library `thin(..., max_num_iter=N)` compatibility path.
- SDF: signed foreground/background distance clipped by fixed `sdf_clip`.
- Distance transform: foreground distance divided by fixed `distance_clip`, never the patch maximum.
- Thickness: medial-axis diameter propagated through foreground pixels and divided by fixed `thickness_max`.

For foreground set `F`, `SDF(x)=d(x,background)` for `x in F`, and `-d(x,F)` otherwise. Fixed scaling gives the same physical width the same value in every patch.

Local thickness uses `skimage.morphology.medial_axis(..., return_distance=True)` and SciPy's exact EDT feature transform propagates medial diameter to foreground pixels without Python pixel loops. The same feature transform propagates axial orientation from the centerline. Crop-border geometry is invalid so a crossing wire is not taught as an endpoint or corner. Deterministic target sets may be cached by mask bytes plus target-config hash. Thinning and distance transforms are CPU costs proportional to patch area and thinning iterations.

## 3. Multi-head models

`HeadSpec` defines channels, target semantics, activation and loss adapter. The registry includes scalar heads, three-channel T/X/Y junctions, two-channel axial orientation/tangent and two-channel foreground/background topology.

The shared head bundle supports EfficientUNet, EfficientUNetMax and QuasiDualScaleUNet. Unsupported legacy architectures raise a configuration error. New checkpoints reconstruct heads from model kwargs; old checkpoints omit those kwargs and remain loadable.

Auxiliary logits are returned only in training mode. Evaluation and export expose mask and optional confidence, so auxiliary heads add no production inference cost. Training cost is one small projection per enabled head.

## 4. Composite and topology-aware losses

Existing BCE, Dice, IoU, focal, boundary and soft-clDice names and formulas remain available. The trainer delegates boundary and clDice to the modular loss package instead of maintaining divergent copies.

- sparse maps: focal BCE plus soft Dice;
- SDF, distance, thickness and curvature: masked Smooth L1;
- orientation and tangent: axial cosine loss on `(cos 2θ, sin 2θ)`;
- junctions: channel-aware focal loss;
- topology: foreground/background critical maps plus overlap separation.

The old boundary term is unchanged. A separate SDF distance-boundary term weights mask classification error by interface distance and requires SDF supervision.

Static weights remain the default. `homoscedastic_uncertainty` learns log variances `s_i` using `exp(-s_i)L_i + 0.5s_i`. Mask precision has a configurable floor; parameters join optimization, synchronize under DDP and are checkpointed. The inverse-EMA helper is not promoted because it may overweight easy tasks. Soft-clDice depth is configurable and remains an ablation candidate.

## 5. Shared SEM preprocessing

Preprocessing is deterministic and runs once on the full frame before local and context crops. Recognition reads it from the model artifact; an enabled override with a different hash is rejected.

Operations are percentile normalization, CLAHE, illumination correction, background subtraction, scan-line residual suppression and optional bilateral denoising. Ordering and parameters are validated.

The pipeline uses float32 normalized space while preserving uint8, uint16 and float input precision. CLAHE uses 16-bit data. Scan suppression removes high-frequency row/column profile residuals rather than subtracting long conductors. Cost is linear for normalization/profile correction; CLAHE, large kernels and denoising cost more.

## 6. SEM augmentation

`legacy_v1` preserves the prior plan. `sem_v2` consolidates overlapping brightness, blur and noise and adds acquisition-only charging bloom/saturation, row-dependent drift, local focus, Poisson plus read noise, smooth detector gain and contamination/scan defects.

Acquisition effects change only the image. Physical conductor defects remain in the existing label-aware PCB/IC path. Seeded tests verify determinism, bounds and alignment. Local focus is normally the largest CPU cost. Each effect must improve its artifact-specific validation subset before retention.

## 7. Hard-example mining

The existing sampler was extended. Features include thin-wire length, boundary density, small components/vias, junction density, routing density and compactness. Sampling combines normalized geometry and per-sample EMA loss with clipping and an exploration floor.

Every epoch has a deterministic plan. Under DDP the same plan is sharded and per-rank loss observations are gathered before the next epoch. This replaces the old hard-mining DDP downgrade.

Offline mode writes JSONL/CSV with frame, ROI, geometry features, historical loss, score and rank. It does not copy crops by default. Generation is linear plus `O(N log N)` ranking.

## 8. Context branch audit

Pooled context fusion and coordinate-aware cross-attention remain unchanged because no real dataset demonstrated a topology gain. QuasiDualScaleUNet now supports auxiliary heads.

`training.experiments.benchmark_model` records parameters, throughput, peak VRAM and context scale. Compare no context, pooled fusion and current cross-attention with identical splits and budgets. Two-dimensional global-token and ROI-relative positions remain a bounded candidate, not an unvalidated default.

## 9. Confidence and uncertainty

Probability, uncertainty and confidence are distinct; public confidence is `1 - calibrated_uncertainty`.

- confidence head: one-pass output with detached correctness target and configurable weight;
- MC Dropout: only dropout becomes stochastic, normalization stays in eval, and all states restore;
- TTA variance: aligned probabilities produce variance/disagreement rather than averaged confidence logits;
- combined: combines configured uncertainty sources.

Validation reports Brier score, ECE, error-detection AURC and histogram counts. MC/TTA cost scales with passes; the confidence head adds one small projection.

## 10. Active Learning export

Active Learning runs after full-frame stitching. It detects low confidence, high entropy, MC/TTA instability and source disagreement, merges uncertain pixels and ranks non-overlapping component ROIs.

`NeedsAnnotation` contains each source once, probability/confidence/uncertainty maps, ROI reasons, model/config/preprocessing hashes and JSON/CSV manifests. IDs are collision-safe, writes atomic, manifest updates cross-process locked, limits global and completed IDs resumable. No annotation GUI is included. Acceptance requires error/topology enrichment over random ROIs.

## 11. Full-frame validation

Patch probability and confidence maps are stitched before advanced metrics. Binary contour postprocessing is not called. Continuous metrics are averaged across frames; component/topology counts and confidence-bin counts are summed.

Metrics are Dice, IoU, tolerance-aware symmetric Boundary IoU/F1, Hausdorff/HD95, connected-component difference, missed/spurious components, foreground/background deltas, confidence histogram, Brier, ECE and AURC.

A wire break is one GT component overlapping multiple predicted components. A false bridge is one predicted component overlapping multiple GT components. Empty cases, shifts, known gaps, bridges and holes have regression tests. Full-frame computation uses distance transforms and connected-component labeling per frame.

## 12. Module boundaries

```text
src/neuralimage/
  configuration/    Versioned aggregate config and presets
  preprocessing/    Shared deterministic frame preprocessing
  augmentations/     SEM acquisition augmentation
  targets/           Basic/geometry targets, validity and cache
  heads/             HeadSpec registry and shared head bundle
  losses/            Composite terms and target-aware adapters
  metrics/           Segmentation, topology and calibration metrics
  uncertainty/       Confidence, MC Dropout and TTA
  active_learning/   ROI scoring and NeedsAnnotation export
  training/          Hard mining and experiment utilities
```

Existing registry and `model/NeuralNetwork` imports remain compatibility facades. Dataset, UI and legacy model code were not rewritten wholesale. Contour code remains outside these dependencies.

## 13. Experiment and acceptance protocol

Create an immutable manifest and split by source frame; no source may cross splits. Run the production EfficientUNet baseline and cumulative variants with seeds 17, 29 and 43, identical optimizer/early-stop policy and effective batch size.

`rank_topology_first` implements ordering and `paired_bootstrap_delta` supplies paired confidence intervals. Reports include dataset/config hashes, runtime and peak VRAM.

1. Minimize wire breaks, false bridges and topology violations.
2. Among topology-equivalent candidates, maximize Boundary F1/IoU and minimize Hausdorff.
3. Use Dice/IoU only as tie-breakers.

A lower-Dice model wins only with supported topology improvement and no topology-category regression. Preprocessing/augmentation need artifact-subset gains; confidence needs calibration/error-detection gains; Active Learning needs enrichment. Record target overhead, training time, throughput and VRAM, and verify 256×256 within 8 GB.

| Variant | Seeds | Breaks | Bridges | Topology violations | Boundary F1 | HD95 | Dice | VRAM MB | img/s | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| production baseline | 17/29/43 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline |
| + basic heads | 17/29/43 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| + loss candidate | 17/29/43 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| + geometry group | 17/29/43 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |

## 14. Geometry-aware supervision

Vertices and Gaussian corners come from scale-aware approximations of foreground and hole boundaries. This is target generation only and does not reuse or modify production contour extraction.

A thinned centerline becomes a graph. Junction clusters collapse and branches are traced between endpoints and junctions. Outputs include endpoint heatmaps and T/X/Y junction channels.

Undirected angle uses `(cos 2θ, sin 2θ)`, eliminating the 0°/180° discontinuity. Tangent is centerline-only; orientation propagates into conductor interiors. Curvature comes from traced axial-direction changes, not Sobel derivatives of a binary skeleton.

Topology targets contain separate foreground/background critical skeletons. Losses encourage centerline coverage, separation and junction preservation. Border validity prevents invented endpoints/corners. Geometry adds CPU graph/contour cost and channels; ablate basic heads first, then geometry groups individually.
