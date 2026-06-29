# NeuralImage SEM Segmentation Enhancements

This document summarizes the modular neural network improvements for SEM IC layout segmentation.

## Module Layout

```
src/neuralimage/
  preprocessing/     Shared SEM preprocessing (training + inference)
  targets/           Automatic supervision target generation
  heads/             Multi-target prediction heads
  losses/            Composite and auxiliary losses
  metrics/           Advanced validation metrics
  uncertainty/       Confidence / MC dropout / TTA variance
  active_learning/   NeedsAnnotation export infrastructure
  training/          Hard example mining helpers
  augmentations/sem.py  SEM-specific augmentations
```

## Configuration Example

```json
{
  "tranining_parameters": {
    "targets": {
      "basic": {
        "boundary": true,
        "skeleton": true,
        "sdf": true,
        "distance_transform": false,
        "thickness": false
      },
      "geometry": {
        "corner": true,
        "junction": true,
        "orientation": true
      },
      "auxiliary_head_weights": {
        "boundary": 0.3,
        "skeleton": 0.25,
        "sdf": 0.2
      }
    },
    "preprocessing": {
      "percentile_normalization": true,
      "clahe": true,
      "scan_line_suppression": true
    },
    "sem_augmentation": {
      "enabled": true,
      "charging_artifacts": true,
      "scan_drift": true
    },
    "uncertainty": {
      "enabled": true,
      "method": "confidence_head"
    },
    "active_learning": {
      "enabled": true,
      "export_dir": "D:/data/NeedsAnnotation"
    },
    "advanced_validation": true
  }
}
```

## Feature Notes

### Multi-target supervision (Feature 1)
Binary polygon masks remain the only manual label format. Boundary, skeleton, SDF, distance transform, thickness, and geometry targets are generated online during dataset preparation.

### Multi-head architecture (Feature 2)
`MultiTargetHeadBundle` adds auxiliary heads from the final decoder features. Inference exports only the binary mask; auxiliary heads are training-only.

### Composite loss (Feature 3)
Existing composite loss in `loss_config.py` and `model_train_and_recognition.py` is preserved. Auxiliary head losses are added via `losses/composite.py` with optional dynamic weighting.

### SEM preprocessing (Feature 4)
`SemPreprocessingPipeline` applies identical preprocessing in dataset loading and can be reused in recognition.

### SEM augmentation (Feature 5)
`SemAugmentor` complements existing tech/IC augmentations with charging, drift, focus variation, detector noise, gradients, and realistic defects.

### Hard example mining (Feature 6)
Existing `LossAwareSampler` online mining is preserved. `training/hard_mining.py` adds geometry-aware difficulty scoring and offline hard dataset generation.

### Context branch (Feature 7)
Existing quasi-dual-scale context branch is unchanged. Coordinate-aware cross-attention and dual-scale fusion remain the default for context-enabled models.

### Confidence estimation (Feature 8)
Supports confidence head (existing), Monte Carlo dropout, and TTA variance via `uncertainty/estimators.py`.

### Active learning (Feature 9)
Infrastructure exports low-confidence / high-entropy / unstable samples into `NeedsAnnotation/` without GUI.

### Advanced validation (Feature 10)
Validation reports Dice, IoU, Boundary IoU/F1, Hausdorff distance, connected-component difference, wire breaks, false bridges, and confidence histogram summaries.

### Geometry-aware supervision (Feature 14)
Corner, junction, orientation, tangent, curvature, vertex, endpoint, and topology targets are derived automatically from polygon masks.

## Computational Cost

| Component | Relative cost |
|-----------|---------------|
| Target generation | Low CPU per patch |
| Auxiliary heads | +5–15% GPU memory / forward |
| CLAHE + denoise preprocessing | Moderate CPU |
| MC dropout uncertainty | Linear in sample count |
| Advanced validation metrics | Moderate CPU per validation batch |

## Integration

1. Enable targets in training config.
2. Select a model that supports `supervision_heads` (EfficientUNet family).
3. Keep composite mask loss weights in `loss_term_weights`.
4. Tune auxiliary head weights separately under `targets.auxiliary_head_weights`.
