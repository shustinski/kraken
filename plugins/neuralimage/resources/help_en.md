# NeuralImage Help

NeuralImage lets you train a segmentation model, continue training an existing model, and run recognition on prepared images.

This help page is organized like a large project knowledge base:

- the catalog is shown on the left;
- articles expand on the right;
- important settings are explained in plain language without requiring deep PyTorch knowledge.

## Quick Start

Use this section for the fastest safe first run.

### Where to begin

If this is your first NeuralImage run, use the following order:

1. Select `Train and recognize`.
2. Choose folders with training images and masks.
3. Keep the default model and optimizer settings.
4. Train for 3-5 epochs.
5. Check `train loss`, `validation loss`, and the output folder.

### Recommended starting values

- Patch size: `256 × 256`
- Patch step: `96-128`
- Validation: `20%`
- Optimizer: `AdamW`
- Learning rate: `0.0005`
- Training batch size: `8-16`
- Mixed precision: `bf16` or `fp16`, if your GPU supports it
- Loss function: `bce_dice`
- Warmup: enabled for `3` epochs

### When to tune settings manually

Change parameters only after you have a baseline run.

- If GPU memory is too low, reduce batch size first.
- If quality is poor, check masks and validation first.
- If output tiles have seams, increase `Overlap`.

## Work Modes

Choose the mode based on the task, not habit, because it changes which fields are required.

### Train and recognize

This mode runs the full pipeline:

- trains a new model on your dataset;
- applies it to source images after training;
- saves the output in the selected folder.

Use it when you need a new model and want to verify the result immediately.

### Continue training and recognize

Use this mode when you already have a `.pth` file and want to improve it with new data.

Required inputs:

- a folder with training images;
- a folder with masks;
- a path to an existing model.

### Recognition only

This mode does not train anything. It simply loads an existing `.pth` file and runs inference on the source images.

It is suitable for production runs, repeated processing, or comparing multiple trained models.

### Training only

This mode trains or fine-tunes the model without running recognition afterward.

It is useful when you want to:

- prepare weights separately;
- compare several training runs;
- avoid spending time on output generation after every experiment.

## Data and Paths

Before running, check your folder layout and make sure images match masks correctly.

### Which folders are required

Different modes require different inputs:

- `Source files` and `Output folder` are required for recognition;
- `Images` and `Labels` are required for training;
- `Model file` is required for fine-tuning and recognition with an existing model.

### How to prepare the training dataset

Each image is expected to have a matching mask or label.

Practical recommendations:

- keep matching file names in sync;
- avoid mixing formats unless you have a reason to do so;
- keep images and masks in separate folders.

### What matters most for labels

Bad labels usually hurt quality more than imperfect training parameters.

Check these before starting:

- masks are not empty when an object should exist;
- masks are not fully filled without a good reason;
- image and label sizes match;
- file pairs match by name.

## Training Settings

These parameters have the strongest effect on quality and speed.

### Patch size and patch step

A `patch` is the image fragment the model sees at one time.

- a large patch provides more context but uses more memory;
- a small patch saves memory but may lose large objects;
- a smaller `Patch step` creates more samples and increases training time.

### Validation, batch size, and mixed precision

- `Validation` shows whether the model improves on held-out data;
- `Training batch size` affects both speed and memory usage;
- `Mixed precision` usually speeds up training and reduces VRAM usage.

If memory is not enough, reduce settings in this order:

1. batch size;
2. patch size;
3. extra augmentation.

### Loss function

If you are unsure, start with one of these:

- `bce_dice` for binary segmentation;
- `ce_dice` when boundary stability matters more;
- `bce_iou` when overlap quality should be optimized more aggressively.

`Dice weight` and `IoU weight` matter only for combined loss functions.

### Warmup, hard mining, and early stopping

- `Warmup` helps stabilize the start of training;
- `Hard mining` shows high-error examples more often;
- `Hard pixel mining` emphasizes the hardest pixels inside each mask;
- `Early stopping` ends the run if quality stops improving.

For a stable first run, this is often enough:

- warmup = on;
- hard mining = off;
- early stopping = on.

## Preprocessing and Augmentation

These settings help adapt input data before training starts.

### When to crop edges

Enable `Crop edges` when image borders contain:

- service frames;
- noisy margins;
- black or white stripes;
- scanning artifacts.

### When to resize

Resize is useful when images are too large, too inconsistent in size, or must be normalized to one scale.

Do not resize without a reason: unnecessary scaling can damage small details.

### How to use augmentation

Augmentation is especially useful when the dataset is small or capture conditions vary a lot.

Safe starting values:

- brightness: `0.1`
- contrast: `0.1`
- noise probability: `0.3-0.5`
- noise strength: `0.01`

## Monitoring and Troubleshooting

The metrics panel helps you quickly understand what slows training down or hurts quality.

### How to read the charts

- `Train loss by epoch` should generally move down;
- `Validation loss` should not keep getting worse;
- if `train loss` goes down but `validation loss` goes up, overfitting is likely starting.

### What batch timing means

- `data wait` is time spent waiting for the next batch from the DataLoader;
- `forward` is the model forward pass;
- `backward` is the gradient computation pass;
- `optimizer` is the optimizer step;
- `total` is the full duration of one batch step.

If most time is spent in `data wait`, the bottleneck is likely data loading or storage, not the model itself.

### Common problems

If something goes wrong, check these first:

- **Not enough VRAM**: reduce batch size and patch size.
- **Low quality**: verify masks, enable validation, and lower the learning rate.
- **Visible seams in output**: increase recognition overlap.
- **Training is very slow**: inspect `data wait`, disable unnecessary preprocessing, and enable mixed precision.
