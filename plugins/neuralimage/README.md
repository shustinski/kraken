# NeuralImage

NeuralImage is a Kraken plugin for neural IC image segmentation, training, and
recognition. It can run as a standalone desktop application, an optional web UI,
or as a Kraken-managed plugin.

## Run

```powershell
python -m neuralimage --help
python -m neuralimage --ui-only
```

### Kraken Agent protocol v1

`python -m neuralimage` automatically enters managed headless mode when
`KRAKEN_JOB_MANIFEST`, `KRAKEN_RESULT_MANIFEST`, and `KRAKEN_STAGING_ROOT`
are present. The same values can be supplied by the `--kraken-*` CLI flags.
Inputs are taken exclusively from the signed-off manifest list and are checked
against SHA-256 before inference.

V1 requires the job parameters `model_relative_path` and `model_sha256`; the
model must also be materialized inside staging. Optional parameters are
`model_version`, `patch_size`, `batch_size`, `overlap`, `threshold`,
`use_auto_threshold`, `tta`, `postprocess_enabled`, and
`postprocess_kernel_size`. A missing or invalid model produces a failed result
manifest and never a fabricated mask. Successful output is always a lossless
grayscale PNG containing only 0 and 255.

## Test

```powershell
pytest
```

## Build

```powershell
.\scripts\build_windows.ps1
```

```bash
./scripts/build_linux.sh
```
