# NeuralImage

NeuralImage is a Kraken plugin for neural IC image segmentation, training, and
recognition. It can run as a standalone desktop application, an optional web UI,
or as a Kraken-managed plugin.

## Run

```powershell
python -m neuralimage --help
python -m neuralimage --ui-only
```

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
