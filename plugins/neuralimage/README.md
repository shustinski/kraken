# NeuralImage

NeuralImage is a Kraken plugin for neural IC image segmentation, training, and
recognition. It can run as a standalone desktop application, an optional web UI,
or as a Kraken-managed plugin.

## Run

From a source checkout, open `run_neuralimage.py` in VS Code and press `F5`.
The launcher configures the local `src` package automatically.

```powershell
python run_neuralimage.py
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
