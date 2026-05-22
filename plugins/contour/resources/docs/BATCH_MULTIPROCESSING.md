# Batch Multiprocessing Architecture

Contour batch processing now uses a three-layer runtime:

```text
PyQt GUI process
  |
  | Qt signals, throttled progress, metadata only
  v
BatchProcessor / BatchQueueRunnable
  |
  | ProcessPoolExecutor, spawn context, chunk requests
  v
Worker processes
  |
  | load image -> preprocess -> detect/vectorize -> save -> return metadata
  v
BatchImageMetadata
```

The GUI remains in the main process. The Qt `QRunnable` is only a lightweight
orchestrator; it never performs frame processing itself.

## IPC Contract

Worker requests contain only serializable control data:

- input image paths
- output directory
- pipeline configuration
- contour extraction settings
- save/display settings

Workers load images locally, process them locally, save results locally, and
return only metadata:

- image path
- polygon count
- saved file paths
- per-frame timing
- worker PID
- error text, if any

Large NumPy arrays, masks, debug maps, `QImage`, and `QPixmap` objects are not
sent through multiprocessing IPC.

## Chunking

The orchestrator submits chunks of image paths instead of one future per image.
Default chunk size is `16` and can be overridden with:

```powershell
$env:CONTOUR_BATCH_CHUNK_SIZE = "32"
```

Each worker processes its assigned chunk sequentially. This reduces executor
scheduling overhead and prevents thousands of pending futures during large runs.

## OpenCV Runtime

Both the GUI process and worker processes configure OpenCV as:

```python
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
```

Process-level parallelism provides CPU scaling. Per-process OpenCV thread pools
are disabled to avoid oversubscription and long-run throughput collapse.

## Diagnostics

Batch logs include:

- process count
- chunk count and chunk size
- per-chunk wall time
- worker utilization
- worker RSS memory when `psutil` is available
- average load, contour, save, and total frame timing
- average throughput
- queued and active chunk counts during runtime diagnostics

Per-frame timing buckets:

- image loading
- threshold operations
- morphology operations
- contour extraction
- postprocessing
- saving
- total frame time

## Benchmark

Run:

```powershell
uv run python plugins\contour\scripts\benchmark_batch.py plugins\contour\dataset\images\frame_1.png --repeats 200 --workers 2 --chunk-size 25
```

The script reports a sequential baseline, the multiprocessing batch path, speedup,
and first-half/second-half degradation ratios. Use production-sized images for
meaningful throughput numbers; tiny test frames are dominated by process startup
and IPC overhead.
