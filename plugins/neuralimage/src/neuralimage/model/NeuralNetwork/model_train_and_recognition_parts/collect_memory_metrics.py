"""model_train_and_recognition: collect memory metrics."""
from __future__ import annotations
import os
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import _resolve_memory_root_process

def _collect_memory_metrics() -> dict[str, float] | None:
    ram_mb: float | None = None
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        root_process = _resolve_memory_root_process(process)
        processes = [root_process, *root_process.children(recursive=True)]
        seen_pids: set[int] = set()
        rss_bytes = 0
        for candidate in processes:
            try:
                pid = int(candidate.pid)
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                rss_bytes += int(candidate.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        ram_mb = float(rss_bytes) / (1024.0 * 1024.0)
    except Exception:
        ram_mb = None

    vram_alloc_mb: float | None = None
    vram_reserved_mb: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    if torch.cuda.is_available():
        try:
            vram_alloc_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            vram_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        except Exception:
            vram_alloc_mb = None
            vram_reserved_mb = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
            vram_total_mb = float(total_bytes) / (1024.0 * 1024.0)
            vram_used_mb = float(total_bytes - free_bytes) / (1024.0 * 1024.0)
        except Exception:
            vram_used_mb = None
            vram_total_mb = None

    if (
        ram_mb is None
        and vram_alloc_mb is None
        and vram_reserved_mb is None
        and vram_used_mb is None
        and vram_total_mb is None
    ):
        return None

    payload: dict[str, float] = {}
    if ram_mb is not None:
        payload['ram_mb'] = ram_mb
    if vram_alloc_mb is not None:
        payload['vram_allocated_mb'] = vram_alloc_mb
    if vram_reserved_mb is not None:
        payload['vram_reserved_mb'] = vram_reserved_mb
    if vram_used_mb is not None:
        payload['vram_used_mb'] = vram_used_mb
    if vram_total_mb is not None:
        payload['vram_total_mb'] = vram_total_mb
    return payload
