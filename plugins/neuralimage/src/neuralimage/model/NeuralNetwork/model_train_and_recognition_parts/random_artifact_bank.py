"""model_train_and_recognition: random artifact bank."""
from __future__ import annotations
import math
from collections import deque
import threading
import torch
import torch.nn.functional as F
from neuralimage.lib.random_artifacts import generate_random_artifact_patch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    _RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY,
    _RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC,
    _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
)

class _RandomArtifactBank:
    def __init__(
        self,
        *,
        channels: int,
        artifact_types: tuple[str, ...],
        target_per_bucket: int = _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
        bucket_granularity: int = _RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY,
        base_seed: int | None = None,
    ) -> None:
        self._channels = max(1, int(channels))
        self._artifact_types = tuple(artifact_types)
        self._target_per_bucket = max(1, int(target_per_bucket))
        self._bucket_granularity = max(1, int(bucket_granularity))
        self._base_seed = int(base_seed if base_seed is not None else (torch.initial_seed() & 0x7FFFFFFF))
        self._seed_counter = 0
        self._cache: dict[tuple[int, int], deque[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._pending_requests: dict[tuple[int, int], int] = {}
        self._request_queue: deque[tuple[int, int]] = deque()
        self._condition = threading.Condition()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def channels(self) -> int:
        return int(self._channels)

    @property
    def artifact_types(self) -> tuple[str, ...]:
        return tuple(self._artifact_types)

    def matches(self, *, channels: int, artifact_types: tuple[str, ...]) -> bool:
        return int(channels) == int(self._channels) and tuple(artifact_types) == tuple(self._artifact_types)

    def start(self, prewarm_buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ()) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name='random-artifact-bank',
                daemon=True,
            )
            self._thread.start()
            for bucket in prewarm_buckets:
                self._ensure_bucket_target_locked(self._normalize_bucket(*bucket))
            self._condition.notify_all()

    def stop(self, *, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout)))
        with self._condition:
            self._thread = None
            self._cache.clear()
            self._pending_requests.clear()
            self._request_queue.clear()

    def wait_until_ready(self, timeout: float) -> bool:
        return bool(self._ready_event.wait(timeout=max(0.0, float(timeout))))

    def acquire(self, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        requested_height = max(1, int(height))
        requested_width = max(1, int(width))
        requested_bucket = self._normalize_bucket(requested_height, requested_width)
        entry, source_bucket = self._try_take_entry(requested_bucket)
        if entry is None:
            self.wait_until_ready(timeout=_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC)
            entry, source_bucket = self._try_take_entry(requested_bucket)
        if entry is None:
            source_bucket = requested_bucket
            entry = self._generate_bucket_entry(requested_bucket)
        overlay, alpha = entry
        with self._condition:
            self._ensure_bucket_target_locked(requested_bucket)
            if source_bucket is not None:
                self._ensure_bucket_target_locked(source_bucket)
            self._condition.notify_all()
        if overlay.shape[-2:] != (requested_height, requested_width):
            overlay = F.interpolate(
                overlay.unsqueeze(0),
                size=(requested_height, requested_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            alpha = F.interpolate(
                alpha.unsqueeze(0),
                size=(requested_height, requested_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            alpha = torch.clamp(alpha, min=0.0, max=1.0)
        return overlay, alpha

    def _next_seed(self) -> int:
        with self._condition:
            seed = (self._base_seed + self._seed_counter) & 0x7FFFFFFF
            self._seed_counter += 1
        return int(seed)

    def _normalize_bucket(self, height: int, width: int) -> tuple[int, int]:
        granularity = int(self._bucket_granularity)
        bucket_height = max(granularity, int(math.ceil(max(1, int(height)) / granularity) * granularity))
        bucket_width = max(granularity, int(math.ceil(max(1, int(width)) / granularity) * granularity))
        return int(bucket_height), int(bucket_width)

    def _ensure_bucket_target_locked(self, bucket: tuple[int, int]) -> None:
        cache = self._cache.setdefault(bucket, deque())
        pending = int(self._pending_requests.get(bucket, 0))
        required = max(0, int(self._target_per_bucket) - (len(cache) + pending))
        for _ in range(required):
            self._request_queue.append(bucket)
            pending += 1
        self._pending_requests[bucket] = pending

    def _try_take_entry(
        self,
        requested_bucket: tuple[int, int],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, tuple[int, int] | None]:
        with self._condition:
            self._ensure_bucket_target_locked(requested_bucket)
            exact_cache = self._cache.get(requested_bucket)
            if exact_cache:
                entry = exact_cache.popleft()
                return entry, requested_bucket
            nearest_bucket: tuple[int, int] | None = None
            nearest_distance: int | None = None
            for bucket, cache in self._cache.items():
                if not cache:
                    continue
                distance = abs(int(bucket[0]) - int(requested_bucket[0])) + abs(int(bucket[1]) - int(requested_bucket[1]))
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_bucket = bucket
            if nearest_bucket is None:
                return None, None
            entry = self._cache[nearest_bucket].popleft()
            return entry, nearest_bucket

    def _generate_bucket_entry(self, bucket: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        return generate_random_artifact_patch(
            channels=int(self._channels),
            height=int(bucket[0]),
            width=int(bucket[1]),
            device=torch.device('cpu'),
            dtype=torch.float32,
            artifact_types=self._artifact_types,
            seed=self._next_seed(),
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while not self._stop_event.is_set() and not self._request_queue:
                    self._condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    return
                bucket = self._request_queue.popleft()
                self._pending_requests[bucket] = max(0, int(self._pending_requests.get(bucket, 0)) - 1)
            entry = self._generate_bucket_entry(bucket)
            with self._condition:
                if self._stop_event.is_set():
                    return
                self._cache.setdefault(bucket, deque()).append(entry)
                self._ready_event.set()
                self._condition.notify_all()
