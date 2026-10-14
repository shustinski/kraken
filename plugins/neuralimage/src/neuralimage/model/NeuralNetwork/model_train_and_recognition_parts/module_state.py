"""model_train_and_recognition: module state."""
from __future__ import annotations
import re

CHECKPOINT_SUFFIX = '.ckpt'

STOP_TOKEN = '__STOP__'

# Global profiler switch (set in code, not via env var).
TRAINING_PROFILER_ENABLED = False

# Diagnostic profiler for the variable-size patch path. Change only this
# constant to True, reproduce the slowdown, then inspect profiles/ next to the
# model. It captures a Chrome trace plus per-batch/per-shape CSV timings.
RANDOM_PATCH_PROFILING_ENABLED = False

RANDOM_PATCH_PROFILE_BATCHES = 40

RANDOM_PATCH_PROFILE_LOG_EVERY = 5

FOCAL_LOSS_ALPHA = 0.25

FOCAL_LOSS_GAMMA = 2.0

FOCAL_TVERSKY_ALPHA = 0.3

FOCAL_TVERSKY_BETA = 0.7

BOUNDARY_LOSS_KERNEL_SIZE = 3

CLDICE_SKELETON_ITERATIONS = 16

VALIDATION_THRESHOLD_CANDIDATES: tuple[float, ...] = tuple(value / 100.0 for value in range(10, 95, 5))

VALIDATION_CHECK_INTERVAL_BATCHES = 10_000

DEEP_SUPERVISION_DECAY = 0.5

CONFIDENCE_LOSS_WEIGHT = 0.2

RECOGNITION_AUX_WORKERS_PER_GPU = 4

VALIDATION_EXPORT_CACHE_MAX_BYTES = 512 * 1024 * 1024

_ARTIFACT_NAME_SANITIZE_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')

_RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET = 4

_RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY = 8

_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC = 0.25

NON_FINITE_GRADIENT_SKIP_LIMIT = 5

TRAINING_INSTABILITY_MESSAGE = (
    'Не удалось выполнить обучение модели из-за нестабильности. '
    'Проверьте обучающую выборку на однозначность интерпретации данных, '
    'уменьшите скорость обучения в 2-10 раз и/или отключите режим точности и попробуйте снова'
)

_TORCH_COMPILE_MODES = {'default', 'reduce-overhead', 'max-autotune'}

_MAX_AUTOTUNE_MIN_SMS = 68
