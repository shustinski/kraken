"""Grid-defect algorithm version: the one place to bump when analysis results change.

The number goes into the disk-cache key (results of another algorithm are never
reused) and into the app version: 0.2.<number>-betaN.
"""

from __future__ import annotations

GRID_ALGORITHM_NUMBER = 94
GRID_ALGORITHM_TAG = "grid_free_normal_bank"
GRID_DAMAGE_ALGORITHM_VERSION = f"grid_damage_v{GRID_ALGORITHM_NUMBER}_{GRID_ALGORITHM_TAG}"
