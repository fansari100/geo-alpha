"""Walk-forward evaluation harness."""

from .metrics import (
    detection_metrics,
    information_coefficient,
    rank_ic,
)
from .walk_forward import (
    DetectorResult,
    WalkForwardConfig,
    walk_forward_threshold_search,
)

__all__ = [
    "DetectorResult",
    "WalkForwardConfig",
    "walk_forward_threshold_search",
    "detection_metrics",
    "information_coefficient",
    "rank_ic",
]
