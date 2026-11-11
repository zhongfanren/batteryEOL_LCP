"""LCP: Localized Conformal Prediction — Python implementation."""

from .core import (
    id_low_search,
    q_low_compute,
    lcp_construction_path_distance,
    lcp_alpha,
    auto_tune_distance,
    lcp_construction_path,
)

from .distance import (
    pairwise_dist,
    lcp_default_distance,
)

# Pipeline needs torch — import lazily.
try:
    from .pipeline import lcp_compare
except ImportError:
    lcp_compare = None

__all__ = [
    "id_low_search",
    "q_low_compute",
    "lcp_construction_path_distance",
    "lcp_alpha",
    "auto_tune_distance",
    "lcp_construction_path",
    "pairwise_dist",
    "lcp_default_distance",
    "lcp_compare",
]
