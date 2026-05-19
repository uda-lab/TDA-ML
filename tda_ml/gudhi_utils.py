"""Helpers for GUDHI 3.12+ Python bindings (nanobind)."""

import numpy as np
from scipy.spatial.distance import squareform


def distance_matrix_for_gudhi(dm: np.ndarray) -> np.ndarray:
    """Return float64, C-contiguous, read-only square distance matrix for RipsComplex."""
    if dm.ndim == 1:
        dm = squareform(dm)
    out = np.array(np.asarray(dm, dtype=np.float64), order="C", copy=True)
    out.setflags(write=False)
    return out
