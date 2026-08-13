"""Shared plotting utilities."""

from pathlib import Path

import joblib
import numpy as np

from .. import PCA_ROOT


def load_pca_rgb() -> tuple[object, np.ndarray, np.ndarray]:
    """Load the 3-D PCA basis and the global normalization range."""
    pca3 = joblib.load(PCA_ROOT / "pca3.joblib")
    norm = np.load(PCA_ROOT / "pca3_norm.npz")
    return pca3, norm["vmin"], norm["vmax"]


def features_to_rgb(features: np.ndarray, pca3, vmin, vmax) -> np.ndarray:
    """Project DINO features to a 3-D PCA space and rescale to RGB in [0, 1]."""
    z = pca3.transform(features)
    z = (z - vmin) / (vmax - vmin + 1e-8)
    return np.clip(z, 0, 1)
