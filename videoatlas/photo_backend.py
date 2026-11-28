"""Collection-adapted scoring: within-class covariance normalization (WCCN).

A general face model spreads one person's faces along directions such as pose,
light, makeup, and age. In a particular collection, those directions can be
larger than the small differences between two similar-looking people. WCCN
estimates the within-person covariance from the collection's own confident
groups (no labels are used) and whitens the embeddings with it, so directions
that separate people count more in the cosine score.

The transform is a fixed matrix per model. It is stored with the run, and new
photos are transformed with the same matrix; the transformed vectors are a new
embedding space with their own key and are never mixed with the original one.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

UNGROUPED = -1


@dataclass(frozen=True)
class Whitening:
    mean: np.ndarray        # subtracted before whitening (zeros when not centering)
    matrix: np.ndarray      # d x d, applied as (x - mean) @ matrix.T
    info: dict

    @property
    def digest(self) -> str:
        data = np.concatenate([self.mean.astype(np.float32).ravel(), self.matrix.astype(np.float32).ravel()]).tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        out = (np.asarray(vectors, np.float64) - self.mean) @ self.matrix.T
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


def within_class_whitening(vectors: np.ndarray, groups: np.ndarray, min_size: int = 10, shrinkage: float = 0.1,
                           center: bool = True) -> Whitening:
    """Whitening from the pooled covariance of faces around their group means.

    Only groups with at least ``min_size`` faces are used. ``shrinkage`` mixes
    the covariance with a scaled identity (0 = none, 1 = no whitening), which
    keeps the transform stable when there are few groups.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    vectors = np.asarray(vectors, np.float64)
    groups = np.asarray(groups)
    dims = vectors.shape[1]
    values, counts = np.unique(groups[groups != UNGROUPED], return_counts=True)
    used = values[counts >= min_size]
    if len(used) < 2:
        raise ValueError(f"need at least two groups with {min_size} or more faces")
    rows = np.flatnonzero(np.isin(groups, used))
    centred = vectors[rows].copy()
    for value in used:
        members = groups[rows] == value
        centred[members] -= centred[members].mean(axis=0)
    covariance = centred.T @ centred / len(rows)
    scale = np.trace(covariance) / dims
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(dims)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12 * scale)
    matrix = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T * np.sqrt(scale)
    mean = vectors.mean(axis=0) if center else np.zeros(dims)
    info = {"method": "wccn", "groups_used": int(len(used)), "faces_used": int(len(rows)), "min_size": min_size,
            "shrinkage": shrinkage, "center": center, "condition_number": float(eigenvalues.max() / eigenvalues.min())}
    return Whitening(mean, matrix, info)
