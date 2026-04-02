"""Embedding visualization with UMAP and PCA."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def plot_umap(
    embeddings: np.ndarray,
    output_path: str | Path,
    labels: np.ndarray | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP of Antibody Embeddings",
) -> None:
    """Generate a UMAP scatter plot of embeddings.

    Args:
        embeddings: Array of shape (n_samples, hidden_size).
        output_path: Path to save the plot image.
        labels: Optional array of category labels for coloring.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        title: Plot title.
    """
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_kwargs: dict = {"s": 3, "alpha": 0.5}
    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", **scatter_kwargs)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("UMAP plot saved to %s", output_path)


def plot_pca(
    embeddings: np.ndarray,
    output_path: str | Path,
    labels: np.ndarray | None = None,
    n_components: int = 2,
    title: str = "PCA of Antibody Embeddings",
) -> None:
    """Generate a PCA scatter plot of embeddings.

    Args:
        embeddings: Array of shape (n_samples, hidden_size).
        output_path: Path to save the plot image.
        labels: Optional category labels for coloring.
        n_components: Number of PCA components (2 for scatter).
        title: Plot title.
    """
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_kwargs: dict = {"s": 3, "alpha": 0.5}
    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", **scatter_kwargs)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

    variance = pca.explained_variance_ratio_
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("PCA plot saved to %s", output_path)
