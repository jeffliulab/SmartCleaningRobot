"""Post-cleaning coverage map visualization.

Renders the coverage grid as a 2D heatmap image showing cleaned vs uncleaned
areas, similar to commercial robot vacuum app displays.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def render_coverage_map(
    coverage_grid: np.ndarray,
    trajectory: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a coverage map as a 2D heatmap.

    Args:
        coverage_grid: 2D array where 0=uncleaned, 1=cleaned.
        trajectory: Optional (N, 2) array of (grid_x, grid_y) robot positions
            to overlay as a path trace.
        save_path: If provided, save the figure to this path.
        show: If True, display the figure interactively.

    Returns:
        The matplotlib Figure object.
    """
    cmap = ListedColormap(["#d32f2f", "#4caf50"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(coverage_grid, cmap=cmap, origin="lower", vmin=0, vmax=1, interpolation="nearest")

    if trajectory is not None and len(trajectory) > 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=0.5, alpha=0.6)
        ax.plot(trajectory[0, 0], trajectory[0, 1], "wo", markersize=6, label="Start")
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "w^", markersize=6, label="End")
        ax.legend(loc="upper right", fontsize=8)

    total_free = max(1, (coverage_grid >= 0).sum())
    cleaned = (coverage_grid == 1).sum()
    ratio = cleaned / total_free

    ax.set_title(f"Cleaning Coverage: {ratio * 100:.1f}%", fontsize=16)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Coverage map saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
