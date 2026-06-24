from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tracking.trajectory_cleaner import TrajectoryPoint


def save_trajectory_plot(
    points: List[TrajectoryPoint],
    output_path: Path,
    pixel_to_meter: Optional[float] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    fig.patch.set_facecolor("#071018")
    ax.set_facecolor("#0b111b")

    if points:
        x = np.array([point.x - points[0].x for point in points], dtype=float)
        y = np.array([points[0].y - point.y for point in points], dtype=float)
        if pixel_to_meter:
            x = x * pixel_to_meter
            y = y * pixel_to_meter
            x_label = "水平位移（米）"
            y_label = "垂直位移（米）"
        else:
            x_label = "水平位移（px）"
            y_label = "垂直位移（px）"

        ax.plot(x, y, color="#00d5ff", linewidth=2.6)
        ax.scatter(x, y, color="#39ff88", s=18, zorder=3)
        ax.scatter([x[0]], [y[0]], color="#ffd23f", s=58, zorder=4, label="起点")
        ax.scatter([x[-1]], [y[-1]], color="#ff3b30", s=58, zorder=4, label="终点")
    else:
        x_label = "水平位移"
        y_label = "垂直位移"

    ax.set_title("估算球路轨迹", color="#f8fbff", fontsize=14, pad=12)
    ax.set_xlabel(x_label, color="#cfe8f7")
    ax.set_ylabel(y_label, color="#cfe8f7")
    ax.grid(color="#263648", alpha=0.58, linewidth=0.8)
    ax.tick_params(colors="#a9c7d8")
    for spine in ax.spines.values():
        spine.set_color("#2a3b4d")
    if points:
        legend = ax.legend(facecolor="#0b111b", edgecolor="#2a3b4d")
        for text in legend.get_texts():
            text.set_color("#e8f6ff")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
