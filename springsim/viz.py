from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .system import System


def plot_configuration(system: System, positions: np.ndarray, out_path: Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_springs(ax, system.springs, positions)
    pos = positions.reshape(-1, 2)
    ax.scatter(pos[:, 0], pos[:, 1], c="tab:blue", s=12, label="particles")
    fixed_coords = [
        (pos[i, 0], pos[i, 1])
        for i, particle in enumerate(system.particles)
        if particle.fixed
    ]
    if fixed_coords:
        fx, fy = zip(*fixed_coords)
        ax.scatter(fx, fy, c="tab:red", s=28, label="fixed", marker="s")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def animate_system(
    system: System,
    positions_sequence: Sequence[np.ndarray],
    out_path: Path,
    *,
    fps: int = 30,
    dpi: int = 120,
) -> Path:
    from matplotlib import animation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    line_collection = _init_animation_plot(ax, system, positions_sequence[0])

    def update(frame_index: int):
        positions = positions_sequence[frame_index].reshape(-1, 2)
        for line, spring in zip(line_collection, system.springs):
            i = spring.i
            j = spring.j
            line.set_data([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]])
        scatter = line_collection[-1]
        scatter.set_offsets(positions)
        return line_collection

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions_sequence),
        blit=True,
        interval=1000.0 / fps,
    )

    writer, output_path = _select_writer(animation, out_path)
    if writer is None:
        plt.close(fig)
        return _save_frame_sequence(system, positions_sequence, out_path)

    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def _draw_springs(ax, springs, positions: np.ndarray):
    pos = positions.reshape(-1, 2)
    for spring in springs:
        i, j = spring.i, spring.j
        xs = [pos[i, 0], pos[j, 0]]
        ys = [pos[i, 1], pos[j, 1]]
        ax.plot(xs, ys, color="lightgray", linewidth=1.0)


def _init_animation_plot(ax, system: System, positions: np.ndarray):
    pos = positions.reshape(-1, 2)
    lines = []
    for spring in system.springs:
        i, j = spring.i, spring.j
        (line,) = ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="lightgray", linewidth=1.0)
        lines.append(line)
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c="tab:blue", s=10)
    lines.append(scatter)
    ax.set_xlim(pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5)
    ax.set_ylim(pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    return lines


def _select_writer(animation_module, requested_path: Path):
    writers = animation_module.writers
    suffix = requested_path.suffix.lower()
    if suffix == ".mp4" and writers.is_available("ffmpeg"):
        return animation_module.FFMpegWriter(fps=30, bitrate=1800), requested_path
    if suffix in {".gif", ".mp4"} and writers.is_available("pillow"):
        fallback = requested_path.with_suffix(".gif")
        return animation_module.PillowWriter(fps=30), fallback
    if suffix == ".gif" and writers.is_available("imagemagick"):
        return animation_module.ImageMagickWriter(fps=30), requested_path
    return None, requested_path


def _save_frame_sequence(system: System, positions_sequence: Sequence[np.ndarray], out_path: Path) -> Path:
    out_dir = out_path if out_path.suffix == "" else out_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, positions in enumerate(positions_sequence):
        frame_path = out_dir / f"frame_{idx:05d}.png"
        plot_configuration(system, positions, frame_path, title=f"Frame {idx}")
    return out_dir


__all__ = ["plot_configuration", "animate_system"]
