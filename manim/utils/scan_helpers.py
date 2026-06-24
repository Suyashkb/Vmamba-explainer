"""Scan path utilities for VMamba explainer scenes."""

import json
from pathlib import Path

from manim import *


def load_scan_paths(data_dir):
    """Load scan paths from stage2_block0.json.

    Returns (H, W, scan_paths_dict) where scan_paths_dict maps
    'd0'..'d3' to list[int] permutations.

    Falls back to analytical_scan_paths on any error.
    """
    data_dir = Path(data_dir)
    json_path = data_dir / "activations" / "stage2_block0.json"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        H = data["H"]
        W = data["W"]
        scan_paths = data["scan_paths"]
        return H, W, scan_paths
    except Exception:
        return analytical_scan_paths(14, 14)


def analytical_scan_paths(H, W):
    """Compute the 4 permutation arrays analytically for an H×W grid.

    Returns (H, W, scan_paths_dict) where scan_paths_dict maps
    'd0'..'d3' to list[int] permutations.

    Directions:
        d0: row-major      (left-to-right, top-to-bottom)
        d1: col-major      (top-to-bottom, left-to-right)
        d2: rev row-major  (right-to-left, bottom-to-top)
        d3: rev col-major  (bottom-to-top, right-to-left)
    """
    L = H * W
    perm0 = list(range(L))
    perm1 = [c * H + r for r in range(H) for c in range(W)]
    perm2 = list(reversed(perm0))
    perm3 = list(reversed(perm1))
    scan_paths = {
        "d0": perm0,
        "d1": perm1,
        "d2": perm2,
        "d3": perm3,
    }
    return H, W, scan_paths


def make_cursor(color, radius=0.13):
    """Return a Dot with given color and radius."""
    dot = Dot(radius=radius, color=color)
    dot.set_fill(color=color, opacity=1.0)
    dot.set_stroke(color=WHITE, width=1.0)
    return dot


def animate_cursors(scene, grid_cells, orders, colors, n_steps=30, dt=0.07):
    """Animate n_steps of scanning across the grid.

    Parameters
    ----------
    scene      : Manim Scene instance
    grid_cells : VGroup of cell Squares (indexed by grid position)
    orders     : list of 4 permutation lists
    colors     : list of 4 colors (one per direction)
    n_steps    : how many steps to animate
    dt         : run_time per step
    """
    # Create cursors at starting positions
    cursors = []
    for d, color in enumerate(colors):
        cursor = make_cursor(color)
        start_cell = grid_cells[orders[d][0]]
        cursor.move_to(start_cell.get_center())
        cursors.append(cursor)
        scene.add(cursor)

    # Track visited sets per direction
    visited = [set() for _ in range(len(orders))]

    for t in range(n_steps):
        anims = []
        for d, cursor in enumerate(cursors):
            order = orders[d]
            if t < len(order):
                target_cell = grid_cells[order[t]]
                anims.append(
                    cursor.animate.move_to(target_cell.get_center())
                )

        if anims:
            scene.play(*anims, run_time=dt, rate_func=linear)

        # Color visited cells
        fill_anims = []
        for d in range(len(orders)):
            order = orders[d]
            if t < len(order):
                cell_idx = order[t]
                visited[d].add(cell_idx)
                cell = grid_cells[cell_idx]
                age = 1.0
                # Dim older cells slightly
                opacity = max(0.15, 0.35 * age)
                fill_anims.append(
                    cell.animate.set_fill(colors[d], opacity=opacity)
                )

        if fill_anims:
            scene.play(*fill_anims, run_time=dt * 0.5, rate_func=linear)
