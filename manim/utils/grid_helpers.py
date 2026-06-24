"""Grid utility helpers for VMamba explainer scenes."""

from manim import *


def make_grid(rows, cols, cell_size=0.4, gap=0.02, color=GREY_B):
    """Create a VGroup of rows*cols Squares arranged row-major, centered at ORIGIN.

    Each cell has a .grid_index attribute (row-major int index).
    """
    cells = VGroup()
    total_w = cols * cell_size + (cols - 1) * gap
    total_h = rows * cell_size + (rows - 1) * gap
    origin_x = -total_w / 2 + cell_size / 2
    origin_y = total_h / 2 - cell_size / 2

    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell_size)
            sq.set_stroke(color=color, width=1.0)
            sq.set_fill(color=BLACK, opacity=0.0)
            x = origin_x + c * (cell_size + gap)
            y = origin_y - r * (cell_size + gap)
            sq.move_to([x, y, 0])
            sq.grid_index = r * cols + c
            cells.add(sq)

    return cells


def heatmap_grid(rows, cols, values, vmin, vmax, cell_size=0.4):
    """Create a VGroup with cells filled via a DARK_BLUE→YELLOW gradient heatmap.

    values: flat list/array of length rows*cols.
    """
    cells = VGroup()
    total_w = cols * cell_size + (cols - 1) * 0.02
    total_h = rows * cell_size + (rows - 1) * 0.02
    origin_x = -total_w / 2 + cell_size / 2
    origin_y = total_h / 2 - cell_size / 2

    range_v = vmax - vmin if (vmax - vmin) != 0 else 1.0

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            val = values[idx] if idx < len(values) else vmin
            t = max(0.0, min(1.0, (val - vmin) / range_v))
            fill_color = interpolate_color(DARK_BLUE, YELLOW, t)

            sq = Square(side_length=cell_size)
            sq.set_stroke(color=GREY_B, width=0.5)
            sq.set_fill(color=fill_color, opacity=1.0)

            x = origin_x + c * (cell_size + 0.02)
            y = origin_y - r * (cell_size + 0.02)
            sq.move_to([x, y, 0])
            sq.grid_index = idx
            cells.add(sq)

    return cells


def patch_overlay(grid, idx, color, opacity=0.5):
    """Return a Square covering grid[idx] for cursor highlighting.

    The overlay matches the cell's size and position.
    """
    cell = grid[idx]
    side = cell.width
    overlay = Square(side_length=side)
    overlay.set_stroke(width=0)
    overlay.set_fill(color=color, opacity=opacity)
    overlay.move_to(cell.get_center())
    return overlay
