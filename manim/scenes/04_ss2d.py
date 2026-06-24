"""Scene 04: SS2D — Cross-Scan Selective State Space (centerpiece scene)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *
from utils.grid_helpers import make_grid, heatmap_grid
from utils.scan_helpers import make_cursor

DATA_DIR = Path(__file__).parent.parent.parent / "interactive" / "src" / "data"

# Scan direction colors
DIR_COLORS = ["#4e9af1", "#f4a442", "#54c97f", "#e05d78"]
DIR_ARROWS = ["→", "↓", "←", "↑"]


def _analytical_scan_paths_7x7():
    """Return (orders, H, W) for a 7×7 grid."""
    H, W, L = 7, 7, 49
    perm0 = list(range(L))
    perm1 = [c * H + r for r in range(H) for c in range(W)]
    perm2 = list(reversed(perm0))
    perm3 = list(reversed(perm1))
    orders = [perm0, perm1, perm2, perm3]
    return orders, H, W


class SS2DScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text(
            "SS2D: Cross-Scan Selective State Space",
            font_size=38,
            color=WHITE,
        )
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ── Scan paths ───────────────────────────────────────────────────────
        orders, H, W = _analytical_scan_paths_7x7()

        # ── 7×7 grid, centered left ───────────────────────────────────────
        cell_size = 0.45
        gap = 0.03
        grid = make_grid(H, W, cell_size=cell_size, gap=gap, color=GREY_B)
        grid.move_to([-3.0, -0.3, 0])

        # Light grey fill for all cells initially
        for cell in grid:
            cell.set_fill(color="#1e2130", opacity=1.0)

        self.play(Create(grid), run_time=0.8)

        # ── Subtitle ─────────────────────────────────────────────────────────
        subtitle = Text(
            "4 directions sweep simultaneously",
            font_size=26,
            color=GREY_B,
        )
        subtitle.next_to(grid, DOWN, buff=0.35)
        self.play(FadeIn(subtitle, shift=UP * 0.1))
        self.wait(0.2)

        # ── Directional labels at grid corners ───────────────────────────────
        dir_label_positions = [
            grid.get_corner(UL) + LEFT * 0.1 + UP * 0.1,   # → top-left
            grid.get_corner(UR) + RIGHT * 0.1 + UP * 0.1,  # ↓ top-right
            grid.get_corner(DR) + RIGHT * 0.1 + DOWN * 0.1, # ← bottom-right
            grid.get_corner(DL) + LEFT * 0.1 + DOWN * 0.1,  # ↑ bottom-left
        ]
        dir_label_mobs = VGroup()
        for i, (sym, pos) in enumerate(zip(DIR_ARROWS, dir_label_positions)):
            lbl = Text(sym, font_size=28, color=DIR_COLORS[i])
            lbl.move_to(pos)
            dir_label_mobs.add(lbl)
        self.play(FadeIn(dir_label_mobs))

        # ── Create 4 cursors at their starting positions ──────────────────
        cursors = []
        start_indices = [
            orders[0][0],  # top-left  for row-major
            orders[1][0],  # top-left  for col-major
            orders[2][0],  # bottom-right for rev row-major
            orders[3][0],  # bottom-right for rev col-major
        ]
        for d in range(4):
            cursor = make_cursor(DIR_COLORS[d], radius=0.11)
            cursor.move_to(grid[start_indices[d]].get_center())
            cursors.append(cursor)

        self.play(*[FadeIn(c) for c in cursors])
        self.wait(0.2)

        # ── Animate 25 scanning steps ────────────────────────────────────────
        n_steps = 25
        dt = 0.09
        visited = [set() for _ in range(4)]

        for t in range(n_steps):
            move_anims = []
            for d, cursor in enumerate(cursors):
                if t < len(orders[d]):
                    target_cell = grid[orders[d][t]]
                    move_anims.append(
                        cursor.animate.move_to(target_cell.get_center())
                    )
            if move_anims:
                self.play(*move_anims, run_time=dt, rate_func=linear)

            # Color visited cells
            fill_anims = []
            for d in range(4):
                if t < len(orders[d]):
                    cell_idx = orders[d][t]
                    visited[d].add(cell_idx)
                    cell = grid[cell_idx]
                    fill_anims.append(
                        cell.animate.set_fill(DIR_COLORS[d], opacity=0.35)
                    )
            if fill_anims:
                self.play(*fill_anims, run_time=dt * 0.4, rate_func=linear)

        self.wait(0.3)

        # ── Feature maps: 4 small 4×4 heatmaps to the right ──────────────────
        import math, random

        random.seed(42)
        mini_maps = []
        mini_labels = []
        feat_x = 1.5
        feat_y_start = 2.8
        feat_spacing = 1.55
        mini_size = 0.28

        for d in range(4):
            vals = [random.random() for _ in range(16)]
            hmap = heatmap_grid(4, 4, vals, 0.0, 1.0, cell_size=mini_size)
            hmap.move_to([feat_x, feat_y_start - d * feat_spacing, 0])
            lbl = Text(f"d{d}", font_size=18, color=DIR_COLORS[d])
            lbl.next_to(hmap, LEFT, buff=0.18)
            mini_maps.append(hmap)
            mini_labels.append(lbl)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(FadeIn(mini_maps[d]), FadeIn(mini_labels[d]))
                    for d in range(4)
                ],
                lag_ratio=0.25,
                run_time=1.2,
            )
        )
        self.wait(0.3)

        # ── ⊕ symbol in center-right ──────────────────────────────────────
        plus_x = 3.0
        plus_y = feat_y_start - 1.5 * feat_spacing
        plus_circle = Circle(radius=0.22, color=WHITE, stroke_width=2)
        plus_circle.move_to([plus_x, plus_y, 0])
        plus_tex = MathTex(r"\oplus", font_size=32, color=WHITE)
        plus_tex.move_to([plus_x, plus_y, 0])
        plus_group = VGroup(plus_circle, plus_tex)
        self.play(FadeIn(plus_group))

        # Arrows from each mini-map to ⊕
        merge_arrows = VGroup()
        for d in range(4):
            arr = Arrow(
                mini_maps[d].get_right(),
                plus_circle.get_left(),
                buff=0.05,
                color=DIR_COLORS[d],
                stroke_width=1.5,
                tip_length=0.12,
            )
            merge_arrows.add(arr)
        self.play(LaggedStart(*[Create(a) for a in merge_arrows], lag_ratio=0.15))
        self.wait(0.3)

        # ── Merged output map ─────────────────────────────────────────────
        random.seed(99)
        merged_vals = [random.random() for _ in range(16)]
        merged_map = heatmap_grid(4, 4, merged_vals, 0.0, 1.0, cell_size=mini_size)
        merged_map.move_to([plus_x + 1.3, plus_y, 0])
        merged_label = Text("merged", font_size=18, color=WHITE)
        merged_label.next_to(merged_map, DOWN, buff=0.1)

        merge_out_arrow = Arrow(
            plus_circle.get_right(),
            merged_map.get_left(),
            buff=0.05,
            color=WHITE,
            stroke_width=2,
            tip_length=0.12,
        )
        self.play(Create(merge_out_arrow), FadeIn(merged_map), FadeIn(merged_label))
        self.wait(0.3)

        # ── Caption ───────────────────────────────────────────────────────
        caption = Text(
            "Each direction captures different spatial dependencies.",
            font_size=22,
            color=LIGHT_GREY,
        )
        caption.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(caption, shift=UP * 0.1))
        self.wait(2)
