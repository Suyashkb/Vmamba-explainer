"""Scene 01: Motivation — Why Not Just Attention?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *


class MotivationScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text(
            "Why Not Just Attention?",
            font_size=44,
            color=WHITE,
        )
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ── Axes ─────────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[1, 8, 1],
            y_range=[0, 64, 8],
            x_length=8,
            y_length=4.5,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=True,
        )
        axes.shift(DOWN * 0.3)

        x_label = axes.get_x_axis_label(
            Text("Sequence length L (×100)", font_size=20, color=GREY_B),
            edge=RIGHT,
            direction=DOWN,
            buff=0.1,
        )
        y_label = axes.get_y_axis_label(
            Text("Compute (relative)", font_size=20, color=GREY_B).rotate(
                90 * DEGREES
            ),
            edge=LEFT,
            direction=LEFT,
            buff=0.3,
        )

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(0.2)

        # ── O(L) curve — SSM / Mamba ─────────────────────────────────────────
        linear_color = "#4e9af1"
        linear_curve = axes.plot(
            lambda x: x,
            x_range=[1, 8],
            color=linear_color,
            stroke_width=3,
        )
        linear_label = Text("O(L)  SSM / Mamba", font_size=22, color=linear_color)
        linear_label.next_to(axes.c2p(8, 8), RIGHT + UP * 0.1, buff=0.1)

        self.play(Create(linear_curve))
        self.play(Write(linear_label))
        self.wait(0.3)

        # ── O(L²) curve — Attention ──────────────────────────────────────────
        quad_color = "#e05d78"
        quad_curve = axes.plot(
            lambda x: x ** 2,
            x_range=[1, 8],
            color=quad_color,
            stroke_width=3,
        )
        quad_label = Text("O(L²)  Attention", font_size=22, color=quad_color)
        quad_label.next_to(axes.c2p(6, 64), UP, buff=0.15)

        self.play(Create(quad_curve))
        self.play(Write(quad_label))
        self.wait(0.5)

        # ── DashedLine at x=6 ────────────────────────────────────────────────
        x_val = 6
        linear_y = x_val        # O(L) → 6
        quad_y = x_val ** 2     # O(L²) → 36  (clamped to axis max 64)
        quad_y_clamped = min(quad_y, 64)

        dashed = DashedLine(
            start=axes.c2p(x_val, 0),
            end=axes.c2p(x_val, quad_y_clamped),
            color=YELLOW,
            stroke_width=1.5,
            dash_length=0.12,
        )
        self.play(Create(dashed))

        gap_label = Text(
            "L=600: 36× ops gap",
            font_size=22,
            color=YELLOW,
        )
        # Position between the two curve heights at x=6
        mid_y = (linear_y + quad_y_clamped) / 2
        gap_label.move_to(axes.c2p(x_val + 0.7, mid_y))
        self.play(Write(gap_label))
        self.wait(0.5)

        # ── Subtitle ─────────────────────────────────────────────────────────
        subtitle = Text(
            "Mamba replaces attention with a recurrent SSM that scales linearly.",
            font_size=22,
            color=LIGHT_GREY,
        )
        subtitle.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(subtitle, shift=UP * 0.1))
        self.wait(2)
