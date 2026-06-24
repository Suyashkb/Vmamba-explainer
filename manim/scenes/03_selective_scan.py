"""Scene 03: S4 → S6 Selective Scan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *


def _make_box(label_tex, color, width=1.8, height=0.6):
    """Return a VGroup of RoundedRectangle + centered MathTex label."""
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        color=color,
        stroke_width=2,
    )
    rect.set_fill(color, opacity=0.15)
    lbl = MathTex(label_tex, font_size=28, color=color)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


class SelectiveScanScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text("S4 → S6: Selective Scan", font_size=44, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL: S4 — Static
        # ═══════════════════════════════════════════════════════════════════
        left_title = Text("S4 — Static", font_size=30, color=GREY_B)
        left_title.move_to([-3.5, 2.0, 0])

        left_A = _make_box("A", RED)
        left_B = _make_box("B", BLUE)
        left_C = _make_box("C", BLUE)

        left_stack = VGroup(left_A, left_B, left_C).arrange(DOWN, buff=0.35)
        left_stack.move_to([-3.5, 0.2, 0])

        # "fixed" labels
        fixed_labels = VGroup()
        for box in left_stack:
            lbl = Text("fixed", font_size=17, color=GREY_B)
            lbl.next_to(box, DOWN, buff=0.08)
            fixed_labels.add(lbl)

        # x_t input arrow → B box
        x_t_left = MathTex("x_t", font_size=28, color=GREEN_C)
        x_t_left.next_to(left_B, LEFT, buff=0.8)
        xt_to_B = Arrow(
            x_t_left.get_right(),
            left_B[0].get_left(),
            buff=0.05,
            color=GREEN_C,
            stroke_width=2,
            tip_length=0.15,
        )

        # y_t output from C box
        y_t_left = MathTex("y_t", font_size=28, color=YELLOW_C)
        y_t_left.next_to(left_C, DOWN, buff=0.55)
        C_to_yt = Arrow(
            left_C[0].get_bottom(),
            y_t_left.get_top(),
            buff=0.05,
            color=YELLOW_C,
            stroke_width=2,
            tip_length=0.15,
        )

        left_panel = VGroup(
            left_title, left_stack, fixed_labels,
            x_t_left, xt_to_B, y_t_left, C_to_yt,
        )

        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANEL: S6 — Selective (Mamba)
        # ═══════════════════════════════════════════════════════════════════
        right_title = Text("S6 — Selective (Mamba)", font_size=30, color=GREY_B)
        right_title.move_to([3.0, 2.0, 0])

        right_A = _make_box("A", RED)
        right_Bt = _make_box("B_t", BLUE)
        right_Ct = _make_box("C_t", BLUE)
        right_Delta = _make_box("\\Delta_t", GREEN)

        right_stack = VGroup(right_A, right_Bt, right_Ct, right_Delta).arrange(
            DOWN, buff=0.28
        )
        right_stack.move_to([3.0, 0.0, 0])

        # "input-dependent" labels under Bt, Ct, Delta
        input_dep_labels = VGroup()
        for box in [right_Bt, right_Ct, right_Delta]:
            lbl = Text("input-dependent", font_size=15, color=TEAL)
            lbl.next_to(box, DOWN, buff=0.06)
            input_dep_labels.add(lbl)

        # x_t node
        x_t_right = MathTex("x_t", font_size=28, color=GREEN_C)
        x_t_right.next_to(right_stack, LEFT, buff=1.0)
        x_t_right.align_to(right_Bt, UP)

        # Curved arrows from x_t → B_t, x_t → C_t, x_t → Delta_t
        def curved_arrow_to(source, target, color, angle=0.3):
            arr = CurvedArrow(
                source.get_right(),
                target[0].get_left(),
                angle=angle,
                color=color,
                stroke_width=2,
                tip_length=0.15,
            )
            return arr

        xt_to_Bt = curved_arrow_to(x_t_right, right_Bt, BLUE, angle=0.2)
        xt_to_Ct = curved_arrow_to(x_t_right, right_Ct, BLUE, angle=0.0)
        xt_to_Delta = curved_arrow_to(x_t_right, right_Delta, GREEN, angle=-0.3)

        # y_t output
        y_t_right = MathTex("y_t", font_size=28, color=YELLOW_C)
        y_t_right.next_to(right_Ct, DOWN, buff=0.9)
        Ct_to_yt = Arrow(
            right_Ct[0].get_bottom(),
            y_t_right.get_top(),
            buff=0.05,
            color=YELLOW_C,
            stroke_width=2,
            tip_length=0.15,
        )

        right_panel = VGroup(
            right_title, right_stack, input_dep_labels,
            x_t_right, xt_to_Bt, xt_to_Ct, xt_to_Delta,
            y_t_right, Ct_to_yt,
        )

        # ── Animate left panel ───────────────────────────────────────────────
        self.play(FadeIn(left_panel))
        self.wait(0.5)

        # ── Animate right panel ──────────────────────────────────────────────
        self.play(FadeIn(right_panel))
        self.wait(0.5)

        # ── Matching arrow: left B → right B_t ──────────────────────────────
        match_arrow = CurvedArrow(
            left_B[0].get_right(),
            right_Bt[0].get_left(),
            angle=-0.4,
            color=WHITE,
            stroke_width=1.5,
            tip_length=0.12,
        )
        match_label = Text("static → dynamic", font_size=18, color=WHITE)
        match_label.move_to(match_arrow.get_center() + UP * 0.3)

        self.play(Create(match_arrow), Write(match_label))
        self.wait(0.5)

        # ── Bottom caption ───────────────────────────────────────────────────
        caption = Text(
            "Selectivity: the model can choose what to remember at each step.",
            font_size=21,
            color=LIGHT_GREY,
        )
        caption.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(caption, shift=UP * 0.1))
        self.wait(2)
