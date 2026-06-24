"""Scene 02: SSM Primer — State Space Models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *


class SSMPrimerScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text("State Space Models", font_size=44, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ── Continuous ODE ───────────────────────────────────────────────────
        ode1 = MathTex(r"h'(t) = A\,h(t) + B\,x(t)", font_size=40, color=WHITE)
        ode2 = MathTex(r"y(t) = C\,h(t)", font_size=40, color=WHITE)
        ode_group = VGroup(ode1, ode2).arrange(DOWN, buff=0.4)
        ode_group.move_to(ORIGIN + UP * 0.8)

        self.play(Write(ode1))
        self.play(Write(ode2))
        self.wait(1)

        # ── Discrete ZOH form ────────────────────────────────────────────────
        disc1 = MathTex(
            r"h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t",
            font_size=40,
            color=WHITE,
        )
        disc2 = MathTex(
            r"y_t = C_t\,h_t + D\,x_t",
            font_size=40,
            color=WHITE,
        )
        disc_group = VGroup(disc1, disc2).arrange(DOWN, buff=0.4)
        disc_group.move_to(ORIGIN + UP * 0.8)

        self.play(
            TransformMatchingTex(ode1, disc1),
            TransformMatchingTex(ode2, disc2),
        )
        self.wait(0.5)

        # ── ZOH formulas ─────────────────────────────────────────────────────
        zoh1 = MathTex(
            r"\bar{A}_t = \exp(\Delta_t A)",
            font_size=34,
            color=TEAL,
        )
        zoh2 = MathTex(
            r"\bar{B}_t = \Delta_t B_t",
            font_size=34,
            color=TEAL,
        )
        zoh_group = VGroup(zoh1, zoh2).arrange(RIGHT, buff=1.0)
        zoh_group.move_to(DOWN * 0.6)

        self.play(FadeIn(zoh1, shift=UP * 0.1))
        self.play(FadeIn(zoh2, shift=UP * 0.1))
        self.wait(0.8)

        # ── Fade out equations ───────────────────────────────────────────────
        self.play(FadeOut(disc_group), FadeOut(zoh_group), run_time=0.6)
        self.wait(0.2)

        # ── Unrolled chain ───────────────────────────────────────────────────
        n_tokens = 4
        node_r = 0.28
        h_spacing = 2.2
        chain_x_start = -3.3

        # State nodes h_0..h_3
        h_nodes = VGroup()
        h_labels = VGroup()
        for i in range(n_tokens):
            cx = chain_x_start + i * h_spacing
            cy = 1.2
            circ = Circle(radius=node_r, color=BLUE_C, stroke_width=2)
            circ.set_fill(BLUE_E, opacity=0.5)
            circ.move_to([cx, cy, 0])
            lbl = MathTex(f"h_{i}", font_size=28, color=BLUE_C)
            lbl.move_to(circ.get_center())
            h_nodes.add(circ)
            h_labels.add(lbl)

        # Token nodes x_1..x_4
        x_nodes = VGroup()
        x_labels = VGroup()
        for i in range(n_tokens):
            cx = chain_x_start + i * h_spacing
            cy = -0.8
            circ = Circle(radius=node_r, color=GREEN_C, stroke_width=2)
            circ.set_fill(GREEN_E, opacity=0.5)
            circ.move_to([cx, cy, 0])
            lbl = MathTex(f"x_{i+1}", font_size=28, color=GREEN_C)
            lbl.move_to(circ.get_center())
            x_nodes.add(circ)
            x_labels.add(lbl)

        # Output nodes y_1..y_4
        y_nodes = VGroup()
        y_labels = VGroup()
        for i in range(n_tokens):
            cx = chain_x_start + i * h_spacing
            cy = 2.7
            circ = Circle(radius=node_r, color=YELLOW_C, stroke_width=2)
            circ.set_fill(YELLOW_E, opacity=0.5)
            circ.move_to([cx, cy, 0])
            lbl = MathTex(f"y_{i+1}", font_size=28, color=YELLOW_C)
            lbl.move_to(circ.get_center())
            y_nodes.add(circ)
            y_labels.add(lbl)

        # x_t → h_t arrows (labeled B)
        b_arrows = VGroup()
        b_arrow_labels = VGroup()
        for i in range(n_tokens):
            arr = Arrow(
                x_nodes[i].get_top(),
                h_nodes[i].get_bottom(),
                buff=0.05,
                color=BLUE_B,
                stroke_width=2,
                tip_length=0.15,
            )
            lbl = MathTex("B", font_size=22, color=BLUE_B)
            lbl.next_to(arr, RIGHT, buff=0.05)
            b_arrows.add(arr)
            b_arrow_labels.add(lbl)

        # h_{t-1} → h_t arrows (labeled Ā)
        a_arrows = VGroup()
        a_arrow_labels = VGroup()
        for i in range(1, n_tokens):
            arr = Arrow(
                h_nodes[i - 1].get_right(),
                h_nodes[i].get_left(),
                buff=0.05,
                color=RED_B,
                stroke_width=2,
                tip_length=0.15,
            )
            lbl = MathTex(r"\bar{A}", font_size=22, color=RED_B)
            lbl.next_to(arr, UP, buff=0.05)
            a_arrows.add(arr)
            a_arrow_labels.add(lbl)

        # h_t → y_t arrows (labeled C)
        c_arrows = VGroup()
        c_arrow_labels = VGroup()
        for i in range(n_tokens):
            arr = Arrow(
                h_nodes[i].get_top(),
                y_nodes[i].get_bottom(),
                buff=0.05,
                color=YELLOW_B,
                stroke_width=2,
                tip_length=0.15,
            )
            lbl = MathTex("C", font_size=22, color=YELLOW_B)
            lbl.next_to(arr, RIGHT, buff=0.05)
            c_arrows.add(arr)
            c_arrow_labels.add(lbl)

        # Animate chain assembly with lag
        all_elements = [
            *[VGroup(x_nodes[i], x_labels[i]) for i in range(n_tokens)],
            *[VGroup(h_nodes[i], h_labels[i]) for i in range(n_tokens)],
            *[VGroup(y_nodes[i], y_labels[i]) for i in range(n_tokens)],
            *[VGroup(b_arrows[i], b_arrow_labels[i]) for i in range(n_tokens)],
            *[VGroup(a_arrows[i], a_arrow_labels[i]) for i in range(n_tokens - 1)],
            *[VGroup(c_arrows[i], c_arrow_labels[i]) for i in range(n_tokens)],
        ]

        self.play(
            LaggedStart(
                *[FadeIn(el, shift=UP * 0.15) for el in all_elements],
                lag_ratio=0.08,
                run_time=2.0,
            )
        )
        self.wait(0.5)

        # ── Highlight Δ: large vs small ──────────────────────────────────────
        # Show Δ_t label on arrow a_arrows[1] (h_1 → h_2)
        delta_label = MathTex(r"\Delta_t", font_size=28, color=GREEN)
        delta_label.next_to(a_arrows[1], DOWN, buff=0.1)
        self.play(FadeIn(delta_label, shift=UP * 0.1))

        # "Large Δ → thick arrow (fast forgetting)"
        thick_note = Text(
            "Large Δ: fast forgetting (thick)",
            font_size=20,
            color=GREEN_B,
        )
        thick_note.to_edge(DOWN, buff=0.8)
        self.play(
            a_arrows[1].animate.set_stroke(width=6),
            FadeIn(thick_note),
        )
        self.wait(0.5)

        # "Small Δ → thin arrow"
        thin_note = Text(
            "Small Δ: slow forgetting (thin)",
            font_size=20,
            color=GREEN_D,
        )
        thin_note.to_edge(DOWN, buff=0.8)
        self.play(
            a_arrows[0].animate.set_stroke(width=1),
            Transform(thick_note, thin_note),
        )
        self.wait(2)
