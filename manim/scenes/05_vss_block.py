"""Scene 05: VSS Block — The Building Unit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *


# ── Helper ────────────────────────────────────────────────────────────────────

def _block(label, color, width=2.0, height=0.55):
    """Return VGroup(RoundedRectangle, Text label)."""
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        color=color,
        stroke_width=2,
    )
    rect.set_fill(color, opacity=0.15)
    lbl = Text(label, font_size=22, color=color)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


def _op_circle(symbol, color=GREY_B, radius=0.22):
    """Return VGroup(Circle, MathTex symbol)."""
    circ = Circle(radius=radius, color=color, stroke_width=2)
    circ.set_fill(BLACK, opacity=0.4)
    sym = MathTex(symbol, font_size=28, color=color)
    sym.move_to(circ.get_center())
    return VGroup(circ, sym)


def _arrow(start, end, color=GREY_B):
    return Arrow(
        start, end,
        buff=0.05,
        color=color,
        stroke_width=2,
        tip_length=0.14,
    )


class VSSBlockScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text("VSS Block: The Building Unit", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        # Build block flowchart (vertical, centered)
        # ════════════════════════════════════════════════════════════════════
        cx = 0.0          # horizontal center of the main spine

        # Vertical positions (top → bottom)
        y_xin     =  3.0
        y_ln      =  2.1
        y_inproj  =  1.1
        y_split   =  0.2   # split level
        y_dwconv  = -0.6   # left branch
        y_ss2d    = -1.5   # left branch
        y_gate    = -0.6   # right branch (SiLU)
        y_mul     = -2.45
        y_outproj = -3.35
        y_add     = -4.1
        y_xout    = -4.85

        left_x  = -1.3
        right_x =  1.3

        # Nodes
        x_in_dot = Dot(point=[cx, y_xin, 0], radius=0.07, color=WHITE)
        x_in_lbl = MathTex("x_{in}", font_size=26, color=WHITE)
        x_in_lbl.next_to(x_in_dot, LEFT, buff=0.15)

        ln_block      = _block("LN",       GREY_B, width=2.0)
        inproj_block  = _block("in_proj ×2", GREY_B, width=2.2)
        dwconv_block  = _block("DWConv",   BLUE_D, width=1.9)
        ss2d_block    = _block("SS2D",     BLUE_D, width=1.9)
        gate_block    = _block("Gate/SiLU", GREEN_D, width=1.9)
        mul_op        = _op_circle(r"\odot", GREY_B)
        outproj_block = _block("out_proj", GREY_B, width=2.0)
        add_op        = _op_circle(r"\oplus", GREY_B)

        x_out_dot = Dot(point=[cx, y_xout, 0], radius=0.07, color=WHITE)
        x_out_lbl = MathTex("x_{out}", font_size=26, color=WHITE)
        x_out_lbl.next_to(x_out_dot, LEFT, buff=0.15)

        # Position blocks
        ln_block.move_to([cx, y_ln, 0])
        inproj_block.move_to([cx, y_inproj, 0])
        dwconv_block.move_to([left_x, y_dwconv, 0])
        ss2d_block.move_to([left_x, y_ss2d, 0])
        gate_block.move_to([right_x, y_gate, 0])
        mul_op.move_to([cx, y_mul, 0])
        outproj_block.move_to([cx, y_outproj, 0])
        add_op.move_to([cx, y_add, 0])

        # Shape annotations
        shape_xin = Text("(B, L, D)", font_size=17, color=GREY_B)
        shape_xin.next_to(x_in_dot, RIGHT, buff=0.1)
        shape_inproj = Text("(B, L, 2D)", font_size=17, color=GREY_B)
        shape_inproj.next_to(inproj_block, RIGHT, buff=0.12)

        # ── All components list (for LaggedStart FadeIn) ──────────────────
        components = [
            VGroup(x_in_dot, x_in_lbl),
            ln_block,
            inproj_block,
            dwconv_block,
            ss2d_block,
            gate_block,
            mul_op,
            outproj_block,
            add_op,
            VGroup(x_out_dot, x_out_lbl),
        ]

        self.play(
            LaggedStart(
                *[FadeIn(c, shift=DOWN * 0.1) for c in components],
                lag_ratio=0.15,
                run_time=2.0,
            )
        )

        # ── Shape annotations ────────────────────────────────────────────────
        self.play(
            FadeIn(shape_xin, shift=LEFT * 0.1),
            FadeIn(shape_inproj, shift=LEFT * 0.1),
        )

        # ── Draw connecting arrows ────────────────────────────────────────────
        arrows = VGroup()

        # x_in → LN
        arrows.add(_arrow(x_in_dot.get_bottom(), ln_block.get_top()))
        # LN → in_proj
        arrows.add(_arrow(ln_block.get_bottom(), inproj_block.get_top()))
        # in_proj → DWConv (left branch)
        arrows.add(_arrow(inproj_block.get_bottom() + LEFT * 0.4, dwconv_block.get_top()))
        # in_proj → Gate (right branch)
        arrows.add(_arrow(inproj_block.get_bottom() + RIGHT * 0.4, gate_block.get_top()))
        # DWConv → SS2D
        arrows.add(_arrow(dwconv_block.get_bottom(), ss2d_block.get_top()))
        # SS2D → ⊙
        arrows.add(_arrow(ss2d_block.get_bottom(), mul_op.get_top() + LEFT * 0.1))
        # Gate → ⊙
        arrows.add(_arrow(gate_block.get_bottom(), mul_op.get_top() + RIGHT * 0.1))
        # ⊙ → out_proj
        arrows.add(_arrow(mul_op.get_bottom(), outproj_block.get_top()))
        # out_proj → ⊕
        arrows.add(_arrow(outproj_block.get_bottom(), add_op.get_top()))
        # ⊕ → x_out
        arrows.add(_arrow(add_op.get_bottom(), x_out_dot.get_top()))

        self.play(
            LaggedStart(
                *[Create(a) for a in arrows],
                lag_ratio=0.1,
                run_time=1.5,
            )
        )
        self.wait(0.3)

        # ── Callout arrow: SS2D → "16-dim state per channel" ─────────────────
        callout_label = Text(
            "16-dim state per channel",
            font_size=19,
            color=YELLOW,
        )
        callout_label.move_to([left_x - 2.0, y_ss2d, 0])
        callout_arrow = Arrow(
            callout_label.get_right(),
            ss2d_block.get_left(),
            buff=0.05,
            color=YELLOW,
            stroke_width=1.5,
            tip_length=0.12,
        )
        self.play(FadeIn(callout_label, shift=RIGHT * 0.1), Create(callout_arrow))
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        # Animate token dot flowing through the graph
        # ════════════════════════════════════════════════════════════════════
        token_dot = Dot(radius=0.1, color=YELLOW)
        token_dot.move_to(x_in_dot.get_center())
        gate_dot = Dot(radius=0.1, color=GREEN)
        gate_dot.move_to(inproj_block.get_center())

        self.play(FadeIn(token_dot))

        # x_in → LN → in_proj
        self.play(token_dot.animate.move_to(ln_block.get_center()), run_time=0.4, rate_func=smooth)
        self.play(token_dot.animate.move_to(inproj_block.get_center()), run_time=0.4, rate_func=smooth)

        # Split: token_dot left branch, gate_dot right branch
        self.play(FadeIn(gate_dot))
        self.play(
            token_dot.animate.move_to(dwconv_block.get_center()),
            gate_dot.animate.move_to(gate_block.get_center()),
            run_time=0.5,
            rate_func=smooth,
        )
        self.play(
            token_dot.animate.move_to(ss2d_block.get_center()),
            run_time=0.4,
            rate_func=smooth,
        )
        self.play(
            token_dot.animate.move_to(mul_op.get_center() + LEFT * 0.1),
            gate_dot.animate.move_to(mul_op.get_center() + RIGHT * 0.1),
            run_time=0.5,
            rate_func=smooth,
        )

        # Merge at ⊙
        self.play(
            token_dot.animate.move_to(mul_op.get_center()),
            FadeOut(gate_dot),
            run_time=0.3,
        )
        self.play(token_dot.animate.move_to(outproj_block.get_center()), run_time=0.4, rate_func=smooth)
        self.play(token_dot.animate.move_to(add_op.get_center()), run_time=0.4, rate_func=smooth)
        self.play(token_dot.animate.move_to(x_out_dot.get_center()), run_time=0.4, rate_func=smooth)
        self.play(FadeOut(token_dot))

        self.wait(2)
