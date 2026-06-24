"""Scene 06: Full VMamba-Tiny Architecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import *


def _arch_block(label, color, width=1.5, height=1.0):
    """Return VGroup(RoundedRectangle, Text label)."""
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        color=color,
        stroke_width=2,
    )
    rect.set_fill(color, opacity=0.18)
    lbl = Text(label, font_size=18, color=color)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


def _downsample_arrow(color=GREY_B):
    """Return a downward-pointing small Arrow as a merge/downsample symbol."""
    arr = Arrow(
        ORIGIN,
        RIGHT * 0.6,
        buff=0,
        color=color,
        stroke_width=2,
        tip_length=0.12,
        max_tip_length_to_length_ratio=0.5,
    )
    return arr


def _feature_square(side, color):
    sq = Square(side_length=side, color=color, stroke_width=1.5)
    sq.set_fill(color, opacity=0.12)
    return sq


class FullVMambaScene(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────────────────────────
        self.camera.background_color = "#0f1117"

        # ── Title ───────────────────────────────────────────────────────────
        title = Text("VMamba-Tiny: Full Architecture", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        # Pipeline blocks — horizontal
        # ════════════════════════════════════════════════════════════════════
        y_pipe = 0.8

        img_block   = _arch_block("Img\n224²",      "#aaaaaa", width=1.1, height=0.9)
        patch_block = _arch_block("PatchEmbed",      GREY_B,   width=1.6, height=0.9)
        s0_block    = _arch_block("Stage 0\n2×VSS\n56²×96",   BLUE_C,  width=1.7, height=1.2)
        ds0_arrow   = Arrow(ORIGIN, RIGHT * 0.55, buff=0, color=GREY_B, stroke_width=2, tip_length=0.12)
        s1_block    = _arch_block("Stage 1\n2×VSS\n28²×192",  BLUE_B,  width=1.7, height=1.2)
        ds1_arrow   = Arrow(ORIGIN, RIGHT * 0.55, buff=0, color=GREY_B, stroke_width=2, tip_length=0.12)
        s2_block    = _arch_block("Stage 2\n9×VSS\n14²×384",  TEAL_C,  width=1.7, height=1.2)
        ds2_arrow   = Arrow(ORIGIN, RIGHT * 0.55, buff=0, color=GREY_B, stroke_width=2, tip_length=0.12)
        s3_block    = _arch_block("Stage 3\n2×VSS\n7²×768",   GREEN_C, width=1.7, height=1.2)
        ln_block    = _arch_block("LN",             GREY_B,   width=0.8, height=0.9)
        head_block  = _arch_block("Head\n1000",     YELLOW_C, width=1.2, height=0.9)

        # Arrange horizontally
        pipeline_items = [
            img_block,
            patch_block,
            s0_block,
            ds0_arrow,
            s1_block,
            ds1_arrow,
            s2_block,
            ds2_arrow,
            s3_block,
            ln_block,
            head_block,
        ]

        pipe_group = VGroup(*pipeline_items).arrange(RIGHT, buff=0.22)
        pipe_group.move_to([0, y_pipe, 0])

        # Animate left → right with Create, each 0.25s apart
        self.play(
            LaggedStart(
                *[
                    (Create(el) if isinstance(el, Arrow) else FadeIn(el, shift=RIGHT * 0.15))
                    for el in pipeline_items
                ],
                lag_ratio=0.18,
                run_time=2.5,
            )
        )
        self.wait(0.3)

        # ── Connecting arrows between blocks ─────────────────────────────────
        block_seq = [img_block, patch_block, s0_block, s1_block, s2_block, s3_block, ln_block, head_block]
        conn_arrows = VGroup()
        for i in range(len(block_seq) - 1):
            a = Arrow(
                block_seq[i][0].get_right(),
                block_seq[i + 1][0].get_left(),
                buff=0.05,
                color=GREY_B,
                stroke_width=1.5,
                tip_length=0.1,
            )
            conn_arrows.add(a)
        self.play(LaggedStart(*[Create(a) for a in conn_arrows], lag_ratio=0.1, run_time=1.0))

        # ── Feature map squares below each stage ─────────────────────────────
        stage_blocks  = [s0_block, s1_block, s2_block, s3_block]
        feat_sizes    = [1.0, 0.7, 0.5, 0.35]   # visual sizes (scaled)
        stage_colors  = [BLUE_C, BLUE_B, TEAL_C, GREEN_C]
        feat_labels   = ["56×56", "28×28", "14×14", "7×7"]

        feat_squares = VGroup()
        feat_label_mobs = VGroup()
        for i, (sb, fs, sc, fl) in enumerate(
            zip(stage_blocks, feat_sizes, stage_colors, feat_labels)
        ):
            sq = _feature_square(fs, sc)
            sq.next_to(sb, DOWN, buff=0.45)
            lbl = Text(fl, font_size=16, color=sc)
            lbl.next_to(sq, DOWN, buff=0.08)
            feat_squares.add(sq)
            feat_label_mobs.add(lbl)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(FadeIn(feat_squares[i]), FadeIn(feat_label_mobs[i]))
                    for i in range(4)
                ],
                lag_ratio=0.2,
                run_time=1.0,
            )
        )
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        # Highlight Stage 2: zoom in
        # ════════════════════════════════════════════════════════════════════
        s2_annotation = Text("9 stacked VSS blocks", font_size=20, color=YELLOW)
        s2_annotation.next_to(s2_block, UP, buff=0.25)
        s2_annotation_arrow = Arrow(
            s2_annotation.get_bottom(),
            s2_block.get_top(),
            buff=0.04,
            color=YELLOW,
            stroke_width=1.5,
            tip_length=0.1,
        )

        self.play(
            self.camera.frame.animate.scale(0.6).move_to(s2_block.get_center() + UP * 0.2),
            run_time=1.0,
        )
        self.play(
            FadeIn(s2_annotation, shift=DOWN * 0.1),
            Create(s2_annotation_arrow),
        )
        self.wait(0.8)
        self.play(
            self.camera.frame.animate.scale(1 / 0.6).move_to(ORIGIN),
            run_time=1.0,
        )
        self.wait(0.3)

        # ── Classification result at Head block ───────────────────────────────
        acc_label = Text("ImageNet top-1: ~82.6%", font_size=20, color=YELLOW)
        acc_label.next_to(head_block, DOWN, buff=0.35)
        self.play(Write(acc_label))
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        # Summary stats panel at bottom
        # ════════════════════════════════════════════════════════════════════
        stats = [
            "Parameters: ~22M",
            "Top-1 Acc: 82.6%",
            "Linear complexity: O(L)",
        ]
        stat_colors = [GREY_B, YELLOW, "#4e9af1"]
        stat_mobs = VGroup()
        for text, color in zip(stats, stat_colors):
            mob = Text(text, font_size=22, color=color)
            stat_mobs.add(mob)
        stat_mobs.arrange(RIGHT, buff=0.8)
        stat_mobs.to_edge(DOWN, buff=0.25)

        self.play(
            LaggedStart(
                *[FadeIn(s, shift=UP * 0.1) for s in stat_mobs],
                lag_ratio=0.3,
                run_time=1.0,
            )
        )
        self.wait(2)
