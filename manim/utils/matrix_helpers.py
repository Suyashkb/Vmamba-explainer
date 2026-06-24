"""Matrix and SSM label utility helpers for VMamba explainer scenes."""

from manim import *


def ssm_label_group(name, shape_str, color):
    """Return a VGroup of a colored Text(name) + small Text(shape_str) below it."""
    name_mob = Text(name, color=color, font_size=28)
    shape_mob = Text(shape_str, color=GREY_B, font_size=18)
    shape_mob.next_to(name_mob, DOWN, buff=0.08)
    group = VGroup(name_mob, shape_mob)
    return group


def _make_placeholder_matrix(label_tex, d_show, n_show, label_color):
    """Internal helper: build a Matrix of '·' with a bold label above."""
    entries = [["\\cdot"] * n_show for _ in range(d_show)]
    mat = Matrix(entries, element_alignment_corner=ORIGIN)
    mat.set_color(GREY_B)

    label = MathTex(label_tex, color=label_color, font_size=32)
    label.next_to(mat, UP, buff=0.15)

    group = VGroup(label, mat)
    return group


def ssm_matrices(d_show=3, n_show=3):
    """Return dict with keys 'A','B','C','Delta','D' mapping to VGroups.

    Each VGroup contains a bold label above a Matrix of '·' placeholders.
    Colors: A=RED, B/C=BLUE, Delta=GREEN, D=GREY.
    """
    specs = {
        "A": ("A", RED),
        "B": ("B", BLUE),
        "C": ("C", BLUE),
        "Delta": ("\\Delta", GREEN),
        "D": ("D", GREY),
    }
    result = {}
    for key, (tex, color) in specs.items():
        result[key] = _make_placeholder_matrix(tex, d_show, n_show, color)
    return result


def discretized_matrices(d_show=3, n_show=3):
    """Return dict with keys 'Abar','Bbar' with overline notation in labels.

    Colors: Abar=RED, Bbar=BLUE.
    """
    specs = {
        "Abar": ("\\bar{A}", RED),
        "Bbar": ("\\bar{B}", BLUE),
    }
    result = {}
    for key, (tex, color) in specs.items():
        result[key] = _make_placeholder_matrix(tex, d_show, n_show, color)
    return result
