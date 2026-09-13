"""Draw report figures from recorded experiment summaries.

Constants are traced to runs in FOVEANET_LOG.txt and report/figures/CAPTIONS.md.
This script renders recorded results; it does not rerun the experiments.

Run from the repository root with ``python make_figures.py``."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e5e4df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# IEEE two-column: 3.5in single, 7.16in double.
W1, W2 = 3.5, 7.16

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    }
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# Figure 1. Accuracy against the fraction of the sensor kept, placed two ways.

SWEEP_FOVEA = [
    (0.030, 0.779),
    (0.099, 0.817),
    (0.237, 0.869),
    (0.436, 0.889),
    (0.595, 0.868),
    (1.000, 0.861),
]
SWEEP_CENTRE = [
    (0.030, 0.472),
    (0.100, 0.655),
    (0.250, 0.746),
    (0.500, 0.843),
    (0.750, 0.849),
    (1.000, 0.861),
]
FULL_ACC = 0.861


def fig_sweep():
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    fx, fy = map(np.array, zip(*SWEEP_FOVEA))
    cx, cy = map(np.array, zip(*SWEEP_CENTRE))

    # Shade the placement premium between equal-area conditions.
    grid = np.linspace(0.03, 1.0, 200)
    fi, ci = np.interp(grid, fx, fy), np.interp(grid, cx, cy)
    ax.fill_between(grid, ci, fi, color=BLUE, alpha=0.10, lw=0, zorder=1)

    ax.axhline(FULL_ACC, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.plot(
        cx,
        cy,
        "-o",
        color=ORANGE,
        label="centre-placed",
        zorder=3,
        markeredgecolor=SURFACE,
        markeredgewidth=1.1,
        markersize=4.5,
    )
    ax.plot(
        fx,
        fy,
        "-o",
        color=BLUE,
        label="fovea-placed",
        zorder=4,
        markeredgecolor=SURFACE,
        markeredgewidth=1.1,
        markersize=4.5,
    )

    ax.annotate(
        "full sensor",
        xy=(1.0, FULL_ACC),
        xytext=(-2, -11),
        textcoords="offset points",
        ha="right",
        color=INK2,
        fontsize=7,
    )
    ax.annotate(
        "fovea-placed",
        xy=(0.72, 0.8645),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        color=BLUE,
        fontsize=8,
        fontweight="bold",
    )
    ax.annotate(
        "centre-placed",
        xy=(0.250, 0.746),
        xytext=(8, -14),
        textcoords="offset points",
        color=ORANGE,
        fontsize=8,
        fontweight="bold",
    )
    ax.annotate(
        "peak +0.032 over the\nfull sensor (10 seeds, two batches)",
        xy=(0.436, 0.889),
        xytext=(0.055, 0.952),
        fontsize=6.5,
        color=INK2,
        ha="left",
        linespacing=1.35,
        arrowprops=dict(
            arrowstyle="->",
            color=INK2,
            lw=0.7,
            shrinkA=2,
            shrinkB=3,
            connectionstyle="arc3,rad=-0.2",
        ),
    )
    ax.annotate(
        "placement\npremium",
        xy=(0.145, 0.70),
        fontsize=6.5,
        color=BLUE,
        ha="center",
        linespacing=1.3,
    )

    ax.set_xlabel("fraction of the sensor kept (achieved)")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(-0.01, 1.06)
    ax.set_ylim(0.43, 0.99)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.legend(loc="lower right", ncol=1, handlelength=1.6, borderpad=0.2, labelspacing=0.25)
    save(fig, "fig01_sensor_fraction_sweep")


# Figure 2. The diagnostic predicts the placement premium.

DIAG = [
    ("DVS128 Gesture", 0.192, +0.309, "left", (8, -3)),
    ("SL-Animals", 0.550, +0.055, "right", (-8, 6)),
    ("UCF-50", 0.592, -0.141, "left", (9, 3)),
    ("DVS-Lip", 0.627, -0.158, "left", (9, -2)),
]


def fig_diagnostic():
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, lw=0.9, zorder=3)

    r = np.array([d[1] for d in DIAG])
    p = np.array([d[2] for d in DIAG])
    ax.plot(r, p, "-", color=MUTED, lw=1.0, zorder=2, alpha=0.7)
    for name, ratio, prem, ha, off in DIAG:
        col = BLUE if prem > 0 else ORANGE
        ax.plot(
            ratio,
            prem,
            "o",
            color=col,
            zorder=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            markersize=7,
        )
        ax.annotate(
            name,
            xy=(ratio, prem),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va="center",
            color=INK,
            fontsize=7.5,
        )

    ax.annotate(
        "fovea beats a\nfixed crop",
        xy=(0.20, 0.175),
        fontsize=6.5,
        color=BLUE,
        style="italic",
        linespacing=1.3,
    )
    ax.annotate("fixed crop wins", xy=(0.20, -0.115), fontsize=6.5, color=ORANGE, style="italic")
    ax.annotate(
        "failure predicted before training",
        xy=(0.627, -0.143),
        xytext=(0.375, -0.183),
        fontsize=6.5,
        color=INK2,
        ha="left",
        arrowprops=dict(
            arrowstyle="->",
            color=INK2,
            lw=0.7,
            shrinkA=2,
            shrinkB=4,
            connectionstyle="arc3,rad=0.25",
        ),
    )

    ax.set_xlabel("blind-box ratio, measured before any training")
    ax.set_ylabel("placement premium\n(fovea \u2212 fixed crop)")
    ax.set_xlim(0.12, 0.76)
    ax.set_ylim(-0.22, 0.40)
    save(fig, "fig02_diagnostic_predicts_outcome")


# Figure 3. The same comparison on four datasets.

PANELS = [
    (
        "DVS128 Gesture",
        0.192,
        {"full sensor": 0.861, "fovea crop": 0.839, "centre crop": 0.530, "random crop": 0.345},
    ),
    (
        "SL-Animals",
        0.550,
        {"full sensor": 0.795, "fovea crop": 0.522, "centre crop": 0.467, "random crop": 0.249},
    ),
    (
        "UCF-50",
        0.592,
        {"full sensor": 0.679, "fovea crop": 0.499, "centre crop": 0.640, "random crop": 0.541},
    ),
    (
        "DVS-Lip",
        0.627,
        {"full sensor": 0.516, "fovea crop": 0.334, "centre crop": 0.492, "random crop": 0.354},
    ),
]
ORDER = ["full sensor", "fovea crop", "centre crop", "random crop"]
COLS = {"full sensor": INK2, "fovea crop": BLUE, "centre crop": ORANGE, "random crop": MUTED}


def fig_four_datasets():
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.15), sharey=True)
    ys = np.arange(len(ORDER))[::-1]
    for ax, (name, ratio, vals) in zip(axes, PANELS):
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for y, cond in zip(ys, ORDER):
            v = vals[cond]
            ax.plot([0, v], [y, y], color=GRID, lw=2.2, zorder=2, solid_capstyle="round")
            ax.plot(
                v,
                y,
                "o",
                color=COLS[cond],
                zorder=4,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
            )
            ax.annotate(
                f"{v:.2f}",
                xy=(v, y),
                xytext=(7, -2.5),
                textcoords="offset points",
                fontsize=6.5,
                color=INK2,
            )
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("accuracy", fontsize=7.5)
        ax.set_title(f"{name}\nblind-box ratio {ratio:.2f}", fontsize=8, color=INK, linespacing=1.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(ORDER, fontsize=7.5)
    for lbl, cond in zip(axes[0].get_yticklabels(), ORDER):
        lbl.set_color(COLS[cond] if cond in ("fovea crop", "centre crop") else INK2)
    axes[0].set_ylim(-0.6, len(ORDER) - 0.4)
    save(fig, "fig03_four_datasets")


# Figure 6. Warm-starting is what makes short frames usable.

WINDOWS = [1, 5, 20, 100]
WARM_DWELL = [0.43, 0.23, 0.81, 1.27]
COLD_DWELL = [0.00, 0.01, 0.04, 0.17]


def fig_warm_cold():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    xs = np.arange(len(WINDOWS))
    w = 0.36
    # Label bars directly as well as distinguishing them by colour.
    ax.bar(xs - w / 2, WARM_DWELL, w, color=BLUE, zorder=3, label="carried forward")
    ax.bar(xs + w / 2, COLD_DWELL, w, color=ORANGE, zorder=3, label="fitted independently")
    for x, v in zip(xs - w / 2, WARM_DWELL):
        ax.annotate(
            f"{v:.2f}",
            xy=(x, v),
            xytext=(0, 3),
            fontsize=7,
            textcoords="offset points",
            ha="center",
            color=INK2,
        )
    for x, v in zip(xs + w / 2, COLD_DWELL):
        ax.annotate(
            f"{v:.2f}",
            xy=(x, v),
            xytext=(0, 3),
            fontsize=7,
            textcoords="offset points",
            ha="center",
            color=INK2,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v} ms" for v in WINDOWS])
    ax.set_xlabel("frame length")
    ax.set_ylabel("fovea dwell (s)")
    ax.set_ylim(0, 1.62)
    # Leave room above the bars for the annotation.
    ax.annotate(
        "at 1 ms a per-frame fit changes\nfovea identity 563 times a second",
        xy=(0.20, 0.055),
        xytext=(0.46, 0.95),
        fontsize=7,
        color=INK2,
        linespacing=1.4,
        ha="left",
        arrowprops=dict(
            arrowstyle="->",
            color=MUTED,
            lw=0.8,
            shrinkA=3,
            shrinkB=4,
            connectionstyle="arc3,rad=0.32",
        ),
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.6,
        fontsize=7.5,
    )
    save(fig, "fig06_warm_vs_cold")


# Figure 7. All conditions on DVS128 Gesture.

COND = [
    ("fovea + periphery", 0.884, 100, AQUA),
    ("full sensor", 0.861, 100, INK2),
    ("size  (instantaneous)", 0.851, 3, BLUE),
    ("persistence (accumulated)", 0.839, 3, BLUE),
    ("density  (instantaneous)", 0.808, 2, MUTED),
    ("persdens (accumulated)", 0.805, 2, MUTED),
    ("centre crop", 0.530, 3, ORANGE),
    ("random crop", 0.345, 3, MUTED),
]


def fig_conditions():
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ys = np.arange(len(COND))[::-1]
    for y, (name, acc, kept, col) in zip(ys, COND):
        ax.plot([0, acc], [y, y], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
        ax.plot(
            acc,
            y,
            "o",
            color=col,
            markersize=6.5,
            zorder=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
        )
        ax.annotate(
            f"{acc:.3f}",
            xy=(acc, y),
            xytext=(7, -2.5),
            textcoords="offset points",
            fontsize=6.5,
            color=INK2,
        )

    ax.axvline(0.861, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c[0]}" for c in COND], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("test accuracy")
    ax.set_ylim(-0.6, len(COND) - 0.4)
    save(fig, "fig07_all_conditions")


# Figure 8. Why the tracker fails: it moves, but not with the target.

TRACKERS = [
    ("target's own motion", 0.667, AQUA),
    ("density (original rule)", 0.022, MUTED),
    ("nearest + velocity", 0.038, MUTED),
    ("persistence", 0.038, MUTED),
    ("SNN appearance", 0.064, BLUE),
]


def fig_direction():
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ys = np.arange(len(TRACKERS))[::-1]
    for y, (name, v, col) in zip(ys, TRACKERS):
        ax.plot([0, v], [y, y], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
        ax.plot(
            v,
            y,
            "o",
            color=col,
            markersize=6.5,
            zorder=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
        )
        ax.annotate(
            f"{v:.3f}",
            xy=(v, y),
            xytext=(7, -2.5),
            textcoords="offset points",
            fontsize=6.5,
            color=INK2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([t[0] for t in TRACKERS], fontsize=7.5)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("direction coherence with the target")
    ax.set_ylim(-0.6, len(TRACKERS) - 0.4)
    ax.annotate(
        "what is available",
        xy=(0.667, 4),
        xytext=(-6, 11),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color=AQUA,
    )
    ax.annotate("what every method extracted", xy=(0.16, 1.75), fontsize=6.5, color=INK2)
    save(fig, "fig08_direction_coherence")


# Figure 9. Appearance carries identity, but only over one frame.

APPEAR = [
    ("same frame\n(positive control)", 0.969, 0.8),
    ("previous frame", 0.796, 5.6),
    ("frame zero\n(as implemented)", 0.033, 45.8),
]


def fig_appearance():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.2))
    ys = np.arange(len(APPEAR))[::-1]
    cols = [MUTED, BLUE, ORANGE]
    for ax, vals, lab, xlim in (
        (a1, [a[1] for a in APPEAR], "correlation at the target", 1.05),
        (a2, [a[2] for a in APPEAR], "% of positions scoring higher", 55),
    ):
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for y, v, c in zip(ys, vals, cols):
            ax.plot([0, v], [y, y], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
            ax.plot(
                v,
                y,
                "o",
                color=c,
                markersize=6,
                zorder=4,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
            )
            ax.annotate(
                f"{v:g}",
                xy=(v, y),
                xytext=(6, -2.5),
                fontsize=6.5,
                textcoords="offset points",
                color=INK2,
            )
        ax.set_xlim(0, xlim)
        ax.set_xlabel(lab, fontsize=7)
        ax.set_ylim(-0.6, len(APPEAR) - 0.4)
        ax.set_yticks(ys)
    a1.set_yticklabels([a[0] for a in APPEAR], fontsize=7, linespacing=1.3)
    a2.set_yticklabels([])
    a2.axvline(50, color=ORANGE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    a2.annotate(
        "chance",
        xy=(50, 2.3),
        xytext=(-4, 0),
        ha="right",
        textcoords="offset points",
        fontsize=6,
        color=ORANGE,
    )
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig09_appearance_controls")


# Figure 10. The method as a pipeline.


def fig_pipeline():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(W2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 112)
    ax.axis("off")

    BW, BH = 29.0, 25.0
    COL_X = [2.0, 35.5, 69.0]
    ROW_Y = [60.0, 12.0]

    # input stages are neutral, the model is blue, what it emits is orange.
    stages = [
        (0, 0, "event stream", MUTED, ["asynchronous", "(x, y, t, p)", "no frames"]),
        (
            0,
            1,
            "10 ms frame",
            MUTED,
            ["events in [s, s+10 ms)", "at least 20 events", "no accumulation"],
        ),
        (
            0,
            2,
            "Gaussian mixture",
            BLUE,
            ["K = 3 components", "reg_covar = 1 px\u00b2", "warm-started"],
        ),
        (
            1,
            0,
            "persistence",
            BLUE,
            [
                "\u03c1 \u2190 \u03bb\u03c1 + (1\u2212\u03bb)\u03c0",
                "one score per component",
                "\u03bb = 0.85",
            ],
        ),
        (
            1,
            1,
            "saliency map",
            ORANGE,
            [
                "S(u) = \u03a3 \u03c1 N(u | \u03bc, \u03a3)",
                "the mixture density,",
                "weighted by \u03c1",
            ],
        ),
        (
            1,
            2,
            "fovea box",
            ORANGE,
            ["the selected component,", "5th to 95th percentile", "hysteresis 1.15"],
        ),
    ]

    for row, col, title, colour, lines in stages:
        x, y = COL_X[col], ROW_Y[row]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                BW,
                BH,
                boxstyle="round,pad=0.4,rounding_size=1.4",
                linewidth=1.0,
                edgecolor=colour,
                facecolor=colour,
                alpha=0.10,
                zorder=2,
            )
        )
        ax.text(
            x + BW / 2,
            y + BH - 5.5,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
            zorder=3,
        )
        for i, ln in enumerate(lines):
            ax.text(
                x + BW / 2,
                y + BH - 12.0 - i * 5.0,
                ln,
                ha="center",
                va="center",
                fontsize=6.2,
                color=INK2,
                zorder=3,
            )

    # flow within each row
    for row in (0, 1):
        for col in (0, 1):
            x0 = COL_X[col] + BW + 0.4
            ax.add_patch(
                FancyArrowPatch(
                    (x0, ROW_Y[row] + BH / 2),
                    (x0 + 3.3, ROW_Y[row] + BH / 2),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=INK2,
                    lw=1.0,
                    zorder=3,
                )
            )

    # wrap from the end of row 1 to the start of row 2
    xa, xb = COL_X[2] + BW / 2, COL_X[0] + BW / 2
    ax.plot(
        [xa, xa, xb, xb],
        [ROW_Y[0], 51.0, 51.0, ROW_Y[1] + BH + 3.2],
        color=INK2,
        lw=1.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.add_patch(
        FancyArrowPatch(
            (xb, ROW_Y[1] + BH + 3.2),
            (xb, ROW_Y[1] + BH + 0.4),
            arrowstyle="-|>",
            mutation_scale=10,
            color=INK2,
            lw=1.0,
            zorder=3,
        )
    )

    # Carry mixture parameters forward before updating persistence.
    mx = COL_X[2]
    ax.add_patch(
        FancyArrowPatch(
            (mx + BW * 0.82, ROW_Y[0] + BH + 0.5),
            (mx + BW * 0.18, ROW_Y[0] + BH + 0.5),
            connectionstyle="arc3,rad=0.28",
            arrowstyle="-|>",
            mutation_scale=10,
            color=BLUE,
            lw=1.3,
            zorder=4,
        )
    )
    # Keep the label inside the canvas and clear of the return arrow.
    ax.text(
        97.0,
        108.5,
        "means, weights and precisions carried forward",
        ha="right",
        va="center",
        fontsize=6.8,
        color=BLUE,
    )
    ax.text(
        97.0,
        103.5,
        "component identity survives, so a per-component",
        ha="right",
        va="center",
        fontsize=6.0,
        color=INK2,
        style="italic",
    )
    ax.text(
        97.0,
        99.7,
        "score accumulated over time is definable",
        ha="right",
        va="center",
        fontsize=6.0,
        color=INK2,
        style="italic",
    )
    save(fig, "fig10_pipeline")


if __name__ == "__main__":
    print("writing figures to report/figures/")
    fig_sweep()
    fig_diagnostic()
    fig_four_datasets()
    fig_warm_cold()
    fig_conditions()
    fig_direction()
    fig_appearance()
    fig_pipeline()
