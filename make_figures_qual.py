"""Render foveal crops, saliency stages and persistence traces from recordings.

Requires the datasets used by each figure. Frame-selection criteria are
specified beside the sampling code. Outputs go to report/figures/.

Run from the repository root with ``python make_figures_qual.py``."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

import foveanet as fn
from make_figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, W1, W2, save

OUT = Path("report/figures")
WINDOW = 10_000

# High-luminance overlays contrast with dark event frames; boxes also have labels.
BOX_FOVEA = "#4da6ff"
BOX_ALT = "#ffd166"


def frame_image(xs, ys, shape=(128, 128)):
    img = np.zeros(shape, dtype=np.float32)
    np.add.at(img, (np.clip(ys, 0, shape[0] - 1), np.clip(xs, 0, shape[1] - 1)), 1.0)
    return img


def run_fovea(x, y, t, sensor=(128, 128), window=WINDOW, k=3):
    """Frames and the fovea box on each, exactly as build_crops computes it."""
    pg = fn.PersistentGMM(k=k)
    out = []
    for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window, min_events=20):
        coords = np.column_stack([xs, ys]).astype(np.float32)
        pg.update(coords)
        box = pg.component_box(pg.select(rule="persistence+hyst"), coords)
        out.append((s, xs, ys, box, pg))
    return out


def draw_frame(ax, xs, ys, sensor, box=None, colour=BOX_FOVEA, accum=None):
    img = frame_image(xs, ys, shape=(sensor[1], sensor[0])) if accum is None else accum
    ax.imshow(np.sqrt(img), cmap="inferno", origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_linewidth(0.6)
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=1.4)
        )


# Where the fovea lands, on three datasets ordered by blind-box ratio.


def fig_where_it_lands():
    rows = []

    ds = fn.load_gesture(train=True)
    ev, _ = ds[7]
    x, y, t, _ = fn.unpack(ev)
    rows.append(("DVS128 Gesture", 0.192, "+0.309", (128, 128), run_fovea(x, y, t)))

    from tonic_convert import SLAnimals

    sl = SLAnimals()
    x, y, t, _ = sl.load(sl.data[3])
    rows.append(("SL-Animals", 0.550, "+0.055", (128, 128), run_fovea(x, y, t)))

    import tonic.datasets as td

    lip = td.DVSLip(save_to="data", train=True)
    e, _ = lip[40]
    x = e["x"].astype(np.int64)
    y = e["y"].astype(np.int64)
    t = e["t"].astype(np.int64)
    rows.append(("DVS-Lip", 0.627, "−0.158", (128, 128), run_fovea(x, y, t)))

    ncol = 5
    fig, axes = plt.subplots(3, ncol, figsize=(W2, 4.5))
    for r, (name, ratio, prem, sensor, frames) in enumerate(rows):
        # Frames are sampled across the recording but restricted to those with
        # an event count above the recording's median. Below that the mixture
        # has too little to fit and the box inflates toward the whole frame,
        # which is a real behaviour but not what this figure is about.
        cand = [f for f in frames if f[3] is not None]
        counts = np.array([len(f[1]) for f in cand])
        good = [f for f, n in zip(cand, counts) if n >= np.median(counts)]
        picks = np.linspace(0, len(good) - 1, ncol).astype(int)
        for c, idx in enumerate(picks):
            s, xs, ys, box, _ = good[idx]
            ax = axes[r, c]
            draw_frame(ax, xs, ys, sensor, box)
            if r == 0:
                ax.set_title(f"t = {(s - good[0][0]) / 1e6:.1f} s", fontsize=7, color=INK2, pad=3)
        area = np.mean([(b[3][2] - b[3][0]) * (b[3][3] - b[3][1]) for b in good]) / (
            sensor[0] * sensor[1]
        )
        axes[r, 0].set_ylabel(
            f"{name}\nratio {ratio:.2f}   premium {prem}\nbox {100 * area:.0f}% of frame",
            fontsize=7.5,
            color=INK,
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            linespacing=1.6,
        )
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    save(fig, "fig04_where_the_fovea_lands")


# The saliency map, stage by stage.


def fig_saliency_stages():
    ds = fn.load_gesture(train=True)
    ev, _ = ds[7]
    x, y, t, _ = fn.unpack(ev)
    frames = run_fovea(x, y, t)
    # Show a representative frame: one with a healthy event count where the
    # saliency peak lands inside the chosen box, which is the case on 88% of
    # frames under the persistence rule. Picking one of the other 12% would
    # illustrate a disagreement rather than the method.
    cand = [f for f in frames if f[3] is not None]
    counts = np.array([len(f[1]) for f in cand])
    pick = None
    for f, n in zip(cand, counts):
        if n < np.percentile(counts, 60):
            continue
        sm = f[4].saliency_map(rule="persistence")
        py, px = np.unravel_index(int(sm.argmax()), sm.shape)
        b = f[3]
        if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
            pick = f
            break
    s, xs, ys, box, pg = pick if pick is not None else cand[len(cand) // 2]
    coords = np.column_stack([xs, ys]).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.05))
    titles = ["one 10 ms frame", "mixture components", "persistence-weighted density", "fovea box"]

    draw_frame(axes[0], xs, ys, (128, 128))

    axes[1].imshow(np.sqrt(frame_image(xs, ys)), cmap="gray_r", origin="upper", alpha=0.55)
    labels = pg.model.predict(coords)
    comp_cols = [BLUE, ORANGE, AQUA]
    for cid in range(pg.k):
        pts = coords[labels == cid]
        if len(pts) < 5:
            continue
        axes[1].scatter(pts[:, 0], pts[:, 1], s=0.6, color=comp_cols[cid], linewidths=0, alpha=0.75)
        m = pg.model.means_[cid]
        axes[1].annotate(
            f"p={pg.persistence[cid]:.2f}",
            xy=(m[0], m[1]),
            fontsize=6,
            color=INK,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8),
        )

    smap = pg.saliency_map(rule="persistence")
    axes[2].imshow(smap, cmap="inferno", origin="upper")
    draw_frame(axes[3], xs, ys, (128, 128), box)

    for ax, ti in zip(axes, titles):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 128)
        ax.set_ylim(128, 0)
        ax.set_title(ti, fontsize=7, color=INK2, pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
            sp.set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig05_saliency_stages")


# Persistence traces: continuous, crossing, and not saturating.


def fig_persistence_traces():
    ds = fn.load_gesture(train=True)
    ev, _ = ds[7]
    x, y, t, _ = fn.unpack(ev)
    pg = fn.PersistentGMM()
    chosen = []
    for s_, xs, ys, ts in fn.iter_frames(x, y, t, window=WINDOW, min_events=20):
        pg.update(np.column_stack([xs, ys]).astype(np.float32))
        chosen.append(pg.select(rule="persistence+hyst"))
    H = np.array(pg.history)
    n = min(len(H), 260)
    H, chosen = H[:n], np.array(chosen[:n])
    tt = np.arange(n) * WINDOW / 1e6

    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for i in np.where(np.diff(chosen) != 0)[0]:
        ax.axvline(tt[i], color=GRID, lw=0.8, zorder=1)
    for c, col in zip(range(H.shape[1]), [BLUE, ORANGE, AQUA]):
        ax.plot(tt, H[:, c], color=col, lw=1.6, zorder=3, label=f"component {c}")
    nsw = int((np.diff(chosen) != 0).sum())
    ax.annotate(
        f"vertical rules mark the {nsw} frames where the\n"
        f"fovea changed component ({100 * nsw / (n - 1):.0f}% of frames)",
        xy=(0.03, 0.115),
        fontsize=6.2,
        color=INK2,
        linespacing=1.3,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("accumulated persistence")
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.9, borderpad=0.2, fontsize=6.5
    )
    save(fig, "fig11_persistence_traces")


if __name__ == "__main__":
    print("writing qualitative figures to report/figures/")
    fig_saliency_stages()
    fig_where_it_lands()
    fig_persistence_traces()
