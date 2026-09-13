"""Plot mixture geometry and component identity on a DVS128 Gesture recording.

Requires local gesture data. Outputs are written to report/figures/.
Run from the repository root with ``python make_figures_method.py``."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans

import foveanet as fn
from make_figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, W2, save

OUT = Path("report/figures")
WINDOW = 10_000
COMP = [BLUE, ORANGE, AQUA]
BOX_FOVEA = "#4da6ff"


def busy_frame(x, y, t, index=None):
    """A frame with enough events for every method to have something to fit."""
    frames = [
        (s, xs, ys) for s, xs, ys, ts in fn.iter_frames(x, y, t, window=WINDOW, min_events=20)
    ]
    counts = np.array([len(f[1]) for f in frames])
    pick = int(np.argsort(counts)[int(0.75 * len(counts))]) if index is None else index
    return frames[pick]


def box_fraction(box):
    return max((box[2] - box[0]) * (box[3] - box[1]), 1) / (fn.SENSOR[0] * fn.SENSOR[1])


def blank(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    if title:
        ax.set_title(title, fontsize=7, color=INK2, pad=4)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_linewidth(0.6)


def draw_box(ax, box, colour, label):
    ax.add_patch(
        Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor=colour,
            linewidth=1.3,
        )
    )
    ax.annotate(label, xy=(box[0], box[1] - 3), fontsize=6, color=colour, ha="left", va="bottom")


# --------------------------------------------------------------------------
# 2.2  Why a mixture rather than a simpler clustering.
# --------------------------------------------------------------------------
def fig_mixture_vs_kmeans():
    ds = fn.load_gesture(train=True)
    x, y, t, _ = fn.unpack(ds[7][0])
    _, xs, ys = busy_frame(x, y, t)
    coords = np.column_stack([xs, ys]).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25))

    axes[0].scatter(coords[:, 0], coords[:, 1], s=0.5, color=INK2, linewidths=0, alpha=0.55)
    blank(axes[0], "one 10 ms frame")

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(coords)
    kbig = int(np.bincount(km.labels_).argmax())
    for cid in range(3):
        pts = coords[km.labels_ == cid]
        axes[1].scatter(pts[:, 0], pts[:, 1], s=0.5, color=COMP[cid], linewidths=0, alpha=0.7)
    kbox = fn.fovea_box(coords[km.labels_ == kbig])
    draw_box(axes[1], kbox, INK, f"{box_fraction(kbox):.3f} of sensor")
    blank(axes[1], "k-means")

    axes[2].set_xlim(0, 128)
    axes[2].set_ylim(128, 0)
    pg = fn.PersistentGMM(k=3)
    pg.update(coords)
    resp = pg.model.predict_proba(coords)
    lab = resp.argmax(1)
    for cid in range(3):
        pts = coords[lab == cid]
        axes[2].scatter(
            pts[:, 0],
            pts[:, 1],
            s=0.5,
            color=COMP[cid],
            linewidths=0,
            alpha=0.35 + 0.5 * resp[lab == cid, cid].mean(),
        )
        mu = pg.model.means_[cid]
        cov = pg.model.covariances_[cid]
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        # one standard deviation: the broad component's 2-sigma contour is
        # larger than the sensor itself, which is the point but not drawable.
        el = Ellipse(
            mu,
            2 * np.sqrt(vals[0]),
            2 * np.sqrt(vals[1]),
            angle=ang,
            fill=False,
            edgecolor=COMP[cid],
            linewidth=0.9,
            alpha=0.9,
        )
        axes[2].add_patch(el)
        el.set_clip_path(Rectangle((0, 0), 128, 128, transform=axes[2].transData))
        axes[2].annotate(
            f"\u03c0={pg.model.weights_[cid]:.2f}",
            xy=(np.clip(mu[0], 8, 120), np.clip(mu[1] + 9, 8, 120)),
            fontsize=5.5,
            color=INK,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85),
        )
    gbig = int(pg.model.weights_.argmax())
    gbox = pg.component_box(gbig, coords)
    draw_box(axes[2], gbox, INK, f"{box_fraction(gbox):.3f} of sensor")
    blank(axes[2], "Gaussian mixture")

    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig12_mixture_vs_kmeans")


# --------------------------------------------------------------------------
# 2.3  What carrying the mixture forward buys: component identity.
# --------------------------------------------------------------------------
def fig_identity_across_frames():
    ds = fn.load_gesture(train=True)
    x, y, t, _ = fn.unpack(ds[7][0])
    frames = [
        (s, xs, ys) for s, xs, ys, ts in fn.iter_frames(x, y, t, window=WINDOW, min_events=20)
    ]
    start = len(frames) // 2
    seq = frames[start : start + 6]

    fig, axes = plt.subplots(2, 6, figsize=(W2, 2.9))
    for row, warm in enumerate([False, True]):
        pg = fn.PersistentGMM(k=3, warm=warm)
        # burn in so the warm row is genuinely carried forward, not first-fit
        for _, xs, ys in frames[max(0, start - 12) : start]:
            pg.update(np.column_stack([xs, ys]).astype(np.float32))
        heavies = []
        for col, (_, xs, ys) in enumerate(seq):
            ax = axes[row, col]
            coords = np.column_stack([xs, ys]).astype(np.float32)
            pg.update(coords)
            lab = pg.model.predict(coords)
            for cid in range(3):
                pts = coords[lab == cid]
                if len(pts) < 5:
                    continue
                ax.scatter(pts[:, 0], pts[:, 1], s=0.5, color=COMP[cid], linewidths=0, alpha=0.75)
            heavy = int(pg.model.weights_.argmax())
            heavies.append(heavy)
            ax.annotate(
                f"heaviest: {heavy}",
                xy=(4, 122),
                fontsize=5.5,
                color=COMP[heavy],
                ha="left",
                va="bottom",
            )
            blank(ax, f"frame {col + 1}" if row == 0 else None)
        sw = sum(a != b for a, b in zip(heavies, heavies[1:]))
        axes[row, -1].annotate(
            f"index changes on {sw} of {len(heavies) - 1}",
            xy=(1.04, 0.5),
            xycoords="axes fraction",
            rotation=270,
            fontsize=6,
            color=INK2,
            ha="left",
            va="center",
        )
    axes[0, 0].set_ylabel("fitted\nindependently", fontsize=6.5, color=INK2)
    axes[1, 0].set_ylabel("carried\nforward", fontsize=6.5, color=INK2)
    fig.subplots_adjust(wspace=0.08, hspace=0.10)
    save(fig, "fig13_component_identity")


if __name__ == "__main__":
    fig_mixture_vs_kmeans()
    fig_identity_across_frames()
