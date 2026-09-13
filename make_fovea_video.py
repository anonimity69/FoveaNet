"""Animate raw events, persistence-weighted saliency and accumulated saliency.

The foveal box is drawn on the first two panels. Outputs go to figures/.

Examples:
    python make_fovea_video.py --index 42
    python make_fovea_video.py --rule persistence+hyst --fps 12"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

import foveanet as fn

try:
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False


def build(index, rule, decay):
    dataset = fn.load_gesture()
    events, label = dataset[index]
    x, y, t, _ = fn.unpack(events)

    pg = fn.PersistentGMM(decay=decay)
    accum = np.zeros(fn.SENSOR)
    steps = []
    held = 0
    previous = None

    for s, xs, ys, ts in fn.iter_frames(x, y, t):
        coords = np.column_stack([xs, ys]).astype(np.float32)
        pg.update(coords)
        cid = pg.select(rule=rule)
        box = pg.component_box(cid, coords)
        if box is None:
            continue
        smap = pg.saliency_map(rule="persistence")
        accum = decay * accum + (1 - decay) * fn.normalise(smap)
        held = held + 1 if cid == previous else 1
        previous = cid
        steps.append(
            {
                "ms": (s - t.min()) / 1000.0,
                "events": fn.frame_image(xs, ys),
                "saliency": smap,
                "accum": accum.copy(),
                "box": box,
                "cid": cid,
                "persistence": float(pg.persistence[cid]),
                "held": held,
            }
        )
    return steps, label, len(dataset)


def render(steps, label, out, fps):
    ev_top = np.percentile([s["events"].max() for s in steps], 90)
    sal_top = np.percentile([s["saliency"].max() for s in steps], 97)
    acc_top = np.percentile([s["accum"].max() for s in steps], 97)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    fig.patch.set_facecolor("white")
    titles = ["events", "persistence-weighted saliency", "saliency accumulated"]
    ims, rects = [], []
    for ax, title, (data, top, cmap) in zip(
        axes,
        titles,
        [
            (steps[0]["events"], ev_top, "hot"),
            (steps[0]["saliency"], sal_top, "magma"),
            (steps[0]["accum"], acc_top, "magma"),
        ],
    ):
        ims.append(ax.imshow(data, cmap=cmap, vmin=0, vmax=top, interpolation="nearest"))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[:2]:
        r = patches.Rectangle((0, 0), 1, 1, lw=2, edgecolor="cyan", facecolor="none")
        ax.add_patch(r)
        rects.append(r)

    banner = fig.suptitle("", fontsize=11, family="monospace")
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    def frame(i):
        st = steps[i]
        ims[0].set_data(st["events"])
        ims[1].set_data(st["saliency"])
        ims[2].set_data(st["accum"])
        bx0, by0, bx1, by1 = st["box"]
        for r in rects:
            r.set_bounds(bx0, by0, bx1 - bx0, by1 - by0)
        banner.set_text(
            f"class {label}   t = {st['ms']:6.0f} ms   component {st['cid']}   "
            f"persistence {st['persistence']:.2f}   held {st['held']:3d} frames "
            f"({st['held'] * fn.WINDOW_US / 1e6:.1f} s)"
        )
        return ims + rects + [banner]

    anim = FuncAnimation(fig, frame, frames=len(steps), interval=1000 / fps, blit=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if out.endswith(".mp4"):
        if not HAVE_FFMPEG:
            raise SystemExit("mp4 needs ffmpeg: pip install imageio-ffmpeg, or ask for a .gif")
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=110)
    else:
        anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--rule", default="persdens+hyst")
    ap.add_argument("--decay", type=float, default=fn.DECAY)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ext = "mp4" if HAVE_FFMPEG else "gif"
    out = args.out or f"figures/fovea_recording_{args.index}.{ext}"
    steps, label, total = build(args.index, args.rule, args.decay)
    ids = [s["cid"] for s in steps]
    mean_run, longest, switch = fn.dwell_stats(ids)
    secs = fn.WINDOW_US / 1e6

    print(f"recording {args.index} of {total}, class {label}, rule {args.rule}")
    print(f"frames: {len(steps)}  ({len(steps) * secs:.1f} s)")
    print(
        f"mean dwell {mean_run:.1f} frames ({mean_run * secs:.2f} s), "
        f"longest {longest} ({longest * secs:.2f} s), "
        f"switches on {100 * switch:.1f}% of frames"
    )
    render(steps, label, out, args.fps)
    print("wrote", out)


if __name__ == "__main__":
    main()
