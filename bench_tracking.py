"""Compare the local GMM tracker with a static-box baseline.

Measures overlap and target-event share on VOT and TrackingDataset recordings.
Run from the repository root with ``python bench_tracking.py``."""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import evaluate as ev
import tracker
from tonic_convert import DVSBenchmark

VOT = "data/DVSBENCH/INI_VOT_30fps_20160610.hdf5"
TRK = "data/DVSBENCH/INI_TrackingDataset_30fps_20160610.hdf5"
WINDOW = 33_333


def static_score(b, t0, t1, window, ceilings=None):
    first = ev.gt_box(b[0])
    preds, truths = [], []
    for s in np.arange(t0, t1, window):
        preds.append(first)
        truths.append(ev.gt_at(b, s + window / 2))
    return tracker.score(preds, truths, ceilings=ceilings)


def ceilings_for(b, t0, t1, window):
    """Per-frame best overlap achievable by the box shape both methods are given.

    The tracker freezes its box at the size of the first annotation and the
    static baseline never moves off it, so the two share one ceiling and the
    relative measure divides both by the same number.
    """
    first = ev.gt_box(b[0])
    w, h = first[2] - first[0], first[3] - first[1]
    return np.array(
        [ev.max_iou_quad(ev.gt_quad_at(b, s + window / 2), w, h) for s in np.arange(t0, t1, window)]
    )


def in_window_share(x, y, t, b, window, scale=tracker.SEARCH_SCALE, sensor=(240, 180)):
    """Of the events the tracker actually sees, how many are on the target."""
    shares = []
    for s in np.arange(t.min(), t.max(), window):
        sel = (t >= s) & (t < s + window)
        if int(sel.sum()) < tracker.MIN_EVENTS:
            continue
        xs, ys = x[sel], y[sel]
        truth = ev.gt_at(b, s + window / 2)
        sw = tracker._search_window(truth, scale, sensor)
        near = (xs >= sw[0]) & (xs <= sw[2]) & (ys >= sw[1]) & (ys <= sw[3])
        if near.sum() < tracker.MIN_EVENTS:
            continue
        inb = (
            (xs[near] >= truth[0])
            & (xs[near] <= truth[2])
            & (ys[near] >= truth[1])
            & (ys[near] <= truth[3])
        )
        shares.append(inb.mean())
    return float(np.mean(shares)) if shares else 0.0


def run(h5, names, label, sensor=(240, 180)):
    print(f"\n=== {label}: {len(names)} sequences ===", flush=True)
    rows = []
    start = time.time()
    for i, nm in enumerate(names):
        e, b = DVSBenchmark(h5, sequences=[nm])[0]
        x, y, t = e["x"], e["y"], e["t"]
        ceil = ceilings_for(b, t.min(), t.max(), WINDOW)
        st = static_score(b, t.min(), t.max(), WINDOW, ceilings=ceil)
        p, g, _ = tracker.track(x, y, t, b, window=WINDOW, alpha=0.3, sensor=sensor)
        tr = tracker.score(p, g, ceilings=ceil)
        share = in_window_share(x, y, t, b, WINDOW, sensor=sensor)
        rows.append(
            (
                nm,
                st["mean_iou"],
                tr["mean_iou"],
                tr["success_50"],
                share,
                tr["mean_ceiling"],
                st["relative_iou"],
                tr["relative_iou"],
            )
        )
        if (i + 1) % 10 == 0:
            el = time.time() - start
            print(
                f"  {i + 1}/{len(names)}  {el:.0f}s elapsed, "
                f"{el / (i + 1) * (len(names) - i - 1):.0f}s left",
                flush=True,
            )

    a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows])
    print(f"\n{label} summary")
    print(
        f"  achievable ceiling         {a[:, 4].mean():.3f}   (upright fixed-size box vs rotated quad)"
    )
    print(f"                     {'raw IoU':>12}{'relative':>12}")
    print(f"  static box       {a[:, 0].mean():>12.3f}{a[:, 5].mean():>12.3f}")
    print(f"  local tracker    {a[:, 1].mean():>12.3f}{a[:, 6].mean():>12.3f}")
    print(
        f"  difference       {a[:, 1].mean() - a[:, 0].mean():>+12.3f}{a[:, 6].mean() - a[:, 5].mean():>+12.3f}"
    )
    print(
        f"  tracker wins on            {int((a[:, 1] > a[:, 0]).sum())}/{len(rows)} sequences raw, "
        f"{int((a[:, 6] > a[:, 5]).sum())}/{len(rows)} relative"
    )
    print(f"  tracker success@0.5        {100 * a[:, 2].mean():.1f}%")
    print(
        f"  sequences above 0.5 IoU    {int((a[:, 1] > 0.5).sum())}/{len(rows)} raw, "
        f"{int((a[:, 6] > 0.5).sum())}/{len(rows)} relative"
    )
    print(f"  events on target, of those the tracker sees: {100 * a[:, 3].mean():.0f}%")

    print("\n  sequences where the ceiling is lowest (annotation shape hurts most)")
    print(f"  {'sequence':>22}{'ceiling':>9}{'static':>9}{'tracker':>9}{'rel.trk':>9}")
    for nm, s, tk, _, _, c, rs, rt in sorted(rows, key=lambda r: r[5])[:5]:
        print(f"  {nm[-22:]:>22}{c:>9.3f}{s:>9.3f}{tk:>9.3f}{rt:>9.3f}")

    print("\n  best 5 by relative tracker IoU")
    print(f"  {'sequence':>22}{'ceiling':>9}{'static':>9}{'tracker':>9}{'rel.trk':>9}")
    for nm, s, tk, _, _, c, rs, rt in sorted(rows, key=lambda r: -r[7])[:5]:
        print(f"  {nm[-22:]:>22}{c:>9.3f}{s:>9.3f}{tk:>9.3f}{rt:>9.3f}")
    return rows


def main():
    vot = DVSBenchmark(VOT)
    run(VOT, vot.data, "VOT Challenge 2015 (all)")

    trk = DVSBenchmark(TRK)
    rng = np.random.default_rng(0)
    sample = [trk.data[i] for i in sorted(rng.choice(len(trk.data), 20, replace=False))]
    run(TRK, sample, "TrackingDataset (20 of 67)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
