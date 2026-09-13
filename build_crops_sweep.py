"""Build equal-area foveal and central crops across retained fractions.

fovNN and cenNN share nominal dimensions W*sqrt(f) by H*sqrt(f). Edge clipping
reduces the achieved area, which is stored with the tensors for plotting.
All conditions retain the same (8, 2, 32, 32) tensor shape; full is shared at 100%.

Example: python build_crops_sweep.py --window 10000 --out data/crops_sweep_10ms.npz"""

import argparse
import time

import numpy as np

import foveanet as fn
from build_crops import N_BINS, add_crop, blank

FRACTIONS = [0.75, 0.50, 0.25, 0.10, 0.03]


def names_for(fractions):
    out = ["full"]
    for f in fractions:
        out += [f"fov{int(round(f * 100)):02d}", f"cen{int(round(f * 100)):02d}"]
    return out


def sized_box(cx, cy, f, sensor):
    """Box of area ``f`` of the sensor, centred at (cx, cy), clipped to bounds."""
    W, H = sensor
    hw, hh = W * np.sqrt(f) / 2, H * np.sqrt(f) / 2
    x0, y0 = max(cx - hw, 0), max(cy - hh, 0)
    x1, y1 = min(cx + hw, W - 1), min(cy + hh, H - 1)
    return (x0, y0, x1, y1)


def area_frac(box, sensor):
    W, H = sensor
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0) / (W * H)


def process(x, y, t, p, window, fractions):
    t0, t1 = t.min(), t.max()
    span = max(t1 - t0, 1)
    W, H = fn.SENSOR
    names = names_for(fractions)
    out = {k: blank() for k in names}
    achieved = {k: [] for k in names}

    pg = fn.PersistentGMM()
    frames, boxes = [], []
    for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window, min_events=20):
        coords = np.column_stack([xs, ys]).astype(np.float32)
        pg.update(coords)
        bp = pg.component_box(pg.select(rule="persistence+hyst"), coords)
        sel = (t >= s) & (t < s + window)
        frames.append((int(N_BINS * (s - t0) / span) % N_BINS, xs, ys, p[sel]))
        boxes.append(bp)

    if not frames or all(b is None for b in boxes):
        return None, None

    full_box = (0, 0, W - 1, H - 1)
    for (b, xs, ys, ps), bp in zip(frames, boxes):
        add_crop(out["full"], b, xs, ys, ps, full_box)
        achieved["full"].append(1.0)
        if bp is None:
            continue
        fcx, fcy = (bp[0] + bp[2]) / 2, (bp[1] + bp[3]) / 2
        for f in fractions:
            fn_name = f"fov{int(round(f * 100)):02d}"
            cn_name = f"cen{int(round(f * 100)):02d}"
            fb = sized_box(fcx, fcy, f, (W, H))
            cb = sized_box(W / 2, H / 2, f, (W, H))
            add_crop(out[fn_name], b, xs, ys, ps, fb)
            add_crop(out[cn_name], b, xs, ys, ps, cb)
            achieved[fn_name].append(area_frac(fb, (W, H)))
            achieved[cn_name].append(area_frac(cb, (W, H)))

    means = {k: float(np.mean(v)) if v else 0.0 for k, v in achieved.items()}
    return out, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crops_sweep_10ms.npz")
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    names = names_for(FRACTIONS)
    print(f"conditions: {', '.join(names)}\n", flush=True)
    store = {k: [] for k in names}
    frac_store = {k: [] for k in names}
    labels, users, splits = [], [], []

    for split in ["train", "test"] if args.split == "both" else [args.split]:
        ds = fn.load_gesture(train=(split == "train"))
        n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
        print(f"{split}: {n} recordings", flush=True)
        start = time.time()
        for i in range(n):
            events, label = ds[i]
            x, y, t, p = fn.unpack(events)
            got, means = process(x, y, t, p, args.window, FRACTIONS)
            if got is None:
                continue
            for k, v in got.items():
                store[k].append(v)
                frac_store[k].append(means[k])
            labels.append(label)
            users.append(ds.users[i])
            splits.append(0 if split == "train" else 1)
            if (i + 1) % 50 == 0:
                el = time.time() - start
                print(
                    f"  {split} {i + 1}/{n}, {el:.0f}s elapsed, "
                    f"{el / (i + 1) * (n - i - 1):.0f}s left",
                    flush=True,
                )

    arrays = {k: np.stack(v) for k, v in store.items()}
    arrays["label"] = np.array(labels)
    arrays["user"] = np.array(users)
    arrays["split"] = np.array(splits)
    for k in names:
        arrays[f"frac_{k}"] = np.array(frac_store[k])
    np.savez_compressed(args.out, **arrays)

    print(
        f"\nwrote {args.out}: {int((arrays['split'] == 0).sum())} train + "
        f"{int((arrays['split'] == 1).sum())} test"
    )
    print(f"\n{'condition':>10}{'nominal':>9}{'achieved':>10}")
    for k in names:
        nominal = 1.0 if k == "full" else int(k[3:]) / 100
        print(f"{k:>10}{nominal:>9.2f}{arrays[f'frac_{k}'].mean():>10.3f}")


if __name__ == "__main__":
    main()
