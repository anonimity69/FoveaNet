"""Build fixed-budget crop representations of DVS128 Gesture.

Each condition produces an (8, 2, 32, 32) event-count tensor. CONDITIONS lists
instantaneous and accumulated selection rules, centre/random baselines,
full-sensor and recentered views, enlarged crops, and fovea-plus-periphery.
Centre and random box sizes match the median persistence box per recording.

Example: python build_crops.py --window 10000 --out data/crops.npz"""

import argparse
import time

import numpy as np

import foveanet as fn

N_BINS = 8
SIZE = 32

# size/density and persistence/persdens compare instantaneous vs accumulated
# activity, with and without area normalisation.
CONDITIONS = [
    "persistence",
    "persdens",
    "size",
    "density",
    "centre",
    "random",
    "full",
    "recentred",
    "persist2x",
    "fovperiph",
]


def blank():
    return np.zeros((N_BINS, 2, SIZE, SIZE), dtype=np.float32)


def scaled_box(box, factor, sensor):
    """The same box about the same centre, ``factor`` times larger, clipped."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw = max(box[2] - box[0], 1) * factor / 2
    hh = max(box[3] - box[1], 1) * factor / 2
    return (
        max(cx - hw, 0),
        max(cy - hh, 0),
        min(cx + hw, sensor[0] - 1),
        min(cy + hh, sensor[1] - 1),
    )


def add_shifted(tensor, bin_idx, xs, ys, ps, box, sensor):
    """Translate the full field so the fovea centre maps to the sensor centre.

    Events shifted beyond the sensor are dropped, not wrapped. This control
    separates spatial registration from cropping."""
    W, H = sensor
    dx = int(round(W / 2 - (box[0] + box[2]) / 2))
    dy = int(round(H / 2 - (box[1] + box[3]) / 2))
    nx, ny = xs + dx, ys + dy
    keep = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
    if not keep.any():
        return
    u = np.clip((nx[keep] / W * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    v = np.clip((ny[keep] / H * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    np.add.at(tensor, (bin_idx, ps[keep], v, u), 1.0)


def add_fovea_periphery(tensor, bin_idx, xs, ys, ps, box, sensor):
    """Render the fovea in the top half and the full sensor in the bottom half.

    Both views share the fixed tensor budget. The multiresolution comparison
    follows the motivation in Gruel et al.; see the experiment log."""
    W, H = sensor
    half = SIZE // 2
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if keep.any():
        u = np.clip(((xs[keep] - x0) / w * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
        v = np.clip(((ys[keep] - y0) / h * (half - 1)).astype(np.int64), 0, half - 1)
        np.add.at(tensor, (bin_idx, ps[keep], v, u), 1.0)
    u = np.clip((xs / W * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    v = np.clip((ys / H * (half - 1)).astype(np.int64) + half, half, SIZE - 1)
    np.add.at(tensor, (bin_idx, ps, v, u), 1.0)


def add_crop(tensor, bin_idx, xs, ys, ps, box):
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if not keep.any():
        return
    u = np.clip(((xs[keep] - x0) / w * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    v = np.clip(((ys[keep] - y0) / h * (SIZE - 1)).astype(np.int64), 0, SIZE - 1)
    np.add.at(tensor, (bin_idx, ps[keep], v, u), 1.0)


def process(x, y, t, p, window, rng, sensor=None):
    t0, t1 = t.min(), t.max()
    span = max(t1 - t0, 1)
    out = {k: blank() for k in CONDITIONS}

    pg = fn.PersistentGMM()
    boxes_p, boxes_d, boxes_s, boxes_n, frames = [], [], [], [], []
    for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window, min_events=20):
        coords = np.column_stack([xs, ys]).astype(np.float32)
        pg.update(coords)
        cid_p = pg.select(rule="persistence+hyst")
        bp = pg.component_box(cid_p, coords)
        pg.fovea_id = None
        cid_d = pg.select(rule="persdens+hyst")
        bd = pg.component_box(cid_d, coords)
        pg.fovea_id = None
        cid_s = pg.select(rule="size+hyst")
        bs = pg.component_box(cid_s, coords)
        pg.fovea_id = None
        cid_n = pg.select(rule="density+hyst")
        bn = pg.component_box(cid_n, coords)
        pg.fovea_id = None
        sel = (t >= s) & (t < s + window)
        frames.append((int(N_BINS * (s - t0) / span) % N_BINS, xs, ys, p[sel]))
        boxes_p.append(bp)
        boxes_d.append(bd)
        boxes_s.append(bs)
        boxes_n.append(bn)

    if not frames:
        return None

    # Match control dimensions to the median persistence box, not persdens.
    sizes = [(b[2] - b[0], b[3] - b[1]) for b in boxes_p if b is not None]
    if not sizes:
        return None
    mw = int(np.median([s[0] for s in sizes]))
    mh = int(np.median([s[1] for s in sizes]))
    mw, mh = max(mw, 4), max(mh, 4)

    W, H = sensor if sensor is not None else fn.SENSOR
    cx, cy = W // 2, H // 2
    centre_box = (cx - mw // 2, cy - mh // 2, cx + mw // 2, cy + mh // 2)
    rx = int(rng.integers(0, max(W - mw, 1)))
    ry = int(rng.integers(0, max(H - mh, 1)))
    random_box = (rx, ry, rx + mw, ry + mh)
    full_box = (0, 0, W - 1, H - 1)

    for (b, xs, ys, ps), bp, bd, bs, bn in zip(frames, boxes_p, boxes_d, boxes_s, boxes_n):
        if bs is not None:
            add_crop(out["size"], b, xs, ys, ps, bs)
        if bn is not None:
            add_crop(out["density"], b, xs, ys, ps, bn)
        if bp is not None:
            add_crop(out["persistence"], b, xs, ys, ps, bp)
            add_crop(out["persist2x"], b, xs, ys, ps, scaled_box(bp, 2.0, (W, H)))
            add_shifted(out["recentred"], b, xs, ys, ps, bp, (W, H))
            add_fovea_periphery(out["fovperiph"], b, xs, ys, ps, bp, (W, H))
        if bd is not None:
            add_crop(out["persdens"], b, xs, ys, ps, bd)
        add_crop(out["centre"], b, xs, ys, ps, centre_box)
        add_crop(out["random"], b, xs, ys, ps, random_box)
        add_crop(out["full"], b, xs, ys, ps, full_box)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crops.npz")
    ap.add_argument("--window", type=int, default=fn.WINDOW_US)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    store = {k: [] for k in CONDITIONS}
    labels, users, splits = [], [], []

    wanted = ["train", "test"] if args.split == "both" else [args.split]
    for split in wanted:
        ds = fn.load_gesture(train=(split == "train"))
        n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
        print(f"{split}: {n} recordings, subjects {sorted(set(ds.users))}", flush=True)
        start = time.time()
        for i in range(n):
            events, label = ds[i]
            x, y, t, p = fn.unpack(events)
            got = process(x, y, t, p, args.window, rng)
            if got is None:
                continue
            for k, v in got.items():
                store[k].append(v)
            labels.append(label)
            users.append(ds.users[i])
            splits.append(0 if split == "train" else 1)
            if (i + 1) % 50 == 0:
                done = time.time() - start
                print(
                    f"  {split} {i + 1}/{n}, {done:.0f}s elapsed, "
                    f"{done / (i + 1) * (n - i - 1):.0f}s left",
                    flush=True,
                )

    arrays = {k: np.stack(v) for k, v in store.items()}
    arrays["label"] = np.array(labels)
    arrays["user"] = np.array(users)
    arrays["split"] = np.array(splits)
    np.savez_compressed(args.out, **arrays)
    n_tr = int((arrays["split"] == 0).sum())
    n_te = int((arrays["split"] == 1).sum())
    print(
        f"wrote {args.out}: {n_tr} train + {n_te} test recordings, "
        f"each {arrays['full'].shape[1:]} per condition"
    )


if __name__ == "__main__":
    main()
