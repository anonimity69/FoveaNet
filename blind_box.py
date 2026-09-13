"""Measure random-box activity relative to a same-sized foveal box.

Ratios near zero indicate spatially concentrated activity; ratios near one
indicate little advantage over random placement. This is an activity diagnostic,
not a classifier accuracy estimate. See PREDICTIONS.txt for historical results.

Example: python blind_box.py --dataset gesture --limit 30"""

import argparse

import numpy as np

import foveanet as fn


def diagnose(
    records, sensor, window=10_000, k=3, limit=30, seed=0, rule="persistence+hyst", verbose=True
):
    """Ratio of blind-box capture to fovea capture over a sample of recordings.

    ``records`` is any iterable of ``(x, y, t)`` arrays. Only ``limit`` of them
    are used. Sampling uncertainty depends on the recordings and should be
    checked before treating the ratio as representative of a dataset.
    """
    rng = np.random.default_rng(seed)
    W, H = sensor
    fovea_hits, blind_hits, boxes, used = [], [], [], 0

    for x, y, t in records:
        if used >= limit:
            break
        pg = fn.PersistentGMM(k=k)
        per_frame_fovea, per_frame_blind, sizes = [], [], []

        for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window, min_events=20):
            coords = np.column_stack([xs, ys]).astype(np.float32)
            pg.update(coords)
            box = pg.component_box(pg.select(rule=rule), coords)
            if box is None:
                continue
            bw, bh = max(box[2] - box[0], 1), max(box[3] - box[1], 1)
            sizes.append((bw, bh))
            inside = ((xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])).sum()
            per_frame_fovea.append(inside)

        if not sizes:
            continue
        mw = int(np.median([s[0] for s in sizes]))
        mh = int(np.median([s[1] for s in sizes]))
        rx = int(rng.integers(0, max(W - mw, 1)))
        ry = int(rng.integers(0, max(H - mh, 1)))

        for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window, min_events=20):
            inside = ((xs >= rx) & (xs <= rx + mw) & (ys >= ry) & (ys <= ry + mh)).sum()
            per_frame_blind.append(inside)

        if not per_frame_fovea or not per_frame_blind:
            continue
        fovea_hits.append(np.mean(per_frame_fovea))
        blind_hits.append(np.mean(per_frame_blind))
        boxes.append(mw * mh / (W * H))
        used += 1
        if verbose and used % 10 == 0:
            r = np.sum(blind_hits) / max(np.sum(fovea_hits), 1e-9)
            print(f"  {used} recordings, running ratio {r:.3f}", flush=True)

    if not fovea_hits:
        raise SystemExit("no usable recordings")
    ratio = float(np.sum(blind_hits) / max(np.sum(fovea_hits), 1e-9))
    per_rec = np.array(blind_hits) / np.maximum(np.array(fovea_hits), 1e-9)
    return {
        "ratio": ratio,
        "ratio_sd": float(per_rec.std()),
        "recordings": used,
        "fovea_events_per_frame": float(np.mean(fovea_hits)),
        "blind_events_per_frame": float(np.mean(blind_hits)),
        "box_fraction": float(np.mean(boxes)),
    }


def gesture_records(train=True):
    ds = fn.load_gesture(train=train)
    for i in range(len(ds)):
        events, _ = ds[i]
        x, y, t, _ = fn.unpack(events)
        yield x, y, t


def ucf_records(path="data/DVSBENCH/INI_UCF50_30fps_20160424.hdf5"):
    import h5py

    from build_crops_ucf import list_sequences

    names = list_sequences(path)
    with h5py.File(path, "r") as f:
        for nm in names:
            g = f[nm]
            yield (
                g["x_pos"][:].astype(np.int64),
                g["y_pos"][:].astype(np.int64),
                g["timestamps"][:].astype(np.int64),
            )


def dvslip_records(train=True):
    """DVS-Lip, via Tonic's own reader. Note the samples are already cropped to
    the mouth at 128x128, so this measures activity spread inside a crop that
    someone else already placed, not across a scene."""
    import tonic

    ds = tonic.datasets.DVSLip(save_to="data", train=train)
    order = np.random.default_rng(0).permutation(len(ds))
    for i in order:
        events, _ = ds[int(i)]
        yield (
            events["x"].astype(np.int64),
            events["y"].astype(np.int64),
            events["t"].astype(np.int64),
        )


def slanimals_records():
    """SL-Animals-DVS. Iterated in dataset order rather than shuffled, because
    the 19 signs of one take are consecutive and the reader caches the decoded
    recording; shuffling would decode a 100 MB file once per sample."""
    from tonic_convert import SLAnimals

    ds = SLAnimals()
    for i in range(len(ds)):
        x, y, t, _ = ds.load(ds.data[i])
        yield x, y, t


LOADERS = {
    "gesture": (gesture_records, (128, 128)),
    "ucf": (ucf_records, (240, 180)),
    "dvslip": (dvslip_records, (128, 128)),
    "slanimals": (slanimals_records, (128, 128)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loader, sensor = LOADERS[args.dataset]
    print(
        f"{args.dataset}: sensor {sensor}, window {args.window / 1000:.0f} ms, "
        f"sampling {args.limit} recordings\n"
    )
    out = diagnose(loader(), sensor, window=args.window, limit=args.limit, seed=args.seed)
    print(f"\n  fovea catches   {out['fovea_events_per_frame']:9.1f} events/frame")
    print(f"  blind box       {out['blind_events_per_frame']:9.1f} events/frame")
    print(f"  box fraction    {100 * out['box_fraction']:9.2f}% of sensor")
    print(
        f"\n  BLIND-BOX RATIO {out['ratio']:.3f}   (sd across recordings "
        f"{out['ratio_sd']:.3f}, n={out['recordings']})"
    )
    verdict = (
        "activity is localised, a fovea has something to find"
        if out["ratio"] < 0.25
        else "borderline, expect a small or unreliable benefit"
        if out["ratio"] < 0.4
        else "activity is spread out, expect a fixed crop to compete or win"
    )
    print(f"  reading: {verdict}")


if __name__ == "__main__":
    main()
