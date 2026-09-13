"""Build crop representations for the UCF-50 DVS benchmark.

Inputs have a 240 x 180 sensor and come from video replayed on a monitor.
Display modulation and scene motion both affect activity; this loader does
not attribute performance differences to either source.

Example: python build_crops_ucf.py --out data/crops_ucf.npz --classes 12 --per-class 40"""

import argparse
import time

import h5py
import numpy as np

from build_crops import CONDITIONS

H5 = "data/DVSBENCH/INI_UCF50_30fps_20160424.hdf5"
SENSOR = (240, 180)


def list_sequences(path):
    out = []

    def walk(group, prefix=""):
        for key, node in group.items():
            sub = f"{prefix}/{key}" if prefix else key
            if isinstance(node, h5py.Group):
                if "x_pos" in node:
                    out.append(sub)
                else:
                    walk(node, sub)

    with h5py.File(path, "r") as f:
        walk(f)
    return sorted(out)


def process(x, y, t, p, window, rng):
    """Run the shared crop conditions with the 240 x 180 sensor dimensions."""
    from build_crops import process as shared

    return shared(x, y, t, p, window, rng, sensor=SENSOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crops_ucf.npz")
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()

    names = list_sequences(H5)
    by_class = {}
    for n in names:
        cls = n.split("/")[0]
        by_class.setdefault(cls, []).append(n)

    rng = np.random.default_rng(0)
    chosen_classes = sorted(by_class)[: args.classes]
    picked = []
    for ci, cls in enumerate(chosen_classes):
        pool = by_class[cls]
        take = (
            pool
            if len(pool) <= args.per_class
            else [pool[i] for i in rng.choice(len(pool), args.per_class, replace=False)]
        )
        picked += [(p, ci) for p in take]
    print(f"{len(by_class)} classes available, using {len(chosen_classes)}")
    print(f"{len(picked)} recordings, window {args.window / 1000:.0f} ms", flush=True)

    store = {k: [] for k in CONDITIONS}
    labels, groups = [], []
    start = time.time()
    with h5py.File(H5, "r") as f:
        for i, (path, cls) in enumerate(picked):
            g = f[path]
            x = g["x_pos"][:].astype(np.int64)
            y = g["y_pos"][:].astype(np.int64)
            t = g["timestamps"][:].astype(np.int64)
            p = g["pol"][:].astype(np.int64)
            got = process(x, y, t, p, args.window, rng)
            if got is None:
                continue
            for k, v in got.items():
                store[k].append(v)
            labels.append(cls)
            groups.append(path.split("/")[-1].split("_g")[-1][:2] if "_g" in path else "00")
            if (i + 1) % 20 == 0:
                el = time.time() - start
                print(
                    f"  {i + 1}/{len(picked)}, {el:.0f}s elapsed, "
                    f"{el / (i + 1) * (len(picked) - i - 1):.0f}s left",
                    flush=True,
                )

    arrays = {k: np.stack(v) for k, v in store.items()}
    arrays["label"] = np.array(labels)
    arrays["user"] = np.array([int(g) if g.isdigit() else 0 for g in groups])
    np.savez_compressed(args.out, **arrays)
    print(
        f"wrote {args.out}: {len(labels)} recordings, "
        f"{len(set(labels))} classes, {len(set(arrays['user'].tolist()))} groups"
    )


if __name__ == "__main__":
    main()
