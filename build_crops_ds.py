"""Build crop representations for SL-Animals-DVS and DVS-Lip.

Uses build_crops.process with 128 x 128 inputs. SL-Animals holds out the last
sorted subject IDs; DVS-Lip uses its supplied split and samples each word class.

Example: python build_crops_ds.py --dataset slanimals --out data/crops_slanimals.npz"""

import argparse
import time

import numpy as np

from build_crops import CONDITIONS, process


def slanimals_samples(args):
    from tonic_convert import SLAnimals

    ds = SLAnimals()
    subjects = sorted(set(ds.subject))
    held = set(subjects[-args.holdout :])
    print(f"SL-Animals: {len(ds)} samples, {len(subjects)} subjects, holding out {sorted(held)}")
    for i in range(len(ds)):
        x, y, t, p = ds.load(ds.data[i])
        yield x, y, t, p, ds.targets[i], ds.subject[i], int(ds.subject[i] in held)


def dvslip_samples(args):
    """DVS-Lip through tonic's own reader.

    Tonic derives the data directory from the class name, so the recordings must
    sit at data/DVSLip/DVS-Lip/{train,test} and the call is save_to="data". This
    is the same layout constraint DVS128 Gesture has, and getting it wrong
    produces a download prompt rather than an error.
    """
    import tonic.datasets as td

    rng = np.random.default_rng(0)
    for split, is_test in (("train", 0), ("test", 1)):
        ds = td.DVSLip(save_to="data", train=(split == "train"))
        targets = [int(ds.targets[i]) for i in range(len(ds))]
        per = args.per_class if not is_test else max(args.per_class // 3, 8)
        chosen = []
        for ci in range(args.classes):
            idx = [i for i, tg in enumerate(targets) if tg == ci]
            if len(idx) > per:
                idx = [idx[j] for j in rng.choice(len(idx), per, replace=False)]
            chosen += [(i, ci) for i in idx]
        print(f"DVS-Lip {split}: {len(chosen)} samples over {args.classes} words", flush=True)
        for i, ci in chosen:
            ev, _ = ds[i]
            yield (
                ev["x"].astype(np.int64),
                ev["y"].astype(np.int64),
                ev["t"].astype(np.int64),
                ev["p"].astype(np.int64),
                ci,
                0,
                is_test,
            )


SOURCES = {"slanimals": slanimals_samples, "dvslip": dvslip_samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--holdout", type=int, default=12, help="slanimals subjects")
    ap.add_argument("--classes", type=int, default=20, help="dvslip words")
    ap.add_argument("--per-class", type=int, default=60, help="dvslip per word")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    store = {k: [] for k in CONDITIONS}
    labels, users, splits = [], [], []
    start = time.time()

    for n, (x, y, t, p, label, user, is_test) in enumerate(SOURCES[args.dataset](args)):
        got = process(x, y, t, p, args.window, rng)
        if got is None:
            continue
        for k, v in got.items():
            store[k].append(v)
        labels.append(label)
        users.append(user)
        splits.append(is_test)
        if (n + 1) % 50 == 0:
            el = time.time() - start
            print(f"  {n + 1} samples, {el:.0f}s elapsed ({el / (n + 1):.2f}s each)", flush=True)

    arrays = {k: np.stack(v) for k, v in store.items()}
    arrays["label"] = np.array(labels)
    arrays["user"] = np.array(users)
    arrays["split"] = np.array(splits)
    np.savez_compressed(args.out, **arrays)
    print(
        f"wrote {args.out}: {int((arrays['split'] == 0).sum())} train + "
        f"{int((arrays['split'] == 1).sum())} test, "
        f"{len(set(labels))} classes, {arrays['full'].shape[1:]} per condition"
    )


if __name__ == "__main__":
    main()
