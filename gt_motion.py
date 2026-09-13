"""Summarise annotated target motion and static-box overlap on VOT.

Distances are reported relative to the median target-box diagonal.
Run from the repository root with ``python gt_motion.py``."""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
import evaluate as ev
from tonic_convert import DVSBenchmark

H5 = "data/DVSBENCH/INI_VOT_30fps_20160610.hdf5"


def main():
    ds = DVSBenchmark(H5)

    rows = []
    for nm in ds.data:
        b = ds._boxes[nm]
        boxes = np.array([ev.gt_box(r) for r in b])
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = np.median(boxes[:, 2] - boxes[:, 0])
        h = np.median(boxes[:, 3] - boxes[:, 1])
        diag = np.hypot(w, h)
        step = np.hypot(np.diff(cx), np.diff(cy))
        path = step.sum()
        net = np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])
        first = boxes[0]
        static_iou = np.mean([ev.iou(tuple(first), tuple(g)) for g in boxes])
        rows.append(
            dict(
                name=nm,
                n=len(boxes),
                diag=diag,
                step=step.mean(),
                step_rel=step.mean() / max(diag, 1),
                path=path,
                path_rel=path / max(diag, 1),
                net_rel=net / max(diag, 1),
                static=static_iou,
            )
        )

    rows.sort(key=lambda r: r["static"])
    print(f"{len(rows)} VOT sequences, sorted by how well a never-moving box scores\n")
    print(
        f"{'sequence':>16}{'frames':>7}{'box diag':>10}{'step/diag':>11}"
        f"{'path/diag':>11}{'net/diag':>10}{'static IoU':>12}"
    )
    for r in rows[:8]:
        print(
            f"{r['name'][:16]:>16}{r['n']:>7}{r['diag']:>10.0f}{r['step_rel']:>11.3f}"
            f"{r['path_rel']:>11.1f}{r['net_rel']:>10.1f}{r['static']:>12.3f}"
        )
    print(f"{'...':>16}")
    for r in rows[-8:]:
        print(
            f"{r['name'][:16]:>16}{r['n']:>7}{r['diag']:>10.0f}{r['step_rel']:>11.3f}"
            f"{r['path_rel']:>11.1f}{r['net_rel']:>10.1f}{r['static']:>12.3f}"
        )

    st = np.array([r["static"] for r in rows])
    pr = np.array([r["path_rel"] for r in rows])
    nr = np.array([r["net_rel"] for r in rows])
    print(
        f"\n{'MEDIAN':>16}{'':>7}{'':>10}"
        f"{np.median([r['step_rel'] for r in rows]):>11.3f}"
        f"{np.median(pr):>11.1f}{np.median(nr):>10.1f}{np.median(st):>12.3f}"
    )
    print(
        f"{'MEAN':>16}{'':>7}{'':>10}"
        f"{np.mean([r['step_rel'] for r in rows]):>11.3f}"
        f"{np.mean(pr):>11.1f}{np.mean(nr):>10.1f}{np.mean(st):>12.3f}"
    )

    print(f"\nsequences where a static box scores above 0.5: {int((st > 0.5).sum())} of {len(st)}")
    print(f"sequences where it scores below 0.1:            {int((st < 0.1).sum())} of {len(st)}")
    print("\npath/diag is total distance travelled in units of the target's own size.")
    print("net/diag is straight-line start-to-end distance in the same units.")


if __name__ == "__main__":
    main()
