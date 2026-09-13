"""Clustering, percentile crops and persistent Gaussian mixtures for event streams.

Coordinates use (x, y); sensor sizes use (width, height), while image arrays
use (height, width). Event timestamps and frame windows are in microseconds."""

import warnings
from pathlib import Path

import numpy as np
import tonic
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

WINDOW_US = 50_000
METHOD = "gmm"
K = 3
TOP_N = 3
SENSOR = (128, 128)

DECAY = 0.85
HYSTERESIS = 1.15
REG_COVAR = 1.0


def load_gesture(root="./data", train=True):
    extracted = Path(root) / "DVSGesture" / ("ibmGestureTrain" if train else "ibmGestureTest")
    n_npy = len(list(extracted.glob("**/*.npy"))) if extracted.is_dir() else 0
    if n_npy < 100:
        raise SystemExit(
            f"DVSGesture recordings are missing: {extracted} holds {n_npy} .npy files, "
            "and at least 100 are needed. See notebook 01 section 2 for the manual download."
        )
    tonic.datasets.DVSGesture._is_file_present = lambda self: True
    return tonic.datasets.DVSGesture(save_to=str(root), train=train)


def unpack(events):
    return (
        events["x"].astype(np.int64),
        events["y"].astype(np.int64),
        events["t"].astype(np.int64),
        events["p"].astype(np.int64),
    )


def cluster_frame(coords, method=METHOD, k=K, random_state=0):
    if len(coords) < k:
        return None
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    elif method == "gmm":
        model = GaussianMixture(n_components=k, random_state=random_state)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError(f"unknown method: {method}")
    return model.fit_predict(coords)


def fovea_box(points, margin=5):
    lo = np.percentile(points, margin, axis=0)
    hi = np.percentile(points, 100 - margin, axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def cluster_table(coords, times, labels, n_bins=5):
    edges = np.linspace(times.min(), times.max() + 1, n_bins + 1)
    rows = []
    for cid in np.unique(labels):
        m = labels == cid
        box = fovea_box(coords[m])
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        size = int(m.sum())
        occupied = np.unique(np.clip(np.digitize(times[m], edges) - 1, 0, n_bins - 1))
        rows.append(
            {
                "id": int(cid),
                "size": size,
                "density": size / area,
                "persistence": len(occupied) / n_bins,
                "box": box,
            }
        )
    return rows


def rank_clusters(coords, times, labels, by="size"):
    return sorted(cluster_table(coords, times, labels), key=lambda r: r[by], reverse=True)


def most_salient(coords, times, labels, by="size"):
    return rank_clusters(coords, times, labels, by=by)[0]


def iter_frames(x, y, t, window=WINDOW_US, min_events=10):
    for s in np.arange(t.min(), t.max(), window):
        m = (t >= s) & (t < s + window)
        if int(m.sum()) < min_events:
            continue
        yield s, x[m], y[m], t[m]


def frame_image(xs, ys, shape=(128, 128)):
    img = np.zeros(shape, dtype=np.float32)
    np.add.at(img, (ys, xs), 1.0)
    return img


def box_centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detections(xs, ys, ts, method=METHOD, k=K, top_n=TOP_N, by="size"):
    coords = np.column_stack([xs, ys]).astype(np.float32)
    labels = cluster_frame(coords, method=method, k=k)
    if labels is None:
        return []
    return rank_clusters(coords, ts, labels, by=by)[:top_n]


class PersistentGMM:
    """A Gaussian mixture carried forward from frame to frame.

    Each frame is fitted starting from the previous frame's parameters, so a
    component keeps its identity instead of being re-initialised at random.
    That is what makes a per-component persistence score meaningful: it is an
    exponential moving average of how much of the frame's activity the
    component has been holding over time.
    """

    def __init__(self, k=K, decay=DECAY, hysteresis=HYSTERESIS, reg_covar=REG_COVAR, warm=True):
        self.k = k
        self.decay = decay
        self.hysteresis = hysteresis
        self.reg_covar = reg_covar
        self.warm = warm
        self.model = None
        self.persistence = np.zeros(k)
        self.fovea_id = None
        self.history = []

    def update(self, coords):
        prev = self.model if self.warm else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if prev is None:
                g = GaussianMixture(n_components=self.k, random_state=0, reg_covar=self.reg_covar)
            else:
                g = GaussianMixture(
                    n_components=self.k,
                    reg_covar=self.reg_covar,
                    means_init=prev.means_,
                    weights_init=prev.weights_,
                    precisions_init=prev.precisions_,
                    max_iter=50,
                )
            g.fit(coords)
        self.persistence = self.decay * self.persistence + (1 - self.decay) * g.weights_
        self.model = g
        self.history.append(self.persistence.copy())
        return g

    def component_area(self):
        return np.array(
            [2 * np.pi * np.sqrt(max(np.linalg.det(c), 1e-9)) for c in self.model.covariances_]
        )

    def score(self, rule="persistence"):
        g = self.model
        if rule.startswith("size"):
            return g.weights_
        if rule.startswith("density"):
            return g.weights_ / self.component_area()
        if rule.startswith("persistence"):
            return self.persistence
        if rule.startswith("persdens"):
            return self.persistence / self.component_area()
        raise ValueError(f"unknown rule: {rule}")

    def select(self, rule="persistence"):
        s = self.score(rule)
        best = int(s.argmax())
        if rule.endswith("+hyst") and self.fovea_id is not None:
            if s[best] <= s[self.fovea_id] * self.hysteresis:
                best = self.fovea_id
        self.fovea_id = best
        return best

    def saliency_map(self, shape=SENSOR, rule="persistence"):
        g = self.model
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
        smap = np.zeros(grid.shape[0])
        weights = g.weights_ if rule.startswith("size") else self.persistence
        for cid in range(self.k):
            cov = g.covariances_[cid]
            det = np.linalg.det(cov)
            if det <= 0:
                continue
            d = grid - g.means_[cid]
            expo = -0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
            smap += weights[cid] * np.exp(expo) / (2 * np.pi * np.sqrt(det))
        return smap.reshape(shape)

    def component_box(self, cid, coords, margin=5):
        pts = coords[self.model.predict(coords) == cid]
        if len(pts) < 5:
            return None
        return fovea_box(pts, margin=margin)


def dwell_stats(ids):
    """Mean and longest unbroken run of the same fovea id, plus the switch rate."""
    runs, cur = [], 1
    for a, b in zip(ids, ids[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b) / max(len(ids) - 1, 1)
    return float(np.mean(runs)), int(max(runs)), switches


def normalise(a):
    top = a.max()
    return a / top if top > 0 else a
