"""Track a target with a GMM fitted inside the previous box's search window.

The first benchmark annotation initialises the target. Later annotations are
used only for scoring. Search is local rather than over the entire sensor."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

import evaluate as ev

SEARCH_SCALE = 2.5
MIN_EVENTS = 30


def _fit(coords, k, prev=None, reg_covar=1.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if prev is None:
            g = GaussianMixture(n_components=k, random_state=0, reg_covar=reg_covar)
        else:
            g = GaussianMixture(
                n_components=k,
                reg_covar=reg_covar,
                max_iter=50,
                means_init=prev.means_,
                weights_init=prev.weights_,
                precisions_init=prev.precisions_,
            )
        g.fit(coords)
    return g


def _search_window(box, scale, sensor):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw = max(box[2] - box[0], 4) * scale / 2
    hh = max(box[3] - box[1], 4) * scale / 2
    return (
        max(cx - hw, 0),
        max(cy - hh, 0),
        min(cx + hw, sensor[0] - 1),
        min(cy + hh, sensor[1] - 1),
    )


def _percentile_box(points, margin=10):
    lo = np.percentile(points, margin, axis=0)
    hi = np.percentile(points, 100 - margin, axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _recentre(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def track(
    x,
    y,
    t,
    boxes,
    window,
    k=2,
    search_scale=SEARCH_SCALE,
    sensor=(240, 180),
    alpha=0.5,
    keep_size=True,
    rule="iou",
    predict=0.0,
    decay=0.85,
    gate=1.0,
):
    """Follow the annotated target through one sequence.

    Initialised from the first ground-truth box, then never told the answer
    again. Returns the predicted box per frame and the ground-truth box it was
    scored against.

    ``rule`` decides which component of the local mixture becomes the new box:

        iou           the component overlapping the previous box most. This is
                      the original rule, and it is worth being explicit that it
                      rewards a component for having moved as little as
                      possible, which is why it produces a nearly stationary
                      box.
        near          the component whose centre is closest to where the target
                      is predicted to be. Carries no preference for staying put.
        density       the most concentrated component, damped by distance.
        persistence   accumulated persistence, damped by distance. This is the
                      project's own saliency measure, applied locally.

    ``predict`` extrapolates the box forward along its last step before the
    search window is placed, so the window is centred on where the target is
    going rather than where it was. 0 disables it, 1 is a full constant-velocity
    step. ``gate`` sets how sharply a candidate is penalised for sitting far
    from the predicted centre, in units of the box half-diagonal.
    """
    first = ev.gt_box(boxes[0])
    box = first
    init_w = first[2] - first[0]
    init_h = first[3] - first[1]
    radius = max(np.hypot(init_w, init_h) / 2, 1.0)

    prev_model = None
    persistence = np.zeros(k)
    vel = np.zeros(2)
    preds, truths, times = [], [], []

    def skip():
        preds.append(box)
        truths.append(ev.gt_at(boxes, start + window / 2))
        times.append(start)

    for start in np.arange(t.min(), t.max(), window):
        sel = (t >= start) & (t < start + window)
        if int(sel.sum()) < MIN_EVENTS:
            skip()
            continue

        cx0, cy0 = _centre(box)
        px, py = cx0 + predict * vel[0], cy0 + predict * vel[1]
        px = float(np.clip(px, 0, sensor[0] - 1))
        py = float(np.clip(py, 0, sensor[1] - 1))
        search_from = _recentre(px, py, init_w, init_h) if keep_size else box

        xs, ys = x[sel], y[sel]
        sw = _search_window(search_from, search_scale, sensor)
        near = (xs >= sw[0]) & (xs <= sw[2]) & (ys >= sw[1]) & (ys <= sw[3])
        if int(near.sum()) < MIN_EVENTS:
            skip()
            continue

        coords = np.column_stack([xs[near], ys[near]]).astype(np.float32)
        g = _fit(coords, k, prev_model)
        prev_model = g
        persistence = decay * persistence + (1 - decay) * g.weights_

        labels = g.predict(coords)
        best, best_score = None, -np.inf
        for cid in range(k):
            pts = coords[labels == cid]
            if len(pts) < 10:
                continue
            cand = _percentile_box(pts)
            ccx, ccy = _centre(cand)
            dist = np.hypot(ccx - px, ccy - py)
            proximity = 1.0 / (1.0 + gate * dist / radius)

            if rule == "iou":
                score = ev.iou(cand, box)
            elif rule == "near":
                score = -dist
            elif rule == "density":
                area = max((cand[2] - cand[0]) * (cand[3] - cand[1]), 1.0)
                score = (g.weights_[cid] / area) * proximity
            elif rule == "persistence":
                score = persistence[cid] * proximity
            else:
                raise ValueError(f"unknown rule: {rule}")

            if score > best_score:
                best_score, best = score, cand
        if best is None:
            skip()
            continue

        if keep_size:
            best = _recentre(*_centre(best), init_w, init_h)

        new_box = tuple(alpha * np.array(best) + (1 - alpha) * np.array(box))
        ncx, ncy = _centre(new_box)
        vel = np.array([ncx - cx0, ncy - cy0])
        box = new_box
        preds.append(box)
        truths.append(ev.gt_at(boxes, start + window / 2))
        times.append(start)

    return preds, truths, np.array(times)


def score(preds, truths, ceilings=None):
    """Standard single-object tracking measures, all against the annotations.

    ``ceilings`` is the per-frame best overlap the predicted box *shape* could
    have reached, from ``evaluate.max_iou_quad``. Supplying it adds the relative
    overlap, which is the raw overlap as a fraction of what was achievable given
    that the annotations are rotated quadrilaterals and the box is an upright
    rectangle of a size fixed before the sequence starts.

    The ceiling must come from that fixed constraint and never from the box the
    model has just produced, or a method is rewarded for predicting a badly
    sized box by being handed a lower bar.
    """
    present = np.array([ev.is_present(g) for g in truths])
    n_absent = int((~present).sum())
    if present.any():
        preds = [p for p, ok in zip(preds, present) if ok]
        truths = [g for g, ok in zip(truths, present) if ok]
        if ceilings is not None:
            ceilings = np.asarray(ceilings, dtype=float)[: len(present)][present]

    ious = np.array([ev.iou(p, g) for p, g in zip(preds, truths)])
    centres = np.array(
        [
            np.hypot((p[0] + p[2]) / 2 - (g[0] + g[2]) / 2, (p[1] + p[3]) / 2 - (g[1] + g[3]) / 2)
            for p, g in zip(preds, truths)
        ]
    )
    _, _, auc = ev.success_curve(ious)
    out = {
        "absent_frames": n_absent,
        "mean_iou": float(ious.mean()),
        "success_50": float((ious > 0.5).mean()),
        "success_auc": auc,
        "precision_20px": float((centres < 20).mean()),
        "median_centre_error": float(np.median(centres)),
        "frames": len(ious),
        "ious": ious,
    }
    if ceilings is not None:
        ceilings = np.asarray(ceilings, dtype=float)[: len(ious)]
        rel, ok = ev.relative_iou(ious, ceilings)
        out["mean_ceiling"] = (
            float(ceilings[ceilings > 1e-3].mean()) if (ceilings > 1e-3).any() else 0.0
        )
        out["relative_iou"] = rel
        out["relative_success_50"] = (
            float((ious[ok] / ceilings[ok] > 0.5).mean()) if ok.any() else 0.0
        )
        out["scored_frames"] = int(ok.sum())
    return out
