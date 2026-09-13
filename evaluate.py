"""Ground-truth geometry and overlap metrics for event-camera tracking.

Includes axis-aligned IoU, quadrilateral overlap ceilings, success curves,
chosen/oracle component scores, and target-event coverage."""

import numpy as np


def gt_box(row):
    """Axis-aligned box from one ground-truth row: timestamp then four corners."""
    xs, ys = row[1::2], row[2::2]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def gt_quad(row):
    """The four annotated corners of one ground-truth row, unmodified."""
    xs, ys = row[1::2], row[2::2]
    return [(float(a), float(b)) for a, b in zip(xs, ys)]


def gt_at(boxes, t_us):
    """The ground-truth box whose timestamp is nearest to ``t_us``."""
    return gt_box(boxes[int(np.abs(boxes[:, 0] - t_us).argmin())])


def gt_quad_at(boxes, t_us):
    """The ground-truth quadrilateral whose timestamp is nearest to ``t_us``."""
    return gt_quad(boxes[int(np.abs(boxes[:, 0] - t_us).argmin())])


def is_present(box):
    """Whether the annotation says the target is visible in this frame.

    TrackingDataset marks a fully occluded or absent target with a row of NaN
    corners; 15 of its 67 sequences contain them, and Other/drunk2 is more than
    half absent. Those frames say nothing about a tracker and must be dropped,
    not scored as a miss, which is what happens if the NaN is allowed to fall
    through ``iou`` and come back as 0.0. VOT has no such rows.
    """
    return box is not None and not any(v != v for v in box)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    union = max((a[2] - a[0]) * (a[3] - a[1]), 0) + max((b[2] - b[0]) * (b[3] - b[1]), 0) - inter
    return inter / union if union > 0 else 0.0


def _poly_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def _clip_rect(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a convex polygon by an axis-aligned rectangle."""
    for kind, val in ((0, x0), (1, x1), (2, y0), (3, y1)):
        if not poly:
            return poly
        out = []
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            if kind == 0:
                ina, inb = ax >= val, bx >= val
            elif kind == 1:
                ina, inb = ax <= val, bx <= val
            elif kind == 2:
                ina, inb = ay >= val, by >= val
            else:
                ina, inb = ay <= val, by <= val
            if ina:
                out.append((ax, ay))
            if ina != inb:
                if kind < 2:
                    s = (val - ax) / (bx - ax)
                    out.append((val, ay + s * (by - ay)))
                else:
                    s = (val - ay) / (by - ay)
                    out.append((ax + s * (bx - ax), val))
        poly = out
    return poly


def max_iou_quad(quad, w, h, levels=6, grid=7):
    """Estimate the best IoU of an upright w-by-h box against a quadrilateral.

    Searches translation using a finite coarse-to-fine grid. This is a numerical
    estimate of the fixed-shape ceiling, not a guaranteed global optimum."""
    area_r = float(w) * float(h)
    area_q = _poly_area(quad)
    if area_r <= 0 or area_q <= 0:
        return 0.0

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    gx0, gx1 = min(xs) - w / 2, max(xs) + w / 2
    gy0, gy1 = min(ys) - h / 2, max(ys) + h / 2

    best, bx, by = 0.0, (gx0 + gx1) / 2, (gy0 + gy1) / 2
    for _ in range(levels):
        sx = (gx1 - gx0) / (grid - 1)
        sy = (gy1 - gy0) / (grid - 1)
        for i in range(grid):
            cx = gx0 + i * sx
            for j in range(grid):
                cy = gy0 + j * sy
                clipped = _clip_rect(quad, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                a = _poly_area(clipped) if len(clipped) >= 3 else 0.0
                if a > best:
                    best, bx, by = a, cx, cy
        gx0, gx1 = bx - sx, bx + sx
        gy0, gy1 = by - sy, by + sy

    return best / (area_r + area_q - best)


def max_iou_aabb(box, w, h):
    """Closed-form ceiling against an axis-aligned ground-truth box.

    Cheaper than ``max_iou_quad`` and removes only the fixed-size handicap, not
    the cost of squaring off a rotated annotation. Kept for comparison.
    """
    W, H = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0 or W <= 0 or H <= 0:
        return 0.0
    inter = min(w, W) * min(h, H)
    return inter / (w * h + W * H - inter)


def relative_iou(ious, ceilings, floor=1e-3):
    """Overlap as a fraction of what the box shape could have achieved.

    Frames whose ceiling is essentially zero carry no information about the
    method and are dropped rather than being allowed to divide by nothing.
    """
    ious = np.asarray(ious, dtype=float)
    ceilings = np.asarray(ceilings, dtype=float)
    ok = ceilings > floor
    if not ok.any():
        return 0.0, ok
    return float(np.mean(ious[ok] / ceilings[ok])), ok


def success_curve(ious, thresholds=None):
    """Share of frames above each overlap threshold, and the area under it."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    ious = np.asarray(ious)
    curve = np.array([(ious > th).mean() for th in thresholds])
    return thresholds, curve, float(curve.mean())


def events_inside(box, xs, ys):
    """Share of a frame's events falling inside a box."""
    if box is None:
        return 0.0
    return float(((xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])).mean())


def score_sequence(pg_factory, x, y, t, boxes, window, rule="persdens+hyst"):
    """Run the fovea over one sequence and score every frame against ground truth.

    Returns the per-frame IoU of the chosen component, the per-frame IoU of the
    best available component, and the share of events that were inside the
    ground-truth box in the first place.
    """
    import foveanet as fn

    pg = pg_factory()
    chosen, oracle, inside = [], [], []
    for s, xs, ys, ts in fn.iter_frames(x, y, t, window=window):
        coords = np.column_stack([xs, ys]).astype(np.float32)
        pg.update(coords)
        truth = gt_at(boxes, s + window / 2)
        inside.append(events_inside(truth, xs, ys))
        cid = pg.select(rule=rule)
        chosen.append(iou(pg.component_box(cid, coords), truth))
        oracle.append(max(iou(pg.component_box(c, coords), truth) for c in range(pg.k)))
    return np.array(chosen), np.array(oracle), np.array(inside)


def dominant_frequencies(t, n=5, bin_us=1000):
    """Strongest frequencies in the event rate, for spotting display artifacts."""
    bins = np.arange(t.min(), t.max(), bin_us)
    counts, _ = np.histogram(t, bins=bins)
    spectrum = np.abs(np.fft.rfft(counts - counts.mean()))
    freqs = np.fft.rfftfreq(len(counts), d=bin_us / 1e6)
    order = np.argsort(spectrum)[::-1][:n]
    peak = spectrum.max()
    return [(float(freqs[i]), float(spectrum[i] / peak)) for i in order]
