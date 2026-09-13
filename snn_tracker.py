"""Appearance tracking with an event surface and an snnTorch spiking readout.

A frame-decayed event-count surface supplies patches for normalised
cross-correlation. Readout neurons accumulate match scores across frames.
The first ground-truth box initialises the template; subsequent annotations
are used only for evaluation. See tracker.py for the GMM baseline."""

import numpy as np
import snntorch as snn
import torch
import torch.nn.functional as F

import evaluate as ev

TAU_US = 50_000
SEARCH_SCALE = 2.5
MIN_EVENTS = 30
BETA = 0.7
THRESHOLD = 0.5
TEMPLATE_LR = 0.05


class MembraneSurface:
    """Frame-decayed event-count surface.

    Old potential decays by exp(-dt/tau); all events in the current frame then
    add unit charge. Intra-frame event times are not represented, so this is
    not exact event-by-event exponential integration."""

    def __init__(self, sensor=(240, 180), tau_us=TAU_US):
        self.w, self.h = sensor
        self.tau = float(tau_us)
        self.v = np.zeros((self.h, self.w), dtype=np.float32)
        self.t = None

    def update(self, xs, ys, t_now):
        if self.t is not None:
            dt = max(float(t_now) - self.t, 0.0)
            self.v *= float(np.exp(-dt / self.tau))
        self.t = float(t_now)
        if len(xs):
            np.add.at(
                self.v,
                (
                    np.clip(ys, 0, self.h - 1).astype(np.int64),
                    np.clip(xs, 0, self.w - 1).astype(np.int64),
                ),
                1.0,
            )
        return self.v

    def patch(self, box):
        x0, y0, x1, y1 = [int(round(c)) for c in box]
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return self.v[y0:y1, x0:x1].copy()


def ncc_map(surface, template):
    """Normalised cross-correlation of ``template`` over ``surface``.

    The response at each position is the correlation between the template and
    the equally sized patch centred there: bounded in [-1, 1] and independent of
    how much activity the patch holds, which is what separates this from a
    density measure.
    """
    th, tw = template.shape
    sh, sw = surface.shape
    if sh < th or sw < tw or th < 3 or tw < 3:
        return None

    s = torch.from_numpy(np.ascontiguousarray(surface)).float()[None, None]
    t = torch.from_numpy(np.ascontiguousarray(template)).float()
    t = t - t.mean()
    t_norm = float(torch.sqrt((t * t).sum()))
    if t_norm < 1e-6:
        return None
    t = (t / t_norm)[None, None]

    ones = torch.ones((1, 1, th, tw))
    n = float(th * tw)
    ssum = F.conv2d(s, ones)
    ssq = F.conv2d(s * s, ones)
    num = F.conv2d(s, t)
    var = ssq - ssum * ssum / n
    denom = torch.sqrt(torch.clamp(var, min=1e-8))
    return (num / denom)[0, 0].numpy()


class SNNAppearanceTracker:
    """Template matching with a spiking readout layer.

    ``beta`` is the readout neuron's membrane decay: 0 decides from the current
    frame alone, closer to 1 accumulates evidence across more frames.
    ``template_lr`` lets the stored appearance drift toward what is currently
    matched, which is needed because a target's event pattern changes with its
    speed and pose.
    """

    def __init__(
        self,
        sensor=(240, 180),
        tau_us=TAU_US,
        beta=BETA,
        threshold=THRESHOLD,
        template_lr=TEMPLATE_LR,
        search_scale=SEARCH_SCALE,
        smooth=2.0,
        anchor_w=0.0,
        update_gate=-1.0,
    ):
        self.sensor = sensor
        self.surface = MembraneSurface(sensor, tau_us)
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.mem = None
        self.template_lr = template_lr
        self.search_scale = search_scale
        self.smooth = smooth
        self.anchor_w = anchor_w
        self.update_gate = update_gate
        self.template = None
        self.anchor = None
        self.box = None
        self.w = self.h = 0.0
        self.spikes = 0
        self.frames = 0

    def _view(self):
        """Smooth the event surface before matching to reduce pixel-level sparsity."""
        if self.smooth <= 0:
            return self.surface.v
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(self.surface.v, self.smooth)

    def init_from(self, box):
        self.box = tuple(float(c) for c in box)
        self.w = max(box[2] - box[0], 4.0)
        self.h = max(box[3] - box[1], 4.0)
        x0, y0, x1, y1 = [int(round(c)) for c in box]
        v = self._view()
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.sensor[0]), min(y1, self.sensor[1])
        self.template = v[y0:y1, x0:x1].copy() if (x1 - x0 >= 3 and y1 - y0 >= 3) else None
        self.anchor = None if self.template is None else self.template.copy()
        return self.box

    def _window(self, cx, cy):
        hw = self.w * self.search_scale / 2
        hh = self.h * self.search_scale / 2
        return (
            int(max(cx - hw, 0)),
            int(max(cy - hh, 0)),
            int(min(cx + hw, self.sensor[0])),
            int(min(cy + hh, self.sensor[1])),
        )

    def step(self, xs, ys, t_now, predicted=None):
        """Advance the surface by one frame and return the updated box."""
        self.surface.update(xs, ys, t_now)
        self.frames += 1
        v = self._view()
        if self.template is None or min(self.template.shape) < 3:
            return self.box

        cx, cy = (
            predicted
            if predicted is not None
            else ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)
        )
        cx = float(np.clip(cx, 0, self.sensor[0] - 1))
        cy = float(np.clip(cy, 0, self.sensor[1] - 1))
        wx0, wy0, wx1, wy1 = self._window(cx, cy)
        region = v[wy0:wy1, wx0:wx1]
        if (
            region.size == 0
            or region.shape[0] < self.template.shape[0]
            or region.shape[1] < self.template.shape[1]
        ):
            return self.box
        resp = ncc_map(region, self.template)
        if resp is None or resp.size == 0:
            return self.box

        # Blend match responses with the initial target to limit template drift.
        # Blending patches instead would blur spatially offset appearances.
        if self.anchor_w > 0 and self.anchor is not None:
            resp_a = ncc_map(region, self.anchor)
            if resp_a is not None and resp_a.shape == resp.shape:
                resp = (1 - self.anchor_w) * resp + self.anchor_w * resp_a

        cur = torch.from_numpy(resp).float()
        if self.mem is None or self.mem.shape != cur.shape:
            self.mem = torch.zeros_like(cur)
        spk, self.mem = self.lif(cur, self.mem)

        if float(spk.sum()) == 0:
            return self.box
        self.spikes += 1

        mem = self.mem.numpy()
        ry, rx = np.unravel_index(int(np.argmax(mem)), mem.shape)
        th, tw = self.template.shape
        ncx = float(np.clip(wx0 + rx + tw / 2, 0, self.sensor[0] - 1))
        ncy = float(np.clip(wy0 + ry + th / 2, 0, self.sensor[1] - 1))
        self.box = (ncx - self.w / 2, ncy - self.h / 2, ncx + self.w / 2, ncy + self.h / 2)

        bx0, by0 = max(int(round(self.box[0])), 0), max(int(round(self.box[1])), 0)
        bx1 = min(int(round(self.box[2])), self.sensor[0])
        by1 = min(int(round(self.box[3])), self.sensor[1])
        # A frame whose best match is weak is more likely to be a distractor
        # than the target, so it is not allowed to write into the template.
        if float(resp.max()) < self.update_gate:
            return self.box

        patch = v[by0:by1, bx0:bx1] if (bx1 - bx0 >= 3 and by1 - by0 >= 3) else None
        if patch is not None and patch.shape == self.template.shape:
            self.template = (1 - self.template_lr) * self.template + self.template_lr * patch
        return self.box


def track(x, y, t, boxes, window, sensor=(240, 180), predict=1.0, **kw):
    """Run the spiking appearance tracker over one sequence.

    Same contract as ``tracker.track``: initialised from the benchmark's first
    annotation, never told the answer again, one predicted and one ground-truth
    box per frame.
    """
    first = ev.gt_box(boxes[0])
    trk = SNNAppearanceTracker(sensor=sensor, **kw)

    preds, truths, times = [], [], []
    vel = np.zeros(2)
    prev_c = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    box = first

    for i, start in enumerate(np.arange(t.min(), t.max(), window)):
        sel = (t >= start) & (t < start + window)
        xs, ys = x[sel], y[sel]
        mid = start + window / 2

        if i == 0:
            trk.surface.update(xs, ys, mid)
            trk.init_from(first)
        elif int(sel.sum()) < MIN_EVENTS:
            trk.surface.update(xs, ys, mid)
            box = trk.box
        else:
            pred_c = (prev_c[0] + predict * vel[0], prev_c[1] + predict * vel[1])
            box = trk.step(xs, ys, mid, predicted=pred_c)

        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        vel = np.array([c[0] - prev_c[0], c[1] - prev_c[1]])
        prev_c = c

        preds.append(box)
        truths.append(ev.gt_at(boxes, mid))
        times.append(start)

    return preds, truths, np.array(times)
