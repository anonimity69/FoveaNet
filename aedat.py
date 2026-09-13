"""Read jAER AEDAT-2.0 recordings from a 128 x 128 DVS.

Each event is an eight-byte big-endian address/timestamp pair. The address
stores polarity in bit 0, x in bits 1-7 and y in bits 8-14."""

import numpy as np


def read_aedat2(path, sensor=128, flip_x=True, file_order=False):
    """Decode one AEDAT-2.0 recording into ``(x, y, t, p)``.

    ``file_order`` returns the events exactly as they sit in the file, with no
    sorting and no dropping of out-of-range addresses. SL-Animals-DVS needs
    that: its tag files mark where each sign begins and ends by *event index*
    rather than by timestamp, so anything that reorders or removes events
    invalidates the boundaries before they can be applied.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    pos = 0
    while pos < len(raw) and raw[pos : pos + 1] == b"#":
        end = raw.find(b"\n", pos)
        if end < 0:
            break
        pos = end + 1

    body = raw[pos:]
    body = body[: len(body) - (len(body) % 8)]
    rec = np.frombuffer(body, dtype=">u4").reshape(-1, 2)
    addr = rec[:, 0]
    t = rec[:, 1].astype(np.int64)

    p = (addr & 1).astype(np.int64)
    x = ((addr >> 1) & 0x7F).astype(np.int64)
    y = ((addr >> 8) & 0x7F).astype(np.int64)
    if flip_x:
        x = (sensor - 1) - x

    if file_order:
        return x, y, t, p

    keep = (x >= 0) & (x < sensor) & (y >= 0) & (y < sensor)
    x, y, t, p = x[keep], y[keep], t[keep], p[keep]

    order = np.argsort(t, kind="stable")
    return x[order], y[order], t[order], p[order]
