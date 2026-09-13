"""Dataset adapters returning structured (x, y, t, p) event arrays.

Readers cover POKER-DVS, SL-Animals-DVS and the Hu et al. DVS benchmark.
Sensor sizes follow Tonic's (width, height, polarities) convention and timestamps
are microseconds. DVS-Lip is loaded through tonic.datasets.DVSLip."""

from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])


def to_structured(x, y, t, p, dtype=EVENT_DTYPE):
    """Pack four parallel arrays into the structured array Tonic expects."""
    order = np.argsort(t, kind="stable")
    ev = np.empty(len(t), dtype=dtype)
    ev["x"] = np.asarray(x)[order]
    ev["y"] = np.asarray(y)[order]
    ev["t"] = np.asarray(t)[order]
    ev["p"] = np.asarray(p)[order]
    return ev


class EventDataset:
    """Minimal Tonic-compatible dataset.

    Subclasses set ``sensor_size`` and fill ``self.data`` with whatever handle
    identifies a recording, ``self.targets`` with its label, then implement
    ``load`` to return ``(x, y, t, p)`` for one handle.
    """

    sensor_size = None
    dtype = EVENT_DTYPE
    ordering = EVENT_DTYPE.names

    def __init__(self, transform=None, target_transform=None, transforms=None):
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.data = []
        self.targets = []

    def load(self, handle):
        raise NotImplementedError

    def __getitem__(self, index):
        events = to_structured(*self.load(self.data[index]), dtype=self.dtype)
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)} recordings, sensor={self.sensor_size})"


class PokerDVSCards(EventDataset):
    """The three original 128x128 POKER-DVS recordings of cards being browsed.

    Tonic ships POKER-DVS as pre-extracted 32x32 pip symbols, which have already
    been tracked and cropped. These are the raw recordings the symbols were cut
    from, downloaded from the IMSE-CNM lab page, and are the ones to use if the
    tracking itself is what you want to test.
    """

    sensor_size = (128, 128, 2)

    def __init__(self, root="./data/POKERDVS", **kw):
        super().__init__(**kw)
        files = sorted(Path(root).glob("cards_*.aedat"))
        if not files:
            raise SystemExit(
                f"no cards_*.aedat under {root}. Fetch them with:\n"
                "  curl -O http://www2.imse-cnm.csic.es/caviar/POKER_DVS/cards_1.aedat"
            )
        self.data = files
        self.names = [f.stem for f in files]
        self.targets = list(range(len(files)))

    def load(self, handle):
        import aedat

        return aedat.read_aedat2(str(handle), sensor=128)


class DVSBenchmark(EventDataset):
    """VOT Challenge 2015 and TrackingDataset recordings from Hu et al. (2016).

    One HDF5 holds every sequence. Each carries its events and a
    ``bounding_box`` array whose rows are a timestamp followed by the four
    corners of the ground-truth quadrilateral, already mapped into sensor
    coordinates. The target returned here is that array, so IoU can be scored
    directly against a predicted box.

    The two files are laid out differently: VOT puts its 60 sequences at the top
    level, TrackingDataset nests its 67 one level deeper under the name of the
    paper each came from. Both are walked here, so a sequence is addressed by
    its path, for example ``bag`` or ``Babenko/girl``.
    """

    sensor_size = (240, 180, 2)

    def __init__(self, h5_path, sequences=None, **kw):
        super().__init__(**kw)
        import h5py

        self.h5_path = str(h5_path)
        found = []

        def walk(group, path=""):
            for key, node in group.items():
                sub = f"{path}/{key}" if path else key
                if isinstance(node, h5py.Group):
                    if "x_pos" in node:
                        found.append(sub)
                    else:
                        walk(node, sub)

        with h5py.File(self.h5_path, "r") as f:
            walk(f)
            found.sort()
            self._boxes = {n: f[n]["bounding_box"][:] for n in found}
        self.data = [n for n in found if sequences is None or n in sequences]
        self.targets = [self._boxes[n] for n in self.data]

    def load(self, handle):
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            g = f[handle]
            return (
                g["x_pos"][:].astype(np.int64),
                g["y_pos"][:].astype(np.int64),
                g["timestamps"][:].astype(np.int64),
                g["pol"][:].astype(np.int64),
            )


class SLAnimals(EventDataset):
    """SL-Animals-DVS: 59 recordings of signers naming 19 animals.

    Unlike every other dataset here, a recording is not a sample. Each subject
    performs all 19 signs in one continuous take, and a matching CSV marks where
    each one begins and ends. Those boundaries are EVENT INDICES rather than
    timestamps, which is why the AEDAT is read in file order: sorting by time or
    dropping out-of-range addresses would shift every index before it could be
    used. Filtering happens after the slice instead.

    Four recording scenarios are present and are worth keeping track of, because
    they are lighting conditions rather than repetitions: indoor (17), dc (25),
    imse (10) and sunlight (7). ``subject`` and ``scenario`` are exposed so a
    split can hold out people rather than samples, as the gesture work does.

        ds = SLAnimals("./data/SLANIMALS")
        events, target = ds[0]
    """

    sensor_size = (128, 128, 2)

    def __init__(self, root="./data/SLANIMALS", scenarios=None, **kw):
        super().__init__(**kw)
        root = Path(root)
        defs = root / "SL-Animals-DVS_gestures_definitions.csv"
        self.classes = []
        if defs.is_file():
            for line in defs.read_text().splitlines()[1:]:
                if "," in line:
                    self.classes.append(line.split(",", 1)[1].strip())

        self.subject, self.scenario = [], []
        for rec in sorted(root.glob("user*.aedat")):
            tags = rec.with_suffix(".csv")
            if not tags.is_file():
                continue
            stem = rec.stem
            who, _, where = stem.partition("_")
            if scenarios is not None and where not in scenarios:
                continue
            for line in tags.read_text().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                cls, lo, hi = int(parts[0]), int(parts[1]), int(parts[2])
                if hi <= lo:
                    continue
                self.data.append((rec, lo, hi))
                self.targets.append(cls - 1)
                self.subject.append(int(who.replace("user", "")))
                self.scenario.append(where)

        if not self.data:
            raise SystemExit(
                f"no SL-Animals recordings under {root}. Expected user*.aedat "
                "beside matching user*.csv tag files."
            )

    _cache = (None, None)

    def load(self, handle):
        import aedat

        rec, lo, hi = handle
        # Cache one recording because consecutive samples share an AEDAT file.
        if SLAnimals._cache[0] != rec:
            SLAnimals._cache = (rec, aedat.read_aedat2(str(rec), file_order=True))
        x, y, t, p = SLAnimals._cache[1]
        hi = min(hi, len(x))
        x, y, t, p = x[lo:hi], y[lo:hi], t[lo:hi], p[lo:hi]
        keep = (x >= 0) & (x < 128) & (y >= 0) & (y < 128)
        x, y, t, p = x[keep], y[keep], t[keep], p[keep]
        order = np.argsort(t, kind="stable")
        return x[order], y[order], t[order], p[order]


# Use tonic.datasets.DVSLip(save_to="data"). Its released 128 x 128 samples
# are already mouth crops, unlike the full 346 x 260 acquisition sensor.


def from_arrays(records, sensor_size, targets=None):
    """Wrap (x, y, t, p) arrays as a Tonic-compatible dataset."""

    class _Wrapped(EventDataset):
        pass

    ds = _Wrapped()
    _Wrapped.sensor_size = sensor_size
    ds.data = list(records)
    ds.targets = list(targets) if targets is not None else [0] * len(ds.data)
    ds.load = lambda handle: handle
    return ds
