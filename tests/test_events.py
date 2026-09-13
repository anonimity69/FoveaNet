import struct

import numpy as np

from aedat import read_aedat2
from tonic_convert import DVSBenchmark, from_arrays


def test_aedat_decoding_sorting_and_file_order(tmp_path):
    path = tmp_path / "events.aedat"
    payload = b"#!AER-DAT2.0\n" + b"".join(
        struct.pack(">II", (y << 8) | (x << 1) | p, t)
        for x, y, t, p in [(3, 4, 20, 1), (5, 6, 10, 0)]
    )
    path.write_bytes(payload)
    x, y, t, p = read_aedat2(path)
    np.testing.assert_array_equal(x, [122, 124])
    np.testing.assert_array_equal(y, [6, 4])
    np.testing.assert_array_equal(t, [10, 20])
    np.testing.assert_array_equal(p, [0, 1])
    np.testing.assert_array_equal(read_aedat2(path, file_order=True)[2], [20, 10])


def test_array_dataset_preserves_events_and_target():
    arrays = tuple(np.array(v) for v in ([1, 2], [3, 4], [5, 6], [0, 1]))
    ds = from_arrays([arrays], sensor_size=(128, 128, 2), targets=[7])
    events, target = ds[0]
    assert target == 7
    assert set(events.dtype.names) == {"x", "y", "t", "p"}
    np.testing.assert_array_equal(events["t"], [5, 6])


def test_nested_benchmark_hdf5(tmp_path):
    import h5py

    path = tmp_path / "benchmark.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("source/sequence")
        for name, value in {
            "x_pos": [1, 2],
            "y_pos": [3, 4],
            "timestamps": [10, 20],
            "pol": [0, 1],
            "bounding_box": [[10, 0, 0, 4, 0, 4, 4, 0, 4]],
        }.items():
            group.create_dataset(name, data=value)
    ds = DVSBenchmark(path)
    assert ds.data == ["source/sequence"]
    events, boxes = ds[0]
    assert len(events) == 2
    assert boxes.shape == (1, 9)
