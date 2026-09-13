import numpy as np
import pytest

import build_crops as crops
import foveanet as fn
from snn_tracker import MembraneSurface, ncc_map


def test_persistence_and_hysteresis():
    rng = np.random.default_rng(12)
    coords = np.concatenate([rng.normal(c, 1, (40, 2)) for c in [(10, 10), (40, 40), (80, 80)]])
    model = fn.PersistentGMM()
    model.update(coords)
    expected = (1 - fn.DECAY) * model.model.weights_
    np.testing.assert_allclose(model.persistence, expected)
    old = model.persistence.copy()
    model.update(coords + 0.1)
    np.testing.assert_allclose(
        model.persistence, fn.DECAY * old + (1 - fn.DECAY) * model.model.weights_
    )
    model.persistence = np.array([1.0, 1.1, 0.1])
    model.fovea_id = 0
    assert model.select("persistence+hyst") == 0
    model.persistence[1] = 1.2
    assert model.select("persistence+hyst") == 1
    assert np.isfinite(model.saliency_map()).all()


def test_crop_counts_and_periphery_budget():
    x, y, p = np.array([1, 2, 20]), np.array([1, 2, 20]), np.array([0, 1, 1])
    tensor = crops.blank()
    crops.add_crop(tensor, 0, x, y, p, (0, 0, 4, 4))
    assert tensor.shape == (8, 2, 32, 32)
    assert tensor.sum() == 2
    tensor = crops.blank()
    crops.add_fovea_periphery(tensor, 0, x, y, p, (0, 0, 4, 4), (128, 128))
    assert tensor[:, :, :16].sum() == 2
    assert tensor[:, :, 16:].sum() == 3


def test_synthetic_recording_all_conditions():
    rng = np.random.default_rng(9)
    xy = np.concatenate([rng.normal(c, 2, (90, 2)) for c in [(25, 25), (60, 60), (90, 90)]])
    xy = np.clip(xy.astype(int), 0, 127)
    t = np.arange(len(xy)) * 100
    output = crops.process(xy[:, 0], xy[:, 1], t, np.arange(len(xy)) % 2, 10000, rng)
    assert set(output) == set(crops.CONDITIONS)
    for tensor in output.values():
        assert tensor.shape == (8, 2, 32, 32)
        assert np.isfinite(tensor).all()
        assert (tensor >= 0).all()


def test_surface_decay_and_template_match():
    surface = MembraneSurface(sensor=(8, 8), tau_us=100)
    surface.update(np.array([2]), np.array([3]), 0)
    surface.update(np.array([], dtype=int), np.array([], dtype=int), 100)
    assert surface.v[3, 2] == pytest.approx(np.exp(-1))
    rng = np.random.default_rng(5)
    image = rng.normal(size=(12, 12)).astype(np.float32)
    template = image[3:7, 5:9].copy()
    response = ncc_map(image, template)
    assert np.unravel_index(response.argmax(), response.shape) == (3, 5)
    assert response[3, 5] == pytest.approx(1, abs=1e-5)
