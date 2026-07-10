import math

import pytest
import torch

from eval.metrics import angular_error_by_energy, mean_angular_error


def test_identical_directions_give_zero_error():
    # float32 acos near +/-1 is numerically sensitive (a consistent ~0.03deg
    # error from float32 rounding in the dot product, verified by direct
    # testing) -- tolerance is loose relative to that, still far tighter
    # than any practical angular-resolution use case.
    az = torch.tensor([0.5, 1.0, 3.0])
    zen = torch.tensor([0.5, 1.0, 1.5])
    err = mean_angular_error(az, zen, az, zen)
    assert err == pytest.approx(0.0, abs=0.1)


def test_known_90_degree_separation():
    # zenith=pi/2 puts both vectors in the xy-plane; azimuth 0 vs pi/2 is a
    # 90-degree great-circle separation.
    pred_az = torch.tensor([0.0])
    pred_zen = torch.tensor([math.pi / 2])
    true_az = torch.tensor([math.pi / 2])
    true_zen = torch.tensor([math.pi / 2])
    err = mean_angular_error(pred_az, pred_zen, true_az, true_zen)
    assert err == pytest.approx(90.0, abs=1e-2)


def test_opposite_directions_give_180_degrees():
    pred_az = torch.tensor([0.0])
    pred_zen = torch.tensor([0.0])  # north pole
    true_az = torch.tensor([0.0])
    true_zen = torch.tensor([math.pi])  # south pole
    err = mean_angular_error(pred_az, pred_zen, true_az, true_zen)
    assert err == pytest.approx(180.0, abs=0.1)


def test_angular_error_by_energy_bins_correctly():
    n = 20
    az = torch.zeros(n)
    zen = torch.zeros(n)
    log_energies = torch.linspace(2.0, 6.9, n)
    result = angular_error_by_energy(az, zen, az, zen, log_energies)
    assert len(result) > 0
    assert all(v == pytest.approx(0.0, abs=0.1) for v in result.values())
    for bin_center in result:
        assert 2.0 <= bin_center <= 7.0


def test_angular_error_by_energy_skips_empty_bins():
    az = torch.zeros(3)
    zen = torch.zeros(3)
    log_energies = torch.tensor([2.1, 2.2, 2.3])  # all in the first bin only
    result = angular_error_by_energy(az, zen, az, zen, log_energies)
    assert len(result) == 1
