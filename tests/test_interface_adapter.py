"""Tests for interface adapter utilities."""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from dso.network import build_cigre_feeder
from interface import apply_reference, measure_boundary, replace_tso_load
from tso.network import build_tso_pandapower


def test_measure_and_apply_reference_tracks_targets() -> None:
    feeder = build_cigre_feeder("mv")

    p0, q0, _ = measure_boundary(feeder.net)
    target_p = p0 - 2.0
    target_q = q0 - 0.5

    result = apply_reference(feeder.net, target_p, target_q)
    assert result.p_mw == pytest.approx(target_p, abs=5e-3)

    p1, q1, _ = measure_boundary(feeder.net)
    assert p1 == pytest.approx(target_p, abs=5e-3)


def test_replace_tso_load_voltage_monotonic() -> None:
    case = build_tso_pandapower()
    net = case.net
    bus = case.boundary_buses[0]

    replace_tso_load(net, bus, p_mw=0.0, q_mvar=0.0)

    prefs = np.linspace(-5.0, 5.0, 11)
    voltages = []
    for pref in prefs:
        replace_tso_load(net, bus, p_mw=-pref, q_mvar=0.0)
        pp.runpp(net, algorithm="nr", numba=False)
        voltages.append(float(net.res_bus.at[bus, "vm_pu"]))

    diffs = np.diff(voltages)
    assert np.all(diffs >= -1e-4)
