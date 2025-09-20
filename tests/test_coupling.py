"""Tests for TSO-DSO coupling interface utilities."""

from __future__ import annotations

import numpy as np
import pytest

from coord.interface import (
    aggregate_dsos_to_tso,
    coupling_residuals,
    define_coupling,
    push_tso_signals_to_dsos,
)
from dso.network import build_ieee33
from tso.network import build_tso_case, mark_boundary_buses


@pytest.fixture()
def sample_coupling():
    case = build_tso_case(6)
    case = mark_boundary_buses(case, [1, 3])
    dso_nets = [build_ieee33(), build_ieee33()]
    mapping = {1: (0, 5), 3: (1, 10)}
    coupler = define_coupling(case, dso_nets, mapping)
    return coupler, case, dso_nets


def test_define_coupling_sets_slots(sample_coupling) -> None:
    coupler, _, dso_nets = sample_coupling
    assert coupler["n_interfaces"] == 2
    assert np.array_equal(coupler["tso_buses"], np.array([1, 3]))
    assert dso_nets[0]["coupling"]["slots"].tolist() == [0]
    assert dso_nets[1]["coupling"]["slots"].tolist() == [1]


def test_push_and_aggregate_roundtrip(sample_coupling) -> None:
    coupler, _, dso_nets = sample_coupling
    signals = np.array([5.0, 7.5])
    push_tso_signals_to_dsos(signals, dso_nets)
    assert np.allclose(dso_nets[0]["coupling"]["p_tso"], np.array([5.0]))
    assert np.allclose(dso_nets[1]["coupling"]["p_tso"], np.array([7.5]))

    for net in dso_nets:
        coupling = net["coupling"]
        coupling["p_response"] = coupling["p_tso"] * 1.1

    aggregated = aggregate_dsos_to_tso(dso_nets)
    assert np.allclose(aggregated, signals * 1.1)


def test_coupling_residuals_metrics() -> None:
    tso = np.array([10.0, -5.0])
    dso = np.array([8.0, -4.5])
    metrics = coupling_residuals(tso, dso)
    assert metrics["max"] == pytest.approx(2.0)
    assert metrics["mean"] == pytest.approx(1.25)
    assert metrics["l2"] == pytest.approx(np.linalg.norm(tso - dso))

