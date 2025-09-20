"""Tests for the IEEE33 DSO network and LinDistFlow linearisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dso.network import ac_power_flow, build_ieee33, export_net, load_net
from models.lindistflow import linearize_lindistflow, predict_voltage_delta, validate_linearization


def test_export_and_load_roundtrip(tmp_path: Path) -> None:
    net = build_ieee33()
    path = tmp_path / "ieee33.json"
    export_net(net, path)

    loaded = load_net(path)
    assert len(loaded.bus) == len(net.bus)
    assert len(loaded.line) == len(net.line)


def test_linearization_accuracy() -> None:
    net = build_ieee33()
    base = ac_power_flow(net)
    sens = linearize_lindistflow(net, base)

    stats = validate_linearization(
        net,
        sens,
        n_samples=30,
        tol=0.03,
        max_delta_mw=0.5,
    )
    assert stats["mape"] <= 0.03
    assert stats["max_abs"] <= 0.06
    assert stats["passed"]


def test_predict_voltage_delta_shapes() -> None:
    net = build_ieee33()
    base = ac_power_flow(net)
    sens = linearize_lindistflow(net, base)

    n = len(net.load)
    dp = np.zeros(n)
    dq = np.zeros(n)
    delta = predict_voltage_delta(sens, dp, dq)
    assert delta.shape == (n,)
