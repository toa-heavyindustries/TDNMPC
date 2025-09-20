"""Tests for baseline network planning helpers."""

from __future__ import annotations

import numpy as np
import pytest

import pandapower as pp

from dso.network import build_cigre_feeder
from sim import assemble_baseline_network, plan_baseline_coupled_system


def test_build_cigre_feeder_scales_loads() -> None:
    feeder = build_cigre_feeder(target_peak_mw=35.0, cos_phi=0.95)
    total_p = feeder.net.load.p_mw.sum()
    total_q = feeder.net.load.q_mvar.sum()
    assert total_p == pytest.approx(35.0, rel=1e-3)

    apparent = np.sqrt(total_p**2 + total_q**2)
    cos_phi = total_p / apparent if apparent else 1.0
    assert cos_phi == pytest.approx(0.95, abs=1e-3)


def test_plan_baseline_coupled_system_marks_boundaries() -> None:
    plan = plan_baseline_coupled_system()
    assert plan.attachments, "Expected at least one DSO attachment"

    boundary_flags = plan.tso.net.bus.get("is_boundary")
    flagged = (
        int(boundary_flags.astype(bool).sum())
        if boundary_flags is not None
        else 0
    )
    assert flagged == len(plan.attachments)

    for attachment in plan.attachments:
        kwargs = attachment.transformer_kwargs
        expected_lv = attachment.feeder.net.bus.at[attachment.feeder.root_bus, "vn_kv"]
        assert kwargs["vn_lv_kv"] == pytest.approx(expected_lv)
        assert "sn_mva" in kwargs


def test_assemble_baseline_network_runpp() -> None:
    plan = plan_baseline_coupled_system(boundary_buses=[16, 18, 21])
    net = assemble_baseline_network(plan)

    pp.runpp(net, algorithm="nr", numba=False)
    voltages = net.res_bus.vm_pu
    assert voltages.min() >= 0.93
    assert voltages.max() <= 1.07
