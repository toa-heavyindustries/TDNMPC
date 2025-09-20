"""Tests for envelope construction utilities."""

from __future__ import annotations

from pathlib import Path

from dso.network import build_cigre_feeder
from interface.adapter import measure_boundary
from interface.envelope import BoxEnvelope, build_box_envelope, export_envelopes


def test_build_box_envelope_bounds() -> None:
    feeder = build_cigre_feeder("mv", target_peak_mw=20.0)
    env = build_box_envelope(feeder, p_margin=5.0, q_margin=3.0)
    p0, q0, _ = measure_boundary(feeder.net)
    assert env.p_min < p0 < env.p_max
    assert env.q_min < q0 < env.q_max
    assert env.v_min == 0.95
    assert env.v_max == 1.05
    assert env.metadata["target_peak_mw"] == feeder.target_peak_mw


def test_export_envelopes(tmp_path: Path) -> None:
    feeder = build_cigre_feeder("mv", target_peak_mw=20.0)
    env = build_box_envelope(feeder, p_margin=5.0, q_margin=3.0)
    out = tmp_path / "boundary.json"
    export_envelopes([env], out)
    assert out.exists()
    data = out.read_text()
    assert "\"type\": \"box\"" in data
