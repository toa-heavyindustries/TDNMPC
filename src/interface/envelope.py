"""Boundary envelope construction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandapower as pp

from dso.network import DsoFeeder
from interface.adapter import BoundaryMeasurement, measure_boundary


@dataclass(slots=True)
class BoxEnvelope:
    """Simple rectangular feasible region for boundary power exchange."""

    p_min: float
    p_max: float
    q_min: float
    q_max: float
    v_min: float
    v_max: float
    base: BoundaryMeasurement
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "box",
            "p_min": self.p_min,
            "p_max": self.p_max,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "voltage_limits": [self.v_min, self.v_max],
            "base": {
                "p_mw": self.base.p_mw,
                "q_mvar": self.base.q_mvar,
                "v_pu": self.base.v_pu,
            },
            "metadata": self.metadata,
        }


def build_box_envelope(
    feeder: DsoFeeder,
    *,
    net: pp.pandapowerNet | None = None,
    measurement: BoundaryMeasurement | None = None,
    p_margin: float | None = None,
    q_margin: float | None = None,
    voltage_limits: tuple[float, float] = (0.95, 1.05),
) -> BoxEnvelope:
    """Construct a conservative rectangular envelope for a feeder boundary."""

    net = feeder.net if net is None else net

    if measurement is None:
        p_mw, q_mvar, v_pu = measure_boundary(net)
        measurement = BoundaryMeasurement(
            p_mw=p_mw,
            q_mvar=q_mvar,
            v_pu=v_pu,
            hv_bus=int(net.trafo.iloc[0].hv_bus) if not net.trafo.empty else 0,
            trafo_index=int(net.trafo.index[0]) if not net.trafo.empty else -1,
        )

    base_p = measurement.p_mw
    base_q = measurement.q_mvar

    if p_margin is None:
        p_margin = feeder.target_peak_mw
    if p_margin <= 0:
        p_margin = max(abs(base_p), 1.0)

    if q_margin is None:
        cos_phi = float(np.clip(feeder.cos_phi, 0.1, 0.999))
        tan_phi = np.sqrt(1.0 / (cos_phi * cos_phi) - 1.0)
        q_margin = feeder.target_peak_mw * tan_phi
    if q_margin <= 0:
        q_margin = max(abs(base_q), 0.5)

    envelope = BoxEnvelope(
        p_min=float(base_p - p_margin),
        p_max=float(base_p + p_margin),
        q_min=float(base_q - q_margin),
        q_max=float(base_q + q_margin),
        v_min=float(voltage_limits[0]),
        v_max=float(voltage_limits[1]),
        base=measurement,
        metadata={
            "target_peak_mw": feeder.target_peak_mw,
            "cos_phi": feeder.cos_phi,
            "hv_bus": measurement.hv_bus,
            "trafo_index": measurement.trafo_index,
        },
    )
    return envelope


def export_envelopes(envelopes: Sequence[BoxEnvelope], path: Path | str) -> None:
    """Write envelope definitions to JSON."""

    data = {
        "type": "collection",
        "elements": [env.to_dict() for env in envelopes],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
