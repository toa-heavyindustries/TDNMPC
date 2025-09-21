"""Baseline network assembly helpers aligned with 基座环境.md."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandapower as pp

from dso import DsoFeeder, build_cigre_feeder
from tso import TsoPandapowerCase, build_tso_pandapower

_DEFAULT_FEEDER_VOLTAGE_KV: dict[Literal["mv", "lv"], float] = {
    "mv": 12.47,
    "lv": 0.4,
}


@dataclass(slots=True)
class TransformerSpec:
    """Named set of transformer parameters used at T–D interfaces."""

    sn_mva: float = 60.0
    vk_percent: float = 8.0
    vkr_percent: float = 0.5
    pfe_kw: float = 30.0
    i0_percent: float = 0.1
    shift_degree: float = 0.0

    def instantiate(self, *, hv_kv: float, lv_kv: float) -> dict[str, float]:
        """Return a dict compatible with ``create_transformer_from_parameters``."""

        return {
            "sn_mva": self.sn_mva,
            "vk_percent": self.vk_percent,
            "vkr_percent": self.vkr_percent,
            "pfe_kw": self.pfe_kw,
            "i0_percent": self.i0_percent,
            "shift_degree": self.shift_degree,
            "vn_hv_kv": hv_kv,
            "vn_lv_kv": lv_kv,
        }


@dataclass(slots=True)
class FeederAttachmentPlan:
    """Plan describing how a distribution feeder connects to a TSO bus."""

    boundary_bus: int
    boundary_bus_name: str
    transformer_kwargs: dict[str, float]
    feeder: DsoFeeder


@dataclass(slots=True)
class BaselineCoupledSystem:
    """High-level description of the coupled TSO/DSO baseline system."""

    tso: TsoPandapowerCase
    attachments: list[FeederAttachmentPlan]


def plan_baseline_coupled_system(
    case_name: str = "case39",
    *,
    boundary_buses: Sequence[int | str] | None = None,
    feeder_type: Literal["mv", "lv"] = "mv",
    feeder_peak_mw: float = 20.0,
    feeder_cos_phi: float = 0.95,
    transformer_spec: TransformerSpec | None = None,
) -> BaselineCoupledSystem:
    """Return a coupled system plan following the recommended baseline."""

    tso_case = build_tso_pandapower(case_name, boundary_buses)
    attachments: list[FeederAttachmentPlan] = []
    spec = transformer_spec or TransformerSpec()

    for bus_idx in tso_case.boundary_buses:
        feeder = build_cigre_feeder(
            feeder_type,
            target_peak_mw=feeder_peak_mw,
            cos_phi=feeder_cos_phi,
        )
        hv_kv = _select_transformer_hv(tso_case.net, bus_idx)
        root_vn = float(feeder.net.bus.at[feeder.root_bus, "vn_kv"])
        lv_kv = root_vn
        transformer_kwargs = spec.instantiate(hv_kv=hv_kv, lv_kv=lv_kv)
        attachments.append(
            FeederAttachmentPlan(
                boundary_bus=bus_idx,
                boundary_bus_name=str(tso_case.net.bus.at[bus_idx, "name"]),
                transformer_kwargs=transformer_kwargs,
                feeder=feeder,
            )
        )

    return BaselineCoupledSystem(tso=tso_case, attachments=attachments)


def _select_transformer_hv(net: pp.pandapowerNet, bus_idx: int) -> float:
    """Pick the HV side voltage for the interface transformer."""

    bus_vn = float(net.bus.at[bus_idx, "vn_kv"])
    return bus_vn


def assemble_baseline_network(
    plan: BaselineCoupledSystem,
    *,
    clear_results: bool = True,
) -> pp.pandapowerNet:
    """Materialise ``plan`` into a single pandapower network."""

    combined = copy.deepcopy(plan.tso.net)
    if clear_results:
        pp.clear_result_tables(combined)
    _ensure_boolean_flags(combined)

    attachments_info: list[dict[str, int]] = []

    for slot, attachment in enumerate(plan.attachments):
        feeder_net = copy.deepcopy(attachment.feeder.net)
        if clear_results:
            pp.clear_result_tables(feeder_net)
        if not feeder_net.ext_grid.empty:
            pp.drop_elements(feeder_net, "ext_grid", feeder_net.ext_grid.index.tolist())
        _ensure_boolean_flags(feeder_net)

        combined, lookup = pp.merge_nets(
            combined,
            feeder_net,
            validate=False,
            merge_results=False,
            return_net2_reindex_lookup=True,
            net2_reindex_log_level=None,
        )
        _ensure_transformer_defaults(combined)

        root_bus = int(lookup["bus"][attachment.feeder.root_bus])
        hv_bus = int(attachment.boundary_bus)

        _downshift_tso_loads(combined, hv_bus, feeder_net)

        trafo_params = attachment.transformer_kwargs.copy()
        transformer = pp.create_transformer_from_parameters(
            combined,
            hv_bus,
            root_bus,
            name=f"baseline_interface_{slot}",
            **trafo_params,
        )

        attachments_info.append(
            {
                "boundary_bus": hv_bus,
                "feeder_root_bus": root_bus,
                "transformer": int(transformer),
                "downshift_mw": float(feeder_net.load.p_mw.sum()),
            }
        )
        _ensure_transformer_defaults(combined)

    combined["baseline_plan"] = {
        "case_name": plan.tso.case_name,
        "attachments": attachments_info,
    }
    combined["baseline_plan"]["boundary_names"] = plan.tso.boundary_bus_names
    return combined


def _ensure_boolean_flags(net: pp.pandapowerNet) -> None:
    """Fill NaNs in controllable-like columns to avoid dtype warnings."""

    for element in ("sgen", "gen", "load", "storage"):
        table = getattr(net, element, None)
        if table is None or table.empty:
            continue
        if "controllable" not in table.columns:
            table.loc[:, "controllable"] = False
        else:
            table.loc[:, "controllable"] = table["controllable"].fillna(False).astype(bool)


def _ensure_transformer_defaults(net: pp.pandapowerNet) -> None:
    """Fill transformer tap metadata with safe defaults."""

    trafo = getattr(net, "trafo", None)
    if trafo is None or trafo.empty:
        return
    if "tap_dependency_table" in trafo.columns:
        trafo.loc[:, "tap_dependency_table"] = False
    if "id_characteristic_table" in trafo.columns:
        trafo.loc[:, "id_characteristic_table"] = (
            trafo["id_characteristic_table"].astype("Int64", copy=False).fillna(-1)
        )


def _downshift_tso_loads(
    tso_net: pp.pandapowerNet,
    hv_bus: int,
    feeder_net: pp.pandapowerNet,
) -> None:
    """Reduce TSO load at ``hv_bus`` to avoid double counting with the feeder."""

    feeder_p = float(feeder_net.load.p_mw.sum())
    feeder_q = float(feeder_net.load.q_mvar.sum())
    if feeder_p <= 0 and feeder_q <= 0:
        return

    mask = tso_net.load.bus == hv_bus
    if not np.any(mask):
        return

    load_slice = tso_net.load.loc[mask]
    base_p = float(load_slice.p_mw.sum())
    base_q = float(load_slice.q_mvar.sum())
    if base_p <= 0:
        return

    p_ratio = np.clip(feeder_p / base_p, 0.0, 1.0)
    tso_net.load.loc[mask, "p_mw"] = load_slice.p_mw * (1.0 - p_ratio)

    if base_q > 0 and feeder_q > 0:
        q_ratio = np.clip(feeder_q / base_q, 0.0, 1.0)
        tso_net.load.loc[mask, "q_mvar"] = load_slice.q_mvar * (1.0 - q_ratio)
    else:
        tso_net.load.loc[mask, "q_mvar"] = 0.0
