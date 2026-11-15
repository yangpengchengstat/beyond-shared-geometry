"""Canonical paper-facing metric labels."""

from __future__ import annotations


PAPER_METRIC_LABELS = {
    "E_eq": r"$E_{\mathrm{eq}}$",
    "E_hom_data": r"$E_{\mathrm{hom}}^{\mathrm{data}}$",
    "E_hom_op": r"$E_{\mathrm{hom}}^{\mathrm{op}}$",
    "E_rel_data": r"$E_{\mathrm{rel}}^{\mathrm{data}}$",
    "E_rel_op": r"$E_{\mathrm{rel}}^{\mathrm{op}}$",
    "E_comp_data": r"$E_{\mathrm{comp}}^{\mathrm{data}}$",
    "E_comp_op": r"$E_{\mathrm{comp}}^{\mathrm{op}}$",
    "E_word": r"$E_{\mathrm{word}}$",
    "E_int": r"$E_{\mathrm{int}}$",
    "E_beh_NLL": r"$E_{\mathrm{beh,NLL}}$",
}


def paper_metric_label(metric_key: str) -> str:
    try:
        return PAPER_METRIC_LABELS[metric_key]
    except KeyError as exc:
        raise KeyError(f"No canonical paper label registered for metric {metric_key!r}") from exc
