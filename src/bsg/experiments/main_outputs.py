"""Canonical output layout and reproducible lightweight paper artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from bsg.experiments.runtime_state import atomic_write_frame, atomic_write_text


SUBDIRS = ("structural", "irrep", "intertwiner", "causal", "bootstrap", "logs", "checkpoints", "results")
STABLE_TABLES = (
    "structural_main.csv", "c7_causal_main.csv", "d4_causal_main.csv",
    "cross_model_main.csv", "latent_dim_sensitivity.csv", "model_scale_summary.csv",
    "joint_vs_ridge_inference.csv", "causal_additive_baseline_summary.csv",
    "d4_order_control_summary.csv",
)
STABLE_FIGURES = (
    "structural_by_layer", "structural_by_model", "c7_causal_by_layer",
    "c7_causal_by_step", "c7_cycle_recovery", "d4_relation_intervention", "irrep_spectra",
    "intertwiner_vs_cka", "scale_analysis",
)
STABLE_TABLE_REQUIRED_COLUMNS = {
    "structural_main.csv": {"model", "group", "method", "depth_slot", "layer", "E_eq", "E_rel_data"},
    "c7_causal_main.csv": {"model", "method", "evaluation_role", "intervention", "depth_slot", "layer", "target_logit_change"},
    "d4_causal_main.csv": {"model", "method", "evaluation_role", "intervention", "depth_slot", "layer", "relation_final_logit_distance"},
    "cross_model_main.csv": {"model_A", "model_B", "group", "depth_slot", "layer_A", "layer_B", "split_seed", "optimization_seed", "latent_dim"},
    "latent_dim_sensitivity.csv": {"model", "group", "method", "latent_dim", "E_eq", "E_rel_data"},
    "model_scale_summary.csv": {"model", "group", "method", "E_eq", "E_rel_data"},
    "joint_vs_ridge_inference.csv": {"model", "group", "metric", "joint_estimate", "independent_ridge_estimate", "paired_difference", "ci_low", "ci_high", "n_pairs"},
    "causal_additive_baseline_summary.csv": {"model", "group", "metric", "learned_operator_estimate", "source_conditioned_additive_estimate", "paired_difference", "ci_low", "ci_high", "n_pairs"},
    "d4_order_control_summary.csv": {"model", "group", "metric", "true_relation_estimate", "wrong_order_estimate", "paired_difference", "ci_low", "ci_high", "n_pairs"},
}


def create_output_tree(root, run_id: str) -> Path:
    out = Path(root) / run_id
    if out.exists():
        if not (out / "run_ledger.json").exists():
            raise FileExistsError(f"existing run directory is not resumable: {out}")
        return out
    out.mkdir(parents=True)
    for name in SUBDIRS:
        (out / name).mkdir()
    (out / "results" / "tables").mkdir()
    (out / "results" / "figures").mkdir()
    return out


def write_frame(frame: pd.DataFrame, path) -> Path:
    return atomic_write_frame(frame, Path(path))


def write_table(frame: pd.DataFrame, path, columns=None) -> Path:
    view = frame if columns is None else frame[columns]
    return atomic_write_text(Path(path), view.to_latex(index=False, float_format=lambda value: f"{value:.4g}"))


def _scaled_points(frame: pd.DataFrame, x: str, y: str):
    clean = frame[[x, y]].dropna().sort_values(x)
    width, height, pad = 720, 420, 55
    if clean.empty:
        return width, height, pad, []
    xs, ys = clean[x].astype(float), clean[y].astype(float)
    xmin, ymin = float(xs.min()), float(ys.min())
    xr = max(float(xs.max() - xmin), 1e-12)
    yr = max(float(ys.max() - ymin), 1e-12)
    points = [
        (pad + (vx - xmin) / xr * (width - 2 * pad), height - pad - (vy - ymin) / yr * (height - 2 * pad))
        for vx, vy in zip(xs, ys)
    ]
    return width, height, pad, points


def write_simple_svg(frame: pd.DataFrame, x: str, y: str, path, title: str) -> Path:
    path = Path(path)
    width, height, pad, points = _scaled_points(frame, x, y)
    point_text = " ".join(f"{vx:.1f},{vy:.1f}" for vx, vy in points)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{pad}" y="30" font-family="sans-serif" font-size="18">{title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>
<polyline points="{point_text}" fill="none" stroke="#176B87" stroke-width="2"/></svg>'''
    return atomic_write_text(path, svg)


def write_simple_pdf(frame: pd.DataFrame, x: str, y: str, path, title: str) -> Path:
    """Write a small vector PDF from the same points used by the SVG."""
    path = Path(path)
    width, height, pad, points = _scaled_points(frame, x, y)
    safe_title = title.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    commands = [
        "0 0 0 RG 1 w",
        f"{pad} {pad} m {width-pad} {pad} l S",
        f"{pad} {pad} m {pad} {height-pad} l S",
        "0.09 0.42 0.53 RG 2 w",
    ]
    if points:
        first_x, first_y = points[0]
        commands.append(f"{first_x:.1f} {height-first_y:.1f} m")
        commands.extend(f"{vx:.1f} {height-vy:.1f} l" for vx, vy in points[1:])
        commands.append("S")
    commands.append(f"BT /F1 16 Tf {pad} {height-30} Td ({safe_title}) Tj ET")
    content = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    payload.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    payload.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def write_figure_pair(frame: pd.DataFrame, x: str, y: str, directory: Path, stem: str, title: str) -> list[Path]:
    source_path = write_frame(frame, directory / f"{stem}.source.csv")
    return [
        write_simple_svg(frame, x, y, directory / f"{stem}.svg", title),
        write_simple_pdf(frame, x, y, directory / f"{stem}.pdf", title),
        source_path,
    ]


def artifact_checksums(output_dir: Path) -> dict:
    checksums = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "checkpoints" in path.parts:
            continue
        checksums[str(path.relative_to(output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def validate_stable_artifacts(results_root: Path) -> None:
    tables = Path(results_root) / "tables"
    figures = Path(results_root) / "figures"
    for name, required in STABLE_TABLE_REQUIRED_COLUMNS.items():
        path = tables / name
        if not path.is_file():
            raise ValueError(f"missing stable table: {path}")
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = (required | {"source_csv"}) - columns
        if missing:
            raise ValueError(f"stable table {name} missing columns: {sorted(missing)}")
    for stem in STABLE_FIGURES:
        source = figures / f"{stem}.source.csv"
        if not source.is_file():
            raise ValueError(f"missing figure source data: {source}")
        for suffix in (".svg", ".pdf"):
            path = figures / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing stable figure: {path}")
