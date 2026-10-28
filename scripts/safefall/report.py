#!/usr/bin/env python3
"""Aggregate SafeFall benchmark shards into tables and figures.

Reads ``trials_shard_*.csv`` produced by ``scripts/safefall/benchmark.py`` and
writes a Table-IV-shaped summary (overall and by primary fall mechanism), plus
PNG/PDF figures and a simple HTML gallery.

Runs on CPU; no simulator dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("max_joint_torque", "Peak joint torque (N·m)", True),
    ("max_torque_ratio", "Peak torque ratio", True),
    ("max_joint_reaction", "Peak joint reaction force (N)", True),
    ("max_contact_force", "Peak contact force (N)", True),
    ("max_impulse", "Max step impulse (N·s)", True),
    ("cumulative_impulse", "Cumulative impulse (N·s)", True),
    ("illegal_contact", "Illegal-contact trials (%)", False),
    ("max_limit_violations", "Max joint-limit violations", False),
]
BASELINES = ["nominal", "default", "damping"]
COLORS = {"nominal": "#d62728", "default": "#1f77b4", "damping": "#2ca02c"}


def load_trials(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("trials_shard_*.csv"))
    if not files:
        raise SystemExit(f"No trials_shard_*.csv found under {run_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Coerce numerics; some columns are empty for non-applicable mechanisms.
    for col in df.columns:
        if col in {"baseline", "mechanisms", "primary", "freeze_mode", "freeze_dofs"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["illegal_pct"] = 100.0 * df["illegal_contact"].fillna(0.0)
    if "primary" not in df or df["primary"].isna().all():
        df["primary"] = df["mechanisms"].astype(str).str.split("+").str[0]
    return df


def summary_table(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    keys = ["baseline"] + ([group] if group else [])
    rows = []
    for group_key, sub in df.groupby(keys, dropna=False):
        key_tuple = (group_key,) if not isinstance(group_key, tuple) else group_key
        row = {k: v for k, v in zip(keys, key_tuple)}
        row["n"] = len(sub)
        for col, _label, _log in METRICS:
            values = sub["illegal_pct"] if col == "illegal_contact" else sub[col]
            values = values.dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trial reduction of each method vs nominal, as percentages."""
    rows = []
    piv = df.pivot_table(index="trial_id", columns="baseline", values=[m for m, _, _ in METRICS])
    if "nominal" not in piv.columns.get_level_values(1):
        return pd.DataFrame()
    for baseline in BASELINES:
        if baseline == "nominal" or baseline not in piv.columns.get_level_values(1):
            continue
        for col, label, _log in METRICS:
            nom = piv[(col, "nominal")]
            val = piv[(col, baseline)]
            mask = nom.notna() & val.notna()
            if not mask.any():
                continue
            # For reduction metrics, a drop is good.  Percent reduction relative to nominal.
            denom = nom[mask].abs().replace(0, np.nan)
            reduction = (100.0 * (nom[mask] - val[mask]) / denom).replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "baseline": baseline,
                    "metric": col,
                    "metric_label": label,
                    "n_pairs": int(mask.sum()),
                    "nominal_mean": float(nom[mask].mean()),
                    "method_mean": float(val[mask].mean()),
                    "mean_reduction_pct": float(reduction.mean()),
                    "median_reduction_pct": float(reduction.median()),
                }
            )
    return pd.DataFrame(rows)


def plot_overall(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for col, label, log in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        data = summary[summary["baseline"].isin(BASELINES)].copy()
        data["baseline"] = pd.Categorical(data["baseline"], categories=BASELINES, ordered=True)
        data = data.sort_values("baseline")
        means = data[f"{col}_mean"].to_numpy()
        stds = data[f"{col}_std"].fillna(0).to_numpy()
        names = data["baseline"].astype(str).tolist()
        ax.bar(names, means, yerr=stds, capsize=4, color=[COLORS.get(n, "#777") for n in names])
        ax.set_ylabel(label)
        ax.set_title(label + " across falls")
        if log and np.nanmax(means) > 0:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        path = out_dir / f"metric_{col}.png"
        fig.savefig(path, dpi=170)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def plot_by_mechanism(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    metrics = [m for m in METRICS if m[0] not in {"max_impulse", "cumulative_impulse"}]
    for col, label, log in metrics:
        fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
        piv = summary.pivot_table(index="primary", columns="baseline", values=f"{col}_mean", aggfunc="first")
        order = [p for p in ("push", "slip", "trip", "freeze") if p in piv.index]
        piv = piv.reindex(order)
        x = np.arange(len(piv.index))
        width = 0.8 / max(1, len(BASELINES))
        for j, baseline in enumerate(BASELINES):
            if baseline not in piv.columns:
                continue
            ax.bar(
                x + (j - (len(BASELINES) - 1) / 2) * width, piv[baseline], width, label=baseline, color=COLORS[baseline]
            )
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by primary fall mechanism")
        if log:
            ax.set_yscale("log")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        path = out_dir / f"mechanism_{col}.png"
        fig.savefig(path, dpi=170)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for col, label, log in [
        m for m in METRICS if m[0] in {"max_joint_torque", "max_joint_reaction", "max_contact_force"}
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        data = [df.loc[df["baseline"] == b, col].dropna().to_numpy() for b in BASELINES]
        ax.boxplot(data, tick_labels=BASELINES, showfliers=False)
        ax.set_ylabel(label)
        ax.set_title(f"{label} distribution")
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        path = out_dir / f"dist_{col}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def write_html(run_dir: Path, out_dir: Path, images: list[Path], summary: pd.DataFrame, deltas: pd.DataFrame) -> Path:
    rel = lambda p: p.relative_to(out_dir).as_posix()  # noqa: E731
    table_rows = []
    for _, row in summary.iterrows():
        cells = "".join(f"<td>{row[f'{m}_mean']:.3g} ± {row[f'{m}_std']:.3g}</td>" for m, _, _ in METRICS)
        table_rows.append(
            f"<tr><td>{row['baseline']}</td><td>{row.get('primary', 'all')}</td>{cells}<td>{row['n']}</td></tr>"
        )
    delta_rows = []
    for _, row in deltas.iterrows():
        delta_rows.append(
            f"<tr><td>{row['baseline']}</td><td>{row['metric_label']}</td>"
            f"<td>{row['nominal_mean']:.3g}</td><td>{row['method_mean']:.3g}</td>"
            f"<td>{row['mean_reduction_pct']:.1f}%</td><td>{row['n_pairs']}</td></tr>"
        )
    videos = sorted((run_dir / "videos").glob("*.mp4")) if (run_dir / "videos").exists() else []

    video_html = "".join(
        f'<p><video controls width="640" src="{os.path.relpath(v, out_dir)}"></video></p>' for v in videos
    )
    metric_headers = "".join(f"<th>{label}</th>" for _, label, _ in METRICS)
    delta_header = (
        "<tr><th>baseline</th><th>metric</th><th>nominal mean</th>"
        "<th>method mean</th><th>mean reduction</th><th>n</th></tr>"
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SafeFall benchmark</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.4}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
img{{max-width:100%;margin:0.5rem 0}}code{{background:#f4f4f4;padding:2px 4px}}</style></head><body>
<h1>SafeFall baseline benchmark</h1>
<p>Run: <code>{run_dir}</code></p>
<h2>Table IV summary</h2>
<table><tr><th>baseline</th><th>mechanism</th>{metric_headers}<th>n</th></tr>{"".join(table_rows)}</table>
<h2>Paired reductions vs nominal</h2>
<table>{delta_header}{"".join(delta_rows)}</table>
<h2>Figures</h2>{"".join(f'<p><img src="{rel(p)}"></p>' for p in images if p.exists())}
<h2>Videos</h2>{video_html or "<p>No videos in <code>videos/</code> yet.</p>"}
</body></html>"""
    path = out_dir / "index.html"
    path.write_text(html)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="SafeFall benchmark report")
    p.add_argument("--run-dir", required=True, help="Directory containing trials_shard_*.csv")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <run-dir>/report)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trials(run_dir)
    overall = summary_table(df)
    by_mech = summary_table(df, group="primary")
    deltas = paired_deltas(df)

    overall.to_csv(out_dir / "summary.csv", index=False)
    by_mech.to_csv(out_dir / "summary_by_mechanism.csv", index=False)
    deltas.to_csv(out_dir / "paired_reductions.csv", index=False)

    images = []
    images += plot_overall(overall, out_dir)
    images += plot_by_mechanism(by_mech, out_dir)
    images += plot_distributions(df, out_dir)
    index = write_html(run_dir, out_dir, images, overall, deltas)

    print(json.dumps({"rows": len(df), "summary": str(out_dir / "summary.csv"), "html": str(index)}, indent=2))


if __name__ == "__main__":
    main()
