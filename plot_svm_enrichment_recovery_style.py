#!/usr/bin/env python3
"""Render SVM parameter enrichment in the recovery-fraction heatmap layout.

This script reads the existing combined SVM enrichment table only. It does not
refit the SVM or recompute enrichment values.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FONT_SIZE = 18.0
RECOVERY_TRACT_TITLE_SIZE = 16
RECOVERY_TICK_SIZE = 13
RECOVERY_CBAR_LABEL_SIZE = 15
EPS = 1e-12

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
    }
)


def display_tract_name(name: object) -> str:
    raw = str(name)
    normalized = re.sub(r"[^A-Za-z0-9]+", "", raw).lower()

    direction = ""
    base = normalized
    if base.endswith("left"):
        direction = "left"
        base = base[:-4]
    elif base.endswith("right"):
        direction = "right"
        base = base[:-5]
    elif base.endswith("l"):
        direction = "left"
        base = base[:-1]
    elif base.endswith("r"):
        direction = "right"
        base = base[:-1]

    aliases = {
        "arcuatefasciculus": "AF",
        "arcuate": "AF",
        "af": "AF",
        "corticospinaltract": "CST",
        "corticospinal": "CST",
        "cst": "CST",
        "fornix": "FX",
        "fx": "FX",
    }
    short = aliases.get(base)
    if short and direction:
        return f"{direction} {short}"
    if short:
        return short
    return raw


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_paths(output_dir: Path, output_png: str) -> tuple[Path, Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    candidates = [
        output_dir / "svm_information_boundary" / "svm_information_region_enrichment.csv",
        output_dir / "svm_information_region_enrichment.csv",
    ]
    enrichment_csv = first_existing(candidates)
    if enrichment_csv is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Could not find the combined SVM enrichment table. Searched:\n  " + searched
        )

    svm_dir = enrichment_csv.parent
    analysis_root = svm_dir.parent if svm_dir.name == "svm_information_boundary" else output_dir

    if output_png:
        png_path = Path(output_png).expanduser().resolve()
    else:
        png_path = svm_dir / "figures" / "svm_parameter_enrichment_recovery_style.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    return enrichment_csv, png_path, analysis_root


def infer_tract_order(enrichment: pd.DataFrame, analysis_root: Path) -> list[str]:
    observed = enrichment["display_tract"].dropna().astype(str).drop_duplicates().tolist()

    # Match the recovery-fraction plot's bundle ordering when its cached table exists.
    recovery_candidates = [
        analysis_root / "model_tables" / "tract_recovery_probability_heatmap_summary.csv",
        analysis_root / "tract_recovery_probability_heatmap_summary.csv",
    ]
    recovery_csv = first_existing(recovery_candidates)
    if recovery_csv is not None:
        recovery = pd.read_csv(recovery_csv, low_memory=False)
        if "bundle" in recovery.columns:
            recovery_order = sorted(recovery["bundle"].dropna().astype(str).unique())
            ordered = [display_tract_name(bundle) for bundle in recovery_order]
            ordered = [tract for tract in ordered if tract in observed]
            return ordered + [tract for tract in observed if tract not in ordered]

    preferred = ["left AF", "right AF", "left FX", "right FX", "left CST", "right CST"]
    return [t for t in preferred if t in observed] + [t for t in observed if t not in preferred]


def parameter_specs(enrichment: pd.DataFrame) -> list[tuple[str, str]]:
    specs = (
        enrichment[["parameter", "parameter_column"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["parameter", "parameter_column"])
    )
    return [(str(row.parameter), str(row.parameter_column)) for row in specs.itertuples(index=False)]


def ordered_level_values(sub: pd.DataFrame) -> list[object]:
    level_rows = sub[["level_index", "level_value"]].drop_duplicates().copy()
    level_rows["level_index"] = pd.to_numeric(level_rows["level_index"], errors="coerce")
    return level_rows.sort_values(["level_index", "level_value"])["level_value"].tolist()


def render(enrichment: pd.DataFrame, output_png: Path, analysis_root: Path, cmap: str, dpi: int) -> None:
    required = {
        "parameter",
        "parameter_column",
        "level_index",
        "level_value",
        "log2_enrichment",
    }
    missing = sorted(required - set(enrichment.columns))
    if missing:
        raise ValueError(f"Enrichment table is missing required columns: {missing}")

    enrichment = enrichment.copy()
    if "display_tract" not in enrichment.columns:
        if "bundle" not in enrichment.columns:
            raise ValueError("Enrichment table must contain display_tract or bundle.")
        enrichment["display_tract"] = enrichment["bundle"].map(display_tract_name)
    enrichment["display_tract"] = enrichment["display_tract"].astype(str)
    enrichment["log2_enrichment"] = pd.to_numeric(enrichment["log2_enrichment"], errors="coerce")

    params = parameter_specs(enrichment)
    if not params:
        raise ValueError("No parameter rows were found in the enrichment table.")
    tract_order = infer_tract_order(enrichment, analysis_root)

    values = enrichment["log2_enrichment"].dropna().to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(vmax) or vmax <= EPS:
        vmax = 1.0

    n_cols = len(params)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.3 * n_cols, 5.4), squeeze=False)
    axes_flat = axes.flatten()
    cbar_ax = fig.add_axes([0.905, 0.22, 0.012, 0.56])

    for idx, (parameter, parameter_column) in enumerate(params):
        ax = axes_flat[idx]
        sub = enrichment.loc[enrichment["parameter_column"].eq(parameter_column)].copy()
        level_order = ordered_level_values(sub)

        pivot = sub.pivot_table(
            index="display_tract",
            columns="level_value",
            values="log2_enrichment",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=tract_order, columns=level_order)

        annot = pivot.copy().astype(object)
        for row_name in pivot.index:
            for col_name in pivot.columns:
                value = pivot.loc[row_name, col_name]
                annot.loc[row_name, col_name] = "" if pd.isna(value) else f"{value:.3f}"

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            cbar=(idx == 0),
            cbar_ax=cbar_ax,
            linewidths=0.5,
            linecolor="white",
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 12},
        )
        ax.set_title(parameter, fontsize=RECOVERY_TRACT_TITLE_SIZE, pad=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=RECOVERY_TICK_SIZE)
        ax.tick_params(axis="y", labelrotation=0, labelsize=RECOVERY_TICK_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")

        if idx == 0:
            ax.set_yticklabels([tick.get_text() for tick in ax.get_yticklabels()], rotation=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    cbar_ax.set_ylabel("log2 enrichment", fontsize=RECOVERY_CBAR_LABEL_SIZE, rotation=90, labelpad=10)
    fig.tight_layout(rect=[0.03, 0.06, 0.895, 0.93])
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SVM parameter enrichment using the recovery-fraction heatmap layout."
    )
    parser.add_argument(
        "output_dir",
        help="Integrated plotting output directory, or its svm_information_boundary directory.",
    )
    parser.add_argument(
        "--output_png",
        default="",
        help="Optional output PNG path. Default: svm_information_boundary/figures/svm_parameter_enrichment_recovery_style.png",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib/seaborn colormap.")
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enrichment_csv, output_png, analysis_root = resolve_paths(Path(args.output_dir), args.output_png)
    enrichment = pd.read_csv(enrichment_csv, low_memory=False)
    if enrichment.empty:
        raise ValueError(f"Enrichment table is empty: {enrichment_csv}")
    render(enrichment, output_png, analysis_root, cmap=args.cmap, dpi=args.dpi)
    print(f"Read: {enrichment_csv}")
    print(f"Wrote: {output_png}")


if __name__ == "__main__":
    main()
