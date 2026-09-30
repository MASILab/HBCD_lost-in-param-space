#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import hypergeom, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, balanced_accuracy_score, pairwise_distances, roc_auc_score
from statsmodels.stats.multitest import multipletests

DEFAULT_PATTERNS = ["factorial_runs_long.sub-*_ses-*.csv"]
DEFAULT_OUTCOME_COL = "hausdorff_mm_vs_reference"
DEFAULT_SHAPE_COLS = ["streamline_count", "tract_volume_mm3", "surface_area_mm2", "mean_length_mm"]
EPS = 1e-12
STATIC_PLOT_FONT_SIZE = 18.0
PARAMETER_AXIS_OVERLAY_FONT_SIZE = 16.0

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": STATIC_PLOT_FONT_SIZE,
        "axes.titlesize": STATIC_PLOT_FONT_SIZE,
        "axes.labelsize": STATIC_PLOT_FONT_SIZE,
        "xtick.labelsize": STATIC_PLOT_FONT_SIZE,
        "ytick.labelsize": STATIC_PLOT_FONT_SIZE,
        "legend.fontsize": STATIC_PLOT_FONT_SIZE,
        "figure.titlesize": STATIC_PLOT_FONT_SIZE,
    }
)



@dataclass(frozen=True)
class MeasureSpec:
    key: str
    label: str
    column: str
    colorbar_title: str
    higher_is_more: bool = True


MEASURE_SPECS = {
    "information_shift": MeasureSpec(
        key="information_shift",
        label="Signed information divergence vs HD",
        column="signed_local_information_divergence_bits",
        colorbar_title="Signed info divergence<br>bits (+ lower HD)",
        higher_is_more=True,
    ),
    "information_shift_scaled": MeasureSpec(
        key="information_shift_scaled",
        label="Scaled signed information divergence vs HD",
        column="signed_local_information_divergence_bits",
        colorbar_title="Signed info divergence<br>bits (+ lower HD)<br>shared across tracts",
        higher_is_more=True,
    ),
    "shape_similarity": MeasureSpec(
        key="shape_similarity",
        label="Relative size/shape similarity",
        column="shape_size_similarity_percentile",
        colorbar_title="Shape/size<br>similarity pct",
        higher_is_more=True,
    ),
    "age_interaction": MeasureSpec(
        key="age_interaction",
        label="Age interaction effect",
        column="lmm_age_interaction_abs_effect",
        colorbar_title="Age interaction<br>|projected effect|",
        higher_is_more=False,
    ),
    "hemisphere_interaction": MeasureSpec(
        key="hemisphere_interaction",
        label="Hemisphere interaction effect",
        column="lmm_hemisphere_interaction_abs_effect",
        colorbar_title="Hemisphere interaction<br>|projected effect|",
        higher_is_more=False,
    ),
    "tract_size": MeasureSpec(
        key="tract_size",
        label="Overall tract size",
        column="tract_size_percentile",
        colorbar_title="Overall tract<br>size percentile",
        higher_is_more=True,
    ),
}

HYPERLATTICE_MEASURE_KEYS = [
    "information_shift",
    "information_shift_scaled",
    "shape_similarity",
    "tract_size",
]

PARAMETER_SPACE_2D_MEASURE_KEYS = [
    "information_shift",
    "shape_similarity",
    "age_interaction",
    "hemisphere_interaction",
    "tract_size",
]


def parse_bool_like(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}
RECOVERY_AXIS_LABEL_SIZE = 16
RECOVERY_TRACT_TITLE_SIZE = 16
RECOVERY_TICK_SIZE = 13
RECOVERY_FIGURE_TITLE_SIZE = 18
RECOVERY_CBAR_LABEL_SIZE = 15


def recovery_parameter_order(df: pd.DataFrame) -> List[str]:
    return sorted(df["parameter"].dropna().astype(str).unique())


def recovery_tract_order(df: pd.DataFrame) -> List[str]:
    return sorted(df["bundle"].dropna().astype(str).unique())


def recovery_sorted_unique_values(series: pd.Series) -> list:
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals, key=lambda x: str(x))


def pretty_tract_family_name(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))


def add_recovery_shared_y_label(fig, text: str, x: float = 0.03) -> None:
    fig.text(x, 0.5, text, rotation=90, va="center", ha="center", fontsize=RECOVERY_AXIS_LABEL_SIZE)


def prepare_recovery_heatmap_summary(prepared_all: pd.DataFrame) -> pd.DataFrame:
    df = prepared_all.copy()
    if "streamline_count" in df.columns:
        df["found"] = pd.to_numeric(df["streamline_count"], errors="coerce").fillna(0) > 0
    elif "status" in df.columns:
        df["found"] = df["status"].astype(str).eq("ok")
    else:
        df["found"] = False

    return (
        df.groupby(["bundle", "tract_family", "parameter", "param_value"], as_index=False)["found"]
        .mean()
        .rename(columns={"found": "recovery_probability"})
    )


def plot_recovery_heatmap_summary_grid(
    summary: pd.DataFrame,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    if summary.empty:
        return

    parameters = recovery_parameter_order(summary)
    tract_order = recovery_tract_order(summary)

    n_cols = len(parameters)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.3 * n_cols, 5.4), squeeze=False)
    axes_flat = axes.flatten()
    cbar_ax = fig.add_axes([0.905, 0.22, 0.012, 0.56])

    for k, parameter in enumerate(parameters):
        ax = axes_flat[k]
        sub = summary.loc[summary["parameter"] == parameter].copy()
        value_order = recovery_sorted_unique_values(sub["param_value"])
        sub["param_value_str"] = sub["param_value"].astype(str)

        pivot = (
            sub.assign(
                param_value_str=pd.Categorical(
                    sub["param_value_str"],
                    categories=[str(v) for v in value_order],
                    ordered=True,
                )
            )
            .pivot(index="bundle", columns="param_value_str", values="recovery_probability")
            .reindex(index=tract_order)
        )

        annot = pivot.copy().astype(object)
        for r in pivot.index:
            for c in pivot.columns:
                val = pivot.loc[r, c]
                annot.loc[r, c] = "" if pd.isna(val) else f"{val:.3f}"

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            cbar=(k == 0),
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

        if k == 0:
            ax.set_yticklabels([display_tract_name(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    cbar_ax.set_ylabel("recovery fraction", fontsize=RECOVERY_CBAR_LABEL_SIZE, rotation=90, labelpad=10)
    fig.tight_layout(rect=[0.03, 0.06, 0.895, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_heatmap_grid(
    prepared_all: pd.DataFrame,
    out_path: Path,
    cmap: str = "viridis",
) -> pd.DataFrame:
    summary = prepare_recovery_heatmap_summary(prepared_all)
    plot_recovery_heatmap_summary_grid(summary, out_path, cmap=cmap)
    return summary


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def stable_combo_id(bundle: str, params: dict[str, object]) -> str:

    clean = {}
    for k, v in params.items():
        key = k.replace("param_", "")
        try:
            fv = float(v)
            clean[key] = int(fv) if fv.is_integer() else fv
        except Exception:
            clean[key] = v
    stable_key = bundle + "|" + "|".join(f"{k}={clean[k]}" for k in sorted(clean))
    return hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12]


def expand_input_paths(paths_or_globs: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for item in paths_or_globs or []:
        p = Path(item)
        matches = sorted(p.parent.glob(p.name)) if any(ch in str(item) for ch in "*?[]") else [p]
        for m in matches:
            if m.exists() and m.is_file():
                key = str(m.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(m)
    return out


def split_bundle(bundle: str) -> tuple[str, str]:
    bundle = str(bundle)
    if bundle.endswith("_L"):
        return bundle[:-2], "L"
    if bundle.endswith("_R"):
        return bundle[:-2], "R"
    if bundle.endswith("L"):
        return bundle[:-1], "L"
    if bundle.endswith("R"):
        return bundle[:-1], "R"
    return bundle, "U"


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
    if direction and base:
        return f"{direction} {raw[:-1]}"
    return raw


def add_family_hemi_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["bundle"].apply(split_bundle) if "bundle" in out.columns else pd.Series([("all", "U")] * len(out))
    if "tract_family" not in out.columns:
        out["tract_family"] = parsed.apply(lambda x: x[0])
    if "hemisphere" not in out.columns:
        out["hemisphere"] = parsed.apply(lambda x: x[1])
    if "subject_session_id" not in out.columns:
        if {"subject", "session"}.issubset(out.columns):
            out["subject_session_id"] = out["subject"].astype(str) + "_" + out["session"].astype(str)
        elif "source_summary_csv" in out.columns:
            out["subject_session_id"] = out["source_summary_csv"].astype(str)
        else:
            out["subject_session_id"] = np.arange(len(out)).astype(str)
    return out


def find_summary_shards(summary_root: Path, patterns: Iterable[str]) -> list[Path]:
    paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(summary_root.rglob(pattern)):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def read_nonempty_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    df["source_summary_csv"] = str(path)
    return df


def load_factorial_shards(summary_root: Path, patterns: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in find_summary_shards(summary_root, patterns):
        df = read_nonempty_csv(path)
        if df is not None:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No non-empty summary shard CSVs found in {summary_root}")

    out = pd.concat(frames, ignore_index=True)
    if "run_type" in out.columns:
        out = out.loc[out["run_type"].astype(str).str.lower().eq("factorial")].copy()
    if out.empty:
        raise ValueError("No factorial rows found in completed summary shards.")

    source_col = "source_summary_csv"
    if source_col in out.columns:
        dedupe_cols = [c for c in out.columns if c != source_col]
        out[source_col] = out.groupby(dedupe_cols, dropna=False)[source_col].transform(
            lambda values: ";".join(sorted(set(map(str, values))))
        )
        out = out.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
    else:
        out = out.drop_duplicates().reset_index(drop=True)

    return out


def apply_optional_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.tract_family:
        out = out.loc[out["tract_family"].astype(str).isin(set(map(str, args.tract_family)))]
    if args.bundle:
        out = out.loc[out["bundle"].astype(str).isin(set(map(str, args.bundle)))]
    if args.session and "session" in out.columns:
        out = out.loc[out["session"].astype(str).isin(set(map(str, args.session)))]
    if args.subject and "subject" in out.columns:
        out = out.loc[out["subject"].astype(str).isin(set(map(str, args.subject)))]
    return out.copy()


def factorial_parameter_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"param_template", "param_fa_threshold", "param_track_voxel_ratio"}
    cols = [c for c in df.columns if c.startswith("param_") and c not in excluded]
    varying = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any() and vals.nunique(dropna=True) > 1:
            varying.append(c)
    preferred = [
        "param_turning_angle",
        "param_step_size",
        "param_smoothing",
        "param_tip_iteration",
        "param_tolerance",
        "param_track_voxel_ratio",
    ]
    return [c for c in preferred if c in varying] + [c for c in sorted(varying) if c not in preferred]


def detect_age_column(df: pd.DataFrame, requested: str = "") -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested --age_col does not exist: {requested}")
        return requested
    candidates = [
        "age",
        "age_years",
        "age_in_years",
        "scan_age",
        "scan_age_years",
        "age_months",
        "scan_age_months",
        "interview_age",
    ]
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def parameter_level_maps(df: pd.DataFrame, param_cols: Sequence[str]) -> dict[str, list[float]]:
    out = {}
    for c in param_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        uniq = np.sort(pd.Series(vals).dropna().unique())
        out[c] = [float(v) for v in uniq]
    return out


def full_parameter_grid(param_cols: Sequence[str], level_maps: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product(*[level_maps[c] for c in param_cols])), columns=list(param_cols))


def parameter_key_df(df: pd.DataFrame, param_cols: Sequence[str]) -> pd.Series:
    return df[list(param_cols)].astype(float).apply(lambda r: tuple(r.values.tolist()), axis=1)


def normalized_index_coordinates(grid: pd.DataFrame, param_cols: Sequence[str], level_maps: dict[str, list[float]]) -> np.ndarray:
    coords = np.zeros((len(grid), len(param_cols)), dtype=float)
    for i, c in enumerate(param_cols):
        levels = [float(v) for v in level_maps[c]]
        idx_map = {float(v): j for j, v in enumerate(levels)}
        denom = max(1, len(levels) - 1)
        coords[:, i] = [idx_map[float(v)] / denom for v in grid[c].to_numpy(dtype=float)]
    return coords


def orient_embedding(coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=float).copy()
    out -= out.mean(axis=0, keepdims=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            out[:, j] *= -1
    scale = np.max(np.linalg.norm(out, axis=1))
    if scale > 0:
        out /= scale
    return out


def compute_parameter_embedding(
    grid: pd.DataFrame,
    param_cols: Sequence[str],
    level_maps: dict[str, list[float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray]:
    nd = normalized_index_coordinates(grid, param_cols, level_maps)

    if args.embedding_method == "pca":

        n_components = min(3, nd.shape[0], nd.shape[1])
        emb = PCA(n_components=n_components, random_state=args.random_state).fit_transform(nd)
    elif args.embedding_method == "metric_mds":

        distances = pairwise_distances(nd, metric="euclidean")
        mds_params = signature(MDS).parameters
        if "metric_mds" in mds_params:
            model = MDS(
                n_components=3,
                metric_mds=True,
                metric="precomputed",
                init="classical_mds",
                random_state=args.random_state,
                n_init=args.mds_n_init,
                max_iter=args.mds_max_iter,
                normalized_stress="auto",
            )
        else:
            try:
                model = MDS(
                    n_components=3,
                    metric=True,
                    dissimilarity="precomputed",
                    random_state=args.random_state,
                    n_init=args.mds_n_init,
                    max_iter=args.mds_max_iter,
                    normalized_stress="auto",
                )
            except TypeError:
                model = MDS(
                    n_components=3,
                    metric=True,
                    dissimilarity="precomputed",
                    random_state=args.random_state,
                    n_init=args.mds_n_init,
                    max_iter=args.mds_max_iter,
                )
        emb = model.fit_transform(distances)
    elif args.embedding_method == "classical_mds":

        distances = pairwise_distances(nd, metric="euclidean")
        d2 = distances ** 2
        n = d2.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ d2 @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order][:3]
        eigvecs = eigvecs[:, order][:, :3]
        eigvals = np.maximum(eigvals, 0.0)
        emb = eigvecs * np.sqrt(eigvals)[None, :]
    else:
        raise ValueError(f"Unknown embedding method: {args.embedding_method}")

    if emb.shape[1] < 3:
        emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])), constant_values=0.0)

    emb = orient_embedding(emb) * args.embedding_scale

    out = grid.copy()
    out["parameter_key"] = parameter_key_df(out, param_cols)
    out["x"], out["y"], out["z"] = emb[:, 0], emb[:, 1], emb[:, 2]

    axis_vecs = []
    for c in param_cols:
        lo = float(level_maps[c][0])
        hi = float(level_maps[c][-1])
        lo_mean = out.loc[np.isclose(out[c].astype(float), lo), ["x", "y", "z"]].mean().to_numpy(dtype=float)
        hi_mean = out.loc[np.isclose(out[c].astype(float), hi), ["x", "y", "z"]].mean().to_numpy(dtype=float)
        vec = hi_mean - lo_mean
        norm = np.linalg.norm(vec)
        if norm <= EPS:
            vec = np.zeros(3)
            vec[min(len(axis_vecs), 2)] = 1.0
        else:
            vec = vec / norm
        axis_vecs.append(vec)

    return out, np.asarray(axis_vecs, dtype=float)


def build_hyperlattice_edges(grid: pd.DataFrame, param_cols: Sequence[str], level_maps: dict[str, list[float]]) -> list[tuple[int, int]]:
    key_to_idx = {}
    level_index_rows = []
    levels_by_param = {c: [float(v) for v in level_maps[c]] for c in param_cols}

    for i, row in grid.reset_index(drop=True).iterrows():
        idxs = []
        for c in param_cols:
            vals = levels_by_param[c]
            value = float(row[c])
            matches = [j for j, v in enumerate(vals) if np.isclose(value, v)]
            if not matches:
                idxs = None
                break
            idxs.append(matches[0])
        if idxs is None:
            level_index_rows.append(None)
            continue
        key = tuple(idxs)
        level_index_rows.append(key)
        key_to_idx[key] = i

    edges = set()
    for i, key in enumerate(level_index_rows):
        if key is None:
            continue
        for j, c in enumerate(param_cols):
            for neighbor in (key[j] - 1, key[j] + 1):
                if neighbor < 0 or neighbor >= len(levels_by_param[c]):
                    continue
                new_key = list(key)
                new_key[j] = neighbor
                new_key = tuple(new_key)
                if new_key in key_to_idx:
                    edges.add(tuple(sorted((i, key_to_idx[new_key]))))
    return sorted(edges)


def aggregate_combinations(df: pd.DataFrame, param_cols: Sequence[str], outcome_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_finite_outcome"] = pd.to_numeric(work[outcome_col], errors="coerce").notna()
    grouped = work.groupby(list(param_cols), dropna=False)

    out = grouped[outcome_col].agg(mean_hd="mean", median_hd="median", sd_hd="std", n_recovered_rows="count")
    counts = grouped["_finite_outcome"].agg(n_attempt_rows="size")
    out = out.join(counts).reset_index()
    out["n_recovered_rows"] = out["n_recovered_rows"].astype(int)
    out["n_attempt_rows"] = out["n_attempt_rows"].astype(int)
    out["n_nonrecovered_rows"] = out["n_attempt_rows"] - out["n_recovered_rows"]
    out["recovery_fraction"] = out["n_recovered_rows"] / out["n_attempt_rows"].replace(0, np.nan)
    out["is_recovered_combination"] = out["n_recovered_rows"] > 0
    out["recovery_status"] = np.where(out["is_recovered_combination"], "recovered", "non_recovered")
    out["parameter_key"] = parameter_key_df(out, param_cols)
    return out


def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    valid = vals.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    unique = valid.nunique()
    bins = max(2, min(int(n_bins), int(unique)))
    if bins <= 1:
        return pd.Series(0, index=values.index)
    try:
        return pd.qcut(vals, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.cut(vals, bins=bins, labels=False, include_lowest=True)


def compute_information_shift_metric(raw: pd.DataFrame, combo_df: pd.DataFrame, param_cols: Sequence[str], outcome_col: str, n_bins: int) -> pd.DataFrame:
    if "is_recovered_row" in raw.columns:
        recovered_mask = raw["is_recovered_row"].astype(bool)
    else:
        recovered_mask = analytically_recovered_mask(raw, outcome_col=outcome_col, require_finite_outcome=True)
    work = raw.loc[recovered_mask & pd.to_numeric(raw[outcome_col], errors="coerce").notna()].copy()
    if work.empty:
        combo_df["local_information_divergence_bits"] = np.nan
        combo_df["signed_local_information_divergence_bits"] = np.nan
        combo_df["information_shift_mean_hd_mm"] = np.nan
        combo_df["information_shift_direction"] = ""
        combo_df["information_shift_n_rows"] = 0
        return combo_df

    global_mean_hd = float(pd.to_numeric(work[outcome_col], errors="coerce").mean())
    global_median_hd = float(pd.to_numeric(work[outcome_col], errors="coerce").median())

    work["_hd_bin"] = quantile_bins(work[outcome_col], n_bins=n_bins)
    work = work.loc[work["_hd_bin"].notna()].copy()
    if work.empty:
        combo_df["local_information_divergence_bits"] = np.nan
        combo_df["signed_local_information_divergence_bits"] = np.nan
        combo_df["information_shift_mean_hd_mm"] = np.nan
        combo_df["information_shift_direction"] = ""
        combo_df["information_shift_n_rows"] = 0
        return combo_df

    work["_hd_bin"] = work["_hd_bin"].astype(int)
    bin_counts = work["_hd_bin"].value_counts(normalize=True).sort_index()
    marginal = bin_counts.to_dict()

    rows = []
    for key, g in work.groupby(list(param_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        conditional = g["_hd_bin"].value_counts(normalize=True).to_dict()
        kl = 0.0
        for b, p_cond in conditional.items():
            p_marg = marginal.get(b, 0.0)
            if p_cond > 0 and p_marg > 0:
                kl += float(p_cond) * math.log(float(p_cond) / float(p_marg), 2)

        combo_mean_hd = float(pd.to_numeric(g[outcome_col], errors="coerce").mean())
        combo_median_hd = float(pd.to_numeric(g[outcome_col], errors="coerce").median())
        mean_shift = global_mean_hd - combo_mean_hd
        # Positive signed divergence = shifted toward lower HD than the tract-wide average.
        signed_kl = float(np.sign(mean_shift)) * kl

        if mean_shift > 0:
            direction = "lower_hd_than_global"
        elif mean_shift < 0:
            direction = "higher_hd_than_global"
        else:
            direction = "no_mean_hd_shift"

        row = {c: v for c, v in zip(param_cols, key)}
        row["local_information_divergence_bits"] = kl
        row["signed_local_information_divergence_bits"] = signed_kl
        row["information_shift_global_mean_hd"] = global_mean_hd
        row["information_shift_global_median_hd"] = global_median_hd
        row["information_shift_combo_mean_hd"] = combo_mean_hd
        row["information_shift_combo_median_hd"] = combo_median_hd
        row["information_shift_mean_hd_mm"] = mean_shift
        row["information_shift_direction"] = direction
        row["information_shift_n_rows"] = int(len(g))
        rows.append(row)

    metric = pd.DataFrame(rows)
    out = combo_df.merge(metric, on=list(param_cols), how="left")
    out["information_shift_n_rows"] = out["information_shift_n_rows"].fillna(0).astype(int)
    return out



def percentile_from_values(values: pd.Series, high_good: bool) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    ranks = vals.rank(method="average", pct=True)
    if high_good:
        return ranks
    return 1.0 - ranks


def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median(axis=0, skipna=True)
    mad = (df - med).abs().median(axis=0, skipna=True)
    scale = 1.4826 * mad
    std = df.std(axis=0, skipna=True)
    scale = scale.where(scale > EPS, std)
    scale = scale.where(scale > EPS, 1.0)
    return (df - med) / scale


def compute_tract_size_metric(raw: pd.DataFrame, combo_df: pd.DataFrame, param_cols: Sequence[str], shape_cols: Sequence[str]) -> pd.DataFrame:
    available_cols = [c for c in shape_cols if c in raw.columns]
    if not available_cols:
        combo_df["tract_size_percentile"] = np.nan
        combo_df["tract_size_score"] = np.nan
        combo_df["tract_size_n_rows"] = 0
        return combo_df

    work = raw.copy()
    for c in available_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = work[available_cols].notna().all(axis=1)
    if "status" in work.columns:
        valid &= work["status"].astype(str).str.lower().isin({"ok", "success", "completed"})
    work = work.loc[valid].copy()

    if work.empty:
        combo_df["tract_size_percentile"] = np.nan
        combo_df["tract_size_score"] = np.nan
        combo_df["tract_size_n_rows"] = 0
        return combo_df

    z = robust_zscore(work[available_cols])
    work["_tract_size_score"] = z.mean(axis=1)
    work["_tract_size_percentile"] = percentile_from_values(work["_tract_size_score"], high_good=True) * 100.0

    grouped = work.groupby(list(param_cols), dropna=False).agg(
        tract_size_percentile=("_tract_size_percentile", "median"),
        tract_size_score=("_tract_size_score", "median"),
        tract_size_n_rows=("_tract_size_score", "count"),
    ).reset_index()

    out = combo_df.merge(grouped, on=list(param_cols), how="left")
    out["tract_size_n_rows"] = out["tract_size_n_rows"].fillna(0).astype(int)
    return out


def compute_shape_similarity_metric(raw: pd.DataFrame, combo_df: pd.DataFrame, param_cols: Sequence[str], shape_cols: Sequence[str]) -> pd.DataFrame:
    available_cols = [c for c in shape_cols if c in raw.columns]
    if not available_cols:
        combo_df["shape_size_similarity_percentile"] = np.nan
        combo_df["shape_size_distance_to_tract_median"] = np.nan
        combo_df["shape_size_n_rows"] = 0
        return combo_df

    work = raw.copy()
    for c in available_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = work[available_cols].notna().all(axis=1)
    if "status" in work.columns:
        valid &= work["status"].astype(str).str.lower().isin({"ok", "success", "completed"})
    work = work.loc[valid].copy()

    if work.empty:
        combo_df["shape_size_similarity_percentile"] = np.nan
        combo_df["shape_size_distance_to_tract_median"] = np.nan
        combo_df["shape_size_n_rows"] = 0
        return combo_df

    z = robust_zscore(work[available_cols])
    work["_shape_distance"] = np.sqrt((z ** 2).sum(axis=1))
    work["_shape_similarity_percentile"] = percentile_from_values(work["_shape_distance"], high_good=False) * 100.0

    grouped = work.groupby(list(param_cols), dropna=False).agg(
        shape_size_similarity_percentile=("_shape_similarity_percentile", "median"),
        shape_size_distance_to_tract_median=("_shape_distance", "median"),
        shape_size_n_rows=("_shape_distance", "count"),
    ).reset_index()

    out = combo_df.merge(grouped, on=list(param_cols), how="left")
    out["shape_size_n_rows"] = out["shape_size_n_rows"].fillna(0).astype(int)
    return out


def load_current_lmm_outputs(lmm_results_csv: str, lmm_term_tests_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not lmm_results_csv:
        return pd.DataFrame(), pd.DataFrame()

    results_path = Path(lmm_results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing --lmm_results_csv: {results_path}")

    results = pd.read_csv(results_path, low_memory=False)
    results["source_lmm_results_csv"] = str(results_path)

    if lmm_term_tests_csv:
        tests_path = Path(lmm_term_tests_csv)
        if not tests_path.exists():
            raise FileNotFoundError(f"Missing --lmm_term_tests_csv: {tests_path}")
        tests = pd.read_csv(tests_path, low_memory=False)
        tests["source_lmm_term_tests_csv"] = str(tests_path)
    else:
        tests = pd.DataFrame()

    return results, tests


def parameter_z_stats_from_raw(raw_all: pd.DataFrame, param_cols: Sequence[str]) -> dict[str, tuple[float, float]]:
    # Mirrors run_lmm_analysis_modesplit_v3.py: after factorial rows are expanded to
    # parameter-long form, param_z is standardized within each parameter using ddof=0.
    stats: dict[str, tuple[float, float]] = {}
    for c in param_cols:
        values = pd.to_numeric(raw_all[c], errors="coerce").dropna()
        if values.empty:
            stats[c] = (0.0, 1.0)
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= EPS:
            sd = 1.0
        stats[c] = (mean, sd)
    return stats


def lmm_parameter_name(param_col: str) -> str:
    return str(param_col).removeprefix("param_")


def _get_lmm_term_tests(
    lmm_term_tests: pd.DataFrame,
    tract_family: str,
    parameter: str,
    metric: str,
    term: str,
) -> dict[str, object]:
    if lmm_term_tests.empty:
        return {}

    tests = lmm_term_tests.copy()
    required = {"tract_family", "parameter", "metric", "term"}
    if not required.issubset(tests.columns):
        return {}

    if term == "param_z:C(hemisphere)[T.R]" and term not in set(tests["term"].astype(str).unique()):
        term_match = "param_z:C(hemisphere)"
    else:
        term_match = term

    sub = tests.loc[
        tests["tract_family"].astype(str).eq(str(tract_family))
        & tests["parameter"].astype(str).eq(str(parameter))
        & tests["metric"].astype(str).eq(str(metric))
        & tests["term"].astype(str).eq(str(term_match))
    ].copy()

    if sub.empty:
        return {}

    row = sub.iloc[0]
    out = {}
    for c in ["p_lr", "p_lr_fdr", "delta_marginal_r2", "delta_conditional_r2", "local_f2", "lr_stat", "df_diff"]:
        if c in row.index:
            out[c] = row[c]
    return out


def prepare_projected_lmm_coefficients(
    lmm_results: pd.DataFrame,
    lmm_term_tests: pd.DataFrame,
    tract_family: str,
    metric: str,
    param_cols: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    # Output:
    #   coeffs["age"][param_col]["beta"], ["p_lr"], ["p_lr_fdr"], ...
    #   coeffs["hemisphere"][param_col]["beta"], ...
    coeffs = {"age": {}, "hemisphere": {}}
    if lmm_results.empty:
        return coeffs

    required = {"tract_family", "parameter", "metric"}
    if not required.issubset(lmm_results.columns):
        raise ValueError(
            "lmm_results.csv must contain tract_family, parameter, and metric columns from run_lmm_analysis_modesplit_v3.py"
        )

    results = lmm_results.copy()
    if "converged" in results.columns:
        converged = results["converged"].apply(parse_bool_like)
        results = results.loc[converged].copy()

    family_results = results.loc[
        results["tract_family"].astype(str).eq(str(tract_family))
        & results["metric"].astype(str).eq(str(metric))
    ].copy()

    for c in param_cols:
        p = lmm_parameter_name(c)
        sub = family_results.loc[family_results["parameter"].astype(str).eq(str(p))].copy()
        if sub.empty:
            continue
        row = sub.iloc[0]

        age_beta = pd.to_numeric(pd.Series([row.get("fe_param_z:age_z", np.nan)]), errors="coerce").iloc[0]
        hemi_beta = pd.to_numeric(pd.Series([row.get("fe_param_z:C(hemisphere)[T.R]", np.nan)]), errors="coerce").iloc[0]

        if pd.notna(age_beta):
            age_tests = _get_lmm_term_tests(
                lmm_term_tests,
                tract_family=tract_family,
                parameter=p,
                metric=metric,
                term="param_z:age_z",
            )
            coeffs["age"][c] = {"beta": float(age_beta), **age_tests}

        if pd.notna(hemi_beta):
            hemi_tests = _get_lmm_term_tests(
                lmm_term_tests,
                tract_family=tract_family,
                parameter=p,
                metric=metric,
                term="param_z:C(hemisphere)[T.R]",
            )
            coeffs["hemisphere"][c] = {"beta": float(hemi_beta), **hemi_tests}

    return coeffs


def project_lmm_interaction_effects_to_combinations(
    combo_df: pd.DataFrame,
    raw_all: pd.DataFrame,
    lmm_results: pd.DataFrame,
    lmm_term_tests: pd.DataFrame,
    bundle_name: str,
    param_cols: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    out = combo_df.copy()
    tract_family, _ = split_bundle(bundle_name)
    z_stats = parameter_z_stats_from_raw(raw_all, param_cols)
    coeffs = prepare_projected_lmm_coefficients(
        lmm_results=lmm_results,
        lmm_term_tests=lmm_term_tests,
        tract_family=tract_family,
        metric=metric,
        param_cols=param_cols,
    )

    for measure_name, prefix in [("age", "lmm_age_interaction"), ("hemisphere", "lmm_hemisphere_interaction")]:
        signed_values = []
        dominant_params = []
        dominant_betas = []
        dominant_p = []
        dominant_q = []
        dominant_delta_r2 = []
        contribution_texts = []

        for _, row in out.iterrows():
            total = 0.0
            has_any = False
            contributions: list[tuple[str, float, dict[str, object]]] = []

            for c in param_cols:
                if c not in coeffs[measure_name]:
                    continue
                beta_info = coeffs[measure_name][c]
                beta = float(beta_info["beta"])
                mean, sd = z_stats[c]
                z = (float(row[c]) - mean) / sd
                contrib = beta * z
                total += contrib
                has_any = True
                contributions.append((lmm_parameter_name(c), contrib, beta_info))

            if not has_any:
                signed_values.append(np.nan)
                dominant_params.append("")
                dominant_betas.append(np.nan)
                dominant_p.append(np.nan)
                dominant_q.append(np.nan)
                dominant_delta_r2.append(np.nan)
                contribution_texts.append("")
                continue

            dominant = max(contributions, key=lambda x: abs(x[1])) if contributions else ("", np.nan, {})
            signed_values.append(total)
            dominant_params.append(dominant[0])
            dominant_betas.append(dominant[2].get("beta", np.nan))
            dominant_p.append(dominant[2].get("p_lr", np.nan))
            dominant_q.append(dominant[2].get("p_lr_fdr", np.nan))
            dominant_delta_r2.append(dominant[2].get("delta_marginal_r2", np.nan))
            contribution_texts.append("; ".join(f"{p}:{v:.5g}" for p, v, _ in contributions))

        out[f"{prefix}_signed_effect"] = signed_values
        out[f"{prefix}_abs_effect"] = pd.to_numeric(out[f"{prefix}_signed_effect"], errors="coerce").abs()
        out[f"{prefix}_dominant_parameter"] = dominant_params
        out[f"{prefix}_dominant_parameter_beta"] = dominant_betas
        out[f"{prefix}_dominant_parameter_p_lr"] = dominant_p
        out[f"{prefix}_dominant_parameter_p_lr_fdr"] = dominant_q
        out[f"{prefix}_dominant_parameter_delta_marginal_r2"] = dominant_delta_r2
        out[f"{prefix}_parameter_contributions"] = contribution_texts

    return out



def add_measure_columns(
    bundle_raw: pd.DataFrame,
    combo_df: pd.DataFrame,
    bundle_name: str,
    param_cols: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = combo_df.copy()
    out["bundle"] = bundle_name
    out["combo_id"] = [
        stable_combo_id(bundle_name, {c: row[c] for c in param_cols})
        for _, row in out.iterrows()
    ]

    out = compute_information_shift_metric(bundle_raw, out, param_cols, args.outcome_col, args.information_divergence_bins)
    out = compute_shape_similarity_metric(bundle_raw, out, param_cols, args.shape_cols)
    out = compute_tract_size_metric(bundle_raw, out, param_cols, args.shape_cols)
    return out


def add_interaction_measure_columns(
    raw_all: pd.DataFrame,
    combo_df: pd.DataFrame,
    bundle_name: str,
    param_cols: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    lmm_results, lmm_term_tests = load_current_lmm_outputs(args.lmm_results_csv, args.lmm_term_tests_csv)
    lmm_metric = args.lmm_metric if args.lmm_metric else args.outcome_col
    return project_lmm_interaction_effects_to_combinations(
        combo_df=combo_df,
        raw_all=raw_all,
        lmm_results=lmm_results,
        lmm_term_tests=lmm_term_tests,
        bundle_name=bundle_name,
        param_cols=param_cols,
        metric=lmm_metric,
    )



def merge_embedding(combo_df: pd.DataFrame, embedding_df: pd.DataFrame, param_cols: Sequence[str]) -> pd.DataFrame:
    cols = list(param_cols) + ["x", "y", "z", "parameter_key"]
    out = combo_df.copy()
    if "parameter_key" in out.columns:
        out = out.drop(columns=["parameter_key"])
    out = out.merge(embedding_df[cols], on=list(param_cols), how="left", validate="one_to_one")
    return out


def fixed_scene_ranges(embedding_df: pd.DataFrame, pad_fraction: float = 0.08) -> dict[str, list[float]]:
    out = {}
    for axis in ["x", "y", "z"]:
        vals = embedding_df[axis].to_numpy(dtype=float)
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        span = hi - lo
        if span <= EPS:
            span = 1.0
        pad = span * pad_fraction
        out[axis] = [lo - pad, hi + pad]
    return out


def edge_xyz_for_subset(embedding_df: pd.DataFrame, edges: Sequence[tuple[int, int]], included_keys: set[tuple]) -> tuple[list[float], list[float], list[float]]:
    indexed = embedding_df.reset_index(drop=True)
    xs, ys, zs = [], [], []
    for i, j in edges:
        ki = indexed.loc[i, "parameter_key"]
        kj = indexed.loc[j, "parameter_key"]
        if ki not in included_keys or kj not in included_keys:
            continue
        p0 = indexed.loc[i, ["x", "y", "z"]].to_numpy(dtype=float)
        p1 = indexed.loc[j, ["x", "y", "z"]].to_numpy(dtype=float)
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]
    return xs, ys, zs


def format_float(x, digits=4) -> str:
    try:
        xf = float(x)
    except Exception:
        return "n/a"
    if not np.isfinite(xf):
        return "n/a"
    return f"{xf:.{digits}g}"


def hover_text(row: pd.Series, param_cols: Sequence[str], measure: MeasureSpec) -> str:
    parts = [f"<b>{measure.label}</b>"]
    for c in param_cols:
        parts.append(f"{c.replace('param_', '')}: {format_float(row[c])}")
    parts.append(f"recovery: {row.get('recovery_status', 'unknown')}")
    parts.append(f"value: {format_float(row.get(measure.column, np.nan))}")
    if measure.key in {"information_shift", "information_shift_scaled"}:
        parts.append(f"mean HD shift (global - combo): {format_float(row.get('information_shift_mean_hd_mm', np.nan))} mm")
        shift_dir = row.get("information_shift_direction", "")
        if isinstance(shift_dir, str) and shift_dir:
            parts.append(f"shift direction: {shift_dir}")
    parts.append(f"mean HD: {format_float(row.get('mean_hd', np.nan))}")
    parts.append(f"median HD: {format_float(row.get('median_hd', np.nan))}")
    return "<br>".join(parts)


def marker_sizes(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    sizes = np.full(len(df), args.simple_node_default_size, dtype=float)
    valid = df["mean_hd"].notna() if "mean_hd" in df.columns else pd.Series(False, index=df.index)
    if valid.any():
        vals = pd.to_numeric(df.loc[valid, "mean_hd"], errors="coerce")
        lo = float(vals.min())
        hi = float(vals.max())
        if hi > lo:
            scaled = 1.0 - (vals - lo) / (hi - lo)
            sizes[df.index.get_indexer(vals.index)] = args.simple_node_min_size + scaled * args.simple_node_size_range
        else:
            sizes[df.index.get_indexer(vals.index)] = args.simple_node_default_size
    sizes = np.clip(sizes, args.simple_node_min_size, args.simple_node_max_size)
    return sizes


def add_axis_traces(fig: go.Figure, embedding_df: pd.DataFrame, axis_vecs: np.ndarray, param_cols: Sequence[str], level_maps: dict[str, list[float]], axis_length: float) -> None:
    center = embedding_df[["x", "y", "z"]].mean().to_numpy(dtype=float)
    for i, c in enumerate(param_cols):
        v = axis_vecs[i]
        start = center - v * axis_length * 0.5
        end = center + v * axis_length * 0.5
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines+text",
                line=dict(width=3),
                text=["", c.replace("param_", "")],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
                name=f"axis {c}",
            )
        )

def compute_parameter_embedding_2d(
    grid: pd.DataFrame,
    param_cols: Sequence[str],
    level_maps: dict[str, list[float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray]:
    nd = normalized_index_coordinates(grid, param_cols, level_maps)

    if args.parameter_space_2d_method == "pca":
        n_components = min(2, nd.shape[0], nd.shape[1])
        emb = PCA(n_components=n_components, random_state=args.random_state).fit_transform(nd)
    elif args.parameter_space_2d_method == "metric_mds":
        distances = pairwise_distances(nd, metric="euclidean")
        try:
            model = MDS(
                n_components=2,
                metric=True,
                dissimilarity="precomputed",
                random_state=args.random_state,
                n_init=args.mds_n_init,
                max_iter=args.mds_max_iter,
                normalized_stress="auto",
            )
        except TypeError:
            model = MDS(
                n_components=2,
                metric=True,
                dissimilarity="precomputed",
                random_state=args.random_state,
                n_init=args.mds_n_init,
                max_iter=args.mds_max_iter,
            )
        emb = model.fit_transform(distances)
    elif args.parameter_space_2d_method == "classical_mds":
        distances = pairwise_distances(nd, metric="euclidean")
        d2 = distances ** 2
        n = d2.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ d2 @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order][:2], 0.0)
        eigvecs = eigvecs[:, order][:, :2]
        emb = eigvecs * np.sqrt(eigvals)[None, :]
    else:
        raise ValueError(f"Unknown 2D parameter-space embedding method: {args.parameter_space_2d_method}")

    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])), constant_values=0.0)

    emb = emb - emb.mean(axis=0, keepdims=True)

    tmp = grid.copy()
    tmp["map_x"] = emb[:, 0]
    tmp["map_y"] = emb[:, 1]

    # Deterministic orientation: align the first parameter's low->high direction with +x,
    # then reflect so the second parameter points upward when available.
    first = param_cols[0]
    lo = float(level_maps[first][0])
    hi = float(level_maps[first][-1])
    lo_mean = tmp.loc[np.isclose(tmp[first].astype(float), lo), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
    hi_mean = tmp.loc[np.isclose(tmp[first].astype(float), hi), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
    vec = hi_mean - lo_mean
    theta = np.arctan2(vec[1], vec[0]) if np.linalg.norm(vec) > EPS else 0.0
    rot = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]], dtype=float)
    emb = emb @ rot.T

    if len(param_cols) > 1:
        tmp2 = grid.copy()
        tmp2["map_x"] = emb[:, 0]
        tmp2["map_y"] = emb[:, 1]
        second = param_cols[1]
        lo2 = float(level_maps[second][0])
        hi2 = float(level_maps[second][-1])
        lo2_mean = tmp2.loc[np.isclose(tmp2[second].astype(float), lo2), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
        hi2_mean = tmp2.loc[np.isclose(tmp2[second].astype(float), hi2), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
        vec2 = hi2_mean - lo2_mean
        if vec2[1] < 0:
            emb[:, 1] *= -1.0

    scale = np.max(np.linalg.norm(emb, axis=1))
    if scale > 0:
        emb = emb / scale * args.embedding_scale

    out = grid.copy()
    out["parameter_key"] = parameter_key_df(out, param_cols)
    out["map_x"], out["map_y"] = emb[:, 0], emb[:, 1]

    axis_vecs = []
    for c in param_cols:
        lo = float(level_maps[c][0])
        hi = float(level_maps[c][-1])
        lo_mean = out.loc[np.isclose(out[c].astype(float), lo), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
        hi_mean = out.loc[np.isclose(out[c].astype(float), hi), ["map_x", "map_y"]].mean().to_numpy(dtype=float)
        vec = hi_mean - lo_mean
        norm = np.linalg.norm(vec)
        if norm <= EPS:
            vec = np.zeros(2)
        else:
            vec = vec / norm
        axis_vecs.append(vec)

    return out, np.asarray(axis_vecs, dtype=float)


def fixed_parameter_space_map_ranges(map_embedding_df: pd.DataFrame, pad_fraction: float = 0.015) -> dict[str, tuple[float, float]]:
    out = {}
    for c in ["map_x", "map_y"]:
        vals = pd.to_numeric(map_embedding_df[c], errors="coerce").to_numpy(dtype=float)
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        pad = span * pad_fraction
        out[c] = (lo - pad, hi + pad)
    return out


def edge_xy_for_parameter_space_map(
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    subset_keys: set[str],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    key_to_row = {str(r["parameter_key"]): r for _, r in map_embedding_df.iterrows()}
    key_list = map_embedding_df["parameter_key"].astype(str).tolist()
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i, j in edges:
        ki = key_list[i]
        kj = key_list[j]
        if ki not in subset_keys or kj not in subset_keys:
            continue
        ri = key_to_row[ki]
        rj = key_to_row[kj]
        segments.append(((float(ri["map_x"]), float(ri["map_y"])), (float(rj["map_x"]), float(rj["map_y"]))))
    return segments


def add_parameter_axes_to_map(
    ax: plt.Axes,
    map_embedding_df: pd.DataFrame,
    axis_vecs_2d: np.ndarray,
    param_cols: Sequence[str],
) -> None:
    center = map_embedding_df[["map_x", "map_y"]].mean().to_numpy(dtype=float)
    xr = ax.get_xlim()
    yr = ax.get_ylim()
    axis_scale = 0.30 * min(xr[1] - xr[0], yr[1] - yr[0])
    short_names = {
        "turning_angle": "angle",
        "step_size": "step",
        "smoothing": "smooth",
        "tip_iteration": "tip",
        "tolerance": "tol",
    }
    for i, c in enumerate(param_cols):
        vec = np.asarray(axis_vecs_2d[i], dtype=float)
        norm = np.linalg.norm(vec)
        if norm <= EPS:
            continue
        vec = vec / norm
        start = center - 0.50 * axis_scale * vec
        end = center + 0.50 * axis_scale * vec
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(arrowstyle="-|>", lw=1.55, color="black", shrinkA=0, shrinkB=0),
            zorder=5,
        )
        raw_label = c.replace("param_", "")
        label = short_names.get(raw_label, raw_label)
        label_offset = 0.15 * axis_scale * vec
        ax.text(
            end[0] + label_offset[0],
            end[1] + label_offset[1],
            label,
            fontsize=PARAMETER_AXIS_OVERLAY_FONT_SIZE,
            ha="center",
            va="center",
            color="black",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.84),
        )


def plot_parameter_space_map_panel(
    ax: plt.Axes,
    combo_df: pd.DataFrame,
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    measure: MeasureSpec,
    param_cols: Sequence[str],
    axis_vecs_2d: np.ndarray,
    map_ranges: dict[str, tuple[float, float]],
    args: argparse.Namespace,
    *,
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: str | None = None,
    show_title: bool = False,
    show_axes: bool = True,
) -> None:
    metric_available = combo_df[measure.column].notna() & combo_df["is_recovered_combination"].astype(bool)
    merged = combo_df.copy()
    if "parameter_key" not in merged.columns:
        merged["parameter_key"] = parameter_key_df(merged, param_cols)
    if "map_x" not in merged.columns or "map_y" not in merged.columns:
        merged = merged.merge(
            map_embedding_df[["parameter_key", "map_x", "map_y"]],
            on="parameter_key",
            how="left",
            validate="many_to_one",
        )

    available_subset = merged.loc[metric_available].copy()
    unavailable_subset = merged.loc[~metric_available].copy()

    if cmin is None or cmax is None or colorscale is None:
        cmin, cmax, colorscale = measure_color_settings(
            available_subset if not available_subset.empty else merged,
            measure,
            args,
        )

    cmap_name = mpl_cmap_name_from_scale(colorscale)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=cmin, vmax=cmax)

    all_keys = set(merged["parameter_key"].astype(str).tolist())
    for (x0, y0), (x1, y1) in edge_xy_for_parameter_space_map(map_embedding_df, edges, all_keys):
        ax.plot([x0, x1], [y0, y1], color=(120/255, 120/255, 120/255, 0.35), lw=args.lattice_line_width, zorder=1)

    if not unavailable_subset.empty:
        ax.scatter(
            unavailable_subset["map_x"],
            unavailable_subset["map_y"],
            s=float(args.parameter_space_2d_unavailable_node_size),
            c=[mpl_color_from_rgb_string(args.nonrecovery_gray)],
            alpha=0.85,
            edgecolors=(60/255, 60/255, 60/255, 0.45),
            linewidths=0.30,
            zorder=2,
        )

    if not available_subset.empty:
        sizes = marker_sizes(available_subset, args)
        ax.scatter(
            available_subset["map_x"],
            available_subset["map_y"],
            s=np.clip(sizes * float(args.parameter_space_2d_node_size_scale), 8.0, float(args.parameter_space_2d_node_size_max)),
            c=available_subset[measure.column],
            cmap=cmap,
            norm=norm,
            alpha=0.95,
            edgecolors=(40/255, 40/255, 40/255, 0.45),
            linewidths=0.30,
            zorder=3,
        )

    ax.set_xlim(*map_ranges["map_x"])
    ax.set_ylim(*map_ranges["map_y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(measure.label, fontsize=STATIC_PLOT_FONT_SIZE, pad=8)
    if show_axes:
        add_parameter_axes_to_map(ax, map_embedding_df, axis_vecs_2d, param_cols)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)

def parameter_space_2d_grid_position(bundle_name: str) -> tuple[int, int, str, str] | None:
    display = display_tract_name(bundle_name).strip().lower()
    parts = display.split()
    if len(parts) < 2:
        return None
    side = parts[0]
    family = parts[-1].upper()
    row_map = {"left": 0, "right": 1}
    col_map = {"AF": 0, "FX": 1, "CST": 2}
    if side not in row_map or family not in col_map:
        return None
    return row_map[side], col_map[family], side.capitalize(), family


def plot_parameter_space_metric_grid_figure(
    combo_by_bundle: dict[str, pd.DataFrame],
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    axis_vecs_2d: np.ndarray,
    map_ranges: dict[str, tuple[float, float]],
    param_cols: Sequence[str],
    args: argparse.Namespace,
    measure: MeasureSpec,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(float(args.parameter_space_2d_figure_width), float(args.parameter_space_2d_figure_height)),
        squeeze=False,
    )

    available_all = []
    for bundle_name, combo_df in combo_by_bundle.items():
        if measure.column not in combo_df.columns:
            continue
        mask = combo_df[measure.column].notna() & combo_df["is_recovered_combination"].astype(bool)
        if mask.any():
            available_all.append(combo_df.loc[mask].copy())
    available_concat = pd.concat(available_all, ignore_index=True) if available_all else pd.DataFrame(columns=[measure.column])
    cmin, cmax, colorscale = measure_color_settings(
        available_concat if not available_concat.empty else next(iter(combo_by_bundle.values())).copy(),
        measure,
        args,
    )
    cmap_name = mpl_cmap_name_from_scale(colorscale)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for bundle_name, combo_df in combo_by_bundle.items():
        pos = parameter_space_2d_grid_position(bundle_name)
        if pos is None:
            continue
        r, c, _, _ = pos
        ax = axes[r, c]
        plot_parameter_space_map_panel(
            ax=ax,
            combo_df=combo_df,
            map_embedding_df=map_embedding_df,
            edges=edges,
            measure=measure,
            param_cols=param_cols,
            axis_vecs_2d=axis_vecs_2d,
            map_ranges=map_ranges,
            args=args,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            show_title=False,
            show_axes=True,
        )

    for j, fam in enumerate(["AF", "FX", "CST"]):
        axes[0, j].text(
            0.5,
            1.095,
            fam,
            transform=axes[0, j].transAxes,
            ha="center",
            va="bottom",
            fontsize=23,
        )
        axes[0, j].plot(
            [0.07, 0.93],
            [1.035, 1.035],
            transform=axes[0, j].transAxes,
            color="black",
            lw=1.2,
            clip_on=False,
        )

    for r, label in [(0, "Left"), (1, "Right")]:
        axes[r, 0].text(
            -0.125,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=23,
        )
        axes[r, 0].plot(
            [-0.05, -0.05],
            [0.07, 0.93],
            transform=axes[r, 0].transAxes,
            color="black",
            lw=1.2,
            clip_on=False,
        )


    cax = fig.add_axes([0.915, 0.10, 0.012, 0.80])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
    cbar.set_label(
        plain_text_label(measure.colorbar_title),
        fontsize=STATIC_PLOT_FONT_SIZE,
        rotation=-90,
        labelpad=34,
    )

    fig.subplots_adjust(left=0.055, right=0.900, top=0.93, bottom=0.035, wspace=0.015, hspace=-0.03)
    return fig



def run_parameter_space_2d_outputs(raw_df: pd.DataFrame, param_cols: list[str], age_col: str | None, args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    level_maps = parameter_level_maps(raw_df, param_cols)
    full_grid = full_parameter_grid(param_cols, level_maps)
    map_embedding_df, axis_vecs_2d = compute_parameter_embedding_2d(full_grid, param_cols, level_maps, args)
    edges = build_hyperlattice_edges(map_embedding_df, param_cols, level_maps)
    map_ranges = fixed_parameter_space_map_ranges(map_embedding_df)

    plot_dir = out_dir / "per_metric_parameter_space_2d_maps"
    table_dir = out_dir / "per_tract_parameter_space_2d_tables"
    ensure_dir(plot_dir)
    ensure_dir(table_dir)

    map_embedding_path = out_dir / f"fixed_parameter_space_2d_embedding_{args.parameter_space_2d_method}.csv"
    map_embedding_df.to_csv(map_embedding_path, index=False)

    bundles = sorted(raw_df["bundle"].dropna().astype(str).unique())
    combo_by_bundle: dict[str, pd.DataFrame] = {}
    bundle_table_rows = []
    for bundle_name in bundles:
        bundle_raw = raw_df.loc[raw_df["bundle"].astype(str).eq(bundle_name)].copy()
        if bundle_raw.empty:
            continue
        combo_df = aggregate_combinations(bundle_raw, param_cols, args.outcome_col)
        combo_df = add_measure_columns(
            bundle_raw,
            combo_df,
            bundle_name,
            param_cols,
            args,
        )
        combo_df = add_interaction_measure_columns(
            raw_df,
            combo_df,
            bundle_name,
            param_cols,
            args,
        )
        combo_df["parameter_key"] = parameter_key_df(combo_df, param_cols)
        combo_df = combo_df.merge(
            map_embedding_df[["parameter_key", "map_x", "map_y"]],
            on="parameter_key",
            how="left",
            validate="many_to_one",
        )
        stem = sanitize_name(bundle_name)
        table_path = table_dir / f"{stem}__factorial_combinations_fixed_2d_parameter_space_embedding_multi_measure.csv"
        combo_df.to_csv(table_path, index=False)
        combo_by_bundle[bundle_name] = combo_df
        bundle_table_rows.append({"bundle": bundle_name, "combo_table_csv": str(table_path)})

    summary_rows = []
    for measure_key in PARAMETER_SPACE_2D_MEASURE_KEYS:
        if measure_key not in args.measures:
            continue
        measure = MEASURE_SPECS[measure_key]
        if not combo_by_bundle:
            continue

        fig = plot_parameter_space_metric_grid_figure(
            combo_by_bundle=combo_by_bundle,
            map_embedding_df=map_embedding_df,
            edges=edges,
            axis_vecs_2d=axis_vecs_2d,
            map_ranges=map_ranges,
            param_cols=param_cols,
            args=args,
            measure=measure,
        )
        stem = sanitize_name(measure.key)
        png_path = plot_dir / f"{stem}__fixed_2d_parameter_space_grid.png"
        fig.savefig(png_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches="tight")
        plt.close(fig)

        n_metric_available = 0
        n_metric_unavailable = 0
        metric_values = []
        for combo_df in combo_by_bundle.values():
            if measure.column not in combo_df.columns:
                continue
            mask = combo_df[measure.column].notna() & combo_df["is_recovered_combination"].astype(bool)
            n_metric_available += int(mask.sum())
            n_metric_unavailable += int((~mask).sum())
            if mask.any():
                metric_values.append(pd.to_numeric(combo_df.loc[mask, measure.column], errors="coerce"))
        if metric_values:
            all_vals = pd.concat(metric_values, ignore_index=True)
            metric_min = float(all_vals.min())
            metric_median = float(all_vals.median())
            metric_max = float(all_vals.max())
        else:
            metric_min = np.nan
            metric_median = np.nan
            metric_max = np.nan

        summary_rows.append(
            {
                "measure": measure_key,
                "parameter_space_2d_method": args.parameter_space_2d_method,
                "n_tracts": int(len(combo_by_bundle)),
                "n_metric_available_combinations_total": int(n_metric_available),
                "n_metric_unavailable_or_nonrecovered_combinations_total": int(n_metric_unavailable),
                "metric_min": metric_min,
                "metric_median": metric_median,
                "metric_max": metric_max,
                "parameter_space_2d_map_png": str(png_path),
                "map_embedding_csv": str(map_embedding_path),
            }
        )

    summary_path = out_dir / "parameter_space_2d_multi_measure_figure_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    bundle_table_manifest = out_dir / "parameter_space_2d_combo_table_manifest.csv"
    pd.DataFrame(bundle_table_rows).to_csv(bundle_table_manifest, index=False)

    return {
        "summary_csv": str(summary_path),
        "map_embedding_csv": str(map_embedding_path),
        "plot_dir": str(plot_dir),
        "table_dir": str(table_dir),
        "combo_table_manifest_csv": str(bundle_table_manifest),
    }


def fixed_map_ranges(map_embedding_df: pd.DataFrame, pad_fraction: float = 0.08) -> dict[str, tuple[float, float]]:
    out = {}
    for c in ["map_x", "map_y"]:
        vals = pd.to_numeric(map_embedding_df[c], errors="coerce").to_numpy(dtype=float)
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        pad = span * pad_fraction
        out[c] = (lo - pad, hi + pad)
    return out


def mpl_cmap_name_from_scale(colorscale: str) -> str:
    lookup = {
        "Magma": "magma",
        "RdBu_r": "RdBu_r",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Cividis": "cividis",
    }
    return lookup.get(str(colorscale), str(colorscale).lower())

def mpl_color_from_rgb_string(color: str) -> tuple[float, float, float]:
    s = str(color).strip()
    if s.startswith("rgb(") and s.endswith(")"):
        parts = [float(x.strip()) for x in s[4:-1].split(",")]
        if len(parts) == 3:
            return (parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0)
    return matplotlib.colors.to_rgb(s)

def plain_text_label(text: str) -> str:
    s = str(text)
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return s




def color_range(vals: pd.Series, robust: bool = True) -> tuple[float, float]:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    if x.empty:
        return 0.0, 1.0
    if robust and len(x) >= 10:
        lo = float(x.quantile(0.02))
        hi = float(x.quantile(0.98))
    else:
        lo = float(x.min())
        hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi
def precision_fraction_values(values: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmax(finite) > 1.0:
        vals = vals / 100.0
    return np.clip(vals, 0.0, 1.0)


def symmetric_percentile_color_norm(values: pd.Series, percentile: float = 98.0) -> plt.Normalize:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if vals.size == 0:
        vmax = 1.0
    else:
        p = float(np.clip(percentile, 50.0, 100.0))
        vmax = float(np.nanpercentile(np.abs(vals), p))
        if not np.isfinite(vmax) or vmax <= EPS:
            vmax = float(np.nanmax(np.abs(vals)))
        if not np.isfinite(vmax) or vmax <= EPS:
            vmax = 1.0
    return plt.Normalize(vmin=-vmax, vmax=vmax)
def add_within_tract_standardized_svm_scores(score_by_bundle: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for bundle_name, df in score_by_bundle.items():
        tmp = df.copy()
        tmp["svm_decision_score_z"] = np.nan
        scores = pd.to_numeric(tmp["svm_decision_score"], errors="coerce")
        recovered = tmp["is_recovered_combination"].astype(bool) if "is_recovered_combination" in tmp.columns else pd.Series(True, index=tmp.index)
        fit_mask = recovered & scores.notna()
        if int(fit_mask.sum()) >= 2 and float(scores.loc[fit_mask].std(ddof=0)) > EPS:
            scaler = StandardScaler()
            tmp.loc[fit_mask, "svm_decision_score_z"] = scaler.fit_transform(scores.loc[fit_mask].to_numpy(dtype=float).reshape(-1, 1)).ravel()
        elif int(fit_mask.sum()) == 1:
            tmp.loc[fit_mask, "svm_decision_score_z"] = 0.0
        out[bundle_name] = tmp
    return out


def measure_color_settings(plot_df: pd.DataFrame, measure: MeasureSpec, args: argparse.Namespace) -> tuple[float, float, str]:
    cmin, cmax = color_range(plot_df[measure.column], robust=True)
    colorscale = args.color_scale
    if measure.key in {"information_shift", "information_shift_scaled"}:
        vmax = max(abs(cmin), abs(cmax))
        if vmax <= EPS:
            vmax = 1.0
        cmin, cmax = -vmax, vmax
        colorscale = args.signed_information_color_scale
    return cmin, cmax, colorscale


def shared_information_color_settings(
    combo_by_bundle: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    measure = MEASURE_SPECS["information_shift"]
    available = []
    for combo_df in combo_by_bundle.values():
        if measure.column not in combo_df.columns:
            continue
        mask = combo_df[measure.column].notna() & combo_df["is_recovered_combination"].astype(bool)
        if mask.any():
            available.append(combo_df.loc[mask, [measure.column]].copy())
    if available:
        plot_df = pd.concat(available, ignore_index=True)
    else:
        plot_df = pd.DataFrame({measure.column: [np.nan]})
    return measure_color_settings(plot_df, measure, args)


def metric_level_masks(plot_df: pd.DataFrame, measure: MeasureSpec) -> dict[str, pd.Series]:
    metric_available = plot_df[measure.column].notna() & plot_df["is_recovered_combination"].astype(bool)
    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=plot_df.index),
        "unavailable": ~metric_available,
    }

    vals = pd.to_numeric(plot_df.loc[metric_available, measure.column], errors="coerce")
    if vals.empty or vals.nunique(dropna=True) < 3:
        masks["low"] = pd.Series(False, index=plot_df.index)
        masks["medium"] = metric_available.copy()
        masks["high"] = pd.Series(False, index=plot_df.index)
        return masks

    q1 = float(vals.quantile(1.0 / 3.0))
    q2 = float(vals.quantile(2.0 / 3.0))

    full_vals = pd.to_numeric(plot_df[measure.column], errors="coerce")
    masks["low"] = metric_available & full_vals.le(q1)
    masks["medium"] = metric_available & full_vals.gt(q1) & full_vals.le(q2)
    masks["high"] = metric_available & full_vals.gt(q2)
    return masks



def build_metric_colorbar_trace(
    measure: MeasureSpec,
    colorscale: str,
    cmin: float,
    cmax: float,
    visible: bool = False,
) -> go.Scatter3d:
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
        cmin, cmax = 0.0, 1.0
    return go.Scatter3d(
        x=[np.nan, np.nan],
        y=[np.nan, np.nan],
        z=[np.nan, np.nan],
        mode="markers",
        marker=dict(
            size=0.1,
            color=[cmin, cmax],
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=measure.colorbar_title),
            opacity=0.0,
            showscale=True,
        ),
        hoverinfo="skip",
        name=f"{measure.label}: colorbar",
        showlegend=False,
        visible=visible,
    )



def html_button_bar(active_metric: str, active_level: str, active_style: str) -> str:
    metrics = [
        ("information_shift", "Signed info divergence"),
        ("information_shift_scaled", "Scaled info divergence"),
        ("shape_similarity", "Shape similarity"),
        ("tract_size", "Tract size"),
    ]
    levels = [
        ("all", "All"),
        ("low", "Low"),
        ("medium", "Med"),
        ("high", "High"),
        ("unavailable", "Non-recovered/unavailable"),
    ]
    styles = [
        ("nodes_lines", "Nodes + lines"),
        ("nodes_only", "Nodes only"),
        ("lines_only", "Lines only"),
    ]

    def buttons(kind: str, items: list[tuple[str, str]], active: str) -> str:
        html = []
        for key, label in items:
            active_class = " active" if key == active else ""
            html.append(
                f'<button type="button" class="hlt-btn{active_class}" '
                f'data-hlt-kind="{kind}" data-hlt-value="{key}">{label}</button>'
            )
        return "\n".join(html)

    return f"""
<div class="hlt-controls">
  <div class="hlt-row"><span class="hlt-label">Metric</span>{buttons("metric", metrics, active_metric)}</div>
  <div class="hlt-row"><span class="hlt-label">Visible level</span>{buttons("level", levels, active_level)}</div>
  <div class="hlt-row"><span class="hlt-label">Style</span>{buttons("style", styles, active_style)}</div>
</div>
"""


def write_toggle_html(
    fig: go.Figure,
    out_path: Path,
    trace_meta: list[dict[str, str]],
    initial_metric: str,
    initial_level: str,
    initial_style: str,
) -> None:
    ensure_dir(out_path.parent)
    plot_id = "hlt_" + re.sub(r"[^A-Za-z0-9_]+", "_", sanitize_name(out_path.stem))
    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id=plot_id)
    controls = html_button_bar(initial_metric, initial_level, initial_style)

    meta_json = json.dumps(trace_meta, allow_nan=False)

    css = """
<style>
  .hlt-controls {
    font-family: Arial, sans-serif;
    margin: 10px 0 8px 0;
    padding: 10px 12px;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    max-width: 1100px;
  }
  .hlt-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 5px 0;
  }
  .hlt-label {
    min-width: 105px;
    font-size: 13px;
    font-weight: 600;
    color: #333;
  }
  .hlt-btn {
    border: 1px solid #b8b8b8;
    border-radius: 6px;
    padding: 5px 9px;
    background: #f6f6f6;
    color: #222;
    cursor: pointer;
    font-size: 13px;
  }
  .hlt-btn.active {
    background: #222;
    color: white;
    border-color: #222;
  }
  .hlt-status {
    font-family: Arial, sans-serif;
    color: #666;
    font-size: 12px;
    margin: 2px 0 8px 2px;
  }
</style>
"""

    js = f"""
<script>
(function() {{
  const plotId = {json.dumps(plot_id)};
  const traceMeta = {meta_json};
  const state = {{ metric: {json.dumps(initial_metric)}, level: {json.dumps(initial_level)}, style: {json.dumps(initial_style)} }};

  function root() {{
    return document.getElementById(plotId + "_container");
  }}

  function updateButtons() {{
    const r = root();
    if (!r) return;
    r.querySelectorAll("button[data-hlt-kind]").forEach(btn => {{
      const kind = btn.getAttribute("data-hlt-kind");
      const value = btn.getAttribute("data-hlt-value");
      btn.classList.toggle("active", state[kind] === value);
    }});
  }}

  function metricLevelMatches(meta) {{
    if (meta.role === "axis") return true;
    return meta.metric === state.metric && meta.level === state.level;
  }}

  function styleShows(meta) {{
    if (meta.role === "axis" || meta.role === "colorbar") return true;
    if (meta.role === "edge") {{
      return state.style === "nodes_lines" || state.style === "lines_only";
    }}
    if (meta.role === "node") {{
      return state.style === "nodes_lines" || state.style === "nodes_only";
    }}
    return false;
  }}

  function computeVisibility() {{
    return traceMeta.map(meta => metricLevelMatches(meta) && styleShows(meta));
  }}

  function applyState() {{
    const gd = document.getElementById(plotId);
    if (!gd || !window.Plotly || !gd.data || gd.data.length === 0) {{
      setTimeout(applyState, 100);
      return;
    }}
    const visible = computeVisibility();
    if (visible.length !== gd.data.length) {{
      console.error("Hyperlattice trace metadata length mismatch", visible.length, gd.data.length);
      return;
    }}
    Plotly.restyle(gd, {{ visible: visible }});
    const titleMap = gd.layout.meta && gd.layout.meta.titleByMetric ? gd.layout.meta.titleByMetric : {{}};
    if (titleMap[state.metric]) {{
      Plotly.relayout(gd, {{ "title.text": titleMap[state.metric] + " — " + state.level + " — " + state.style }});
    }}
    updateButtons();
  }}

  function installControls() {{
    const r = root();
    if (!r) {{
      setTimeout(installControls, 100);
      return;
    }}
    r.querySelectorAll("button[data-hlt-kind]").forEach(btn => {{
      btn.addEventListener("click", () => {{
        const kind = btn.getAttribute("data-hlt-kind");
        const value = btn.getAttribute("data-hlt-value");
        state[kind] = value;
        applyState();
      }});
    }});
    applyState();
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", installControls);
  }} else {{
    installControls();
  }}
}})();
</script>
"""

    container_open = f'<div id="{plot_id}_container">\n'
    container_close = "</div>\n"
    html = html.replace("<body>", "<body>\n" + container_open + css + controls + '<div class="hlt-status">If the plot is still loading, wait for Plotly to finish initializing before clicking controls.</div>\n', 1)
    html = html.replace("</body>", container_close + js + "\n</body>", 1)
    out_path.write_text(html)
def plot_combined_metric_toggle_hyperlattice(
    combo_df: pd.DataFrame,
    embedding_full: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    axis_vecs: np.ndarray,
    scene_ranges: dict[str, list[float]],
    param_cols: Sequence[str],
    level_maps: dict[str, list[float]],
    args: argparse.Namespace,
    bundle_name: str,
    shared_information_scale: tuple[float, float, str] | None = None,
) -> tuple[go.Figure, list[dict[str, str]], str, str, str]:
    measures = [
        MEASURE_SPECS[k]
        for k in HYPERLATTICE_MEASURE_KEYS
        if (
            (k in args.measures or (k == "information_shift_scaled" and "information_shift" in args.measures))
            and MEASURE_SPECS[k].column in combo_df.columns
        )
    ]
    initial_metric = measures[0].key if measures else "information_shift"
    initial_level = "all"
    initial_style = "nodes_lines"

    plot_df = combo_df.copy()
    plot_df["parameter_key"] = parameter_key_df(plot_df, param_cols)

    fig = go.Figure()
    trace_meta: list[dict[str, str]] = []
    metric_title_map = {}
    bundle_display_name = display_tract_name(bundle_name)

    def add_empty_node_trace(measure: MeasureSpec, level: str, is_initial: bool) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=args.simple_node_default_size),
                hoverinfo="skip",
                showlegend=False,
                name=f"{measure.label}: {level} nodes",
                visible=is_initial,
            )
        )
        trace_meta.append({"role": "node", "metric": measure.key, "level": level})

    def add_node_trace(
        subset: pd.DataFrame,
        measure: MeasureSpec,
        level: str,
        colorscale: str,
        cmin: float,
        cmax: float,
        is_initial: bool,
        unavailable_style: bool,
        suffix: str,
    ) -> None:
        if subset.empty:
            return

        if unavailable_style:
            node_marker = dict(
                size=args.simple_node_min_size,
                color=args.nonrecovery_gray,
                opacity=0.72,
                line=dict(width=0.3, color="rgba(60,60,60,0.45)"),
            )
        else:
            node_marker = dict(
                size=marker_sizes(subset, args),
                color=subset[measure.column],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
                opacity=0.93,
                line=dict(width=0.3, color="rgba(40,40,40,0.55)"),
            )

        fig.add_trace(
            go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="markers",
                marker=node_marker,
                hovertext=[hover_text(row, param_cols, measure) for _, row in subset.iterrows()],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name=f"{measure.label}: {level} {suffix} nodes",
                visible=is_initial,
            )
        )
        trace_meta.append({"role": "node", "metric": measure.key, "level": level})

    for measure in measures:
        metric_title_map[measure.key] = f"{args.analysis_label}: {measure.label} — {bundle_display_name}"
        masks = metric_level_masks(plot_df, measure)
        metric_available = plot_df[measure.column].notna() & plot_df["is_recovered_combination"].astype(bool)
        available_for_color = plot_df.loc[metric_available]
        if measure.key == "information_shift_scaled" and shared_information_scale is not None:
            cmin, cmax, colorscale = shared_information_scale
        else:
            cmin, cmax, colorscale = measure_color_settings(available_for_color if not available_for_color.empty else plot_df, measure, args)

        for level in ["all", "low", "medium", "high", "unavailable"]:
            subset = plot_df.loc[masks[level]].copy()
            subset_keys = set(subset["parameter_key"].tolist())
            is_initial = measure.key == initial_metric and level == initial_level

            ex, ey, ez = edge_xyz_for_subset(embedding_full, edges, subset_keys)
            fig.add_trace(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(width=args.lattice_line_width, color="rgba(120,120,120,0.35)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{measure.label}: {level} lines",
                    visible=is_initial,
                )
            )
            trace_meta.append({"role": "edge", "metric": measure.key, "level": level})

            if subset.empty:
                add_empty_node_trace(measure, level, is_initial)
                continue

            if level == "all":
                available_subset = subset.loc[metric_available.reindex(subset.index).fillna(False)].copy()
                unavailable_subset = subset.loc[~metric_available.reindex(subset.index).fillna(False)].copy()

                if available_subset.empty and unavailable_subset.empty:
                    add_empty_node_trace(measure, level, is_initial)
                else:
                    add_node_trace(
                        subset=available_subset,
                        measure=measure,
                        level=level,
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        is_initial=is_initial,
                        unavailable_style=False,
                        suffix="recovered",
                    )
                    add_node_trace(
                        subset=unavailable_subset,
                        measure=measure,
                        level=level,
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        is_initial=is_initial,
                        unavailable_style=True,
                        suffix="unavailable",
                    )
                    if not available_subset.empty:
                        fig.add_trace(
                            build_metric_colorbar_trace(
                                measure=measure,
                                colorscale=colorscale,
                                cmin=cmin,
                                cmax=cmax,
                                visible=is_initial,
                            )
                        )
                        trace_meta.append({"role": "colorbar", "metric": measure.key, "level": level})
                continue

            if level == "unavailable":
                add_node_trace(
                    subset=subset,
                    measure=measure,
                    level=level,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    is_initial=is_initial,
                    unavailable_style=True,
                    suffix="unavailable",
                )
                continue

            add_node_trace(
                subset=subset,
                measure=measure,
                level=level,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                is_initial=is_initial,
                unavailable_style=False,
                suffix="recovered",
            )
            if not subset.empty:
                fig.add_trace(
                    build_metric_colorbar_trace(
                        measure=measure,
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        visible=is_initial,
                    )
                )
                trace_meta.append({"role": "colorbar", "metric": measure.key, "level": level})

    axis_length = args.embedding_scale * 1.10
    before_axes_n = len(fig.data)
    add_axis_traces(fig, embedding_full, axis_vecs, param_cols, level_maps, axis_length=axis_length)
    for _ in range(len(fig.data) - before_axes_n):
        trace_meta.append({"role": "axis", "metric": "", "level": ""})

    fig.update_layout(
        title=dict(
            text=f"{args.analysis_label}: {MEASURE_SPECS[initial_metric].label} — {bundle_display_name} — all — nodes_lines",
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        meta=dict(titleByMetric=metric_title_map),
        scene=dict(
            xaxis_title="MDS x" if "mds" in args.embedding_method else "embedding x",
            yaxis_title="MDS y" if "mds" in args.embedding_method else "embedding y",
            zaxis_title="MDS z" if "mds" in args.embedding_method else "embedding z",
            xaxis=dict(range=scene_ranges["x"], autorange=False),
            yaxis=dict(range=scene_ranges["y"], autorange=False),
            zaxis=dict(range=scene_ranges["z"], autorange=False),
            aspectmode="cube",
        ),
        height=900,
        width=1000,
        margin=dict(l=10, r=10, b=10, t=150),
        annotations=[
            dict(
                text=(
                    "Metric buttons switch the colored layer; level buttons filter that metric; "
                    "style buttons switch nodes and parameter-adjacent lines. "
                    "All includes recovered metric-colored nodes and gray non-recovered/unavailable nodes."
                ),
                x=0.0,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
            )
        ],
    )
    return fig, trace_meta, initial_metric, initial_level, initial_style


# ---------------------------------------------------------------------
# LMM helpers adapted from uploaded factorial LMM script
# ---------------------------------------------------------------------

def bh_adjust(df: pd.DataFrame, p_col: str, out_col: str, group_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out

    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, idx in out.groupby(group_key, dropna=False).groups.items():
        pvals = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        ok = pvals.notna()
        if ok.sum() == 0:
            continue
        adj = np.full(len(pvals), np.nan, dtype=float)
        adj_vals = multipletests(pvals.loc[ok], method="fdr_bh")[1]
        adj[np.where(ok)[0]] = adj_vals
        out.loc[idx, out_col] = adj
    return out

def fit_main_lmm(df: pd.DataFrame, tract_family: str, parameter: str, metric: str, reml: bool = True):
    sub = df.loc[(df["tract_family"] == tract_family) & (df["parameter"] == parameter)].copy()
    sub = sub.dropna(subset=[metric, "param_z", "age_z", "subject", "hemisphere"])

    if sub.empty:
        raise ValueError("No successful rows after filtering.")
    if sub["subject"].nunique() < 3:
        raise ValueError("Too few subjects for MixedLM.")

    formula = FULL_FORMULA_TEMPLATE.format(metric=metric)
    md = smf.mixedlm(
        formula=formula,
        data=sub,
        groups=sub["subject"],
        re_formula="1 + age_z",
    )
    fit = md.fit(reml=reml, method="lbfgs")
    return fit, sub

def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _compute_mixedlm_r2(fit, data: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    exog = np.asarray(fit.model.exog, dtype=float)
    beta = np.asarray(fit.fe_params, dtype=float)
    fixed_pred = exog @ beta
    var_fixed = float(np.var(fixed_pred, ddof=0))

    var_random = 0.0
    cov_re = getattr(fit, "cov_re", None)
    exog_re = getattr(fit.model, "exog_re", None)
    if cov_re is not None and exog_re is not None:
        z = np.asarray(exog_re, dtype=float)
        g = np.asarray(cov_re, dtype=float)
        if z.ndim == 2 and g.ndim == 2 and z.shape[1] == g.shape[0]:
            var_random = float(np.mean(np.einsum("ij,jk,ik->i", z, g, z)))

    var_resid = float(fit.scale)
    total = var_fixed + var_random + var_resid
    if total <= 0 or not np.isfinite(total):
        return math.nan, math.nan, var_fixed, var_random, var_resid

    marginal_r2 = var_fixed / total
    conditional_r2 = (var_fixed + var_random) / total
    return marginal_r2, conditional_r2, var_fixed, var_random, var_resid

def summarize_lmm_result(fit, tract_family: str, parameter: str, metric: str, n_rows: int, n_subjects: int) -> Dict[str, float]:
    marginal_r2, conditional_r2, var_fixed, var_random, var_resid = _compute_mixedlm_r2(fit, fit.model.data.frame)
    out: Dict[str, float] = {
        "tract_family": tract_family,
        "parameter": parameter,
        "metric": metric,
        "n_rows": n_rows,
        "n_subjects": n_subjects,
        "converged": bool(getattr(fit, "converged", False)),
        "llf": _safe_float(fit.llf),
        "aic": _safe_float(fit.aic),
        "bic": _safe_float(fit.bic),
        "residual_variance": _safe_float(fit.scale),
        "marginal_r2": marginal_r2,
        "conditional_r2": conditional_r2,
        "var_fixed": var_fixed,
        "var_random": var_random,
        "var_residual": var_resid,
    }

    for name, val in fit.fe_params.items():
        out[f"fe_{name}"] = float(val)
    for name, val in fit.bse_fe.items():
        out[f"se_{name}"] = float(val)
    for name, val in fit.pvalues.items():
        out[f"p_{name}"] = float(val)

    cov_re = fit.cov_re
    if cov_re is not None:
        for i, row_name in enumerate(cov_re.index):
            for j, col_name in enumerate(cov_re.columns):
                out[f"cov_re_{row_name}_{col_name}"] = float(cov_re.iloc[i, j])

    return out

def fit_reduced_ml_model(sub: pd.DataFrame, metric: str, reduced_formula: str):
    md = smf.mixedlm(
        formula=reduced_formula.format(metric=metric),
        data=sub,
        groups=sub["subject"],
        re_formula="1 + age_z",
    )
    fit = md.fit(reml=False, method="lbfgs")
    return fit

def term_test_rows(full_fit_ml, sub: pd.DataFrame, tract_family: str, parameter: str, metric: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    full_marginal_r2, full_conditional_r2, _, _, _ = _compute_mixedlm_r2(full_fit_ml, sub)

    for term, reduced_formula in REDUCED_FORMULAS.items():
        row: Dict[str, object] = {
            "tract_family": tract_family,
            "parameter": parameter,
            "metric": metric,
            "term": term,
            "full_formula": FULL_FORMULA_TEMPLATE.format(metric=metric),
            "reduced_formula": reduced_formula.format(metric=metric),
            "full_llf_ml": _safe_float(full_fit_ml.llf),
            "full_aic_ml": _safe_float(full_fit_ml.aic),
            "full_bic_ml": _safe_float(full_fit_ml.bic),
            "full_marginal_r2": full_marginal_r2,
            "full_conditional_r2": full_conditional_r2,
            "converged_full_ml": bool(getattr(full_fit_ml, "converged", False)),
            "converged_reduced_ml": False,
            "fe_name": TERM_TO_FE_NAME.get(term, ""),
            "estimate_reml": math.nan,
            "se_reml": math.nan,
            "p_reml": math.nan,
            "estimate_ml": math.nan,
            "se_ml": math.nan,
            "p_ml": math.nan,
            "reduced_llf_ml": math.nan,
            "reduced_aic_ml": math.nan,
            "reduced_bic_ml": math.nan,
            "reduced_marginal_r2": math.nan,
            "reduced_conditional_r2": math.nan,
            "delta_marginal_r2": math.nan,
            "delta_conditional_r2": math.nan,
            "local_f2": math.nan,
            "lr_stat": math.nan,
            "df_diff": math.nan,
            "p_lr": math.nan,
            "error": "",
        }

        fe_name = row["fe_name"]
        if fe_name in full_fit_ml.fe_params.index:
            row["estimate_ml"] = float(full_fit_ml.fe_params[fe_name])
        if fe_name in full_fit_ml.bse_fe.index:
            row["se_ml"] = float(full_fit_ml.bse_fe[fe_name])
        if fe_name in full_fit_ml.pvalues.index:
            row["p_ml"] = float(full_fit_ml.pvalues[fe_name])

        try:
            reduced_fit_ml = fit_reduced_ml_model(sub, metric, reduced_formula)
            reduced_marginal_r2, reduced_conditional_r2, _, _, _ = _compute_mixedlm_r2(reduced_fit_ml, sub)

            lr_stat = 2.0 * (float(full_fit_ml.llf) - float(reduced_fit_ml.llf))
            full_terms = set(full_fit_ml.fe_params.index)
            reduced_terms = set(reduced_fit_ml.fe_params.index)
            df_diff = max(len(full_terms - reduced_terms), 1)
            p_lr = float(stats.chi2.sf(max(lr_stat, 0.0), df_diff))

            delta_marginal_r2 = full_marginal_r2 - reduced_marginal_r2 if pd.notna(full_marginal_r2) and pd.notna(reduced_marginal_r2) else math.nan
            delta_conditional_r2 = full_conditional_r2 - reduced_conditional_r2 if pd.notna(full_conditional_r2) and pd.notna(reduced_conditional_r2) else math.nan
            local_f2 = math.nan
            if pd.notna(full_marginal_r2) and full_marginal_r2 < 1 and pd.notna(delta_marginal_r2):
                local_f2 = delta_marginal_r2 / (1.0 - full_marginal_r2)

            row.update({
                "converged_reduced_ml": bool(getattr(reduced_fit_ml, "converged", False)),
                "reduced_llf_ml": _safe_float(reduced_fit_ml.llf),
                "reduced_aic_ml": _safe_float(reduced_fit_ml.aic),
                "reduced_bic_ml": _safe_float(reduced_fit_ml.bic),
                "reduced_marginal_r2": reduced_marginal_r2,
                "reduced_conditional_r2": reduced_conditional_r2,
                "delta_marginal_r2": delta_marginal_r2,
                "delta_conditional_r2": delta_conditional_r2,
                "local_f2": local_f2,
                "lr_stat": lr_stat,
                "df_diff": df_diff,
                "p_lr": p_lr,
            })
        except Exception as e:
            row["error"] = str(e)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Integrated factorial-only driver
# ---------------------------------------------------------------------


MORPHOLOGY_METRICS = [
    "streamline_count",
    "tract_volume_mm3",
    "surface_area_mm2",
    "mean_length_mm",
]

HAUSDORFF_METRIC = "hausdorff_mm_vs_reference"
MODEL_METRICS = MORPHOLOGY_METRICS + [HAUSDORFF_METRIC]

FULL_FORMULA_TEMPLATE = (
    "{metric} ~ param_z * age_z + C(hemisphere) "
    "+ param_z:C(hemisphere) + age_z:C(hemisphere)"
)

REDUCED_FORMULAS = {
    "param_z": "{metric} ~ age_z + C(hemisphere) + age_z:C(hemisphere)",
    "param_z:age_z": "{metric} ~ param_z + age_z + C(hemisphere) + param_z:C(hemisphere) + age_z:C(hemisphere)",
    "param_z:C(hemisphere)[T.R]": "{metric} ~ param_z * age_z + C(hemisphere) + age_z:C(hemisphere)",
    "age_z:C(hemisphere)[T.R]": "{metric} ~ param_z * age_z + C(hemisphere) + param_z:C(hemisphere)",
}

TERM_TO_FE_NAME = {
    "param_z": "fe_param_z",
    "param_z:age_z": "fe_param_z:age_z",
    "param_z:C(hemisphere)[T.R]": "fe_param_z:C(hemisphere)[T.R]",
    "age_z:C(hemisphere)[T.R]": "fe_age_z:C(hemisphere)[T.R]",
}

RECOVERED_STATUSES = {"ok", "success", "completed"}
VALID_ANALYSIS_STATUSES = RECOVERED_STATUSES | {"no_output"}


def normalized_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def valid_analysis_status_mask(df: pd.DataFrame) -> pd.Series:
    if "status" not in df.columns:
        return pd.Series(True, index=df.index)
    return normalized_status(df["status"]).isin(VALID_ANALYSIS_STATUSES)


def recovered_status_mask(df: pd.DataFrame) -> pd.Series:
    if "status" not in df.columns:
        return pd.Series(False, index=df.index)
    return normalized_status(df["status"]).isin(RECOVERED_STATUSES)


def streamline_positive_mask(df: pd.DataFrame) -> pd.Series:
    if "streamline_count" not in df.columns:
        return recovered_status_mask(df)
    return pd.to_numeric(df["streamline_count"], errors="coerce").fillna(0).gt(0)


def finite_outcome_mask(df: pd.DataFrame, outcome_col: str) -> pd.Series:
    if outcome_col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[outcome_col], errors="coerce").notna()


def analytically_recovered_mask(df: pd.DataFrame, outcome_col: str | None = None, require_finite_outcome: bool = False) -> pd.Series:
    mask = recovered_status_mask(df) & streamline_positive_mask(df)
    if require_finite_outcome and outcome_col is not None:
        mask = mask & finite_outcome_mask(df, outcome_col)
    return mask


def status_consistency_qc(df: pd.DataFrame, outcome_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc = df.copy()
    qc["status_norm"] = normalized_status(qc["status"]) if "status" in qc.columns else "missing_status"
    qc["valid_analysis_status"] = valid_analysis_status_mask(qc)
    qc["recovered_status"] = recovered_status_mask(qc)
    qc["streamline_positive"] = streamline_positive_mask(qc)
    qc["finite_outcome"] = finite_outcome_mask(qc, outcome_col)
    qc["status_streamline_inconsistent"] = (
        (qc["recovered_status"] & ~qc["streamline_positive"])
        | (~qc["recovered_status"] & qc["streamline_positive"])
    )
    qc["status_hd_inconsistent"] = qc["finite_outcome"] & ~qc["recovered_status"]

    group_cols = [
        "status_norm",
        "valid_analysis_status",
        "recovered_status",
        "streamline_positive",
        "finite_outcome",
        "status_streamline_inconsistent",
        "status_hd_inconsistent",
    ]
    summary = qc.groupby(group_cols, dropna=False).size().reset_index(name="n_rows")
    detail_cols = [c for c in ["subject", "session", "bundle", "combo_id", "status", "streamline_count", outcome_col, "source_summary_csv"] if c in qc.columns]
    detail = qc.loc[
        (~qc["valid_analysis_status"]) | qc["status_streamline_inconsistent"] | qc["status_hd_inconsistent"],
        detail_cols + ["valid_analysis_status", "status_streamline_inconsistent", "status_hd_inconsistent"],
    ].copy()
    return summary, detail


def apply_valid_status_filter(df: pd.DataFrame, keep_invalid_status: bool = False) -> pd.DataFrame:
    if keep_invalid_status:
        return df.copy()
    return df.loc[valid_analysis_status_mask(df)].copy()


def lmm_factorial_parameter_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "param_template",
        "param_fa_threshold",
        "param_track_voxel_ratio",
    }
    cols = [c for c in df.columns if c.startswith("param_") and c not in excluded]
    varying = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        if vals.dropna().nunique() > 1:
            varying.append(col)
    return sorted(varying)


def expand_factorial_to_parameter_long_factorial_only(df: pd.DataFrame) -> pd.DataFrame:
    param_cols = lmm_factorial_parameter_columns(df)
    if not param_cols:
        raise ValueError("No varying param_* columns found for factorial analysis.")

    frames = []
    for col in param_cols:
        param = col.removeprefix("param_")
        tmp = df.copy()
        tmp["parameter"] = param
        tmp["value"] = tmp[col]
        tmp["param_value"] = pd.to_numeric(tmp[col], errors="coerce")
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    bad = out.loc[out["param_value"].isna(), ["parameter", "value"]].drop_duplicates()
    if not bad.empty:
        raise ValueError(f"Non-numeric factorial parameter values found:\n{bad}")
    return out


def load_factorial_runs_for_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    if args.runs_csv:
        frames = []
        for p in expand_input_paths(args.runs_csv):
            df = pd.read_csv(p, low_memory=False)
            if df.empty:
                continue
            df["source_summary_csv"] = str(p)
            frames.append(df)
        if not frames:
            raise FileNotFoundError("No non-empty --runs_csv files were found.")
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = load_factorial_shards(Path(args.summary_root), args.patterns)

    if "run_type" in raw.columns:
        raw = raw.loc[raw["run_type"].astype(str).str.lower().eq("factorial")].copy()
    if raw.empty:
        raise ValueError("No factorial rows found.")

    raw = add_family_hemi_columns(raw)
    raw = apply_optional_filters(raw, args)

    if raw.empty:
        raise ValueError("No factorial rows remained after loading/filter selection.")

    source_col = "source_summary_csv"
    if source_col in raw.columns:
        dedupe_cols = [c for c in raw.columns if c != source_col]
        raw[source_col] = raw.groupby(dedupe_cols, dropna=False)[source_col].transform(
            lambda values: ";".join(sorted(set(map(str, values))))
        )
        raw = raw.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
    else:
        raw = raw.drop_duplicates().reset_index(drop=True)

    return raw


def prepare_factorial_model_inputs(raw_runs: pd.DataFrame, age_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = raw_runs.copy()
    ages = pd.read_csv(age_csv)

    required_runs = {"subject", "session", "bundle", "status"}
    missing_runs = required_runs - set(runs.columns)
    if missing_runs:
        raise ValueError(f"Missing required columns in runs data: {sorted(missing_runs)}")

    required_age = {"subject", "session", "age_weeks"}
    missing_age = required_age - set(ages.columns)
    if missing_age:
        raise ValueError(f"Missing required columns in age CSV: {sorted(missing_age)}")

    df = expand_factorial_to_parameter_long_factorial_only(runs)
    df = df.merge(
        ages[["subject", "session", "age_weeks"]],
        on=["subject", "session"],
        how="left",
        validate="many_to_one",
    )
    if df["age_weeks"].isna().any():
        missing = df.loc[df["age_weeks"].isna(), ["subject", "session"]].drop_duplicates()
        raise ValueError(f"Missing age_weeks for some subject/session rows:\n{missing}")

    parsed = df["bundle"].apply(split_bundle)
    df["tract_family"] = parsed.apply(lambda x: x[0])
    df["hemisphere"] = parsed.apply(lambda x: x[1])
    df = df.loc[df["hemisphere"].isin(["L", "R"])].copy()
    if df.empty:
        raise ValueError("No L/R bilateral rows remained after parsing bundle names.")

    df["tract_found"] = analytically_recovered_mask(df, require_finite_outcome=False).astype(int)

    age_mean = df["age_weeks"].mean()
    age_std = df["age_weeks"].std(ddof=0)
    if age_std == 0:
        raise ValueError("age_weeks has zero variance.")
    df["age_z"] = (df["age_weeks"] - age_mean) / age_std

    def _z_param(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    df["param_z"] = df.groupby("parameter")["param_value"].transform(_z_param)
    df["hemisphere"] = pd.Categorical(df["hemisphere"], categories=["L", "R"])
    df["session"] = pd.Categorical(df["session"], categories=sorted(df["session"].astype(str).unique()), ordered=True)
    df["subject_session_id"] = df["subject"].astype(str) + "_" + df["session"].astype(str)
    df["sweep_type"] = "factorial"

    success = df.loc[df["tract_found"] == 1].copy()
    return df, success


def run_all_lmms_minimal(df_success: pd.DataFrame, metrics: list[str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    model_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []

    for tract_family in sorted(df_success["tract_family"].unique()):
        for parameter in sorted(df_success["parameter"].unique()):
            for metric in metrics:
                try:
                    fit_reml, sub = fit_main_lmm(df_success, tract_family, parameter, metric, reml=True)
                    fit_ml, _ = fit_main_lmm(df_success, tract_family, parameter, metric, reml=False)
                    model_rows.append(
                        summarize_lmm_result(
                            fit=fit_reml,
                            tract_family=tract_family,
                            parameter=parameter,
                            metric=metric,
                            n_rows=len(sub),
                            n_subjects=sub["subject"].nunique(),
                        )
                    )
                    term_rows.extend(term_test_rows(fit_ml, sub, tract_family, parameter, metric))
                except Exception as e:
                    model_rows.append({
                        "tract_family": tract_family,
                        "parameter": parameter,
                        "metric": metric,
                        "converged": False,
                        "error": str(e),
                    })
                    for term in REDUCED_FORMULAS:
                        term_rows.append({
                            "tract_family": tract_family,
                            "parameter": parameter,
                            "metric": metric,
                            "term": term,
                            "error": str(e),
                        })

    model_df = pd.DataFrame(model_rows)
    term_df = pd.DataFrame(term_rows)

    for fe_name in ["fe_param_z", "fe_param_z:age_z", "fe_param_z:C(hemisphere)[T.R]", "fe_age_z:C(hemisphere)[T.R]"]:
        p_col = f"p_{fe_name.replace('fe_', '')}"
        if p_col in model_df.columns:
            model_df = bh_adjust(model_df, p_col, f"{p_col}_fdr", ["metric"])

    if not term_df.empty and "p_lr" in term_df.columns:
        term_df = bh_adjust(term_df, "p_lr", "p_lr_fdr", ["term", "metric"])

    return model_df, term_df


def prepare_hyper_raw_from_filtered_runs(raw_runs: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str | None]:
    raw = raw_runs.copy()
    raw = add_family_hemi_columns(raw)

    if "bundle" not in raw.columns:
        raise ValueError("Expected a 'bundle' column to create one plot per tract.")

    if args.outcome_col not in raw.columns:
        raise ValueError(f"Missing outcome column: {args.outcome_col}")
    raw[args.outcome_col] = pd.to_numeric(raw[args.outcome_col], errors="coerce")

    raw["is_recovered_row"] = analytically_recovered_mask(
        raw,
        outcome_col=args.outcome_col,
        require_finite_outcome=True,
    )
    raw["is_attempted_row"] = True

    param_cols = factorial_parameter_columns(raw)
    if not param_cols:
        raise ValueError("No varying factorial param_* columns found.")

    for col in param_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=param_cols).copy()

    for c in DEFAULT_SHAPE_COLS:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    age_col = detect_age_column(raw, args.age_col)
    if age_col is not None:
        raw[age_col] = pd.to_numeric(raw[age_col], errors="coerce")

    if raw.empty:
        raise ValueError("No rows remained after filtering and parameter cleanup.")
    return raw, param_cols, age_col


def tract_similarity_inputs(raw_df: pd.DataFrame, param_cols: Sequence[str], outcome_col: str) -> pd.DataFrame:
    work = raw_df.copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work["_recovered_for_hd"] = analytically_recovered_mask(
        work,
        outcome_col=outcome_col,
        require_finite_outcome=True,
    )
    work["parameter_key"] = parameter_key_df(work, param_cols)

    grouped = work.groupby(["bundle", "parameter_key"], dropna=False)
    rows = []
    for (bundle, key), g in grouped:
        attempted = int(len(g))
        recovered = int(g["_recovered_for_hd"].sum())
        mean_hd = float(g.loc[g["_recovered_for_hd"], outcome_col].mean()) if recovered else np.nan
        row = {
            "bundle": bundle,
            "parameter_key": str(key),
            "attempted_rows": attempted,
            "recovered_rows": recovered,
            "recovery_fraction": recovered / attempted if attempted else np.nan,
            "mean_hd": mean_hd,
            "attempted_not_recovered": attempted > 0 and recovered == 0,
        }
        if isinstance(key, tuple):
            for c, v in zip(param_cols, key):
                row[c] = v
        rows.append(row)
    out = pd.DataFrame(rows)

    score_frames = []
    for bundle, sub in out.groupby("bundle", dropna=False):
        sub = sub.copy()
        recovered = sub["mean_hd"].notna()
        sub["reference_alignment_percentile_recovered_only"] = np.nan
        if recovered.any():
            raw_score = 1.0 - sub.loc[recovered, "mean_hd"].rank(method="average", pct=True)
            if raw_score.max() > raw_score.min():
                raw_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
            else:
                raw_score = pd.Series(1.0, index=raw_score.index)
            sub.loc[recovered, "reference_alignment_percentile_recovered_only"] = raw_score
        sub.loc[sub["attempted_not_recovered"], "reference_alignment_percentile_recovered_only"] = 0.0
        sub["recovery_weighted_reference_alignment_score"] = (
            sub["recovery_fraction"].fillna(0.0) * sub["reference_alignment_percentile_recovered_only"].fillna(0.0)
        )
        score_frames.append(sub)

    return pd.concat(score_frames, ignore_index=True) if score_frames else out


def jaccard_matrix_from_sets(names: list[str], sets: dict[str, set]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = sets.get(a, set()) | sets.get(b, set())
            inter = sets.get(a, set()) & sets.get(b, set())
            mat.loc[a, b] = np.nan if not union else len(inter) / len(union)
    return mat


def upper_triangle_bh(pmat: pd.DataFrame) -> pd.DataFrame:
    qmat = pd.DataFrame(np.nan, index=pmat.index, columns=pmat.columns, dtype=float)
    ij = []
    pvals = []
    for i, a in enumerate(pmat.index):
        for j, b in enumerate(pmat.columns):
            if j <= i:
                continue
            p = pd.to_numeric(pd.Series([pmat.loc[a, b]]), errors="coerce").iloc[0]
            if np.isfinite(p):
                ij.append((a, b))
                pvals.append(float(p))
    if pvals:
        adj = multipletests(pvals, method="fdr_bh")[1]
        for (a, b), q in zip(ij, adj):
            qmat.loc[a, b] = q
            qmat.loc[b, a] = q
    return qmat


def pairwise_spearman_with_pvalues(
    sim_input: pd.DataFrame,
    score_col: str,
    min_common: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = sim_input.pivot(index="parameter_key", columns="bundle", values=score_col)
    names = sorted(sim_input["bundle"].dropna().astype(str).unique())
    corr = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    pmat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    n_common = pd.DataFrame(0, index=names, columns=names, dtype=int)

    for a in names:
        for b in names:
            xa = wide[a] if a in wide.columns else pd.Series(dtype=float)
            xb = wide[b] if b in wide.columns else pd.Series(dtype=float)
            mask = xa.notna() & xb.notna()
            n = int(mask.sum())
            n_common.loc[a, b] = n
            if a == b:
                corr.loc[a, b] = 1.0
                continue
            if n < min_common:
                continue
            r, p = spearmanr(xa.loc[mask], xb.loc[mask])
            corr.loc[a, b] = float(r) if np.isfinite(r) else np.nan
            pmat.loc[a, b] = float(p) if np.isfinite(p) else np.nan

    qmat = upper_triangle_bh(pmat)
    return corr, pmat, qmat, n_common


def pairwise_jaccard_with_hypergeom(
    names: list[str],
    sets: dict[str, set],
    universe_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jmat = jaccard_matrix_from_sets(names, sets)
    pmat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            A = sets.get(a, set())
            B = sets.get(b, set())
            if universe_size <= 0 or not A or not B:
                continue
            k = len(A & B)
            pmat.loc[a, b] = float(hypergeom.sf(k - 1, universe_size, len(A), len(B)))
    qmat = upper_triangle_bh(pmat)
    return jmat, pmat, qmat


def compute_tract_similarity_matrices(
    sim_input: pd.DataFrame,
    best_fraction: float,
    min_common: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_col = "recovery_weighted_reference_alignment_score"
    full_corr, full_p, full_q, n_common = pairwise_spearman_with_pvalues(sim_input, score_col, min_common=min_common)

    names = sorted(sim_input["bundle"].dropna().astype(str).unique())
    best_sets = {}
    worst_sets = {}
    set_rows = []
    for bundle, sub in sim_input.groupby("bundle", dropna=False):
        b = str(bundle)
        usable = sub.loc[sub[score_col].notna()].copy()
        if usable.empty:
            best_sets[b] = set()
            worst_sets[b] = set()
            continue
        n = max(1, int(math.ceil(len(usable) * best_fraction)))
        best = usable.sort_values([score_col, "mean_hd"], ascending=[False, True], na_position="last").head(n)
        worst = usable.sort_values([score_col, "mean_hd"], ascending=[True, False], na_position="last").head(n)
        best_sets[b] = set(best["parameter_key"].astype(str))
        worst_sets[b] = set(worst["parameter_key"].astype(str))
        set_rows.append({
            "bundle": b,
            "n_attempted_combinations": int(len(usable)),
            "n_best_set": int(len(best)),
            "n_worst_set": int(len(worst)),
            "best_fraction": best_fraction,
            "n_recovered_combinations": int(sub["mean_hd"].notna().sum()),
            "n_attempted_nonrecovered_combinations": int(sub["attempted_not_recovered"].sum()),
            "median_recovery_fraction": float(sub["recovery_fraction"].median(skipna=True)),
            "median_recovery_weighted_reference_alignment_score": float(sub[score_col].median(skipna=True)),
        })

    universe_size = int(sim_input["parameter_key"].astype(str).nunique())
    best_jaccard, best_p, best_q = pairwise_jaccard_with_hypergeom(names, best_sets, universe_size=universe_size)
    worst_jaccard, worst_p, worst_q = pairwise_jaccard_with_hypergeom(names, worst_sets, universe_size=universe_size)
    set_summary = pd.DataFrame(set_rows)
    return full_corr, full_p, full_q, best_jaccard, best_p, best_q, worst_jaccard, worst_p, worst_q, n_common, set_summary


def significance_star(p: float, q: float) -> str:
    p = float(p) if pd.notna(p) else np.nan
    q = float(q) if pd.notna(q) else np.nan
    if np.isfinite(q) and q < 0.05:
        return "**"
    if np.isfinite(p) and p < 0.05:
        return "*"
    return ""


def annotation_matrix(values: pd.DataFrame, pmat: pd.DataFrame | None = None, qmat: pd.DataFrame | None = None) -> pd.DataFrame:
    annot = pd.DataFrame("", index=values.index, columns=values.columns, dtype=object)
    for a in values.index:
        for b in values.columns:
            val = pd.to_numeric(pd.Series([values.loc[a, b]]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                annot.loc[a, b] = ""
                continue
            p = np.nan if pmat is None else pd.to_numeric(pd.Series([pmat.loc[a, b]]), errors="coerce").iloc[0]
            q = np.nan if qmat is None else pd.to_numeric(pd.Series([qmat.loc[a, b]]), errors="coerce").iloc[0]
            star = significance_star(p, q)
            annot.loc[a, b] = f"{val:.2f}{star}"
    return annot


def plot_tract_similarity_figure(
    full_corr: pd.DataFrame,
    full_p: pd.DataFrame,
    full_q: pd.DataFrame,
    best_jaccard: pd.DataFrame,
    best_p: pd.DataFrame,
    best_q: pd.DataFrame,
    worst_jaccard: pd.DataFrame,
    worst_p: pd.DataFrame,
    worst_q: pd.DataFrame,
    out_path: Path,
) -> None:
    matrices = [
        (
            "Full landscape similarity\nSpearman ρ of reference-alignment score",
            full_corr,
            full_p,
            full_q,
            -1.0,
            1.0,
            "vlag",
        ),
        (
            "High reference-alignment overlap\nJaccard of highest reference-alignment score",
            best_jaccard,
            best_p,
            best_q,
            0.0,
            1.0,
            "viridis",
        ),
        (
            "Worst-space overlap\nJaccard of lowest reference-alignment score",
            worst_jaccard,
            worst_p,
            worst_q,
            0.0,
            1.0,
            "viridis",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), squeeze=False)
    for ax, (title, mat, pmat, qmat, vmin, vmax, cmap) in zip(axes.flatten(), matrices):
        plot_mat = mat.copy()
        plot_mat.index = [display_tract_name(x) for x in plot_mat.index]
        plot_mat.columns = [display_tract_name(x) for x in plot_mat.columns]

        annot = annotation_matrix(mat, pmat, qmat)
        annot.index = plot_mat.index
        annot.columns = plot_mat.columns

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="3%", pad=0.06)
        sns.heatmap(
            plot_mat,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            cbar_ax=cbar_ax,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 11},
        )
        ax.set_title(title, fontsize=15, pad=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=12)
        ax.tick_params(axis="y", labelrotation=0, labelsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.14, wspace=0.38)
    fig.text(
        0.5,
        0.045,
        "* nominal p < 0.05; ** FDR q < 0.05",
        ha="center",
        va="center",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def tract_similarity_plot_summary_long(
    full_corr: pd.DataFrame,
    full_p: pd.DataFrame,
    full_q: pd.DataFrame,
    best_jaccard: pd.DataFrame,
    best_p: pd.DataFrame,
    best_q: pd.DataFrame,
    worst_jaccard: pd.DataFrame,
    worst_p: pd.DataFrame,
    worst_q: pd.DataFrame,
    n_common: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        ("full_landscape_spearman", "Full landscape similarity", full_corr, full_p, full_q),
        ("high_reference_alignment_jaccard", "High reference-alignment overlap", best_jaccard, best_p, best_q),
        ("worst_space_jaccard", "Worst-space overlap", worst_jaccard, worst_p, worst_q),
    ]
    rows = []
    for matrix_id, matrix_label, values, pvals, qvals in specs:
        for row_name in values.index:
            for col_name in values.columns:
                value = pd.to_numeric(pd.Series([values.loc[row_name, col_name]]), errors="coerce").iloc[0]
                p = pd.to_numeric(pd.Series([pvals.loc[row_name, col_name]]), errors="coerce").iloc[0] if row_name in pvals.index and col_name in pvals.columns else np.nan
                q = pd.to_numeric(pd.Series([qvals.loc[row_name, col_name]]), errors="coerce").iloc[0] if row_name in qvals.index and col_name in qvals.columns else np.nan
                n = pd.to_numeric(pd.Series([n_common.loc[row_name, col_name]]), errors="coerce").iloc[0] if row_name in n_common.index and col_name in n_common.columns else np.nan
                rows.append({
                    "matrix_id": matrix_id,
                    "matrix_label": matrix_label,
                    "row_tract": str(row_name),
                    "column_tract": str(col_name),
                    "row_tract_display": display_tract_name(row_name),
                    "column_tract_display": display_tract_name(col_name),
                    "value": value,
                    "p_value": p,
                    "fdr_q_value": q,
                    "stars": significance_star(p, q),
                    "n_common_combinations": n,
                })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------
# Per-tract SVM boundary summaries for high positive information shift
# ---------------------------------------------------------------------


def parse_svm_gamma_grid(values: Sequence[str]) -> list[object]:
    out: list[object] = []
    for value in values:
        s = str(value).strip()
        if s in {"scale", "auto"}:
            out.append(s)
        else:
            out.append(float(s))
    return out


def svm_parameter_feature_matrix(df: pd.DataFrame, param_cols: Sequence[str], level_maps: dict[str, list[float]]) -> np.ndarray:
    x = np.zeros((len(df), len(param_cols)), dtype=float)
    for j, c in enumerate(param_cols):
        levels = [float(v) for v in level_maps[c]]
        idx_map = {float(v): i for i, v in enumerate(levels)}
        denom = max(len(levels) - 1, 1)
        x[:, j] = [idx_map[float(v)] / denom for v in pd.to_numeric(df[c], errors="coerce")]
    return x


def svm_information_threshold(combo_df: pd.DataFrame, metric_col: str, quantile: float) -> float:
    vals = pd.to_numeric(combo_df.loc[combo_df["is_recovered_combination"].astype(bool), metric_col], errors="coerce")
    vals = vals.loc[vals > 0].dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(float(quantile)))


def fit_svm_information_boundary(
    combo_df: pd.DataFrame,
    param_cols: Sequence[str],
    level_maps: dict[str, list[float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metric_col = MEASURE_SPECS["information_shift"].column
    out = combo_df.copy()
    threshold = svm_information_threshold(out, metric_col, args.svm_positive_quantile)
    metric = pd.to_numeric(out[metric_col], errors="coerce")
    recovered = out["is_recovered_combination"].astype(bool)
    y = (recovered & metric.notna() & metric.ge(threshold)).astype(int)
    out["svm_information_target"] = y
    out["svm_information_threshold"] = threshold

    qc: dict[str, object] = {
        "information_shift_threshold": threshold,
        "svm_positive_quantile": float(args.svm_positive_quantile),
        "svm_region_decision_quantile": float(args.svm_region_decision_quantile),
        "svm_refit_metric": str(args.svm_refit_metric),
        "svm_class_weight": str(args.svm_class_weight),
        "n_combinations": int(len(out)),
        "n_recovered_combinations": int(recovered.sum()),
        "n_nonrecovered_combinations": int((~recovered).sum()),
        "n_target_high_positive_information_shift": int(y.sum()),
        "target_fraction": float(y.mean()) if len(y) else np.nan,
        "status": "ok",
    }

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if not np.isfinite(threshold):
        qc["status"] = "skipped_no_positive_information_shift"
    elif n_pos < int(args.svm_min_positive_count):
        qc["status"] = "skipped_too_few_target_combinations"
    elif n_neg < int(args.svm_min_negative_count):
        qc["status"] = "skipped_too_few_non_target_combinations"

    out["svm_decision_score"] = np.nan
    out["svm_raw_boundary_positive"] = False
    out["svm_predicted_high_information_region"] = False
    out["svm_region_decision_threshold"] = np.nan
    out["svm_decision_percentile"] = np.nan

    if qc["status"] != "ok":
        return out, qc

    x = svm_parameter_feature_matrix(out, param_cols, level_maps)
    y_arr = y.to_numpy(dtype=int)
    cv_folds = min(int(args.svm_cv_folds), n_pos, n_neg)
    if cv_folds < 2:
        qc["status"] = "skipped_insufficient_cv_folds"
        return out, qc

    cv_repeats = max(1, int(args.svm_cv_repeats))
    if cv_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=args.random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    class_weight = None if str(args.svm_class_weight).lower() == "none" else "balanced"
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight=class_weight,
                    cache_size=float(args.svm_cache_size_mb),
                    tol=float(args.svm_tol),
                    max_iter=int(args.svm_max_iter),
                ),
            ),
        ]
    )
    param_grid = {
        "svm__C": [float(v) for v in args.svm_c_grid],
        "svm__gamma": parse_svm_gamma_grid(args.svm_gamma_grid),
    }
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=str(args.svm_refit_metric),
        cv=cv,
        n_jobs=int(args.svm_n_jobs),
        error_score=np.nan,
        return_train_score=True,
    )
    grid.fit(x, y_arr)
    model = grid.best_estimator_
    score = np.asarray(model.decision_function(x), dtype=float)
    raw_pred = score >= 0.0

    target_scores = score[y_arr == 1]
    target_scores = target_scores[np.isfinite(target_scores)]
    if len(target_scores):
        region_threshold = float(np.quantile(target_scores, float(args.svm_region_decision_quantile)))
    else:
        region_threshold = 0.0
    pred = score >= region_threshold

    out["svm_decision_score"] = score
    out["svm_raw_boundary_positive"] = raw_pred
    out["svm_predicted_high_information_region"] = pred
    out["svm_region_decision_threshold"] = region_threshold
    out["svm_decision_percentile"] = pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)

    best_idx = int(grid.best_index_)
    cv_bal = grid.cv_results_.get("mean_test_balanced_accuracy", [np.nan])[best_idx]
    cv_auc = grid.cv_results_.get("mean_test_roc_auc", [np.nan])[best_idx]
    cv_ap = grid.cv_results_.get("mean_test_average_precision", [np.nan])[best_idx]
    train_bal = grid.cv_results_.get("mean_train_balanced_accuracy", [np.nan])[best_idx]
    train_auc = grid.cv_results_.get("mean_train_roc_auc", [np.nan])[best_idx]
    train_ap = grid.cv_results_.get("mean_train_average_precision", [np.nan])[best_idx]

    qc.update(
        {
            "best_C": grid.best_params_.get("svm__C", np.nan),
            "best_gamma": grid.best_params_.get("svm__gamma", np.nan),
            "cv_folds": int(cv_folds),
            "cv_repeats": int(cv_repeats),
            "n_cv_splits_total": int(cv_folds * cv_repeats),
            "n_grid_candidates": int(len(param_grid["svm__C"]) * len(param_grid["svm__gamma"])),
            "cv_balanced_accuracy": float(cv_bal) if np.isfinite(cv_bal) else np.nan,
            "cv_roc_auc": float(cv_auc) if np.isfinite(cv_auc) else np.nan,
            "cv_average_precision": float(cv_ap) if np.isfinite(cv_ap) else np.nan,
            "cv_train_balanced_accuracy": float(train_bal) if np.isfinite(train_bal) else np.nan,
            "cv_train_roc_auc": float(train_auc) if np.isfinite(train_auc) else np.nan,
            "cv_train_average_precision": float(train_ap) if np.isfinite(train_ap) else np.nan,
            "svm_region_decision_threshold": float(region_threshold),
            "n_svm_raw_boundary_positive": int(raw_pred.sum()),
            "svm_raw_boundary_positive_fraction": float(raw_pred.mean()) if len(raw_pred) else np.nan,
            "n_svm_predicted_high_information_region": int(pred.sum()),
            "svm_predicted_fraction": float(pred.mean()) if len(pred) else np.nan,
            "training_balanced_accuracy_raw_boundary": float(balanced_accuracy_score(y_arr, raw_pred)),
            "training_balanced_accuracy_restrictive_region": float(balanced_accuracy_score(y_arr, pred)),
        }
    )
    try:
        qc["training_roc_auc"] = float(roc_auc_score(y_arr, score))
    except Exception:
        qc["training_roc_auc"] = np.nan
    try:
        qc["training_average_precision"] = float(average_precision_score(y_arr, score))
    except Exception:
        qc["training_average_precision"] = np.nan
    return out, qc


def compute_svm_information_enrichment(
    score_df: pd.DataFrame,
    param_cols: Sequence[str],
    level_maps: dict[str, list[float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    pred = score_df["svm_predicted_high_information_region"].astype(bool)
    rows = []
    pseudo = float(args.svm_enrichment_pseudocount)
    n_all = len(score_df)
    n_region = int(pred.sum())
    for c in param_cols:
        parameter = c.replace("param_", "")
        levels = [float(v) for v in level_maps[c]]
        for level_index, level in enumerate(levels):
            all_count = int(np.isclose(pd.to_numeric(score_df[c], errors="coerce"), level).sum())
            region_count = int(np.isclose(pd.to_numeric(score_df.loc[pred, c], errors="coerce"), level).sum()) if n_region else 0
            p_all = (all_count + pseudo) / (n_all + pseudo * len(levels)) if n_all else np.nan
            p_region = (region_count + pseudo) / (n_region + pseudo * len(levels)) if n_region else np.nan
            enrichment = math.log(p_region / p_all, 2) if p_all > 0 and p_region > 0 else np.nan
            rows.append(
                {
                    "parameter": parameter,
                    "parameter_column": c,
                    "level_index": int(level_index),
                    "level_value": level,
                    "n_all": all_count,
                    "n_svm_region": region_count,
                    "p_all": p_all,
                    "p_svm_region": p_region,
                    "log2_enrichment": enrichment,
                }
            )
    return pd.DataFrame(rows)










def _plot_svm_information_boundary_panel(
    ax: plt.Axes,
    score_df: pd.DataFrame,
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    axis_vecs_2d: np.ndarray,
    map_ranges: dict[str, tuple[float, float]],
    param_cols: Sequence[str],
    args: argparse.Namespace,
    *,
    norm: plt.Normalize,
    cmap,
) -> None:
    merged = score_df.copy()
    all_keys = set(merged["parameter_key"].astype(str).tolist())
    for (x0, y0), (x1, y1) in edge_xy_for_parameter_space_map(map_embedding_df, edges, all_keys):
        ax.plot([x0, x1], [y0, y1], color=(120/255, 120/255, 120/255, 0.30), lw=args.lattice_line_width, zorder=1)

    recovered = merged["is_recovered_combination"].astype(bool)
    nonrecovered = ~recovered
    scores = pd.to_numeric(merged["svm_decision_score_z"], errors="coerce")

    if nonrecovered.any():
        ax.scatter(
            merged.loc[nonrecovered, "map_x"],
            merged.loc[nonrecovered, "map_y"],
            s=float(args.parameter_space_2d_unavailable_node_size),
            c=[mpl_color_from_rgb_string(args.nonrecovery_gray)],
            alpha=0.85,
            edgecolors=(60/255, 60/255, 60/255, 0.45),
            linewidths=0.30,
            zorder=2,
        )

    rec_scored = recovered & scores.notna()
    if rec_scored.any():
        ax.scatter(
            merged.loc[rec_scored, "map_x"],
            merged.loc[rec_scored, "map_y"],
            s=np.clip(marker_sizes(merged.loc[rec_scored].copy(), args) * float(args.parameter_space_2d_node_size_scale), 8.0, float(args.parameter_space_2d_node_size_max)),
            c=scores.loc[rec_scored],
            cmap=cmap,
            norm=norm,
            alpha=0.95,
            edgecolors=(40/255, 40/255, 40/255, 0.45),
            linewidths=0.30,
            zorder=3,
        )

    svm_region = merged["svm_predicted_high_information_region"].astype(bool)
    if svm_region.any():
        ax.scatter(
            merged.loc[svm_region, "map_x"],
            merged.loc[svm_region, "map_y"],
            s=float(args.parameter_space_2d_node_size_max) * 1.20,
            facecolors="none",
            edgecolors="black",
            linewidths=0.65,
            zorder=4,
        )

    target = merged["svm_information_target"].astype(bool)
    if target.any():
        ax.scatter(
            merged.loc[target, "map_x"],
            merged.loc[target, "map_y"],
            s=float(args.parameter_space_2d_node_size_max) * 0.55,
            facecolors="none",
            edgecolors="white",
            linewidths=0.80,
            zorder=5,
        )

    ax.set_xlim(*map_ranges["map_x"])
    ax.set_ylim(*map_ranges["map_y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    add_parameter_axes_to_map(ax, map_embedding_df, axis_vecs_2d, param_cols)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)


def plot_svm_information_boundary_grid_figure(
    score_by_bundle: dict[str, pd.DataFrame],
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    axis_vecs_2d: np.ndarray,
    map_ranges: dict[str, tuple[float, float]],
    param_cols: Sequence[str],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(float(args.parameter_space_2d_figure_width), float(args.parameter_space_2d_figure_height)),
        squeeze=False,
    )

    score_by_bundle = add_within_tract_standardized_svm_scores(score_by_bundle)
    all_scores = []
    for combo_df in score_by_bundle.values():
        rec_scored = combo_df["is_recovered_combination"].astype(bool) & pd.to_numeric(combo_df["svm_decision_score_z"], errors="coerce").notna()
        if rec_scored.any():
            all_scores.append(pd.to_numeric(combo_df.loc[rec_scored, "svm_decision_score_z"], errors="coerce"))
    all_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.Series(dtype=float)
    norm = symmetric_percentile_color_norm(all_scores, percentile=98.0)
    vmax = max(abs(float(norm.vmin)), abs(float(norm.vmax)), EPS)
    cmap = plt.get_cmap(mpl_cmap_name_from_scale(args.signed_information_color_scale))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for bundle_name, combo_df in score_by_bundle.items():
        pos = parameter_space_2d_grid_position(bundle_name)
        if pos is None:
            continue
        r, c, _, _ = pos
        _plot_svm_information_boundary_panel(
            axes[r, c],
            combo_df,
            map_embedding_df,
            edges,
            axis_vecs_2d,
            map_ranges,
            param_cols,
            args,
            norm=norm,
            cmap=cmap,
        )

    for j, fam in enumerate(["AF", "FX", "CST"]):
        axes[0, j].text(0.5, 1.095, fam, transform=axes[0, j].transAxes, ha="center", va="bottom", fontsize=23)
        axes[0, j].plot([0.07, 0.93], [1.035, 1.035], transform=axes[0, j].transAxes, color="black", lw=1.2, clip_on=False)

    for r, label in [(0, "Left"), (1, "Right")]:
        axes[r, 0].text(-0.125, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=23)
        axes[r, 0].plot([-0.05, -0.05], [0.07, 0.93], transform=axes[r, 0].transAxes, color="black", lw=1.2, clip_on=False)


    cax = fig.add_axes([0.915, 0.10, 0.012, 0.80])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
    cbar.set_label(
        "Within-tract standardized RBF-SVM decision score",
        fontsize=STATIC_PLOT_FONT_SIZE,
        rotation=-90,
        labelpad=34,
    )

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=mpl_color_from_rgb_string(args.nonrecovery_gray), markeredgecolor=(60/255, 60/255, 60/255, 0.75), markersize=6, label='Non-recovered / unavailable'),
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=cmap(norm(0.65 * vmax)), markeredgecolor=(40/255, 40/255, 40/255, 0.75), markersize=7, label='Recovered combinations'),
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='black', markersize=9, label='SVM high-information region'),
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='white', markersize=6, label='Upper positive information-shift target'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.48, 0.008), ncol=4, frameon=False, fontsize=STATIC_PLOT_FONT_SIZE)
    fig.subplots_adjust(left=0.055, right=0.900, top=0.93, bottom=0.08, wspace=0.015, hspace=-0.03)
    fig.savefig(out_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches='tight')
    plt.close(fig)


def plot_svm_information_enrichment_grid_heatmap(
    enrichment_df: pd.DataFrame,
    param_cols: Sequence[str],
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    if enrichment_df.empty:
        return

    tract_order = [
        "left AF",
        "right AF",
        "left FX",
        "right FX",
        "left CST",
        "right CST",
    ]
    enrichment = enrichment_df.copy()
    enrichment["display_tract"] = enrichment["display_tract"].astype(str)
    enrichment["tract_order"] = enrichment["display_tract"].map({t: i for i, t in enumerate(tract_order)})
    enrichment = enrichment.sort_values(["tract_order", "parameter", "level_index"]).copy()

    values = pd.to_numeric(enrichment["log2_enrichment"], errors='coerce')
    vmax = float(np.nanmax(np.abs(values.to_numpy(dtype=float)))) if np.isfinite(values).any() else 1.0
    if not np.isfinite(vmax) or vmax <= EPS:
        vmax = 1.0

    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(3.0 * n_params + 3.4, 5.8), squeeze=False)
    axes = axes[0]
    cmap = 'RdBu_r'
    last_mesh = None
    param_display = [c.replace('param_', '') for c in param_cols]

    for idx, (ax, c, title) in enumerate(zip(axes, param_cols, param_display)):
        sub = enrichment.loc[enrichment['parameter_column'].eq(c)].copy()
        mat = sub.pivot(index='display_tract', columns='level_value', values='log2_enrichment')
        ordered_rows = [t for t in tract_order if t in mat.index]
        mat = mat.reindex(ordered_rows)
        if not mat.empty:
            ordered_cols = sub.sort_values('level_index')['level_value'].drop_duplicates().tolist()
            mat = mat.reindex(columns=ordered_cols)
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': 12.0},
            cbar=False,
            linewidths=0.4,
            linecolor='white',
        )
        last_mesh = ax.collections[0] if ax.collections else last_mesh
        ax.set_title(title, fontsize=STATIC_PLOT_FONT_SIZE, pad=10)
        ax.set_xlabel('Parameter value', fontsize=STATIC_PLOT_FONT_SIZE)
        ax.tick_params(axis='x', labelrotation=45, labelsize=STATIC_PLOT_FONT_SIZE)
        if idx == 0:
            ax.set_ylabel('Tract', fontsize=STATIC_PLOT_FONT_SIZE)
            ax.tick_params(axis='y', labelrotation=0, labelsize=STATIC_PLOT_FONT_SIZE)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False)

    cax = fig.add_axes([0.915, 0.13, 0.012, 0.72])
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, cax=cax)
        cbar.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
        cbar.set_label('log2 enrichment', fontsize=STATIC_PLOT_FONT_SIZE, rotation=-90, labelpad=34)
    fig.subplots_adjust(left=0.12, right=0.900, top=0.87, bottom=0.17, wspace=0.06)
    fig.savefig(out_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches='tight')
    plt.close(fig)
def format_param_level_value(value: object) -> str:
    try:
        fv = float(value)
        if np.isfinite(fv):
            if float(fv).is_integer():
                return str(int(fv))
            return f"{fv:g}"
    except Exception:
        pass
    return str(value)


def _series_matches_level(series: pd.Series, level: object) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        level_f = float(level)
        if np.isfinite(level_f):
            return numeric.notna() & np.isclose(numeric.to_numpy(dtype=float), level_f)
    except Exception:
        pass
    return series.astype(str).eq(str(level))


def logit_clip(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return math.log(p / (1.0 - p))


def compute_svm_higher_order_dependence(
    score_df: pd.DataFrame,
    param_cols: Sequence[str],
    bundle_name: str,
    *,
    orders: Sequence[int] = (1, 2, 3, 4, 5),
    min_support_count: int = 3,
    min_region_count: int = 5,
    precision_threshold: float = 0.80,
    coverage_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Replace the previous higher-order dependence residual analysis with an explicit
    minimum-order dependent-range analysis.

    For each tract and each rule order k, search over all contiguous parameter ranges
    on exactly k parameters and display the best rule at every order. Prefer a rule
    meeting the requested precision, coverage, and high-information-count thresholds
    when one exists at that order; otherwise display the best available rule. The
    minimum order meeting the thresholds is identified separately for the panel outline.
    """
    if score_df.empty:
        return pd.DataFrame()

    out = score_df.copy()
    out["svm_predicted_high_information_region"] = out["svm_predicted_high_information_region"].astype(bool)
    n_region_total = int(out["svm_predicted_high_information_region"].sum())
    if n_region_total == 0:
        return pd.DataFrame()

    # Ordered parameter levels present for this tract.
    level_maps: dict[str, list[float]] = {}
    level_to_index: dict[str, dict[float, int]] = {}
    for c in param_cols:
        vals = np.sort(pd.to_numeric(out[c], errors="coerce").dropna().unique())
        level_maps[c] = [float(v) for v in vals]
        level_to_index[c] = {float(v): i for i, v in enumerate(level_maps[c])}
    dims = [len(level_maps[c]) for c in param_cols]
    if not dims or any(d <= 0 for d in dims):
        return pd.DataFrame()

    # Populate total and positive grids across the observed parameter-space lattice.
    total_grid = np.zeros(dims, dtype=np.int32)
    positive_grid = np.zeros(dims, dtype=np.int32)
    for _, row in out.iterrows():
        try:
            idx = tuple(level_to_index[c][float(row[c])] for c in param_cols)
        except Exception:
            continue
        total_grid[idx] += 1
        if bool(row["svm_predicted_high_information_region"]):
            positive_grid[idx] += 1

    total_prefix = total_grid.copy()
    positive_prefix = positive_grid.copy()
    for axis in range(len(dims)):
        total_prefix = total_prefix.cumsum(axis=axis)
        positive_prefix = positive_prefix.cumsum(axis=axis)

    def query_prefix_sum(prefix: np.ndarray, starts: list[int], ends: list[int]) -> int:
        total = 0
        n_dim = len(starts)
        for bits in itertools.product([0, 1], repeat=n_dim):
            corner = []
            valid = True
            sign = 1
            for ax, bit in enumerate(bits):
                if bit == 0:
                    corner.append(ends[ax])
                else:
                    if starts[ax] == 0:
                        valid = False
                        break
                    corner.append(starts[ax] - 1)
                    sign *= -1
            if valid:
                total += sign * int(prefix[tuple(corner)])
        return int(total)

    def range_label(param: str, start_idx: int, end_idx: int) -> str:
        lo = format_param_level_value(level_maps[param][start_idx])
        hi = format_param_level_value(level_maps[param][end_idx])
        pname = param.replace("param_", "")
        if start_idx == end_idx:
            return f"{pname}={lo}"
        return f"{pname} in [{lo}, {hi}]"

    range_options: dict[str, list[tuple[int, int, str]]] = {}
    for c in param_cols:
        opts = []
        n_levels = len(level_maps[c])
        for i in range(n_levels):
            for j in range(i, n_levels):
                opts.append((i, j, range_label(c, i, j)))
        range_options[c] = opts

    order_rows = []
    index_lookup = {c: i for i, c in enumerate(param_cols)}

    for order in orders:
        if order < 1 or order > len(param_cols):
            continue
        best_row = None
        for subset in itertools.combinations(param_cols, order):
            option_lists = [range_options[c] for c in subset]
            for range_combo in itertools.product(*option_lists):
                starts = [0] * len(param_cols)
                ends = [dims[i] - 1 for i in range(len(param_cols))]
                label_parts = []
                for c, (start_idx, end_idx, label) in zip(subset, range_combo):
                    pos = index_lookup[c]
                    starts[pos] = int(start_idx)
                    ends[pos] = int(end_idx)
                    label_parts.append(label)

                support_count = query_prefix_sum(total_prefix, starts, ends)
                if support_count < int(min_support_count):
                    continue
                region_count = query_prefix_sum(positive_prefix, starts, ends)
                if region_count <= 0:
                    continue

                precision = region_count / support_count
                coverage = region_count / n_region_total if n_region_total else np.nan
                support_fraction = support_count / int(total_grid.sum()) if int(total_grid.sum()) > 0 else np.nan
                qualifies = (
                    (precision >= float(precision_threshold))
                    and (coverage >= float(coverage_threshold))
                    and (region_count >= int(min_region_count))
                )

                row = {
                    "bundle": bundle_name,
                    "display_tract": display_tract_name(bundle_name),
                    "order": int(order),
                    "parameter_subset": ";".join(subset),
                    "range_label": " & ".join(label_parts),
                    "support_count": int(support_count),
                    "support_fraction": float(support_fraction),
                    "svm_region_count": int(region_count),
                    "precision_for_svm_region": float(precision),
                    "coverage_of_svm_region": float(coverage),
                    "qualifies_threshold": bool(qualifies),
                }

                if best_row is None:
                    best_row = row
                else:
                    best_key = (
                        int(best_row["qualifies_threshold"]),
                        float(best_row["precision_for_svm_region"]),
                        float(best_row["coverage_of_svm_region"]),
                        int(best_row["svm_region_count"]),
                        int(best_row["support_count"]),
                    )
                    row_key = (
                        int(row["qualifies_threshold"]),
                        float(row["precision_for_svm_region"]),
                        float(row["coverage_of_svm_region"]),
                        int(row["svm_region_count"]),
                        int(row["support_count"]),
                    )
                    if row_key > best_key:
                        best_row = row
        if best_row is not None:
            order_rows.append(best_row)

    if not order_rows:
        return pd.DataFrame()

    dep = pd.DataFrame(order_rows).sort_values("order").reset_index(drop=True)
    qualifying_orders = dep.loc[dep["qualifies_threshold"].astype(bool), "order"].tolist()
    first_q = min(qualifying_orders) if qualifying_orders else np.nan
    dep["first_qualifying_order"] = first_q
    dep["is_first_qualifying_order"] = dep["order"].eq(first_q) if np.isfinite(first_q) else False
    return dep
def wrap_dependent_range_cell_text(label: str, precision: float, coverage: float, region_n: int, width: int = 27, max_lines: int = 7) -> str:
    import textwrap as _textwrap

    clean = str(label).replace(" & ", "\n")
    wrapped_lines: list[str] = []
    for part in clean.splitlines():
        part = part.strip()
        if not part:
            continue
        wrapped_lines.extend(_textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [part])

    metric_line = f"P={precision:.2f}; C={coverage:.2f}; n={region_n}"
    if len(wrapped_lines) >= max_lines:
        wrapped_lines = wrapped_lines[: max_lines - 1]
        if wrapped_lines:
            wrapped_lines[-1] = wrapped_lines[-1].rstrip(" .;") + "…"
    wrapped_lines.append(metric_line)
    return "\n".join(wrapped_lines)


def plot_svm_higher_order_dependence_heatmaps(
    dependence_df: pd.DataFrame,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    if dependence_df.empty:
        return

    tract_order = [
        "left AF",
        "right AF",
        "left FX",
        "right FX",
        "left CST",
        "right CST",
    ]
    order_list = [1, 2, 3, 4, 5]
    dep = dependence_df.copy()
    dep["display_tract"] = dep["display_tract"].astype(str)
    dep["order"] = pd.to_numeric(dep["order"], errors="coerce").astype("Int64")

    value_mat = np.full((len(tract_order), len(order_list)), np.nan, dtype=float)
    annotation_mat = [["" for _ in order_list] for _ in tract_order]
    first_qualifying: dict[str, int] = {}

    for i, tract in enumerate(tract_order):
        sub = dep.loc[dep["display_tract"].eq(tract)].copy().sort_values("order")
        if sub["qualifies_threshold"].astype(bool).any():
            first_qualifying[tract] = int(sub.loc[sub["qualifies_threshold"].astype(bool), "order"].min())

        for j, order in enumerate(order_list):
            row = sub.loc[sub["order"].eq(order)]
            if row.empty:
                annotation_mat[i][j] = "No rule"
                continue
            row = row.iloc[0]
            precision = float(row["precision_for_svm_region"])
            coverage = float(row["coverage_of_svm_region"])
            region_n = int(row["svm_region_count"])
            value_mat[i, j] = precision
            annotation_mat[i][j] = wrap_dependent_range_cell_text(
                row["range_label"],
                precision=precision,
                coverage=coverage,
                region_n=region_n,
                width=26,
                max_lines=7,
            )

    fig, ax = plt.subplots(1, 1, figsize=(23.5, 10.2))
    cmap = plt.get_cmap("viridis")
    cmap = cmap.copy()
    cmap.set_bad(color=(0.92, 0.92, 0.92))
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    im = ax.imshow(value_mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(order_list)))
    ax.set_xticklabels([str(x) for x in order_list], fontsize=STATIC_PLOT_FONT_SIZE)
    ax.set_yticks(np.arange(len(tract_order)))
    ax.set_yticklabels(tract_order, fontsize=STATIC_PLOT_FONT_SIZE)
    ax.set_xlabel("Rule order", fontsize=STATIC_PLOT_FONT_SIZE, labelpad=14)
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=STATIC_PLOT_FONT_SIZE, pad=5)
    ax.tick_params(axis="y", labelsize=STATIC_PLOT_FONT_SIZE)
    ax.set_ylabel("Tract", fontsize=STATIC_PLOT_FONT_SIZE, labelpad=10)

    ax.set_xticks(np.arange(-0.5, len(order_list), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tract_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, tract in enumerate(tract_order):
        for j, order in enumerate(order_list):
            val = value_mat[i, j]
            text_color = "white" if np.isfinite(val) and val < 0.52 else "black"
            clip_rect = mpl.patches.Rectangle(
                (j - 0.49, i - 0.49),
                0.98,
                0.98,
                transform=ax.transData,
                facecolor="none",
                edgecolor="none",
            )
            ax.add_patch(clip_rect)
            txt = ax.text(
                j,
                i,
                annotation_mat[i][j],
                ha="center",
                va="center",
                fontsize=STATIC_PLOT_FONT_SIZE,
                color=text_color,
                linespacing=1.02,
                clip_on=True,
            )
            txt.set_clip_path(clip_rect)
            if first_qualifying.get(tract) == order:
                rect = mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=3.0,
                    zorder=6,
                )
                ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
    cbar.set_label("Precision  P(H=1 | R=1)", fontsize=STATIC_PLOT_FONT_SIZE, rotation=-90, labelpad=34)

    fig.subplots_adjust(left=0.075, right=0.900, top=0.965, bottom=0.075)
    fig.savefig(out_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches="tight")
    plt.close(fig)





def plot_svm_combined_three_panel_grid(
    score_by_bundle: dict[str, pd.DataFrame],
    enrichment_df: pd.DataFrame,
    dependence_df: pd.DataFrame,
    map_embedding_df: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
    axis_vecs_2d: np.ndarray,
    map_ranges: dict[str, tuple[float, float]],
    param_cols: Sequence[str],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    tract_order = [
        "left AF",
        "right AF",
        "left FX",
        "right FX",
        "left CST",
        "right CST",
    ]

    fig = plt.figure(figsize=(31.5, 19.0))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.55, 1.05],
        height_ratios=[0.95, 1.05],
        left=0.032,
        right=0.985,
        top=0.975,
        bottom=0.055,
        wspace=0.12,
        hspace=0.075,
    )

    cbar_width_ratio = 0.052

    map_gs = outer[0, 0].subgridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.18],
        width_ratios=[1.0, 1.0, 1.0, cbar_width_ratio],
        wspace=0.015,
        hspace=0.00,
    )
    map_axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            map_axes[r, c] = fig.add_subplot(map_gs[r, c])
    map_legend_ax = fig.add_subplot(map_gs[2, :3])
    map_legend_ax.axis("off")
    map_cax = fig.add_subplot(map_gs[:2, 3])

    score_by_bundle = add_within_tract_standardized_svm_scores(score_by_bundle)
    all_scores = []
    for combo_df in score_by_bundle.values():
        rec_scored = combo_df["is_recovered_combination"].astype(bool) & pd.to_numeric(combo_df["svm_decision_score_z"], errors="coerce").notna()
        if rec_scored.any():
            all_scores.append(pd.to_numeric(combo_df.loc[rec_scored, "svm_decision_score_z"], errors="coerce"))
    all_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.Series(dtype=float)
    norm_map = symmetric_percentile_color_norm(all_scores, percentile=98.0)
    vmax = max(abs(float(norm_map.vmin)), abs(float(norm_map.vmax)), EPS)
    cmap_map = plt.get_cmap(mpl_cmap_name_from_scale(args.signed_information_color_scale))
    sm_map = plt.cm.ScalarMappable(norm=norm_map, cmap=cmap_map)
    sm_map.set_array([])

    for bundle_name, combo_df in score_by_bundle.items():
        pos = parameter_space_2d_grid_position(bundle_name)
        if pos is None:
            continue
        r, c, _, _ = pos
        _plot_svm_information_boundary_panel(
            map_axes[r, c],
            combo_df,
            map_embedding_df,
            edges,
            axis_vecs_2d,
            map_ranges,
            param_cols,
            args,
            norm=norm_map,
            cmap=cmap_map,
        )

    for j, fam in enumerate(["AF", "FX", "CST"]):
        map_axes[0, j].text(0.5, 1.085, fam, transform=map_axes[0, j].transAxes, ha="center", va="bottom", fontsize=STATIC_PLOT_FONT_SIZE)
        map_axes[0, j].plot([0.07, 0.93], [1.03, 1.03], transform=map_axes[0, j].transAxes, color="black", lw=1.1, clip_on=False)

    for r, label in [(0, "Left"), (1, "Right")]:
        map_axes[r, 0].text(-0.095, 0.5, label, transform=map_axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=STATIC_PLOT_FONT_SIZE)
        map_axes[r, 0].plot([-0.030, -0.030], [0.07, 0.93], transform=map_axes[r, 0].transAxes, color="black", lw=1.1, clip_on=False)

    cbar_map = fig.colorbar(sm_map, cax=map_cax)
    cbar_map.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
    cbar_map.set_label(
        "Within-tract standardized RBF-SVM decision score",
        fontsize=STATIC_PLOT_FONT_SIZE,
        rotation=-90,
        labelpad=34,
    )

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=mpl_color_from_rgb_string(args.nonrecovery_gray), markeredgecolor=(60/255, 60/255, 60/255, 0.75), markersize=6, label='Non-recovered / unavailable'),
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=cmap_map(norm_map(0.65 * vmax)), markeredgecolor=(40/255, 40/255, 40/255, 0.75), markersize=7, label='Recovered combinations'),
        matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='black', markersize=9, label='SVM high-information region'),
    ]
    map_legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=STATIC_PLOT_FONT_SIZE,
        columnspacing=1.3,
        handletextpad=0.6,
    )

    enrich_gs = outer[:, 1].subgridspec(
        len(param_cols),
        2,
        width_ratios=[1.0, cbar_width_ratio],
        hspace=0.12,
        wspace=0.055,
    )
    enrich_axes = []
    enrich = enrichment_df.copy()
    enrich["display_tract"] = enrich["display_tract"].astype(str)
    enrich["tract_order"] = enrich["display_tract"].map({t: i for i, t in enumerate(tract_order)})
    enrich = enrich.sort_values(["parameter", "tract_order", "level_index"]).copy()

    values = pd.to_numeric(enrich["log2_enrichment"], errors="coerce")
    vmax_enrich = float(np.nanmax(np.abs(values.to_numpy(dtype=float)))) if np.isfinite(values).any() else 1.0
    if not np.isfinite(vmax_enrich) or vmax_enrich <= EPS:
        vmax_enrich = 1.0

    param_display = [c.replace("param_", "") for c in param_cols]
    last_mesh = None
    for idx, (param_col, param_title) in enumerate(zip(param_cols, param_display)):
        ax = fig.add_subplot(enrich_gs[idx, 0])
        enrich_axes.append(ax)
        sub = enrich.loc[enrich["parameter_column"].eq(param_col)].copy()
        mat = sub.pivot(index="level_value", columns="display_tract", values="log2_enrichment")
        ordered_cols = [t for t in tract_order if t in mat.columns]
        mat = mat.reindex(columns=ordered_cols)
        ordered_rows = sub.sort_values("level_index")["level_value"].drop_duplicates().tolist()
        mat = mat.reindex(index=ordered_rows)
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax_enrich,
            vmax=vmax_enrich,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": max(STATIC_PLOT_FONT_SIZE - 3, 8)},
            cbar=False,
            linewidths=0.4,
            linecolor="white",
        )
        last_mesh = ax.collections[0] if ax.collections else last_mesh
        ax.set_ylabel(param_title, fontsize=STATIC_PLOT_FONT_SIZE, rotation=90, labelpad=8)
        if idx == 0:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Tract", fontsize=STATIC_PLOT_FONT_SIZE, labelpad=10)
            ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=45, labelsize=STATIC_PLOT_FONT_SIZE)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.tick_params(axis="x", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelrotation=0, labelsize=STATIC_PLOT_FONT_SIZE)
        ax.tick_params(axis="both", length=0)

    enrich_cax = fig.add_subplot(enrich_gs[:, 1])
    if last_mesh is not None:
        cbar_enrich = fig.colorbar(last_mesh, cax=enrich_cax)
        cbar_enrich.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
        cbar_enrich.set_label("log2 enrichment", fontsize=STATIC_PLOT_FONT_SIZE, rotation=-90, labelpad=34)

    rule_gs = outer[1, 0].subgridspec(1, 3, width_ratios=[0.082, 1.0, cbar_width_ratio], wspace=0.025)
    rule_ax = fig.add_subplot(rule_gs[0, 1])
    rule_cax = fig.add_subplot(rule_gs[0, 2])

    dep = dependence_df.copy()
    dep["display_tract"] = dep["display_tract"].astype(str)
    dep["order"] = pd.to_numeric(dep["order"], errors="coerce").astype("Int64")
    order_list = [1, 2, 3, 4, 5]

    value_mat = np.full((len(tract_order), len(order_list)), np.nan, dtype=float)
    annotation_mat = [["" for _ in order_list] for _ in tract_order]
    first_qualifying: dict[str, int] = {}

    for i, tract in enumerate(tract_order):
        sub = dep.loc[dep["display_tract"].eq(tract)].copy().sort_values("order")
        if sub["qualifies_threshold"].astype(bool).any():
            first_qualifying[tract] = int(sub.loc[sub["qualifies_threshold"].astype(bool), "order"].min())
        for j, order in enumerate(order_list):
            row = sub.loc[sub["order"].eq(order)]
            if row.empty:
                annotation_mat[i][j] = "No rule"
                continue
            row = row.iloc[0]
            precision = float(row["precision_for_svm_region"])
            coverage = float(row["coverage_of_svm_region"])
            region_n = int(row["svm_region_count"])
            value_mat[i, j] = precision
            annotation_mat[i][j] = wrap_dependent_range_cell_text(
                row["range_label"],
                precision=precision,
                coverage=coverage,
                region_n=region_n,
                width=26,
                max_lines=7,
            )

    cmap_rule = plt.get_cmap("viridis").copy()
    cmap_rule.set_bad(color=(0.92, 0.92, 0.92))
    norm_rule = plt.Normalize(vmin=0.0, vmax=1.0)
    im = rule_ax.imshow(value_mat, aspect="auto", cmap=cmap_rule, norm=norm_rule)

    rule_ax.set_xticks(np.arange(len(order_list)))
    rule_ax.set_xticklabels([str(x) for x in order_list], fontsize=STATIC_PLOT_FONT_SIZE)
    rule_ax.set_yticks(np.arange(len(tract_order)))
    rule_ax.set_yticklabels(tract_order, fontsize=STATIC_PLOT_FONT_SIZE)
    rule_ax.set_xlabel("Rule order", fontsize=STATIC_PLOT_FONT_SIZE, labelpad=14)
    rule_ax.xaxis.set_label_position("bottom")
    rule_ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=STATIC_PLOT_FONT_SIZE, pad=5)
    rule_ax.tick_params(axis="y", labelsize=STATIC_PLOT_FONT_SIZE)
    rule_ax.set_ylabel("Tract", fontsize=STATIC_PLOT_FONT_SIZE, labelpad=10)
    rule_ax.set_xticks(np.arange(-0.5, len(order_list), 1), minor=True)
    rule_ax.set_yticks(np.arange(-0.5, len(tract_order), 1), minor=True)
    rule_ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    rule_ax.tick_params(which="minor", bottom=False, left=False)

    for i, tract in enumerate(tract_order):
        for j, order in enumerate(order_list):
            val = value_mat[i, j]
            text_color = "white" if np.isfinite(val) and val < 0.52 else "black"
            clip_rect = mpl.patches.Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, transform=rule_ax.transData, facecolor="none", edgecolor="none")
            rule_ax.add_patch(clip_rect)
            txt = rule_ax.text(
                j,
                i,
                annotation_mat[i][j],
                ha="center",
                va="center",
                fontsize=STATIC_PLOT_FONT_SIZE,
                color=text_color,
                linespacing=1.02,
                clip_on=True,
            )
            txt.set_clip_path(clip_rect)
            if first_qualifying.get(tract) == order:
                rect = mpl.patches.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=3.0, zorder=6)
                rule_ax.add_patch(rect)

    for spine in rule_ax.spines.values():
        spine.set_visible(False)
    cbar_rule = fig.colorbar(im, cax=rule_cax)
    cbar_rule.ax.tick_params(labelsize=STATIC_PLOT_FONT_SIZE)
    cbar_rule.set_label("Precision  P(H=1 | R=1)", fontsize=STATIC_PLOT_FONT_SIZE, rotation=-90, labelpad=34)

    fig.canvas.draw()
    map_pos = map_axes[0, 0].get_position()
    rule_pos = rule_ax.get_position()
    map_cbar_pos = map_cax.get_position()
    rule_cbar_pos = rule_cax.get_position()
    enrich_cbar_pos = enrich_cax.get_position()
    fixed_cbar_w = min(map_cbar_pos.width, rule_cbar_pos.width, enrich_cbar_pos.width)
    map_grid_right = max(ax.get_position().x1 for ax in map_axes.ravel())
    rule_grid_right = rule_ax.get_position().x1
    shared_ac_cbar_x = max(map_grid_right, rule_grid_right) + 0.004
    map_cax.set_position([shared_ac_cbar_x, map_cbar_pos.y0, fixed_cbar_w, map_cbar_pos.height])
    rule_cax.set_position([shared_ac_cbar_x, rule_cbar_pos.y0, fixed_cbar_w, rule_cbar_pos.height])
    enrich_cax.set_position([enrich_cbar_pos.x0, enrich_cbar_pos.y0, fixed_cbar_w, enrich_cbar_pos.height])

    label_dx = 0.018
    enrich_pos = enrich_axes[0].get_position()
    rule_pos = rule_ax.get_position()
    fig.text(max(map_pos.x0 - label_dx, 0.002), map_pos.y1 + 0.012, "A", ha="left", va="bottom", fontsize=STATIC_PLOT_FONT_SIZE + 4, fontweight="bold")
    fig.text(max(enrich_pos.x0 - label_dx, 0.002), enrich_pos.y1 + 0.018, "B", ha="left", va="bottom", fontsize=STATIC_PLOT_FONT_SIZE + 4, fontweight="bold")
    fig.text(max(rule_pos.x0 - label_dx, 0.002), rule_pos.y1 + 0.012, "C", ha="left", va="bottom", fontsize=STATIC_PLOT_FONT_SIZE + 4, fontweight="bold")

    fig.savefig(out_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches="tight")
    plt.close(fig)




def compute_recovery_new_default_summary(raw_df: pd.DataFrame, param_cols: Sequence[str]) -> pd.DataFrame:
    required = {"bundle", "subject", "session"}
    if raw_df.empty or not required.issubset(raw_df.columns):
        return pd.DataFrame()

    df = raw_df.copy()
    if "display_tract" not in df.columns:
        df["display_tract"] = df["bundle"].astype(str).map(display_tract_name)

    if "is_recovered_row" in df.columns:
        recovered = df["is_recovered_row"].astype(bool)
    else:
        recovered = analytically_recovered_mask(
            df,
            outcome_col="hausdorff_mm_vs_reference" if "hausdorff_mm_vs_reference" in df.columns else DEFAULT_OUTCOME_COL,
            require_finite_outcome=False,
        )
    df["__recovered_for_summary"] = recovered.astype(bool)

    stage_col = "stage" if "stage" in df.columns else None
    run_index_col = None
    for candidate in ("run_index", "attempt_index", "attempt_number", "combo_rank", "combination_index"):
        if candidate in df.columns:
            run_index_col = candidate
            break

    group_cols = ["bundle", "display_tract", "subject", "session"]
    per_subject_session_rows = []

    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        bundle, display_tract, subject, session = keys
        g = g.copy()
        n_rows = int(len(g))
        n_recovered = int(g["__recovered_for_summary"].sum())
        recovery_fraction = float(n_recovered / n_rows) if n_rows > 0 else np.nan

        if stage_col is not None:
            strict = g.loc[g[stage_col].astype(str).eq("strict_search")].copy()
        else:
            strict = g.copy()

        strict_recovered = strict.loc[strict["__recovered_for_summary"]].copy()
        new_default = pd.DataFrame()
        spaces_attempted = np.nan

        if not strict_recovered.empty:
            if run_index_col is not None:
                strict_recovered["__attempt_sort"] = pd.to_numeric(strict_recovered[run_index_col], errors="coerce")
            else:
                strict_recovered["__attempt_sort"] = np.nan

            if strict_recovered["__attempt_sort"].notna().any():
                new_default = strict_recovered.sort_values("__attempt_sort", kind="mergesort").head(1)
                spaces_attempted = float(new_default["__attempt_sort"].iloc[0])
                if run_index_col in {"combo_rank", "run_index", "combination_index"} and spaces_attempted >= 0:
                    spaces_attempted += 1.0
            else:
                first_idx = strict_recovered.index[0]
                strict_order = list(strict.index)
                spaces_attempted = float(strict_order.index(first_idx) + 1)
                new_default = strict_recovered.loc[[first_idx]]

        row = {
            "bundle": bundle,
            "tract": display_tract,
            "subject": subject,
            "session": session,
            "n_parameter_space_rows": n_rows,
            "n_recovered_parameter_spaces": n_recovered,
            "recovery_fraction": recovery_fraction,
            "spaces_attempted_until_new_default": spaces_attempted,
            "new_default_found": bool(not new_default.empty),
        }

        if not new_default.empty:
            nd = new_default.iloc[0]
            for c in param_cols:
                row[f"new_default_{c.replace('param_', '')}"] = pd.to_numeric(pd.Series([nd.get(c, np.nan)]), errors="coerce").iloc[0]
        else:
            for c in param_cols:
                row[f"new_default_{c.replace('param_', '')}"] = np.nan

        per_subject_session_rows.append(row)

    per_subject_session = pd.DataFrame(per_subject_session_rows)
    if per_subject_session.empty:
        return pd.DataFrame()

    summary_rows = []
    param_default_cols = [f"new_default_{c.replace('param_', '')}" for c in param_cols]

    for (bundle, tract), g in per_subject_session.groupby(["bundle", "tract"], dropna=False, sort=False):
        out = {
            "bundle": bundle,
            "tract": tract,
            "n_subject_sessions": int(g[["subject", "session"]].drop_duplicates().shape[0]),
            "mean_recovery_fraction": float(g["recovery_fraction"].mean()) if g["recovery_fraction"].notna().any() else np.nan,
            "mean_spaces_attempted_until_new_default": float(g["spaces_attempted_until_new_default"].mean()) if g["spaces_attempted_until_new_default"].notna().any() else np.nan,
            "n_subject_sessions_with_new_default": int(g["new_default_found"].sum()),
        }
        for c in param_default_cols:
            out[f"mean_{c}"] = float(g[c].mean()) if g[c].notna().any() else np.nan
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["tract_order"] = summary["tract"].map({
            "left AF": 0,
            "right AF": 1,
            "left FX": 2,
            "right FX": 3,
            "left CST": 4,
            "right CST": 5,
        })
        summary = summary.sort_values(["tract_order", "tract"], na_position="last").drop(columns=["tract_order"])
    return summary


def write_recovery_new_default_summary(raw_df: pd.DataFrame, param_cols: Sequence[str], out_dir: Path) -> str:
    ensure_dir(out_dir)
    out_path = out_dir / "tract_recovery_new_default_summary.csv"
    summary = compute_recovery_new_default_summary(raw_df, param_cols)
    summary.to_csv(out_path, index=False)
    return str(out_path)



def run_svm_information_boundary_outputs(raw_df: pd.DataFrame, param_cols: list[str], age_col: str | None, args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    level_maps = parameter_level_maps(raw_df, param_cols)
    full_grid = full_parameter_grid(param_cols, level_maps)
    map_embedding_df, axis_vecs_2d = compute_parameter_embedding_2d(full_grid, param_cols, level_maps, args)
    edges = build_hyperlattice_edges(map_embedding_df, param_cols, level_maps)
    map_ranges = fixed_parameter_space_map_ranges(map_embedding_df)

    figure_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    score_rows = []
    qc_rows = []
    enrichment_rows = []
    dependence_rows = []
    manifest_rows = []
    score_by_bundle: dict[str, pd.DataFrame] = {}

    bundles = sorted(raw_df["bundle"].dropna().astype(str).unique())
    for bundle_name in bundles:
        bundle_raw = raw_df.loc[raw_df["bundle"].astype(str).eq(bundle_name)].copy()
        if bundle_raw.empty:
            continue
        combo_df = aggregate_combinations(bundle_raw, param_cols, args.outcome_col)
        combo_df = compute_information_shift_metric(bundle_raw, combo_df, param_cols, args.outcome_col, args.information_divergence_bins)
        combo_df["parameter_key"] = parameter_key_df(combo_df, param_cols)
        combo_df = combo_df.merge(
            map_embedding_df[["parameter_key", "map_x", "map_y"]],
            on="parameter_key",
            how="left",
            validate="many_to_one",
        )

        score_df, qc = fit_svm_information_boundary(combo_df, param_cols, level_maps, args)
        score_df["bundle"] = bundle_name
        score_df["display_tract"] = display_tract_name(bundle_name)
        score_by_bundle[bundle_name] = score_df.copy()
        qc["bundle"] = bundle_name
        qc["display_tract"] = display_tract_name(bundle_name)

        stem = sanitize_name(bundle_name)
        score_path = table_dir / f"{stem}__svm_information_boundary_scores.csv"
        enrich_path = table_dir / f"{stem}__svm_information_region_enrichment.csv"
        dependence_path = table_dir / f"{stem}__svm_higher_order_dependence.csv"

        score_df.to_csv(score_path, index=False)
        score_rows.append(score_df)
        qc_rows.append(qc)

        if qc.get("status") == "ok":
            enrichment_df = compute_svm_information_enrichment(score_df, param_cols, level_maps, args)
            enrichment_df["bundle"] = bundle_name
            enrichment_df["display_tract"] = display_tract_name(bundle_name)
            enrichment_df.to_csv(enrich_path, index=False)
            enrichment_rows.append(enrichment_df)

            dependence_df = compute_svm_higher_order_dependence(
                score_df,
                param_cols,
                bundle_name=bundle_name,
                orders=(1, 2, 3, 4, 5),
                min_support_count=3,
                min_region_count=5,
                precision_threshold=0.80,
                coverage_threshold=0.25,
            )
            dependence_df.to_csv(dependence_path, index=False)
            dependence_rows.append(dependence_df)
        else:
            pd.DataFrame().to_csv(enrich_path, index=False)
            pd.DataFrame().to_csv(dependence_path, index=False)

        manifest_rows.append(
            {
                "bundle": bundle_name,
                "display_tract": display_tract_name(bundle_name),
                "score_table_csv": str(score_path),
                "enrichment_table_csv": str(enrich_path),
                "dependence_table_csv": str(dependence_path),
                **{f"qc_{k}": v for k, v in qc.items()},
            }
        )

    score_summary_path = out_dir / "svm_information_boundary_scores.csv"
    qc_path = out_dir / "svm_information_boundary_model_qc.csv"
    enrichment_summary_path = out_dir / "svm_information_region_enrichment.csv"
    dependence_summary_path = out_dir / "svm_higher_order_dependence.csv"
    manifest_path = out_dir / "svm_information_boundary_manifest.csv"
    combined_grid_png = figure_dir / "svm_combined_three_panel_grid.png"

    pd.concat(score_rows, ignore_index=True).to_csv(score_summary_path, index=False) if score_rows else pd.DataFrame().to_csv(score_summary_path, index=False)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    enrichment_all = pd.concat(enrichment_rows, ignore_index=True) if enrichment_rows else pd.DataFrame()
    dependence_all = pd.concat(dependence_rows, ignore_index=True) if dependence_rows else pd.DataFrame(
        columns=[
            "bundle", "display_tract", "order", "range_label",
            "svm_region_count", "precision_for_svm_region",
            "coverage_of_svm_region", "qualifies_threshold", "first_qualifying_order",
            "is_first_qualifying_order",
        ]
    )
    enrichment_all.to_csv(enrichment_summary_path, index=False)
    dependence_all.to_csv(dependence_summary_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    ok_scores = {k: v for k, v in score_by_bundle.items() if not v.empty}
    if ok_scores and (not enrichment_all.empty):
        plot_svm_combined_three_panel_grid(
            ok_scores,
            enrichment_all,
            dependence_all,
            map_embedding_df,
            edges,
            axis_vecs_2d,
            map_ranges,
            param_cols,
            args,
            combined_grid_png,
        )

    return {
        "score_summary_csv": str(score_summary_path),
        "qc_summary_csv": str(qc_path),
        "enrichment_summary_csv": str(enrichment_summary_path),
        "dependence_summary_csv": str(dependence_summary_path),
        "manifest_csv": str(manifest_path),
        "figure_dir": str(figure_dir),
        "table_dir": str(table_dir),
        "svm_combined_grid_png": str(combined_grid_png),
    }


def run_tract_similarity_outputs(raw_df: pd.DataFrame, param_cols: Sequence[str], args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    sim_input = tract_similarity_inputs(raw_df, param_cols, args.outcome_col)
    (
        full_corr,
        full_p,
        full_q,
        best_jaccard,
        best_p,
        best_q,
        worst_jaccard,
        worst_p,
        worst_q,
        n_common,
        set_summary,
    ) = compute_tract_similarity_matrices(
        sim_input,
        best_fraction=args.similarity_best_fraction,
        min_common=args.similarity_min_common,
    )

    summary = tract_similarity_plot_summary_long(
        full_corr=full_corr,
        full_p=full_p,
        full_q=full_q,
        best_jaccard=best_jaccard,
        best_p=best_p,
        best_q=best_q,
        worst_jaccard=worst_jaccard,
        worst_p=worst_p,
        worst_q=worst_q,
        n_common=n_common,
    )

    paths = {
        "summary_csv": out_dir / "tract_similarity_plot_summary.csv",
        "figure_png": out_dir / "tract_similarity_reference_alignment_three_matrix_summary.png",
    }
    summary.to_csv(paths["summary_csv"], index=False)
    plot_tract_similarity_figure(full_corr, full_p, full_q, best_jaccard, best_p, best_q, worst_jaccard, worst_p, worst_q, paths["figure_png"])
    return {k: str(v) for k, v in paths.items()}



def run_hyperlattice_outputs(raw_df: pd.DataFrame, param_cols: list[str], args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    level_maps = parameter_level_maps(raw_df, param_cols)
    full_grid = full_parameter_grid(param_cols, level_maps)

    embedding_df, axis_vecs = compute_parameter_embedding(full_grid, param_cols, level_maps, args)
    edges = build_hyperlattice_edges(embedding_df, param_cols, level_maps)
    scene_ranges = fixed_scene_ranges(embedding_df)

    toggle_plot_dir = out_dir / "per_tract_hyperlattice_toggle_plots"
    table_dir = out_dir / "per_tract_hyperlattice_tables"
    ensure_dir(toggle_plot_dir)
    ensure_dir(table_dir)

    embedding_path = out_dir / f"fixed_parameter_space_embedding_{args.embedding_method}.csv"
    embedding_df.to_csv(embedding_path, index=False)

    summary_rows = []
    bundles = sorted(raw_df["bundle"].dropna().astype(str).unique())
    combo_by_bundle: dict[str, pd.DataFrame] = {}
    raw_by_bundle: dict[str, pd.DataFrame] = {}
    table_path_by_bundle: dict[str, Path] = {}

    for bundle_name in bundles:
        bundle_raw = raw_df.loc[raw_df["bundle"].astype(str).eq(bundle_name)].copy()
        if bundle_raw.empty:
            continue

        combo_df = aggregate_combinations(bundle_raw, param_cols, args.outcome_col)
        combo_df = add_measure_columns(
            bundle_raw,
            combo_df,
            bundle_name,
            param_cols,
            args,
        )
        combo_df = merge_embedding(combo_df, embedding_df, param_cols)

        stem = sanitize_name(bundle_name)
        table_path = table_dir / f"{stem}__factorial_combinations_fixed_embedding_multi_measure.csv"
        combo_df.to_csv(table_path, index=False)
        combo_by_bundle[bundle_name] = combo_df
        raw_by_bundle[bundle_name] = bundle_raw
        table_path_by_bundle[bundle_name] = table_path

    shared_information_scale = shared_information_color_settings(combo_by_bundle, args)

    for bundle_name in bundles:
        if bundle_name not in combo_by_bundle:
            continue
        bundle_raw = raw_by_bundle[bundle_name]
        combo_df = combo_by_bundle[bundle_name]
        table_path = table_path_by_bundle[bundle_name]
        stem = sanitize_name(bundle_name)

        toggle_fig, toggle_trace_meta, initial_metric, initial_level, initial_style = plot_combined_metric_toggle_hyperlattice(
            combo_df=combo_df,
            embedding_full=embedding_df,
            edges=edges,
            axis_vecs=axis_vecs,
            scene_ranges=scene_ranges,
            param_cols=param_cols,
            level_maps=level_maps,
            args=args,
            bundle_name=bundle_name,
            shared_information_scale=shared_information_scale,
        )
        toggle_html = toggle_plot_dir / f"{stem}__hyperlattice_metric_toggle_{args.embedding_method}.html"
        write_toggle_html(
            toggle_fig,
            toggle_html,
            toggle_trace_meta,
            initial_metric=initial_metric,
            initial_level=initial_level,
            initial_style=initial_style,
        )

        for measure_key in HYPERLATTICE_MEASURE_KEYS:
            if measure_key == "information_shift_scaled":
                continue
            if measure_key not in args.measures:
                continue
            measure = MEASURE_SPECS[measure_key]
            if measure.column not in combo_df.columns:
                continue
            metric_available = combo_df[measure.column].notna() & combo_df["is_recovered_combination"].astype(bool)
            summary_rows.append(
                {
                    "bundle": bundle_name,
                    "measure": measure_key,
                    "embedding_method": args.embedding_method,
                    "n_loaded_rows": int(len(bundle_raw)),
                    "n_strict_search_rows": int(bundle_raw["search_stage"].astype(str).eq("strict_search").sum()) if "search_stage" in bundle_raw.columns else 0,
                    "n_strict_search_nonrecovered_rows": int((bundle_raw["search_stage"].astype(str).eq("strict_search") & ~bundle_raw["is_recovered_row"].astype(bool)).sum()) if "search_stage" in bundle_raw.columns else 0,
                    "n_observed_parameter_combinations": int(len(combo_df)),
                    "n_recovered_parameter_combinations": int(combo_df["is_recovered_combination"].sum()),
                    "n_nonrecovered_parameter_combinations": int((~combo_df["is_recovered_combination"]).sum()),
                    "n_metric_available_combinations": int(metric_available.sum()),
                    "n_metric_unavailable_or_nonrecovered_combinations": int((~metric_available).sum()),
                    "metric_min": float(combo_df.loc[metric_available, measure.column].min()) if metric_available.any() else np.nan,
                    "metric_median": float(combo_df.loc[metric_available, measure.column].median()) if metric_available.any() else np.nan,
                    "metric_max": float(combo_df.loc[metric_available, measure.column].max()) if metric_available.any() else np.nan,
                    "outcome_col": args.outcome_col,
                    "shape_cols": ",".join([c for c in args.shape_cols if c in raw_df.columns]),
                    "metric_toggle_html_plot": str(toggle_html),
                    "combo_table_csv": str(table_path),
                }
            )

    summary_path = out_dir / "per_tract_hyperlattice_multi_measure_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    return {
        "summary_csv": str(summary_path),
        "embedding_csv": str(embedding_path),
        "toggle_plot_dir": str(toggle_plot_dir),
        "table_dir": str(table_dir),
    }



def _read_cached_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached analysis table required for --rerender_plots_only: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_cached_boolean_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].apply(parse_bool_like)
    return out


def _cached_embedding_path(directory: Path, prefix: str, requested_method: str) -> Path:
    exact = directory / f"{prefix}_{requested_method}.csv"
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"{prefix}_*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No cached embedding CSV found under {directory} with prefix {prefix}")
    raise FileNotFoundError(
        f"Multiple cached embedding CSVs found under {directory}; pass the matching embedding method explicitly: "
        + ", ".join(path.name for path in matches)
    )


def _embedding_method_from_cached_path(path: Path, prefix: str) -> str:
    expected_prefix = f"{prefix}_"
    stem = path.stem
    if not stem.startswith(expected_prefix):
        raise ValueError(f"Cannot infer embedding method from cached path: {path}")
    return stem[len(expected_prefix):]


def _restore_parameter_keys(df: pd.DataFrame, param_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["parameter_key"] = parameter_key_df(out, param_cols)
    return out


def _axis_vectors_from_cached_embedding(
    embedding_df: pd.DataFrame,
    param_cols: Sequence[str],
    coordinate_cols: Sequence[str],
) -> np.ndarray:
    vectors = []
    for i, column in enumerate(param_cols):
        values = pd.to_numeric(embedding_df[column], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            vector = np.zeros(len(coordinate_cols), dtype=float)
            vector[min(i, len(coordinate_cols) - 1)] = 1.0
            vectors.append(vector)
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        lo_mean = embedding_df.loc[np.isclose(values, lo), list(coordinate_cols)].mean().to_numpy(dtype=float)
        hi_mean = embedding_df.loc[np.isclose(values, hi), list(coordinate_cols)].mean().to_numpy(dtype=float)
        vector = hi_mean - lo_mean
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= EPS:
            vector = np.zeros(len(coordinate_cols), dtype=float)
            vector[min(i, len(coordinate_cols) - 1)] = 1.0
        else:
            vector = vector / norm
        vectors.append(vector)
    return np.asarray(vectors, dtype=float)


def _load_cached_combo_tables(table_dir: Path, pattern: str) -> dict[str, pd.DataFrame]:
    paths = sorted(table_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No cached per-tract tables found in {table_dir} matching {pattern}")
    out: dict[str, pd.DataFrame] = {}
    boolean_columns = [
        "is_recovered_combination",
        "attempted_not_recovered",
        "svm_information_target",
        "svm_raw_boundary_positive",
        "svm_predicted_high_information_region",
    ]
    for path in paths:
        df = _read_cached_csv(path)
        if df.empty:
            continue
        df = _normalize_cached_boolean_columns(df, boolean_columns)
        if "bundle" not in df.columns or df["bundle"].dropna().empty:
            raise ValueError(f"Cached table does not contain a usable bundle column: {path}")
        bundle_name = str(df["bundle"].dropna().iloc[0])
        out[bundle_name] = df
    if not out:
        raise ValueError(f"Cached tables in {table_dir} were empty")
    return out


def rerender_recovery_plot_from_cache(model_dir: Path) -> Path:
    summary_path = model_dir / "tract_recovery_probability_heatmap_summary.csv"
    out_path = model_dir / "tract_recovery_probability_heatmaps.png"
    summary = _read_cached_csv(summary_path)
    plot_recovery_heatmap_summary_grid(summary, out_path)
    return out_path


def rerender_parameter_space_2d_plots_from_cache(args: argparse.Namespace, map2d_dir: Path) -> list[Path]:
    embedding_path = _cached_embedding_path(
        map2d_dir,
        "fixed_parameter_space_2d_embedding",
        args.parameter_space_2d_method,
    )
    render_args = argparse.Namespace(**vars(args))
    render_args.parameter_space_2d_method = _embedding_method_from_cached_path(
        embedding_path,
        "fixed_parameter_space_2d_embedding",
    )
    embedding_df = _read_cached_csv(embedding_path)
    param_cols = factorial_parameter_columns(embedding_df)
    if not param_cols:
        raise ValueError(f"No varying parameter columns found in cached 2D embedding: {embedding_path}")
    embedding_df = _restore_parameter_keys(embedding_df, param_cols)
    level_maps = parameter_level_maps(embedding_df, param_cols)
    edges = build_hyperlattice_edges(embedding_df, param_cols, level_maps)
    axis_vecs_2d = _axis_vectors_from_cached_embedding(embedding_df, param_cols, ["map_x", "map_y"])
    map_ranges = fixed_parameter_space_map_ranges(embedding_df)

    combo_by_bundle = _load_cached_combo_tables(
        map2d_dir / "per_tract_parameter_space_2d_tables",
        "*__factorial_combinations_fixed_2d_parameter_space_embedding_multi_measure.csv",
    )
    combo_by_bundle = {
        bundle: _restore_parameter_keys(df, param_cols)
        for bundle, df in combo_by_bundle.items()
    }

    plot_dir = map2d_dir / "per_metric_parameter_space_2d_maps"
    ensure_dir(plot_dir)
    outputs = []
    for measure_key in PARAMETER_SPACE_2D_MEASURE_KEYS:
        if measure_key not in args.measures:
            continue
        measure = MEASURE_SPECS[measure_key]
        if not any(measure.column in df.columns for df in combo_by_bundle.values()):
            continue
        fig = plot_parameter_space_metric_grid_figure(
            combo_by_bundle=combo_by_bundle,
            map_embedding_df=embedding_df,
            edges=edges,
            axis_vecs_2d=axis_vecs_2d,
            map_ranges=map_ranges,
            param_cols=param_cols,
            args=render_args,
            measure=measure,
        )
        out_path = plot_dir / f"{sanitize_name(measure.key)}__fixed_2d_parameter_space_grid.png"
        fig.savefig(out_path, dpi=args.parameter_space_2d_plot_dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def rerender_hyperlattice_plots_from_cache(args: argparse.Namespace, hyper_dir: Path) -> list[Path]:
    embedding_path = _cached_embedding_path(
        hyper_dir,
        "fixed_parameter_space_embedding",
        args.embedding_method,
    )
    render_args = argparse.Namespace(**vars(args))
    render_args.embedding_method = _embedding_method_from_cached_path(
        embedding_path,
        "fixed_parameter_space_embedding",
    )
    embedding_df = _read_cached_csv(embedding_path)
    param_cols = factorial_parameter_columns(embedding_df)
    if not param_cols:
        raise ValueError(f"No varying parameter columns found in cached 3D embedding: {embedding_path}")
    embedding_df = _restore_parameter_keys(embedding_df, param_cols)
    level_maps = parameter_level_maps(embedding_df, param_cols)
    edges = build_hyperlattice_edges(embedding_df, param_cols, level_maps)
    axis_vecs = _axis_vectors_from_cached_embedding(embedding_df, param_cols, ["x", "y", "z"])
    scene_ranges = fixed_scene_ranges(embedding_df)

    combo_by_bundle = _load_cached_combo_tables(
        hyper_dir / "per_tract_hyperlattice_tables",
        "*__factorial_combinations_fixed_embedding_multi_measure.csv",
    )
    output_dir = hyper_dir / "per_tract_hyperlattice_toggle_plots"
    ensure_dir(output_dir)
    shared_information_scale = shared_information_color_settings(combo_by_bundle, render_args)
    outputs = []
    for bundle_name, combo_df in combo_by_bundle.items():
        combo_df = _restore_parameter_keys(combo_df, param_cols)
        fig, trace_meta, initial_metric, initial_level, initial_style = plot_combined_metric_toggle_hyperlattice(
            combo_df=combo_df,
            embedding_full=embedding_df,
            edges=edges,
            axis_vecs=axis_vecs,
            scene_ranges=scene_ranges,
            param_cols=param_cols,
            level_maps=level_maps,
            args=render_args,
            bundle_name=bundle_name,
            shared_information_scale=shared_information_scale,
        )
        out_path = output_dir / f"{sanitize_name(bundle_name)}__hyperlattice_metric_toggle_{render_args.embedding_method}.html"
        write_toggle_html(
            fig,
            out_path,
            trace_meta,
            initial_metric=initial_metric,
            initial_level=initial_level,
            initial_style=initial_style,
        )
        outputs.append(out_path)
    return outputs


def rerender_svm_plot_from_cache(args: argparse.Namespace, svm_dir: Path) -> Path | None:
    score_path = svm_dir / "svm_information_boundary_scores.csv"
    enrichment_path = svm_dir / "svm_information_region_enrichment.csv"
    score_all = _read_cached_csv(score_path)
    enrichment_all = _read_cached_csv(enrichment_path)
    if score_all.empty or enrichment_all.empty:
        return None

    score_all = _normalize_cached_boolean_columns(
        score_all,
        [
            "is_recovered_combination",
            "svm_information_target",
            "svm_raw_boundary_positive",
            "svm_predicted_high_information_region",
        ],
    )
    param_cols = factorial_parameter_columns(score_all)
    if not param_cols:
        raise ValueError(f"No varying parameter columns found in cached SVM scores: {score_path}")
    score_all = _restore_parameter_keys(score_all, param_cols)
    map_embedding_df = (
        score_all[list(param_cols) + ["parameter_key", "map_x", "map_y"]]
        .drop_duplicates(subset=["parameter_key"])
        .reset_index(drop=True)
    )
    level_maps = parameter_level_maps(map_embedding_df, param_cols)
    edges = build_hyperlattice_edges(map_embedding_df, param_cols, level_maps)
    axis_vecs_2d = _axis_vectors_from_cached_embedding(map_embedding_df, param_cols, ["map_x", "map_y"])
    map_ranges = fixed_parameter_space_map_ranges(map_embedding_df)
    score_by_bundle = {
        str(bundle): group.copy()
        for bundle, group in score_all.groupby("bundle", sort=False)
    }
    dependence_rows = []
    for bundle_name, bundle_scores in score_by_bundle.items():
        dep = compute_svm_higher_order_dependence(
            bundle_scores,
            param_cols,
            bundle_name=bundle_name,
            orders=(1, 2, 3, 4, 5),
            min_support_count=3,
            min_region_count=5,
            precision_threshold=0.80,
            coverage_threshold=0.25,
        )
        if not dep.empty:
            dependence_rows.append(dep)
    dependence_all = pd.concat(dependence_rows, ignore_index=True) if dependence_rows else pd.DataFrame(
        columns=[
            "bundle", "display_tract", "order", "range_label",
            "svm_region_count", "precision_for_svm_region",
            "qualifies_threshold", "first_qualifying_order",
            "is_first_qualifying_order",
        ]
    )
    out_path = svm_dir / "figures" / "svm_combined_three_panel_grid.png"
    ensure_dir(out_path.parent)
    plot_svm_combined_three_panel_grid(
        score_by_bundle,
        enrichment_all,
        dependence_all,
        map_embedding_df,
        edges,
        axis_vecs_2d,
        map_ranges,
        param_cols,
        args,
        out_path,
    )
    return out_path


def _similarity_matrix_from_long(summary: pd.DataFrame, matrix_id: str, value_column: str) -> pd.DataFrame:
    sub = summary.loc[summary["matrix_id"].astype(str).eq(matrix_id)].copy()
    if sub.empty:
        raise ValueError(f"Cached similarity summary is missing matrix_id={matrix_id}")
    row_order = list(dict.fromkeys(sub["row_tract"].astype(str).tolist()))
    col_order = list(dict.fromkeys(sub["column_tract"].astype(str).tolist()))
    return (
        sub.pivot(index="row_tract", columns="column_tract", values=value_column)
        .reindex(index=row_order, columns=col_order)
    )


def rerender_tract_similarity_plot_from_cache(sim_dir: Path) -> Path:
    summary_path = sim_dir / "tract_similarity_plot_summary.csv"
    summary = _read_cached_csv(summary_path)
    full_corr = _similarity_matrix_from_long(summary, "full_landscape_spearman", "value")
    full_p = _similarity_matrix_from_long(summary, "full_landscape_spearman", "p_value")
    full_q = _similarity_matrix_from_long(summary, "full_landscape_spearman", "fdr_q_value")
    best_jaccard = _similarity_matrix_from_long(summary, "high_reference_alignment_jaccard", "value")
    best_p = _similarity_matrix_from_long(summary, "high_reference_alignment_jaccard", "p_value")
    best_q = _similarity_matrix_from_long(summary, "high_reference_alignment_jaccard", "fdr_q_value")
    worst_jaccard = _similarity_matrix_from_long(summary, "worst_space_jaccard", "value")
    worst_p = _similarity_matrix_from_long(summary, "worst_space_jaccard", "p_value")
    worst_q = _similarity_matrix_from_long(summary, "worst_space_jaccard", "fdr_q_value")
    out_path = sim_dir / "tract_similarity_reference_alignment_three_matrix_summary.png"
    plot_tract_similarity_figure(
        full_corr,
        full_p,
        full_q,
        best_jaccard,
        best_p,
        best_q,
        worst_jaccard,
        worst_p,
        worst_q,
        out_path,
    )
    return out_path


def rerender_all_plots_from_cached_outputs(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    outputs.append(rerender_recovery_plot_from_cache(out_dir / "model_tables"))
    outputs.extend(rerender_hyperlattice_plots_from_cache(args, out_dir / "hyperlattice"))
    outputs.extend(rerender_parameter_space_2d_plots_from_cache(args, out_dir / "parameter_space_2d_embedding"))
    svm_output = rerender_svm_plot_from_cache(args, out_dir / "svm_information_boundary")
    if svm_output is not None:
        outputs.append(svm_output)
    outputs.append(rerender_tract_similarity_plot_from_cache(out_dir / "tract_similarity_reference_alignment"))
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated factorial-only analysis and plotting pipeline.")
    p.add_argument("--summary_root", default="")
    p.add_argument("--runs_csv", nargs="*", default=[])
    p.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    p.add_argument("--age_csv", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rerender_plots_only", action="store_true")
    p.add_argument("--analysis_label", default="Factorial sweep")
    p.add_argument("--outcome_col", default=DEFAULT_OUTCOME_COL)
    p.add_argument("--lmm_metric", default=DEFAULT_OUTCOME_COL)
    p.add_argument("--lmm_results_csv", default="", help=argparse.SUPPRESS)
    p.add_argument("--lmm_term_tests_csv", default="", help=argparse.SUPPRESS)
    p.add_argument("--model_metrics", nargs="+", default=[DEFAULT_OUTCOME_COL])
    p.add_argument("--skip_lmm", action="store_true")
    p.add_argument("--age_col", default="")
    p.add_argument(
        "--measures",
        nargs="+",
        default=PARAMETER_SPACE_2D_MEASURE_KEYS,
        choices=sorted(MEASURE_SPECS),
    )
    p.add_argument("--shape_cols", nargs="+", default=DEFAULT_SHAPE_COLS)
    p.add_argument("--information_divergence_bins", type=int, default=8)
    p.add_argument("--tract_family", nargs="+")
    p.add_argument("--bundle", nargs="+")
    p.add_argument("--session", nargs="+")
    p.add_argument("--subject", nargs="+")
    p.add_argument("--keep_invalid_status", action="store_true")
    p.add_argument("--embedding_method", default="metric_mds", choices=["metric_mds", "classical_mds", "pca"])
    p.add_argument("--mds_n_init", type=int, default=1)
    p.add_argument("--mds_max_iter", type=int, default=300)
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--embedding_scale", type=float, default=1.6)
    p.add_argument("--color_scale", default="Magma")
    p.add_argument("--signed_information_color_scale", default="RdBu_r")
    p.add_argument("--nonrecovery_gray", default="rgb(185,185,185)")
    p.add_argument("--lattice_line_width", type=float, default=1.2)
    p.add_argument("--simple_node_min_size", type=float, default=3.5)
    p.add_argument("--simple_node_size_range", type=float, default=8.0)
    p.add_argument("--simple_node_max_size", type=float, default=12.0)
    p.add_argument("--simple_node_default_size", type=float, default=7.0)
    p.add_argument("--parameter_space_2d_method", default="classical_mds", choices=["classical_mds", "metric_mds", "pca"])
    p.add_argument("--parameter_space_2d_plot_dpi", type=int, default=220)
    p.add_argument("--parameter_space_2d_figure_width", type=float, default=24.0)
    p.add_argument("--parameter_space_2d_figure_height", type=float, default=16.0)
    p.add_argument("--parameter_space_2d_node_size_scale", type=float, default=6.0)
    p.add_argument("--parameter_space_2d_node_size_max", type=float, default=72.0)
    p.add_argument("--parameter_space_2d_unavailable_node_size", type=float, default=9.0)
    p.add_argument("--svm_positive_quantile", type=float, default=0.975)
    p.add_argument("--svm_region_decision_quantile", type=float, default=0.50)
    p.add_argument("--svm_min_positive_count", type=int, default=8)
    p.add_argument("--svm_min_negative_count", type=int, default=20)
    p.add_argument("--svm_cv_folds", type=int, default=5)
    p.add_argument("--svm_cv_repeats", type=int, default=3)
    p.add_argument("--svm_refit_metric", default="average_precision", choices=["average_precision", "balanced_accuracy", "roc_auc"])
    p.add_argument("--svm_class_weight", default="balanced", choices=["balanced", "none"])
    p.add_argument("--svm_c_grid", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--svm_gamma_grid", nargs="+", default=["scale", "0.05", "0.10", "0.20", "0.35", "0.50", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0"])
    p.add_argument("--svm_n_jobs", type=int, default=1)
    p.add_argument("--svm_cache_size_mb", type=float, default=1000.0)
    p.add_argument("--svm_tol", type=float, default=0.0005)
    p.add_argument("--svm_max_iter", type=int, default=-1)
    p.add_argument("--svm_enrichment_pseudocount", type=float, default=0.5)
    p.add_argument("--similarity_best_fraction", type=float, default=0.05)
    p.add_argument("--similarity_min_common", type=int, default=10)
    args = p.parse_args()
    if not args.rerender_plots_only and not args.summary_root and not args.runs_csv:
        raise ValueError("Provide either --summary_root or --runs_csv.")
    if not args.rerender_plots_only and not args.age_csv:
        raise ValueError("Provide --age_csv unless --rerender_plots_only is used.")
    if args.similarity_best_fraction <= 0 or args.similarity_best_fraction >= 1:
        raise ValueError("--similarity_best_fraction must be between 0 and 1.")
    if args.svm_positive_quantile <= 0 or args.svm_positive_quantile >= 1:
        raise ValueError("--svm_positive_quantile must be between 0 and 1.")
    if args.svm_region_decision_quantile < 0 or args.svm_region_decision_quantile >= 1:
        raise ValueError("--svm_region_decision_quantile must be in [0, 1).")
    if args.svm_cv_folds < 2:
        raise ValueError("--svm_cv_folds must be at least 2.")
    if args.svm_cv_repeats < 1:
        raise ValueError("--svm_cv_repeats must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.rerender_plots_only:
        rendered = rerender_all_plots_from_cached_outputs(args, out_dir)
        print(f"Re-rendered {len(rendered):,} plots from cached analysis outputs.")
        for path in rendered:
            print(path)
        return

    model_dir = out_dir / "model_tables"
    lmm_dir = out_dir / "mixedlm_morphology"
    hyper_dir = out_dir / "hyperlattice"
    map2d_dir = out_dir / "parameter_space_2d_embedding"
    svm_dir = out_dir / "svm_information_boundary"
    sim_dir = out_dir / "tract_similarity_reference_alignment"
    ensure_dir(model_dir)
    if not args.skip_lmm:
        ensure_dir(lmm_dir)
    ensure_dir(hyper_dir)
    ensure_dir(map2d_dir)
    ensure_dir(svm_dir)
    ensure_dir(sim_dir)

    raw_runs_unfiltered = load_factorial_runs_for_pipeline(args)
    raw_runs_hyperlattice_universe = raw_runs_unfiltered.copy()

    status_qc_summary, status_qc_detail = status_consistency_qc(raw_runs_unfiltered, args.outcome_col)
    status_qc_summary_path = model_dir / "status_consistency_qc_summary.csv"
    status_qc_detail_path = model_dir / "status_consistency_qc_flagged_rows.csv"
    status_qc_summary.to_csv(status_qc_summary_path, index=False)
    status_qc_detail.to_csv(status_qc_detail_path, index=False)

    raw_runs = apply_valid_status_filter(raw_runs_unfiltered, keep_invalid_status=args.keep_invalid_status)
    if raw_runs.empty:
        raise ValueError("No rows remained after valid-status filtering.")

    prepared_all, prepared_success = prepare_factorial_model_inputs(raw_runs, args.age_csv)
    prepared_all_path = model_dir / "prepared_all_factorial_parameter_long.csv"
    prepared_all.to_csv(prepared_all_path, index=False)

    recovery_heatmap_path = model_dir / "tract_recovery_probability_heatmaps.png"
    recovery_heatmap_summary_path = model_dir / "tract_recovery_probability_heatmap_summary.csv"
    recovery_heatmap_summary = plot_recovery_heatmap_grid(prepared_all, recovery_heatmap_path)
    recovery_heatmap_summary.to_csv(recovery_heatmap_summary_path, index=False)

    requested_metrics: list[str] = []
    if not args.skip_lmm:
        prepared_success_path = model_dir / "prepared_success_only.csv"
        prepared_success.to_csv(prepared_success_path, index=False)

        requested_metrics = [m for m in args.model_metrics if m in prepared_success.columns]
        if args.lmm_metric not in requested_metrics and args.lmm_metric in prepared_success.columns:
            requested_metrics.append(args.lmm_metric)
        if not requested_metrics:
            raise ValueError("None of the requested --model_metrics are present in the success-only table.")

        lmm_results, lmm_term_tests = run_all_lmms_minimal(prepared_success, requested_metrics, lmm_dir)
        lmm_results_path = lmm_dir / "lmm_results.csv"
        lmm_term_tests_path = lmm_dir / "lmm_term_tests.csv"
        lmm_results.to_csv(lmm_results_path, index=False)
        lmm_term_tests.to_csv(lmm_term_tests_path, index=False)

        args.lmm_results_csv = str(lmm_results_path)
        args.lmm_term_tests_csv = str(lmm_term_tests_path)
    else:
        args.lmm_results_csv = ""
        args.lmm_term_tests_csv = ""

    hyper_raw, hyper_param_cols, age_col = prepare_hyper_raw_from_filtered_runs(raw_runs_hyperlattice_universe, args)
    hyper_records = run_hyperlattice_outputs(hyper_raw, hyper_param_cols, args, hyper_dir)
    map2d_records = run_parameter_space_2d_outputs(hyper_raw, hyper_param_cols, age_col, args, map2d_dir)
    svm_records = run_svm_information_boundary_outputs(hyper_raw, hyper_param_cols, age_col, args, svm_dir)

    sim_raw, sim_param_cols, _ = prepare_hyper_raw_from_filtered_runs(raw_runs, args)
    sim_records = run_tract_similarity_outputs(sim_raw, sim_param_cols, args, sim_dir)

    print(f"Loaded factorial rows for LMM/similarity: {len(raw_runs):,}")
    print(f"Loaded full attempted-factorial rows for hyperlattices: {len(raw_runs_hyperlattice_universe):,}")
    print(f"Prepared factorial parameter-long rows: {len(prepared_all):,}")
    print(f"Prepared success-only rows: {len(prepared_success):,}")
    print(f"Recovery heatmap PNG: {recovery_heatmap_path}")
    print(f"Recovery heatmap summary CSV: {recovery_heatmap_summary_path}")
    if args.skip_lmm:
        print("LMM outputs: skipped (--skip_lmm)")
    else:
        print(f"LMM metrics: {requested_metrics}")
    print(f"Hyperlattice parameter columns: {hyper_param_cols}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
