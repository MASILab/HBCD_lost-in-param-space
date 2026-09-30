#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import yaml
from nibabel.affines import apply_affine
from scipy.spatial.distance import directed_hausdorff


@dataclass
class Subject:
    subject: str
    session: str
    fib: str
    mask: str
    references: Dict[str, str]

    @property
    def subject_session_id(self) -> str:
        return f"{self.subject}_{self.session}"


@dataclass
class RunSpec:
    subject: str
    session: str
    fib: str
    mask: str
    bundle: str
    run_type: str
    parameter: str
    value: str
    combo_id: str
    changed_params: List[str]
    resolved_params: Dict[str, Any]
    out_dir: str
    output_hint: str
    trk_gz_file: str
    tck_file: str
    metrics_file: str
    log_file: str
    search_stage: str = ""
    permissiveness_rank: Optional[int] = None
    is_new_default: bool = False

    @property
    def subject_session_id(self) -> str:
        return f"{self.subject}_{self.session}"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def quoted_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _atomic_tmp_path(out_path: str | Path, suffix: str = ".tmp") -> Path:
    out = Path(out_path)
    return out.with_name(out.name + suffix)


def write_json(data: Dict[str, Any], out_path: str) -> None:
    ensure_dir(Path(out_path).parent)
    tmp_path = _atomic_tmp_path(out_path)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, out_path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
def write_json_if_absent(data: Dict[str, Any], out_path: str) -> bool:
    out = Path(out_path)
    ensure_dir(out.parent)
    try:
        with open(out, "x") as f:
            json.dump(data, f, indent=2)
        return True
    except FileExistsError:
        return False


def write_yaml_if_absent(data: Dict[str, Any], out_path: str) -> bool:
    out = Path(out_path)
    ensure_dir(out.parent)
    try:
        with open(out, "x") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return True
    except FileExistsError:
        return False


def metrics_contain_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(metrics_contain_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(metrics_contain_nonfinite(v) for v in value)
    return False


def run_dir_has_existing_content(path: str | Path) -> bool:
    run_path = Path(path)
    if not run_path.exists():
        return False
    try:
        for child in run_path.rglob("*"):
            if child.is_file():
                return True
        return False
    except OSError:
        return True


def metric_source_path(spec: RunSpec) -> str:
    redo = Path(spec.out_dir) / "metrics_redo.json"
    if redo.exists() and redo.is_file() and redo.stat().st_size > 0:
        return str(redo)
    return spec.metrics_file


def completed_metrics_for_summary(
    spec: RunSpec,
    reference_tck: Optional[str],
    include_reference: bool,
) -> Tuple[Dict[str, Any], str]:
    metrics_path = metric_source_path(spec)
    metrics = read_json(metrics_path)
    status = str(metrics.get("status", ""))

    recovered_with_existing_metrics = (
        status == "ok"
        and is_nonempty_file(spec.tck_file)
        and metrics_path == spec.metrics_file
        and metrics_contain_nonfinite(metrics)
    )
    if recovered_with_existing_metrics:
        redo_path = Path(spec.out_dir) / "metrics_redo.json"
        if redo_path.exists() and redo_path.is_file() and redo_path.stat().st_size > 0:
            metrics_path = str(redo_path)
            metrics = read_json(metrics_path)
        else:
            redo_metrics = compute_metrics_for_tck(
                tck_path=spec.tck_file,
                mask_path=spec.mask,
                reference_tck=reference_tck if include_reference else None,
                include_reference=include_reference,
            )
            redo_metrics["status"] = "ok"
            redo_metrics["return_code"] = 0
            redo_metrics["metrics_redo_reason"] = "original_metrics_contained_nan_or_inf"
            write_json_if_absent(redo_metrics, str(redo_path))
            metrics_path = str(redo_path)
            metrics = read_json(metrics_path)

    return metrics, metrics_path


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    ensure_dir(Path(out_path).parent)
    tmp_path = _atomic_tmp_path(out_path)

    if not rows:
        with open(tmp_path, "w") as f:
            f.write("")
        os.replace(tmp_path, out_path)
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, out_path)




def run_cmd(
    cmd: List[str],
    log_file: str,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    ensure_dir(Path(log_file).parent)
    if dry_run:
        print("[DRY-RUN]", quoted_cmd(cmd))
        return 0

    with open(log_file, "a") as log:
        log.write(quoted_cmd(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
        return int(proc.returncode)




def is_nonempty_file(path: str) -> bool:
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def canonical_base(subject: str, session: str, bundle: str) -> str:
    return f"{subject}_{session}_{bundle}"


def append_log(log_file: str, message: str) -> None:
    ensure_dir(Path(log_file).parent)
    with open(log_file, "a") as log:
        log.write(message.rstrip() + "\n")


def split_bundle(bundle: str) -> Tuple[str, str]:
    if bundle.endswith("L"):
        return bundle[:-1], "L"
    if bundle.endswith("R"):
        return bundle[:-1], "R"
    return bundle, "U"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return cfg


def save_resolved_config(cfg: Dict[str, Any]) -> None:
    out_path = Path(cfg["paths"]["summary_root"]) / "resolved_config.yaml"
    write_yaml_if_absent(cfg, str(out_path))


def save_run_manifest(
    cfg: Dict[str, Any],
    n_subject_sessions: int,
    n_runs_planned: int,
) -> None:
    sweeps = cfg.get("sweeps", {})
    n_values = len(next(iter(sweeps.values()))["values"]) if sweeps else 0
    n_params = len(sweeps)
    manifest = {
        "experiment_name": cfg.get("experiment_name", "unnamed"),
        "runner_mode": "factorial_only_full_cartesian_with_backfill",
        "execution_mode": cfg.get("execution", {}).get("mode", "subject"),
        "max_workers": int(cfg.get("execution", {}).get("max_workers", 1)),
        "resume": bool(cfg.get("execution", {}).get("resume", False)),
        "dry_run": bool(cfg.get("execution", {}).get("dry_run", False)),
        "fail_fast": bool(cfg.get("execution", {}).get("fail_fast", False)),
        "n_subject_sessions": n_subject_sessions,
        "n_bundles": len(cfg.get("bundles", [])),
        "n_parameters": n_params,
        "n_values_per_parameter": n_values,
        "n_factorial_combinations_per_bundle": (n_values ** n_params) if sweeps else 0,
        "n_runs_planned_total": n_runs_planned,
        "bundles": cfg.get("bundles", []),
        "parameters": list(sweeps.keys()),
        "summary_outputs": [
            "factorial_runs_long.csv or factorial_runs_long.<subject>_<session>.csv shards",
            "failures.csv or failures.<subject>_<session>.csv shards",
            "status_counts.json or status_counts.<subject>_<session>.json shards",
            "run_manifest.json",
            "resolved_config.yaml",
        ],
        "no_merge_policy": True,
        "backfill_stage": "backfill_strict",
        "metrics_redo_policy": "Recovered runs whose original metrics.json contains NaN or Inf are summarized from metrics_redo.json when available or newly created without overwriting metrics.json.",
    }
    write_json_if_absent(manifest, str(Path(cfg["paths"]["summary_root"]) / "run_manifest.json"))


def load_dataset_json(json_path: str) -> List[Subject]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("dataset_json must be a non-empty list")

    required = {"subject", "session", "fib", "mask", "references"}
    subjects: List[Subject] = []

    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset item {i} must be an object")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Dataset item {i} missing keys: {sorted(missing)}")

        refs = row["references"]
        if not isinstance(refs, dict):
            raise ValueError(f"Dataset item {i} references must be a mapping of bundle -> path")

        subjects.append(
            Subject(
                subject=str(row["subject"]).strip(),
                session=str(row["session"]).strip(),
                fib=str(row["fib"]).strip(),
                mask=str(row["mask"]).strip(),
                references={str(k): str(v).strip() for k, v in refs.items()},
            )
        )
    return subjects


def validate_config_schema(cfg: Dict[str, Any]) -> None:
    required_top = [
        "paths",
        "execution",
        "bundles",
        "baseline_params",
        "sweeps",
    ]
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")

    required_paths = [
        "dsi_studio_bin",
        "dataset_json",
        "output_root",
        "summary_root",
    ]
    for key in required_paths:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing paths.{key}")

    if cfg.get("execution", {}).get("mode", "subject") != "subject":
        raise ValueError("Only execution.mode=subject is supported")

    for param, spec in cfg["sweeps"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"sweeps.{param} must be a mapping")
        if "values" not in spec or "permissive_direction" not in spec:
            raise ValueError(f"sweeps.{param} must contain values and permissive_direction")
        if spec["permissive_direction"] not in {"higher", "lower"}:
            raise ValueError(f"sweeps.{param}.permissive_direction must be 'higher' or 'lower'")
        if not isinstance(spec["values"], list) or not spec["values"]:
            raise ValueError(f"sweeps.{param}.values must be a non-empty list")


def validate_uniform_value_counts(cfg: Dict[str, Any]) -> None:
    counts = [len(v["values"]) for v in cfg["sweeps"].values()]
    if len(set(counts)) != 1:
        raise ValueError("All parameters in sweeps must have the same number of values")


def validate_baselines_in_sweeps(cfg: Dict[str, Any]) -> None:
    baseline = cfg["baseline_params"]
    sweeps = cfg["sweeps"]
    for param, spec in sweeps.items():
        if param not in baseline:
            raise ValueError(f"sweeps.{param} has no matching baseline_params.{param}")
        if str(baseline[param]) not in {str(v) for v in spec["values"]}:
            raise ValueError(f"Baseline value for {param} ({baseline[param]}) is not included in sweeps.{param}.values")


def validate_paths_exist(cfg: Dict[str, Any], subjects: List[Subject]) -> None:
    dsi_bin = Path(cfg["paths"]["dsi_studio_bin"])
    dataset_json = Path(cfg["paths"]["dataset_json"])

    if not dsi_bin.exists():
        raise FileNotFoundError(f"DSI Studio binary not found: {dsi_bin}")
    if not os.access(str(dsi_bin), os.X_OK):
        raise PermissionError(f"DSI Studio binary is not executable: {dsi_bin}")
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")

    for sub in subjects:
        if not Path(sub.fib).exists():
            raise FileNotFoundError(f"Missing FIB for {sub.subject}_{sub.session}: {sub.fib}")
        if not Path(sub.mask).exists():
            raise FileNotFoundError(f"Missing mask for {sub.subject}_{sub.session}: {sub.mask}")

        for bundle, ref in sub.references.items():
            if ref and not Path(ref).exists():
                raise FileNotFoundError(
                    f"Missing reference tract for {sub.subject}_{sub.session} {bundle}: {ref}"
                )


def run_preflight(cfg: Dict[str, Any]) -> List[Subject]:
    validate_config_schema(cfg)
    validate_uniform_value_counts(cfg)
    validate_baselines_in_sweeps(cfg)
    subjects = load_dataset_json(cfg["paths"]["dataset_json"])
    validate_paths_exist(cfg, subjects)
    return subjects


def strict_to_permissive_values(spec: Dict[str, Any]) -> List[Any]:
    values = list(spec["values"])
    direction = spec["permissive_direction"]
    if direction == "higher":
        return sorted(values)
    return sorted(values, reverse=True)


def strict_corner_params(sweeps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {param: strict_to_permissive_values(spec)[0] for param, spec in sweeps.items()}


def stable_combo_id(bundle: str, params: Dict[str, Any]) -> str:
    key = bundle + "|" + "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def generate_ranked_strict_search_combos(
    sweeps: Dict[str, Dict[str, Any]]
) -> List[Tuple[int, Tuple[int, ...], Dict[str, Any]]]:
    param_names = list(sweeps.keys())
    ordered_lists = [strict_to_permissive_values(sweeps[p]) for p in param_names]
    index_ranges = [range(len(vs)) for vs in ordered_lists]

    combos = []
    for idx_tuple in itertools.product(*index_ranges):
        params = {p: ordered_lists[i][idx_tuple[i]] for i, p in enumerate(param_names)}
        rank = sum(idx_tuple)
        combos.append((rank, tuple(idx_tuple), params))

    combos.sort(key=lambda x: (x[0], x[1]))
    return combos


def generate_post_default_factorial_combos(
    sweeps: Dict[str, Dict[str, Any]],
    new_default_params: Dict[str, Any],
) -> List[Tuple[int, Tuple[int, ...], Dict[str, Any]]]:
    param_names = list(sweeps.keys())
    trunc_lists = []
    default_idx_tuple = []

    for p in param_names:
        ordered = strict_to_permissive_values(sweeps[p])
        idx_map = {str(v): i for i, v in enumerate(ordered)}
        d_idx = idx_map[str(new_default_params[p])]
        default_idx_tuple.append(d_idx)
        trunc_lists.append(ordered[d_idx:])

    combos = []
    for offsets in itertools.product(*[range(len(vs)) for vs in trunc_lists]):
        params = {}
        actual_idx_tuple = []
        for i, p in enumerate(param_names):
            ordered = strict_to_permissive_values(sweeps[p])
            d_idx = default_idx_tuple[i]
            actual_idx = d_idx + offsets[i]
            actual_idx_tuple.append(actual_idx)
            params[p] = ordered[actual_idx]
        rank = sum(actual_idx_tuple)
        combos.append((rank, tuple(actual_idx_tuple), params))

    combos.sort(key=lambda x: (x[0], x[1]))
    return combos




def count_planned_runs(cfg: Dict[str, Any], subjects: List[Subject]) -> int:
    sweeps = cfg["sweeps"]
    n_values = len(next(iter(sweeps.values()))["values"])
    n_params = len(sweeps)
    n_bundles = len(cfg["bundles"])
    n_subject_sessions = len(subjects)
    return n_subject_sessions * n_bundles * (n_values ** n_params)


def build_autotrack_cmd(
    dsi_studio_bin: str,
    subject: Subject,
    bundle: str,
    params: Dict[str, Any],
    output_hint: str,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=atk",
        f"--source={subject.fib}",
        f"--track_id={bundle}",
        f"--output={output_hint}",
        "--trk_format=tt.gz",
        f"--tolerance={params.get('tolerance', 24)}",
        f"--track_voxel_ratio={params.get('track_voxel_ratio', 2)}",
        f"--turning_angle={params['turning_angle']}",
        f"--step_size={params['step_size']}",
        f"--smoothing={params['smoothing']}",
        f"--tip_iteration={params['tip_iteration']}",
        "--overwrite=1",
    ]


def build_tt_to_trkgz_cmd(
    dsi_studio_bin: str,
    tt_file: str,
    trk_gz_file: str,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=exp",
        f"--source={tt_file}",
        f"--output={trk_gz_file}",
    ]


def find_nonempty_tt_output(run_dir: str) -> Optional[str]:
    run_path = Path(run_dir)
    candidates = sorted(run_path.rglob("*.tt.gz"))
    valid = []
    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size > 0:
                valid.append(p)
        except OSError:
            continue
    if not valid:
        return None
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(valid[0])


def gunzip_trk_gz_to_bytes(trk_gz_file: str) -> bytes:
    with gzip.open(trk_gz_file, "rb") as fin:
        return fin.read()


def convert_trkgz_to_tck(trk_gz_file: str, tck_file: str, dry_run: bool = False) -> str:
    dst = Path(tck_file)
    ensure_dir(dst.parent)

    if dry_run:
        print(f"[DRY-RUN] convert {trk_gz_file} -> {tck_file}")
        return str(dst)

    raw = gunzip_trk_gz_to_bytes(trk_gz_file)
    tmp_trk = dst.with_suffix(".tmp.trk")
    try:
        with open(tmp_trk, "wb") as f:
            f.write(raw)
        obj = nib.streamlines.load(str(tmp_trk))
        nib.streamlines.save(obj.tractogram, str(dst))
    finally:
        if tmp_trk.exists():
            tmp_trk.unlink()

    return str(dst)


def empty_metrics(include_reference: bool = True) -> Dict[str, Any]:
    out = {
        "streamline_count": 0,
        "tract_volume_mm3": 0.0,
        "surface_area_mm2": math.nan,
        "mean_length_mm": math.nan,
    }
    if include_reference:
        out.update({
            "dice_vs_reference": math.nan,
            "hausdorff_mm_vs_reference": math.nan,
        })
    return out


def failed_metrics(return_code: int, include_reference: bool = True) -> Dict[str, Any]:
    m = empty_metrics(include_reference=include_reference)
    m["status"] = "command_failed"
    m["return_code"] = return_code
    return m


def conversion_failed_metrics(include_reference: bool = True) -> Dict[str, Any]:
    m = empty_metrics(include_reference=include_reference)
    m["status"] = "conversion_failed"
    return m


def load_tck_rasmm(tck_path: str) -> List[np.ndarray]:
    tck = nib.streamlines.load(tck_path)
    tractogram = tck.tractogram
    aff = tractogram.affine_to_rasmm
    if aff is None:
        aff = np.eye(4)

    out: List[np.ndarray] = []
    for sl in tractogram.streamlines:
        if len(sl) == 0:
            out.append(np.zeros((0, 3), dtype=float))
        else:
            out.append(apply_affine(aff, sl))
    return out


def streamline_lengths_mm(streamlines_rasmm: List[np.ndarray]) -> np.ndarray:
    lengths = []
    for sl in streamlines_rasmm:
        if len(sl) < 2:
            lengths.append(0.0)
            continue
        diffs = np.diff(sl, axis=0)
        lengths.append(float(np.linalg.norm(diffs, axis=1).sum()))
    return np.asarray(lengths, dtype=float)


def voxelize_streamlines(
    streamlines_rasmm: List[np.ndarray],
    ref_img: nib.Nifti1Image,
) -> np.ndarray:
    inv_aff = np.linalg.inv(ref_img.affine)
    shape = ref_img.shape[:3]
    occ = np.zeros(shape, dtype=bool)

    for sl in streamlines_rasmm:
        if len(sl) == 0:
            continue
        vox = apply_affine(inv_aff, sl)
        ijk = np.round(vox).astype(int)
        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        ijk = ijk[valid]
        if len(ijk):
            occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occ


def surface_area_mm2_from_occ(occ: np.ndarray, voxel_sizes: np.ndarray) -> float:
    if occ.sum() == 0:
        return 0.0

    sx, sy, sz = map(float, voxel_sizes)
    area_yz = sy * sz
    area_xz = sx * sz
    area_xy = sx * sy

    pad = np.pad(occ.astype(np.uint8), 1, mode="constant", constant_values=0)

    dx = np.abs(np.diff(pad, axis=0)).sum() * area_yz
    dy = np.abs(np.diff(pad, axis=1)).sum() * area_xz
    dz = np.abs(np.diff(pad, axis=2)).sum() * area_xy

    return float(dx + dy + dz)


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return math.nan
    return float((2.0 * inter) / denom)


def sampled_points(streamlines_rasmm: List[np.ndarray], max_points: int = 5000) -> np.ndarray:
    pts = [sl for sl in streamlines_rasmm if len(sl) > 0]
    if not pts:
        return np.zeros((0, 3), dtype=float)
    pts_arr = np.vstack(pts)
    if len(pts_arr) <= max_points:
        return pts_arr
    idx = np.linspace(0, len(pts_arr) - 1, max_points).astype(int)
    return pts_arr[idx]


def hausdorff_distance_mm(a_pts: np.ndarray, b_pts: np.ndarray) -> float:
    if len(a_pts) == 0 or len(b_pts) == 0:
        return math.nan
    h1 = directed_hausdorff(a_pts, b_pts)[0]
    h2 = directed_hausdorff(b_pts, a_pts)[0]
    return float(max(h1, h2))


def compute_reference_metrics_for_tck(
    generated_tck: str,
    reference_tck: str,
    mask_path: str,
) -> Dict[str, Any]:
    mask_img = nib.load(mask_path)

    gen_sl = load_tck_rasmm(generated_tck)
    ref_sl = load_tck_rasmm(reference_tck)

    gen_occ = voxelize_streamlines(gen_sl, mask_img)
    ref_occ = voxelize_streamlines(ref_sl, mask_img)

    gen_pts = sampled_points(gen_sl)
    ref_pts = sampled_points(ref_sl)

    return {
        "dice_vs_reference": dice_binary(gen_occ, ref_occ),
        "hausdorff_mm_vs_reference": hausdorff_distance_mm(gen_pts, ref_pts),
    }


def compute_metrics_for_tck(
    tck_path: str,
    mask_path: str,
    reference_tck: Optional[str] = None,
    include_reference: bool = True,
) -> Dict[str, Any]:
    mask_img = nib.load(mask_path)
    vox_sizes = np.sqrt((mask_img.affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume_mm3 = float(np.prod(vox_sizes))

    streamlines = load_tck_rasmm(tck_path)
    lengths = streamline_lengths_mm(streamlines)
    occ = voxelize_streamlines(streamlines, mask_img)
    occ_vox = int(occ.sum())
    surface_area = surface_area_mm2_from_occ(occ, vox_sizes)

    out = {
        "streamline_count": len(streamlines),
        "tract_volume_mm3": float(occ_vox * voxel_volume_mm3),
        "surface_area_mm2": surface_area,
        "mean_length_mm": float(np.mean(lengths)) if len(lengths) else math.nan,
    }

    if include_reference:
        out.update({
            "dice_vs_reference": math.nan,
            "hausdorff_mm_vs_reference": math.nan,
        })
        if reference_tck and is_nonempty_file(reference_tck):
            out.update(compute_reference_metrics_for_tck(tck_path, reference_tck, mask_path))

    return out










def completed_run_is_valid(spec: RunSpec) -> bool:
    metrics_path = Path(spec.metrics_file)
    if not metrics_path.exists() or not is_nonempty_file(spec.metrics_file):
        return False

    try:
        metrics = read_json(spec.metrics_file)
    except Exception:
        return False

    status = str(metrics.get("status", ""))

    if status == "ok":
        return (
            is_nonempty_file(spec.metrics_file)
            and is_nonempty_file(spec.trk_gz_file)
            and is_nonempty_file(spec.tck_file)
        )

    if status in {"no_output", "command_failed", "conversion_failed"}:
        return is_nonempty_file(spec.metrics_file)

    return False


def execute_one_run(
    spec: RunSpec,
    subject: Subject,
    dsi_studio_bin: str,
    dry_run: bool,
    resume: bool,
    reference_tck: Optional[str],
    include_reference: bool,
) -> Dict[str, Any]:
    preexisting_content = run_dir_has_existing_content(spec.out_dir)

    if completed_run_is_valid(spec):
        metrics, metrics_path = completed_metrics_for_summary(
            spec=spec,
            reference_tck=reference_tck,
            include_reference=include_reference,
        )

        status = metrics.get("status", "ok" if is_nonempty_file(spec.tck_file) else "no_output")
        tt_found = find_nonempty_tt_output(spec.out_dir) or ""

        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": spec.tck_file if is_nonempty_file(spec.tck_file) else "",
            "tt_file": tt_found if is_nonempty_file(tt_found) else "",
            "trk_gz_file": spec.trk_gz_file if is_nonempty_file(spec.trk_gz_file) else "",
            "reference_file": reference_tck if include_reference else "",
            "status": status,
            "return_code": metrics.get("return_code", 0 if status != "command_failed" else None),
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": metrics_path,
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **metrics,
        }

    if preexisting_content:
        tt_found = find_nonempty_tt_output(spec.out_dir) or ""
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": spec.tck_file if is_nonempty_file(spec.tck_file) else "",
            "tt_file": tt_found if is_nonempty_file(tt_found) else "",
            "trk_gz_file": spec.trk_gz_file if is_nonempty_file(spec.trk_gz_file) else "",
            "reference_file": reference_tck if include_reference else "",
            "status": "existing_incomplete_preserved",
            "return_code": "",
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": "",
            "preservation_note": "Run directory contained existing files but did not satisfy completed-run validity checks; no command was executed and no existing file was overwritten.",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **empty_metrics(include_reference=include_reference),
        }

    ensure_dir(spec.out_dir)
    ensure_dir(spec.output_hint)
    write_json_if_absent(spec.resolved_params, str(Path(spec.out_dir) / "params.json"))

    atk_cmd = build_autotrack_cmd(
        dsi_studio_bin=dsi_studio_bin,
        subject=subject,
        bundle=spec.bundle,
        params=spec.resolved_params,
        output_hint=spec.output_hint,
    )

    return_code = run_cmd(
        atk_cmd,
        log_file=spec.log_file,
        dry_run=dry_run,
    )

    if dry_run:
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": "",
            "tt_file": "",
            "trk_gz_file": "",
            "reference_file": reference_tck if include_reference else "",
            "status": "dry_run",
            "return_code": 0,
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": "",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
        }

    discovered_tt = find_nonempty_tt_output(spec.out_dir) or ""

    if return_code != 0:
        metrics = failed_metrics(int(return_code), include_reference=include_reference)
        write_json_if_absent(metrics, spec.metrics_file)
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": "",
            "tt_file": discovered_tt if is_nonempty_file(discovered_tt) else "",
            "trk_gz_file": "",
            "reference_file": reference_tck if include_reference else "",
            "status": "command_failed",
            "return_code": return_code,
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": spec.metrics_file if is_nonempty_file(spec.metrics_file) else "",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **metrics,
        }

    if not discovered_tt or not is_nonempty_file(discovered_tt):
        metrics = empty_metrics(include_reference=include_reference)
        metrics["status"] = "no_output"
        metrics["return_code"] = 0
        write_json_if_absent(metrics, spec.metrics_file)
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": "",
            "tt_file": "",
            "trk_gz_file": "",
            "reference_file": reference_tck if include_reference else "",
            "status": "no_output",
            "return_code": 0,
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": spec.metrics_file if is_nonempty_file(spec.metrics_file) else "",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **metrics,
        }

    trkgz_cmd = build_tt_to_trkgz_cmd(
        dsi_studio_bin=dsi_studio_bin,
        tt_file=discovered_tt,
        trk_gz_file=spec.trk_gz_file,
    )
    conv_rc = run_cmd(
        trkgz_cmd,
        log_file=spec.log_file,
        dry_run=False,
    )

    if conv_rc != 0 or not is_nonempty_file(spec.trk_gz_file):
        metrics = conversion_failed_metrics(include_reference=include_reference)
        metrics["return_code"] = conv_rc
        write_json_if_absent(metrics, spec.metrics_file)
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": "",
            "tt_file": discovered_tt,
            "trk_gz_file": spec.trk_gz_file if is_nonempty_file(spec.trk_gz_file) else "",
            "reference_file": reference_tck if include_reference else "",
            "status": "conversion_failed",
            "return_code": conv_rc,
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": spec.metrics_file if is_nonempty_file(spec.metrics_file) else "",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **metrics,
        }

    append_log(spec.log_file, f"[TT] found: {discovered_tt}")
    append_log(spec.log_file, f"[TRK.GZ] canonical: {spec.trk_gz_file}")

    try:
        convert_trkgz_to_tck(spec.trk_gz_file, spec.tck_file, dry_run=False)
        append_log(spec.log_file, f"[TCK] canonical: {spec.tck_file}")
    except Exception as e:
        metrics = conversion_failed_metrics(include_reference=include_reference)
        metrics["error"] = f"trkgz_to_tck_failed: {e}"
        write_json_if_absent(metrics, spec.metrics_file)
        return {
            **asdict(spec),
            "subject_session_id": spec.subject_session_id,
            "tract_file": "",
            "tt_file": discovered_tt,
            "trk_gz_file": spec.trk_gz_file,
            "reference_file": reference_tck if include_reference else "",
            "status": "conversion_failed",
            "return_code": 0,
            "changed_params": ",".join(spec.changed_params),
            "metrics_file_used": spec.metrics_file if is_nonempty_file(spec.metrics_file) else "",
            **{f"param_{k}": v for k, v in spec.resolved_params.items()},
            **metrics,
        }

    metrics = compute_metrics_for_tck(
        tck_path=spec.tck_file,
        mask_path=spec.mask,
        reference_tck=reference_tck if include_reference else None,
        include_reference=include_reference,
    )
    metrics["status"] = "ok"
    metrics["return_code"] = 0
    write_json_if_absent(metrics, spec.metrics_file)

    return {
        **asdict(spec),
        "subject_session_id": spec.subject_session_id,
        "tract_file": spec.tck_file,
        "tt_file": discovered_tt,
        "trk_gz_file": spec.trk_gz_file,
        "reference_file": reference_tck if include_reference else "",
        "status": "ok",
        "return_code": 0,
        "changed_params": ",".join(spec.changed_params),
        "metrics_file_used": spec.metrics_file if is_nonempty_file(spec.metrics_file) else "",
        **{f"param_{k}": v for k, v in spec.resolved_params.items()},
        **metrics,
    }


def process_subject_session(
    subject: Subject,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    paths = cfg["paths"]
    exec_cfg = cfg.get("execution", {})
    bundles = cfg["bundles"]
    sweeps = cfg["sweeps"]

    dry_run = bool(exec_cfg.get("dry_run", False))
    resume = bool(exec_cfg.get("resume", False))

    output_root = paths["output_root"]
    dsi_studio_bin = paths["dsi_studio_bin"]

    all_rows: List[Dict[str, Any]] = []
    strict_corner = strict_corner_params(sweeps)
    full_ranked = generate_ranked_strict_search_combos(sweeps)

    for bundle in bundles:
        reference_tck = subject.references.get(bundle, "")
        staged_combo_ids: set[str] = set()

        new_default_params = None
        new_default_combo_id = None
        new_default_rank = None

        for rank, idx_tuple, params in full_ranked:
            combo_id = stable_combo_id(bundle, params)
            staged_combo_ids.add(combo_id)
            changed_params = sorted([
                k for k in params
                if str(params[k]) != str(strict_corner[k])
            ])

            out_dir = (
                Path(output_root)
                / subject.subject
                / subject.session
                / bundle
                / "factorial"
                / "strict_search"
                / combo_id
            )
            base = canonical_base(subject.subject, subject.session, bundle)

            spec = RunSpec(
                subject=subject.subject,
                session=subject.session,
                fib=subject.fib,
                mask=subject.mask,
                bundle=bundle,
                run_type="factorial",
                parameter="__factorial__",
                value=combo_id,
                combo_id=combo_id,
                changed_params=changed_params,
                resolved_params=params,
                out_dir=str(out_dir),
                output_hint=str(out_dir / "atk_output/"),
                trk_gz_file=str(out_dir / f"{base}.trk.gz"),
                tck_file=str(out_dir / f"{base}.tck"),
                metrics_file=str(out_dir / "metrics.json"),
                log_file=str(out_dir / "run.log"),
                search_stage="strict_search",
                permissiveness_rank=rank,
                is_new_default=False,
            )

            row = execute_one_run(
                spec=spec,
                subject=subject,
                dsi_studio_bin=dsi_studio_bin,
                dry_run=dry_run,
                resume=resume,
                reference_tck=reference_tck,
                include_reference=True,
            )
            all_rows.append(row)

            if row["status"] == "ok" and row.get("streamline_count", 0) > 0:
                new_default_params = params
                new_default_combo_id = combo_id
                new_default_rank = rank
                row["is_new_default"] = True
                break

        bundle_summary_dir = (
            Path(output_root)
            / subject.subject
            / subject.session
            / bundle
            / "factorial"
        )
        ensure_dir(bundle_summary_dir)

        if new_default_params is None:
            write_json_if_absent(
                {
                    "subject": subject.subject,
                    "session": subject.session,
                    "bundle": bundle,
                    "new_default_found": False,
                },
                str(bundle_summary_dir / "new_default.json"),
            )
        else:
            write_json_if_absent(
                {
                    "subject": subject.subject,
                    "session": subject.session,
                    "bundle": bundle,
                    "new_default_found": True,
                    "new_default_params": new_default_params,
                    "new_default_combo_id": new_default_combo_id,
                    "new_default_rank": new_default_rank,
                },
                str(bundle_summary_dir / "new_default.json"),
            )

            post_default = generate_post_default_factorial_combos(sweeps, new_default_params)
            for rank, idx_tuple, params in post_default:
                combo_id = stable_combo_id(bundle, params)
                if combo_id == new_default_combo_id:
                    continue
                staged_combo_ids.add(combo_id)

                changed_params = sorted([
                    k for k in params
                    if str(params[k]) != str(new_default_params[k])
                ])
                tag = "__default_or_more_permissive__"

                out_dir = (
                    Path(output_root)
                    / subject.subject
                    / subject.session
                    / bundle
                    / "factorial"
                    / tag
                    / combo_id
                )
                base = canonical_base(subject.subject, subject.session, bundle)

                spec = RunSpec(
                    subject=subject.subject,
                    session=subject.session,
                    fib=subject.fib,
                    mask=subject.mask,
                    bundle=bundle,
                    run_type="factorial",
                    parameter="__factorial__",
                    value=combo_id,
                    combo_id=combo_id,
                    changed_params=changed_params,
                    resolved_params=params,
                    out_dir=str(out_dir),
                    output_hint=str(out_dir / "atk_output/"),
                    trk_gz_file=str(out_dir / f"{base}.trk.gz"),
                    tck_file=str(out_dir / f"{base}.tck"),
                    metrics_file=str(out_dir / "metrics.json"),
                    log_file=str(out_dir / "run.log"),
                    search_stage="post_default_factorial",
                    permissiveness_rank=rank,
                    is_new_default=False,
                )

                row = execute_one_run(
                    spec=spec,
                    subject=subject,
                    dsi_studio_bin=dsi_studio_bin,
                    dry_run=dry_run,
                    resume=resume,
                    reference_tck=reference_tck,
                    include_reference=True,
                )
                all_rows.append(row)

        for rank, idx_tuple, params in full_ranked:
            combo_id = stable_combo_id(bundle, params)
            if combo_id in staged_combo_ids:
                continue

            changed_params = sorted([
                k for k in params
                if str(params[k]) != str(strict_corner[k])
            ])

            out_dir = (
                Path(output_root)
                / subject.subject
                / subject.session
                / bundle
                / "factorial"
                / "backfill_strict"
                / combo_id
            )
            base = canonical_base(subject.subject, subject.session, bundle)

            spec = RunSpec(
                subject=subject.subject,
                session=subject.session,
                fib=subject.fib,
                mask=subject.mask,
                bundle=bundle,
                run_type="factorial",
                parameter="__factorial__",
                value=combo_id,
                combo_id=combo_id,
                changed_params=changed_params,
                resolved_params=params,
                out_dir=str(out_dir),
                output_hint=str(out_dir / "atk_output/"),
                trk_gz_file=str(out_dir / f"{base}.trk.gz"),
                tck_file=str(out_dir / f"{base}.tck"),
                metrics_file=str(out_dir / "metrics.json"),
                log_file=str(out_dir / "run.log"),
                search_stage="backfill_strict",
                permissiveness_rank=rank,
                is_new_default=False,
            )

            row = execute_one_run(
                spec=spec,
                subject=subject,
                dsi_studio_bin=dsi_studio_bin,
                dry_run=dry_run,
                resume=resume,
                reference_tck=reference_tck,
                include_reference=True,
            )
            all_rows.append(row)

    return all_rows


def select_one_subject_session(subjects: List[Subject], subject: str, session: str) -> Subject:
    for sub in subjects:
        if sub.subject == subject and sub.session == session:
            return sub
    raise ValueError(f"Subject/session not found in dataset_json: {subject} {session}")


def write_subject_session_job_file(subjects: List[Subject], out_path: str) -> None:
    ensure_dir(Path(out_path).parent)
    tmp_path = _atomic_tmp_path(out_path)
    with open(tmp_path, "w") as f:
        for sub in subjects:
            f.write(f"{sub.subject} {sub.session}\n")
    os.replace(tmp_path, out_path)


def _subject_session_suffix(subject: str, session: str) -> str:
    return f".{subject}_{session}"


def write_summary_shards(
    summary_root: Path,
    factorial_rows: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    suffix: str,
) -> None:
    write_csv([r for r in factorial_rows if r.get("run_type") == "factorial"], str(summary_root / f"factorial_runs_long{suffix}.csv"))
    write_csv(failures, str(summary_root / f"failures{suffix}.csv"))

    status_counts: Dict[str, int] = {}
    for row in factorial_rows:
        s = str(row.get("status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1
    write_json(status_counts, str(summary_root / f"status_counts{suffix}.json"))




def merge_summary_shards(summary_root: Path) -> None:
    raise RuntimeError("Merging summary shards is disabled. Keep per-subject/session factorial_runs_long, failures, and status_counts shards separate.")


def run_parallel_subject_mode(
    cfg: Dict[str, Any],
    subjects: List[Subject],
) -> None:
    paths = cfg["paths"]
    exec_cfg = cfg.get("execution", {})

    ensure_dir(paths["output_root"])
    ensure_dir(paths["summary_root"])

    max_workers = int(exec_cfg.get("max_workers", 1))
    fail_fast = bool(exec_cfg.get("fail_fast", False))
    summary_root = Path(paths["summary_root"])

    if max_workers <= 1:
        for subject in subjects:
            failures: List[Dict[str, Any]] = []
            rows: List[Dict[str, Any]] = []
            try:
                rows = process_subject_session(subject, cfg)
                print(f"[DONE] {subject.subject}_{subject.session}")
            except Exception as e:
                failures.append({
                    "subject": subject.subject,
                    "session": subject.session,
                    "subject_session_id": subject.subject_session_id,
                    "error": str(e),
                })
                print(f"[FAILED] {subject.subject}_{subject.session}: {e}", file=sys.stderr)
                if fail_fast:
                    raise
            finally:
                suffix = _subject_session_suffix(subject.subject, subject.session)
                write_summary_shards(summary_root, rows, failures, suffix=suffix)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(process_subject_session, subject, cfg): subject
                for subject in subjects
            }
            for fut in as_completed(futures):
                subj = futures[fut]
                failures: List[Dict[str, Any]] = []
                rows: List[Dict[str, Any]] = []
                try:
                    rows = fut.result()
                    print(f"[DONE] {subj.subject}_{subj.session}")
                except Exception as e:
                    failures.append({
                        "subject": subj.subject,
                        "session": subj.session,
                        "subject_session_id": subj.subject_session_id,
                        "error": str(e),
                    })
                    print(f"[FAILED] {subj.subject}_{subj.session}: {e}", file=sys.stderr)
                    if fail_fast:
                        raise
                finally:
                    suffix = _subject_session_suffix(subj.subject, subj.session)
                    write_summary_shards(summary_root, rows, failures, suffix=suffix)


def run_one_subject_session(cfg: Dict[str, Any], subject_id: str, session_id: str) -> None:
    subjects = run_preflight(cfg)
    subject = select_one_subject_session(subjects, subject_id, session_id)

    summary_root = Path(cfg["paths"]["summary_root"])
    ensure_dir(summary_root)

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    try:
        all_rows = process_subject_session(subject, cfg)
        print(f"[DONE] {subject.subject}_{subject.session}")
    except Exception as e:
        failures.append({
            "subject": subject.subject,
            "session": subject.session,
            "subject_session_id": subject.subject_session_id,
            "error": str(e),
        })
        print(f"[FAILED] {subject.subject}_{subject.session}: {e}", file=sys.stderr)
        raise
    finally:
        suffix = _subject_session_suffix(subject.subject, subject.session)
        write_summary_shards(summary_root, all_rows, failures, suffix=suffix)


def run_experiment(cfg: Dict[str, Any]) -> None:
    subjects = run_preflight(cfg)
    n_runs = count_planned_runs(cfg, subjects)

    save_resolved_config(cfg)
    save_run_manifest(cfg, n_subject_sessions=len(subjects), n_runs_planned=n_runs)

    mode = cfg.get("execution", {}).get("mode", "subject")
    if mode != "subject":
        raise NotImplementedError("Only subject-level parallelism is implemented.")
    run_parallel_subject_mode(cfg, subjects)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoTrack factorial-only runner with full 3125-combination backfill")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--make_job_file", action="store_true", help="Write subject/session job file and exit")
    p.add_argument("--job_file_out", default="", help="Path to write subject/session jobs")
    p.add_argument("--run_one_subject_session", action="store_true", help="Run exactly one subject/session")
    p.add_argument("--subject", default="", help="Subject ID for --run_one_subject_session")
    p.add_argument("--session", default="", help="Session ID for --run_one_subject_session")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    subjects = run_preflight(cfg)

    if args.make_job_file:
        if not args.job_file_out:
            raise ValueError("--job_file_out is required with --make_job_file")
        save_resolved_config(cfg)
        n_runs = count_planned_runs(cfg, subjects)
        save_run_manifest(cfg, n_subject_sessions=len(subjects), n_runs_planned=n_runs)
        write_subject_session_job_file(subjects, args.job_file_out)
        print(f"Wrote job file: {args.job_file_out}")
        return

    if args.run_one_subject_session:
        if not args.subject or not args.session:
            raise ValueError("--subject and --session are required with --run_one_subject_session")
        n_runs = count_planned_runs(cfg, subjects)
        save_resolved_config(cfg)
        save_run_manifest(cfg, n_subject_sessions=len(subjects), n_runs_planned=n_runs)
        run_one_subject_session(cfg, args.subject, args.session)
        return

    run_experiment(cfg)


if __name__ == "__main__":
    main()
