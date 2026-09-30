#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparser


RECON_MODE = "gqi"
GQI_METHOD_CODE = 4
GQI_PARAM0 = 1.25
GQI_TEMPLATE = None

ROOT_DIR = Path("/nfs/masi/taylogc1")
CODE_DIR = ROOT_DIR / "code"
BIN_DIR = ROOT_DIR / "bin"

DSI_STUDIO_BIN = str(BIN_DIR / "dsi_studio_singularity.sh")

SUBJECTS_JSON_OUT = ROOT_DIR / "subjects.json"
AGES_CSV_OUT = ROOT_DIR / "ages.csv"
AGES_DETAILS_OUT = ROOT_DIR / "detailed_ages.csv"

REFERENCE_BUNDLES = {
    "ArcuateFasciculusL": "",
    "ArcuateFasciculusR": "",
    "FornixL": "",
    "FornixR": "",
    "CorticospinalTractL": "",
    "CorticospinalTractR": "",
}

BIRTH_KEYS = [
    "BirthDate",
    "PatientBirthDate",
    "PatientBirthDateTime",
    "birth_date",
    "birthdate",
    "date_of_birth",
    "DOB",
    "dob",
    "dateOfBirth",
    "Birth-Date",
    "birth-date",
    "subject_birthdate",
]

ACQ_KEYS = [
    "AcquisitionDateTime",
    "AcquisitionDate",
    "StudyDate",
    "SeriesDate",
    "AcquisitionTime",
    "StudyTime",
    "SeriesTime",
    "AcqDateTime",
    "AcquiredDateTime",
    "ImageAcquisitionDateTime",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subject_session_stem(subject: str, session: str) -> str:
    return f"{subject}_{session}"


def session_output_dir(subject: str, session: str) -> Path:
    return ROOT_DIR / subject / session


def default_t2w_path(subject: str, session: str) -> Path:
    return Path(f"/nfs2/harmonization/BIDS/HBCD/{subject}/{session}/anat/{subject}_{session}_T2w.nii.gz")


def infer_mask_from_dwi(dwi_path: Path) -> Path:
    return dwi_path.with_name("mask.nii.gz")


def run_cmd(cmd: List[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf8") as log:
        log.write(" ".join(str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(str(x) for x in cmd)}")


def load_manifest(manifest_json: Path) -> List[Dict[str, Any]]:
    data = load_json(manifest_json)
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest JSON must be a non-empty list.")

    required = {"subject", "session", "dwi", "bvec", "bval"}
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {i} is not an object.")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Manifest row {i} missing required keys: {sorted(missing)}")

        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()

        dwi = Path(str(row["dwi"])).expanduser()
        bvec = Path(str(row["bvec"])).expanduser()
        bval = Path(str(row["bval"])).expanduser()

        if "t2w" in row and str(row["t2w"]).strip():
            t2w = Path(str(row["t2w"])).expanduser()
        else:
            t2w = default_t2w_path(subject, session)

        for pth, label in [(dwi, "dwi"), (bvec, "bvec"), (bval, "bval")]:
            if not pth.exists():
                raise FileNotFoundError(f"Missing {label} file for {subject} {session}: {pth}")

        out.append({
            "subject": subject,
            "session": session,
            "dwi": dwi,
            "bvec": bvec,
            "bval": bval,
            "t2w": t2w,
        })

    return out


def build_src_cmd(
    dsi_studio_bin: str,
    dwi: Path,
    bval: Path,
    bvec: Path,
    out_sz: Path,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=src",
        f"--source={dwi}",
        f"--bval={bval}",
        f"--bvec={bvec}",
        f"--output={out_sz}",
        "--overwrite=1",
    ]


def build_rec_cmd(
    dsi_studio_bin: str,
    sz_path: Path,
    out_fz: Path,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=rec",
        f"--source={sz_path}",
        f"--method={GQI_METHOD_CODE}",
        f"--param0={GQI_PARAM0}",
        f"--output={out_fz}",
        "--record_odf=1",
        "--overwrite=1",
    ]


def reconstruct_subject_session(
    row: Dict[str, Any],
    dsi_studio_bin: str,
    copy_mask: bool = True,
) -> Dict[str, Any]:
    subject = row["subject"]
    session = row["session"]
    dwi = Path(row["dwi"])
    bvec = Path(row["bvec"])
    bval = Path(row["bval"])

    for p in [dwi, bvec, bval]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    out_dir = session_output_dir(subject, session)
    ensure_dir(out_dir)

    stem = subject_session_stem(subject, session)
    sz_path = out_dir / f"{stem}.sz"
    fz_path = out_dir / f"{stem}.gqi.fz"

    mask_original = infer_mask_from_dwi(dwi)
    mask_copied = out_dir / "mask.nii.gz"
    log_file = out_dir / "recon.log"
    meta_file = out_dir / "recon_meta.json"

    if copy_mask:
        if not mask_original.exists():
            raise FileNotFoundError(f"Expected mask not found next to dwi: {mask_original}")
        shutil.copy2(mask_original, mask_copied)

    print("DEBUG manifest row:")
    print("  subject:", subject)
    print("  session:", session)
    print("  dwi:", dwi)
    print("  bval:", bval)
    print("  bvec:", bvec)

    if not sz_path.exists():
        run_cmd(
            build_src_cmd(
                dsi_studio_bin=dsi_studio_bin,
                dwi=dwi,
                bval=bval,
                bvec=bvec,
                out_sz=sz_path,
            ),
            log_path=log_file,
        )

    if not fz_path.exists():
        run_cmd(
            build_rec_cmd(
                dsi_studio_bin=dsi_studio_bin,
                sz_path=sz_path,
                out_fz=fz_path,
            ),
            log_path=log_file,
        )

    meta = {
        "subject": subject,
        "session": session,
        "subject_session_id": f"{subject}_{session}",
        "dwi": str(dwi),
        "bval": str(bval),
        "bvec": str(bvec),
        "mask_original": str(mask_original),
        "mask_copied": str(mask_copied),
        "sz": str(sz_path),
        "fz": str(fz_path),
        "recon_mode": RECON_MODE,
        "param0": GQI_PARAM0,
        "template": GQI_TEMPLATE,
    }
    save_json(meta, meta_file)
    return meta


def build_subjects_json(recon_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in recon_rows:
        out.append({
            "subject": row["subject"],
            "session": row["session"],
            "fib": row["fz"],
            "mask": row["mask_copied"],
            "references": dict(REFERENCE_BUNDLES),
        })
    return out


def parse_to_datetime(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime.combine(val, time())

    s = str(val).strip()
    if not s:
        return None

    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?(?:\.\d+)?", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh = int(m[4]) if m[4] else 0
        mm = int(m[5]) if m[5] else 0
        ss = int(m[6]) if m[6] else 0
        try:
            return datetime(y, mo, d, hh, mm, ss)
        except Exception:
            pass

    try:
        return dateparser.parse(s)
    except Exception:
        return None


def extract_dates_from_json(jpath: Path) -> Tuple[Optional[date], Optional[date], str]:
    try:
        with open(jpath, "r", encoding="utf8") as fh:
            data = json.load(fh)
    except Exception as e:
        return None, None, f"json_read_error:{e}"

    birth_dt = None
    acq_dt = None

    for k in BIRTH_KEYS:
        if k in data and data[k]:
            parsed = parse_to_datetime(data[k])
            if parsed:
                birth_dt = parsed.date()
                break

    for k in ACQ_KEYS:
        if k in data and data[k]:
            parsed = parse_to_datetime(data[k])
            if parsed:
                acq_dt = parsed.date()
                break

    if acq_dt is None:
        for k in ["AcquisitionDate", "StudyDate", "SeriesDate"]:
            if k in data and data[k]:
                parsed = parse_to_datetime(data[k])
                if parsed:
                    acq_dt = parsed.date()
                    break

    note = ""
    if birth_dt is None:
        note = "no_birth"
    if acq_dt is None:
        note = "no_acq" if not note else note + "|no_acq"

    return birth_dt, acq_dt, note


def compute_age_weeks_decimal(
    birth_date_obj: Optional[date],
    acq_date_obj: Optional[date],
) -> Optional[float]:
    if birth_date_obj is None or acq_date_obj is None:
        return None
    delta_days = (acq_date_obj - birth_date_obj).days
    return round(delta_days / 7.0, 1)


def load_gestational_age_tsv(tsv_path: Path) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    exact_map: Dict[Tuple[str, str], float] = {}
    subject_map: Dict[str, float] = {}

    with open(tsv_path, "r", encoding="utf8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {
            "participant_id",
            "session_id",
            "sed_basic_demographics_gestational_age_delivery",
        }
        if reader.fieldnames is None:
            raise ValueError("Gestational-age TSV has no header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Gestational-age TSV missing columns: {sorted(missing)}")

        for row in reader:
            subject = str(row["participant_id"]).strip()
            session = str(row["session_id"]).strip()
            val_str = str(row["sed_basic_demographics_gestational_age_delivery"]).strip()
            if not subject or not val_str:
                continue
            try:
                ga = float(val_str)
            except ValueError:
                continue

            exact_map[(subject, session)] = ga
            if subject not in subject_map:
                subject_map[subject] = ga

    return exact_map, subject_map


def json_sidecar_from_nifti(nifti_path: Path) -> Path:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[:-7] + ".json")
    if name.endswith(".nii"):
        return nifti_path.with_suffix(".json")
    return nifti_path.with_suffix(".json")


def compute_ages_from_manifest(
    manifest_rows: List[Dict[str, Any]],
    gestational_age_tsv: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ga_exact, ga_subject = load_gestational_age_tsv(gestational_age_tsv)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for row in manifest_rows:
        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()
        t2w_path = Path(row["t2w"]).expanduser()
        anat_raw = t2w_path.parent

        note = ""
        birth_date = None
        acq_date = None
        postnatal_age_weeks = None
        gestational_age_birth_weeks = None
        post_conceptual_age_weeks = None
        json_path = None

        if not t2w_path.exists():
            note = "missing_t2w"
        else:
            json_path = json_sidecar_from_nifti(t2w_path)
            if not json_path.exists():
                note = "missing_json" if not note else note + "|missing_json"
            else:
                birth_date, acq_date, json_note = extract_dates_from_json(json_path)
                if json_note:
                    note = json_note if not note else note + "|" + json_note
                postnatal_age_weeks = compute_age_weeks_decimal(birth_date, acq_date)
                if postnatal_age_weeks is None and not note:
                    note = "age_failed"

        if (subject, session) in ga_exact:
            gestational_age_birth_weeks = ga_exact[(subject, session)]
        elif subject in ga_subject:
            gestational_age_birth_weeks = ga_subject[subject]
            note = "ga_subject_fallback" if not note else note + "|ga_subject_fallback"
        else:
            note = "missing_gest_age" if not note else note + "|missing_gest_age"

        if postnatal_age_weeks is not None and gestational_age_birth_weeks is not None:
            post_conceptual_age_weeks = round(postnatal_age_weeks + gestational_age_birth_weeks, 1)

        summary_rows.append({
            "subject": subject,
            "session": session,
            "age_weeks": post_conceptual_age_weeks if post_conceptual_age_weeks is not None else "",
        })

        detail_rows.append({
            "subject": subject,
            "session": session,
            "anat_raw": str(anat_raw),
            "t2w": str(t2w_path),
            "json_path": str(json_path) if json_path else "",
            "acq_date": acq_date.isoformat() if acq_date else "",
            "birth_date": birth_date.isoformat() if birth_date else "",
            "postnatal_age_weeks": postnatal_age_weeks if postnatal_age_weeks is not None else "",
            "gestational_age_birth_weeks": gestational_age_birth_weeks if gestational_age_birth_weeks is not None else "",
            "post_conceptual_age_weeks": post_conceptual_age_weeks if post_conceptual_age_weeks is not None else "",
            "note": note,
        })

    return summary_rows, detail_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DSI Studio GQI reconstruction from a manifest and write per-session recon files, subjects.json, ages.csv, and detailed_ages.csv under /nfs/masi/taylogc1."
    )
    p.add_argument("manifest_json", type=Path, help="Input manifest JSON with subject/session/dwi/bvec/bval and optionally t2w")
    p.add_argument("gestational_age_tsv", type=Path, help="TSV with gestational age at birth")
    p.add_argument("--dsi_studio_bin", default=DSI_STUDIO_BIN)
    p.add_argument("--copy_mask", action="store_true", help="Copy mask.nii.gz from the dwi directory into each session folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest_json)

    recon_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        recon_rows.append(
            reconstruct_subject_session(
                row=row,
                dsi_studio_bin=args.dsi_studio_bin,
                copy_mask=args.copy_mask,
            )
        )

    subjects_json = build_subjects_json(recon_rows)
    ages_rows, ages_detail_rows = compute_ages_from_manifest(
        manifest_rows=manifest_rows,
        gestational_age_tsv=args.gestational_age_tsv,
    )

    save_json(subjects_json, SUBJECTS_JSON_OUT)

    write_csv(
        ages_rows,
        AGES_CSV_OUT,
        fieldnames=["subject", "session", "age_weeks"],
    )
    write_csv(
        ages_detail_rows,
        AGES_DETAILS_OUT,
        fieldnames=[
            "subject",
            "session",
            "anat_raw",
            "t2w",
            "json_path",
            "acq_date",
            "birth_date",
            "postnatal_age_weeks",
            "gestational_age_birth_weeks",
            "post_conceptual_age_weeks",
            "note",
        ],
    )

    print(f"Wrote subjects.json: {SUBJECTS_JSON_OUT}")
    print(f"Wrote ages.csv: {AGES_CSV_OUT}")
    print(f"Wrote detailed ages CSV: {AGES_DETAILS_OUT}")
    print("Per-session recon files, meta, and logs were written under /nfs/masi/taylogc1/sub-*/ses-*/")


if __name__ == "__main__":
    main()