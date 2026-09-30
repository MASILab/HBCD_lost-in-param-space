#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation, gaussian_filter, zoom


# Glass-brain shell

def _compute_shell(binary_data, vox_sizes, target_vox=0.5, stdev_mm=2.0, dilate_passes=2):
    zoom_factors = vox_sizes / target_vox
    data_up = zoom(binary_data.astype(float), zoom_factors, order=1)
    stdev_vox = stdev_mm / target_vox
    data_smooth = gaussian_filter(data_up, sigma=stdev_vox)
    data_thres = (data_smooth >= 0.5).astype(np.uint8)
    data_dilated = binary_dilation(data_thres, iterations=dilate_passes).astype(np.uint8)
    shell = (data_dilated - data_thres).astype(np.uint8)
    return shell, zoom_factors


def compute_glass_brain(mask_path, target_vox=0.5, stdev_mm=2.0, dilate_passes=2):
    img = nib.load(mask_path)
    affine = img.affine
    vox_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    binary_data = img.get_fdata() > 0
    shell, zoom_factors = _compute_shell(
        binary_data, vox_sizes, target_vox, stdev_mm, dilate_passes
    )
    return shell, zoom_factors


# Projection helpers

def _orient_projection(proj, axis):
    return proj.T[::-1, :]


def project_glass_brain(shell, axis):
    proj = shell.astype(float).sum(axis=axis)
    return _orient_projection(proj, axis)


# Streamline loading

def load_streamlines(tck_path, mask_img, debug=False):

    tck_file = nib.streamlines.load(tck_path)
    tractogram = tck_file.tractogram
    n_sl = len(tractogram.streamlines)

    if n_sl == 0:
        return [], [], 0

    inv_mask_affine = np.linalg.inv(mask_img.affine)
    tg_affine = tractogram.affine_to_rasmm
    if tg_affine is None:
        tg_affine = np.eye(4)

    vox_sls = []
    dec_colors = []
    all_pts_vox = []

    for sl_native in tractogram.streamlines:
        if len(sl_native) == 0:
            vox_sls.append(np.zeros((0, 3)))
            dec_colors.append(np.zeros((0, 3)))
            continue

        sl_rasmm = nib.affines.apply_affine(tg_affine, sl_native)
        sl_vox = nib.affines.apply_affine(inv_mask_affine, sl_rasmm)

        vox_sls.append(sl_vox)
        all_pts_vox.append(sl_vox)

        if len(sl_rasmm) >= 2:
            tangents = np.diff(sl_rasmm, axis=0)
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            dec = np.abs(tangents / norms)
            dec_colors.append(dec)
        else:
            dec_colors.append(np.zeros((0, 3)))

    if debug and all_pts_vox:
        pts = np.vstack(all_pts_vox)
        print(f"[DEBUG] {Path(tck_path).name}")
        print(f"        tractogram affine_to_rasmm:\n{tg_affine}")
        print(f"        mask affine:\n{mask_img.affine}")
        print(f"        voxel bounds min: {pts.min(axis=0)}")
        print(f"        voxel bounds max: {pts.max(axis=0)}")
        print(f"        mask shape: {mask_img.shape[:3]}")

    return vox_sls, dec_colors, n_sl


def streamlines_to_segments(streamlines_vox, dec_colors, zoom_factors, axis, shell_shape):
    keep = [a for a in range(3) if a != axis]
    n_rows = shell_shape[keep[1]]

    seg_list = []
    col_list = []

    for sl, colors in zip(streamlines_vox, dec_colors):
        if len(sl) < 2:
            continue

        pts_up = sl * zoom_factors
        col = pts_up[:, keep[0]]
        row = (n_rows - 1) - pts_up[:, keep[1]]
        pts_2d = np.column_stack([col, row])
        segments = np.stack([pts_2d[:-1], pts_2d[1:]], axis=1)

        seg_list.append(segments)
        col_list.append(colors)

    if not seg_list:
        return np.empty((0, 2, 2)), np.empty((0, 3))

    return np.concatenate(seg_list), np.concatenate(col_list)


def streamlines_to_segments_solid(streamlines_vox, zoom_factors, axis, shell_shape):
    keep = [a for a in range(3) if a != axis]
    n_rows = shell_shape[keep[1]]

    seg_list = []

    for sl in streamlines_vox:
        if len(sl) < 2:
            continue

        pts_up = sl * zoom_factors
        col = pts_up[:, keep[0]]
        row = (n_rows - 1) - pts_up[:, keep[1]]
        pts_2d = np.column_stack([col, row])
        segments = np.stack([pts_2d[:-1], pts_2d[1:]], axis=1)

        seg_list.append(segments)

    if not seg_list:
        return np.empty((0, 2, 2))

    return np.concatenate(seg_list)


# Utility helpers

def _run_cmd(cmd, verbose=True, force=False):
    cmd = list(cmd)
    if force:
        cmd.append("-force")
    if verbose:
        print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True)


def _strip_known_suffixes(name):
    suffixes = [".tck", ".trk", ".nii.gz", ".nii", ".mif"]
    for s in suffixes:
        if name.endswith(s):
            return name[:-len(s)]
    return os.path.splitext(name)[0]


def _read_pair_tsv(tsv_path):

    pairs = []

    with open(tsv_path, "r", newline="") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty TSV: {tsv_path}")

    first_cols = [x.strip() for x in lines[0].split("\t")]
    has_header = (
        len(first_cols) >= 2 and
        first_cols[0].lower() == "tck" and
        first_cols[1].lower() == "tck2"
    )

    start_idx = 1 if has_header else 0

    for i, line in enumerate(lines[start_idx:], start=start_idx + 1):
        cols = [x.strip() for x in line.split("\t")]
        if len(cols) < 2:
            raise ValueError(f"TSV row {i} must have at least 2 tab-separated columns.")
        tck, tck2 = cols[0], cols[1]
        if tck and tck2:
            pairs.append((tck, tck2))

    if not pairs:
        raise ValueError(f"No valid tract pairs found in TSV: {tsv_path}")

    return pairs


def _coerce_sort_value(value):
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        return s


def _read_sweep_tsv(tsv_path):

    rows = []

    with open(tsv_path, "r", newline="") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty TSV: {tsv_path}")

    header = [x.strip().lower() for x in lines[0].split("\t")]
    if "value" not in header or "tck" not in header:
        raise ValueError(f"{tsv_path} must contain header columns 'value' and 'tck'")

    value_idx = header.index("value")
    tck_idx = header.index("tck")

    for i, line in enumerate(lines[1:], start=2):
        cols = [x.strip() for x in line.split("\t")]
        if max(value_idx, tck_idx) >= len(cols):
            raise ValueError(
                f"TSV row {i} is missing required columns 'value' and/or 'tck'"
            )
        value = cols[value_idx]
        tck = cols[tck_idx]
        if value and tck:
            rows.append((value, tck))

    if not rows:
        raise ValueError(f"No valid sweep rows found in TSV: {tsv_path}")

    rows.sort(key=lambda x: _coerce_sort_value(x[0]))
    values = [r[0] for r in rows]
    tcks = [r[1] for r in rows]
    return values, tcks


# MRtrix transform helpers

def prepare_streamline_warp(flirt_affine, flirt_in_image, flirt_ref_image,
                            template_image, work_dir, verbose=True, force=False):

    flirt_affine = os.path.abspath(flirt_affine)
    flirt_in_image = os.path.abspath(flirt_in_image)
    flirt_ref_image = os.path.abspath(flirt_ref_image)
    template_image = os.path.abspath(template_image)
    work_dir = os.path.abspath(work_dir)

    os.makedirs(work_dir, exist_ok=True)

    stem = _strip_known_suffixes(Path(flirt_affine).name)
    mrtrix_txt = os.path.join(work_dir, f"{stem}_mrtrix.txt")
    inv_txt = os.path.join(work_dir, f"{stem}_mrtrix_inv.txt")
    identity_warp = os.path.join(work_dir, "identity_warp.mif")
    warp_out = os.path.join(work_dir, f"{stem}_streamline_warp.mif")

    _run_cmd([
        "transformconvert",
        flirt_affine,
        flirt_in_image,
        flirt_ref_image,
        "flirt_import",
        mrtrix_txt,
    ], verbose=verbose, force=force)

    _run_cmd([
        "transformcalc",
        mrtrix_txt,
        "invert",
        inv_txt,
    ], verbose=verbose, force=force)

    _run_cmd([
        "warpinit",
        flirt_ref_image,
        identity_warp,
    ], verbose=verbose, force=force)

    _run_cmd([
        "transformcompose",
        identity_warp,
        inv_txt,
        warp_out,
        "-template",
        template_image,
    ], verbose=verbose, force=force)

    if not os.path.exists(warp_out):
        raise RuntimeError(f"Expected warp not found: {warp_out}")

    return warp_out


def apply_warp_to_tck(in_tck, warp_image, out_tck, verbose=True, force=False):

    in_tck = os.path.abspath(in_tck)
    warp_image = os.path.abspath(warp_image)
    out_tck = os.path.abspath(out_tck)

    out_parent = os.path.dirname(out_tck)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    _run_cmd([
        "tcktransform",
        in_tck,
        warp_image,
        out_tck,
    ], verbose=verbose, force=force)

    if not os.path.exists(out_tck):
        raise RuntimeError(f"Expected transformed tract not found: {out_tck}")

    return out_tck


# Rendering

def render_tract_qa(mask_path, tck_path, out_png, title=None,
                    target_vox=0.5, glass_alpha=0.4,
                    tract_alpha=0.6, tract_lw=0.3,
                    precomputed_shell=None, precomputed_zoom=None,
                    precomputed_mask_img=None, debug=False):
    if precomputed_shell is not None and precomputed_zoom is not None:
        shell = precomputed_shell
        zoom_factors = precomputed_zoom
    else:
        shell, zoom_factors = compute_glass_brain(mask_path, target_vox=target_vox)

    mask_img = precomputed_mask_img if precomputed_mask_img is not None else nib.load(mask_path)
    streamlines_vox, dec_colors, n_sl = load_streamlines(tck_path, mask_img, debug=debug)

    if title is None:
        title = Path(tck_path).stem

    view_labels = ["Sagittal", "Coronal", "Axial"]
    proj_axes = [0, 1, 2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.suptitle(f"{title}  ({n_sl} streamlines)",
                 color="black", fontsize=16, fontweight="bold")

    for ax, proj_axis, vlabel in zip(axs, proj_axes, view_labels):
        gb_proj = project_glass_brain(shell, proj_axis)
        gb_max = gb_proj.max()
        if gb_max > 0:
            gb_proj = gb_proj / gb_max

        ax.imshow(gb_proj, cmap="gray_r", vmin=0, vmax=1,
                  alpha=glass_alpha, aspect="equal")

        if n_sl > 0:
            segs, colors = streamlines_to_segments(
                streamlines_vox, dec_colors, zoom_factors,
                proj_axis, shell.shape
            )
            if len(segs) > 0:
                lc = LineCollection(segs, colors=colors,
                                    alpha=tract_alpha, linewidths=tract_lw)
                lc.set_rasterized(True)
                ax.add_collection(lc)

        ax.set_title(vlabel, color="black", fontsize=11)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


def render_parameter_sweep_montage(mask_path, tck_paths, value_labels, out_png,
                                   title=None, target_vox=0.5,
                                   glass_alpha=0.4, tract_alpha=0.6,
                                   tract_lw=0.3,
                                   precomputed_shell=None,
                                   precomputed_zoom=None,
                                   precomputed_mask_img=None,
                                   debug=False,
                                   max_rows_per_fig=None):
    if len(tck_paths) != len(value_labels):
        raise ValueError("tck_paths and value_labels must have the same length")

    if not tck_paths:
        raise ValueError("No tck files provided to render_parameter_sweep_montage")

    if precomputed_shell is not None and precomputed_zoom is not None:
        shell = precomputed_shell
        zoom_factors = precomputed_zoom
    else:
        shell, zoom_factors = compute_glass_brain(mask_path, target_vox=target_vox)

    mask_img = precomputed_mask_img if precomputed_mask_img is not None else nib.load(mask_path)

    if title is None:
        title = "Parameter sweep"

    view_labels = ["Sagittal", "Coronal", "Axial"]
    proj_axes = [0, 1, 2]

    rows = list(zip(value_labels, tck_paths))

    if max_rows_per_fig is None or max_rows_per_fig <= 0:
        chunks = [rows]
    else:
        chunks = [rows[i:i + max_rows_per_fig] for i in range(0, len(rows), max_rows_per_fig)]

    out_png = str(out_png)
    out_dir = os.path.dirname(out_png) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_base, out_ext = os.path.splitext(out_png)
    if not out_ext:
        out_ext = ".png"

    for page_idx, chunk in enumerate(chunks, start=1):
        n_rows = len(chunk)
        fig_height = max(3.8 * n_rows + 1.2, 4.5)
        fig, axs = plt.subplots(
            n_rows, 3,
            figsize=(15, fig_height),
            facecolor="white",
            squeeze=False
        )

        if len(chunks) == 1:
            sup_title = title
        else:
            sup_title = f"{title}  (page {page_idx}/{len(chunks)})"

        fig.suptitle(sup_title, color="black", fontsize=16, fontweight="bold")

        for row_idx, (value_label, tck_path) in enumerate(chunk):
            streamlines_vox, dec_colors, n_sl = load_streamlines(
                tck_path, mask_img, debug=debug
            )

            for col_idx, (proj_axis, vlabel) in enumerate(zip(proj_axes, view_labels)):
                ax = axs[row_idx, col_idx]

                gb_proj = project_glass_brain(shell, proj_axis)
                gb_max = gb_proj.max()
                if gb_max > 0:
                    gb_proj = gb_proj / gb_max

                ax.imshow(
                    gb_proj, cmap="gray_r", vmin=0, vmax=1,
                    alpha=glass_alpha, aspect="equal"
                )

                if n_sl > 0:
                    segs, colors = streamlines_to_segments(
                        streamlines_vox, dec_colors, zoom_factors,
                        proj_axis, shell.shape
                    )
                    if len(segs) > 0:
                        lc = LineCollection(
                            segs,
                            colors=colors,
                            alpha=tract_alpha,
                            linewidths=tract_lw
                        )
                        lc.set_rasterized(True)
                        ax.add_collection(lc)

                if row_idx == 0:
                    ax.set_title(vlabel, color="black", fontsize=11)

                ax.axis("off")

            axs[row_idx, 0].text(
                -0.06, 0.5,
                f"{value_label}\n({n_sl} streamlines)",
                transform=axs[row_idx, 0].transAxes,
                ha="right", va="center",
                fontsize=10, color="black"
            )

        fig.tight_layout(rect=[0.06, 0, 1, 0.95])

        if len(chunks) == 1:
            page_out = out_png
        else:
            page_out = f"{out_base}_page{page_idx:02d}{out_ext}"

        fig.savefig(page_out, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {page_out}")


def render_all_tracts(mask_path, tract_dir, out_dir, target_vox=0.5,
                      glass_alpha=0.4, tract_alpha=0.6, tract_lw=0.3,
                      debug=False):
    tract_dir = Path(tract_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tcks = sorted(tract_dir.glob("*.tck"))
    if not tcks:
        print(f"WARNING: no .tck files found in {tract_dir}")
        return

    print(f"Computing glass brain from {mask_path} ...")
    shell, zoom_factors = compute_glass_brain(mask_path, target_vox=target_vox)
    mask_img = nib.load(mask_path)
    print(f"Glass brain shell shape: {shell.shape}")

    for tck in tcks:
        stem = tck.stem
        out_png = out_dir / f"{stem}.png"
        print(f"Rendering {stem} ...")
        render_tract_qa(
            mask_path=mask_path,
            tck_path=str(tck),
            out_png=str(out_png),
            title=stem,
            target_vox=target_vox,
            glass_alpha=glass_alpha,
            tract_alpha=tract_alpha,
            tract_lw=tract_lw,
            precomputed_shell=shell,
            precomputed_zoom=zoom_factors,
            precomputed_mask_img=mask_img,
            debug=debug,
        )

    print(f"All tract QA PNGs saved to: {out_dir}")


def render_pair_list_from_tsv(mask_path, pair_tsv, out_dir,
                              warp_image,
                              transformed_tck2_out=None,
                              label1="Tract 1", label2="Tract 2",
                              color1="tab:blue", color2="tab:orange",
                              target_vox=0.5, glass_alpha=0.4,
                              tract_alpha=0.6, tract_lw=0.3,
                              debug=False, force=False):
    pairs = _read_pair_tsv(pair_tsv)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transformed_dir = None
    if transformed_tck2_out is not None:
        transformed_dir = Path(transformed_tck2_out).resolve()
        transformed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing glass brain from {mask_path} ...")
    shell, zoom_factors = compute_glass_brain(mask_path, target_vox=target_vox)
    mask_img = nib.load(mask_path)
    print(f"Glass brain shell shape: {shell.shape}")
    print(f"Found {len(pairs)} tract pair(s) in {pair_tsv}")
    print(f"Using warp image: {warp_image}")

    for idx, (tck, tck2) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Processing:")
        print(f"    tck : {tck}")
        print(f"    tck2: {tck2}")

        tck_abs = os.path.abspath(tck)
        tck2_abs = os.path.abspath(tck2)

        if transformed_dir is not None:
            transformed_tck2 = str(transformed_dir / Path(tck2_abs).name)
        else:
            transformed_tck2 = str(out_dir / Path(tck2_abs).name)

        tck2_for_plot = apply_warp_to_tck(
            in_tck=tck2_abs,
            warp_image=warp_image,
            out_tck=transformed_tck2,
            verbose=True,
            force=force,
        )

        png_name = _strip_known_suffixes(Path(tck2_abs).name) + ".png"
        out_png = str(out_dir / png_name)
        title = f"{Path(tck_abs).stem} vs {Path(tck2_abs).stem}"

        render_two_tracts_qa(
            mask_path=mask_path,
            tck_path_1=tck_abs,
            tck_path_2=tck2_for_plot,
            out_png=out_png,
            title=title,
            labels=(label1, label2),
            tract_colors=(color1, color2),
            target_vox=target_vox,
            glass_alpha=glass_alpha,
            tract_alpha=tract_alpha,
            tract_lw=tract_lw,
            precomputed_shell=shell,
            precomputed_zoom=zoom_factors,
            precomputed_mask_img=mask_img,
            debug=debug,
        )

    print(f"All batch pair PNGs saved to: {out_dir}")


def render_two_tracts_qa(mask_path, tck_path_1, tck_path_2, out_png,
                         title=None, labels=("Tract 1", "Tract 2"),
                         tract_colors=("tab:blue", "tab:orange"),
                         target_vox=0.5, glass_alpha=0.4,
                         tract_alpha=0.7, tract_lw=0.35,
                         precomputed_shell=None, precomputed_zoom=None,
                         precomputed_mask_img=None, debug=False):
    if precomputed_shell is not None and precomputed_zoom is not None:
        shell = precomputed_shell
        zoom_factors = precomputed_zoom
    else:
        shell, zoom_factors = compute_glass_brain(mask_path, target_vox=target_vox)

    mask_img = precomputed_mask_img if precomputed_mask_img is not None else nib.load(mask_path)

    streamlines_vox_1, _, n_sl_1 = load_streamlines(tck_path_1, mask_img, debug=debug)
    streamlines_vox_2, _, n_sl_2 = load_streamlines(tck_path_2, mask_img, debug=debug)

    if title is None:
        title = f"{Path(tck_path_1).stem} vs {Path(tck_path_2).stem}"

    view_labels = ["Sagittal", "Coronal", "Axial"]
    proj_axes = [0, 1, 2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.suptitle(f"{title}  ({n_sl_1} vs {n_sl_2} streamlines)",
                 color="black", fontsize=16, fontweight="bold")

    for ax, proj_axis, vlabel in zip(axs, proj_axes, view_labels):
        gb_proj = project_glass_brain(shell, proj_axis)
        gb_max = gb_proj.max()
        if gb_max > 0:
            gb_proj = gb_proj / gb_max

        ax.imshow(gb_proj, cmap="gray_r", vmin=0, vmax=1,
                  alpha=glass_alpha, aspect="equal")

        segs_1 = streamlines_to_segments_solid(
            streamlines_vox_1, zoom_factors, proj_axis, shell.shape
        )
        if len(segs_1) > 0:
            lc1 = LineCollection(
                segs_1,
                colors=[tract_colors[0]] * len(segs_1),
                alpha=tract_alpha,
                linewidths=tract_lw,
            )
            lc1.set_rasterized(True)
            ax.add_collection(lc1)

        segs_2 = streamlines_to_segments_solid(
            streamlines_vox_2, zoom_factors, proj_axis, shell.shape
        )
        if len(segs_2) > 0:
            lc2 = LineCollection(
                segs_2,
                colors=[tract_colors[1]] * len(segs_2),
                alpha=tract_alpha,
                linewidths=tract_lw,
            )
            lc2.set_rasterized(True)
            ax.add_collection(lc2)

        ax.set_title(vlabel, color="black", fontsize=11)
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=tract_colors[0], lw=3, label=labels[0]),
        Line2D([0], [0], color=tract_colors[1], lw=3, label=labels[1]),
    ]
    axs[0].legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


# CLI

def _build_argparser():
    p = argparse.ArgumentParser(
        description="Render sagittal/coronal/axial QA views of tracts with glass-brain overlay."
    )
    p.add_argument("--mask", required=True,
                   help="Brain mask NIfTI (binary), used only for glass-brain rendering.")

    p.add_argument("--tck", default=None,
                   help="Primary .tck file. In overlay mode, this is the static tract.")
    p.add_argument("--tck2", default=None,
                   help="Second .tck file for overlay mode.")
    p.add_argument("--out", default=None,
                   help="Output PNG path (single or overlay mode).")

    p.add_argument("--tract_dir", default=None,
                   help="Directory of .tck files (batch mode).")
    p.add_argument("--out_dir", default=None,
                   help="Output directory for PNGs (batch mode).")

    p.add_argument("--pair_tsv", default=None,
                   help="TSV with columns tck and tck2, or two tab-separated columns without a header.")
    p.add_argument("--pair_out_dir", default=None,
                   help="Output directory for batch pair overlay PNGs.")

    p.add_argument("--sweep_tsv", default=None,
                   help="TSV with columns value and tck for parameter-sweep montage mode.")
    p.add_argument("--sweep_out", default=None,
                   help="Output PNG path for parameter-sweep montage mode.")
    p.add_argument("--max_rows_per_fig", type=int, default=None,
                   help="Optional max rows per montage page. If set, split tall sweeps across multiple PNGs.")

    p.add_argument("--title", default=None,
                   help="Figure title.")
    p.add_argument("--label1", default="Tract 1",
                   help="Legend label for --tck in overlay mode.")
    p.add_argument("--label2", default="Tract 2",
                   help="Legend label for --tck2 in overlay mode.")
    p.add_argument("--color1", default="tab:blue",
                   help="Solid color for first tract in overlay mode.")
    p.add_argument("--color2", default="tab:orange",
                   help="Solid color for second tract in overlay mode.")

    p.add_argument("--target_vox", type=float, default=0.5,
                   help="Glass-brain resolution in mm (default 0.5).")
    p.add_argument("--glass_alpha", type=float, default=0.4,
                   help="Glass-brain opacity (default 0.4).")
    p.add_argument("--tract_alpha", type=float, default=0.6,
                   help="Tract line opacity (default 0.6).")
    p.add_argument("--tract_lw", type=float, default=0.3,
                   help="Tract line width (default 0.3).")
    p.add_argument("--debug", action="store_true",
                   help="Print coordinate-space debug information.")

    p.add_argument("--flirt_fa_affine", default=None,
                   help="Raw FLIRT affine to convert.")
    p.add_argument("--flirt_in_image", default=None,
                   help="Exact image used with FLIRT -in.")
    p.add_argument("--flirt_ref_image", default=None,
                   help="Exact image used with FLIRT -ref.")
    p.add_argument("--warp_work_dir", default=None,
                   help="Working directory for MRtrix warp construction.")
    p.add_argument("--warp_template_image", default=None,
                   help="Template image for transformcompose. Defaults to --flirt_in_image.")
    p.add_argument("--transformed_tck2_out", default=None,
                   help="Directory to save transformed tck2, using the same filename as input --tck2.")
    p.add_argument("--force", action="store_true",
                   help="Pass -force to MRtrix commands to allow repeat runs.")

    return p


def main():
    args = _build_argparser().parse_args()

    warp_image = None
    if args.flirt_fa_affine is not None:
        if args.flirt_in_image is None or args.flirt_ref_image is None:
            print("ERROR: using --flirt_fa_affine requires both --flirt_in_image and --flirt_ref_image")
            raise SystemExit(2)

        warp_work_dir = args.warp_work_dir
        if warp_work_dir is None:
            warp_work_dir = os.path.join(
                tempfile.gettempdir(),
                "tract_qa_mrtrix_warp"
            )

        warp_template = args.warp_template_image or args.flirt_in_image

        try:
            warp_image = prepare_streamline_warp(
                flirt_affine=args.flirt_fa_affine,
                flirt_in_image=args.flirt_in_image,
                flirt_ref_image=args.flirt_ref_image,
                template_image=warp_template,
                work_dir=warp_work_dir,
                verbose=True,
                force=args.force,
            )
        except Exception as e:
            print(f"ERROR preparing streamline warp: {e}")
            raise SystemExit(2)

    if args.pair_tsv:
        if not args.pair_out_dir:
            print("ERROR: --pair_out_dir is required with --pair_tsv")
            raise SystemExit(2)
        if warp_image is None:
            print("ERROR: --flirt_fa_affine is required with --pair_tsv")
            raise SystemExit(2)

        render_pair_list_from_tsv(
            mask_path=args.mask,
            pair_tsv=args.pair_tsv,
            out_dir=args.pair_out_dir,
            warp_image=warp_image,
            transformed_tck2_out=args.transformed_tck2_out,
            label1=args.label1,
            label2=args.label2,
            color1=args.color1,
            color2=args.color2,
            target_vox=args.target_vox,
            glass_alpha=args.glass_alpha,
            tract_alpha=args.tract_alpha,
            tract_lw=args.tract_lw,
            debug=args.debug,
            force=args.force,
        )
        return

    if args.sweep_tsv:
        if not args.sweep_out:
            print("ERROR: --sweep_out is required with --sweep_tsv")
            raise SystemExit(2)

        values, tcks = _read_sweep_tsv(args.sweep_tsv)

        print(f"Computing glass brain from {args.mask} ...")
        shell, zoom_factors = compute_glass_brain(args.mask, target_vox=args.target_vox)
        mask_img = nib.load(args.mask)
        print(f"Glass brain shell shape: {shell.shape}")
        print(f"Found {len(tcks)} tract(s) in sweep TSV: {args.sweep_tsv}")

        render_parameter_sweep_montage(
            mask_path=args.mask,
            tck_paths=[os.path.abspath(x) for x in tcks],
            value_labels=values,
            out_png=args.sweep_out,
            title=args.title,
            target_vox=args.target_vox,
            glass_alpha=args.glass_alpha,
            tract_alpha=args.tract_alpha,
            tract_lw=args.tract_lw,
            precomputed_shell=shell,
            precomputed_zoom=zoom_factors,
            precomputed_mask_img=mask_img,
            debug=args.debug,
            max_rows_per_fig=args.max_rows_per_fig,
        )
        return

    if args.tract_dir:
        if not args.out_dir:
            print("ERROR: --out_dir is required in batch mode")
            raise SystemExit(2)

        render_all_tracts(
            mask_path=args.mask,
            tract_dir=args.tract_dir,
            out_dir=args.out_dir,
            target_vox=args.target_vox,
            glass_alpha=args.glass_alpha,
            tract_alpha=args.tract_alpha,
            tract_lw=args.tract_lw,
            debug=args.debug,
        )
        return

    if args.tck and args.tck2:
        out = args.out or "tract_overlay_qa.png"
        tck2_for_plot = os.path.abspath(args.tck2)

        if warp_image is not None:
            if args.transformed_tck2_out is not None:
                out_dir_for_tck2 = os.path.abspath(args.transformed_tck2_out)
                os.makedirs(out_dir_for_tck2, exist_ok=True)
                transformed_tck2 = os.path.join(
                    out_dir_for_tck2,
                    Path(args.tck2).name
                )
            else:
                if args.out:
                    out_base = os.path.splitext(os.path.abspath(args.out))[0]
                    transformed_tck2 = out_base + "_tck2_fixed.tck"
                else:
                    transformed_tck2 = os.path.join(
                        tempfile.gettempdir(),
                        Path(args.tck2).name
                    )

            print("Applying streamline warp to tract 2 before overlay...")
            tck2_for_plot = apply_warp_to_tck(
                in_tck=os.path.abspath(args.tck2),
                warp_image=warp_image,
                out_tck=transformed_tck2,
                verbose=True,
                force=args.force,
            )
            print(f"Using transformed tract 2: {tck2_for_plot}")

        render_two_tracts_qa(
            mask_path=args.mask,
            tck_path_1=os.path.abspath(args.tck),
            tck_path_2=tck2_for_plot,
            out_png=out,
            title=args.title,
            labels=(args.label1, args.label2),
            tract_colors=(args.color1, args.color2),
            target_vox=args.target_vox,
            glass_alpha=args.glass_alpha,
            tract_alpha=args.tract_alpha,
            tract_lw=args.tract_lw,
            debug=args.debug,
        )
        return

    if args.tck:
        out = args.out or "tract_qa.png"
        render_tract_qa(
            mask_path=args.mask,
            tck_path=os.path.abspath(args.tck),
            out_png=out,
            title=args.title,
            target_vox=args.target_vox,
            glass_alpha=args.glass_alpha,
            tract_alpha=args.tract_alpha,
            tract_lw=args.tract_lw,
            debug=args.debug,
        )
        return

    print("ERROR: provide either --pair_tsv, --sweep_tsv, --tck, --tck + --tck2, or --tract_dir")
    raise SystemExit(2)


if __name__ == "__main__":
    main()