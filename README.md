# Lost in (parameter) space

This project evaluates how tractography settings influence the recovery, measured morphology, and anatomical agreement of infant white matter pathways. It runs DSI Studio AutoTrack across a five-dimensional parameter grid, compares the resulting pathways with expert-extracted reference tracts, and identifies combinations associated with favorable agreement to expert-extracted reference tracts.

The study includes 12 infants with paired HBCD Visit 2 and Visit 3 scans, giving 24 subject-session datasets. Each dataset contributes six targets: the left and right arcuate fasciculus (AF), fornix (FX), and corticospinal tract (CST). Five ordered values for each of five parameters produces 3,125 combinations per target, yielding a complete design of 450,000 AutoTrack run attempts.

A **streamline** is a computed trajectory through diffusion MRI data. A **tractogram** is a collection of these trajectories. A **reference tract** is a subject-session-specific tractogram selected by an expert using anatomical criteria. A **parameter combination** specifies one value for each of the five settings listed below.

## Research questions:

1. **Recovery:** Which settings produce a non-zero streamline count for a requested pathway?
2. **Morphometric variability:** How does tract size (given by streamline count, length, volume, and surface-area metrics) change across settings?
3. **Anatomical agreement:** How closely does each recovered tractogram match its corresponding expert reference?
4. **Joint parameter dependence:** Which parameter levels and combinations characterize regions with favorable anatomical agreement?

## Workflow and scripts:

| Script | Inputs | Work performed | Main outputs |
|---|---|---|---|
| [`dsi_recon.py`](dsi_recon.py) | Preprocessed DWI manifest, gradient files, gestational-age table, anatomical image sidecars | Creates DSI Studio source files and GQI reconstructions; assembles the sweep input manifest and age tables | `.sz`, `.gqi.fz`, `recon.log`, `recon_meta.json`, `subjects.json`, `ages.csv`, `detailed_ages.csv` |
| [`autotrack_param_sweep_parallel_backfillmode.py`](autotrack_param_sweep_parallel_backfillmode.py) | YAML configuration, reconstructed fiber files, reference-grid masks, expert reference tractograms | Attempts the complete factorial grid; converts tractograms; calculates per-run metrics; records outcomes | Per-run tractograms, parameters, logs and metrics; one summary shard per subject-session |
| [`integrated_factorial_plots_v13.py`](integrated_factorial_plots_v13.py) | Sweep summary shards and age table | Calculates recovery and morphometric summaries, signed information divergence, SVM regions, enrichment, parameter-range rules, and additional analyses | CSV tables, static figures, and interactive parameter-space maps |
| [`plot_svm_enrichment_recovery_style.py`](plot_svm_enrichment_recovery_style.py) | Combined enrichment CSV from the integrated analysis | Arranges existing enrichment values in horizontal layout | `svm_parameter_enrichment_recovery_style.png` |
| [`tract_qa_v3.py`](tract_qa_v3.py) | Brain mask and selected `.tck` tractograms | Creates anatomical QA views, overlays, and parameter-sweep montages | PNG images and, when explicitly requested, transformed tractograms |

## Parameter grid

| Parameter | Values | Meaning | Direction labeled more permissive in this experiment |
|---|---|---|---|
| `turning_angle` | 30, 45, 60, 75, 90 degrees | Maximum permitted change in tracking direction | Higher |
| `step_size` | 0.5, 0.75, 1.0, 1.25, 1.5 mm | Distance advanced during a tracking step | Higher |
| `smoothing` | 0.0, 0.2, 0.4, 0.6, 0.8 | Fraction of the previous direction incorporated during tracking | Lower |
| `tip_iteration` | 0, 4, 8, 16, 32 | Number of topology-informed pruning iterations | Lower |
| `tolerance` | 22, 24, 26, 28, 30 mm | AutoTrack recognition tolerance relative to template pathway | Higher |

## Required inputs:

### Reconstruction manifest

`dsi_recon.py` accepts a JSON list with one record per subject-session:

```json
[
  {
    "subject": "sub-EXAMPLE",
    "session": "ses-V02",
    "dwi": "/absolute/path/preprocessed_dwi.nii.gz",
    "bval": "/absolute/path/preprocessed_dwi.bval",
    "bvec": "/absolute/path/preprocessed_dwi.bvec",
    "t2w": "/absolute/path/sub-EXAMPLE_ses-V02_T2w.nii.gz"
  }
]
```

The required keys are `subject`, `session`, `dwi`, `bval`, and `bvec`. Supply `t2w` explicitly to identify the anatomical sidecar used for ages. With `--copy_mask`, the script expects `mask.nii.gz` beside the DWI file.

The gestational-age TSV requires these headers:

```text
participant_id	session_id	sed_basic_demographics_gestational_age_delivery
```

Birth and acquisition dates come from supported fields in the T2w JSON sidecar. The script first calculates postnatal age in weeks, rounded to one decimal place. It adds gestational age at delivery and writes the rounded sum as `age_weeks` in `ages.csv`. This represents gestational age at birth plus elapsed postnatal age, commonly termed postmenstrual age. `detailed_ages.csv` retains the components and uses the source field name `post_conceptual_age_weeks` for their sum.

### Sweep manifest

The sweep requires a JSON list with these fields:

```json
[
  {
    "subject": "sub-EXAMPLE",
    "session": "ses-V02",
    "fib": "/absolute/path/sub-EXAMPLE_ses-V02.gqi.fz",
    "mask": "/absolute/path/mask.nii.gz",
    "references": {
      "ArcuateFasciculusL": "/absolute/path/AF_L.tck",
      "ArcuateFasciculusR": "/absolute/path/AF_R.tck",
      "FornixL": "/absolute/path/FX_L.tck",
      "FornixR": "/absolute/path/FX_R.tck",
      "CorticospinalTractL": "/absolute/path/CST_L.tck",
      "CorticospinalTractR": "/absolute/path/CST_R.tck"
    }
  }
]
```

The `fib` key accepts the reconstructed fiber-file path, including the `.gqi.fz` output from `dsi_recon.py`. That script initializes reference paths as empty strings. Populate all six paths after expert extraction and conversion to `.tck`.

### Age table

The integrated analysis requires a CSV with `subject`, `session`, and numeric `age_weeks`. Each subject-session must have one row, every included session must have an age, and the table must contain age variation.

### Sweep configuration

Copy the example configuration to a local working file:

```bash
cp config.example.yaml config.local.yaml
```

Set `paths.dsi_studio_bin`, `paths.dataset_json`, `paths.output_root`, and `paths.summary_root` to absolute paths. YAML paths are read literally. Preserve the five parameter grids and their order. Set `execution.max_workers` to the intended number of local subject-session workers. For GNU Parallel, use `max_workers: 1` because each external job already owns one subject-session.

The supplied runner processes the manifest and configured `bundles`.

## Study documentation and source provenance

The study documents are:

1. Taylor GC et al. *Lost in (parameter) space: quantifying parameter-dependent recovery and morphometric variability of infant white matter pathways in the HBCD cohort.* Supplied manuscript: `Taylor_SPIE_manuscript_08-01-26_8-page_FINAL_v11.docx`.

Relevant software references in the manuscript include: Yeh, *Nature Methods* 22, 1617–1619 (2025), for DSI Studio; Cai et al., *Magnetic Resonance in Medicine* 86, 456–470 (2021), for PreQual; and Pedregosa et al., *Journal of Machine Learning Research* 12, 2825–2830 (2011), for scikit-learn. The supplied manuscript contains the full scientific reference list.
