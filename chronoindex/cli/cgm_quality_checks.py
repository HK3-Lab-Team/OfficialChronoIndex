"""Run unified CGM quality checks from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from chronoindex.dataset_ingestion.dataset_unification import DEFAULT_UNIFIED_METADATA_COLUMNS
from chronoindex.dataset_loader import load_datasets
from chronoindex.dataset_unifier import UnifiedCGMDataset
from chronoindex.glucose_series_processing import cgm_checking_functions as checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run quality checks on a unified CGM dataframe. "
            "Input can be an existing unified parquet, or datasets loaded from config."
        )
    )
    parser.add_argument(
        "--unified-parquet",
        default=None,
        help="Path to an existing unified CGM parquet file.",
    )
    parser.add_argument(
        "--config",
        default="datasets.toml",
        help="Path to datasets TOML config (used when --unified-parquet is not provided).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional subset of dataset names to load when building unified CGM from config.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include datasets marked as enabled=false in config.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Load datasets in parallel threads (config mode).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker threads for --parallel (config mode).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first loading error (config mode).",
    )
    parser.add_argument(
        "--unified-metadata-columns",
        nargs="+",
        default=DEFAULT_UNIFIED_METADATA_COLUMNS,
        help="Metadata columns passed to UnifiedCGMDataset (config mode).",
    )
    parser.add_argument(
        "--min-unique-days",
        type=int,
        default=14,
        help="Minimum unique recording days required per subject.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=100,
        help="Minimum valid CGM recordings required per subject.",
    )
    parser.add_argument("--id-col", default="Id", help="Subject identifier column name.")
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Dataset column name (if missing, checks run only by id).",
    )
    parser.add_argument("--cgm-col", default="CGM", help="CGM values column name.")
    parser.add_argument("--time-col", default="CGMTime", help="CGM timestamp column name.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pre-check and progress prints from run_all_unified_checks.",
    )
    parser.add_argument(
        "--save-report-dir",
        default=None,
        help=(
            "Optional output directory to save per-check parquet reports "
            "(duplicate_cgms/null_values/few_days/few_cgm_records/non_consecutive_days)."
        ),
    )
    return parser.parse_args()


def _load_unified_cgm_from_config(args: argparse.Namespace) -> pl.DataFrame:
    datasets, errors = load_datasets(
        config_path=args.config,
        only=args.only,
        include_disabled=args.include_disabled,
        fail_fast=args.fail_fast,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    if errors:
        print(f"[warning] some datasets failed to load: {sorted(errors.keys())}")
    if not datasets:
        raise ValueError("No datasets were loaded. Cannot run checks.")

    unified = UnifiedCGMDataset(
        datasets,
        metadata_columns=args.unified_metadata_columns,
        config_path=args.config,
    ).unified_data
    return unified


def _load_input_df(args: argparse.Namespace) -> pl.DataFrame:
    if args.unified_parquet:
        path = Path(args.unified_parquet).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Unified parquet not found: {path}")
        print(f"[input] unified parquet: {path}")
        return pl.read_parquet(path)

    print(f"[input] building unified CGM from config: {args.config}")
    return _load_unified_cgm_from_config(args)


def _save_reports(results: dict[str, pl.DataFrame], output_dir: str) -> None:
    outdir = Path(output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        outpath = outdir / f"{name}.parquet"
        frame.write_parquet(outpath)
        print(f"[saved] {outpath} ({frame.height} rows)")


def main() -> None:
    args = _parse_args()
    df = _load_input_df(args)
    print(f"[unified_cgm] shape=({df.height}, {df.width})")

    results = checks.run_all_unified_checks(
        df,
        min_unique_days=args.min_unique_days,
        min_records=args.min_records,
        patient_identifier_col=args.id_col,
        dataset_col=args.dataset_col,
        cgm_col=args.cgm_col,
        time_col=args.time_col,
        verbose=not args.quiet,
    )

    print("[summary]")
    for name, frame in results.items():
        print(f"- {name}: {frame.height} flagged subjects")

    if args.save_report_dir:
        _save_reports(results, args.save_report_dir)


if __name__ == "__main__":
    main()
