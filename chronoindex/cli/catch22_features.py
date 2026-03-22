"""Compute catch22 features from a CGM table."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from chronoindex.dataset_ingestion.dataset_unification import DEFAULT_UNIFIED_METADATA_COLUMNS
from chronoindex.dataset_loader import load_datasets
from chronoindex.dataset_unifier import UnifiedCGMDataset
from chronoindex.statistical_summaries import catchfeatures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute catch22 features from a CGM table. "
            "Input can be an external parquet/csv file, or datasets loaded from config."
        )
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to an external CGM table (.parquet or .csv).",
    )
    parser.add_argument(
        "--unified-parquet",
        default=None,
        help="Deprecated alias for parquet input. Prefer --input-file.",
    )
    parser.add_argument(
        "--config",
        default="datasets.toml",
        help="Path to datasets TOML config (used when --input-file is not provided).",
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
    parser.add_argument("--id-col", default="Id", help="Subject identifier column name.")
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Dataset column name (if missing, output groups only by id).",
    )
    parser.add_argument("--cgm-col", default="CGM", help="CGM values column name.")
    parser.add_argument(
        "--time-col",
        default="CGMTime",
        help="CGM timestamp column name. Use --no-time-col for non-temporal aggregation.",
    )
    parser.add_argument(
        "--no-time-col",
        action="store_true",
        help="Disable timestamp usage and aggregate catch22 per subject/dataset across all records.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Unnest the catch22 struct into one column per feature.",
    )
    parser.add_argument(
        "--save-output",
        default=None,
        help="Optional output parquet path for catch22 results.",
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
        raise ValueError("No datasets were loaded. Cannot compute catch22 features.")

    return UnifiedCGMDataset(
        datasets,
        metadata_columns=args.unified_metadata_columns,
        config_path=args.config,
    ).unified_data


def _read_input_file(input_path: str) -> pl.DataFrame:
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        print(f"[input] external parquet: {path}")
        return pl.read_parquet(path)
    if suffix == ".csv":
        print(f"[input] external csv: {path}")
        return pl.read_csv(path)
    raise ValueError("Unsupported input format. Expected .parquet or .csv")


def _load_input_df(args: argparse.Namespace) -> pl.DataFrame:
    if args.input_file:
        return _read_input_file(args.input_file)

    if args.unified_parquet:
        return _read_input_file(args.unified_parquet)

    print(f"[input] building unified CGM from config: {args.config}")
    return _load_unified_cgm_from_config(args)


def _save_output(df: pl.DataFrame, output_path: str) -> None:
    out = Path(output_path).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"[saved] {out} ({df.height} rows, {df.width} columns)")


def main() -> None:
    args = _parse_args()
    df = _load_input_df(args)
    print(f"[input_df] shape=({df.height}, {df.width})")

    time_col = None if args.no_time_col else args.time_col
    out = catchfeatures.compute_catch22_features(
        df,
        id_col=args.id_col,
        cgm_col=args.cgm_col,
        cgm_time_col=time_col,
        dataset_col=args.dataset_col,
    )
    if args.flatten and "catch22_features" in out.columns:
        out = out.unnest("catch22_features")

    print(f"[catch22] output_shape=({out.height}, {out.width})")
    print(out.head(10))

    if args.save_output:
        _save_output(out, args.save_output)


if __name__ == "__main__":
    main()
