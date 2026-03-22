"""Generate CGM plots from a CGM table."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

_MATPLOTLIB_CACHE_DIR = (Path.cwd() / ".mpl_cache").resolve()
_MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MATPLOTLIB_CACHE_DIR))

import polars as pl

from chronoindex.dataset_ingestion.dataset_unification import DEFAULT_UNIFIED_METADATA_COLUMNS
from chronoindex.dataset_loader import load_datasets
from chronoindex.dataset_unifier import UnifiedCGMDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CGM plots from a CGM table. Input can be an external parquet/csv file, "
            "or datasets loaded from config."
        )
    )
    parser.add_argument(
        "--plot",
        required=True,
        choices=("time-series", "frequency", "mean"),
        help="Plot type to generate.",
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=("subject", "dataset"),
        help="Aggregation level for plotting.",
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
    parser.add_argument(
        "--subject-id",
        nargs="*",
        default=None,
        help="Subject id(s) to plot (subject level only). If omitted, all subjects are plotted.",
    )
    parser.add_argument(
        "--dataset-name",
        nargs="*",
        default=None,
        help="Dataset name(s) to plot (dataset level only). If omitted, all datasets are plotted.",
    )
    parser.add_argument(
        "--daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Time-series only. If true, one subplot per day with hour-of-day on x-axis. "
            "If false, one plot per subject/dataset with day on x-axis."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins for frequency plots.",
    )
    parser.add_argument("--id-col", default="Id", help="Subject identifier column name.")
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Dataset column name.",
    )
    parser.add_argument("--cgm-col", default="CGM", help="CGM values column name.")
    parser.add_argument("--time-col", default="CGMTime", help="CGM timestamp column name.")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory to save generated plot PNG files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI when using --save-dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
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
        raise ValueError("No datasets were loaded. Cannot generate plots.")

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


def _safe_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return token or "item"


def _save_figures(
    figures: list[Any],
    *,
    output_dir: str,
    level: str,
    plot_type: str,
    labels: list[str] | None,
    dpi: int,
) -> None:
    out_dir = Path(output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, fig in enumerate(figures, start=1):
        label = labels[idx - 1] if labels and idx - 1 < len(labels) else f"{idx:03d}"
        filename = f"{level}_{plot_type}_{_safe_token(label)}.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out_path}")


def _resolve_requested_labels(args: argparse.Namespace) -> list[str] | None:
    if args.level == "subject":
        return args.subject_id if args.subject_id else None
    return args.dataset_name if args.dataset_name else None


def _build_plots(df: pl.DataFrame, args: argparse.Namespace) -> list[Any]:
    from chronoindex.statistical_summaries import plot as plotters

    common_kwargs = {
        "cgm_col": args.cgm_col,
        "dataset_col": args.dataset_col,
        "show": args.show,
    }

    if args.level == "subject":
        common_kwargs["id_col"] = args.id_col
        if args.plot == "time-series":
            return plotters.plot_subject_glucose_time_series(
                df,
                subject_id=args.subject_id,
                daily=args.daily,
                cgm_time_col=args.time_col,
                **common_kwargs,
            )
        if args.plot == "frequency":
            return plotters.plot_subject_glucose_frequency(
                df,
                subject_id=args.subject_id,
                bins=args.bins,
                **common_kwargs,
            )
        return plotters.plot_subject_mean_glucose(
            df,
            subject_id=args.subject_id,
            cgm_time_col=args.time_col,
            **common_kwargs,
        )

    if args.plot == "time-series":
        return plotters.plot_dataset_glucose_time_series(
            df,
            dataset=args.dataset_name,
            daily=args.daily,
            cgm_time_col=args.time_col,
            **common_kwargs,
        )
    if args.plot == "frequency":
        return plotters.plot_dataset_glucose_frequency(
            df,
            dataset=args.dataset_name,
            bins=args.bins,
            **common_kwargs,
        )
    return plotters.plot_dataset_mean_glucose(
        df,
        dataset=args.dataset_name,
        cgm_time_col=args.time_col,
        **common_kwargs,
    )


def main() -> None:
    args = _parse_args()
    if not args.show and not args.save_dir:
        raise ValueError("No output requested. Use --show and/or --save-dir.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    df = _load_input_df(args)
    print(f"[input_df] shape=({df.height}, {df.width})")

    figures = _build_plots(df, args)
    print(f"[plots] generated={len(figures)}")

    if args.save_dir:
        labels = _resolve_requested_labels(args)
        _save_figures(
            figures,
            output_dir=args.save_dir,
            level=args.level,
            plot_type=args.plot,
            labels=labels,
            dpi=args.dpi,
        )

    # Avoid leaving many open figures in non-interactive runs.
    if not args.show:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
