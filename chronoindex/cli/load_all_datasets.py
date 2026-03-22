"""Load all datasets declared in datasets.toml."""

from __future__ import annotations

import argparse
from pathlib import Path

from chronoindex.dataset_ingestion.dataset_unification import DEFAULT_UNIFIED_METADATA_COLUMNS
from chronoindex.dataset_loader import list_dataset_entries, load_datasets
from chronoindex.dataset_unifier import (
    UnifiedCGMDataset,
    UnifiedClinicalData,
    UnifiedFoodData,
    UnifiedPhysicalActivityData,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load dataset handlers from TOML config."
    )
    parser.add_argument(
        "--config",
        default="datasets.toml",
        help="Path to datasets TOML config (relative to project root or absolute).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional subset of dataset names to load.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include datasets marked as enabled=false in config.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Load datasets in parallel threads.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker threads for --parallel.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first loading error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved entries without loading data.",
    )
    parser.add_argument(
        "--save-unified",
        default=None,
        help="Optional output parquet path for a unified dataset.",
    )
    parser.add_argument(
        "--unified-type",
        choices=("cgm", "clinical", "food", "physical-activity"),
        default="cgm",
        help=(
            "Unified output kind for --save-unified: "
            "'cgm' (UnifiedCGMDataset), "
            "'clinical' (UnifiedClinicalData), "
            "'food' (UnifiedFoodData), "
            "'physical-activity' (UnifiedPhysicalActivityData)."
        ),
    )
    parser.add_argument(
        "--unified-metadata-columns",
        nargs="+",
        default=DEFAULT_UNIFIED_METADATA_COLUMNS,
        help=(
            "Metadata columns passed to UnifiedCGMDataset "
            "(used when --unified-type is cgm)."
        ),
    )
    return parser.parse_args()


def _print_summary(datasets: dict[str, object], errors: dict[str, str]) -> None:
    for name in sorted(datasets.keys()):
        ds = datasets[name]
        metadata_rows = getattr(getattr(ds, "metadata", None), "height", None)
        cgm_rows = getattr(getattr(ds, "cgm_data", None), "height", None)
        joined_rows = getattr(getattr(ds, "joined_data", None), "height", None)
        print(
            f"[ok] {name}: "
            f"metadata_rows={metadata_rows}, cgm_rows={cgm_rows}, joined_rows={joined_rows}"
        )

    for name in sorted(errors.keys()):
        print(f"[error] {name}: {errors[name]}")

    print(f"[summary] loaded={len(datasets)} failed={len(errors)}")


def _save_unified_parquet(
    output_path: str | Path,
    datasets: dict[str, object],
    unified_type: str,
    metadata_columns: list[str] | None = None,
    config_path: str | None = None,
) -> None:
    if not datasets:
        raise ValueError("No loaded datasets available to unify.")

    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    metadata_columns = metadata_columns or list(DEFAULT_UNIFIED_METADATA_COLUMNS)
    if unified_type == "cgm":
        unified = UnifiedCGMDataset(
            datasets,
            metadata_columns=metadata_columns,
            config_path=config_path or "datasets.toml",
        ).unified_data
    elif unified_type == "clinical":
        unified = UnifiedClinicalData(
            datasets,
            config_path=config_path or "datasets.toml",
        ).unified_data
    elif unified_type == "food":
        unified = UnifiedFoodData(
            datasets,
            config_path=config_path or "datasets.toml",
        ).unified_data
    elif unified_type == "physical-activity":
        unified = UnifiedPhysicalActivityData(
            datasets,
            config_path=config_path or "datasets.toml",
        ).unified_data
    else:
        raise ValueError(f"Unsupported unified type: {unified_type}")

    output.parent.mkdir(parents=True, exist_ok=True)
    unified.write_parquet(output)
    print(
        f"[saved] {output} ({unified.height} rows, {unified.width} columns) "
        f"[type={unified_type}]"
    )


def main() -> None:
    args = _parse_args()

    if args.dry_run:
        config_path, entries = list_dataset_entries(
            config_path=args.config,
            only=args.only,
            include_disabled=args.include_disabled,
        )
        print(f"[dry-run] config: {config_path}")
        for name in sorted(entries.keys()):
            entry = entries[name]
            print(f"[entry] {name}: {entry['handler']} kwargs={entry['kwargs']}")
        print(f"[summary] entries={len(entries)}")
        return

    datasets, errors = load_datasets(
        config_path=args.config,
        only=args.only,
        include_disabled=args.include_disabled,
        fail_fast=args.fail_fast,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    _print_summary(datasets, errors)

    if args.save_unified:
        _save_unified_parquet(
            args.save_unified,
            datasets,
            unified_type=args.unified_type,
            metadata_columns=args.unified_metadata_columns,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
