from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from typing import Any

import polars as pl

from chronoindex.dataset_harmonization import (
    HarmonizedBlocks,
    harmonize_loaded_aleppo,
    harmonize_loaded_colas,
    harmonize_loaded_praes,
    harmonize_loaded_zhao,
)
from chronoindex.dataset_loader import list_dataset_entries


Harmonizer = Callable[[Any], HarmonizedBlocks]

_ID_COLUMN_CANDIDATES = (
    "Id",
    "id",
    "participantId",
    "Patient Number",
    "PtID",
    "unique_ds_id",
)

_DEFAULT_HARMONIZERS: dict[str, Harmonizer] = {
    "aleppo": harmonize_loaded_aleppo,
    "colas": harmonize_loaded_colas,
    "zhao": harmonize_loaded_zhao,
    "chinese": harmonize_loaded_zhao,
    "praes": harmonize_loaded_praes,
}

_PHYSICAL_ACTIVITY_COLUMNS_TO_EXCLUDE = {"steps", "calories", "startDate"}


def _pick_id_column(frame: pl.DataFrame, dataset_name: str, block_name: str) -> str:
    for candidate in _ID_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not infer id column for dataset '{dataset_name}' in '{block_name}' block. "
        f"Tried: {list(_ID_COLUMN_CANDIDATES)}"
    )


def _resolve_harmonizer(
    dataset_name: str,
    dataset: Any,
    harmonizer_by_dataset: Mapping[str, Harmonizer] | None,
    config_harmonizer_by_dataset: Mapping[str, Harmonizer] | None = None,
) -> Harmonizer:
    if harmonizer_by_dataset and dataset_name in harmonizer_by_dataset:
        return harmonizer_by_dataset[dataset_name]
    if config_harmonizer_by_dataset and dataset_name in config_harmonizer_by_dataset:
        return config_harmonizer_by_dataset[dataset_name]

    lowered_name = dataset_name.lower()
    for key, harmonizer in _DEFAULT_HARMONIZERS.items():
        if key in lowered_name:
            return harmonizer

    class_fingerprint = f"{dataset.__class__.__module__}.{dataset.__class__.__name__}".lower()
    for key, harmonizer in _DEFAULT_HARMONIZERS.items():
        if key in class_fingerprint:
            return harmonizer

    raise ValueError(
        f"Unable to resolve harmonizer for dataset '{dataset_name}'. "
        "Set [datasets.<name>].harmonizer in datasets.toml or pass "
        "`harmonizer_by_dataset={'dataset_name': harmonizer_function}`."
    )


def _load_harmonizer_from_spec(harmonizer_spec: str, dataset_name: str) -> Harmonizer:
    if ":" not in harmonizer_spec:
        raise ValueError(
            f"Invalid harmonizer spec for dataset '{dataset_name}': '{harmonizer_spec}'. "
            "Expected 'module.path:function_name'."
        )

    module_path, function_name = harmonizer_spec.split(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, function_name):
        raise ValueError(
            f"Harmonizer function '{function_name}' not found in module '{module_path}' "
            f"for dataset '{dataset_name}'."
        )

    harmonizer = getattr(module, function_name)
    if not callable(harmonizer):
        raise ValueError(
            f"Harmonizer '{harmonizer_spec}' for dataset '{dataset_name}' is not callable."
        )
    return harmonizer


def _load_harmonizers_from_config(
    selected_names: Sequence[str],
    config_path: str | None,
) -> dict[str, Harmonizer]:
    if config_path is None:
        return {}

    try:
        _, entries = list_dataset_entries(
            config_path=config_path,
            include_disabled=True,
        )
    except FileNotFoundError:
        return {}

    configured: dict[str, Harmonizer] = {}
    for dataset_name in selected_names:
        entry = entries.get(dataset_name)
        if not entry:
            continue
        harmonizer_spec = entry.get("harmonizer")
        if isinstance(harmonizer_spec, str) and harmonizer_spec.strip():
            configured[dataset_name] = _load_harmonizer_from_spec(
                harmonizer_spec,
                dataset_name=dataset_name,
            )
    return configured


def _normalize_requested_columns(
    columns: Sequence[str] | str | None,
) -> tuple[bool, list[str]]:
    # None means "do not include this block".
    if columns is None:
        return False, []

    if isinstance(columns, str):
        if columns.strip().lower() == "all":
            return True, []
        return False, [columns]

    normalized = [column for column in columns if column]
    if len(normalized) == 1 and normalized[0].strip().lower() == "all":
        return True, []
    return False, normalized


def _resolve_selected_names(
    datasets: Mapping[str, Any],
    dataset_names: Sequence[str] | None,
) -> list[str]:
    selected_names = list(dataset_names) if dataset_names else list(datasets.keys())
    missing = [name for name in selected_names if name not in datasets]
    if missing:
        raise ValueError(f"Selected datasets are not loaded: {missing}")
    return selected_names


def _harmonize_selected_datasets(
    datasets: Mapping[str, Any],
    selected_names: Sequence[str],
    harmonizer_by_dataset: Mapping[str, Harmonizer] | None,
    config_path: str | None,
    use_toml_harmonizers: bool,
) -> dict[str, HarmonizedBlocks]:
    config_harmonizer_by_dataset = (
        _load_harmonizers_from_config(selected_names, config_path=config_path)
        if use_toml_harmonizers
        else {}
    )

    harmonized: dict[str, HarmonizedBlocks] = {}
    for dataset_name in selected_names:
        harmonizer = _resolve_harmonizer(
            dataset_name=dataset_name,
            dataset=datasets[dataset_name],
            harmonizer_by_dataset=harmonizer_by_dataset,
            config_harmonizer_by_dataset=config_harmonizer_by_dataset,
        )
        blocks = harmonizer(datasets[dataset_name])
        if not isinstance(blocks.CGMData, pl.DataFrame):
            raise ValueError(
                f"Harmonizer for '{dataset_name}' did not return a Polars CGMData block."
            )
        harmonized[dataset_name] = blocks
    return harmonized


def _standardize_id_column(
    frame: pl.DataFrame,
    dataset_name: str,
    block_name: str,
) -> pl.DataFrame:
    id_col = _pick_id_column(frame, dataset_name, block_name)
    if id_col == "Id":
        return frame.with_columns(pl.col("Id").cast(pl.Utf8, strict=False))
    return (
        frame.with_columns(pl.col(id_col).cast(pl.Utf8, strict=False).alias("Id"))
        .drop(id_col)
    )


def _ensure_columns(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    out = frame
    for column_name in columns:
        if column_name not in out.columns:
            out = out.with_columns(pl.lit(None).alias(column_name))
    return out


def _union_all_block_columns(
    block_by_dataset: Mapping[str, pl.DataFrame | None],
    block_name: str,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for dataset_name, frame in block_by_dataset.items():
        if frame is None:
            continue
        id_col = _pick_id_column(frame, dataset_name, block_name)
        for col_name in frame.columns:
            if col_name == id_col:
                continue
            if col_name in seen:
                continue
            ordered.append(col_name)
            seen.add(col_name)
    return ordered


def _empty_frame(columns: Sequence[str]) -> pl.DataFrame:
    schema = {col_name: pl.Null for col_name in columns}
    return pl.DataFrame(schema=schema)


def _normalize_food_frame_for_unification(
    frame: pl.DataFrame,
    dataset_name: str,
) -> pl.DataFrame:
    normalized = frame
    if "FoodTimepoint" not in normalized.columns:
        if "Timepoint" in normalized.columns:
            normalized = normalized.rename({"Timepoint": "FoodTimepoint"})
        elif "timepoint" in normalized.columns:
            normalized = normalized.rename({"timepoint": "FoodTimepoint"})
    id_col = _pick_id_column(normalized, dataset_name, "Food")
    sort_cols = [col for col in [id_col, "FoodTimepoint"] if col in normalized.columns]
    if len(sort_cols) > 1:
        normalized = normalized.sort(sort_cols)
    return normalized


def _normalize_physical_activity_frame_for_unification(
    frame: pl.DataFrame,
    dataset_name: str,
) -> pl.DataFrame:
    normalized = frame
    if "ActivityTimepoint" not in normalized.columns:
        if "activity_timepoint" in normalized.columns:
            normalized = normalized.rename({"activity_timepoint": "ActivityTimepoint"})
        elif "startDate" in normalized.columns and "startTime" in normalized.columns:
            normalized = normalized.with_columns(
                (
                    pl.concat_str(
                        [
                            pl.col("startDate").cast(pl.Utf8, strict=False),
                            pl.col("startTime").cast(pl.Utf8, strict=False),
                        ],
                        separator=" ",
                    )
                    .str.to_datetime(strict=False)
                    .alias("ActivityTimepoint")
                )
            )
        elif "activity_date" in normalized.columns:
            normalized = normalized.with_columns(
                pl.col("activity_date")
                .cast(pl.Datetime("us"), strict=False)
                .alias("ActivityTimepoint")
            )
    id_col = _pick_id_column(normalized, dataset_name, "PhysicalActivity")
    sort_cols = [col for col in [id_col, "ActivityTimepoint"] if col in normalized.columns]
    if len(sort_cols) > 1:
        normalized = normalized.sort(sort_cols)
    return normalized


class UnifiedCGMDataset:
    def __init__(
        self,
        datasets: Mapping[str, Any],
        metadata_columns: Sequence[str] | str | None = None,
        dataset_names: Sequence[str] | None = None,
        harmonizer_by_dataset: Mapping[str, Harmonizer] | None = None,
        config_path: str | None = "datasets.toml",
        use_toml_harmonizers: bool = True,
    ):
        if not datasets:
            raise ValueError("No datasets were provided.")

        selected_names = _resolve_selected_names(datasets, dataset_names)
        harmonized = _harmonize_selected_datasets(
            datasets=datasets,
            selected_names=selected_names,
            harmonizer_by_dataset=harmonizer_by_dataset,
            config_path=config_path,
            use_toml_harmonizers=use_toml_harmonizers,
        )

        metadata_by_dataset: dict[str, pl.DataFrame | None] = {
            name: blocks.MetaData for name, blocks in harmonized.items()
        }
        use_all_metadata, requested_metadata_columns = _normalize_requested_columns(
            metadata_columns
        )
        if use_all_metadata:
            target_metadata_columns = _union_all_block_columns(
                metadata_by_dataset,
                block_name="MetaData",
            )
        else:
            target_metadata_columns = list(requested_metadata_columns)

        self.dataset_names = selected_names
        self.harmonized = harmonized
        self.metadata_columns = target_metadata_columns

        frames: list[pl.DataFrame] = []
        for dataset_name in selected_names:
            cgm = _standardize_id_column(
                harmonized[dataset_name].CGMData,
                dataset_name=dataset_name,
                block_name="CGMData",
            )

            if target_metadata_columns:
                metadata = metadata_by_dataset[dataset_name]
                if metadata is None:
                    cgm = cgm.with_columns(
                        [pl.lit(None).alias(col) for col in target_metadata_columns]
                    )
                else:
                    metadata_std = _standardize_id_column(
                        metadata,
                        dataset_name=dataset_name,
                        block_name="MetaData",
                    )
                    metadata_std = _ensure_columns(metadata_std, target_metadata_columns)
                    cgm = cgm.join(
                        metadata_std.select(["Id", *target_metadata_columns]),
                        on="Id",
                        how="left",
                    )

            cgm = cgm.with_columns(pl.lit(dataset_name).alias("dataset"))
            frames.append(cgm)

        self.unified_data = pl.concat(frames, how="diagonal_relaxed")
        self.cgm_data = self.unified_data


class UnifiedClinicalData:
    def __init__(
        self,
        datasets: Mapping[str, Any],
        lab_columns: Sequence[str] | str | None = "all",
        visit_timepoint_columns: Sequence[str] | str | None = "all",
        dataset_names: Sequence[str] | None = None,
        harmonizer_by_dataset: Mapping[str, Harmonizer] | None = None,
        config_path: str | None = "datasets.toml",
        use_toml_harmonizers: bool = True,
    ):
        if not datasets:
            raise ValueError("No datasets were provided.")

        selected_names = _resolve_selected_names(datasets, dataset_names)
        harmonized = _harmonize_selected_datasets(
            datasets=datasets,
            selected_names=selected_names,
            harmonizer_by_dataset=harmonizer_by_dataset,
            config_path=config_path,
            use_toml_harmonizers=use_toml_harmonizers,
        )

        lab_by_dataset: dict[str, pl.DataFrame | None] = {
            name: blocks.LabData for name, blocks in harmonized.items()
        }
        visit_by_dataset: dict[str, pl.DataFrame | None] = {
            name: blocks.VisitTimepoints for name, blocks in harmonized.items()
        }

        use_all_lab, requested_lab_columns = _normalize_requested_columns(lab_columns)
        if use_all_lab:
            target_lab_columns = _union_all_block_columns(lab_by_dataset, block_name="LabData")
        else:
            target_lab_columns = list(requested_lab_columns)

        has_any_visit_block = any(
            isinstance(frame, pl.DataFrame) and not frame.is_empty()
            for frame in visit_by_dataset.values()
        )
        use_all_visit, requested_visit_columns = _normalize_requested_columns(
            visit_timepoint_columns
        )
        if has_any_visit_block:
            if use_all_visit:
                raw_visit_columns = _union_all_block_columns(
                    visit_by_dataset,
                    block_name="VisitTimepoints",
                )
            else:
                raw_visit_columns = list(requested_visit_columns)
        else:
            raw_visit_columns = []

        visit_columns_for_join = [
            col_name for col_name in raw_visit_columns if col_name not in {"Id", "visit"}
        ]

        self.dataset_names = selected_names
        self.harmonized = harmonized
        self.lab_columns = target_lab_columns
        self.visit_timepoint_columns = visit_columns_for_join

        frames: list[pl.DataFrame] = []
        for dataset_name in selected_names:
            lab = lab_by_dataset[dataset_name]
            if lab is None or lab.is_empty():
                continue

            base = _standardize_id_column(lab, dataset_name, "LabData")

            if visit_columns_for_join:
                visit = visit_by_dataset[dataset_name]
                if (
                    isinstance(visit, pl.DataFrame)
                    and not visit.is_empty()
                    and "visit" in base.columns
                    and "visit" in visit.columns
                ):
                    visit_std = _standardize_id_column(visit, dataset_name, "VisitTimepoints")
                    visit_std = _ensure_columns(visit_std, visit_columns_for_join)
                    visit_std = visit_std.select(["Id", "visit", *visit_columns_for_join])

                    # Protect against accidental row explosion when visit block contains duplicates.
                    visit_std = visit_std.with_row_index("_row_idx").group_by(
                        ["Id", "visit"], maintain_order=True
                    ).agg(
                        [
                            pl.col(col_name).sort_by("_row_idx").first().alias(col_name)
                            for col_name in visit_columns_for_join
                        ]
                    )
                    base = base.join(visit_std, on=["Id", "visit"], how="left")
                else:
                    base = base.with_columns(
                        [pl.lit(None).alias(col_name) for col_name in visit_columns_for_join]
                    )

            selected_columns = ["Id", *target_lab_columns, *visit_columns_for_join]
            base = _ensure_columns(base, selected_columns)
            base = base.select(selected_columns)
            base = base.with_columns(pl.lit(dataset_name).alias("dataset"))
            frames.append(base)

        output_columns = ["Id", *target_lab_columns, *visit_columns_for_join, "dataset"]
        if not frames:
            self.unified_data = _empty_frame(output_columns)
        else:
            self.unified_data = pl.concat(frames, how="diagonal_relaxed")
        self.clinical_data = self.unified_data


class UnifiedFoodData:
    def __init__(
        self,
        datasets: Mapping[str, Any],
        food_columns: Sequence[str] | str | None = "all",
        dataset_names: Sequence[str] | None = None,
        harmonizer_by_dataset: Mapping[str, Harmonizer] | None = None,
        config_path: str | None = "datasets.toml",
        use_toml_harmonizers: bool = True,
    ):
        if not datasets:
            raise ValueError("No datasets were provided.")

        selected_names = _resolve_selected_names(datasets, dataset_names)
        harmonized = _harmonize_selected_datasets(
            datasets=datasets,
            selected_names=selected_names,
            harmonizer_by_dataset=harmonizer_by_dataset,
            config_path=config_path,
            use_toml_harmonizers=use_toml_harmonizers,
        )

        food_by_dataset: dict[str, pl.DataFrame | None] = {
            name: (
                _normalize_food_frame_for_unification(blocks.Food, name)
                if isinstance(blocks.Food, pl.DataFrame)
                else None
            )
            for name, blocks in harmonized.items()
        }

        use_all_food, requested_food_columns = _normalize_requested_columns(food_columns)
        if use_all_food:
            target_food_columns = _union_all_block_columns(food_by_dataset, block_name="Food")
        else:
            target_food_columns = list(requested_food_columns)

        self.dataset_names = selected_names
        self.harmonized = harmonized
        self.food_columns = target_food_columns

        self.unified_data = self._build_event_block(
            block_by_dataset=food_by_dataset,
            selected_names=selected_names,
            block_name="Food",
            target_columns=target_food_columns,
            primary_sort_columns=("FoodTimepoint",),
        )
        self.food_data = self.unified_data

    @staticmethod
    def _build_event_block(
        block_by_dataset: Mapping[str, pl.DataFrame | None],
        selected_names: Sequence[str],
        block_name: str,
        target_columns: list[str],
        primary_sort_columns: Sequence[str],
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for dataset_name in selected_names:
            frame = block_by_dataset.get(dataset_name)
            if frame is None or frame.is_empty():
                continue

            standardized = _standardize_id_column(frame, dataset_name, block_name)
            standardized = _ensure_columns(standardized, target_columns)
            out = standardized.select(["Id", *target_columns]).with_columns(
                pl.lit(dataset_name).alias("dataset")
            )
            frames.append(out)

        output_columns = ["Id", *target_columns, "dataset"]
        if not frames:
            return _empty_frame(output_columns)

        unified = pl.concat(frames, how="diagonal_relaxed")
        sort_columns = [
            "dataset",
            "Id",
            *[col_name for col_name in primary_sort_columns if col_name in unified.columns],
        ]
        if len(sort_columns) > 2:
            unified = unified.sort(sort_columns)
        return unified


class UnifiedPhysicalActivityData:
    def __init__(
        self,
        datasets: Mapping[str, Any],
        physical_activity_columns: Sequence[str] | str | None = "all",
        dataset_names: Sequence[str] | None = None,
        harmonizer_by_dataset: Mapping[str, Harmonizer] | None = None,
        config_path: str | None = "datasets.toml",
        use_toml_harmonizers: bool = True,
    ):
        if not datasets:
            raise ValueError("No datasets were provided.")

        selected_names = _resolve_selected_names(datasets, dataset_names)
        harmonized = _harmonize_selected_datasets(
            datasets=datasets,
            selected_names=selected_names,
            harmonizer_by_dataset=harmonizer_by_dataset,
            config_path=config_path,
            use_toml_harmonizers=use_toml_harmonizers,
        )

        activity_by_dataset: dict[str, pl.DataFrame | None] = {
            name: (
                _normalize_physical_activity_frame_for_unification(
                    blocks.PhysicalActivity,
                    name,
                )
                if isinstance(blocks.PhysicalActivity, pl.DataFrame)
                else None
            )
            for name, blocks in harmonized.items()
        }

        use_all_activity, requested_activity_columns = _normalize_requested_columns(
            physical_activity_columns
        )
        if use_all_activity:
            target_activity_columns = _union_all_block_columns(
                activity_by_dataset,
                block_name="PhysicalActivity",
            )
        else:
            target_activity_columns = list(requested_activity_columns)

        target_activity_columns = [
            col_name
            for col_name in target_activity_columns
            if col_name not in _PHYSICAL_ACTIVITY_COLUMNS_TO_EXCLUDE
        ]

        self.dataset_names = selected_names
        self.harmonized = harmonized
        self.physical_activity_columns = target_activity_columns

        self.unified_data = UnifiedFoodData._build_event_block(
            block_by_dataset=activity_by_dataset,
            selected_names=selected_names,
            block_name="PhysicalActivity",
            target_columns=target_activity_columns,
            primary_sort_columns=("ActivityTimepoint", "startDate", "startTime", "activity_date"),
        )
        self.physical_activity_data = self.unified_data
