from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

import polars as pl


DEFAULT_UNIFIED_METADATA_COLUMNS = ["Age", "Type of Diabetes", "BMI", "HbA1c"]
_TIME_NORMALIZATION_BASE_DATE = date(2000, 1, 1)

_COMMON_ID_COLUMN_CANDIDATES = (
    "unique_ds_id",
    "id",
    "PtID",
    "Patient Number",
    "participantId",
    "subject_id",
    "subjectId",
    "patient_id",
)
_COMMON_GLUCOSE_COLUMN_CANDIDATES = (
    "CGM",
    "glucemia_series",
    "glucose_series",
    "glucose",
)
_COMMON_TIME_COLUMN_CANDIDATES = (
    "time_series",
    "deviceTimestamp",
    "Date",
    "timestamp_series",
)
_COMMON_DAY_OFFSET_COLUMN_CANDIDATES = ("days_from_enroll", "days_from_enrollment")

_EXPECTED_DTYPE_BY_COLUMN = {
    "time_series": pl.List(pl.Datetime("us")),
    "CGM": pl.List(pl.Float64),
    "Age": pl.Float64,
    "BMI": pl.Float64,
    "HbA1c": pl.Float64,
    "Type of Diabetes": pl.Utf8,
}


def _dtype_equals(left: pl.DataType, right: pl.DataType) -> bool:
    return str(left) == str(right)


def _normalize_metadata_columns(columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return list(DEFAULT_UNIFIED_METADATA_COLUMNS)
    normalized: list[str] = []
    for col_name in columns:
        normalized.append("Age" if col_name == "age" else col_name)
    return normalized


def _to_datetime_list(
    time_values: list[Any] | None,
    day_values: list[Any] | None = None,
) -> list[datetime | None] | None:
    if time_values is None:
        return None
    if day_values is None:
        return [
            datetime.combine(_TIME_NORMALIZATION_BASE_DATE, value)
            if value is not None
            else None
            for value in time_values
        ]

    result: list[datetime | None] = []
    for idx, value in enumerate(time_values):
        day = day_values[idx] if idx < len(day_values) else None
        if value is None or day is None:
            result.append(None)
            continue
        result.append(
            datetime.combine(_TIME_NORMALIZATION_BASE_DATE + timedelta(days=int(day)), value)
        )
    return result


def _pick_single_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    label: str,
    required: bool = False,
) -> str | None:
    found = [col for col in candidates if col in columns]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous {label} columns found {found}. Pass explicit column mapping."
        )
    if required:
        raise ValueError(
            f"Missing {label} column. Tried candidates: {list(candidates)}"
        )
    return None


def infer_id_column(
    df: pl.DataFrame,
    preferred_id_column: str | None = None,
    strict: bool = True,
) -> str:
    if preferred_id_column:
        if preferred_id_column not in df.columns:
            raise ValueError(
                f"Preferred id column '{preferred_id_column}' not found in {df.columns}"
            )
        return preferred_id_column

    by_candidate = [col for col in _COMMON_ID_COLUMN_CANDIDATES if col in df.columns]
    if len(by_candidate) == 1:
        return by_candidate[0]
    if len(by_candidate) > 1:
        if strict:
            raise ValueError(
                f"Ambiguous id columns {by_candidate}. Pass explicit id_column_by_dataset."
            )
        return by_candidate[0]

    heuristic = [
        col
        for col in df.columns
        if col.lower() == "id" or col.lower().endswith("id") or "patient" in col.lower()
    ]
    if len(heuristic) == 1:
        return heuristic[0]
    if len(heuristic) > 1 and not strict:
        return heuristic[0]

    raise ValueError(
        "Could not infer a unique id column. Pass explicit id_column_by_dataset."
    )


def normalize_for_unifier(
    df: pl.DataFrame,
    dataset_name: str | None = None,
    metadata_columns: list[str] | None = None,
    id_column: str | None = None,
    glucose_column: str | None = None,
    time_column: str | None = None,
    day_offset_column: str | None = None,
) -> pl.DataFrame:
    normalized = df
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    display_name = dataset_name or "dataset"

    # Standardize age naming across handlers/adapters.
    if "Age" not in normalized.columns and "age" in normalized.columns:
        normalized = normalized.rename({"age": "Age"})

    glucose_col = glucose_column
    if glucose_col is None:
        glucose_col = _pick_single_column(
            normalized.columns,
            _COMMON_GLUCOSE_COLUMN_CANDIDATES,
            label=f"glucose for {display_name}",
            required=True,
        )
    if glucose_col != "CGM":
        normalized = normalized.rename({glucose_col: "CGM"})

    time_col = time_column
    if time_col is None:
        time_col = _pick_single_column(
            normalized.columns,
            _COMMON_TIME_COLUMN_CANDIDATES,
            label=f"time for {display_name}",
            required=True,
        )
    if time_col != "time_series":
        normalized = normalized.rename({time_col: "time_series"})

    ts_dtype = normalized.schema.get("time_series")
    if ts_dtype is not None and _dtype_equals(ts_dtype, pl.List(pl.Time)):
        day_col = day_offset_column
        if day_col is None:
            day_col = _pick_single_column(
                normalized.columns,
                _COMMON_DAY_OFFSET_COLUMN_CANDIDATES,
                label=f"day-offset for {display_name}",
                required=False,
            )
        if day_col:
            normalized = normalized.with_columns(
                pl.struct(["time_series", day_col])
                .map_elements(
                    lambda row: _to_datetime_list(row["time_series"], row[day_col] or []),
                    return_dtype=pl.List(pl.Datetime("us")),
                )
                .alias("time_series")
            )
        else:
            normalized = normalized.with_columns(
                pl.col("time_series")
                .map_elements(
                    lambda values: _to_datetime_list(values),
                    return_dtype=pl.List(pl.Datetime("us")),
                )
                .alias("time_series")
            )

    if "time_series" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("time_series").cast(pl.List(pl.Datetime("us")), strict=False)
        )
    if "CGM" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("CGM").cast(pl.List(pl.Float64), strict=False)
        )

    for column_name in metadata_columns:
        expected_dtype = _EXPECTED_DTYPE_BY_COLUMN.get(column_name)
        if expected_dtype is None or column_name not in normalized.columns:
            continue
        normalized = normalized.with_columns(
            pl.col(column_name).cast(expected_dtype, strict=False)
        )

    if id_column and id_column in normalized.columns:
        normalized = normalized.with_columns(
            pl.col(id_column).cast(pl.Utf8, strict=False)
        )

    return normalized


def preflight_unifier_inputs(
    joined_frames: Mapping[str, pl.DataFrame],
    id_column_by_dataset: Mapping[str, str],
    metadata_columns: list[str] | None = None,
    strict_dtypes: bool = True,
    dtype_expectations: Mapping[str, pl.DataType] | None = None,
) -> None:
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    dtype_expectations = dict(_EXPECTED_DTYPE_BY_COLUMN) | dict(dtype_expectations or {})
    errors: list[str] = []

    if not joined_frames:
        raise ValueError("Unification preflight failed:\n- No datasets provided.")

    for dataset_name in joined_frames.keys():
        if dataset_name not in id_column_by_dataset:
            errors.append(
                f"{dataset_name}: missing id-column mapping in id_column_by_dataset."
            )

    for dataset_name, frame in joined_frames.items():
        if dataset_name not in id_column_by_dataset:
            continue
        id_column = id_column_by_dataset[dataset_name]
        required_columns = ["time_series", "CGM", id_column]
        missing_columns = [col for col in required_columns if col not in frame.columns]
        if missing_columns:
            errors.append(f"{dataset_name}: missing columns {missing_columns}")
            continue

        if not strict_dtypes:
            continue

        for column_name in [*required_columns, *metadata_columns]:
            if column_name not in frame.columns:
                continue
            expected_dtype = dtype_expectations.get(column_name)
            if expected_dtype is None:
                continue
            current_dtype = frame.schema.get(column_name)
            if current_dtype is None:
                continue
            if not _dtype_equals(current_dtype, expected_dtype):
                errors.append(
                    f"{dataset_name}: column '{column_name}' has dtype {current_dtype}, "
                    f"expected {expected_dtype}"
                )

    if strict_dtypes:
        for column_name in metadata_columns:
            if column_name in dtype_expectations:
                continue
            present = {
                dataset_name: frame.schema[column_name]
                for dataset_name, frame in joined_frames.items()
                if column_name in frame.columns
            }
            unique_types = {str(dtype) for dtype in present.values()}
            if len(unique_types) > 1:
                type_text = ", ".join(
                    f"{dataset_name}={dtype}" for dataset_name, dtype in present.items()
                )
                errors.append(
                    f"Column '{column_name}' has inconsistent dtypes across datasets: {type_text}"
                )

    if errors:
        raise ValueError(
            "Unification preflight failed:\n- "
            + "\n- ".join(errors)
            + "\nHint: run normalize_for_unifier(...) and/or provide explicit id/column mappings."
        )


def _resolve_metadata_target_dtypes(
    joined_frames: Mapping[str, pl.DataFrame],
    metadata_columns: list[str],
    strict_dtypes: bool = True,
    dtype_expectations: Mapping[str, pl.DataType] | None = None,
) -> dict[str, pl.DataType]:
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    dtype_expectations = dict(_EXPECTED_DTYPE_BY_COLUMN) | dict(dtype_expectations or {})
    targets: dict[str, pl.DataType] = {}

    for column_name in metadata_columns:
        if column_name in dtype_expectations:
            targets[column_name] = dtype_expectations[column_name]
            continue

        found_dtypes: list[pl.DataType] = []
        seen: set[str] = set()
        for frame in joined_frames.values():
            if column_name not in frame.columns:
                continue
            dtype = frame.schema[column_name]
            key = str(dtype)
            if key in seen or key == "Null":
                continue
            seen.add(key)
            found_dtypes.append(dtype)

        if not found_dtypes:
            targets[column_name] = pl.Null
        elif len(found_dtypes) == 1:
            targets[column_name] = found_dtypes[0]
        else:
            dtype_list = ", ".join(str(dtype) for dtype in found_dtypes)
            if strict_dtypes:
                raise ValueError(
                    f"Unification preflight failed:\n- Column '{column_name}' has inconsistent "
                    f"dtypes across datasets: {dtype_list}"
                )
            targets[column_name] = found_dtypes[0]

    return targets


def _align_metadata_columns_for_unification(
    joined_frames: Mapping[str, pl.DataFrame],
    metadata_columns: list[str],
    strict_dtypes: bool = True,
    dtype_expectations: Mapping[str, pl.DataType] | None = None,
) -> dict[str, pl.DataFrame]:
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    target_dtypes = _resolve_metadata_target_dtypes(
        joined_frames,
        metadata_columns=metadata_columns,
        strict_dtypes=strict_dtypes,
        dtype_expectations=dtype_expectations,
    )
    aligned: dict[str, pl.DataFrame] = {}
    for dataset_name, frame in joined_frames.items():
        exprs = []
        for column_name in metadata_columns:
            target_dtype = target_dtypes[column_name]
            if column_name in frame.columns:
                exprs.append(
                    pl.col(column_name).cast(target_dtype, strict=False).alias(column_name)
                )
            else:
                exprs.append(pl.lit(None).cast(target_dtype).alias(column_name))
        aligned[dataset_name] = frame.with_columns(exprs) if exprs else frame
    return aligned


class _JoinedFrameAdapter:
    def __init__(self, joined_data: pl.DataFrame):
        self.joined_data = joined_data


def _extract_joined_data_frame(dataset: Any, dataset_name: str) -> pl.DataFrame:
    if isinstance(dataset, pl.DataFrame):
        return dataset
    joined = getattr(dataset, "joined_data", None)
    if not isinstance(joined, pl.DataFrame):
        raise ValueError(
            f"Dataset '{dataset_name}' does not expose 'joined_data' as a Polars DataFrame."
        )
    return joined


def _standardize_for_concat(
    frame: pl.DataFrame,
    dataset_name: str,
    id_column: str,
    metadata_columns: list[str],
) -> pl.DataFrame:
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    columns = ["time_series", "CGM", id_column, *metadata_columns]
    standardized = frame.select(columns)
    standardized = standardized.with_columns(
        pl.col(id_column).cast(pl.Utf8, strict=False).alias("unique_ds_id")
    ).drop(id_column)
    standardized = standardized.with_columns(pl.lit(dataset_name).alias("dataset_name"))
    return standardized


def unify_loaded_datasets(
    datasets: Mapping[str, Any],
    metadata_columns: list[str] | None = None,
    dataset_names: Sequence[str] | None = None,
    id_column_by_dataset: Mapping[str, str] | None = None,
    strict_preflight: bool = True,
    strict_id_inference: bool = True,
) -> pl.DataFrame:
    metadata_columns = _normalize_metadata_columns(metadata_columns)
    selected_names = list(dataset_names) if dataset_names else list(datasets.keys())
    if not selected_names:
        raise ValueError("No datasets selected for unification.")

    missing = [name for name in selected_names if name not in datasets]
    if missing:
        raise ValueError(f"Selected datasets not loaded: {missing}")

    provided_id_map = dict(id_column_by_dataset or {})
    normalized_frames: dict[str, pl.DataFrame] = {}
    resolved_id_map: dict[str, str] = {}

    for dataset_name in selected_names:
        raw_frame = _extract_joined_data_frame(datasets[dataset_name], dataset_name)
        id_column = infer_id_column(
            raw_frame,
            preferred_id_column=provided_id_map.get(dataset_name),
            strict=strict_id_inference,
        )
        normalized = normalize_for_unifier(
            raw_frame,
            dataset_name=dataset_name,
            metadata_columns=metadata_columns,
            id_column=id_column,
        )
        normalized_frames[dataset_name] = normalized
        resolved_id_map[dataset_name] = id_column

    normalized_frames = _align_metadata_columns_for_unification(
        normalized_frames,
        metadata_columns=metadata_columns,
        strict_dtypes=strict_preflight,
    )

    preflight_unifier_inputs(
        normalized_frames,
        id_column_by_dataset=resolved_id_map,
        metadata_columns=metadata_columns,
        strict_dtypes=strict_preflight,
    )

    standardized_frames = [
        _standardize_for_concat(
            normalized_frames[dataset_name],
            dataset_name=dataset_name,
            id_column=resolved_id_map[dataset_name],
            metadata_columns=metadata_columns,
        )
        for dataset_name in selected_names
    ]
    return pl.concat(standardized_frames, how="vertical")


class UnifiedCGMDataset:
    def __init__(
        self,
        *datasets: Any,
        metadata_columns: list[str] | None = None,
        dataset_names: Sequence[str] | None = None,
        id_column_by_dataset: Mapping[str, str] | None = None,
        strict_preflight: bool = True,
    ):
        if metadata_columns is None:
            if datasets and isinstance(datasets[-1], list):
                metadata_columns = datasets[-1]
                datasets = datasets[:-1]
            else:
                metadata_columns = list(DEFAULT_UNIFIED_METADATA_COLUMNS)
        metadata_columns = _normalize_metadata_columns(metadata_columns)

        self.metadata_columns = metadata_columns
        self.unified_data: pl.DataFrame | None = None

        if len(datasets) == 1 and isinstance(datasets[0], Mapping):
            dataset_map = datasets[0]
            selected_names = list(dataset_names) if dataset_names else list(dataset_map.keys())
            self.unified_data = unify_loaded_datasets(
                dataset_map,
                metadata_columns=metadata_columns,
                dataset_names=selected_names,
                id_column_by_dataset=id_column_by_dataset,
                strict_preflight=strict_preflight,
            )
            return

        selected_names = (
            list(dataset_names)
            if dataset_names
            else [f"dataset_{idx + 1}" for idx in range(len(datasets))]
        )
        if len(selected_names) != len(datasets):
            raise ValueError(
                "dataset_names length must match number of positional dataset arguments."
            )

        dataset_map: dict[str, Any] = {}
        for dataset_name, dataset in zip(selected_names, datasets):
            if isinstance(dataset, pl.DataFrame):
                dataset_map[dataset_name] = _JoinedFrameAdapter(dataset)
            else:
                dataset_map[dataset_name] = dataset

        self.unified_data = unify_loaded_datasets(
            dataset_map,
            metadata_columns=metadata_columns,
            dataset_names=selected_names,
            id_column_by_dataset=id_column_by_dataset,
            strict_preflight=strict_preflight,
        )
