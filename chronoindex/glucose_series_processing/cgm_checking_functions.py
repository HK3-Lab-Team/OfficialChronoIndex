from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import polars as pl


def _is_list_dtype(dtype: pl.DataType) -> bool:
    return str(dtype).lower().startswith("list")


def _resolve_group_keys(
    df: pl.DataFrame,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
) -> list[str]:
    if patient_identifier_col not in df.columns:
        raise ValueError(
            f"Column '{patient_identifier_col}' not found in dataframe columns: {df.columns}"
        )

    keys = [patient_identifier_col]
    if dataset_col in df.columns:
        keys = [dataset_col, patient_identifier_col]
    return keys


def _to_recording_level(
    df: pl.DataFrame,
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    if cgm_col not in df.columns or time_col not in df.columns:
        raise ValueError(
            f"Expected columns '{cgm_col}' and '{time_col}' in dataframe."
        )

    cgm_is_list = _is_list_dtype(df.schema[cgm_col])
    time_is_list = _is_list_dtype(df.schema[time_col])

    if cgm_is_list and time_is_list:
        out = df.explode(cgm_col, time_col)
    elif cgm_is_list != time_is_list:
        raise ValueError(
            f"Columns '{cgm_col}' and '{time_col}' must both be list-like or both scalar."
        )
    else:
        out = df

    return out.with_columns(
        [
            pl.col(cgm_col).cast(pl.Float64, strict=False).alias(cgm_col),
            pl.col(time_col).cast(pl.Datetime("us"), strict=False).alias(time_col),
        ]
    )


def _print_check_result(
    *,
    found_message: str,
    not_found_message: str,
    frame: pl.DataFrame,
    display_columns: list[str] | None = None,
) -> None:
    if frame.is_empty():
        print(not_found_message)
        return
    print(found_message)
    if display_columns:
        print(frame.select(display_columns))
    else:
        print(frame)


def check_duplicate_cgms(
    df: pl.DataFrame,
    *,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    """
    Check duplicate CGM readings (same glucose and same CGMTime) per subject.
    Returns one row per (dataset, subject) with duplicate summary counts.
    """
    records = _to_recording_level(df, cgm_col=cgm_col, time_col=time_col)
    group_keys = _resolve_group_keys(
        records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
    )

    duplicates = (
        records
        .filter(pl.col(cgm_col).is_not_null() & pl.col(time_col).is_not_null())
        .group_by([*group_keys, cgm_col, time_col])
        .len()
        .filter(pl.col("len") > 1)
        .group_by(group_keys)
        .agg(
            [
                pl.sum("len").alias("duplicate_rows"),
                pl.len().alias("duplicate_pairs"),
            ]
        )
        .sort(group_keys)
    )

    _print_check_result(
        found_message="Duplicate CGM readings found for the following subjects:",
        not_found_message="No duplicate CGM readings found.",
        frame=duplicates,
        display_columns=[*group_keys, "duplicate_rows", "duplicate_pairs"],
    )
    return duplicates


def check_null_values(
    df: pl.DataFrame,
    *,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    """
    Check null glucose or null timestamp values per subject.
    Returns one row per (dataset, subject) with null counts.
    """
    records = _to_recording_level(df, cgm_col=cgm_col, time_col=time_col)
    group_keys = _resolve_group_keys(
        records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
    )

    nulls = (
        records
        .filter(pl.col(cgm_col).is_null() | pl.col(time_col).is_null())
        .group_by(group_keys)
        .agg(
            [
                pl.col(cgm_col).is_null().sum().alias("null_glucose_values"),
                pl.col(time_col).is_null().sum().alias("null_time_values"),
                pl.len().alias("rows_with_any_null"),
            ]
        )
        .sort(group_keys)
    )

    _print_check_result(
        found_message="Null glucose/timestamp values found for the following subjects:",
        not_found_message="No null glucose/timestamp values found.",
        frame=nulls,
        display_columns=[
            *group_keys,
            "null_glucose_values",
            "null_time_values",
            "rows_with_any_null",
        ],
    )
    return nulls


def check_few_days(
    df: pl.DataFrame,
    min_unique_days: int,
    *,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    """
    Check subjects with fewer than `min_unique_days` unique recording days.
    """
    if min_unique_days < 1:
        raise ValueError("min_unique_days must be >= 1.")

    records = _to_recording_level(df, cgm_col=cgm_col, time_col=time_col)
    group_keys = _resolve_group_keys(
        records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
    )

    all_subjects = records.select(group_keys).unique()
    days_per_subject = (
        records
        .filter(pl.col(time_col).is_not_null())
        .group_by(group_keys)
        .agg(pl.col(time_col).dt.date().n_unique().alias("unique_days"))
    )

    few_days = (
        all_subjects
        .join(days_per_subject, on=group_keys, how="left")
        .with_columns(pl.col("unique_days").fill_null(0).cast(pl.Int64))
        .filter(pl.col("unique_days") < min_unique_days)
        .sort(group_keys)
    )

    _print_check_result(
        found_message=(
            "Subjects below the minimum unique recording days threshold found:"
        ),
        not_found_message=(
            f"All subjects have at least {min_unique_days} unique recording days."
        ),
        frame=few_days,
        display_columns=[*group_keys, "unique_days"],
    )
    return few_days


def check_few_cgm_records(
    df: pl.DataFrame,
    min_records: int,
    *,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    """
    Check subjects with fewer than `min_records` valid CGM readings.
    A valid reading has non-null CGM and non-null CGMTime.
    """
    if min_records < 1:
        raise ValueError("min_records must be >= 1.")

    records = _to_recording_level(df, cgm_col=cgm_col, time_col=time_col)
    group_keys = _resolve_group_keys(
        records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
    )

    all_subjects = records.select(group_keys).unique()
    counts = (
        records
        .filter(pl.col(cgm_col).is_not_null() & pl.col(time_col).is_not_null())
        .group_by(group_keys)
        .len()
        .rename({"len": "n_records"})
    )

    few_records = (
        all_subjects
        .join(counts, on=group_keys, how="left")
        .with_columns(pl.col("n_records").fill_null(0).cast(pl.Int64))
        .filter(pl.col("n_records") < min_records)
        .sort(group_keys)
    )

    _print_check_result(
        found_message="Subjects below the minimum CGM readings threshold found:",
        not_found_message=(
            f"All subjects have at least {min_records} valid CGM readings."
        ),
        frame=few_records,
        display_columns=[*group_keys, "n_records"],
    )
    return few_records


def _non_consecutive_day_pairs(days: Sequence[date] | None) -> list[str]:
    if days is None:
        return []
    ordered = sorted(days)
    if len(ordered) < 2:
        return []
    out: list[str] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        if (right - left).days > 1:
            out.append(f"{left.isoformat()} -> {right.isoformat()}")
    return out


def check_non_consecutive_days(
    df: pl.DataFrame,
    *,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
) -> pl.DataFrame:
    """
    For each subject, detect gaps between recorded days.
    Returns one row per subject with all non-consecutive day pairs.
    """
    records = _to_recording_level(df, cgm_col=cgm_col, time_col=time_col)
    group_keys = _resolve_group_keys(
        records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
    )

    gaps = (
        records
        .filter(pl.col(time_col).is_not_null())
        .with_columns(pl.col(time_col).dt.date().alias("_record_day"))
        .group_by(group_keys)
        .agg(pl.col("_record_day").drop_nulls().unique().sort())
        .with_columns(
            pl.col("_record_day")
            .map_elements(_non_consecutive_day_pairs, return_dtype=pl.List(pl.Utf8))
            .alias("non_consecutive_day_pairs")
        )
        .drop("_record_day")
        .filter(pl.col("non_consecutive_day_pairs").list.len() > 0)
        .sort(group_keys)
    )

    _print_check_result(
        found_message="Non-consecutive recording days found for the following subjects:",
        not_found_message="All subjects have consecutive recording days.",
        frame=gaps,
        display_columns=[*group_keys, "non_consecutive_day_pairs"],
    )
    return gaps


def make_consecutive_day_checker(**kwargs) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """
    Factory wrapper for convenience, mirroring old naming style.
    Returns a checker function configured with keyword options.
    """
    return lambda df: check_non_consecutive_days(df, **kwargs)


def run_all_unified_checks(
    df: pl.DataFrame,
    *,
    min_unique_days: int,
    min_records: int,
    patient_identifier_col: str = "Id",
    dataset_col: str = "dataset",
    cgm_col: str = "CGM",
    time_col: str = "CGMTime",
    verbose: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Run all unified-CGM quality checks and return their outputs.

    Keys in returned dict:
    - duplicate_cgms
    - null_values
    - few_days
    - few_cgm_records
    - non_consecutive_days
    """
    if verbose:
        grouping = (
            f"'{dataset_col} + {patient_identifier_col}'"
            if dataset_col in df.columns
            else f"'{patient_identifier_col}'"
        )
        print("Running unified CGM checks")
        print(f"- Input shape: rows={df.height}, cols={df.width}")
        print(f"- Grouping key: {grouping}")
        print(f"- Thresholds: min_unique_days={min_unique_days}, min_records={min_records}")
        print("[1/5] Checking duplicate CGM readings...")

    duplicate_cgms = check_duplicate_cgms(
        df,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
        cgm_col=cgm_col,
        time_col=time_col,
    )

    if verbose:
        print("[2/5] Checking null glucose/timestamp values...")

    null_values = check_null_values(
        df,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
        cgm_col=cgm_col,
        time_col=time_col,
    )

    if verbose:
        print("[3/5] Checking minimum unique recording days...")

    few_days = check_few_days(
        df,
        min_unique_days=min_unique_days,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
        cgm_col=cgm_col,
        time_col=time_col,
    )

    if verbose:
        print("[4/5] Checking minimum number of CGM readings...")

    few_cgm_records = check_few_cgm_records(
        df,
        min_records=min_records,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
        cgm_col=cgm_col,
        time_col=time_col,
    )

    if verbose:
        print("[5/5] Checking non-consecutive recording days...")

    non_consecutive_days = check_non_consecutive_days(
        df,
        patient_identifier_col=patient_identifier_col,
        dataset_col=dataset_col,
        cgm_col=cgm_col,
        time_col=time_col,
    )

    if verbose:
        print("Completed unified CGM checks.")

    return {
        "duplicate_cgms": duplicate_cgms,
        "null_values": null_values,
        "few_days": few_days,
        "few_cgm_records": few_cgm_records,
        "non_consecutive_days": non_consecutive_days,
    }
