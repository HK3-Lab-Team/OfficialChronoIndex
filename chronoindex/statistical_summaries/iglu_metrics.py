"""
CGM metrics adapted from the R `iglu` package:
https://irinagain.github.io/iglu/
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


_ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "Id",
    "id",
    "participantId",
    "Patient Number",
    "PtID",
    "subject_id",
    "subjectId",
    "patient_id",
)

_CGM_COLUMN_CANDIDATES: tuple[str, ...] = (
    "CGM",
    "cgm",
    "historicGlucoseMmolL",
    "glucose",
)


def _infer_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not infer {label} column. Tried candidates: {list(candidates)}. "
        f"Available columns: {list(columns)}"
    )


def _is_list_dtype(dtype: pl.DataType) -> bool:
    return str(dtype).lower().startswith("list")


def _prepare_cgm_values(df: pl.DataFrame, cgm_col: str) -> pl.DataFrame:
    working = df
    if _is_list_dtype(df.schema[cgm_col]):
        working = working.explode(cgm_col)

    return (
        working
        .with_columns(pl.col(cgm_col).cast(pl.Float64, strict=False))
        .filter(pl.col(cgm_col).is_not_null())
        .filter(pl.col(cgm_col).is_finite())
    )


def _normalize_target_ranges(
    target_ranges: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]]:
    default_ranges: list[tuple[float, float]] = [(70.0, 180.0), (63.0, 140.0)]
    if target_ranges is None:
        return default_ranges

    normalized: list[tuple[float, float]] = []
    for pair in target_ranges:
        if len(pair) != 2:
            raise ValueError(
                "Each target range must contain exactly 2 values, "
                f"but got: {pair}"
            )
        low, high = sorted((float(pair[0]), float(pair[1])))
        normalized.append((low, high))
    return normalized


def _range_token(value: float) -> str:
    # Keep column names compact when values are integer-like.
    return str(int(value)) if float(value).is_integer() else str(value)


def _normalize_targets(targets: Sequence[float] | None) -> list[float]:
    default_targets = [140.0, 180.0, 250.0]
    if targets is None:
        return default_targets
    return [float(t) for t in targets]


def _normalize_targets_below(targets: Sequence[float] | None) -> list[float]:
    default_targets = [54.0, 70.0]
    if targets is None:
        return default_targets
    return [float(t) for t in targets]


def summary(
    df: pl.DataFrame,
    *,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute subject-level summary stats for CGM values.

    Accepts either:
    - Subject-level data (one row per subject with list-like CGM column), or
    - Recording-level data (multiple rows per subject with scalar CGM values).

    If `dataset_col` exists, grouping is done by (`dataset_col`, `id_col`).
    Otherwise, grouping is done by (`id_col`) only.
    """
    if df is None or df.height == 0:
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "Id": pl.Utf8,
                "Min.": pl.Float64,
                "1st Qu.": pl.Float64,
                "Median": pl.Float64,
                "Mean": pl.Float64,
                "3rd Qu.": pl.Float64,
                "Max.": pl.Float64,
            }
        )

    id_col = id_col or _infer_column(df.columns, _ID_COLUMN_CANDIDATES, "subject id")
    cgm_col = cgm_col or _infer_column(df.columns, _CGM_COLUMN_CANDIDATES, "CGM")

    group_keys = [id_col]
    if dataset_col in df.columns:
        group_keys = [dataset_col, id_col]

    # Normalize values so summary metrics are robust on heterogeneous source dtypes.
    working = _prepare_cgm_values(df, cgm_col)

    out = (
        working
        .group_by(group_keys)
        .agg(
            [
                pl.col(cgm_col).min().alias("Min."),
                pl.col(cgm_col).quantile(0.25).alias("1st Qu."),
                pl.col(cgm_col).median().alias("Median"),
                pl.col(cgm_col).mean().alias("Mean"),
                pl.col(cgm_col).quantile(0.75).alias("3rd Qu."),
                pl.col(cgm_col).max().alias("Max."),
            ]
        )
        .sort(group_keys)
    )

    return out


def in_range_percent(
    data: pl.DataFrame | Sequence[float] | pl.Series,
    *,
    target_ranges: Sequence[Sequence[float]] | None = None,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute percent of CGM values in target ranges.

    For DataFrame input: returns one row per subject (and dataset when present).
    For vector input: returns a single-row DataFrame without subject id columns.

    Percent values are between 0 and 100; null/non-finite CGM values are ignored.
    """
    normalized_ranges = _normalize_target_ranges(target_ranges)

    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            schema = {
                "dataset": pl.Utf8,
                "Id": pl.Utf8,
            }
            for low, high in normalized_ranges:
                schema[f"in_range_{_range_token(low)}_{_range_token(high)}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = _prepare_cgm_values(data, cgm_col)

        metrics = []
        for low, high in normalized_ranges:
            col_name = f"in_range_{_range_token(low)}_{_range_token(high)}"
            metrics.append(
                (pl.col(cgm_col).is_between(low, high).mean() * 100.0).alias(col_name)
            )

        return (
            working
            .group_by(group_keys)
            .agg(metrics)
            .sort(group_keys)
        )

    # Numeric-vector mode (compatible with R iglu behavior on vectors).
    if isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("CGM", list(data))

    values_df = (
        pl.DataFrame({"CGM": values_series})
        .with_columns(pl.col("CGM").cast(pl.Float64, strict=False))
        .filter(pl.col("CGM").is_not_null() & pl.col("CGM").is_finite())
    )

    out: dict[str, float | None] = {}
    if values_df.height == 0:
        for low, high in normalized_ranges:
            out[f"in_range_{_range_token(low)}_{_range_token(high)}"] = None
        return pl.DataFrame([out])

    for low, high in normalized_ranges:
        col_name = f"in_range_{_range_token(low)}_{_range_token(high)}"
        pct = (
            values_df
            .select((pl.col("CGM").is_between(low, high).mean() * 100.0).alias(col_name))
            .item()
        )
        out[col_name] = float(pct) if pct is not None else None
    return pl.DataFrame([out])


def above_percent(
    data: pl.DataFrame | Sequence[float] | pl.Series,
    *,
    targets_above: Sequence[float] | None = None,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute percent of CGM values above target thresholds.

    For DataFrame input: returns one row per subject (and dataset when present).
    For vector input: returns a single-row DataFrame without subject id columns.

    Percent values are between 0 and 100; null/non-finite CGM values are ignored.
    """
    targets = _normalize_targets(targets_above)

    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            schema = {
                "dataset": pl.Utf8,
                "Id": pl.Utf8,
            }
            for target in targets:
                schema[f"above_{_range_token(target)}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = _prepare_cgm_values(data, cgm_col)

        metrics = []
        for target in targets:
            col_name = f"above_{_range_token(target)}"
            metrics.append((pl.col(cgm_col) > target).mean().mul(100.0).alias(col_name))

        return (
            working
            .group_by(group_keys)
            .agg(metrics)
            .sort(group_keys)
        )

    # Numeric-vector mode (compatible with R iglu behavior on vectors).
    if isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("CGM", list(data))

    values_df = (
        pl.DataFrame({"CGM": values_series})
        .with_columns(pl.col("CGM").cast(pl.Float64, strict=False))
        .filter(pl.col("CGM").is_not_null() & pl.col("CGM").is_finite())
    )

    out: dict[str, float | None] = {}
    if values_df.height == 0:
        for target in targets:
            out[f"above_{_range_token(target)}"] = None
        return pl.DataFrame([out])

    for target in targets:
        col_name = f"above_{_range_token(target)}"
        pct = values_df.select(((pl.col("CGM") > target).mean() * 100.0).alias(col_name)).item()
        out[col_name] = float(pct) if pct is not None else None
    return pl.DataFrame([out])


def below_percent(
    data: pl.DataFrame | Sequence[float] | pl.Series,
    *,
    targets_below: Sequence[float] | None = None,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute percent of CGM values below target thresholds.

    For DataFrame input: returns one row per subject (and dataset when present).
    For vector input: returns a single-row DataFrame without subject id columns.

    Percent values are between 0 and 100; null/non-finite CGM values are ignored.
    """
    targets = _normalize_targets_below(targets_below)

    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            schema = {
                "dataset": pl.Utf8,
                "Id": pl.Utf8,
            }
            for target in targets:
                schema[f"below_{_range_token(target)}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = _prepare_cgm_values(data, cgm_col)

        metrics = []
        for target in targets:
            col_name = f"below_{_range_token(target)}"
            metrics.append((pl.col(cgm_col) < target).mean().mul(100.0).alias(col_name))

        return (
            working
            .group_by(group_keys)
            .agg(metrics)
            .sort(group_keys)
        )

    # Numeric-vector mode (compatible with R iglu behavior on vectors).
    if isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("CGM", list(data))

    values_df = (
        pl.DataFrame({"CGM": values_series})
        .with_columns(pl.col("CGM").cast(pl.Float64, strict=False))
        .filter(pl.col("CGM").is_not_null() & pl.col("CGM").is_finite())
    )

    out: dict[str, float | None] = {}
    if values_df.height == 0:
        for target in targets:
            out[f"below_{_range_token(target)}"] = None
        return pl.DataFrame([out])

    for target in targets:
        col_name = f"below_{_range_token(target)}"
        pct = values_df.select(((pl.col("CGM") < target).mean() * 100.0).alias(col_name)).item()
        out[col_name] = float(pct) if pct is not None else None
    return pl.DataFrame([out])


def cv_glu(
    data: pl.DataFrame | Sequence[float] | pl.Series,
    *,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute coefficient of variation (CV) of glucose values.

    CV is calculated as: 100 * sd(glucose) / mean(glucose)

    For DataFrame input: returns one row per subject (and dataset when present).
    For vector input: returns a single-row DataFrame with column `CV`.

    Null/non-finite CGM values are ignored.
    """
    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            return pl.DataFrame(
                schema={
                    "dataset": pl.Utf8,
                    "Id": pl.Utf8,
                    "CV": pl.Float64,
                }
            )

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = _prepare_cgm_values(data, cgm_col)

        return (
            working
            .group_by(group_keys)
            .agg((pl.col(cgm_col).std() / pl.col(cgm_col).mean() * 100.0).alias("CV"))
            .sort(group_keys)
        )

    # Numeric-vector mode (compatible with R iglu behavior on vectors).
    if isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("CGM", list(data))

    values_df = (
        pl.DataFrame({"CGM": values_series})
        .with_columns(pl.col("CGM").cast(pl.Float64, strict=False))
        .filter(pl.col("CGM").is_not_null() & pl.col("CGM").is_finite())
    )

    if values_df.height == 0:
        return pl.DataFrame([{"CV": None}])

    cv_value = values_df.select(
        (pl.col("CGM").std() / pl.col("CGM").mean() * 100.0).alias("CV")
    ).item()
    return pl.DataFrame([{"CV": float(cv_value) if cv_value is not None else None}])


def gmi(
    data: pl.DataFrame | Sequence[float] | pl.Series,
    *,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    """
    Compute Glucose Management Indicator (GMI).

    GMI is calculated as: 3.31 + (0.02392 * mean(glucose))

    For DataFrame input: returns one row per subject (and dataset when present).
    For vector input: returns a single-row DataFrame with column `GMI`.

    Null/non-finite CGM values are ignored.
    """
    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            return pl.DataFrame(
                schema={
                    "dataset": pl.Utf8,
                    "Id": pl.Utf8,
                    "GMI": pl.Float64,
                }
            )

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = _prepare_cgm_values(data, cgm_col)

        return (
            working
            .group_by(group_keys)
            .agg((pl.lit(3.31) + pl.col(cgm_col).mean().mul(0.02392)).alias("GMI"))
            .sort(group_keys)
        )

    # Numeric-vector mode (compatible with R iglu behavior on vectors).
    if isinstance(data, pl.Series):
        values_series = data
    else:
        values_series = pl.Series("CGM", list(data))

    values_df = (
        pl.DataFrame({"CGM": values_series})
        .with_columns(pl.col("CGM").cast(pl.Float64, strict=False))
        .filter(pl.col("CGM").is_not_null() & pl.col("CGM").is_finite())
    )

    if values_df.height == 0:
        return pl.DataFrame([{"GMI": None}])

    gmi_value = values_df.select(
        (pl.lit(3.31) + pl.col("CGM").mean().mul(0.02392)).alias("GMI")
    ).item()
    return pl.DataFrame([{"GMI": float(gmi_value) if gmi_value is not None else None}])
