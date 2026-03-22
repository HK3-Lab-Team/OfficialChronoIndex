"""
Plotting helpers for CGM data at subject and dataset level.

These functions accept DataFrames in either:
- Subject-level form (list-like `CGM` and `CGMTime` per row), or
- Recording-level form (scalar `CGM` and `CGMTime` per row).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
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

_CGM_TIME_COLUMN_CANDIDATES: tuple[str, ...] = (
    "CGMTime",
    "cgm_time",
    "timestamp",
    "time",
    "DateTime",
)

_ALL_DATASETS_TOKEN = "all_datasets"


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


def _resolve_time_col(df: pl.DataFrame, cgm_time_col: str | None) -> str | None:
    if cgm_time_col is not None and cgm_time_col in df.columns:
        return cgm_time_col
    if cgm_time_col is None:
        for candidate in _CGM_TIME_COLUMN_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None
    if cgm_time_col == "CGMTime":
        for candidate in _CGM_TIME_COLUMN_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None
    return None


def _normalize_glucose_df(
    df: pl.DataFrame,
    *,
    id_col: str | None,
    cgm_col: str | None,
    cgm_time_col: str | None,
    dataset_col: str,
    require_id: bool,
    require_time: bool,
) -> pl.DataFrame:
    if df.height == 0:
        schema: dict[str, pl.DataType] = {
            "__dataset__": pl.Utf8,
            "__cgm__": pl.Float64,
        }
        if require_id:
            schema["__subject__"] = pl.Utf8
        if require_time:
            schema["__time__"] = pl.Datetime
            schema["__date__"] = pl.Date
        return pl.DataFrame(schema=schema)

    resolved_cgm = cgm_col or _infer_column(df.columns, _CGM_COLUMN_CANDIDATES, "CGM")

    resolved_id: str | None = None
    if require_id:
        resolved_id = id_col or _infer_column(df.columns, _ID_COLUMN_CANDIDATES, "subject id")
    elif id_col is not None and id_col in df.columns:
        resolved_id = id_col
    else:
        for candidate in _ID_COLUMN_CANDIDATES:
            if candidate in df.columns:
                resolved_id = candidate
                break

    resolved_time = _resolve_time_col(df, cgm_time_col)
    if require_time and resolved_time is None:
        raise ValueError(
            "Could not infer CGM time column. Provide `cgm_time_col` or use one of: "
            f"{list(_CGM_TIME_COLUMN_CANDIDATES)}"
        )

    working = df
    cgm_is_list = _is_list_dtype(working.schema[resolved_cgm])

    if resolved_time is not None:
        time_is_list = _is_list_dtype(working.schema[resolved_time])
        if cgm_is_list and time_is_list:
            working = working.explode(resolved_cgm, resolved_time)
        elif cgm_is_list != time_is_list:
            raise ValueError(
                f"Columns '{resolved_cgm}' and '{resolved_time}' must both be list-like or both scalar."
            )
    elif cgm_is_list:
        working = working.explode(resolved_cgm)

    exprs: list[pl.Expr] = [
        pl.col(resolved_cgm).cast(pl.Float64, strict=False).alias("__cgm__"),
    ]
    if resolved_id is not None:
        exprs.append(pl.col(resolved_id).cast(pl.Utf8, strict=False).alias("__subject__"))
    if dataset_col in working.columns:
        exprs.append(pl.col(dataset_col).cast(pl.Utf8, strict=False).alias("__dataset__"))
    else:
        exprs.append(pl.lit(_ALL_DATASETS_TOKEN).alias("__dataset__"))
    if resolved_time is not None:
        exprs.append(pl.col(resolved_time).cast(pl.Datetime, strict=False).alias("__time__"))

    normalized = working.with_columns(exprs).filter(
        pl.col("__cgm__").is_not_null() & pl.col("__cgm__").is_finite()
    )

    if resolved_time is not None:
        normalized = (
            normalized
            .filter(pl.col("__time__").is_not_null())
            .with_columns(pl.col("__time__").dt.date().alias("__date__"))
        )

    select_cols = ["__dataset__", "__cgm__"]
    if resolved_id is not None:
        select_cols.insert(1, "__subject__")
    if resolved_time is not None:
        select_cols.extend(["__time__", "__date__"])

    out = normalized.select(select_cols)

    sort_cols = ["__dataset__"]
    if resolved_id is not None:
        sort_cols.append("__subject__")
    if resolved_time is not None:
        sort_cols.append("__time__")
    return out.sort(sort_cols)


def _normalize_selection(
    selected: str | Sequence[str] | None,
    all_values: list[str],
    label: str,
) -> list[str]:
    if selected is None:
        return all_values
    values = [selected] if isinstance(selected, str) else [str(v) for v in selected]
    missing = sorted(set(values) - set(all_values))
    if missing:
        raise ValueError(f"Unknown {label}(s): {missing}. Available: {all_values}")
    return values


def _build_daily_time_series_figure(group_df: pl.DataFrame, *, title_prefix: str) -> Figure:
    days = group_df.select(pl.col("__date__").unique().sort()).to_series().to_list()
    fig_height = max(3.0, 2.6 * len(days))
    fig, axes = plt.subplots(len(days), 1, figsize=(13, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for ax, day in zip(axes_flat, days):
        day_df = group_df.filter(pl.col("__date__") == day).sort("__time__")
        ax.plot(day_df["__time__"].to_list(), day_df["__cgm__"].to_list(), linewidth=1.25)
        ax.set_ylabel("Glucose")
        ax.set_title(f"{title_prefix} | {day}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, linestyle="--", alpha=0.35)

    axes_flat[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def _build_single_time_series_figure(group_df: pl.DataFrame, *, title: str) -> Figure:
    daily_means = (
        group_df
        .group_by("__date__")
        .agg(pl.col("__cgm__").mean().alias("mean_glucose"))
        .sort("__date__")
    )

    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    ax.plot(
        daily_means["__date__"].to_list(),
        daily_means["mean_glucose"].to_list(),
        marker="o",
        linewidth=1.2,
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Glucose")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def _build_frequency_figure(group_df: pl.DataFrame, *, title: str, bins: int) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(group_df["__cgm__"].to_list(), bins=bins, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_mean_figure(group_df: pl.DataFrame, *, title: str) -> Figure:
    overall_mean = group_df.select(pl.col("__cgm__").mean().alias("mean")).item()
    daily_means = (
        group_df
        .group_by("__date__")
        .agg(pl.col("__cgm__").mean().alias("mean_glucose"))
        .sort("__date__")
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 3]})
    ax_overall, ax_daily = axes

    ax_overall.bar(["Overall"], [overall_mean], color="#1f77b4")
    ax_overall.set_ylabel("Mean Glucose")
    ax_overall.set_title("Overall Mean")
    ax_overall.grid(True, axis="y", linestyle="--", alpha=0.25)

    ax_daily.plot(
        daily_means["__date__"].to_list(),
        daily_means["mean_glucose"].to_list(),
        marker="o",
        linewidth=1.2,
    )
    ax_daily.set_xlabel("Date")
    ax_daily.set_ylabel("Mean Glucose")
    ax_daily.set_title("Daily Mean")
    ax_daily.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_subject_glucose_time_series(
    df: pl.DataFrame,
    *,
    subject_id: str | Sequence[str] | None = None,
    daily: bool = True,
    id_col: str | None = None,
    cgm_col: str | None = None,
    cgm_time_col: str | None = "CGMTime",
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=id_col,
        cgm_col=cgm_col,
        cgm_time_col=cgm_time_col,
        dataset_col=dataset_col,
        require_id=True,
        require_time=True,
    )
    subjects = normalized.select(pl.col("__subject__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(subject_id, subjects, "subject")

    figures: list[Figure] = []
    for sid in selected:
        group_df = normalized.filter(pl.col("__subject__") == sid)
        if daily:
            fig = _build_daily_time_series_figure(group_df, title_prefix=f"Subject {sid}")
        else:
            fig = _build_single_time_series_figure(group_df, title=f"Subject {sid} | Glucose Time Series")
        figures.append(fig)
        if show:
            plt.show()
    return figures


def plot_subject_glucose_frequency(
    df: pl.DataFrame,
    *,
    subject_id: str | Sequence[str] | None = None,
    bins: int = 30,
    id_col: str | None = None,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=id_col,
        cgm_col=cgm_col,
        cgm_time_col=None,
        dataset_col=dataset_col,
        require_id=True,
        require_time=False,
    )
    subjects = normalized.select(pl.col("__subject__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(subject_id, subjects, "subject")

    figures: list[Figure] = []
    for sid in selected:
        group_df = normalized.filter(pl.col("__subject__") == sid)
        fig = _build_frequency_figure(group_df, title=f"Subject {sid} | Glucose Frequency", bins=bins)
        figures.append(fig)
        if show:
            plt.show()
    return figures


def plot_subject_mean_glucose(
    df: pl.DataFrame,
    *,
    subject_id: str | Sequence[str] | None = None,
    id_col: str | None = None,
    cgm_col: str | None = None,
    cgm_time_col: str | None = "CGMTime",
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=id_col,
        cgm_col=cgm_col,
        cgm_time_col=cgm_time_col,
        dataset_col=dataset_col,
        require_id=True,
        require_time=True,
    )
    subjects = normalized.select(pl.col("__subject__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(subject_id, subjects, "subject")

    figures: list[Figure] = []
    for sid in selected:
        group_df = normalized.filter(pl.col("__subject__") == sid)
        fig = _build_mean_figure(group_df, title=f"Subject {sid} | Mean Glucose")
        figures.append(fig)
        if show:
            plt.show()
    return figures


def plot_dataset_glucose_time_series(
    df: pl.DataFrame,
    *,
    dataset: str | Sequence[str] | None = None,
    daily: bool = True,
    cgm_col: str | None = None,
    cgm_time_col: str | None = "CGMTime",
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=None,
        cgm_col=cgm_col,
        cgm_time_col=cgm_time_col,
        dataset_col=dataset_col,
        require_id=False,
        require_time=True,
    )
    datasets = normalized.select(pl.col("__dataset__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(dataset, datasets, "dataset")

    figures: list[Figure] = []
    for name in selected:
        group_df = normalized.filter(pl.col("__dataset__") == name)
        if daily:
            fig = _build_daily_time_series_figure(group_df, title_prefix=f"Dataset {name}")
        else:
            fig = _build_single_time_series_figure(group_df, title=f"Dataset {name} | Glucose Time Series")
        figures.append(fig)
        if show:
            plt.show()
    return figures


def plot_dataset_glucose_frequency(
    df: pl.DataFrame,
    *,
    dataset: str | Sequence[str] | None = None,
    bins: int = 30,
    cgm_col: str | None = None,
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=None,
        cgm_col=cgm_col,
        cgm_time_col=None,
        dataset_col=dataset_col,
        require_id=False,
        require_time=False,
    )
    datasets = normalized.select(pl.col("__dataset__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(dataset, datasets, "dataset")

    figures: list[Figure] = []
    for name in selected:
        group_df = normalized.filter(pl.col("__dataset__") == name)
        fig = _build_frequency_figure(group_df, title=f"Dataset {name} | Glucose Frequency", bins=bins)
        figures.append(fig)
        if show:
            plt.show()
    return figures


def plot_dataset_mean_glucose(
    df: pl.DataFrame,
    *,
    dataset: str | Sequence[str] | None = None,
    cgm_col: str | None = None,
    cgm_time_col: str | None = "CGMTime",
    dataset_col: str = "dataset",
    show: bool = True,
) -> list[Figure]:
    normalized = _normalize_glucose_df(
        df,
        id_col=None,
        cgm_col=cgm_col,
        cgm_time_col=cgm_time_col,
        dataset_col=dataset_col,
        require_id=False,
        require_time=True,
    )
    datasets = normalized.select(pl.col("__dataset__").unique().sort()).to_series().to_list()
    selected = _normalize_selection(dataset, datasets, "dataset")

    figures: list[Figure] = []
    for name in selected:
        group_df = normalized.filter(pl.col("__dataset__") == name)
        fig = _build_mean_figure(group_df, title=f"Dataset {name} | Mean Glucose")
        figures.append(fig)
        if show:
            plt.show()
    return figures
