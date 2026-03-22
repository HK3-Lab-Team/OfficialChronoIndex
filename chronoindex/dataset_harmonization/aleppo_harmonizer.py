from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from .zhao_harmonizer import HarmonizedBlocks


_BASE_DATE = date(1990, 1, 1)

_ID_INPUT_CANDIDATES = ("Id", "id", "PtID", "participantId")
_CGM_INPUT_CANDIDATES = ("CGM", "glucemia_series")
_TIME_INPUT_CANDIDATES = ("CGMTime", "Timestamp", "time_series")
_DAYS_INPUT_CANDIDATES = ("days_from_enroll",)
_VISIT_INPUT_CANDIDATES = ("Visit", "visit")
_HBA1C_INPUT_CANDIDATES = ("HbA1cTestRes", "HbA1c")
_HBA1C_DAY_INPUT_CANDIDATES = ("HbA1cTestDtDaysAfterEnroll", "days_from_enroll")

_HARMONIZED_ID_COLUMN = "Id"
_HARMONIZED_CGM_COLUMN = "CGM"
_HARMONIZED_TIME_COLUMN = "CGMTime"
_METADATA_PRIORITY_COLUMNS = (
    _HARMONIZED_ID_COLUMN,
    "Sex",
    "Age",
    "Height",
    "Weight",
    "BMI",
)
_ALEPPO_TIMEPOINT_NOTE = (
    "👉 Data Harmonization: The Aleppo dataset provides timestamps only as days relative to enrollment. Absolute calendar dates are "
    "not available due to de-identification. During harmonization, a placeholder reference date (1990-01-01) "
    "is used as the enrollment date to reconstruct pseudo-datetimes."
)


def _pick_existing_column(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col_name in candidates:
        if col_name in frame.columns:
            return col_name
    return None


def _resolve_cgm_source(dataset: Any) -> tuple[pl.DataFrame, str, str, str, str]:
    candidates = [getattr(dataset, "cgm_data", None), getattr(dataset, "joined_data", None)]
    for frame in candidates:
        if not isinstance(frame, pl.DataFrame):
            continue
        id_col = _pick_existing_column(frame, _ID_INPUT_CANDIDATES)
        cgm_col = _pick_existing_column(frame, _CGM_INPUT_CANDIDATES)
        time_col = _pick_existing_column(frame, _TIME_INPUT_CANDIDATES)
        days_col = _pick_existing_column(frame, _DAYS_INPUT_CANDIDATES)
        if id_col and cgm_col and time_col and days_col:
            return frame, id_col, cgm_col, time_col, days_col

    raise ValueError(
        "Aleppo harmonizer expects cgm_data/joined_data with id, CGM/glucemia_series, "
        "time_series, and days_from_enroll."
    )


def _resolve_metadata_source(dataset: Any) -> pl.DataFrame:
    load_metadata = getattr(dataset, "load_metadata", None)
    if callable(load_metadata):
        try:
            metadata = load_metadata(mode="all")
            if isinstance(metadata, pl.DataFrame):
                return metadata
        except TypeError:
            try:
                metadata = load_metadata()
                if isinstance(metadata, pl.DataFrame):
                    return metadata
            except Exception:
                pass
        except Exception:
            pass

    metadata = getattr(dataset, "metadata", None)
    if isinstance(metadata, pl.DataFrame):
        return metadata

    raise ValueError("Aleppo harmonizer expects `metadata` as a Polars DataFrame.")


def _resolve_hba1c_source(dataset: Any) -> pl.DataFrame:
    load_hba1c = getattr(dataset, "load_hba1c", None)
    if callable(load_hba1c):
        hba1c = load_hba1c()
        if isinstance(hba1c, pl.DataFrame):
            return hba1c

    raise ValueError("Aleppo harmonizer expects `load_hba1c()` to return a Polars DataFrame.")


def _coerce_scalar_expr(frame: pl.DataFrame, column_name: str, dtype: pl.DataType) -> pl.Expr:
    current_dtype = frame.schema.get(column_name)
    if current_dtype is None:
        return pl.lit(None).cast(dtype)
    if str(current_dtype).startswith("List("):
        return pl.col(column_name).list.first().cast(dtype, strict=False)
    return pl.col(column_name).cast(dtype, strict=False)


def _first_available_expr(
    frame: pl.DataFrame,
    candidates: tuple[str, ...],
    dtype: pl.DataType,
) -> pl.Expr:
    col_name = _pick_existing_column(frame, candidates)
    if col_name is None:
        return pl.lit(None).cast(dtype)
    return _coerce_scalar_expr(frame, col_name, dtype)


def _order_metadata_columns(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = [col for col in _METADATA_PRIORITY_COLUMNS if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered]
    return frame.select([*ordered, *remaining])


def _to_datetime_list(
    time_values: list[Any] | None,
    day_values: list[Any] | None,
) -> list[datetime | None] | None:
    if time_values is None:
        return None

    result: list[datetime | None] = []
    for idx, time_value in enumerate(time_values):
        day_value = day_values[idx] if day_values is not None and idx < len(day_values) else None
        if time_value is None or day_value is None:
            result.append(None)
            continue

        if isinstance(time_value, datetime):
            time_part = time_value.time()
        else:
            time_part = time_value

        result.append(datetime.combine(_BASE_DATE + timedelta(days=int(day_value)), time_part))

    return result


def _build_cgm_block(
    frame: pl.DataFrame,
    id_col: str,
    cgm_col: str,
    time_col: str,
    days_col: str,
) -> pl.DataFrame:
    cgm = frame.select(
        [
            pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            pl.col(cgm_col).cast(pl.List(pl.Float64), strict=False).alias(_HARMONIZED_CGM_COLUMN),
            pl.struct([time_col, days_col])
            .map_elements(
                lambda row: _to_datetime_list(row[time_col], row[days_col]),
                return_dtype=pl.List(pl.Datetime("us")),
            )
            .alias(_HARMONIZED_TIME_COLUMN),
        ]
    )
    return cgm.select([_HARMONIZED_ID_COLUMN, _HARMONIZED_CGM_COLUMN, _HARMONIZED_TIME_COLUMN])


def _build_metadata_block(metadata: pl.DataFrame) -> pl.DataFrame:
    id_col = _pick_existing_column(metadata, _ID_INPUT_CANDIDATES)
    if id_col is None:
        raise ValueError(
            "Aleppo metadata harmonization requires an id column (one of Id/id/PtID/participantId)."
        )

    meta = metadata.select(
        [
            _coerce_scalar_expr(metadata, id_col, pl.Utf8).alias(_HARMONIZED_ID_COLUMN),
            _first_available_expr(metadata, ("AgeAsOfEnrollDt", "Age"), pl.Int32).alias("Age"),
            _first_available_expr(metadata, ("Weight", "screening__Weight"), pl.Float64).alias("Weight"),
            _first_available_expr(metadata, ("Height", "screening__Height"), pl.Float64).alias("Height"),
            _first_available_expr(metadata, ("BMI",), pl.Float64).alias("BMI"),
            _first_available_expr(metadata, ("Gender", "Sex", "screening__Gender"), pl.Utf8).alias("Sex"),
            _first_available_expr(metadata, ("Type of Diabetes",), pl.Utf8).alias("Type of Diabetes"),
        ]
    )

    meta = meta.with_columns(
        pl.when(pl.col("BMI").is_not_null())
        .then(pl.col("BMI"))
        .when(
            (pl.col("Weight").is_not_null())
            & (pl.col("Height").is_not_null())
            & (pl.col("Height") != 0)
        )
        .then(pl.col("Weight") / ((pl.col("Height") / 100) ** 2))
        .otherwise(pl.lit(None))
        .cast(pl.Float64)
        .alias("BMI")
    )
    return _order_metadata_columns(meta)


def _build_clinical_visits_block(hba1c: pl.DataFrame) -> pl.DataFrame:
    id_col = _pick_existing_column(hba1c, _ID_INPUT_CANDIDATES)
    hba1c_col = _pick_existing_column(hba1c, _HBA1C_INPUT_CANDIDATES)
    day_col = _pick_existing_column(hba1c, _HBA1C_DAY_INPUT_CANDIDATES)
    visit_col = _pick_existing_column(hba1c, _VISIT_INPUT_CANDIDATES)

    if id_col is None or hba1c_col is None or day_col is None:
        raise ValueError(
            "Aleppo clinical-visit harmonization requires PtID/id, HbA1cTestRes/HbA1c, "
            "and HbA1cTestDtDaysAfterEnroll."
        )

    visit_expr = (
        pl.col(visit_col).cast(pl.Utf8, strict=False).alias("visit")
        if visit_col is not None
        else pl.lit(None).cast(pl.Utf8).alias("visit")
    )

    # Keep one clinical-visit row per HLocalHbA1c record:
    # for each subject, the number of visits equals the number of rows in load_hba1c().
    return (
        hba1c.select(
            [
                pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
                visit_expr,
                pl.col(day_col)
                .cast(pl.Int64, strict=False)
                .map_elements(
                    lambda value: _BASE_DATE + timedelta(days=int(value))
                    if value is not None
                    else None,
                    return_dtype=pl.Date,
                )
                .alias("VisitTimepoint"),
            ]
        )
        .filter(pl.col(_HARMONIZED_ID_COLUMN).is_not_null())
        .sort([_HARMONIZED_ID_COLUMN, "VisitTimepoint"])
    )


def _build_biomarkers_block(hba1c: pl.DataFrame) -> pl.DataFrame:
    id_col = _pick_existing_column(hba1c, _ID_INPUT_CANDIDATES)
    hba1c_col = _pick_existing_column(hba1c, _HBA1C_INPUT_CANDIDATES)
    visit_col = _pick_existing_column(hba1c, _VISIT_INPUT_CANDIDATES)

    if id_col is None or hba1c_col is None:
        raise ValueError(
            "Aleppo biomarkers harmonization requires PtID/id and HbA1cTestRes/HbA1c."
        )

    visit_expr = (
        pl.col(visit_col).cast(pl.Utf8, strict=False).alias("visit")
        if visit_col is not None
        else pl.lit(None).cast(pl.Utf8).alias("visit")
    )

    return hba1c.select(
        [
            pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            visit_expr,
            pl.col(hba1c_col).cast(pl.Float64, strict=False).alias("HbA1c"),
        ]
    ).filter(pl.col(_HARMONIZED_ID_COLUMN).is_not_null())


def harmonize_aleppo_dataset(dataset: Any) -> HarmonizedBlocks:
    print(_ALEPPO_TIMEPOINT_NOTE)
    frame, id_col, cgm_col, time_col, days_col = _resolve_cgm_source(dataset)
    metadata_source = _resolve_metadata_source(dataset)
    hba1c_source = _resolve_hba1c_source(dataset)
    cgm_data = _build_cgm_block(
        frame,
        id_col=id_col,
        cgm_col=cgm_col,
        time_col=time_col,
        days_col=days_col,
    )
    meta_data = _build_metadata_block(metadata_source)
    clinical_visits = _build_clinical_visits_block(hba1c_source)
    biomarkers = _build_biomarkers_block(hba1c_source)
    harmonized = HarmonizedBlocks(
        CGMData=cgm_data,
        MetaData=meta_data,
        LabData=biomarkers,
        VisitTimepoints=clinical_visits,
    )
    harmonized.add_block("Biomarkers", biomarkers)
    return harmonized


def harmonize_loaded_aleppo(aleppo_dataset: Any) -> HarmonizedBlocks:
    """Explicit harmonization entrypoint for a loaded Aleppo ingestion dataset."""
    return harmonize_aleppo_dataset(aleppo_dataset)
