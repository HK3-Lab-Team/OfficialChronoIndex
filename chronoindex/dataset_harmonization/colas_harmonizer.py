from __future__ import annotations

from typing import Any

import polars as pl

from .zhao_harmonizer import HarmonizedBlocks


_RAW_ID_COLUMN = "id"
_HARMONIZED_ID_COLUMN = "Id"
_RAW_TIME_COLUMN = "time_series"
_HARMONIZED_TIME_COLUMN = "CGMTime"
_RAW_CGM_COLUMN = "CGM"
_ID_INPUT_CANDIDATES = (_HARMONIZED_ID_COLUMN, _RAW_ID_COLUMN)
_TIME_INPUT_CANDIDATES = (_HARMONIZED_TIME_COLUMN, "Timestamp", _RAW_TIME_COLUMN)
_SEX_INPUT_CANDIDATES = ("Sex", "gender")
_METADATA_PRIORITY_COLUMNS = (
    _HARMONIZED_ID_COLUMN,
    "Sex",
    "Age",
    "Height",
    "Weight",
    "BMI",
)


def _pick_existing_column(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col_name in candidates:
        if col_name in frame.columns:
            return col_name
    return None


def _resolve_cgm_source(dataset: Any) -> tuple[pl.DataFrame, str, str]:
    joined = getattr(dataset, "joined_data", None)
    if isinstance(joined, pl.DataFrame):
        id_col = _pick_existing_column(joined, _ID_INPUT_CANDIDATES)
        time_col = _pick_existing_column(joined, _TIME_INPUT_CANDIDATES)
        if id_col and time_col and _RAW_CGM_COLUMN in joined.columns:
            return joined, id_col, time_col

    cgm_data = getattr(dataset, "cgm_data", None)
    if isinstance(cgm_data, pl.DataFrame):
        id_col = _pick_existing_column(cgm_data, _ID_INPUT_CANDIDATES)
        time_col = _pick_existing_column(cgm_data, _TIME_INPUT_CANDIDATES)
        if id_col and time_col and _RAW_CGM_COLUMN in cgm_data.columns:
            return cgm_data, id_col, time_col

    raise ValueError(
        "Colas harmonizer expects joined_data or cgm_data with columns "
        f"including id/Id, time_series/CGMTime/Timestamp, and {_RAW_CGM_COLUMN}."
    )


def _normalize_type_of_diabetes_column(metadata: pl.DataFrame) -> pl.DataFrame:
    if "Type of Diabetes" in metadata.columns:
        return metadata

    if "T2DM" in metadata.columns:
        return metadata.with_columns(
            pl.col("T2DM")
            .cast(pl.Utf8, strict=False)
            .map_elements(
                lambda value: "T2DM" if str(value).lower() == "true" else "Control",
                return_dtype=pl.Utf8,
            )
            .alias("Type of Diabetes")
        )

    return metadata.with_columns(pl.lit(None).cast(pl.Utf8).alias("Type of Diabetes"))


def _select_or_null(frame: pl.DataFrame, column_name: str, dtype: pl.DataType) -> pl.Expr:
    if column_name in frame.columns:
        return pl.col(column_name).cast(dtype, strict=False).alias(column_name)
    return pl.lit(None).cast(dtype).alias(column_name)


def _order_metadata_columns(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = [col for col in _METADATA_PRIORITY_COLUMNS if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered]
    return frame.select([*ordered, *remaining])


def _build_cgm_block(cgm_source: pl.DataFrame, id_col: str, time_col: str) -> pl.DataFrame:
    cgm = cgm_source.select(
        [
            pl.col(time_col).alias(_HARMONIZED_TIME_COLUMN),
            pl.col(_RAW_CGM_COLUMN).cast(pl.List(pl.Float64), strict=False).alias(_RAW_CGM_COLUMN),
            pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
        ]
    )
    return cgm.select([_HARMONIZED_ID_COLUMN, _RAW_CGM_COLUMN, _HARMONIZED_TIME_COLUMN])


def _build_metadata_block(metadata: pl.DataFrame) -> pl.DataFrame:
    metadata = _normalize_type_of_diabetes_column(metadata)
    id_col = _pick_existing_column(metadata, _ID_INPUT_CANDIDATES) or _RAW_ID_COLUMN
    sex_col = _pick_existing_column(metadata, _SEX_INPUT_CANDIDATES)
    sex_expr = (
        pl.when(pl.col(sex_col).cast(pl.Utf8, strict=False) == "0")
        .then(pl.lit("M"))
        .when(pl.col(sex_col).cast(pl.Utf8, strict=False) == "1")
        .then(pl.lit("F"))
        .otherwise(pl.col(sex_col).cast(pl.Utf8, strict=False))
        .alias("Sex")
        if sex_col is not None
        else pl.lit(None).cast(pl.Utf8).alias("Sex")
    )
    metadata_out = metadata.select(
        [
            _select_or_null(metadata, id_col, pl.Utf8).alias(_HARMONIZED_ID_COLUMN),
            sex_expr,
            _select_or_null(metadata, "Age", pl.Int32).alias("Age"),
            _select_or_null(metadata, "BMI", pl.Float64).alias("BMI"),
            _select_or_null(metadata, "Type of Diabetes", pl.Utf8).alias("Type of Diabetes"),
        ]
    )
    return _order_metadata_columns(metadata_out)


def _build_clinical_block(metadata: pl.DataFrame) -> pl.DataFrame:
    id_col = _pick_existing_column(metadata, _ID_INPUT_CANDIDATES) or _RAW_ID_COLUMN
    return metadata.select(
        [
            _select_or_null(metadata, id_col, pl.Utf8).alias(_HARMONIZED_ID_COLUMN),
            _select_or_null(metadata, "HbA1c", pl.Float64).alias("HbA1c"),
        ]
    )


def harmonize_colas_dataset(dataset: Any) -> HarmonizedBlocks:
    metadata = getattr(dataset, "metadata", None)
    if not isinstance(metadata, pl.DataFrame):
        raise ValueError("Colas harmonizer expects `metadata` as a Polars DataFrame.")

    cgm_source, id_col, time_col = _resolve_cgm_source(dataset)
    cgm_data = _build_cgm_block(cgm_source, id_col=id_col, time_col=time_col)
    meta_data = _build_metadata_block(metadata)
    clinical_data = _build_clinical_block(metadata)

    return HarmonizedBlocks(
        CGMData=cgm_data,
        MetaData=meta_data,
        LabData=clinical_data,
    )


def harmonize_loaded_colas(colas_dataset: Any) -> HarmonizedBlocks:
    """Explicit harmonization entrypoint for a loaded Colas ingestion dataset."""
    return harmonize_colas_dataset(colas_dataset)
