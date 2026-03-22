from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl


_PAREN_CONTENT_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
_MULTISPACE_PATTERN = re.compile(r"\s+")

_CGM_SAFE_RENAMES = {
    "DI (Eng)": "Dietary Intake Eng",
    "DI (Ch)": "Dietary Intake Ch",
    "Insulin (s.c.)": "Insulin sc",
    "Insulin (i.v.)": "Insulin iv",
}

_METADATA_RENAMES = {
    "Gender": "Sex",
}

_META_DATA_COLUMNS = {
    "Patient Number",
    "Sex",
    "Age",
    "Height",
    "Weight",
    "BMI",
    "Type of Diabetes",
}

_RAW_ID_COLUMN = "Patient Number"
_HARMONIZED_ID_COLUMN = "Id"
_RAW_TIME_COLUMN = "Date"
_HARMONIZED_TIME_COLUMN = "CGMTime"
_RAW_CGM_COLUMN = "CGM"
_RAW_FOOD_COLUMN_EN = "DI (Eng)"
_HARMONIZED_FOOD_TIME_COLUMN = "FoodTimepoint"

_CLEANED_CGM_COLUMNS_TO_EXCLUDE_FROM_CLINICAL = {
    _RAW_ID_COLUMN,
    _RAW_TIME_COLUMN,
    _RAW_CGM_COLUMN,
    "CBG",
    "Blood Ketone",
    "Dietary Intake Eng",
    "Dietary Intake Ch",
    "Insulin sc",
    "NIHA",
    "CSII Bolus",
    "CSII Basal",
    "Insulin iv",
}

_CLEANED_METADATA_COLUMNS_TO_EXCLUDE_FROM_CLINICAL = {
    "Acute Diabetic Complications",
    "Diabetic Macrovascular Complications",
    "Diabetic Microvascular Complications",
    "Hypoglycemic Agents",
    "Other Agents",
    "Smoking History",
    "Alcohol Drinking History",
    "Duration of diabetes",
    "Comorbidities",
}

_METADATA_PRIORITY_COLUMNS = (
    _HARMONIZED_ID_COLUMN,
    "Sex",
    "Age",
    "Height",
    "Weight",
    "BMI",
)


@dataclass
class HarmonizedBlocks:
    CGMData: pl.DataFrame
    MetaData: pl.DataFrame | None = None
    LabData: pl.DataFrame | None = None
    Food: pl.DataFrame | None = None
    PhysicalActivity: pl.DataFrame | None = None
    VisitTimepoints: pl.DataFrame | None = None
    ClinicalMeasurementTimepoints: pl.DataFrame | None = None
    ExtraBlocks: dict[str, pl.DataFrame] = field(default_factory=dict)

    def add_block(self, name: str, frame: pl.DataFrame) -> None:
        if name in {
            "CGMData",
            "MetaData",
            "LabData",
            "Food",
            "PhysicalActivity",
            "VisitTimepoints",
            "ClinicalMeasurementTimepoints",
        }:
            setattr(self, name, frame)
            return
        self.ExtraBlocks[name] = frame

    def get_block(self, name: str) -> pl.DataFrame | None:
        if hasattr(self, name):
            return getattr(self, name)
        return self.ExtraBlocks.get(name)


def _strip_parenthetical_text(column_name: str) -> str:
    stripped = _PAREN_CONTENT_PATTERN.sub(" ", column_name)
    stripped = _MULTISPACE_PATTERN.sub(" ", stripped).strip()
    return stripped


def _dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count == 1:
            deduped.append(name)
        else:
            deduped.append(f"{name}_{count}")
    return deduped


def _clean_columns(
    frame: pl.DataFrame,
    pre_renames: dict[str, str] | None = None,
    post_renames: dict[str, str] | None = None,
) -> pl.DataFrame:
    if pre_renames:
        existing_pre_renames = {k: v for k, v in pre_renames.items() if k in frame.columns}
        if existing_pre_renames:
            frame = frame.rename(existing_pre_renames)

    cleaned = [_strip_parenthetical_text(col) for col in frame.columns]
    deduped = _dedupe_names(cleaned)
    rename_map = {
        old: new for old, new in zip(frame.columns, deduped) if old != new
    }
    if rename_map:
        frame = frame.rename(rename_map)

    if post_renames:
        existing_post_renames = {k: v for k, v in post_renames.items() if k in frame.columns}
        if existing_post_renames:
            frame = frame.rename(existing_post_renames)

    return frame


def _order_metadata_columns(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = [col for col in _METADATA_PRIORITY_COLUMNS if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered]
    return frame.select([*ordered, *remaining])


def _order_lab_columns(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = [col for col in (_HARMONIZED_ID_COLUMN, "visit") if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered]
    return frame.select([*ordered, *remaining])


def _extract_food_block(raw_cgm_data: pl.DataFrame) -> pl.DataFrame:
    id_col = _RAW_ID_COLUMN
    time_col = _RAW_TIME_COLUMN
    food_col = _RAW_FOOD_COLUMN_EN

    if (
        id_col not in raw_cgm_data.columns
        or time_col not in raw_cgm_data.columns
        or food_col not in raw_cgm_data.columns
    ):
        return pl.DataFrame(
            schema={
                _HARMONIZED_ID_COLUMN: pl.Utf8,
                "food": pl.Utf8,
                _HARMONIZED_FOOD_TIME_COLUMN: pl.Datetime("us"),
            }
        )

    return (
        raw_cgm_data.select([id_col, time_col, food_col])
        .explode([time_col, food_col])
        .select(
            [
                pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
                pl.col(food_col).cast(pl.Utf8, strict=False).alias("food"),
                pl.col(time_col)
                .cast(pl.Datetime("us"), strict=False)
                .alias(_HARMONIZED_FOOD_TIME_COLUMN),
            ]
        )
        .filter(pl.col("food").is_not_null())
        .filter(pl.col("food").str.strip_chars() != "")
        .unique(subset=[_HARMONIZED_ID_COLUMN, "food", _HARMONIZED_FOOD_TIME_COLUMN])
        .sort([_HARMONIZED_ID_COLUMN, _HARMONIZED_FOOD_TIME_COLUMN])
    )


def _extract_first_cgm_timepoint(raw_cgm_data: pl.DataFrame) -> pl.DataFrame:
    id_col = _RAW_ID_COLUMN
    time_col = _RAW_TIME_COLUMN

    if id_col not in raw_cgm_data.columns or time_col not in raw_cgm_data.columns:
        return pl.DataFrame(
            schema={_HARMONIZED_ID_COLUMN: pl.Utf8, "timepoint": pl.Datetime("us")}
        )

    return raw_cgm_data.select(
        [
            pl.col(id_col).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            pl.col(time_col)
            .map_elements(
                lambda values: next((val for val in values if val is not None), None)
                if values is not None
                else None,
                return_dtype=pl.Datetime("us"),
            )
            .alias("timepoint"),
        ]
    )


def _build_clinical_measurement_timepoints(
    clinical_data: pl.DataFrame | None,
    raw_cgm_data: pl.DataFrame,
) -> pl.DataFrame:
    id_col = _HARMONIZED_ID_COLUMN
    if clinical_data is None or id_col not in clinical_data.columns:
        return pl.DataFrame(
            schema={
                id_col: pl.Utf8,
                "measurement": pl.Utf8,
                "value": pl.Utf8,
                "timepoint": pl.Datetime("us"),
            }
        )

    clinical_columns = [col for col in clinical_data.columns if col != id_col]
    if not clinical_columns:
        return pl.DataFrame(
            schema={
                id_col: pl.Utf8,
                "measurement": pl.Utf8,
                "value": pl.Utf8,
                "timepoint": pl.Datetime("us"),
            }
        )

    long_clinical = (
        clinical_data.unpivot(
            index=[id_col],
            on=clinical_columns,
            variable_name="measurement",
            value_name="value",
        )
        .filter(pl.col("value").is_not_null())
        .with_columns(pl.col("value").cast(pl.Utf8, strict=False))
    )

    first_cgm_timepoint = _extract_first_cgm_timepoint(raw_cgm_data)
    return (
        long_clinical.join(first_cgm_timepoint, on=id_col, how="left")
        .select([id_col, "measurement", "value", "timepoint"])
        .sort([id_col, "measurement"])
    )


def harmonize_zhao_dataset(dataset: Any) -> HarmonizedBlocks:
    if not hasattr(dataset, "metadata") or not hasattr(dataset, "cgm_data"):
        raise ValueError(
            "Zhao harmonizer expects an object exposing `metadata` and `cgm_data`."
        )

    if not isinstance(dataset.metadata, pl.DataFrame) or not isinstance(dataset.cgm_data, pl.DataFrame):
        raise ValueError(
            "Zhao harmonizer expects `metadata` and `cgm_data` to be Polars DataFrames."
        )

    metadata_clean = _clean_columns(
        dataset.metadata,
        post_renames=_METADATA_RENAMES,
    )

    cgm_clean = _clean_columns(
        dataset.cgm_data,
        pre_renames=_CGM_SAFE_RENAMES,
    )

    metadata_cols_present = [col for col in metadata_clean.columns if col in _META_DATA_COLUMNS]
    if _RAW_ID_COLUMN in metadata_clean.columns and _RAW_ID_COLUMN not in metadata_cols_present:
        metadata_cols_present = [_RAW_ID_COLUMN, *metadata_cols_present]

    meta_data = (
        metadata_clean.select(metadata_cols_present)
        if metadata_cols_present
        else None
    )
    if meta_data is not None and "Sex" in meta_data.columns:
        meta_data = meta_data.with_columns(
            pl.when(pl.col("Sex").cast(pl.Utf8, strict=False) == "1")
            .then(pl.lit("F"))
            .when(pl.col("Sex").cast(pl.Utf8, strict=False) == "2")
            .then(pl.lit("M"))
            .otherwise(pl.col("Sex").cast(pl.Utf8, strict=False))
            .alias("Sex")
        )

    clinical_feature_cols = [
        col
        for col in metadata_clean.columns
        if col not in _META_DATA_COLUMNS
        and col not in _CLEANED_METADATA_COLUMNS_TO_EXCLUDE_FROM_CLINICAL
    ]
    clinical_cols = (
        [_RAW_ID_COLUMN, *clinical_feature_cols]
        if _RAW_ID_COLUMN in metadata_clean.columns
        else clinical_feature_cols
    )
    clinical_data_from_metadata = (
        metadata_clean.select(clinical_cols)
        if clinical_cols
        else None
    )

    rename_id_map = {_RAW_ID_COLUMN: _HARMONIZED_ID_COLUMN}
    if meta_data is not None and _RAW_ID_COLUMN in meta_data.columns:
        meta_data = meta_data.rename(rename_id_map)
    if meta_data is not None:
        meta_data = _order_metadata_columns(meta_data)
    if clinical_data_from_metadata is not None and _RAW_ID_COLUMN in clinical_data_from_metadata.columns:
        clinical_data_from_metadata = clinical_data_from_metadata.rename(rename_id_map)

    cgm_clinical_columns = [
        col_name
        for col_name in cgm_clean.columns
        if col_name not in _CLEANED_CGM_COLUMNS_TO_EXCLUDE_FROM_CLINICAL
    ]
    clinical_data_from_cgm = (
        cgm_clean.select([_RAW_ID_COLUMN, *cgm_clinical_columns])
        if _RAW_ID_COLUMN in cgm_clean.columns and cgm_clinical_columns
        else None
    )
    if clinical_data_from_cgm is not None and _RAW_ID_COLUMN in clinical_data_from_cgm.columns:
        clinical_data_from_cgm = clinical_data_from_cgm.rename(rename_id_map)

    if clinical_data_from_metadata is None:
        clinical_data = clinical_data_from_cgm
    elif clinical_data_from_cgm is None:
        clinical_data = clinical_data_from_metadata
    else:
        clinical_data = clinical_data_from_metadata.join(
            clinical_data_from_cgm,
            on=_HARMONIZED_ID_COLUMN,
            how="left",
        )
    if clinical_data is not None:
        clinical_data = _order_lab_columns(clinical_data)

    food = _extract_food_block(dataset.cgm_data)
    cgm_data_out = cgm_clean.rename(rename_id_map) if _RAW_ID_COLUMN in cgm_clean.columns else cgm_clean
    if _RAW_TIME_COLUMN in cgm_data_out.columns:
        cgm_data_out = cgm_data_out.rename({_RAW_TIME_COLUMN: _HARMONIZED_TIME_COLUMN})

    required_cgm_cols = [_HARMONIZED_ID_COLUMN, _RAW_CGM_COLUMN, _HARMONIZED_TIME_COLUMN]
    missing_cgm_cols = [col for col in required_cgm_cols if col not in cgm_data_out.columns]
    if missing_cgm_cols:
        raise ValueError(
            f"Cannot build harmonized CGMData; missing required columns: {missing_cgm_cols}"
        )
    cgm_data_out = cgm_data_out.select(required_cgm_cols)

    return HarmonizedBlocks(
        CGMData=cgm_data_out,
        MetaData=meta_data,
        LabData=clinical_data,
        Food=food,
    )


def harmonize_loaded_zhao(zhao_dataset: Any) -> HarmonizedBlocks:
    """Explicit harmonization entrypoint for a loaded Zhao ingestion dataset."""
    return harmonize_zhao_dataset(zhao_dataset)
