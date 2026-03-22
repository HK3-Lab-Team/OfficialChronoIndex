from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl

from .zhao_harmonizer import HarmonizedBlocks


_RAW_ID_COLUMN = "participantId"
_RAW_TIME_COLUMN = "time_series"
_RAW_CGM_COLUMN = "CGM"

_HARMONIZED_ID_COLUMN = "Id"
_HARMONIZED_TIME_COLUMN = "CGMTime"

_METADATA_OUTPUT_COLUMNS = (
    _HARMONIZED_ID_COLUMN,
    "Age",
    "Sex",
    "Weight",
    "Height",
    "BMI",
)


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _select_or_null(frame: pl.DataFrame, column_name: str, dtype: pl.DataType | None = None) -> pl.Expr:
    if column_name in frame.columns:
        if dtype is None:
            return pl.col(column_name).alias(column_name)
        return pl.col(column_name).cast(dtype, strict=False).alias(column_name)
    if dtype is None:
        return pl.lit(None).alias(column_name)
    return pl.lit(None).cast(dtype).alias(column_name)


def _build_cgm_block(cgm_data: pl.DataFrame) -> pl.DataFrame:
    required = (_RAW_ID_COLUMN, _RAW_TIME_COLUMN, _RAW_CGM_COLUMN)
    missing = [col_name for col_name in required if col_name not in cgm_data.columns]
    if missing:
        raise ValueError(
            f"PRAES harmonizer cannot build CGMData; missing required columns: {missing}"
        )

    return cgm_data.select(
        [
            pl.col(_RAW_ID_COLUMN).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            pl.col(_RAW_TIME_COLUMN)
            .cast(pl.List(pl.Datetime("us")), strict=False)
            .alias(_HARMONIZED_TIME_COLUMN),
            pl.col(_RAW_CGM_COLUMN)
            .cast(pl.List(pl.Float64), strict=False)
            .map_elements(
                lambda values: [value * 18.018 if value is not None else None for value in values]
                if values is not None
                else None,
                return_dtype=pl.List(pl.Float64),
            )
            .alias(_RAW_CGM_COLUMN),
        ]
    )


def _build_metadata_block(metadata: pl.DataFrame) -> pl.DataFrame:
    if _RAW_ID_COLUMN not in metadata.columns:
        raise ValueError(
            "PRAES harmonizer cannot build MetaData; missing required column: participantId"
        )

    out = metadata.select(
        [
            pl.col(_RAW_ID_COLUMN).cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            _select_or_null(metadata, "Age", pl.Float64),
            _select_or_null(metadata, "Sex", pl.Utf8),
            _select_or_null(metadata, "Weight", pl.Float64),
            _select_or_null(metadata, "Height", pl.Float64),
            _select_or_null(metadata, "BMI", pl.Float64),
        ]
    )

    out = out.with_columns(
        pl.when(pl.col("Sex").cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase() == "female")
        .then(pl.lit("F"))
        .when(pl.col("Sex").cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase() == "male")
        .then(pl.lit("M"))
        .when(pl.col("Sex").cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase() == "f")
        .then(pl.lit("F"))
        .when(pl.col("Sex").cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase() == "m")
        .then(pl.lit("M"))
        .otherwise(pl.col("Sex").cast(pl.Utf8, strict=False))
        .alias("Sex")
    )

    height_m = (
        pl.when(pl.col("Height") > 10)
        .then(pl.col("Height") / 100.0)
        .otherwise(pl.col("Height"))
    )
    out = out.with_columns(
        pl.when(pl.col("BMI").is_not_null())
        .then(pl.col("BMI"))
        .when(
            pl.col("Weight").is_not_null()
            & pl.col("Height").is_not_null()
            & (height_m > 0)
        )
        .then(pl.col("Weight") / (height_m**2))
        .otherwise(pl.lit(None))
        .cast(pl.Float64, strict=False)
        .alias("BMI")
    )
    return out.select(list(_METADATA_OUTPUT_COLUMNS))


def _resolve_subject_lookup(dataset: Any) -> dict[str, Any]:
    raw_dataset = getattr(dataset, "dataset", dataset)
    subjects = getattr(raw_dataset, "subjects", None)
    if not isinstance(subjects, list):
        return {}
    return {
        str(getattr(subject, "participantId", "")): subject
        for subject in subjects
        if getattr(subject, "participantId", None) is not None
    }


def _resolve_analysis_source(dataset: Any) -> pl.DataFrame:
    raw_dataset = getattr(dataset, "dataset", dataset)
    getter = getattr(raw_dataset, "get_aggregated_analysis_frame", None)
    if callable(getter):
        analyses = getter()
        if isinstance(analyses, pl.DataFrame):
            return analyses
    return pl.DataFrame()


def _resolve_food_source(dataset: Any) -> pl.DataFrame:
    raw_dataset = getattr(dataset, "dataset", dataset)
    getter = getattr(raw_dataset, "get_aggregated_food_frame", None)
    if callable(getter):
        foods = getter()
        if isinstance(foods, pl.DataFrame):
            return foods
    return pl.DataFrame()


def _resolve_macronutrients_source(dataset: Any) -> pl.DataFrame:
    raw_dataset = getattr(dataset, "dataset", dataset)
    getter = getattr(raw_dataset, "macronutrients_frame", None)
    if callable(getter):
        macros = getter()
        if isinstance(macros, pl.DataFrame):
            return macros
    return pl.DataFrame()


def _resolve_activity_source(dataset: Any) -> pl.DataFrame:
    raw_dataset = getattr(dataset, "dataset", dataset)
    getter = getattr(raw_dataset, "get_aggregated_activity_frame", None)
    if callable(getter):
        activity = getter()
        if isinstance(activity, pl.DataFrame):
            return activity
    return pl.DataFrame()


def _format_quantity_with_unit(quantity: Any, unit: Any) -> str | None:
    if quantity is None:
        return None
    quantity_str = f"{float(quantity):g}"
    if unit is None:
        return quantity_str
    unit_str = str(unit).strip()
    if not unit_str:
        return quantity_str
    return f"{quantity_str} {unit_str}"


def _format_food_label(name: Any, quantity: Any, unit: Any) -> str | None:
    if name is None:
        return None
    name_str = str(name).strip()
    if not name_str:
        return None

    qty_with_unit = _format_quantity_with_unit(quantity, unit)
    if qty_with_unit is None:
        return name_str
    return f"{name_str} ({qty_with_unit})"


def _build_food_block(food: pl.DataFrame, macronutrients: pl.DataFrame) -> pl.DataFrame:
    if food.is_empty():
        return pl.DataFrame(
            schema={
                _HARMONIZED_ID_COLUMN: pl.Utf8,
                "FoodTimepoint": pl.Datetime("us"),
                "meal": pl.Utf8,
                "food": pl.Utf8,
                "calories": pl.Float64,
                "carbs": pl.Float64,
                "proteins": pl.Float64,
                "fats": pl.Float64,
            }
        )

    required_food_cols = {"participantId", "date", "meal", "quantity"}
    missing_food_cols = sorted(required_food_cols.difference(set(food.columns)))
    if missing_food_cols:
        raise ValueError(
            f"PRAES harmonizer cannot build Food block; missing food columns: {missing_food_cols}"
        )
    if "foodId" not in food.columns:
        food = food.with_columns(pl.lit(None).cast(pl.Utf8).alias("foodId"))
    food = food.with_columns(pl.col("foodId").cast(pl.Utf8, strict=False))

    macros_selected = (
        macronutrients.select(["foodId", "name", "calories", "carbs", "proteins", "fats", "unit"])
        if not macronutrients.is_empty()
        and {"foodId", "name", "calories", "carbs", "proteins", "fats", "unit"}.issubset(set(macronutrients.columns))
        else pl.DataFrame(
            schema={
                "foodId": pl.Utf8,
                "name": pl.Utf8,
                "calories": pl.Float64,
                "carbs": pl.Float64,
                "proteins": pl.Float64,
                "fats": pl.Float64,
                "unit": pl.Utf8,
            }
        )
    )

    macros_fallback = macros_selected.rename(
        {
            "foodId": "_fallback_food_id",
            "name": "_name_fallback",
            "calories": "_calories_fallback",
            "carbs": "_carbs_fallback",
            "proteins": "_proteins_fallback",
            "fats": "_fats_fallback",
            "unit": "_unit_fallback",
        }
    )

    joined = (
        food.with_columns(
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(
                pl.concat_str(
                    [
                        pl.col("participantId").cast(pl.Utf8, strict=False),
                        pl.col("date").cast(pl.Datetime("us"), strict=False).dt.strftime("%Y-%m-%d"),
                        pl.col("meal").cast(pl.Utf8, strict=False),
                    ],
                    separator="_",
                )
            )
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("_fallback_food_id")
        )
        .join(macros_selected, on="foodId", how="left")
        .join(macros_fallback, on="_fallback_food_id", how="left")
    )

    joined = joined.with_columns(
        [
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(pl.col("_name_fallback").cast(pl.Utf8, strict=False))
            .otherwise(pl.col("name").cast(pl.Utf8, strict=False))
            .alias("_name_base"),
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(pl.col("_calories_fallback").cast(pl.Float64, strict=False))
            .otherwise(
                pl.col("calories").cast(pl.Float64, strict=False)
                * pl.col("quantity").cast(pl.Float64, strict=False)
                / 100.0
            )
            .alias("calories"),
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(pl.col("_carbs_fallback").cast(pl.Float64, strict=False))
            .otherwise(
                pl.col("carbs").cast(pl.Float64, strict=False)
                * pl.col("quantity").cast(pl.Float64, strict=False)
                / 100.0
            )
            .alias("carbs"),
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(pl.col("_proteins_fallback").cast(pl.Float64, strict=False))
            .otherwise(
                pl.col("proteins").cast(pl.Float64, strict=False)
                * pl.col("quantity").cast(pl.Float64, strict=False)
                / 100.0
            )
            .alias("proteins"),
            pl.when(pl.col("foodId").cast(pl.Utf8, strict=False).is_null())
            .then(pl.col("_fats_fallback").cast(pl.Float64, strict=False))
            .otherwise(
                pl.col("fats").cast(pl.Float64, strict=False)
                * pl.col("quantity").cast(pl.Float64, strict=False)
                / 100.0
            )
            .alias("fats"),
        ]
    )
    joined = joined.with_columns(
        pl.struct(["_name_base", "quantity", "unit"])
        .map_elements(
            lambda row: _format_food_label(row["_name_base"], row["quantity"], row["unit"]),
            return_dtype=pl.Utf8,
        )
        .alias("name")
    )

    per_food = (
        joined.select(
            [
                pl.col("participantId").cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
                pl.col("date").cast(pl.Datetime("us"), strict=False).alias("FoodTimepoint"),
                pl.col("meal").cast(pl.Utf8, strict=False).alias("meal"),
                pl.col("name").cast(pl.Utf8, strict=False).alias("name"),
                pl.col("calories").cast(pl.Float64, strict=False).alias("calories"),
                pl.col("carbs").cast(pl.Float64, strict=False).alias("carbs"),
                pl.col("proteins").cast(pl.Float64, strict=False).alias("proteins"),
                pl.col("fats").cast(pl.Float64, strict=False).alias("fats"),
            ]
        )
        .with_row_index("_row_idx")
    )

    per_timepoint = (
        per_food.group_by([_HARMONIZED_ID_COLUMN, "FoodTimepoint"], maintain_order=True)
        .agg(
            [
                pl.col("meal").drop_nulls().first().alias("meal"),
                pl.col("name")
                .sort_by("_row_idx")
                .drop_nulls()
                .str.concat(". ")
                .alias("food"),
                pl.col("calories").sum().alias("calories"),
                pl.col("carbs").sum().alias("carbs"),
                pl.col("proteins").sum().alias("proteins"),
                pl.col("fats").sum().alias("fats"),
            ]
        )
        .select(
            [
                _HARMONIZED_ID_COLUMN,
                "FoodTimepoint",
                "meal",
                "food",
                "calories",
                "carbs",
                "proteins",
                "fats",
            ]
        )
    )
    return per_timepoint.sort([_HARMONIZED_ID_COLUMN, "FoodTimepoint"])


def _build_physical_activity_block(daily_activity: pl.DataFrame) -> pl.DataFrame:
    if daily_activity.is_empty():
        return pl.DataFrame(
            schema={
                _HARMONIZED_ID_COLUMN: pl.Utf8,
                "activity_name": pl.Utf8,
                "steps": pl.Int64,
                "calories": pl.Int64,
                "startDate": pl.Utf8,
                "startTime": pl.Utf8,
                "duration": pl.Int64,
            }
        )

    activity_col = "Activities" if "Activities" in daily_activity.columns else "activities"
    required_cols = {"participantId", activity_col}
    missing_cols = sorted(required_cols.difference(set(daily_activity.columns)))
    if missing_cols:
        raise ValueError(
            f"PRAES harmonizer cannot build PhysicalActivity block; missing columns: {missing_cols}"
        )

    flattened = (
        daily_activity
        .explode(activity_col)
        .unnest(activity_col)
    )
    if "name" in flattened.columns:
        flattened = flattened.filter(pl.col("name").is_not_null())

    return flattened.select(
        [
            pl.col("participantId").cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            _select_or_null(flattened, "name", pl.Utf8).alias("activity_name"),
            _select_or_null(flattened, "steps", pl.Int64).alias("steps"),
            _select_or_null(flattened, "calories", pl.Int64).alias("calories"),
            _select_or_null(flattened, "startDate", pl.Utf8).alias("startDate"),
            _select_or_null(flattened, "startTime", pl.Utf8).alias("startTime"),
            _select_or_null(flattened, "duration", pl.Int64).alias("duration"),
        ]
    ).sort([_HARMONIZED_ID_COLUMN, "startDate", "startTime"])


def _build_lab_block(analyses: pl.DataFrame) -> pl.DataFrame:
    if analyses.is_empty():
        return pl.DataFrame(schema={_HARMONIZED_ID_COLUMN: pl.Utf8, "visit": pl.Utf8})

    if "participantId" not in analyses.columns:
        raise ValueError(
            "PRAES harmonizer cannot build LabData; missing required column: participantId"
        )

    formid_expr = pl.col("formId").cast(pl.Utf8, strict=False).str.to_uppercase()
    visit_expr = (
        pl.when(formid_expr.str.contains("VISIT1"))
        .then(pl.lit("visit 1"))
        .when(formid_expr.str.contains("VISIT2"))
        .then(pl.lit("visit 2"))
        .when(formid_expr.str.contains("VISIT3"))
        .then(pl.lit("visit 3"))
        .otherwise(pl.lit(None).cast(pl.Utf8))
        .alias("visit")
    )

    value_columns = [
        col_name
        for col_name in analyses.columns
        if col_name not in {"participantId", "formId", "missingData"}
    ]

    lab = analyses.select(
        [
            pl.col("participantId").cast(pl.Utf8, strict=False).alias(_HARMONIZED_ID_COLUMN),
            visit_expr,
            *[pl.col(col_name) for col_name in value_columns],
        ]
    )
    return lab.sort([_HARMONIZED_ID_COLUMN, "visit"])


def _visit2_date_from_subject(subject: Any) -> date | None:
    if subject is None:
        return None
    visit2_date = _first_non_null(
        getattr(getattr(getattr(subject, "visit2Form", None), "content", None), "Visit2Date", None),
        getattr(getattr(getattr(subject, "visit1Form", None), "content", None), "Visit2Date", None),
    )
    return _coerce_date(visit2_date)


def _visit3_date_from_subject(subject: Any) -> date | None:
    if subject is None:
        return None
    visit3_date = _first_non_null(
        getattr(getattr(getattr(subject, "visit3Form", None), "content", None), "Visit3Date", None),
        getattr(getattr(getattr(subject, "visit1Form", None), "content", None), "Visit3Date", None),
    )
    return _coerce_date(visit3_date)


def _build_visit_timepoints_block(metadata: pl.DataFrame, subject_lookup: dict[str, Any]) -> pl.DataFrame:
    if _RAW_ID_COLUMN not in metadata.columns:
        raise ValueError(
            "PRAES harmonizer cannot build VisitTimepoints; missing required column: participantId"
        )

    ids = metadata.select(pl.col(_RAW_ID_COLUMN).cast(pl.Utf8, strict=False)).to_series().to_list()

    first_visit_dates: dict[str, date | None] = {}
    if "firstVisitDate" in metadata.columns:
        id_values = metadata[_RAW_ID_COLUMN].cast(pl.Utf8, strict=False).to_list()
        first_dates = metadata["firstVisitDate"].to_list()
        first_visit_dates = {
            str(subject_id): _coerce_date(first_date)
            for subject_id, first_date in zip(id_values, first_dates)
        }

    rows: list[dict[str, Any]] = []
    for subject_id in ids:
        sid = str(subject_id)
        subject = subject_lookup.get(sid)
        rows.append(
            {
                _HARMONIZED_ID_COLUMN: sid,
                "visit": "visit 1",
                "VisitTimepoint": first_visit_dates.get(sid),
            }
        )
        rows.append(
            {
                _HARMONIZED_ID_COLUMN: sid,
                "visit": "visit 2",
                "VisitTimepoint": _visit2_date_from_subject(subject),
            }
        )
        rows.append(
            {
                _HARMONIZED_ID_COLUMN: sid,
                "visit": "visit 3",
                "VisitTimepoint": _visit3_date_from_subject(subject),
            }
        )

    return (
        pl.DataFrame(
            rows,
            schema={
                _HARMONIZED_ID_COLUMN: pl.Utf8,
                "visit": pl.Utf8,
                "VisitTimepoint": pl.Date,
            },
        )
        .sort([_HARMONIZED_ID_COLUMN, "visit"])
    )


def harmonize_praes_dataset(dataset: Any) -> HarmonizedBlocks:
    cgm_data = getattr(dataset, "cgm_data", None)
    metadata = getattr(dataset, "metadata", None)

    if not isinstance(cgm_data, pl.DataFrame):
        raise ValueError("PRAES harmonizer expects `cgm_data` as a Polars DataFrame.")
    if not isinstance(metadata, pl.DataFrame):
        raise ValueError("PRAES harmonizer expects `metadata` as a Polars DataFrame.")

    cgm_block = _build_cgm_block(cgm_data)
    metadata_block = _build_metadata_block(metadata)
    food_source = _resolve_food_source(dataset)
    macronutrients_source = _resolve_macronutrients_source(dataset)
    food_block = _build_food_block(food_source, macronutrients_source)
    activity_source = _resolve_activity_source(dataset)
    physical_activity_block = _build_physical_activity_block(activity_source)
    analyses = _resolve_analysis_source(dataset)
    lab_block = _build_lab_block(analyses)
    subject_lookup = _resolve_subject_lookup(dataset)
    visit_timepoints_block = _build_visit_timepoints_block(metadata, subject_lookup)

    harmonized = HarmonizedBlocks(
        CGMData=cgm_block,
        MetaData=metadata_block,
        LabData=lab_block,
        Food=food_block,
        VisitTimepoints=visit_timepoints_block,
    )
    harmonized.add_block("PhysicalActivity", physical_activity_block)
    return harmonized


def harmonize_loaded_praes(praes_dataset: Any) -> HarmonizedBlocks:
    """Explicit harmonization entrypoint for a loaded PRAES ingestion dataset."""
    return harmonize_praes_dataset(praes_dataset)
