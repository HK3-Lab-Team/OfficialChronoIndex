from collections.abc import Sequence
import math

import polars as pl
import pycatch22

_CATCH22_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]
_CATCH22_FEATURE_DTYPE = pl.Struct([pl.Field(name, pl.Float64) for name in _CATCH22_NAMES])

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


def _empty_catch22_dict() -> dict[str, float | None]:
    return {name: None for name in _CATCH22_NAMES}


def catch22(cgm_series):
    if cgm_series is None:
        return _empty_catch22_dict()

    cleaned: list[float] = []
    for value in cgm_series:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        cleaned.append(parsed)

    # pycatch22 expects real-valued series without nulls.
    if len(cleaned) < 3:
        return _empty_catch22_dict()

    try:
        catch = pycatch22.catch22_all(cleaned)
        values = {k: v for k, v in zip(catch["names"], catch["values"])}
        return {
            name: (float(values[name]) if values.get(name) is not None else None)
            for name in _CATCH22_NAMES
        }
    except Exception:
        return _empty_catch22_dict()

def compute_catch22_features(
    data: pl.DataFrame | pl.Series | Sequence[float],
    *,
    id_col: str | None = None,
    cgm_col: str | None = None,
    cgm_time_col: str | None = "CGMTime",
    dataset_col: str = "dataset",
) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        if data.height == 0:
            return pl.DataFrame(
                schema={
                    "dataset": pl.Utf8,
                    "Id": pl.Utf8,
                    "date": pl.Date,
                    "CGM": pl.List(pl.Float64),
                    "catch22_features": _CATCH22_FEATURE_DTYPE,
                }
            )

        id_col = id_col or _infer_column(data.columns, _ID_COLUMN_CANDIDATES, "subject id")
        cgm_col = cgm_col or _infer_column(data.columns, _CGM_COLUMN_CANDIDATES, "CGM")

        resolved_time_col: str | None = None
        if cgm_time_col is not None and cgm_time_col in data.columns:
            resolved_time_col = cgm_time_col
        elif cgm_time_col is None:
            for candidate in _CGM_TIME_COLUMN_CANDIDATES:
                if candidate in data.columns:
                    resolved_time_col = candidate
                    break
        elif cgm_time_col == "CGMTime" and cgm_time_col not in data.columns:
            # Fallback to known timestamp aliases when default CGMTime is absent.
            for candidate in _CGM_TIME_COLUMN_CANDIDATES:
                if candidate in data.columns:
                    resolved_time_col = candidate
                    break

        group_keys = [id_col]
        if dataset_col in data.columns:
            group_keys = [dataset_col, id_col]

        working = data
        if resolved_time_col is not None:
            cgm_is_list = _is_list_dtype(working.schema[cgm_col])
            time_is_list = _is_list_dtype(working.schema[resolved_time_col])

            if cgm_is_list and time_is_list:
                working = working.explode(cgm_col, resolved_time_col)
            elif cgm_is_list != time_is_list:
                raise ValueError(
                    f"Columns '{cgm_col}' and '{resolved_time_col}' must both be list-like "
                    "or both scalar."
                )

            working = (
                working
                .with_columns(
                    [
                        pl.col(cgm_col).cast(pl.Float64, strict=False).alias(cgm_col),
                        pl.col(resolved_time_col).cast(pl.Datetime, strict=False).alias(resolved_time_col),
                    ]
                )
                .filter(pl.col(cgm_col).is_not_null() & pl.col(cgm_col).is_finite())
                .filter(pl.col(resolved_time_col).is_not_null())
            )

            tsdf = (
                working
                .group_by([*group_keys, pl.col(resolved_time_col).dt.date().alias("date")])
                .agg(pl.col(cgm_col).alias("CGM"))
                .sort([*group_keys, "date"])
            )
        else:
            if _is_list_dtype(working.schema[cgm_col]):
                working = working.explode(cgm_col)

            working = (
                working
                .with_columns(pl.col(cgm_col).cast(pl.Float64, strict=False).alias(cgm_col))
                .filter(pl.col(cgm_col).is_not_null() & pl.col(cgm_col).is_finite())
            )

            tsdf = (
                working
                .group_by(group_keys)
                .agg(pl.col(cgm_col).alias("CGM"))
                .sort(group_keys)
            )

        return tsdf.with_columns(
            catch22_features=pl.col("CGM").map_elements(
                catch22,
                return_dtype=_CATCH22_FEATURE_DTYPE,
                skip_nulls=False,
            )
        )

    if isinstance(data, pl.Series):
        values = data.to_list()
    else:
        values = list(data)

    return pl.DataFrame([{"CGM": values}]).with_columns(
        catch22_features=pl.col("CGM").map_elements(
            catch22,
            return_dtype=_CATCH22_FEATURE_DTYPE,
            skip_nulls=False,
        )
    )
