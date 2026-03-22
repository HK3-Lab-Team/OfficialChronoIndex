import polars as pl
from collections.abc import Sequence





class TimeSeriesManager:
    def __init__(self, dataframe, is_subject_form: bool = True, namespace_by_dataset: bool = True):
        self.df = dataframe
        self.is_subject_form = is_subject_form
        self.namespace_by_dataset = namespace_by_dataset

    def _resolve_group_keys(self, patient_identifier_col: str | Sequence[str]) -> list[str]:
        keys = (
            [patient_identifier_col]
            if isinstance(patient_identifier_col, str)
            else list(patient_identifier_col)
        )
        if keys == ["unique_id"] and "unique_id" not in self.df.columns and "Id" in self.df.columns:
            keys = ["Id"]
        if self.namespace_by_dataset and "dataset" in self.df.columns and "dataset" not in keys:
            keys = ["dataset", *keys]
        return keys

    def sort_df(self, patient_identifier_col: str | Sequence[str] = "unique_id"):
        # Create a sort key from the first timestamp in each CGMTime list.
        group_keys = [
            col for col in self._resolve_group_keys(patient_identifier_col)
            if col in self.df.columns
        ]
        sort_cols = [*group_keys, "start_date"]
        self.df = (
            self.df
            .with_columns(pl.col("CGMTime").list.first().alias("start_date"))
            .sort(by=sort_cols, descending=False)
        )

    def to_subject_df(self, patient_identifier_col: str | Sequence[str] = "unique_id"):
        if self.is_subject_form:
            raise ValueError("DataFrame is already in subject form.")

        group_keys = self._resolve_group_keys(patient_identifier_col)
        self.sort_df(patient_identifier_col=patient_identifier_col)

        # Group by patient keys and concatenate daily lists back into subject-level lists.
        self.df = (
            self.df
            .group_by(group_keys)
            .agg([
                pl.col("CGM").flatten().alias("CGM"),
                pl.col("CGMTime").flatten().alias("CGMTime"),
            ])
        )

        self.is_subject_form = True

    def to_daily_df(
        self,
        patient_identifier_col: str | Sequence[str] = "unique_id",
        temporal_resolution: str = "daily",
    ):
        if not self.is_subject_form:
            raise ValueError("DataFrame is already in daily form.")

        group_keys = self._resolve_group_keys(patient_identifier_col)

        # Split each subject row into temporal segments.
        match temporal_resolution:
            case "daily":
                self.df = (
                    self.df
                    .explode("CGM", "CGMTime")
                    .group_by([*group_keys, pl.col("CGMTime").dt.date().alias("date")])
                    .agg([
                        pl.col("CGM").alias("CGM"),
                        pl.col("CGMTime").alias("CGMTime"),
                    ])
                )
            case "hourly":
                self.df = (
                    self.df
                    .explode("CGM", "CGMTime")
                    .group_by([
                        *group_keys,
                        pl.col("CGMTime").dt.hour().alias("hour"),
                        pl.col("CGMTime").dt.date().alias("date"),
                    ])
                    .agg([
                        pl.col("CGM").alias("CGM"),
                        pl.col("CGMTime").alias("CGMTime"),
                    ])
                )
            case "day_night":
                self.df = (
                    self.df
                    .explode("CGM", "CGMTime")
                    .with_columns((pl.col("CGMTime").dt.hour() > 12).alias("AM_PM"))
                    .group_by([
                        *group_keys,
                        pl.col("AM_PM"),
                        pl.col("CGMTime").dt.date().alias("date"),
                    ])
                    .agg([
                        pl.col("CGM").alias("CGM"),
                        pl.col("CGMTime").alias("CGMTime"),
                    ])
                )
            case _:
                raise ValueError("Invalid temporal resolution")

        self.is_subject_form = False

    def to_individual_recordings(self, patient_identifier_col: str | Sequence[str] = "unique_id"):
        """
        Convert list-based CGM rows into one row per timestamped CGM recording.

        Works with unified subject-level data (lists in `CGM`/`CGMTime`) and is a
        no-op for already scalar recording-level inputs.
        """
        if "CGM" not in self.df.columns or "CGMTime" not in self.df.columns:
            raise ValueError("Expected columns 'CGM' and 'CGMTime' in dataframe.")

        group_keys = [
            col for col in self._resolve_group_keys(patient_identifier_col)
            if col in self.df.columns
        ]

        cgm_is_list = str(self.df.schema["CGM"]).lower().startswith("list")
        cgm_time_is_list = str(self.df.schema["CGMTime"]).lower().startswith("list")

        if cgm_is_list and cgm_time_is_list:
            self.df = self.df.explode("CGM", "CGMTime")
        elif cgm_is_list != cgm_time_is_list:
            raise ValueError(
                "Columns 'CGM' and 'CGMTime' must both be list-like or both scalar."
            )

        sort_cols = [*group_keys, "CGMTime"] if group_keys else ["CGMTime"]
        self.df = self.df.sort(sort_cols)
        self.is_subject_form = False

    def cgms_around_events(
        self,
        events_df: pl.DataFrame,
        *,
        event_column: str,
        patient_identifier_col: str | Sequence[str] = "unique_id",
        hour_window: float = 2.0,
    ) -> pl.DataFrame:
        """
        Return event-level rows enriched with CGM values around each event.

        For each event timestamp in `events_df[event_column]`, collect:
        - `CGMBefore` / `CGMTimeBefore`: CGM recordings in `[event - hour_window, event]`
        - `CGMAfter` / `CGMTimeAfter`: CGM recordings in `(event, event + hour_window]`
        """
        if "CGM" not in self.df.columns or "CGMTime" not in self.df.columns:
            raise ValueError("Expected columns 'CGM' and 'CGMTime' in CGM dataframe.")
        if event_column not in events_df.columns:
            raise ValueError(f"Event column '{event_column}' not found in events dataframe.")
        if hour_window <= 0:
            raise ValueError("hour_window must be > 0.")

        # Normalize CGM table to recording-level rows.
        cgm_df = self.df
        cgm_is_list = str(cgm_df.schema["CGM"]).lower().startswith("list")
        cgm_time_is_list = str(cgm_df.schema["CGMTime"]).lower().startswith("list")
        if cgm_is_list and cgm_time_is_list:
            cgm_df = cgm_df.explode("CGM", "CGMTime")
        elif cgm_is_list != cgm_time_is_list:
            raise ValueError(
                "Columns 'CGM' and 'CGMTime' must both be list-like or both scalar."
            )

        candidate_keys = self._resolve_group_keys(patient_identifier_col)
        join_keys = [
            key for key in candidate_keys
            if key in cgm_df.columns and key in events_df.columns
        ]
        if not join_keys:
            raise ValueError(
                "No shared identifier columns between CGM dataframe and events dataframe. "
                f"Tried candidate keys: {candidate_keys}"
            )

        window_seconds = int(round(hour_window * 3600))
        events_tagged = (
            events_df
            .with_row_count("__event_idx")
            .with_columns(pl.col(event_column).cast(pl.Datetime, strict=False).alias(event_column))
        )

        cgm_recordings = cgm_df.with_columns(
            [
                pl.col("CGM").cast(pl.Float64, strict=False).alias("CGM"),
                pl.col("CGMTime").cast(pl.Datetime, strict=False).alias("CGMTime"),
            ]
        )

        joined = events_tagged.join(
            cgm_recordings.select([*join_keys, "CGM", "CGMTime"]),
            on=join_keys,
            how="left",
        )

        before_agg = (
            joined
            .filter(pl.col("CGMTime").is_not_null() & pl.col(event_column).is_not_null())
            .filter(
                (pl.col("CGMTime") <= pl.col(event_column))
                & (pl.col("CGMTime") >= (pl.col(event_column) - pl.duration(seconds=window_seconds)))
            )
            .sort(["__event_idx", "CGMTime"])
            .group_by("__event_idx")
            .agg(
                [
                    pl.col("CGM").alias("CGMBefore"),
                    pl.col("CGMTime").alias("CGMTimeBefore"),
                ]
            )
        )

        after_agg = (
            joined
            .filter(pl.col("CGMTime").is_not_null() & pl.col(event_column).is_not_null())
            .filter(
                (pl.col("CGMTime") > pl.col(event_column))
                & (pl.col("CGMTime") <= (pl.col(event_column) + pl.duration(seconds=window_seconds)))
            )
            .sort(["__event_idx", "CGMTime"])
            .group_by("__event_idx")
            .agg(
                [
                    pl.col("CGM").alias("CGMAfter"),
                    pl.col("CGMTime").alias("CGMTimeAfter"),
                ]
            )
        )

        return (
            events_tagged
            .join(before_agg, on="__event_idx", how="left")
            .join(after_agg, on="__event_idx", how="left")
            .drop("__event_idx")
        )

