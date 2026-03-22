import os
from typing import Dict

import polars as pl

from .utils.pl_utils import is_float, validate_schema
from .aleppo_data_schemas import (
    CGM_DATA_SCHEMA,
    META_DATA_SCHEMA,
    ROSTER_SCHEMA,
    SCREENING_DATA_SCHEMA,
    HLOCAL_HBA1C_SCHEMA,
    HINSULIN_SCHEMA,
    HMEDICAL_CONDITION_SCHEMA,
    HMEDICATION_DATA_SCHEMA,
    HVISIT_INFO_SCHEMA,
    DEVICE_UPLOADS_SCHEMA,
    AleppoSubjectData
)

from .base_cgm_dataset import CGMDataset


class AleppoCGMDataset(CGMDataset):

    def __init__(self, metadata_root_folder: str = "/data/aleppo2017/Replace-BG/Data Tables/"):
        super().__init__()
        self.METADATA_SCHEMA = META_DATA_SCHEMA
        self.CGM_SCHEMA = CGM_DATA_SCHEMA
        self.bad_subjects = []
        self.cgm_data_root_folder = metadata_root_folder
        self.metadata_tables = {}
        self.metadata = self.load_metadata(metadata_root_folder, mode="summary")
        self.cgm_data = self.construct_cgm_dataframe()
        self.joined_data = self.construct_joined_dataframe()
        self.ids = self.metadata["PtID"].to_list()

    def _resolve_metadata_folder(self, metadata_folder: str | None = None) -> str:
        return metadata_folder if metadata_folder is not None else self.cgm_data_root_folder

    def load_roster(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HPtRoster.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=[
                "RecID",
                "PtID",
                "SiteOrig",
                "SiteID",
                "RandDtDaysAfterEnroll",
                "PtStatus",
                "TrtGroup",
                "AgeAsOfEnrollDt",
            ],
            dtypes=ROSTER_SCHEMA,
            null_values="NA",
        )

    def load_screening(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HScreening.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(SCREENING_DATA_SCHEMA.keys()),
            dtypes=SCREENING_DATA_SCHEMA,
            null_values="NA",
        )

    def load_hba1c(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HLocalHbA1c.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(HLOCAL_HBA1C_SCHEMA.keys()),
            dtypes=HLOCAL_HBA1C_SCHEMA,
            null_values="NA",
        )

    def load_medical_condition(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HMedicalCondition.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(HMEDICAL_CONDITION_SCHEMA.keys()),
            dtypes=HMEDICAL_CONDITION_SCHEMA,
            null_values="NA",
        )

    def load_insulin(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HInsulin.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(HINSULIN_SCHEMA.keys()),
            dtypes=HINSULIN_SCHEMA,
            null_values="NA",
        )

    def load_medication(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HMedication.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(HMEDICATION_DATA_SCHEMA.keys()),
            dtypes=HMEDICATION_DATA_SCHEMA,
            null_values="NA",
        )

    def load_visit_info(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HVisitInfo.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(HVISIT_INFO_SCHEMA.keys()),
            dtypes=HVISIT_INFO_SCHEMA,
            null_values="NA",
        )

    def load_device_uploads(self, metadata_folder: str | None = None) -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        data_path = os.path.join(metadata_folder, "HDeviceUploads.txt")
        return pl.read_csv(
            data_path,
            separator="|",
            new_columns=list(DEVICE_UPLOADS_SCHEMA.keys()),
            dtypes=DEVICE_UPLOADS_SCHEMA,
            null_values="NA",
        )

    def load_metadata_tables(self, metadata_folder: str | None = None) -> Dict[str, pl.DataFrame]:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        tables = {
            "roster": self.load_roster(metadata_folder),
            "screening": self.load_screening(metadata_folder),
            "hba1c": self.load_hba1c(metadata_folder),
            "medical_condition": self.load_medical_condition(metadata_folder),
            "insulin": self.load_insulin(metadata_folder),
            "medication": self.load_medication(metadata_folder),
        }
        self.metadata_tables = tables
        return tables

    def _normalize_ptid_column(self, table: pl.DataFrame) -> pl.DataFrame:
        if "PtID" in table.columns:
            return table
        if "PtId" in table.columns:
            return table.rename({"PtId": "PtID"})
        raise ValueError(f"No subject id column found. Columns: {table.columns}")

    def _group_metadata_table(self, table: pl.DataFrame, prefix: str) -> pl.DataFrame:
        grouped_table = self._normalize_ptid_column(table)
        agg_columns = [
            pl.col(col_name).alias(f"{prefix}__{col_name}")
            for col_name in grouped_table.columns
            if col_name != "PtID"
        ]
        if not agg_columns:
            return grouped_table.select("PtID").unique()
        return grouped_table.group_by("PtID").agg(agg_columns)

    def _build_summary_metadata(self, tables: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        df_roster = tables["roster"]
        df_screening = tables["screening"]
        df_hba1c = tables["hba1c"]

        df_hba1c = df_hba1c.group_by("PtID").agg(pl.mean("HbA1cTestRes"))
        # select PtID and Height and Weight from df_screening and calculate BMI
        df_screening = df_screening.select(["PtID", "Height", "Weight"])
        df_screening = df_screening.with_columns(
            (df_screening["Weight"] / ((df_screening["Height"] / 100) ** 2)).alias("BMI")
        )

        # add a column "Type of Diabetes":pl.Utf8 to the dataframe each entry should be "T1DM"
        df_roster = df_roster.with_columns(
            pl.Series(["T1DM"] * len(df_roster), dtype=pl.Utf8).alias("Type of Diabetes")
        )
        # add a column "HbA1c" to the dataframe
        df_hba1c = df_hba1c.select(["PtID", "HbA1cTestRes"])
        # rename HbA1cTestRes to HbA1c
        df_hba1c = df_hba1c.rename({"HbA1cTestRes": "HbA1c"})
        #

        df_meta = df_roster.join(df_screening, on="PtID", how="left")
        df_meta = df_meta.join(df_hba1c, on="PtID", how="left")
        return self.clean_metadata(df_meta)

    def load_metadata(self, metadata_folder: str | None = None, mode: str = "all") -> pl.DataFrame:
        metadata_folder = self._resolve_metadata_folder(metadata_folder)
        tables = self.load_metadata_tables(metadata_folder)

        if mode == "summary":
            data = self._build_summary_metadata(tables)
            validate_schema(data, self.METADATA_SCHEMA)
            return data

        if mode != "all":
            raise ValueError("mode must be either 'all' or 'summary'")

        df_roster = tables["roster"].with_columns(
            pl.Series(["T1DM"] * len(tables["roster"]), dtype=pl.Utf8).alias("Type of Diabetes")
        )
        df_screening_grouped = self._group_metadata_table(tables["screening"], "screening")
        df_hba1c_grouped = self._group_metadata_table(tables["hba1c"], "hba1c")
        df_medical_condition_grouped = self._group_metadata_table(
            tables["medical_condition"], "medical_condition"
        )
        df_insulin_grouped = self._group_metadata_table(tables["insulin"], "insulin")
        df_medication_grouped = self._group_metadata_table(tables["medication"], "medication")

        df_meta = (
            df_roster.join(df_screening_grouped, on="PtID", how="left")
            .join(df_hba1c_grouped, on="PtID", how="left")
            .join(df_medical_condition_grouped, on="PtID", how="left")
            .join(df_insulin_grouped, on="PtID", how="left")
            .join(df_medication_grouped, on="PtID", how="left")
            .with_columns(
                [
                    pl.col("screening__Height")
                    .list.first()
                    .cast(pl.Float32, strict=False)
                    .alias("Height"),
                    pl.col("screening__Weight")
                    .list.first()
                    .cast(pl.Float32, strict=False)
                    .alias("Weight"),
                    (
                        pl.col("screening__Weight").list.first()
                        / ((pl.col("screening__Height").list.first() / 100) ** 2)
                    )
                    .cast(pl.Float32, strict=False)
                    .alias("BMI"),
                    pl.col("hba1c__HbA1cTestRes")
                    .list.mean()
                    .cast(pl.Float64, strict=False)
                    .alias("HbA1c"),
                ]
            )
        )
        data = self.clean_metadata(df_meta)
        validate_schema(data, self.METADATA_SCHEMA)
        return data

    def clean_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        for col_name in metadata.columns:
            desired_dtype = self.METADATA_SCHEMA.get(col_name, None)
            if desired_dtype is pl.Float64:
                metadata = metadata.with_columns(
                    metadata[col_name].map_elements(
                        lambda x: float(x) if is_float(x) else None, 
                        return_dtype=pl.Float64
                    )
                )
            elif str(desired_dtype) == str(pl.Datetime(time_unit='us', time_zone=None)):
                format_0= "%m/%d/%y %H:%M"
                format_1="%Y-%m-%d %H:%M:%S"
                try:
                    metadata = metadata.with_columns(metadata[col_name].str.strptime(dtype=pl.Datetime, format=format_0))

                except:
                    metadata = metadata.with_columns(metadata[col_name].str.strptime(dtype=pl.Datetime, format=format_1))
            elif desired_dtype == pl.Utf8:
                metadata = metadata.with_columns(
                    metadata[col_name].cast(pl.Utf8)
                )
            elif desired_dtype == pl.Boolean:
                #substitute yes/no with True/False
                metadata = metadata.with_columns(
                    metadata[col_name].map_elements(
                        lambda x: True if x == "yes" else False,
                        return_dtype=pl.Boolean
                    )
                )
            # For other types, you can add more conditions here if needed
        #filter metadata to remove rows containing bad subjects
        filtered_metadata = metadata.filter(~metadata["PtID"].is_in(self.bad_subjects))
        return filtered_metadata

    def get_subject_data(self, subject_id: str) -> AleppoSubjectData:
        # Implementation for getting subject data in the Aleppo dataset
        subject_metadata = self.metadata.filter(self.metadata["PtID"] == subject_id)
        return AleppoSubjectData(
            subject_metadata, cgm_data_root_folder=self.cgm_data_root_folder
        )

    def _load_all_cgm_readings(self) -> pl.DataFrame:
        data_path = os.path.join(self.cgm_data_root_folder, "HDeviceCGM.txt")
        return (
            pl.read_csv(
                data_path,
                separator="|",
                null_values="NA",
                columns=[
                    "PtID",
                    "DeviceDtTmDaysFromEnroll",
                    "DeviceTm",
                    "DexInternalTm",
                    "GlucoseValue",
                ],
            )
            .with_columns(
                [
                    pl.col("PtID").cast(pl.Int32, strict=False),
                    pl.col("GlucoseValue")
                    .cast(pl.Float64, strict=False)
                    .alias("glucemia"),
                    pl.col("DeviceTm")
                    .str.strptime(pl.Time, "%H:%M:%S", strict=False)
                    .alias("time_series"),
                    pl.col("DexInternalTm")
                    .str.strptime(pl.Time, "%H:%M:%S", strict=False)
                    .alias("time_series_internal"),
                    pl.col("DeviceDtTmDaysFromEnroll")
                    .cast(pl.Int64, strict=False)
                    .alias("days_from_enroll"),
                ]
            )
            .select(
                [
                    "PtID",
                    "glucemia",
                    "time_series",
                    "time_series_internal",
                    "days_from_enroll",
                ]
            )
        )

    def construct_cgm_dataframe(self):
        # Build per-subject list columns from one full-file scan of HDeviceCGM.
        cgm_long = self._load_all_cgm_readings()
        grouped = (
            cgm_long.sort(["PtID", "days_from_enroll", "time_series"])
            .group_by("PtID", maintain_order=True)
            .agg(
                [
                    pl.col("glucemia").alias("glucemia_series"),
                    pl.col("time_series"),
                    pl.col("time_series_internal"),
                    pl.col("days_from_enroll"),
                ]
            )
        )

        all_subjects = self.metadata.select(pl.col("PtID").cast(pl.Int32))
        cgm_data = all_subjects.join(grouped, on="PtID", how="left").with_columns(
            [
                pl.when(pl.col("glucemia_series").is_null())
                .then(pl.lit([], dtype=pl.List(pl.Float64)))
                .otherwise(pl.col("glucemia_series"))
                .alias("glucemia_series"),
                pl.when(pl.col("time_series").is_null())
                .then(pl.lit([], dtype=pl.List(pl.Time)))
                .otherwise(pl.col("time_series"))
                .alias("time_series"),
                pl.when(pl.col("time_series_internal").is_null())
                .then(pl.lit([], dtype=pl.List(pl.Time)))
                .otherwise(pl.col("time_series_internal"))
                .alias("time_series_internal"),
                pl.when(pl.col("days_from_enroll").is_null())
                .then(pl.lit([], dtype=pl.List(pl.Int64)))
                .otherwise(pl.col("days_from_enroll"))
                .alias("days_from_enroll"),
            ]
        )
        return cgm_data

    def construct_joined_dataframe(self):
        df_roster_selected = self.metadata.select([
            'PtID',
            'PtStatus',
            'Type of Diabetes',
            'BMI',
            'HbA1c',
            pl.col('AgeAsOfEnrollDt').alias('Age')  # Rename the column
            ])

        # Perform the join operation
        df_combined = self.cgm_data.join(
            df_roster_selected,
            on='PtID',
            how='left'  # This is a left join
        )

        # Standardize glucose list column name for downstream unification.
        if 'glucemia_series' in df_combined.columns and 'CGM' not in df_combined.columns:
            df_combined = df_combined.rename({'glucemia_series': 'CGM'})

        return df_combined

    def concat(self, other):
        pass
