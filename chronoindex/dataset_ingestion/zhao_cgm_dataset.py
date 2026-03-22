import polars as pl
from pathlib import Path
from typing import Optional

from .utils.pl_utils import is_float, validate_schema
from .base_cgm_dataset import CGMDataset
from .zhao_data_schemas import CGM_SCHEMA, METADATA_SCHEMA, ZhaoSubjectData


class ZhaoCGMDataset(CGMDataset):
    def __init__(
        self,
        metadata_file: str = "./data/zhao2023/Shanghai_T2DM_Summary.xlsx",
        cgm_folder: Optional[str] = None,
    ):
        super().__init__()
        self.METADATA_SCHEMA = METADATA_SCHEMA
        self.CGM_SCHEMA = CGM_SCHEMA
        self.bad_subjects = ["2029_0_20210526"]
        metadata_path = Path(metadata_file)
        if cgm_folder is None:
            self.cgm_folder = metadata_path.parent / "Shanghai_T2DM"
        else:
            self.cgm_folder = Path(cgm_folder)
        self.metadata = self.load_metadata(metadata_file)
        self.cgm_data = self.construct_cgm_dataframe()
        self.joined_data = self.construct_joined_dataframe()
        self.ids = self.metadata["Patient Number"].to_list()

    def load_metadata(self, metadata_file: str) -> pl.DataFrame:
        # Implementation for loading metadata specific to the Shanghai dataset
        metadata = pl.read_excel(source=metadata_file)
        metadata = self.clean_metadata(metadata)
        metadata = metadata.rename({"Age (years)": "Age", "BMI (kg/m2)": "BMI"})
    
        metadata = metadata.with_columns(pl.col("Age").cast(pl.Int32))
        metadata = metadata.with_columns(pl.col("BMI").cast(pl.Float32))
        validate_schema(metadata, self.METADATA_SCHEMA)
        return metadata

    def clean_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        for col_name in metadata.columns:
            desired_dtype = self.METADATA_SCHEMA.get(col_name, None)
            if desired_dtype is pl.Float64:
                metadata = metadata.with_columns(
                    metadata[col_name].map_elements(
                        lambda x: float(x) if is_float(x) else None,
                        return_dtype=pl.Float64,
                    )
                )
            elif str(desired_dtype) == str(pl.Datetime(time_unit="us", time_zone=None)):
                format_0 = "%m/%d/%y %H:%M"
                format_1 = "%Y-%m-%d %H:%M:%S"
                try:
                    metadata = metadata.with_columns(
                        metadata[col_name].str.strptime(
                            dtype=pl.Datetime, format=format_0
                        )
                    )

                except:
                    metadata = metadata.with_columns(
                        metadata[col_name].str.strptime(
                            dtype=pl.Datetime, format=format_1
                        )
                    )
            elif desired_dtype == pl.Utf8:
                metadata = metadata.with_columns(metadata[col_name].cast(pl.Utf8))
            elif desired_dtype == pl.Boolean:
                # substitute yes/no with True/False
                metadata = metadata.with_columns(
                    metadata[col_name].map_elements(
                        lambda x: True if x == "yes" else False, return_dtype=pl.Boolean
                    )
                )
            # For other types, you can add more conditions here if needed
        # filter metadata to remove rows containing bad subjects
        filtered_metadata = metadata.filter(
            ~metadata["Patient Number"].is_in(self.bad_subjects)
        )
        return filtered_metadata

    def get_subject_data(self, subject_id: str) -> ZhaoSubjectData:
        # Fetch and return subject data for a specific subject ID
        subject_metadata = self.metadata.filter(
            self.metadata["Patient Number"] == subject_id
        )
        return ZhaoSubjectData(subject_metadata, cgm_folder=str(self.cgm_folder))

    def construct_cgm_dataframe(self):
        # Construct multi-row DataFrame by concatenating cgm_to_single_row outputs
        cgm_rows = [
            self.get_subject_data(subject_id).cgm_to_single_row()
            for subject_id in self.metadata["Patient Number"]
        ]
        return pl.concat(cgm_rows, how="vertical")

    def construct_joined_dataframe(self):
        # joins the metadata and cgm_data frames using the Patient Number column
        joined_df = self.metadata.join(self.cgm_data, on="Patient Number", how="inner")
        # rename 'Date' column to 'time_series'
        joined_df = joined_df.rename({"Date": "time_series"})
        #Formula HbA1c(%)=(HbA1c(mmol/mol)×0.09148)+2.152
        joined_df = joined_df.with_columns(pl.col('HbA1c (mmol/mol)').map_batches(lambda x: (x*0.09148)+2.152).alias('HbA1c'))
        joined_df = joined_df.drop(["HbA1c (mmol/mol)"])
        return joined_df

    def concat(self, other):
        if isinstance(other, ZhaoCGMDataset):
            # Concatenate with another ZhaoCGMDataset
            self.metadata = pl.concat([self.metadata, other.metadata], how="vertical")
            self.construct_cgm_dataframe()

        elif isinstance(other, ZhaoSubjectData):
            # Concatenate with a ZhaoSubjectData instance
            new_metadata = other.metadata
            self.metadata = pl.concat([self.metadata, new_metadata], how="vertical")
            new_cgm_data = other.cgm_to_single_row()
            self.cgm_data = pl.concat([self.cgm_data, new_cgm_data], how="vertical")

        else:
            raise ValueError(
                "The provided input is neither a ZhaoCGMDataset nor a ZhaoSubjectData instance."
            )
