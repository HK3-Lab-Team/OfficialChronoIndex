import polars as pl

import os
from typing import Optional


from .utils.pl_utils import is_float, validate_schema
from .base_subject_data import SubjectData

CGM_SCHEMA = {
    "id": pl.Int64,
    "time_series": pl.List(inner=pl.Time),
    "CGM": pl.List(inner=pl.Float64),
}

CGM_JOINED_SCHEMA = {
    "id": pl.Int64,
    "time_series": pl.List(inner=pl.Float64),
    "CGM": pl.List(inner=pl.Time),
    "gender": pl.Int32,
    "Age": pl.Int32,
    "BMI": pl.Float32,
    "glycaemia": pl.Float32,
    "HbA1c": pl.Float64,
    "follow.up": pl.Int32,
    "Type of Diabetes": pl.Utf8,
}

METADATA_SCHEMA = {
    "id": pl.Int64,
    "gender": pl.Int32,
    "Age": pl.Int32,
    "BMI": pl.Float32,
    "glycaemia": pl.Float32,
    "HbA1c": pl.Float64,
    "follow.up": pl.Int32,
    "T2DM": pl.Boolean,
}


transformation_rules = {}


class ColasSubjectData(SubjectData):
    def __init__(
        self,
        metadata: pl.DataFrame,
        cgm_readings: Optional[pl.DataFrame] = None,
        cgm_data_root_folder: Optional[str] = None,
    ):
        self.METADATA_SCHEMA = METADATA_SCHEMA
        self.CGM_SCHEMA = CGM_SCHEMA
        if cgm_data_root_folder is None:
            current_path = os.getcwd()
            self.cgm_data_root_folder = os.path.join(current_path, "data", "colas2019")
        else:
            self.cgm_data_root_folder = cgm_data_root_folder
        self.subject_id = metadata["id"][0]
        if cgm_readings is not None:
            validate_schema(cgm_readings, self.CGM_SCHEMA)
        else:
            cgm_readings = self.load_cgm_from_subject_id(self.subject_id)
        
        
        
        super().__init__(metadata, cgm_readings)

    def fix_cgm_types(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        for col_name in cgm_readings.columns:
            desired_dtype = self.CGM_SCHEMA.get(col_name, None)
            if desired_dtype is pl.Float64:
                cgm_readings = cgm_readings.with_columns(
                    cgm_readings[col_name].map_elements(
                        lambda x: float(x) if is_float(x) else None,
                        return_dtype=pl.Float64,
                    )
                )
            elif str(desired_dtype) == str(pl.Datetime(time_unit="us", time_zone=None)):
                format_0 = "%m/%d/%y %H:%M"
                format_1 = "%Y-%m-%d %H:%M:%S"
                try:
                    cgm_readings = cgm_readings.with_columns(
                        cgm_readings[col_name].str.strptime(
                            dtype=pl.Datetime, format=format_0
                        )
                    )

                except:
                    cgm_readings = cgm_readings.with_columns(
                        cgm_readings[col_name].str.strptime(
                            dtype=pl.Datetime, format=format_1
                        )
                    )
            elif desired_dtype == pl.Utf8:
                cgm_readings = cgm_readings.with_columns(
                    cgm_readings[col_name].cast(pl.Utf8)
                )
            # For other types, you can add more conditions here if needed

        return cgm_readings

    def upsample_cgm(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        cgm_readings = cgm_readings.set_sorted()
        df_upsampled = (
            cgm_readings.upsample(time_column="time_series", every="15m")
            .interpolate()
            .fill_null(strategy="forward")
        )
        # Forward fill the missing values
        return df_upsampled

    def remove_none_rows(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        # Start with a mask of all False values (indicating all rows to be dropped)
        combined_mask = pl.col(cgm_readings.columns[0]).is_null()
        # Combine the masks for each column
        for col in cgm_readings.columns:
            combined_mask = combined_mask | cgm_readings[col].is_not_null()
        # Filter the original dataframe using the combined mask
        cgm_readings = cgm_readings.filter(combined_mask)
        return cgm_readings

    def load_cgm_from_subject_id(self, subject_id: str) -> pl.DataFrame:
        data_path = os.path.join(self.cgm_data_root_folder, f"case  {subject_id}.csv")
        cgm_readings = pl.read_csv(
            data_path,
            null_values={"glucemia": "NA"},
            dtypes={"hora": pl.Utf8, "glucemia": pl.Int64},
        )
        time_column = (
            cgm_readings["hora"].str.strptime(pl.Time, "%H:%M:%S").alias("time_series")
        )
        cgm_readings = cgm_readings.with_columns(time_column.alias("time_series"))
        # Convert 'hora' to DateTime
        cgm_readings = cgm_readings.drop("hora")

        # Applying transformation rules to rename columns
        # df.rename(columns=transformation_rules, inplace=True)

        # Convert pandas dataframe to polars dataframe
        # cgm_readings = pl.from_pandas(df)
        cgm_readings = self.fix_cgm_types(cgm_readings)

        original_height = cgm_readings.height

        # Create a boolean mask indicating non-null values
        cgm_readings = self.remove_none_rows(cgm_readings)

        # Print if rows have been deleted
        deleted_rows = original_height - cgm_readings.height
        if deleted_rows > 0:
            print(
                f"For subject {subject_id}, {deleted_rows} rows with all null values have been deleted."
            )
        #cgm_readings_filled = self.upsample_cgm(cgm_readings)

        cgm_readings_filled = cgm_readings  # self.upsample_cgm(cgm_readings)

        # validate_schema(cgm_readings_filled, self.CGM_SCHEMA)
        if cgm_readings_filled.height - cgm_readings.height < 0:
            print(
                "Upsampling failed, for {}, with a difference of {} rows".format(
                    subject_id, cgm_readings_filled.height - cgm_readings.height
                )
            )
            print(cgm_readings["Date"].to_list())
            print(cgm_readings_filled["Date"].to_list())
        return cgm_readings_filled

    def cgm_to_single_row(self) -> pl.DataFrame:
        # Convert CGM readings to a single row DataFrame
        glucose_list = self.cgm_readings["glucemia"].to_list()
        cgm_series = pl.DataFrame(
            data={
                "id": [self.subject_id],
                "CGM": [glucose_list],
                "time_series": [self.cgm_readings["time_series"].to_list()],
            },
            schema=CGM_SCHEMA,
        )
        return cgm_series

    def single_row_to_cgm(self, single_row: pl.DataFrame) -> pl.DataFrame:
        # Convert single row DataFrame back to CGM readings
        validate_schema(single_row, self.list_schema)
        data_dict = {col: single_row[col][0] for col in single_row.columns}
        data_dict.pop("id")
        return pl.DataFrame(data_dict, schema=self.CGM_SCHEMA)

    def __repr__(self):
        return f"<ColasSubjectData: {self.metadata.width} metadata columns, {self.cgm_readings.height} cgm readings>"
