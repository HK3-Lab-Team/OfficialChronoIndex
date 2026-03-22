from pathlib import Path
import polars as pl

from .utils.pl_utils import is_float, validate_schema
from .base_cgm_dataset import CGMDataset
from .colas_data_schemas import CGM_SCHEMA, METADATA_SCHEMA, ColasSubjectData
from datetime import datetime, timedelta


def _count_days(time_samples):
    # Assume `time_samples` is a sorted list of time values.
    time_deltas = [0]
    if not time_samples:
        return 0, []

    day_count = 0
    for i in range(len(time_samples) - 1):
        current_time = time_samples[i]
        next_time = time_samples[i + 1]
        if current_time >= next_time:
            day_count += 1
        time_deltas.append(day_count)

    return day_count, time_deltas


class ColasCGMDataset(CGMDataset):
    def __init__(self, metadata_root_folder: str = "/data/colas2019/S1"):
        super().__init__()
        self.METADATA_SCHEMA = METADATA_SCHEMA
        self.CGM_SCHEMA = CGM_SCHEMA
        self.bad_subjects = [209]
        self.metadata, self.cgm_data_root_folder = self.load_metadata(metadata_root_folder)
        self.cgm_data = self.construct_cgm_dataframe()
        self.joined_data = self.construct_joined_dataframe()
        self.ids = self.metadata["id"].to_list()

    def _resolve_metadata_path(self, metadata_folder: str) -> Path:
        metadata_path = Path(metadata_folder)
        if not metadata_path.is_dir():
            return metadata_path

        txt_candidate = metadata_path / "clinical_data.txt"
        if txt_candidate.exists():
            return txt_candidate

        csv_candidate = metadata_path / "clinical_data.csv"
        if csv_candidate.exists():
            return csv_candidate

        raise FileNotFoundError(
            f"No metadata file found in {metadata_path}. Expected clinical_data.txt or clinical_data.csv."
        )

    def _read_metadata_table(self, metadata_path: Path) -> pl.DataFrame:
        column_names = [
            "id",
            "gender",
            "Age",
            "BMI",
            "glycaemia",
            "HbA1c",
            "follow.up",
            "T2DM",
        ]
        dtypes = {
            "id": pl.Int64,
            "gender": pl.Int32,
            "Age": pl.Int32,
            "BMI": pl.Float32,
            "glycaemia": pl.Float32,
            "HbA1c": pl.Float32,
            "follow.up": pl.Int32,
            "T2DM": pl.Boolean,
        }

        if metadata_path.suffix.lower() == ".txt":
            return pl.read_csv(
                str(metadata_path),
                separator=" ",
                has_header=False,
                skip_rows=1,
                quote_char='"',
                new_columns=column_names,
                dtypes=dtypes,
                null_values="NA",
            )

        return pl.read_csv(
            str(metadata_path),
            separator=",",
            has_header=True,
            new_columns=column_names,
            dtypes=dtypes,
            null_values="NA",
        )

    def load_metadata(self, metadata_folder: str) -> tuple[pl.DataFrame, str]:
        metadata_path = self._resolve_metadata_path(metadata_folder)
        cgm_data_root_folder = str(metadata_path.parent)

        clinical_df = self._read_metadata_table(metadata_path)
        clinical_df = clinical_df.with_columns(pl.col("HbA1c").cast(pl.Float64))
        clinical_df = clinical_df.with_columns(
            pl.col("T2DM")
            .cast(pl.Utf8)
            .map_elements(lambda value: "T2DM" if value == "True" else "Control")
            .alias("Type of Diabetes")
        )

        data = self.clean_metadata(clinical_df)
        validate_schema(data, self.METADATA_SCHEMA)
        return data, cgm_data_root_folder

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

        filtered_metadata = metadata.filter(~metadata["id"].is_in(self.bad_subjects))
        return filtered_metadata

    def get_subject_data(self, subject_id: str) -> ColasSubjectData:
        # Extract the metadata for the given subject_id
        subject_metadata = self.metadata.filter(self.metadata["id"] == subject_id)
        # Create and return a ChineseSubjectData instance
        return ColasSubjectData(
            subject_metadata, cgm_data_root_folder=self.cgm_data_root_folder
        )

    def construct_cgm_dataframe(self):
        # Construct multi-row DataFrame by concatenating cgm_to_single_row outputs
        cgm_rows = [
            self.get_subject_data(subject_id).cgm_to_single_row()
            for subject_id in self.metadata["id"]
        ]
        return pl.concat(cgm_rows, how="vertical")

    def construct_joined_dataframe(self):
        df_meta_selected = self.metadata.select(
            [
                "id",
                "gender",
                "Age",
                "BMI",
                "glycaemia",
                "HbA1c",
                "follow.up",
                "Type of Diabetes",
            ]
        )

        # Perform the join operation
        df_combined = self.cgm_data.join(
            df_meta_selected, on="id", how="left"  # This is a left join
        )
        # Example usage
        root_datetime = datetime(2012, 1, 1)  # This is the root datetime

        time_series = df_combined['time_series'].to_list()
        for time_list in time_series:
            _, time_deltas = _count_days(time_list)
            assert len(time_list) == len(time_deltas)

            for i in range(len(time_list)):
                # Extract the time component (hours, minutes, seconds) from time_list[i]
                time_component = time_list[i]

                # Determine the date component

                new_date = root_datetime + timedelta(days=time_deltas[i])


                # Combine the date and time components
                time_list[i] = datetime.combine(new_date, time_component)

        df_combined = df_combined.with_columns(pl.Series(name="time_series", values=time_series, dtype=pl.List(pl.Datetime(time_unit='us', time_zone=None))))

        return df_combined

    def concat(self, other):
        pass
