import polars as pl
from typing import Optional
from pathlib import Path
from .utils.pl_utils import is_float, validate_schema
from .base_subject_data import SubjectData

CGM_SCHEMA = {
    "Date": pl.Datetime(time_unit="us", time_zone=None),
    "CGM": pl.Float64,
    "CBG": pl.Float64,
    "Blood Ketone": pl.Float64,
    "DI (Eng)": pl.Utf8,
    "DI (Ch)": pl.Utf8,
    "Insulin (s.c.)": pl.Utf8,
    "NIHA": pl.Utf8,
    "CSII Bolus": pl.Float64,
    "CSII Basal": pl.Float64,
    "Insulin (i.v.)": pl.Utf8,
}

METADATA_SCHEMA = {
    "Patient Number": pl.Utf8,
    "Gender (Female=1, Male=2)": pl.Int64,
    "Age": pl.Int32,
    "Height (m)": pl.Float64,
    "Weight (kg)": pl.Float64,
    "BMI": pl.Float32,
    "Smoking History (pack year)": pl.Float64,
    "Alcohol Drinking History (drinker/non-drinker)": pl.Utf8,
    "Type of Diabetes": pl.Utf8,
    "Duration of diabetes (years)": pl.Float64,
    "Acute Diabetic Complications": pl.Utf8,
    "Diabetic Macrovascular  Complications": pl.Utf8,
    "Diabetic Microvascular Complications": pl.Utf8,
    "Comorbidities": pl.Utf8,
    "Hypoglycemic Agents": pl.Utf8,
    "Other Agents": pl.Utf8,
    "Fasting Plasma Glucose (mg/dl)": pl.Float64,
    "2-hour Postprandial Plasma Glucose (mg/dl)": pl.Float64,
    "Fasting C-peptide (nmol/L)": pl.Float64,
    "2-hour Postprandial C-peptide (nmol/L)": pl.Float64,
    "Fasting Insulin (pmol/L)": pl.Float64,
    "2-hour Postprandial insulin (pmol/L)": pl.Float64,
    "HbA1c (mmol/mol)": pl.Float64,
    "Glycated Albumin (%)": pl.Float64,
    "Total Cholesterol (mmol/L)": pl.Float64,
    "Triglyceride (mmol/L)": pl.Float64,
    "High-Density Lipoprotein Cholesterol (mmol/L)": pl.Float64,
    "Low-Density Lipoprotein Cholesterol (mmol/L)": pl.Float64,
    "Creatinine (umol/L)": pl.Float64,
    "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ": pl.Float64,
    "Uric Acid (mmol/L)": pl.Float64,
    "Blood Urea Nitrogen (mmol/L)": pl.Float64,
    "Hypoglycemia (yes/no)": pl.Boolean,
}


transformation_rules = {
    "Date": "Date",
    "CGM (mg / dl)": "CGM",
    "CBG (mg / dl)": "CBG",
    "Blood Ketone (mmol / L)": "Blood Ketone",
    "Dietary intake": "DI (Eng)",
    "饮食": "DI (Ch)",
    "Insulin dose - s.c.": "Insulin (s.c.)",
    "Non-insulin hypoglycemic agents": "NIHA",
    "CSII - bolus insulin (Novolin R, IU)": "CSII Bolus",
    "CSII - basal insulin (Novolin R, IU / H)": "CSII Basal",
    "Insulin dose - i.v.": "Insulin (i.v.)",
    "胰岛素泵基础量 (Novolin R, IU / H)": "CSII Basal",
    "进食量": "DI (Ch)",
    "CSII - bolus insulin (Novolin R  IU)": "CSII Bolus",
    "CSII - basal insulin": "CSII Basal",
    "CSII - bolus insulin": "CSII Bolus",
    "CSII - basal insulin (Novolin R  IU / H)": "CSII Basal",
}


class ZhaoSubjectData(SubjectData):
    def __init__(
        self,
        metadata: pl.DataFrame,
        cgm_readings: Optional[pl.DataFrame] = None,
        cgm_folder: Optional[str] = None,
    ):
        self.METADATA_SCHEMA = METADATA_SCHEMA
        self.CGM_SCHEMA = CGM_SCHEMA
        self.cgm_folder = (
            Path(cgm_folder)
            if cgm_folder is not None
            else Path("./data/zhao2023/Shanghai_T2DM")
        )
        if cgm_readings is not None:
            validate_schema(cgm_readings, self.CGM_SCHEMA)
        else:
            subject_id = metadata["Patient Number"][0]
            cgm_readings = self.load_cgm_from_subject_id(subject_id)

        self.metadata = metadata
        self.cgm_readings = cgm_readings
        self.list_schema = {
            col_name: pl.List(inner=dtype)
            for col_name, dtype in self.CGM_SCHEMA.items()
        }
        self.list_schema["Patient Number"] = pl.Utf8
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
                # Normalize datetime column to microseconds or parse if it's a string
                current_dtype = cgm_readings.schema.get(col_name)
                if current_dtype == pl.Utf8:
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
                elif current_dtype == pl.Datetime("ms"):
                    cgm_readings = cgm_readings.with_columns(
                        cgm_readings[col_name].cast(pl.Datetime("us"))
                    )
            elif desired_dtype == pl.Utf8:
                cgm_readings = cgm_readings.with_columns(
                    cgm_readings[col_name].cast(pl.Utf8)
                )
            # For other types, you can add more conditions here if needed

        return cgm_readings

    def upsample_cgm(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        # Ensure the dataframe is sorted by 'date'
        cgm_readings = cgm_readings.sort("Date").set_sorted("Date")

        # Upsample the dataframe every 15 minutes
        df_upsampled = cgm_readings.upsample(
            time_column="Date", every="15m", maintain_order=True
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
        # Load data with polars and normalize column names in a polars-native way.
        # Build schema overrides from known CGM schema + transformation rules, but
        # only for columns that actually exist in the file to avoid KeyError.
        def read_with_overrides(path: str) -> pl.DataFrame:
            header = pl.read_excel(path, read_options={"n_rows": 0})
            # Default every column to Utf8 to avoid dtype inference warnings,
            # then override known columns to their expected CGM dtypes.
            schema_overrides = {col: pl.Utf8 for col in header.columns}
            for raw_name, std_name in transformation_rules.items():
                if std_name in self.CGM_SCHEMA and raw_name in schema_overrides:
                    schema_overrides[raw_name] = self.CGM_SCHEMA[std_name]
            return pl.read_excel(path, schema_overrides=schema_overrides)

        candidates = [
            self.cgm_folder / f"{subject_id}.xlsx",
            self.cgm_folder / f"{subject_id}.xls",
        ]
        selected_path: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                selected_path = candidate
                break
        if selected_path is None:
            looked_up = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"No CGM workbook found for subject '{subject_id}'. Looked in: {looked_up}"
            )

        df = read_with_overrides(str(selected_path))

        # Strip whitespace from column names
        df = df.rename({c: c.strip() for c in df.columns})
        # Apply transformation rules to rename columns (only those that exist)
        existing_rules = {k: v for k, v in transformation_rules.items() if k in df.columns}
        df = df.rename(existing_rules)

        # Keep as polars dataframe
        cgm_readings = df
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

        cgm_readings_filled = self.upsample_cgm(cgm_readings)

        validate_schema(cgm_readings_filled, self.CGM_SCHEMA)
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
        # Create the single row DataFrame using the list_schema
        # {@daniel, @gpt} suboptimal method going through python can be substituted with polars expression
        single_row = {
            col: [self.cgm_readings[col].to_list()] for col in self.cgm_readings.columns
        }
        single_row["Patient Number"] = [self.metadata["Patient Number"][0]]
        # Construct DataFrame with defined schema
        return pl.DataFrame(data=single_row, schema=self.list_schema)

    def single_row_to_cgm(self, single_row: pl.DataFrame) -> pl.DataFrame:
        # validate schema
        validate_schema(single_row, self.list_schema)

        # Convert single row DataFrame to a dictionary
        data_dict = {col: single_row[col][0] for col in single_row.columns}
        # Remove the Patient Number from the dictionary
        data_dict.pop("Patient Number")

        # Construct DataFrame with the data dictionary and the original CGM_SCHEMA
        return pl.DataFrame(data_dict, schema=self.CGM_SCHEMA)

    def __repr__(self):
        return f"<ZhaoSubjectData: {self.metadata.width} metadata columns, {self.cgm_readings.height} cgm readings>"
