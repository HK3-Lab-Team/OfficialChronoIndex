import polars as pl




def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def validate_schema(df: pl.DataFrame, schema: dict):
    """
    Validate the dataframe schema against the provided schema.

    Parameters:
    - df (pl.DataFrame): The dataframe to validate.
    - schema (dict): The schema to validate against. It should map column names to polars data types.

    Raises:
    - ValueError: If a column is missing from the dataframe.
    - TypeError: If a column has an incorrect data type.
    """
    for col_name, dtype in schema.items():
        if col_name not in df.columns:
            raise ValueError(
                f"Column {col_name} missing from dataframe. Found columns: {df.columns}"
            )

        if str(df[col_name].dtype) != str(dtype):
            raise TypeError(
                f"Expected {dtype} for column {col_name}, but got {df[col_name].dtype}."
            )
