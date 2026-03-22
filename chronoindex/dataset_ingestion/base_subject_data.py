from abc import ABC, abstractmethod
import polars as pl
from typing import Optional

class SubjectData(ABC):
    def __init__(self, metadata: pl.DataFrame, cgm_readings: Optional[pl.DataFrame] = None):
        self.metadata = metadata
        self.cgm_readings = cgm_readings
        self.CGM_SCHEMA = None

    @abstractmethod
    def fix_cgm_types(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def upsample_cgm(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def remove_none_rows(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def load_cgm_from_subject_id(self, subject_id: str) -> pl.DataFrame:
        pass

    @abstractmethod
    def cgm_to_single_row(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def single_row_to_cgm(self, single_row: pl.DataFrame) -> pl.DataFrame:
        pass

    def __repr__(self):
        return f"<SubjectData: {self.metadata.width} metadata columns, {self.cgm_readings.height} cgm readings>"