from abc import ABC, abstractmethod
import polars as pl

class CGMDataset(ABC):
    def __init__(self):
        self.METADATA_SCHEMA = None
        self.CGM_SCHEMA = None
        self.bad_subjects = []
        self.metadata = None
        self.cgm_data = None
        self.joined_data = None
        self.ids = None

    @abstractmethod
    def load_metadata(self, source: str) -> pl.DataFrame:
        pass

    @abstractmethod
    def clean_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_subject_data(self, subject_id: str):
        pass

    @abstractmethod
    def construct_cgm_dataframe(self):
        pass

    @abstractmethod
    def construct_joined_dataframe(self):
        pass

    @abstractmethod
    def concat(self, other):
        pass

