## 03.02.2026
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date as date_cls
from typing import List, Tuple, Optional, Literal, Set
import json
import re

import polars as pl

from .praes_data_schemas import PSubject, FoodMacronutrients




@dataclass
class PraesRawDataset:
    """
    Dataset wrapper for PRAES/UNIL/Graz JSON exports.
    """
    subjects: List[PSubject]
    data_path: str
    macros_path: Optional[str] = None

    # -------------------------
    # Constructors / loaders
    # -------------------------
    @classmethod
    def from_json(cls, path: str, macros_path: Optional[str] = None, clean: bool = True) -> PraesRawDataset:
        """
        Clean is set to true s.t. the cgm deduplication function is called when loading the data,
        since the json files contain duplicates across visits.
        """
        subjects = cls._load_subjects(path)
        ds = cls(subjects=subjects, data_path=path, macros_path=macros_path)
        if clean:
            ds = ds.clean()
        return ds

    @staticmethod
    def _load_subjects(path: str) -> List[PSubject]:
        with open(path, "r") as f:
            data = json.load(f)

        outs: List[PSubject] = []
        for idx, subj in enumerate(data):
            try:
                parsed = PSubject.model_validate(subj)
                outs.append(parsed)
            except Exception as e:
                pid = subj.get("participantId", f"row_{idx}")
                print(
                    f"Error validating item with id {pid}: {e} "
                    f"item number {idx} in the raw data, data: {subj}"
                )
        return outs

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _concat_non_empty(frames: List[pl.DataFrame]) -> pl.DataFrame:
        frames = [df for df in frames if df is not None and df.height > 0]
        return pl.concat(frames) if frames else pl.DataFrame()

    def get_subject(self, participant_id: str) -> PSubject:
        try:
            return next(s for s in self.subjects if s.participantId == participant_id)
        except StopIteration:
            raise KeyError(f"Subject '{participant_id}' not found")

    def __getitem__(self, participant_id: str) -> PSubject:
        return self.get_subject(participant_id)

    def clean(self) -> PraesRawDataset:
        """
        Deduplicate CGM entries across visits and rebuild the top-level CGM list.
        Returns a new PraesRawDataset instance with cleaned subjects.
        """
        cleaned_subjects = clean_cgms(self.subjects)
        return PraesRawDataset(
            subjects=cleaned_subjects,
            data_path=self.data_path,
            macros_path=self.macros_path
        )

    # -------------------------
    # Subject-level frame helpers
    # -------------------------
    @staticmethod
    def get_activity_frame(subj: PSubject) -> pl.DataFrame:
        if not getattr(subj, "activities", None) or len(subj.activities) == 0:
            return pl.DataFrame()

        return (
            pl.DataFrame(subj.activities, infer_schema_length=1_000_000)
            .unnest("data")
            .unnest("activitySummary")
            .unnest("Summary")
            .sort("activity_date", descending=False)
            .with_columns(pl.lit(subj.participantId).alias("participantId"))
        )

    # -------------------------
    # Aggregations
    # -------------------------
    def get_aggregated_cgm_frame(self, group_by: bool = False) -> pl.DataFrame:
        cgm_frames = [subj.get_cgm_frame() for subj in self.subjects]
        out = self._concat_non_empty(cgm_frames)
        if out.height == 0:
            return out
        if group_by:
            out = (
                out.sort(["participantId", "deviceTimestamp"], descending=False)
                .group_by("participantId")
                .agg(pl.all().exclude("device", "serialNumber", "recordType"))
            )
        return out

    def get_aggregated_activity_frame(self, group_by: bool = False) -> pl.DataFrame:
        all_activities = [
            subj.activities for subj in self.subjects
            if getattr(subj, "activities", None) and len(subj.activities) > 0
        ]
        if not all_activities:
            return pl.DataFrame()

        subj_ids = [
            [subj.participantId] * len(subj.activities)
            for subj in self.subjects
            if getattr(subj, "activities", None) and len(subj.activities) > 0
        ]

        flattened_activities = [item for sublist in all_activities for item in sublist]
        flattened_subj_ids = [item for sublist in subj_ids for item in sublist]

        out = pl.DataFrame(flattened_activities, infer_schema_length=1_000_000)
        out = pl.concat([pl.DataFrame({"participantId": flattened_subj_ids}), out], how="horizontal")

        out = (
            out.unnest("data")
            .unnest("activitySummary")
            .unnest("Summary")
            .sort("activity_date", descending=False)
        )

        if group_by:
            good_cols = [
                "Steps","CaloriesOut","ActivityCalories","MarginalCalories",
                "RestingHeartRate","SedentaryMinutes","VeryActiveMinutes",
                "FairlyActiveMinutes","LightlyActiveMinutes","activity_date",
            ]
            out = (
                out.sort(["participantId", "activity_date"], descending=False)
                .group_by("participantId")
                .agg(pl.col(good_cols))
            )
        return out

    def get_aggregated_analysis_frame(self, group_by: bool = False) -> pl.DataFrame:
        frames = [
            subj.get_analysis_frame().with_columns(pl.col("missingData").cast(pl.Utf8))
            for subj in self.subjects
        ]
        out = self._concat_non_empty(frames)
        if out.height == 0:
            return out
        if group_by:
            out = (
                out.sort(["participantId", "formId"])
                .group_by("participantId")
                .agg(pl.all())
            )
        return out

    def get_aggregated_visit_frame(self, group_by: bool = False) -> pl.DataFrame:
        frames = [subj.get_visit_frame() for subj in self.subjects]
        out = self._concat_non_empty(frames)
        if out.height == 0:
            return out
        if group_by:
            out = out.sort("visitId").group_by("participantId").agg(pl.all())
        return out

    def get_aggregated_food_frame(self, group_by: bool = False) -> pl.DataFrame:
        frames = [subj.get_food_frame() for subj in self.subjects]
        out = self._concat_non_empty(frames)
        if out.height == 0:
            return out
        if group_by:
            out = (
                out.sort(["participantId", "date"], descending=False)
                .group_by("participantId")
                .agg(pl.all())
            )
        return out

    def get_aggregated_frames(
        self, group_by: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        out_cgm = self.get_aggregated_cgm_frame(group_by)
        out_activity = self.get_aggregated_activity_frame(group_by)
        out_analysis = self.get_aggregated_analysis_frame(group_by)
        out_visit = self.get_aggregated_visit_frame(group_by)
        return out_cgm, out_activity, out_analysis, out_visit

    # -------------------------
    # Visit CGM helpers (Graz-compatible)
    # -------------------------
    @staticmethod
    def _empty_visit_cgm_frame_schema() -> dict[str, pl.DataType]:
        return {
            "deviceTimestamp": pl.Datetime,
            "historicGlucoseMmolL": pl.Float64,
            "participantId": pl.Utf8,
            "visitId": pl.Utf8,
            "visitDate": pl.Date,
            "date_notime": pl.Date,
            "hour": pl.Int8,
        }

    @classmethod
    def get_visit_2_cgm_frame(cls, subject: PSubject) -> pl.DataFrame:
        try:
            entries = subject.visit2Form.content.CGMUpload.cgm_entries
            if not entries:
                return pl.DataFrame(schema=cls._empty_visit_cgm_frame_schema())
            visit2_date = getattr(subject.visit2Form.content, "Visit2Date", None)
            if isinstance(visit2_date, datetime):
                visit2_date = visit2_date.date()
            elif not isinstance(visit2_date, date_cls):
                visit2_date = None
            return (
                pl.DataFrame(entries)
                .select(["deviceTimestamp", "historicGlucoseMmolL"])
                .with_columns(
                    pl.lit(subject.participantId).alias("participantId"),
                    pl.lit("visit2").alias("visitId"),
                    pl.lit(visit2_date).cast(pl.Date).alias("visitDate"),
                    pl.col("deviceTimestamp").dt.date().alias("date_notime"),
                    pl.col("deviceTimestamp").dt.hour().cast(pl.Int8).alias("hour"),
                )
                .sort("deviceTimestamp", descending=False)
            )
        except AttributeError:
            return pl.DataFrame(schema=cls._empty_visit_cgm_frame_schema())

    @classmethod
    def get_visit_3_cgm_frame(cls, subject: PSubject) -> pl.DataFrame:
        try:
            entries = subject.visit3Form.content.CGMUpload.cgm_entries
            if not entries:
                return pl.DataFrame(schema=cls._empty_visit_cgm_frame_schema())
            visit3_date = getattr(subject.visit3Form.content, "Visit3Date", None)
            if isinstance(visit3_date, datetime):
                visit3_date = visit3_date.date()
            elif not isinstance(visit3_date, date_cls):
                visit3_date = None
            return (
                pl.DataFrame(entries)
                .select(["deviceTimestamp", "historicGlucoseMmolL"])
                .with_columns(
                    pl.lit(subject.participantId).alias("participantId"),
                    pl.lit("visit3").alias("visitId"),
                    pl.lit(visit3_date).cast(pl.Date).alias("visitDate"),
                    pl.col("deviceTimestamp").dt.date().alias("date_notime"),
                    pl.col("deviceTimestamp").dt.hour().cast(pl.Int8).alias("hour"),
                )
                .sort("deviceTimestamp", descending=False)
            )
        except AttributeError:
            return pl.DataFrame(schema=cls._empty_visit_cgm_frame_schema())

    def get_all_cgm_visit_frames(self, visit_id: Literal["visit2", "visit3"]) -> pl.DataFrame:
        assert visit_id in ("visit2", "visit3")
        frames = (
            [self.get_visit_2_cgm_frame(s) for s in self.subjects]
            if visit_id == "visit2"
            else [self.get_visit_3_cgm_frame(s) for s in self.subjects]
        )
        return self._concat_non_empty(frames)

    def get_all_cgm_long_frame(
        self,
        visit_ids: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Build a long-format CGM DataFrame (one row per reading) across visits.
        """
        if visit_ids is None:
            visit_ids = ["visit2", "visit3"]

        frames: List[pl.DataFrame] = []
        for vid in visit_ids:
            if vid == "visit2" or vid == "visit3":
                frames.append(self.get_all_cgm_visit_frames(vid))
            else:
                raise ValueError(
                    f"Unsupported visit_id '{vid}'. Supported visits: visit2, visit3."
                )

        return self._concat_non_empty(frames)

    # -------------------------
    # Macronutrients (unified)
    # -------------------------
    def load_macronutrients(self) -> List[FoodMacronutrients]:
        if self.macros_path:
            return self._load_macronutrients_from_file(self.macros_path)
        return self._load_macronutrients_from_subject_json(self.data_path)

    @staticmethod
    def _load_macronutrients_from_file(path: str) -> List[FoodMacronutrients]:
        with open(path, "r") as f:
            data = json.load(f)

        validated: List[FoodMacronutrients] = []
        for idx, item in enumerate(data):
            try:
                validated.append(FoodMacronutrients.model_validate(item))
            except Exception as e:
                item_id = item.get("id", f"row_{idx}")
                print(
                    f"Error validating item with id {item_id}: {e} "
                    f"item number {idx} in the raw data, data: {item}"
                )
        return validated

    def _load_macronutrients_from_subject_json(self, path: str) -> List[FoodMacronutrients]:
        with open(path, "r") as f:
            data = json.load(f)

        valid_ids = {s.participantId for s in self.subjects}
        outs: List[FoodMacronutrients] = []

        for line in data:
            participantId = line.get("participantId")
            if not participantId or participantId not in valid_ids:
                continue

            foodint_visit1 = line.get("visit1Form", {}).get("content", {}).get("FoodIntake", []) or []
            foodint_visit2 = line.get("visit2Form", {}).get("content", {}).get("FoodIntake", []) or []
            foodint_visit3 = line.get("visit3Form", {}).get("content", {}).get("FoodIntake", []) or []

            foods = [d for d in (foodint_visit1 + foodint_visit2 + foodint_visit3) if d]

            for i, mydict in enumerate(foods):
                meal_raw = mydict.get("meal")
                date_raw = mydict.get("date")
                if meal_raw is None or date_raw is None:
                    continue

                current_meal = meal_raw if " " not in meal_raw else re.sub(" ", "_", meal_raw)
                mydict["id"] = f"{participantId}_{date_raw}_{current_meal}_{i}"
                mydict["name"] = str(None)

                try:
                    outs.append(FoodMacronutrients.model_validate(mydict))
                except Exception as e:
                    print(f"Error validating macronutrient item for participant {participantId}: {e} data: {mydict}")

        return outs

    def macronutrients_frame(self) -> pl.DataFrame:
        items = self.load_macronutrients()
        if not items:
            return pl.DataFrame()

        df = pl.DataFrame([x.model_dump(by_alias=True) for x in items])

        if "id" in df.columns:
            df = df.rename({"id":"foodId"})

        # Only add participantId/meal when derived from subject JSON (Graz-mode)
        if not self.macros_path and "foodId" in df.columns:
            df = (
                df.with_columns(
                    participantId=pl.col("foodId").str.split("_").list.get(0),
                    meal=pl.col("foodId").str.split("_").list.get(2),
                )
                .with_columns(pl.col("meal").str.to_titlecase().alias("meal"))
            )
        return df


#### Adapter to base_cgm_dataset ####
from .base_cgm_dataset import CGMDataset
from .base_subject_data import SubjectData


PRAES_CGM_SCHEMA = {
    "participantId": pl.Utf8,
    "time_series": pl.List(inner=pl.Datetime(time_unit="us", time_zone=None)),
    "CGM": pl.List(inner=pl.Float64),
}

PRAES_METADATA_SCHEMA = {
    "participantId": pl.Utf8,
}


class PraesSubjectDataAdapter(SubjectData):
    def __init__(self, subject: PSubject):
        self.subject = subject
        self.CGM_SCHEMA = PRAES_CGM_SCHEMA
        metadata = pl.DataFrame([{"participantId": subject.participantId}])
        cgm_readings = self.load_cgm_from_subject_id(subject.participantId)
        super().__init__(metadata, cgm_readings)

    def fix_cgm_types(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        if "deviceTimestamp" in cgm_readings.columns:
            cgm_readings = cgm_readings.with_columns(
                pl.col("deviceTimestamp").cast(pl.Datetime("us"))
            )
        if "historicGlucoseMmolL" in cgm_readings.columns:
            cgm_readings = cgm_readings.with_columns(
                pl.col("historicGlucoseMmolL").cast(pl.Float64)
            )
        return cgm_readings

    def upsample_cgm(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        # No upsampling by default for PRAES CGM; keep original resolution.
        return cgm_readings

    def remove_none_rows(self, cgm_readings: pl.DataFrame) -> pl.DataFrame:
        if cgm_readings.is_empty():
            return cgm_readings
        combined_mask = pl.col(cgm_readings.columns[0]).is_null()
        for col in cgm_readings.columns:
            combined_mask = combined_mask | cgm_readings[col].is_not_null()
        return cgm_readings.filter(combined_mask)

    def load_cgm_from_subject_id(self, subject_id: str) -> pl.DataFrame:
        if self.subject.participantId != subject_id:
            raise KeyError(f"Subject '{subject_id}' not found in adapter.")
        cgm_readings = self.subject.get_cgm_frame()
        if cgm_readings.is_empty():
            return cgm_readings
        cgm_readings = self.fix_cgm_types(cgm_readings)
        return self.remove_none_rows(cgm_readings)

    def cgm_to_single_row(self) -> pl.DataFrame:
        cgm = self.cgm_readings
        if cgm.is_empty():
            return pl.DataFrame(schema=PRAES_CGM_SCHEMA)
        return pl.DataFrame(
            data={
                "participantId": [self.subject.participantId],
                "time_series": [cgm["deviceTimestamp"].to_list()],
                "CGM": [cgm["historicGlucoseMmolL"].to_list()],
            },
            schema=PRAES_CGM_SCHEMA,
        )

    def single_row_to_cgm(self, single_row: pl.DataFrame) -> pl.DataFrame:
        data_dict = {col: single_row[col][0] for col in single_row.columns}
        data_dict.pop("participantId", None)
        if "time_series" in data_dict:
            data_dict["deviceTimestamp"] = data_dict.pop("time_series")
        if "CGM" in data_dict:
            data_dict["historicGlucoseMmolL"] = data_dict.pop("CGM")
        return pl.DataFrame(data_dict)


class PRAESIIDIUMDataset(CGMDataset):
    def __init__(self, data_path: str, macros_path: Optional[str] = None, clean: bool = True):
        super().__init__()
        self.METADATA_SCHEMA = PRAES_METADATA_SCHEMA
        self.CGM_SCHEMA = PRAES_CGM_SCHEMA
        self.bad_subjects = []
        self.dataset = PraesRawDataset.from_json(
            path=data_path, macros_path=macros_path, clean=clean
        )
        self.metadata = self.load_metadata(data_path)
        self.cgm_data = self.construct_cgm_dataframe()
        self.joined_data = self.construct_joined_dataframe()
        if "participantId" in self.metadata.columns:
            self.ids = self.metadata["participantId"].to_list()
        else:
            self.ids = []

    def load_metadata(self, source: str) -> pl.DataFrame:
        rows = []
        for subj in self.dataset.subjects:
            content = getattr(subj, "newParticipantForm", None)
            if content and getattr(content, "content", None):
                data = content.content.model_dump()
            else:
                data = {}
            # Keep metadata strictly from NewParticipantForm.
            data["participantId"] = subj.participantId
            rows.append(data)
        if not rows:
            return pl.DataFrame(schema=PRAES_METADATA_SCHEMA)
        metadata = pl.DataFrame(rows)
        return self.clean_metadata(metadata)

    def clean_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        if "participantId" in metadata.columns:
            metadata = metadata.with_columns(pl.col("participantId").cast(pl.Utf8))
        for col_name in ("Age", "Height", "Weight", "HbA1c"):
            if col_name in metadata.columns:
                metadata = metadata.with_columns(
                    pl.col(col_name).cast(pl.Float64, strict=False)
                )

        if "Height" in metadata.columns and "Weight" in metadata.columns:
            # BMI is derived from participant-level Height/Weight (kg/m^2).
            # Height is interpreted as centimeters when values are > 10.
            height_m = (
                pl.when(pl.col("Height") > 10)
                .then(pl.col("Height") / 100.0)
                .otherwise(pl.col("Height"))
            )
            metadata = metadata.with_columns(
                pl.when(
                    pl.col("Weight").is_not_null()
                    & pl.col("Height").is_not_null()
                    & (height_m > 0)
                )
                .then(pl.col("Weight") / (height_m**2))
                .otherwise(pl.lit(None))
                .cast(pl.Float64, strict=False)
                .alias("BMI")
            )
        return metadata

    def get_subject_data(self, subject_id: str) -> PraesSubjectDataAdapter:
        subject = self.dataset.get_subject(subject_id)
        return PraesSubjectDataAdapter(subject)

    def construct_cgm_dataframe(self):
        rows = [
            self.get_subject_data(subject_id).cgm_to_single_row()
            for subject_id in self.metadata["participantId"]
        ]
        return pl.concat(rows, how="vertical") if rows else pl.DataFrame()

    def construct_joined_dataframe(self):
        if self.cgm_data.is_empty():
            return self.metadata
        return self.metadata.join(self.cgm_data, on="participantId", how="left")

    def concat(self, other):
        if not isinstance(other, PRAESIIDIUMDataset):
            raise ValueError("Can only concatenate with PRAESIIDIUMDataset.")
        combined_subjects = self.dataset.subjects + other.dataset.subjects
        self.dataset = PraesRawDataset(
            subjects=combined_subjects,
            data_path=self.dataset.data_path,
            macros_path=self.dataset.macros_path,
        )
        self.metadata = pl.concat([self.metadata, other.metadata], how="vertical")
        self.cgm_data = pl.concat([self.cgm_data, other.cgm_data], how="vertical")
        self.joined_data = self.construct_joined_dataframe()
        self.ids = self.metadata["participantId"].to_list()
        return self




def _set_entries(asubject, which: str, entries: List) -> None:
    try:
        if which == "v2":
            asubject.visit2Form.content.CGMUpload.cgm_entries = entries or []
        elif which == "v3":
            asubject.visit3Form.content.CGMUpload.cgm_entries = entries or []
        else:
            raise ValueError("which must be 'v2' or 'v3'")
    except AttributeError:
        pass

def decide_kill_visit(s_copy, cross_dupes: set) -> str:
    """
    Decide which visit (v2 or v3) to purge duplicates or near duplicates from.

    Rules:
    1. If only one visit has CGMData == "Yes", drop from the other visit.
    2. If both visits have CGMData == "Yes", pick the visit whose VisitDate
       is furthest (in days) from the earliest duplicate timestamp.
    3. Default: return 'v3'.
    """
    cgm2_flag = getattr(s_copy.visit2Form.content, "CGMData", None)
    cgm3_flag = getattr(s_copy.visit3Form.content, "CGMData", None)
    d2 = getattr(s_copy.visit2Form.content, "Visit2Date", None)
    d3 = getattr(s_copy.visit3Form.content, "Visit3Date", None)

    v2_entries = getattr(s_copy.visit2Form.content.CGMUpload, "cgm_entries", []) or []
    v3_entries = getattr(s_copy.visit3Form.content.CGMUpload, "cgm_entries", []) or []

    # Rule 1
    if cgm2_flag != "Yes" and cgm3_flag == "Yes":
        return "v2"
    if cgm2_flag == "Yes" and cgm3_flag != "Yes":
        return "v3"

    # Rule 2
    if cgm2_flag == "Yes" and cgm3_flag == "Yes" and (d2 and d3) and cross_dupes:

        # V2 and V3 have identical entries (V2 has dupes from V3 and vice versa)
        if set(map(cgm_key, v2_entries)) == cross_dupes and \
           set(map(cgm_key, v3_entries)) == cross_dupes:
            return "split"

        # V2 and V3 partially overlap (V3 has dupes from V2 plus other CGMS)
        dup_times = [k[0] for k in cross_dupes if k[0] is not None]
        if dup_times:
            earliest_dup = min(dup_times)
            dist2 = (d2.date() - earliest_dup.date()).days
            dist3 = (d3.date() - earliest_dup.date()).days
            return "v2" if dist2 > dist3 else "v3"

    # Rule 3 (fallback)
    return "v3"


###### FUNCTION TO REMOVE EXACT DUPLICATES ACROSS VISITS ######
def remove_exact_dupes(s_copy, v2, v3):
    def dedupe_and_filter(entries, *, exclude_keys=None):
        key_fn = cgm_key
        entries = entries or []
        exclude = set(exclude_keys or ())
        seen, out = set(), []
        removed_within, removed_across = [], []
        for e in entries:
            k = key_fn(e)
            if k in exclude:
                removed_across.append(e)
                continue
            if k in seen:
                removed_within.append(e)
                continue
            seen.add(k)
            out.append(e)
        return out, removed_within, removed_across, seen


    v2_keys = {cgm_key(e) for e in (v2 or [])}
    v3_keys = {cgm_key(e) for e in (v3 or [])}
    cross_dupes = v2_keys & v3_keys

    v2_clean, v3_clean = v2 or [], v3 or []


    if cross_dupes:
        kill_v = decide_kill_visit(s_copy, cross_dupes)

        if kill_v == "split":
            v2_date = getattr(s_copy.visit2Form.content, "Visit2Date", None)
            cutoff = _date_as_is(v2_date) if v2_date else None
            v2_new, v3_new, removed_v2, removed_v3 = [], [], [], []
            for e in v2_clean:  # identical so v2_clean == v3_clean
                ts = getattr(e, "deviceTimestamp", None)
                d = _date_as_is(ts)
                if cutoff and d < cutoff:
                    removed_v3.append(e)
                    v2_new.append(e)
                else:
                    v3_new.append(e)
            for e in v3_clean:
                ts = getattr(e, "deviceTimestamp", None)
                d = _date_as_is(ts)
                if cutoff and d >= cutoff:
                    removed_v2.append(e)

            v2_clean, v3_clean = v2_new, v3_new

        else:
            entries = v2_clean if kill_v == "v2" else v3_clean

            v_clean, _, v_removed_across, _ = dedupe_and_filter(entries, exclude_keys=cross_dupes)

            if kill_v == "v2":
                v2_clean = v_clean
            else:
                v3_clean = v_clean

    # --- Update subject ---
    _set_entries(s_copy, "v2", v2_clean)
    _set_entries(s_copy, "v3", v3_clean)



def clean_cgms(
    allsubjects
):

    cleaned = []

    for subj in allsubjects:
        s_copy = subj.model_copy(deep=True)   # keep original intact
        pid = getattr(s_copy, "participantId", "<unknown>")
        v2 = _safe_entries(s_copy, "v2")
        v3 = _safe_entries(s_copy, "v3")

        # Skip if nothing to process
        if not v2 and not v3:
            cleaned.append(s_copy)
            continue

        remove_exact_dupes(s_copy,v2,v3)
        

        # === Rebuild top-level from CURRENT visits ===
        v2_now = _safe_entries(s_copy, "v2")
        v3_now = _safe_entries(s_copy, "v3")
        s_copy.cgm = v2_now + v3_now


        cleaned.append(s_copy)

    return cleaned


def cgm_key(e) -> Tuple:
    return (getattr(e, "deviceTimestamp", None),
            getattr(e, "historicGlucoseMmolL", None))


def _safe_entries(asubject, which: str) -> List:
    try:
        if which == "v2":
            entries = asubject.visit2Form.content.CGMUpload.cgm_entries
        elif which == "v3":
            entries = asubject.visit3Form.content.CGMUpload.cgm_entries
        else:
            raise ValueError("which must be 'v2' or 'v3'")
    except AttributeError:
        entries = None
    return entries or []

def _visit_dates(entries) -> Set:
    """Set of unique dates (UTC) from a list of CGM entries."""
    dates = set()
    for e in entries or []:
        d = _date_as_is(getattr(e, "deviceTimestamp", None))
        if d is not None:
            dates.add(d)
    return dates

def _date_as_is(ts):
    """Return date without any timezone conversion.
    For naive or tz-aware datetimes, just take .date() as-is."""
    if ts is None:
        return None
    if hasattr(ts, "date"):
        return ts.date()
    return ts