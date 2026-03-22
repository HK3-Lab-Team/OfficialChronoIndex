from datetime import datetime
from typing import List, Optional, Dict, Union, Self, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import polars as pl
import csv


class VO2Data(BaseModel):
    vo2Max: str = Field(default_factory=str)


class SPO2Data(BaseModel):
    Avg: float = Field(default=0.0)
    Max: float = Field(default=0.0)
    Min: float = Field(default=0.0)


class ActiveZonesData(BaseModel):
    activeZoneMinutes: int = Field(default=0)
    peakActiveZoneMinutes: int = Field(default=0)
    cardioActiveZoneMinutes: int = Field(default=0)
    fatBurnActiveZoneMinutes: int = Field(default=0)


class ActivityLevel(BaseModel):
    minutes: int = Field(default=0)
    name: str = Field(default_factory=str)


class DetailedActivity(BaseModel):
    activeDuration: int = Field(default=0)
    # Make list fields Optional[List[...]] with default_factory=list
    activityLevel: Optional[List[ActivityLevel]] = Field(default_factory=list)
    activityName: str = Field(default_factory=str)
    activityTypeId: int = Field(default=0)
    averageHeartRate: Optional[int] = Field(default=None)
    calories: int = Field(default=0)
    dateOfActivity: str = Field(default_factory=str)
    distance: float = Field(default=0.0)
    distanceUnit: Optional[str] = Field(default=None)
    duration: int = Field(default=0)
    elevationGain: float = Field(default=0.0) # MODIFIED BY ENRI, WAS int which caused error for a Graz subject
    lastModified: datetime = Field(...)
    logId: int = Field(default=0)
    logType: str = Field(default_factory=str)
    originalDuration: int = Field(default=0)
    originalStartTime: datetime = Field(...)
    pace: float = Field(default=0.0)
    speed: float = Field(default=0.0)
    startTime: datetime = Field(...)
    steps: int = Field(default=0)
    tcxLink: Optional[str] = Field(default=None)


class Distance(BaseModel):
    Activity: str = Field(default_factory=str)
    Distance: float = Field(default=0.0)


class HeartRateZone(BaseModel):
    max: int = Field(default=0)
    min: int = Field(default=0)
    name: str = Field(default_factory=str)
    minutes: int = Field(default=0)
    caloriesOut: float = Field(default=0.0)


class ActivitySummary(BaseModel):
    Steps: int = Field(default=0)
    Floors: Optional[int] = Field(default=None)
    # List-type fields are now Optional[List[...]] with default_factory=list
    Distances: Optional[List[Distance]] = Field(default_factory=list)
    Elevation: Optional[float] = Field(default=None)
    CaloriesBMR: int = Field(default=0)
    CaloriesOut: int = Field(default=0)
    HeartRateZones: Optional[List[HeartRateZone]] = Field(default_factory=list)
    ActivityCalories: int = Field(default=0)
    MarginalCalories: int = Field(default=0)
    RestingHeartRate: Optional[float] = Field(default=None) # MODIFIED BY ENRI, WAS integer
    SedentaryMinutes: int = Field(default=0)
    VeryActiveMinutes: int = Field(default=0)
    FairlyActiveMinutes: int = Field(default=0)
    LightlyActiveMinutes: int = Field(default=0)


class SummaryActivity(BaseModel):
    name: str = Field(default_factory=str)
    logId: int = Field(default=0)
    steps: int = Field(default=0)
    calories: int = Field(default=0)
    duration: int = Field(default=0)
    startDate: str = Field(default_factory=str)
    startTime: str = Field(default_factory=str)
    activityId: int = Field(default=0)
    isFavorite: bool = Field(default=False)
    description: str = Field(default_factory=str)
    hasStartTime: bool = Field(default=True)
    lastModified: datetime = Field(...)
    activityParentId: int = Field(default=0)
    activityParentName: str = Field(default_factory=str)
    hasActiveZoneMinutes: bool = Field(default=True)
    distance: Optional[float] = Field(default=None)


class ActivitySummaryWrapper(BaseModel):
    Summary: Optional[ActivitySummary] = Field(...)
    # Make this a list but allow null input
    Activities: Optional[List[SummaryActivity]] = Field(default_factory=list)


class ActivityData(BaseModel):
    vo2: Optional[VO2Data] = Field(default=None)
    spo2: Optional[SPO2Data] = Field(default=None)
    activeZones: Optional[ActiveZonesData] = Field(default=None)
    # Similarly for the "activities" list
    activities: Optional[List[DetailedActivity]] = Field(default_factory=list)
    # MADE OPTIONAL BY ENRI
    activitySummary: Optional[ActivitySummaryWrapper] = Field(default=None)


class Activity(BaseModel):
    data: ActivityData = Field(...)
    activity_date: datetime = Field(...,alias="date")

class Drug(BaseModel):
    drug: str


class Disease(BaseModel):
    disease: Optional[str]



class DiabetesFamiliarityItem(BaseModel):
    FamiliarityOrder: Optional[str] = None
    FamilyMember1stOrder: Optional[str] = None
    FamilyMember2ndOrder: Optional[str] = None
    DiabetesType: Optional[str] = None


#
# 2. Main content of the "NEW_PARTICIPANT_UNIL" form
#
class NewParticipantFormContent(BaseModel):
    participantId: str
    nickname: Optional[str] = None
    email: Optional[str] = None
    emailVerification: Optional[str] = None
    phone: Optional[str] = None
    phoneVerification: Optional[str] = None
    firstVisitDate: Optional[datetime] = None

    Age: Optional[float] = None
    Sex: Optional[str] = None 
    PregnancyTest: Optional[str] = None
    Weight: Optional[float] = None
    Height: Optional[float] = None

    TakesDrugs: Optional[str] = None
    Drugs: Optional[List[Drug]] = Field(default_factory=list)

    HasDiseases: Optional[str] = None
    Diseases: Optional[List[Disease]] = Field(default_factory=list)

    HasDiabetesFamiliarity: Optional[str] = None
    DiabetesFamiliarity: Optional[List[DiabetesFamiliarityItem]] = Field(default_factory=list)

    BeerPerWeek: Optional[float] = None
    WinePerWeek: Optional[float] = None
    LiquorsPerWeek: Optional[float] = None

    missingData: Optional[bool] = None
    DoctorNotes: Optional[str] = None

    @field_validator("Diseases", mode="before")
    def skip_empty_diseases(cls, value):
        """
        Remove any items in 'Diseases' that are empty dicts or
        do not have a 'disease' key.
        """
        if not isinstance(value, list):
            return value
        filtered = []
        for item in value:
            if isinstance(item, dict) and item.get("disease"):
                # Keep items that have a non-empty 'disease' field
                filtered.append(item)
            # Skip items that are empty or missing "disease"
        return filtered

    


#
# 3. Wrapper for the entire form (includes formId + content)
#
class NewParticipantForm(BaseModel):
    formId: str
    content: NewParticipantFormContent


class CardiovascularDiseaseFamiliarityItem(BaseModel):
    FamiliarityOrder: Optional[str] = None
    FamilyMember1stOrder: Optional[str] = None
    CardiovascularDiseaseType: Optional[str] = None


class HypertensionFamiliarityItem(BaseModel):
    FamiliarityOrder: Optional[str] = None
    FamilyMember1stOrder: Optional[str] = None


class CigaretteData(BaseModel):
    NumberOfCigarettes: Optional[str] = None
    StartPeriod: Optional[str] = None  # Changed to datetime
    NumberOfPuffs: Optional[str] = None
    EndPeriod: Optional[str] = None  # Changed to datetime




# Model for individual CGM readings
class CGMEntry(BaseModel):
    device: str
    serialNumber: str
    deviceTimestamp: datetime
    recordType: int
    historicGlucoseMmolL: float
    
    class Config:
        extra = "ignore"

# Model for the CGM upload in visit forms
class CGMUploadData(BaseModel):
    fileName: Optional[str] = None
    data: Optional[str] = None  # Raw CSV data as string
    cgm_entries: Optional[List[CGMEntry]] = None

    @model_validator(mode="after")
    def validate_and_parse_cgm_entries(self) -> Self:
        if self.data:
            self.cgm_entries = self.parse_cgm_entries()
        return self


    
    def parse_cgm_entries(self) -> List[CGMEntry]:
        """Parse the raw CSV data into a list of CGMEntry objects"""
        if not self.data:
            return []
        
        entries = []
        # Skip the header lines (first 3 lines are metadata)
        lines = self.data.strip().split('\r\n')

        lines = clean_csv_lines(lines)

        
        # Find the line with column headers
        header_index = 0
        for i, line in enumerate(lines):
            if "Device,Serial Number,Device Timestamp,Record Type" in line or "Gerät,Seriennummer,Gerätezeitstempel,Aufzeichnungstyp" in line:
                header_index = i
                break
        
        # Set up a CSV reader with the actual data rows
        reader = csv.reader(lines[header_index+1:])
        for row in reader:
            if len(row) >= 5:  # Ensure we have enough columns
                try:
                    entry = CGMEntry(
                        device=row[0],
                        serialNumber=row[1],
                        deviceTimestamp=datetime.strptime(row[2], '%d-%m-%Y %H:%M'),
                        recordType=int(row[3]),
                        historicGlucoseMmolL=float(row[4].replace(",", ".")) if row[4] else 0.0 # modified by Enri, to convert comma decimals into dot decimals
                    )
                    entries.append(entry)
                except (ValueError, IndexError):
                    # Skip rows with parsing errors
                    continue
                    
        return entries

class BaseFormContent(BaseModel):
    """Base class for shared fields across different visit forms"""
    # Basic measurements
    Fasted: Optional[str] = None
    Weight: Optional[float] = None
    Height: Optional[float] = None
    Waist: Optional[float] = None
    Hip: Optional[float] = None
    UrineTest: Optional[str] = None
    PregnancyTest: Optional[str] = None
    BloodTest: Optional[str] = None
    
    # 24-hour recall
    recall_24h: Optional[str] = Field(default=None, alias="24hRecall")
    
    # Additional metrics
    Distance: Optional[float] = None
    RestingHeartRate: Optional[float] = None # MODIFIED BY ENRI, WAS integer
    Sex: Optional[str] = None
    Age: Optional[float] = None
    
    # CGM related
    CGM: Optional[str] = None

    # Follow-up information
    FecalSample: Optional[str] = None
    
    
    # Notes
    missingData: Optional[bool] = None
    DoctorNotes: Optional[str] = None


class BaseForm(BaseModel):
    """Base class for all visit forms"""
    formId: str
    content: BaseFormContent


class Visit1FormContent(BaseFormContent):

    # Food frequency items
    Grain: Optional[str] = None
    BreadAndBakedGoods: Optional[str] = None
    ConfectioneryProducts: Optional[str] = None
    PotatoSweetPotato: Optional[str] = None
    Legumes: Optional[str] = None
    FastFoodAndSnacks: Optional[str] = None
    SweetsAndSoftDrinks: Optional[str] = None
    MeatAndOffal: Optional[str] = None
    MeatProducts: Optional[str] = None
    FishAndSeafood: Optional[str] = None
    FishProducts: Optional[str] = None
    Eggs: Optional[str] = None
    MilkAndMilkProducts: Optional[str] = None
    OilsFatsButter: Optional[str] = None
    Vegetables: Optional[str] = None
    FruitsBerries: Optional[str] = None
    DriedFruits: Optional[str] = None
    NutsSeeds: Optional[str] = None
    CoffeeTeaHotDrinks: Optional[str] = None
    Alcohol: Optional[str] = None
    VitaminsDietarySupplements: Optional[str] = None
    MeatSubstituteProducts: Optional[str] = None
    Sweeteners: Optional[str] = None
    
    # Family history
    HasCardiovascularDiseaseFamiliarity: Optional[str] = None
    CardiovascularDiseaseFamiliarity: Optional[List[CardiovascularDiseaseFamiliarityItem]] = Field(default_factory=list)
    HasHypertensionFamiliarity: Optional[str] = None
    HypertensionFamiliarity: Optional[List[HypertensionFamiliarityItem]] = Field(default_factory=list)
    
    # Smoking
    Smoker: Optional[str] = None
    NumberOfCigarettesPerDay: Optional[List[CigaretteData]] = Field(default_factory=list)

    # Extra fields to catch the visit 2 and 3 dates in Latvia
    Visit2Date: Optional[datetime] = None
    Visit3Date: Optional[datetime] = None



class Visit1Form(BaseForm):
    formId: str = Field(default="VISIT1_UNIL")
    content: Visit1FormContent = Field(default_factory=Visit1FormContent)

class Visit2FormContent(BaseFormContent):
    # Study dates
    WeekDay1_AV1: Optional[datetime] = None  # Changed to datetime
    WeekDay2_AV1: Optional[datetime] = None  # Changed to datetime
    Weekend1_AV1: Optional[datetime] = None  # Changed to datetime
    Weekend1_BF2: Optional[datetime] = None  # Changed to datetime
    WeekDay1_BF2: Optional[datetime] = None  # Changed to datetime
    WeekDay2_BF2: Optional[datetime] = None  # Changed to datetime
    Visit2Date: Optional[datetime] = Field(default=None, alias="CurrentDate")  # This alias works for GRAZ but not Latvia. The Latvia date is updated in PSubject thanks to the _extras dict in Visit1Form
    # CGM specific data
    CGMData: Optional[str] = None
    CGMUpload: Optional[CGMUploadData] = None


class Visit2Form(BaseForm):
    formId: str = Field(default="VISIT2_UNIL")
    content: Visit2FormContent = Field(default_factory=Visit2FormContent)

class Visit3FormContent(BaseFormContent):
    # Visit3 specific dates
    WeekDay1_BV3: Optional[datetime] = None
    WeekDay2_BV3: Optional[datetime] = None
    Weekend1_BV3: Optional[datetime] = None
    
    # Food frequency items (same as Visit1)
    Grain: Optional[str] = None
    BreadAndBakedGoods: Optional[str] = None
    ConfectioneryProducts: Optional[str] = None
    PotatoSweetPotato: Optional[str] = None
    Legumes: Optional[str] = None
    FastFoodAndSnacks: Optional[str] = None
    SweetsAndSoftDrinks: Optional[str] = None
    MeatAndOffal: Optional[str] = None
    MeatProducts: Optional[str] = None
    FishAndSeafood: Optional[str] = None
    FishProducts: Optional[str] = None
    Eggs: Optional[str] = None
    MilkAndMilkProducts: Optional[str] = None
    OilsFatsButter: Optional[str] = None
    Vegetables: Optional[str] = None
    FruitsBerries: Optional[str] = None
    DriedFruits: Optional[str] = None
    NutsSeeds: Optional[str] = None
    CoffeeTeaHotDrinks: Optional[str] = None
    Alcohol: Optional[str] = None
    VitaminsDietarySupplements: Optional[str] = None
    MeatSubstituteProducts: Optional[str] = None
    Sweeteners: Optional[str] = None
    Visit3Date: Optional[datetime] = Field(default=None, alias="CurrentDate")  # This alias works for GRAZ but not Latvia. The Latvia date is updated in PSubject thanks to the _extras dict in Visit1Form

    
    # CGM specific data (similar to Visit2)
    CGMData: Optional[str] = None
    CGMUpload: Optional[CGMUploadData] = None
    
    # HasAllFoods flag seen in sample data
    HasAllFoods: Optional[bool] = None

    @field_validator('missingData', mode='before')
    @classmethod
    def validate_missing_data(cls, value):
        """Convert string values to appropriate boolean values for missingData field"""
        if isinstance(value, str):
            # Handle specific string cases
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif value == 'missingFoodData':
                # You might want to decide how to handle this specific value
                # For now, treating it as a marker that there is missing data
                return True
            else:
                # For any other string, return None or a default value
                return None
        return value

class Visit3Form(BaseForm):
    formId: str = Field(default="VISIT3_UNIL")
    content: Visit3FormContent = Field(default_factory=Visit3FormContent)

class VisitAnalysisResultContent(BaseModel):
    # Blood glucose and related
    HbA1c: Optional[float] = None
    Fasting_blood_glucose: Optional[float] = None
    C_peptide: Optional[float] = Field(default=None, alias="C-peptide")
    
    # Blood count values
    RBC: Optional[float] = None
    Hemoglobin_Hb: Optional[float] = Field(default=None, alias="Hemoglobin_(Hb)")
    Hematocrit_Hct: Optional[float] = Field(default=None, alias="Hematocrit_(Hct)")
    Platelets: Optional[float] = None
    
    # White blood cells
    Leukocytes: Optional[float] = None
    Neutrophils: Optional[float] = None
    Eosinophils: Optional[float] = None
    Basophils: Optional[float] = None
    Lymphocytes: Optional[float] = None
    Monocytes: Optional[float] = None
    
    # Inflammation and iron markers
    ESR: Optional[float] = None
    Ferritin: Optional[float] = None
    Transferrin: Optional[float] = None
    CRP: Optional[float] = None
    hsCRP: Optional[float] = None
    
    # Lipid panel
    Lipid_panel_HDL: Optional[float] = Field(default=None, alias="Lipid_panel_-_HDL")
    Lipid_panel_LDL: Optional[float] = Field(default=None, alias="Lipid_panel_-_LDL")
    Lipid_panel_Total_cholesterol: Optional[float] = Field(default=None, alias="Lipid_panel_-_Total_cholesterol")
    Lipid_panel_Triglycerides: Optional[float] = Field(default=None, alias="Lipid_panel_-_Triglycerides")
    Lipid_panel_remnant_cholesterol: Optional[float] = Field(default=None, alias="Lipid_panel_-_remnant_cholesterol")
    
    # Liver function tests
    ASAT: Optional[float] = None
    ALAT: Optional[float] = None
    CGT: Optional[float] = None
    
    # Kidney function
    serum_creatinine: Optional[float] = None
    GFR_CKD_EPI: Optional[float] = Field(default=None, alias="GFR_(CKD_EPI)")
    Albumin: Optional[float] = None
    
    # Other markers
    Ceruloplasmin: Optional[float] = None
    Uric_acid: Optional[float] = None
    IL_2: Optional[float] = Field(default=None, alias="IL-2")
    IL_6: Optional[float] = Field(default=None, alias="IL-6")
    IL_10: Optional[float] = Field(default=None, alias="IL-10")
    Leptin: Optional[float] = Field(default=None, alias="leptin")
    
    # Missing data flag - needs special handling
    missingData: Optional[Union[bool, str]] = None
    
    @field_validator('missingData', mode='before')
    @classmethod
    def validate_missing_data(cls, value):
        """Handle various string values for missingData field"""
        if isinstance(value, str):
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            # Handle various string values like 'missingFifData'
            return value
        return value
    
    class Config:
        populate_by_name = True  # Allow populating by alias
        extra = "ignore"  # Ignore extra fields


class VisitAnalysisResultForm(BaseModel):
    formId: str = Field(default="VISIT1_ANALYSISRESULT_UNIL")
    content: VisitAnalysisResultContent
    
    class Config:
        extra = "ignore"

class MealType(str, Enum):
    BREAKFAST = "Breakfast"
    MORNING = "Morning"
    LUNCH = "Lunch"
    AFTERNOON = "Afternoon"
    DINNER = "Dinner"


class Food(BaseModel):
    """Model representing a food entry with timestamp, identifier, meal category, and portion size."""
    date: datetime
    foodId: Optional[str] = None# ADDED BY ENRI food id is optional
    meal: MealType
    quantity: Optional[float]= None # ADDED BY ENRI quantity is optional

    meal_aliases: ClassVar[dict[str, str]] = {
        "morning snack": "Morning",
        "afternoon snack": "Afternoon",
    }

    @field_validator("meal", mode="before")
    @classmethod
    def normalize_meal(cls, value):
        if not isinstance(value, str):
            return value 

        original = value
        value = value.strip()

        capitalized = value.capitalize() # Capitalize first letter (e.g., 'lunch' → 'Lunch')
        if capitalized in MealType._value2member_map_:
            return capitalized

        alias = cls.meal_aliases.get(value.lower()) # Fall back on aliases if the problem was not capitalization
        if alias:
            return alias

        raise ValueError(
            f"Invalid meal value: {original!r}. Expected one of: "
            f"{list(MealType._value2member_map_.keys()) + list(cls.meal_aliases.keys())}"
        )
    
    class Config:
        populate_by_name = True
        extra = "ignore"



class PSubject(BaseModel):
    participantId: str = Field(default_factory=str)
    # Let activities be a list, but accept null by defaulting to []
    activities: Optional[List[Activity]] = Field(default_factory=list)
    newParticipantForm: NewParticipantForm = Field(default_factory=NewParticipantForm)
    visit1Form: Visit1Form = Field(default_factory=Visit1Form)
    visit2Form: Visit2Form = Field(default_factory=Visit2Form) # I had made it Optional in the previous version. Now it is required
    visit3Form: Visit3Form = Field(default_factory=Visit3Form) # I had made it Optional in the previous version. Now it is required
    newParticipantAnalysisForm: Optional[Dict] = None
    visit1AnalysisForm: VisitAnalysisResultForm = Field(default_factory=dict)
    visit2AnalysisForm: VisitAnalysisResultForm = Field(default_factory=dict) # ADDED BY ENRI
    visit3AnalysisForm: VisitAnalysisResultForm = Field(default_factory=dict) # ADDED BY ENRI
    # Let foods/cgm also default to empty lists if not present
    foods: Optional[List[Food]] = Field(default_factory=list)
    cgm: Optional[List[CGMEntry]] = Field(default_factory=list)


    # Allow for SOME food entries in food intake to be missing
    @field_validator("foods", mode="before")
    @classmethod
    def filter_invalid_foods(cls, value):
        return [f for f in value if isinstance(f, dict) and f.get("meal") is not None]

    def get_cgm_frame(self) -> pl.DataFrame:

        if not self.cgm:
            return pl.DataFrame()  # Safely return empty if no CGM data
        
        return pl.DataFrame(self.cgm).sort("deviceTimestamp", descending=False).with_columns(pl.lit(self.participantId).alias("participantId"))
    
    
    def get_analysis_frame(self) -> pl.DataFrame:
        return pl.DataFrame([self.visit1AnalysisForm,self.visit2AnalysisForm,self.visit3AnalysisForm]).unnest("content").with_columns(pl.lit(self.participantId).alias("participantId"))

    def get_visit_frame(self) -> pl.DataFrame:
        v1 = pl.DataFrame([self.visit1Form.content]).with_columns(pl.lit(self.participantId).alias("participantId"),pl.lit("visit1").alias("visitId"))
        v2 = pl.DataFrame([self.visit2Form.content]).select(pl.all().exclude("CGMUpload")).with_columns(pl.lit(self.participantId).alias("participantId"),pl.lit("visit2").alias("visitId"))
        v3 = pl.DataFrame([self.visit3Form.content]).select(pl.all().exclude("CGMUpload")).with_columns(pl.lit(self.participantId).alias("participantId"),pl.lit("visit3").alias("visitId"))
        shared_columns = sorted(set(v1.columns) & set(v2.columns) & set(v3.columns))
        return pl.concat(
            [v1.select(shared_columns), v2.select(shared_columns), v3.select(shared_columns)],
            how="vertical_relaxed",
        )
    

    def get_food_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.foods).with_columns(pl.lit(self.participantId).alias("participantId"))

    # Added to populate the dates in Latvia (visit2 and visit3 Forms content)
    @model_validator(mode="after")
    def resolve_visit_dates(self):
        v1 = self.visit1Form.content
        v2 = self.visit2Form.content
        v3 = self.visit3Form.content

        # fallback for Visit2Date since it is not in the Visit2Form content
        if not v2.Visit2Date and v1.Visit2Date:
            v2.Visit2Date = v1.Visit2Date

        # fallback for Visit3Date 
        if not v3.Visit3Date and v1.Visit3Date:
            v3.Visit3Date = v1.Visit3Date

        return self



class UnitType(str, Enum):
    GRAM = "g"
    MILLILITER = "ml"

class FoodMacronutrients(BaseModel):
    """Model representing nutritional information for foods in the database."""
    foodId: Optional[str] = Field(..., alias="id")
    name: str
    calories: Optional[float] = None
    carbs: Optional[float] = None
    proteins: Optional[float] = None
    fats: Optional[float] = None
    unit: Optional[UnitType] = None


def clean_csv_lines(lines: List[str]) -> List[str]:
    ### Some lines in the  Graz file have extra quotes that break the validation
    cleaned = []
    for line in lines:
        # Remove outer quotes if entire line is quoted
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        # Replace double-double quotes with single
        line = line.replace('""', '"')
        cleaned.append(line)
    return cleaned