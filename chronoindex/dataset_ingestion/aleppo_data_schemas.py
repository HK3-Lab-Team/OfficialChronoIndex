import os
import polars as pl
from .base_subject_data import SubjectData
from typing import Optional

from .utils.pl_utils import is_float, validate_schema

CGM_JOINED_SCHEMA = {
    "PtID": pl.Int32,
    "CGM": pl.List(inner=pl.Float64),
    "time_series": pl.List(inner=pl.Datetime(time_unit="us", time_zone=None)),
    "PtStatus": pl.Utf8,
    "Age": pl.Int32,
    "Type of Diabetes": pl.Utf8,
}

META_DATA_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteOrig": pl.Int32,
    "SiteID": pl.Int32,
    "RandDtDaysAfterEnroll": pl.Int32,
    "PtStatus": pl.Utf8,
    "TrtGroup": pl.Utf8,
    "AgeAsOfEnrollDt": pl.Int32,
    "Type of Diabetes": pl.Utf8,
    "BMI": pl.Float32,
    "Height": pl.Float32,
    "Weight": pl.Float32,
}

ADVERSE_EVENT_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "AENotifiedDaysFromEnroll": pl.Int32,
    "MedCond": pl.Utf8,
    "AdverseEventType": pl.Utf8,
    "AEOnsetDaysFromEnroll": pl.Int32,
    "AEPrEnroll": pl.Utf8,
    "AENotedStdyVisExam": pl.Utf8,
    "AEIntensity": pl.Utf8,
    "AERelStdyTrtUncertain": pl.Utf8,
    "AERelStdyProc": pl.Utf8,
    "AEEffectTrt": pl.Utf8,
    "AESerious": pl.Utf8,
    "AETrt": pl.Utf8,
    "AESurg": pl.Utf8,
    "AESurgDaysFromEnroll": pl.Int32,
    "AEMeds": pl.Utf8,
    "AEOthTrt": pl.Utf8,
    "AEOutcome": pl.Utf8,
    "AEResDaysFromEnroll": pl.Int32,
    "AEDeathDaysFromEnroll": pl.Int32,
    "AEDeath": pl.Utf8,
    "AEConAnomaly": pl.Utf8,
    "AELifeThreat": pl.Utf8,
    "AEHosp": pl.Utf8,
    "AEDisability": pl.Utf8,
    "AEOther": pl.Utf8,
    "Weight": pl.Float64,
    "WeightMeas": pl.Utf8,
    "WeightNotAvail": pl.Utf8,
    "AERelLabData": pl.Utf8,
    "AEOthRelHx": pl.Utf8,
    "AEMedProd": pl.Utf8,
    "AERelStdyDrugDevice": pl.Utf8,
    "AERelStdyDrugDeviceUncertain": pl.Utf8,
}

DEVICE_PROBLEMS_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "SHSinceLastVis": pl.Utf8,
    "DKASinceLastVis": pl.Utf8,
    "OthAESinceLastVis": pl.Utf8,
    "DevProbSinceLastVis": pl.Utf8,
}

COMPL_UNBLIND_CGM_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "CGMMinUse": pl.Utf8,
    "BGMMinMeas": pl.Utf8,
    "CGMGlucUnder60": pl.Utf8,
    "UnblindCGMSensor": pl.Utf8,
    "DtTm1MinCell": pl.Utf8,
    "PtRunInStatus": pl.Utf8,
}

DEVICE_BGM_SCHEMA = {
    "RecID": pl.Int64,
    "ParentHDeviceUploadsID": pl.Int32,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "DeviceDtTmDaysFromEnroll": pl.Int32,
    "DeviceTm": pl.Utf8,
    "RecordType": pl.Utf8,
    "RecordSubType": pl.Utf8,
    "GlucoseValue": pl.Float64,
}

CGM_DATA_SCHEMA = {"id": pl.Int32, "time": pl.Utf8, "gl": pl.Float64}

CGM_UPDATED_SCHEMA = {
    "PtID": pl.Int32,
    "glucemia_series": pl.List(inner=pl.Float64),
    "time_series": pl.List(inner=pl.Time),
    "time_series_internal": pl.List(inner=pl.Time),
    "days_from_enroll": pl.List(inner=pl.Int64),
}

DEVICEDTTMVER_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "DtTm1MinCell": pl.Utf8,
    "CellDtDaysFromEnroll": pl.Int32,
    "CellHr": pl.Int32,
    "CellMin": pl.Int32,
    "CellAmPm": pl.Utf8,
    "CGMDtDaysFromEnroll": pl.Int32,
    "CGMHr": pl.Int32,
    "CGMMin": pl.Int32,
    "CGMAmPm": pl.Utf8,
    "StandBGMDtDaysFromEnroll": pl.Int32,
    "StandBGMHr": pl.Int32,
    "StandBGMMin": pl.Int32,
    "StandBGMAmPm": pl.Utf8,
    "BlindBGMDtDaysFromEnroll": pl.Int32,
    "BlindBGMHr": pl.Int32,
    "BlindBGMMin": pl.Int32,
    "BlindBGMAmPm": pl.Utf8,
    "PumpDtDaysFromEnroll": pl.Int32,
    "PumpHr": pl.Int32,
    "PumpMin": pl.Int32,
    "PumpAmPm": pl.Utf8,
    "DtTm1MinCellUnk": pl.Utf8,
    "CGMNotAtVis": pl.Utf8,
    "StandBGMNotAtVis": pl.Utf8,
    "BlindBGMNotAtVis": pl.Utf8,
    "PumpNotAtVis": pl.Utf8,
    "KetoneMetDtDaysFromEnroll": pl.Int32,
    "KetoneMetHr": pl.Int32,
    "KetoneMetMin": pl.Int32,
    "KetoneMetAMPM": pl.Utf8,
    "KetoneMetNotAtVis": pl.Utf8,
}

HDEVICE_EVENTS_SCHEMA = {
    "RecID": pl.Int64,
    "ParentHDeviceUploadsID": pl.Int64,
    "PtId": pl.Int32,
    "SiteID": pl.Int32,
    "DeviceDtTmDaysFromEnrollDt": pl.Int32,
    "DeviceTm": pl.Utf8,
    "RecordType": pl.Utf8,
    "TmChFromDaysFromEnrollDt": pl.Int32,
    "TmChFrom": pl.Utf8,
    "TmChToDaysFromEnrollDt": pl.Int32,
    "TmChTo": pl.Utf8,
    "TmChAgent": pl.Utf8,
}

DEVICE_ISSUES_SCHEMA = {
    "RecID": pl.Int64,
    "PtId": pl.Int32,
    "SiteID": pl.Int32,
    "InvDevice": pl.Utf8,
    "DevIssueType": pl.Utf8,
    "DevIssueOnsetDtDaysFromEnroll": pl.Int32,
    "DevIssueLocation": pl.Utf8,
    "DevIssueFrequency": pl.Utf8,
    "DevIssueEffect": pl.Utf8,
    "DevIssueReplaceDtDaysFromEnroll": pl.Int32,
    "DevIssueAE": pl.Utf8,
    "DevIssueAELikely": pl.Utf8,
}

DEVICE_UPLOADS_SCHEMA = {
    "RecID": pl.Int64,
    "PtId": pl.Int32,
    "SiteID": pl.Int32,
    "DeviceManufact": pl.Utf8,
    "DeviceModel": pl.Utf8,
    "DeviceType": pl.Utf8,
    "Visit": pl.Utf8,
    "UploadDtTmDaysFromEnroll": pl.Int32,
    "UploadTm": pl.Utf8,
    "DataSource": pl.Utf8,
    "InventoryItemDs": pl.Utf8,
}

DEVICE_WIZARD_SCHEMA = {
    "RecID": pl.Int64,
    "ParentHDeviceUploadsID": pl.Int64,
    "PtId": pl.Int32,
    "SiteID": pl.Int32,
    "DeviceDtTmDaysFromEnroll": pl.Int32,
    "DeviceTm": pl.Utf8,
    "RecommendedCarb": pl.Float64,
    "RecommendedCorrection": pl.Float64,
    "RecommendedNet": pl.Float64,
    "BgInput": pl.Float64,
    "CarbInput": pl.Float64,
    "InsulinOnBoard": pl.Float64,
    "InsulinCarbRatio": pl.Float64,
    "InsulinSensitivity": pl.Float64,
    "BgTargetLow": pl.Float64,
    "BgTargetHigh": pl.Float64,
    "ParentHDeviceBolusID": pl.Int64,
    "BgTargetTarget": pl.Float64,
    "BgTargetRange": pl.Float64,
}

HFOLLOW_UP_SCHEMA = {
    "RecID": pl.Int64,
    "PtId": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "PumpUse": pl.Utf8,
    "PumpStopDtDaysFromEnroll": pl.Int32,
    "CGMUse6Days": pl.Utf8,
    "CGMUseSkin": pl.Utf8,
    "CGMUseAlarms": pl.Utf8,
    "CGMUseAccuracy": pl.Utf8,
    "CGMUseDifficult": pl.Utf8,
    "CGMUseTooBusy": pl.Utf8,
    "CGMUseForget": pl.Utf8,
    "CGMUseNoHelp": pl.Utf8,
    "CGMUseOth": pl.Utf8,
    "CGMDiabMgmtDec": pl.Utf8,
    "BGMProtCompl": pl.Utf8,
    "HbA1cCollected": pl.Utf8,
    "StandBGMProtCompl": pl.Utf8,
    "BlindBGMProtCompl": pl.Utf8,
}

HHYPO_EVENT_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "HypoOccurDtDaysFromEnroll": pl.Int32,
    "HypoOccurDtApprox": pl.Utf8,
    "HypoApproxTime": pl.Utf8,
    "GlucMeterCk": pl.Utf8,
    "GlucMeterRes": pl.Float64,
    "GlucMeterResUnk": pl.Utf8,
    "SensorWear": pl.Utf8,
    "SensorGluc": pl.Float64,
    "SensorGlucUnk": pl.Utf8,
    "LastInsDoseDtDaysFromEnroll": pl.Int32,
    "LastInsDoseApprox": pl.Utf8,
    "LastInsDoseUnk": pl.Utf8,
    "LastInsApproxTime": pl.Utf8,
    "LastInsPriorToHypo": pl.Utf8,
    "HypoSeizure": pl.Utf8,
    "HypoLossCons": pl.Utf8,
    "HypoReqAssist": pl.Utf8,
    "HypoAmbulance": pl.Utf8,
    "HypoEMT": pl.Utf8,
    "HypoHthCareProv": pl.Utf8,
    "GlucGiven": pl.Utf8,
    "HospOrER": pl.Utf8,
    "HospERTrtLoc": pl.Utf8,
    "HospNumDays": pl.Int32,
    "HospNumDaysUnk": pl.Utf8,
    "HypoOutcome": pl.Utf8,
    "EventCauseStdyDev": pl.Utf8,
    "EventCauseBGM": pl.Utf8,
    "EventCauseCGM": pl.Utf8,
    "EventCauseExpl": pl.Utf8,
    "EventCauseNonStdy": pl.Utf8,
}

HINITIAL_STUDY_CGM_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "CGMTypeProvided": pl.Utf8,
    "DtTmSyncConfirm": pl.Int32,
}

HINSULIN_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "InsName": pl.Utf8,
    "InsRoute": pl.Utf8,
    "InsInjectionFreq": pl.Utf8,
    "InsTypeStart": pl.Utf8,
    "InsTypeStartDtDaysFromEnroll": pl.Int32,
    "InsTypeStartUnknown": pl.Utf8,
    "InsTypeStopDtDaysFromEnroll": pl.Int32,
    "InsTypeStopUnknown": pl.Utf8,
    "InsTypeStartEstimate": pl.Utf8,
    "InsTypeStopEstimate": pl.Utf8,
}

HLOCAL_HBA1C_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "HbA1cTestDtDaysAfterEnroll": pl.Int32,
    "HbA1cTestMethod": pl.Utf8,
    "HbA1cTestRes": pl.Float64,
    "HbA1cNotDone": pl.Utf8,
}

HMEDICAL_CONDITION_SCHEMA = {
    "RecID": pl.Int64,
    "PtId": pl.Int32,
    "SiteId": pl.Int32,
    "MedCond": pl.Utf8,
    "MedCondPreStart": pl.Utf8,
    "MedCondPreStartCat": pl.Utf8,
    "MedCondPreStartTreat": pl.Utf8,
    "MedCondDiagDtDaysFromEnroll": pl.Int32,
    "MedCondDiagDtApprox": pl.Int32,
    "MedCondDiagDtUnk": pl.Int32,
    "MedCondTrt": pl.Utf8,
    "MedCondResDtDaysFromEnroll": pl.Int32,
    "MedCondResDtApprox": pl.Int32,
    "MedCondStatus": pl.Utf8,
    "MedCondDiagMonthsAfterEnroll": pl.Int32,
    "MedCondDiagYearsAfterEnroll": pl.Int32,
    "MedCondCurrTreatMed": pl.Utf8,
}

HMEDICATION_DATA_SCHEMA = {
    "RecID": pl.Int32,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "DrugName": pl.Utf8,
    "MedDose": pl.Utf8,
    "MedUnit": pl.Utf8,
    "MedDoseUnk": pl.Int32,
    "MedRoute": pl.Utf8,
    "MedLocSide": pl.Utf8,
    "MedFreqType": pl.Utf8,
    "MedFreqNum": pl.Int32,
    "MedFreqPer": pl.Utf8,
    "MedFreqUnk": pl.Int32,
    "MedInd": pl.Utf8,
    "MedicalCondition1": pl.Utf8,
    "MedicalCondition2": pl.Utf8,
    "AdverseEvent1": pl.Utf8,
    "AdverseEvent2": pl.Utf8,
    "PreExistingCondition1": pl.Utf8,
    "PreExistingCondition2": pl.Utf8,
    "MedStartTrtCat": pl.Utf8,
    "MedStartDtDaysFromEnroll": pl.Int32,
    "MedStartDtApprox": pl.Int32,
    "MedStopDtDaysFromEnroll": pl.Int32,
    "MedStopDtApprox": pl.Int32,
    "MedOngoing": pl.Int32,
    "MedStartPreEnrRange": pl.Utf8,
    "MedStartMonthsAfterEnroll": pl.Int32,
    "MedStartYearsAfterEnroll": pl.Int32,
    "MedStopMonthsAfterEnroll": pl.Int32,
    "MedStopYearsAfterEnroll": pl.Int32,
    "MedStopDtUnk": pl.Int32,
    "MedStartDtUnk": pl.Int32,
    "MedCondNotReqd": pl.Int32,
    "PreExistCondNotReqd": pl.Int32,
    "AdvEventNotReqd": pl.Int32,
}

SCREENING_DATA_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int64,
    "EligCritMet": pl.Utf8,
    "ExclCritAbsent": pl.Utf8,
    "Gender": pl.Utf8,
    "Ethnicity": pl.Utf8,
    "Race": pl.Utf8,
    "DiagAge": pl.Int64,
    "DiagAgeApprox": pl.Utf8,
    "SHMostRec": pl.Utf8,
    "SHNumLast12Mon": pl.Int64,
    "DKAMostRec": pl.Utf8,
    "DKANumLast12Mon": pl.Int64,
    "OthGlucLowerMed": pl.Utf8,
    "Fingerstick7DayAve": pl.Float64,
    "EduLevel": pl.Utf8,
    "EduLevelUnk": pl.Utf8,
    "EduLevelNoAns": pl.Utf8,
    "AnnualInc": pl.Utf8,
    "AnnualIncUnk": pl.Utf8,
    "AnnualIncNoAns": pl.Utf8,
    "InsPrivate": pl.Utf8,
    "InsMedicare": pl.Utf8,
    "InsMediGap": pl.Utf8,
    "InsMedicaid": pl.Utf8,
    "InsSCHIP": pl.Utf8,
    "InsMilitary": pl.Utf8,
    "InsIndian": pl.Utf8,
    "InsState": pl.Utf8,
    "InsOtherGov": pl.Utf8,
    "InsSingleService": pl.Utf8,
    "InsNoCoverage": pl.Utf8,
    "InsUnknown": pl.Utf8,
    "InsNoAns": pl.Utf8,
    "Weight": pl.Float32,
    "Height": pl.Float32,
    "PEAbnormal": pl.Utf8,
    "CGMUseStatus": pl.Utf8,
    "CGMUseDuration": pl.Utf8,
    "CGMUseDevice": pl.Utf8,
    "CGMDLoadMinDays": pl.Utf8,
    "CGMGlucUnder60": pl.Utf8,
}

HPOST_RAND_PT_FINAL_STATUS_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "HPostRandFinalStatReas": pl.Utf8,
    "PtWithdrawReas": pl.Utf8,
    "DeathDtDaysAfterEnroll": pl.Int32,
}

ROSTER_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteOrig": pl.Int32,
    "SiteID": pl.Int32,
    "RandDtDaysAfterEnroll": pl.Int32,
    "PtStatus": pl.Utf8,
    "TrtGroup": pl.Utf8,
    "AgeAsOfEnrollDt": pl.Int32,
    "Type of Diabetes": pl.Utf8,
}

HQUEST_DIAB_TECH_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "WorryHighBGNow": pl.Int32,
    "WorryHighBGChg": pl.Int32,
    "LowBGEffortNow": pl.Int32,
    "LowBGEffortChg": pl.Int32,
    "LowBGSleepNow": pl.Int32,
    "LowBGSleepChg": pl.Int32,
    "FeelDiffNow": pl.Int32,
    "FeelDiffChg": pl.Int32,
    "DiabTimeNow": pl.Int32,
    "DiabTimeChg": pl.Int32,
    "EatBGNow": pl.Int32,
    "EatBGChg": pl.Int32,
    "DiabEffortNow": pl.Int32,
    "DiabEffortChg": pl.Int32,
    "LTHealthNow": pl.Int32,
    "LTHealthChg": pl.Int32,
    "DayLowBGNow": pl.Int32,
    "DayLowBGChg": pl.Int32,
    "HighBGNow": pl.Int32,
    "HighBGChg": pl.Int32,
    "FingStkPainNow": pl.Int32,
    "FindStkPainChg": pl.Int32,
    "PumpInjPainNow": pl.Int32,
    "PumpInjPainChg": pl.Int32,
    "FamWorryNow": pl.Int32,
    "FamWorryChg": pl.Int32,
    "TroubleSleepNow": pl.Int32,
    "TroubleSleepChg": pl.Int32,
    "StrictMealNow": pl.Int32,
    "StrictMealChg": pl.Int32,
    "WorkSchNow": pl.Int32,
    "WorkSchChg": pl.Int32,
    "SportExerNow": pl.Int32,
    "SportExerChg": pl.Int32,
    "InsAmtNow": pl.Int32,
    "InsAmtChg": pl.Int32,
    "KeepUpPeersNow": pl.Int32,
    "KeepUpPeersChg": pl.Int32,
    "BGReactNow": pl.Int32,
    "BGReactChg": pl.Int32,
    "DiabQuestNow": pl.Int32,
    "DiabQuestChg": pl.Int32,
    "DiabRespNow": pl.Int32,
    "DiabRespChg": pl.Int32,
    "PreMealInsNow": pl.Int32,
    "PreMealInsChg": pl.Int32,
    "SkipMealInsNow": pl.Int32,
    "SkipMealInsChg": pl.Int32,
    "AlarmReactNow": pl.Int32,
    "AlarmReactChg": pl.Int32,
    "InsSickDayNow": pl.Int32,
    "InsSickDayChg": pl.Int32,
    "DeviceRunLifeNow": pl.Int32,
    "DeviceRunLifeChg": pl.Int32,
    "ExerInsAmtNow": pl.Int32,
    "ExerInsAmtChg": pl.Int32,
    "SeveralDeviceNow": pl.Int32,
    "SeveralDeviceChg": pl.Int32,
    "LookDiffNow": pl.Int32,
    "LookDiffChg": pl.Int32,
    "QuestNotDone": pl.Utf8,
}

HQUEST_HYPO_FEAR_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "LgSnackBed": pl.Int32,
    "AvoidAloneLowBG": pl.Int32,
    "TestBGRunHigh": pl.Int32,
    "HighBGAlone": pl.Int32,
    "EatFirstSignLowBG": pl.Int32,
    "RedInsThinkLowBG": pl.Int32,
    "KeepHighBGMtg": pl.Int32,
    "CarryFastActSug": pl.Int32,
    "AvoidExThinkLowBG": pl.Int32,
    "CkSugOftMtg": pl.Int32,
    "WorryNotRecLowBG": pl.Int32,
    "WorryNoFood": pl.Int32,
    "WorryPassOut": pl.Int32,
    "WorryEmbarSocial": pl.Int32,
    "WorryReacAlone": pl.Int32,
    "WorryAppStupDrunk": pl.Int32,
    "WorryLoseCntrl": pl.Int32,
    "WorryNoHelp": pl.Int32,
    "WorryReactDrive": pl.Int32,
    "WorryMistAcc": pl.Int32,
    "WorryBadEvalCrit": pl.Int32,
    "WorryRespForOthers": pl.Int32,
    "WorryDizzy": pl.Int32,
    "QuestNotDone": pl.Utf8,
}

HQUEST_HYPO_UNAWARE_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "LowBGSympCat": pl.Utf8,
    "LowBGLostSymp": pl.Utf8,
    "ModHypoEpPast6Mon": pl.Utf8,
    "SevHypoEpPastYear": pl.Utf8,
    "Bel70PastMonWSymp": pl.Utf8,
    "Bel70PastMonNoSymp": pl.Utf8,
    "FeelSympLowBG": pl.Utf8,
    "ExtentSympLowBG": pl.Utf8,
}

HRANDOMIZATION_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "NoExclEventOccur": pl.Int32,
    "EligCritMet": pl.Int32,
    "MedicalCondsEntered": pl.Utf8,
    "MedicationEntered": pl.Utf8,
    "HbA1cCollected": pl.Utf8,
    "BioRepCollected": pl.Utf8,
}

HRUN_IN_VISIT_STATUS_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "RunInStatus": pl.Utf8,
}

HSCREENING_SCHEMA = {
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "EligCritMet": pl.Int32,
    "ExclCritAbsent": pl.Int32,
    "Gender": pl.Utf8,
    "Ethnicity": pl.Utf8,
    "Race": pl.Utf8,
    "DiagAge": pl.Int32,
    "DiagAgeApprox": pl.Utf8,
    "SHMostRec": pl.Utf8,
    "SHNumLast12Mon": pl.Int32,
    "DKAMostRec": pl.Utf8,
    "DKANumLast12Mon": pl.Int32,
    "OthGlucLowerMed": pl.Utf8,
    "Fingerstick7DayAve": pl.Float64,
    "EduLevel": pl.Utf8,
    "EduLevelUnk": pl.Utf8,
    "EduLevelNoAns": pl.Utf8,
    "AnnualInc": pl.Utf8,
    "AnnualIncUnk": pl.Utf8,
    "AnnualIncNoAns": pl.Utf8,
    "InsPrivate": pl.Int32,
    "InsMedicare": pl.Int32,
    "InsMediGap": pl.Int32,
    "InsMedicaid": pl.Int32,
    "InsSCHIP": pl.Int32,
    "InsMilitary": pl.Int32,
    "InsIndian": pl.Int32,
    "InsState": pl.Int32,
    "InsOtherGov": pl.Int32,
    "InsSingleService": pl.Int32,
    "InsNoCoverage": pl.Int32,
    "InsUnknown": pl.Int32,
    "InsNoAns": pl.Int32,
    "Weight": pl.Float64,
    "Height": pl.Float64,
    "PEAbnormal": pl.Utf8,
    "CGMUseStatus": pl.Utf8,
    "CGMUseDuration": pl.Utf8,
    "CGMUseDevice": pl.Utf8,
    "CGMDLoadMinDays": pl.Utf8,
    "CGMGlucUnder60": pl.Utf8,
}

HUNSCHEDULED_VISIT_SCHEMA = {
    "ParentLoginVisitID": pl.Int64,
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "VisitReasCGMTrain": pl.Int32,
    "VisitReasDiabMgmt": pl.Int32,
    "VisitReasPotentialAE": pl.Int32,
    "VisitReasOth": pl.Int32,
}

HVISIT_INFO_SCHEMA = {
    "ParentLoginVisitID": pl.Int64,
    "RecID": pl.Int64,
    "PtID": pl.Int32,
    "SiteID": pl.Int32,
    "Visit": pl.Utf8,
    "VisitDtDaysFromEnroll": pl.Int32,
    "OutOfWin": pl.Utf8,
    "OutOfWinReason": pl.Utf8,
    "VisitMiss": pl.Utf8,
    "VisitMissReason": pl.Utf8,
}


class AleppoSubjectData(SubjectData):
    def __init__(
        self,
        metadata: pl.DataFrame,
        cgm_readings: Optional[pl.DataFrame] = None,
        cgm_data_root_folder: Optional[str] = None,
    ):
        self.METADATA_SCHEMA = ROSTER_SCHEMA
        self.CGM_SCHEMA = CGM_DATA_SCHEMA
        self.subject_id = metadata["PtID"][0]
        if cgm_data_root_folder is None:
            current_path = os.getcwd()
            self.cgm_data_root_folder = os.path.join(
                current_path, "data", "Replace-BG", "repli_aleppo", "Data Tables"
            )
        else:
            self.cgm_data_root_folder = cgm_data_root_folder
        if cgm_readings is not None:
            validate_schema(cgm_readings, self.CGM_SCHEMA)
        else:
            cgm_readings = self.load_cgm_from_subject_id(self.subject_id)
        
        self.list_schema = {
            "PtID": pl.Int32,
            "glucemia_series": pl.List(inner=pl.Float64),
            "time_series": pl.List(inner=pl.Time),
            "time_series_internal": pl.List(inner=pl.Time),
            "days_from_enroll": pl.List(inner=pl.Int64),
        }
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
        data_path = os.path.join(self.cgm_data_root_folder, "HDeviceCGM.txt")
        cgm_df = pl.read_csv(
            data_path,
            separator="|",
            null_values="NA",
            columns=[
                "PtID",
                "DeviceDtTmDaysFromEnroll",
                "DeviceTm",
                "DexInternalDtTmDaysFromEnroll",
                "DexInternalTm",
                "GlucoseValue",
            ],
        )

        cgm_readings = (
            cgm_df.filter(pl.col("PtID") == int(subject_id))
            .with_columns(
                [
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

        original_height = cgm_readings.height

        # Create a boolean mask indicating non-null values
        cgm_readings = self.remove_none_rows(cgm_readings)

        # Print if rows have been deleted
        deleted_rows = original_height - cgm_readings.height
        if deleted_rows > 0:
            print(
                f"For subject {subject_id}, {deleted_rows} rows with all null values have been deleted."
            )

        cgm_readings_filled = cgm_readings  # self.upsample_cgm(cgm_readings)

        # validate_schema(cgm_readings_filled, self.CGM_SCHEMA)
        """
        if cgm_readings_filled.height - cgm_readings.height < 0:
            print("Upsampling failed, for {}, with a difference of {} rows".format(subject_id, cgm_readings_filled.height - cgm_readings.height))
            print(cgm_readings["Date"].to_list())
            print(cgm_readings_filled["Date"].to_list())
        """
        cgm_readings_filled = cgm_readings_filled.sort(
            ["days_from_enroll", "time_series"]
        )

        return cgm_readings_filled

    def cgm_to_single_row(self) -> pl.DataFrame:
        # Convert CGM readings to a single row DataFrame
        glucose_list = self.cgm_readings["glucemia"].to_list()
        cgm_series = pl.DataFrame(
            data={
                "PtID": [self.subject_id],
                "glucemia_series": [glucose_list],
                "time_series": [self.cgm_readings["time_series"].to_list()],
                "time_series_internal": [
                    self.cgm_readings["time_series_internal"].to_list()
                ],
                "days_from_enroll": [
                    self.cgm_readings["days_from_enroll"].to_list()
                ],
            },
            schema=CGM_UPDATED_SCHEMA,
        )
        return cgm_series

    def single_row_to_cgm(self, single_row: pl.DataFrame) -> pl.DataFrame:
        # Convert single row DataFrame back to CGM readings
        validate_schema(single_row, self.list_schema)
        data_dict = {col: single_row[col][0] for col in single_row.columns}
        data_dict.pop("PtID")
        return pl.DataFrame(data_dict, schema=self.CGM_SCHEMA)

    def __repr__(self):
        return f"<AleppoSubjectData: {self.metadata.width} metadata columns, {self.cgm_readings.height} cgm readings>"
