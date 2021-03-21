from typing import Any

from pydantic import BaseModel, validator

# Death Event: if the patient deceased during the follow-up period (boolean)
TARGET_NAME = "DEATH_EVENT"

BOOL_FEATS_LIST = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
NUM_FEATS_LIST = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]


class PatientData(BaseModel):
    """
    Patient Data features:
        age: age of the patient (years)
        anaemia: decrease of red blood cells or hemoglobin (boolean)
        high blood pressure: if the patient has hypertension (boolean)
        creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
        diabetes: if the patient has diabetes (boolean)
        ejection fraction: percentage of blood leaving the heart at each contraction (%)
        platelets: platelets in the blood (kiloplatelets/mL)
        sex: woman or man (binary)
        serum creatinine: level of serum creatinine in the blood (mg/dL)
        serum sodium: level of serum sodium in the blood (mEq/L)
        smoking: if the patient smokes or not (boolean)
        time: follow-up period (days)
    """

    age: float
    anaemia: bool
    creatinine_phosphokinase: float
    diabetes: bool
    ejection_fraction: float
    high_blood_pressure: bool
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: bool
    smoking: bool
    time: float

    def __getitem__(self, item):
        return getattr(self, item)

    # noinspection PyMethodParameters
    @validator("*")
    def check_data_integrity(cls, v: Any):
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError(
                    f"All numerical features should be greater or equal than 0"
                )
            return float(v)
