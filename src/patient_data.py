from typing import Any

from pydantic import BaseModel, validator

TARGET_NAME = "DEATH_EVENT"
BOOL_FEATS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
NUM_FEATS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]


class PatientData(BaseModel):
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
