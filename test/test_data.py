import unittest

from src.patient_data import PatientData


class TestData(unittest.TestCase):
    def test_check_num_features_are_positive(self):
        with self.assertRaises(ValueError):
            _ = PatientData(
                **{
                    "age": -70,
                    "anaemia": True,
                    "creatinine_phosphokinase": 580.3,
                    "diabetes": True,
                    "ejection_fraction": -62.3,
                    "high_blood_pressure": 1,
                    "platelets": 263359,
                    "serum_creatinine": 9.0,
                    "serum_sodium": 137,
                    "sex": 0,
                    "smoking": 0,
                    "time": 132,
                }
            )
