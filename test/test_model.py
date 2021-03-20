import os
import unittest

from src.model import EnsembleModel
from src.patient_data import PatientData

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = EnsembleModel()
        cls.scores = cls.model.train(
            data_path=os.path.join(
                TEST_DIR_PATH,
                os.pardir,
                "data",
                "heart_failure_clinical_records_dataset.csv",
            )
        )

        cls.neg_data = PatientData(
            **{
                "age": 32,
                "anaemia": False,
                "creatinine_phosphokinase": 580.3,
                "diabetes": False,
                "ejection_fraction": 39,
                "high_blood_pressure": 1,
                "platelets": 263359,
                "serum_creatinine": 1.4,
                "serum_sodium": 137,
                "sex": 0,
                "smoking": 0,
                "time": 132,
            }
        )
        cls.pos_data = PatientData(
            **{
                "age": 70,
                "anaemia": True,
                "creatinine_phosphokinase": 580.3,
                "diabetes": True,
                "ejection_fraction": 62.3,
                "high_blood_pressure": 1,
                "platelets": 263359,
                "serum_creatinine": 9.0,
                "serum_sodium": 137,
                "sex": 1,
                "smoking": True,
                "time": 132,
            }
        )

    def test_model_quality(self):
        self.assertGreaterEqual(self.scores["MCC"], 0.5)
        self.assertGreaterEqual(self.scores["F1"], 0.65)
        self.assertGreaterEqual(self.scores["ROC_AUC"], 0.8)

    def test_predict_neg_correct(self):
        prob, pred = self.model.predict(data=self.neg_data)
        self.assertLessEqual(prob, self.model.cl_thr)
        self.assertFalse(pred)

    def test_predict_pos_correct(self):
        prob, pred = self.model.predict(data=self.pos_data)
        self.assertGreaterEqual(prob, self.model.cl_thr)
        self.assertTrue(pred)

    def test_predict_error_if_not_trained(self):
        model = EnsembleModel()
        with self.assertRaises(ValueError):
            model.predict(data=self.pos_data)
