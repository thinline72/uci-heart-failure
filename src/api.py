import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .model import EnsembleModel
from .patient_data import PatientData


class TrainResponse(BaseModel):
    message: Optional[str]
    mcc_score: Optional[float]
    f1_score: Optional[float]
    roc_auc_score: Optional[float]


class PredictResponse(BaseModel):
    message: Optional[str]
    death_event_probability: Optional[float]
    death_event_prediction: Optional[bool]


app = FastAPI()
app.model = EnsembleModel()


@app.on_event("startup")
async def startup_event():
    app.model.train(
        data_path=os.path.join("data", "heart_failure_clinical_records_dataset.csv")
    )


@app.get("/")
async def root() -> str:
    return (
        "This is a test endpoint. Please, use "
        "/train for training an Ensemble Model and "
        "/predict for predicting on a new patient data."
    )


@app.get("/train", response_model=TrainResponse)
async def explain_train_api() -> TrainResponse:
    return TrainResponse(message="Send a POST request to train an Ensemble Model.")


@app.post("/train", response_model=TrainResponse)
async def train_model(
    data_path: str = os.path.join("data", "heart_failure_clinical_records_dataset.csv"),
    num_folds: int = 5,
    seed: int = 72,
) -> TrainResponse:
    scores_dict = app.model.train(data_path=data_path, num_folds=num_folds, seed=seed)

    scores_str = ""
    for metric_name, metric_value in scores_dict.items():
        scores_str += f"{metric_name}={metric_value:.4f} "

    return TrainResponse(
        message=f"Model was trained successfully with the following Out of Fold "
        f"validation scores: {scores_str}",
        mcc_score=scores_dict.get("MCC", None),
        f1_score=scores_dict.get("F1", None),
        roc_auc_score=scores_dict.get("ROC_AUC", None),
    )


@app.get("/predict", response_model=PredictResponse)
async def explain_predict_api() -> PredictResponse:
    return PredictResponse(
        message="Send a POST request to this endpoint with Patient data json."
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(patient_data: PatientData) -> PredictResponse:
    try:
        prob, pred = app.model.predict(data=patient_data)
        return PredictResponse(
            message="Successful prediction.",
            death_event_probability=prob,
            death_event_prediction=pred,
        )
    except ValueError as e:
        return PredictResponse(message=f"Failed prediction due to {e}")
