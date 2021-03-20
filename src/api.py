import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .model import EnsembleModel
from .patient_data import PatientData


class ModelResponse(BaseModel):
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


@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(
        message="This is a test endpoint. Please, use "
        "/train for training an Ensemble Model and "
        "/predict for predicting on new data "
    )


@app.get("/train", response_model=ModelResponse)
async def explain_train_api() -> ModelResponse:
    return ModelResponse(message="Send a POST request to train an Ensemble Model.")


@app.post("/train", response_model=ModelResponse)
async def train_model(
    data_path: str = os.path.join("data", "heart_failure_clinical_records_dataset.csv"),
    num_folds: int = 5,
    seed: int = 72,
) -> ModelResponse:
    scores_dict = app.model.train(data_path=data_path, num_folds=num_folds, seed=seed)

    scores_str = ""
    for metric_name, metric_value in scores_dict.items():
        scores_str += f"{metric_name}={metric_value:.4f} "

    return ModelResponse(
        message=f"Model was trained successfully with the following Out of Fold "
        f"validation scores: {scores_str}"
    )


@app.get("/predict", response_model=ModelResponse)
async def explain_predict_api() -> ModelResponse:
    return ModelResponse(
        message="Send a POST request to this endpoint with Patient data json."
    )


@app.post("/predict", response_model=ModelResponse)
async def predict(patient_data: PatientData) -> ModelResponse:
    try:
        prob, pred = app.model.predict(data=patient_data)
        return ModelResponse(
            message="Successful prediction.",
            death_event_probability=prob,
            death_event_prediction=pred,
        )
    except ValueError as e:
        return ModelResponse(message=f"Failed prediction due to {e}")
