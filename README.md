# uci-heart-failure
A minimal ML model training and deployment example with FastAPI, Uvicorn and Docker based on [Heart failure clinical records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) dataset. 

## Running a service
In order to run service locally, `make run-local` command can be used. It'll install necessary dependencies and start a service. 
You'll need to have Python 3.6+ for running this command.

Alternatively, `make run-docker` command will build Docker image and start service within a Docker container. 
You'll need to have [Docker](https://www.docker.com/) installed for running this command.

## Service and API Description

Once service is started, a simple ensemble baseline model will be trained automatically which takes several seconds. 
There are two API endpoints: `/train` and `/predict`. Both should be called with a POST request.
- `/train` endpoint retrains the ensemble model by using new or default params. The resulting Out of Fold validation metrics will be returned.
- `/predict` endpoint makes model prediction on a new patient data that should be provided in JSON in the POST request (see test commands below). 
  It returns `death_event_probability` and `death_event_prediction` values.
  
The FastAPI generated docs can be accessed at `http://127.0.0.1:8000/docs` 

## Test curl commands

### Negative example
```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 32,
  "anaemia": false,
  "creatinine_phosphokinase": 580.3,
  "diabetes": false,
  "ejection_fraction": 39,
  "high_blood_pressure": 1,
  "platelets": 263359,
  "serum_creatinine": 1.4,
  "serum_sodium": 137,
  "sex": 0,
  "smoking": false,
  "time": 132
  }'
```

Expected result:
```json
{"message":"Successful prediction.","death_event_probability":0.05885180085897446,"death_event_prediction":false}
```

### Positive example
```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 70,
  "anaemia": true,
  "creatinine_phosphokinase": 580.3,
  "diabetes": true,
  "ejection_fraction": 62.3,
  "high_blood_pressure": true,
  "platelets": 263359,
  "serum_creatinine": 9.0,
  "serum_sodium": 137,
  "sex": 1,
  "smoking": true,
  "time": 132
}'
```

Expected result:
```json
{"message":"Successful prediction.","death_event_probability":0.6206271052360535,"death_event_prediction":true}
```

### Re-training model
```shell
curl -X 'POST' 'http://127.0.0.1:8000/train?num_folds=5&seed=82'
```

Expected result:
```json
{"message":"Model was trained successfully with the following Out of Fold validation scores: MCC=0.5386 F1=0.6931 ROC_AUC=0.8739 ","mcc_score":0.5385856891631564,"f1_score":0.693069306930693,"roc_auc_score":0.8738711001642037}
```

## Running tests locally
In order to run tests locally, `make run-unit-tests` command can be used. It'll install necessary dependencies and start a service. 
You'll need to have Python 3.6+ for running this command.