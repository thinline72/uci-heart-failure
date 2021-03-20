# uci-heart-failure


## Test curl commands

Negative example
```
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

Positive example
```
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
  "platelets": 0,
  "serum_creatinine": 9.0,
  "serum_sodium": 137,
  "sex": 1,
  "smoking": true,
  "time": 132
}'
```

Re-training model
```
curl -X 'POST' 'http://127.0.0.1:8000/train?num_folds=5&seed=82'
```