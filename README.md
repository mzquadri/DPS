# DPS: Traffic Accident Prediction API

A small FastAPI demonstration that serves a regression model trained from Munich traffic-accident data. Given a calendar year and month, it returns the model's rounded predicted number of deaths.

## Scope

This is an educational regression prototype, not an official public-safety forecast. The tracked model and CSV are repository artifacts from a historical-data experiment; they do not establish forecast accuracy for future decisions. Verify the source dataset, feature preparation, and evaluation in the included notebooks before reusing results.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/deaths/ -H "Content-Type: application/json" -d "{\"year\": 2025, \"month\": 6}"
```

## Model artifact

`Regressionmodel.pkl` is loaded relative to `app.py`, so the service can be started from another directory. To use a different trusted local artifact, set `DPS_MODEL_PATH` to its path before starting the service.

Pickle files can execute code when loaded. Do not configure the API to load model files from untrusted sources.

## Project files

- `app.py`: FastAPI application and input validation.
- `Regressionmodel.pkl`: tracked local regression artifact.
- `monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv`: source data snapshot.
- `*.ipynb`: exploratory training and test notebooks.

## License

The code is available under the MIT License. Verify the source-data terms separately before redistributing the dataset.
