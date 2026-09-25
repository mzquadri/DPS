"""FastAPI service over the Munich monthly accident forecaster.

The service needs to know which series to forecast. There are seven of them and
they are not interchangeable: total traffic accidents run near 3,200 a month and
alcohol-related injuries near 20, so a request that names only a year and a month
does not identify a quantity.

    uvicorn app:app --reload

The artifact is produced by `python -m accidents.pipeline` and is loaded once at
startup rather than on every request.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = Path(__file__).with_name("models") / "forecaster.joblib"


class PredictionRequest(BaseModel):
    category: Annotated[str, Field(description="Accident category, for example Alkoholunfälle")]
    kind: Annotated[str, Field(description="Accident type, for example insgesamt")]
    year: Annotated[int, Field(ge=2000, le=2100, description="Calendar year")]
    month: Annotated[int, Field(ge=1, le=12, description="Calendar month, 1 to 12")]


class PredictionResponse(BaseModel):
    category: str
    kind: str
    year: int
    month: int
    prediction: int


def load_bundle(model_path: Path):
    """Load the forecaster bundle written by the training pipeline."""
    return joblib.load(model_path)


def create_app(model_path: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Munich accident forecaster",
        description=(
            "Monthly forecasts for Munich road accident counts, by category and "
            "accident type, from the city's published monthly statistics."
        ),
        version="2.0.0",
    )
    resolved = Path(model_path or os.environ.get("ACCIDENTS_MODEL_PATH", DEFAULT_MODEL_PATH))
    # Loaded once. Reloading per request would re-read the artifact on every call
    # and let the service drift if the file changed underneath it.
    app.state.bundle = load_bundle(resolved) if resolved.is_file() else None
    app.state.model_path = resolved

    @app.get("/")
    def index():
        ready = app.state.bundle is not None
        return {
            "service": "munich-accident-forecaster",
            "status": "ok" if ready else "model artifact unavailable",
            "model_loaded": ready,
        }

    @app.get("/series")
    def series():
        """The series the model can forecast. A caller cannot guess these."""
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available.")
        return {"series": app.state.bundle["series"],
                "trained_through": app.state.bundle["train_end_year"]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available.")
        key = f"{request.category}|{request.kind}"
        model = app.state.bundle["models"].get(key)
        if model is None:
            known = sorted(app.state.bundle["models"])
            raise HTTPException(
                status_code=404,
                detail={"message": "Unknown series.", "requested": key, "available": known},
            )
        value = float(model.predict([request.year], [request.month])[0])
        return PredictionResponse(
            category=request.category,
            kind=request.kind,
            year=request.year,
            month=request.month,
            prediction=round(value),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
