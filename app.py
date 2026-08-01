"""FastAPI service for the traffic-accident regression demonstration."""

from pathlib import Path
import os
import pickle
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path(__file__).with_name("Regressionmodel.pkl")


class PredictionRequest(BaseModel):
    year: Annotated[int, Field(ge=2000, le=2100, description="Calendar year")]
    month: Annotated[int, Field(ge=1, le=12, description="Calendar month")]


def load_model(model_path: Path):
    """Load a trusted, local scikit-learn model artifact."""
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def create_app(model_path: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Traffic Accident Prediction API",
        description="A demonstration API for a regression model trained on Munich traffic-accident data.",
        version="1.0.0",
    )
    resolved_model_path = model_path or Path(os.environ.get("DPS_MODEL_PATH", DEFAULT_MODEL_PATH))

    @app.get("/")
    def index():
        return {"service": "traffic-accident-prediction", "status": "ok"}

    @app.post("/deaths/")
    def predict_deaths(request: PredictionRequest):
        if not resolved_model_path.is_file():
            raise HTTPException(status_code=503, detail="Model artifact is not available.")

        # Only load artifacts produced by a trusted local training workflow.
        model = load_model(resolved_model_path)
        prediction = model.predict([[request.year, request.month]])
        return {"prediction": round(float(prediction[0]))}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
