from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app

BUNDLE = Path(__file__).resolve().parents[1] / "models" / "forecaster.joblib"


def client():
    return TestClient(create_app(BUNDLE))


def test_health_reports_the_model_is_loaded():
    body = client().get("/").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_series_endpoint_lists_every_series():
    body = client().get("/series").json()

    assert len(body["series"]) == 7
    assert body["trained_through"] == 2020
    keys = {s["key"] for s in body["series"]}
    assert "Alkoholunfälle|insgesamt" in keys


def test_prediction_is_series_specific():
    """The whole point of the rewrite: two series must not return the same number."""
    c = client()
    small = c.post("/predict", json={"category": "Alkoholunfälle", "kind": "insgesamt",
                                     "year": 2021, "month": 6}).json()
    large = c.post("/predict", json={"category": "Verkehrsunfälle", "kind": "insgesamt",
                                     "year": 2021, "month": 6}).json()

    assert small["prediction"] < 200
    assert large["prediction"] > 2000


def test_prediction_is_never_negative():
    body = client().post("/predict", json={"category": "Alkoholunfälle",
                                           "kind": "Verletzte und Getötete",
                                           "year": 2099, "month": 1}).json()

    assert body["prediction"] >= 0


def test_unknown_series_is_rejected_with_the_available_options():
    response = client().post("/predict", json={"category": "Nonexistent", "kind": "insgesamt",
                                               "year": 2021, "month": 6})

    assert response.status_code == 404
    assert response.json()["detail"]["available"]


def test_invalid_month_is_rejected():
    response = client().post("/predict", json={"category": "Alkoholunfälle",
                                               "kind": "insgesamt",
                                               "year": 2025, "month": 13})

    assert response.status_code == 422


def test_missing_artifact_reports_unavailable(tmp_path):
    c = TestClient(create_app(tmp_path / "absent.joblib"))

    assert c.get("/").json()["model_loaded"] is False
    assert c.post("/predict", json={"category": "Alkoholunfälle", "kind": "insgesamt",
                                    "year": 2021, "month": 6}).status_code == 503
