from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import create_app


def test_health_endpoint():
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_uses_year_and_month(tmp_path):
    model_path = tmp_path / "model.pkl"
    model_path.touch()
    model = Mock()
    model.predict.return_value = [42.4]

    with patch("app.load_model", return_value=model):
        response = TestClient(create_app(model_path)).post("/deaths/", json={"year": 2025, "month": 6})

    assert response.status_code == 200
    assert response.json() == {"prediction": 42}
    model.predict.assert_called_once_with([[2025, 6]])


def test_invalid_month_is_rejected():
    response = TestClient(create_app()).post("/deaths/", json={"year": 2025, "month": 13})

    assert response.status_code == 422
