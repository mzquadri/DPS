"""The number the API serves must be the number the pipeline would report.

Two paths reach a forecast and they are not the same code. The pipeline scores with
`accidents.model.predict_frame`, which dispatches each row to its series through
`Series(category, kind).key`. The API loads the artifact, builds a key by hand as
`f"{category}|{kind}"`, looks the model up in a dict and rounds the result to an integer.

Nothing connected the two. A change to how a series is keyed, a category and a kind
swapped in the request handler, or a dropped rounding step would all leave every existing
test passing: the app tests assert that a prediction is series-specific and non-negative,
never that it equals what the pipeline produces for the same input.

The offline side here comes from the pipeline's own function rather than from a copy of
the API's arithmetic, which is the point. Comparing the service against a restatement of
itself proves only that the restatement was faithful.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from accidents.model import predict_frame
from app import create_app

MODEL_PATH = "models/forecaster.joblib"

#: Inside the training range, just after it, far outside it, and each month of a year.
DATES = [(2019, 6), (2021, 1), (2023, 7), (2024, 12), (2030, 6)]


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def request_rows(bundle) -> pd.DataFrame:
    """Every series crossed with every date, as the pipeline would see it."""
    rows = [
        {"category": entry["category"], "kind": entry["kind"], "year": year, "month": month}
        for entry in bundle["series"]
        for year, month in DATES
    ]
    return pd.DataFrame(rows)


def test_every_series_and_date_agrees(bundle, client):
    frame = request_rows(bundle)
    expected = predict_frame(bundle["models"], frame)

    mismatches = []
    for position, row in frame.iterrows():
        response = client.post("/predict", json={
            "category": row["category"], "kind": row["kind"],
            "year": int(row["year"]), "month": int(row["month"]),
        })
        assert response.status_code == 200, response.text
        served = response.json()["prediction"]
        if served != round(float(expected[position])):
            mismatches.append(
                f"{row['category']}|{row['kind']} {row['year']}-{row['month']:02d}: "
                f"pipeline {expected[position]:.3f} -> {round(float(expected[position]))}, "
                f"api {served}"
            )
    assert not mismatches, "\n".join(mismatches)


def test_the_comparison_actually_covered_every_series(bundle):
    """A parity test that silently checks nothing is worse than no parity test."""
    assert len(bundle["series"]) == 7
    assert len(request_rows(bundle)) == 7 * len(DATES)


def test_the_api_keys_a_series_the_way_the_data_module_does(bundle):
    """The API builds "category|kind" by hand; Series.key is the definition."""
    from accidents.data import Series

    for entry in bundle["series"]:
        built_by_hand = f"{entry['category']}|{entry['kind']}"
        assert built_by_hand == Series(entry["category"], entry["kind"]).key
        assert built_by_hand in bundle["models"]


def test_a_swapped_category_and_kind_is_not_silently_accepted(client, bundle):
    """The two fields are not interchangeable, so swapping them must not resolve."""
    entry = bundle["series"][0]
    response = client.post("/predict", json={
        "category": entry["kind"], "kind": entry["category"], "year": 2023, "month": 5,
    })
    assert response.status_code == 404


def test_umlauts_survive_the_round_trip(bundle, client):
    """The categories carry umlauts, and a mangled one would 404 rather than mispredict."""
    with_umlaut = [e for e in bundle["series"] if any(c in e["category"] for c in "äöüÄÖÜ")]
    assert with_umlaut, "expected at least one category with an umlaut"
    for entry in with_umlaut:
        response = client.post("/predict", json={
            "category": entry["category"], "kind": entry["kind"], "year": 2023, "month": 5,
        })
        assert response.status_code == 200, f"{entry['category']} did not round-trip"
        assert response.json()["category"] == entry["category"]
