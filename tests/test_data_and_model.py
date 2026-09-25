"""Tests for the things that were actually wrong before.

Each of these exists because the original implementation got it wrong, so they
are regression guards rather than coverage filler.
"""

import numpy as np
import pandas as pd
import pytest

from accidents.data import load_monthly, series_list, split
from accidents.model import SeasonalNaive, SeasonalTrend, design


def test_umlauts_survive_loading():
    """The file carries a UTF-8 BOM over a cp1252 body, which corrupts naive reads."""
    df = load_monthly()

    assert "Alkoholunfälle" in set(df["category"])
    assert not any("�" in c for c in df["category"].unique())


def test_month_is_a_month_not_yyyymm():
    """The MONAT column holds 202401. Feeding that to a model is meaningless."""
    df = load_monthly()

    assert df["month"].min() == 1
    assert df["month"].max() == 12


def test_annual_subtotals_are_excluded():
    """Summe rows are yearly totals and would swamp the monthly series."""
    df = load_monthly()

    assert len(df) == 1932
    assert len(series_list(df)) == 7


def test_unpublished_months_are_dropped_not_zero_filled():
    """2023 and 2024 rows exist with no value. Zero-filling invents an accident drop."""
    df = load_monthly()

    assert df["year"].max() == 2022
    assert (df["value"] == 0).sum() < 20


def test_split_is_temporal_with_no_overlap():
    train, test = split(load_monthly(), 2020)

    assert train["year"].max() == 2020
    assert test["year"].min() == 2021
    assert set(train["year"]).isdisjoint(set(test["year"]))


def test_design_matrix_encodes_january_as_the_reference():
    X = design(np.array([2020, 2020]), np.array([1, 7]), origin=2000.0)

    assert X.shape == (2, 12)
    assert X[0, 1:].sum() == 0        # January sets no dummy
    assert X[1, 1:].sum() == 1        # July sets exactly one


def test_trend_window_limits_the_fitting_years():
    df = load_monthly()
    one = df[(df["category"] == "Alkoholunfälle") & (df["kind"] == "insgesamt")]
    wide = SeasonalTrend(window=None).fit(one)
    narrow = SeasonalTrend(window=6).fit(one)

    assert wide.yearly_change != pytest.approx(narrow.yearly_change)


def test_predictions_are_clipped_at_zero():
    df = load_monthly()
    one = df[(df["category"] == "Alkoholunfälle") & (df["kind"] == "Verletzte und Getötete")]
    model = SeasonalTrend(window=6).fit(one)

    assert model.predict([2400], [1])[0] >= 0.0


def test_seasonal_naive_repeats_the_last_observed_year():
    frame = pd.DataFrame({
        "year": [2019] * 12 + [2020] * 12,
        "month": list(range(1, 13)) * 2,
        "value": [float(i) for i in range(12)] + [float(100 + i) for i in range(12)],
    })
    model = SeasonalNaive().fit(frame)

    assert model.predict([2021], [3])[0] == 102.0
