"""The forecasting model, and the baselines it has to beat.

Each of the seven series is modelled on its own. They are not comparable: total
traffic accidents run in the thousands per month while alcohol-related injuries
run in the tens, so a single model without a series input can only ever predict
their shared average. Fitting per series also makes the fitted trend and the
seasonal shape readable for each one.

Within a series the model is deliberately plain: a linear trend in time plus a
monthly seasonal offset. Twenty-one years of monthly data is 252 points, which
supports twelve seasonal terms and a slope, and not much more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .data import Series


def design(year: np.ndarray, month: np.ndarray, origin: float) -> np.ndarray:
    """Trend in years since the origin, plus eleven month dummies.

    January is the reference level, so its offset is folded into the intercept
    and the remaining eleven coefficients read as differences from January.
    """
    year = np.asarray(year, dtype=float)
    month = np.asarray(month, dtype=int)
    trend = (year + (month - 1) / 12.0 - origin).reshape(-1, 1)
    dummies = np.zeros((len(month), 11), dtype=float)
    for i, m in enumerate(month):
        if m >= 2:
            dummies[i, m - 2] = 1.0
    return np.hstack([trend, dummies])


@dataclass
class SeasonalTrend:
    """Linear trend plus monthly seasonality, fitted to one series.

    `window` limits fitting to the most recent N years. Munich traffic in 2003
    says little about Munich traffic in 2021, and a slope fitted across the whole
    record is dragged by history the forecast does not need. The window is chosen
    on a validation split, never on the test years.
    """

    origin: float = 2000.0
    window: int | None = None
    model: LinearRegression = field(default_factory=LinearRegression)

    def fit(self, frame: pd.DataFrame) -> SeasonalTrend:
        if self.window is not None:
            frame = frame[frame["year"] > frame["year"].max() - self.window]
        X = design(frame["year"].to_numpy(), frame["month"].to_numpy(), self.origin)
        self.model.fit(X, frame["value"].to_numpy(dtype=float))
        return self

    def predict(self, year, month) -> np.ndarray:
        X = design(np.atleast_1d(year), np.atleast_1d(month), self.origin)
        # A count cannot be negative, and the linear form does not know that.
        return np.clip(self.model.predict(X), 0.0, None)

    @property
    def yearly_change(self) -> float:
        return float(self.model.coef_[0])


@dataclass
class SeasonalNaive:
    """Predict the same calendar month of the most recent year that has one.

    This is the baseline that matters. Monthly accident counts are strongly
    seasonal and change slowly, so repeating last year is genuinely hard to
    beat and any model that cannot is not earning its complexity.
    """

    table: dict[int, float] = field(default_factory=dict)
    last_year: int = 0
    fallback: float = 0.0

    def fit(self, frame: pd.DataFrame) -> SeasonalNaive:
        self.last_year = int(frame["year"].max())
        recent = frame[frame["year"] == self.last_year]
        self.table = dict(zip(recent["month"].astype(int), recent["value"].astype(float),
                              strict=True))
        self.fallback = float(frame["value"].mean())
        return self

    def predict(self, year, month) -> np.ndarray:
        month = np.atleast_1d(month).astype(int)
        return np.array([self.table.get(int(m), self.fallback) for m in month], dtype=float)


@dataclass
class SeriesMean:
    """Predict the training mean of the series. The floor any model must clear."""

    mean: float = 0.0

    def fit(self, frame: pd.DataFrame) -> SeriesMean:
        self.mean = float(frame["value"].mean())
        return self

    def predict(self, year, month) -> np.ndarray:
        return np.full(len(np.atleast_1d(month)), self.mean, dtype=float)


#: Everything trained per series, keyed by the series identity.
Fitted = dict[str, object]


def fit_all(train: pd.DataFrame, factory) -> Fitted:
    out: Fitted = {}
    for (category, kind), frame in train.groupby(["category", "kind"], sort=True):
        out[Series(category, kind).key] = factory().fit(frame)
    return out


def predict_frame(fitted: Fitted, frame: pd.DataFrame) -> np.ndarray:
    """Predict for a tidy frame, dispatching each row to its own series model."""
    preds = np.empty(len(frame), dtype=float)
    for (category, kind), part in frame.groupby(["category", "kind"], sort=False):
        model = fitted[Series(category, kind).key]
        preds[part.index.to_numpy()] = model.predict(part["year"].to_numpy(),
                                                     part["month"].to_numpy())
    return preds
