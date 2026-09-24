"""Tests for the pandemic-free re-run.

The README's most consequential claim is now the one that says most of the
published margin does not survive a move to an ordinary period. These guard the
two things that would make that claim meaningless: the re-implementation drifting
away from the published pipeline, and the "ordinary" window quietly including a
pandemic year.
"""

import json

import pytest

from dps.data import load_monthly
from dps.pipeline import RESULTS as REFERENCE_RUN
from dps.robustness import PANDEMIC_FREE, PUBLISHED, RESULTS, run_window

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def df():
    return load_monthly()


@pytest.fixture(scope="module")
def published(df):
    return run_window(df, *PUBLISHED, label="published")


@pytest.fixture(scope="module")
def pandemic_free(df):
    return run_window(df, *PANDEMIC_FREE, label="pandemic-free")


def test_the_rerun_reproduces_the_published_window(published):
    """The comparison is only worth reading if both halves come from the same code.

    This is the load-bearing test. `run_window` re-implements the protocol so it
    can be pointed at a different period; if it drifted from `dps.pipeline` the
    second row of the README table would be measuring something else.
    """
    stored = json.loads(REFERENCE_RUN.read_text(encoding="utf-8"))

    for method in ("seasonal_trend", "seasonal_naive", "series_mean"):
        for metric in ("scaled_mae", "mape_percent", "mae", "rmse"):
            expected = stored["macro"][method][metric]
            actual = published.macro_scores[method][metric]
            assert abs(expected - actual) / max(abs(expected), 1e-300) < TOLERANCE, (
                f"{method}.{metric}: pipeline says {expected}, robustness says {actual}"
            )


def test_the_ordinary_window_contains_no_pandemic_year(pandemic_free):
    """2020 and 2021 are the years the whole exercise is trying to step around."""
    first, last = pandemic_free.test_years

    assert last < 2020, f"held out {first}-{last}, which reaches into the pandemic"
    assert pandemic_free.train_end_year < 2020


def test_both_windows_hold_out_the_same_number_of_rows(published, pandemic_free):
    """Two years of seven monthly series. Different sizes would confound the comparison."""
    assert published.test_rows == pandemic_free.test_rows == 7 * 12 * 2


def test_the_window_is_selected_before_the_held_out_years(pandemic_free):
    """Selection may not see the years it will later be scored on."""
    validation_end = pandemic_free.validation_years[1]
    first_test_year = pandemic_free.test_years[0]

    assert validation_end <= pandemic_free.train_end_year < first_test_year


def test_margin_is_relative_to_the_baseline(published):
    """A share, not a difference, so the two periods can be compared at different levels."""
    model = published.macro_scores["seasonal_trend"]["scaled_mae"]
    baseline = published.macro_scores["seasonal_naive"]["scaled_mae"]

    assert published.margin() == pytest.approx((baseline - model) / baseline)


def test_the_published_window_still_wins_and_the_ordinary_one_does_not(published, pandemic_free):
    """The finding itself, pinned.

    If a later change makes the model win on an ordinary period too, this test
    fails and the README paragraph has to be rewritten rather than left standing.
    """
    assert published.margin() > 0
    assert published.beats_baseline_on() == 7

    assert pandemic_free.margin() < 0
    assert pandemic_free.beats_baseline_on() < 7


def test_the_stored_run_matches_the_readme_table():
    """Guards against the artifact and the prose drifting apart."""
    if not RESULTS.is_file():
        pytest.skip("results/pandemic_free_run.json not generated in this environment")

    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    clean = stored["windows"]["pandemic_free"]

    assert stored["margin_survives"] is False
    assert clean["series_where_the_model_wins"] == 3
    assert clean["series_total"] == 7
    assert clean["test_years"] == [2018, 2019]
