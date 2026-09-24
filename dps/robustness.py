"""How much of the margin over seasonal naive is the pandemic, and how much is the model?

    python -m dps.robustness
    python -m dps.robustness --check

The published result holds out 2021 and 2022. The README already says the obvious
thing about that: the seasonal naive baseline carries 2020 forward, 2020 was a
pandemic year, so the baseline forecasts low on all seven series and part of the
margin is that timing rather than the model. Saying it is not the same as
measuring it.

This runs the identical protocol three years earlier, where nothing unusual
happened. Window selected on 2016 and 2017, refitted on everything to 2017,
scored once on 2018 and 2019. The baseline there carries 2017 forward, which is
an ordinary year, so whatever margin survives is the model's.

Nothing here touches `results/reference_run.json`. This is a second, separate
measurement with its own artifact, because the published result is the answer to
the challenge as it was set and should not move.

Writes:
    results/pandemic_free_run.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .data import ROOT, load_monthly, series_list, split
from .model import SeasonalNaive, SeasonalTrend, SeriesMean, fit_all
from .pipeline import TOLERANCE, choose_window, comparable, differences, macro, scores

RESULTS = ROOT / "results" / "pandemic_free_run.json"

#: The published protocol, restated here so the two are visibly the same shape.
PUBLISHED = (2019, 2020), 2020

#: The same protocol moved back three years. 2018 and 2019 are both ordinary, and
#: so is 2017, which is what the seasonal naive baseline repeats.
PANDEMIC_FREE = (2016, 2017), 2017

METHODS = ("seasonal_trend", "seasonal_naive", "series_mean")


@dataclass(frozen=True)
class Window:
    """One run of the protocol over one stretch of the record."""

    label: str
    validation_years: tuple[int, int]
    train_end_year: int
    chosen_window: int | None
    test_years: tuple[int, int]
    test_rows: int
    macro_scores: dict
    per_series: dict

    def margin(self, metric: str = "scaled_mae") -> float:
        """How far seasonal trend sits below seasonal naive, as a share of the baseline.

        Expressed as a share rather than a difference so the two periods can be
        compared: the levels are not the same, and a raw difference in scaled MAE
        would mostly reflect that.
        """
        model = self.macro_scores["seasonal_trend"][metric]
        baseline = self.macro_scores["seasonal_naive"][metric]
        return float((baseline - model) / baseline) if baseline else float("nan")

    def beats_baseline_on(self) -> int:
        """Series where seasonal trend has the lower scaled MAE."""
        return sum(
            1
            for entry in self.per_series.values()
            if entry["methods"]["seasonal_trend"]["scaled_mae"]
            < entry["methods"]["seasonal_naive"]["scaled_mae"]
        )

    def as_dict(self) -> dict:
        return {
            "validation_years": list(self.validation_years),
            "train_end_year": self.train_end_year,
            "test_years": list(self.test_years),
            "test_rows": self.test_rows,
            "chosen_trend_window_years": self.chosen_window,
            "macro": self.macro_scores,
            "margin_over_seasonal_naive_scaled_mae": round(self.margin(), 6),
            "margin_over_seasonal_naive_mape": round(self.margin("mape_percent"), 6),
            "series_where_the_model_wins": self.beats_baseline_on(),
            "series_total": len(self.per_series),
        }


def run_window(df: pd.DataFrame, validation_years: tuple[int, int], train_end_year: int,
               label: str) -> Window:
    """Select, refit and score exactly as the published pipeline does."""
    train, test = split(df, train_end_year)
    # Nothing after the held-out years may be visible, or the earlier window
    # would be scored with knowledge the published one did not have.
    test = test[test["year"] <= train_end_year + 2].reset_index(drop=True)

    if test.empty:
        raise SystemExit(f"{label}: no rows left after {train_end_year}")

    window, _ = choose_window(df[df["year"] <= train_end_year], validation_years)

    factories = {
        "seasonal_trend": lambda: SeasonalTrend(window=window),
        "seasonal_naive": SeasonalNaive,
        "series_mean": SeriesMean,
    }
    fitted = {name: fit_all(train, factory) for name, factory in factories.items()}

    per_series: dict[str, dict] = {}
    for s in series_list(df):
        mask = (test["category"] == s.category) & (test["kind"] == s.kind)
        part = test[mask]
        entry = {"category": s.category, "kind": s.kind, "test_rows": len(part),
                 "test_mean": float(part["value"].mean()), "methods": {}}
        for name in METHODS:
            predicted = fitted[name][s.key].predict(
                part["year"].to_numpy(), part["month"].to_numpy()
            )
            entry["methods"][name] = scores(part["value"].to_numpy(), predicted)
        per_series[s.key] = entry

    macro_scores = {
        name: {m: macro(per_series, name, m)
               for m in ("mae", "rmse", "mape_percent", "scaled_mae")}
        for name in METHODS
    }

    return Window(
        label=label,
        validation_years=validation_years,
        train_end_year=train_end_year,
        chosen_window=window,
        test_years=(int(test["year"].min()), int(test["year"].max())),
        test_rows=len(test),
        macro_scores=macro_scores,
        per_series=per_series,
    )


def report(window: Window) -> None:
    print(f"  {window.label}")
    print(f"    window selected on {window.validation_years[0]}-{window.validation_years[1]}, "
          f"trained to {window.train_end_year}, scored on "
          f"{window.test_years[0]}-{window.test_years[1]} ({window.test_rows} rows)")
    print(f"    trend window chosen: {window.chosen_window or 'all'} years")
    for name in METHODS:
        s = window.macro_scores[name]
        print(f"      {name:<16} scaled MAE {s['scaled_mae']:.4f}   "
              f"MAPE {s['mape_percent']:6.2f}%")
    print(f"    margin over seasonal naive: {window.margin():.1%} of the baseline, "
          f"winning on {window.beats_baseline_on()} of {len(window.per_series)} series\n")


def main(check: bool = False) -> int:
    df = load_monthly()

    published = run_window(df, *PUBLISHED, label="published window")
    clean = run_window(df, *PANDEMIC_FREE, label="pandemic-free window")

    print()
    report(published)
    report(clean)

    drop = published.margin() - clean.margin()
    survives = clean.margin() > 0 and clean.beats_baseline_on() >= 4

    if survives:
        finding = (
            f"The model still beats the baseline away from the pandemic, by "
            f"{clean.margin():.1%} against {published.margin():.1%} on the published window. "
            f"Of the published margin, {drop / published.margin():.0%} does not reproduce on an "
            f"ordinary period and should be read as the baseline's timing."
        )
    else:
        finding = (
            f"The margin does not survive the move to an ordinary period: "
            f"{clean.margin():.1%} against {published.margin():.1%}, winning on "
            f"{clean.beats_baseline_on()} of {len(clean.per_series)} series. The published "
            f"result should be read as mostly the baseline's timing."
        )
    print("  FINDING:", finding)

    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(),
        },
        "question": (
            "The published margin over seasonal naive is measured on 2021-2022, where the "
            "baseline repeats a pandemic year. How much of it survives on a period the "
            "pandemic does not touch?"
        ),
        "method": (
            "The published protocol run unchanged three years earlier: trend window selected "
            "on the two validation years, refitted on everything up to the train end year, "
            "scored once on the two years after it."
        ),
        "windows": {
            "published": published.as_dict(),
            "pandemic_free": clean.as_dict(),
        },
        "margin_lost_to_the_pandemic": round(drop, 6),
        "margin_share_lost": round(drop / published.margin(), 6) if published.margin() else None,
        "margin_survives": bool(survives),
        "finding": finding,
        "per_series": {
            "published": published.per_series,
            "pandemic_free": clean.per_series,
        },
    }

    if check:
        if not RESULTS.is_file():
            print(f"\n  {RESULTS.name} is missing, nothing to check against", file=sys.stderr)
            return 1
        stored = json.loads(RESULTS.read_text(encoding="utf-8"))
        diffs = differences(comparable(stored), comparable(payload))
        if not diffs:
            print(f"\n  {RESULTS.name} matches this run to within {TOLERANCE:g} relative")
            return 0
        print(f"\n  {RESULTS.name} does not match this run ({len(diffs)} values beyond "
              f"{TOLERANCE:g} relative):", file=sys.stderr)
        for line in diffs[:25]:
            print(f"    {line}", file=sys.stderr)
        return 1

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"\n  wrote {RESULTS.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run the protocol away from the pandemic.")
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare against the stored run without writing")
    raise SystemExit(main(**vars(parser.parse_args())))
