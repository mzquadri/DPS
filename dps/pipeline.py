"""Train, evaluate against baselines, and write the artifacts the API serves.

    python -m dps.pipeline

The trend window is chosen on 2019 and 2020, then the chosen setting is refitted
on everything up to 2020 and scored once on 2021 and 2022. The test years are
never used to make a modelling decision.

Two aggregates are reported and they answer different questions. The micro
figures pool every row, so they are dominated by the largest series: total
traffic accidents run near 3,200 a month while alcohol-related injuries run near
20. The macro figures average across the seven series after scaling each by its
own level, which is the fairer comparison and the one to read first.

Writes:
    models/forecaster.joblib     the seven fitted series models
    results/reference_run.json   every number this repository publishes
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import ROOT, TRAIN_END_YEAR, load_monthly, series_list, split
from .model import SeasonalNaive, SeasonalTrend, SeriesMean, fit_all, predict_frame

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "forecaster.joblib"
RESULTS = ROOT / "results" / "reference_run.json"

#: Trend windows considered, in years. None means the whole record.
WINDOWS = (None, 15, 12, 10, 8, 6, 4)
VALIDATION_YEARS = (2019, 2020)


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    nz = actual != 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_percent": (float(np.mean(np.abs(err[nz] / actual[nz])) * 100)
                         if nz.any() else float("nan")),
        "scaled_mae": (float(np.mean(np.abs(err)) / actual.mean())
                       if actual.mean() else float("nan")),
        "rows": len(actual),
        "rows_nonzero": int(nz.sum()),
    }


def macro(per_series: dict, method: str, metric: str) -> float:
    """Average a metric across the seven series, so no single series dominates."""
    vals = [v["methods"][method][metric] for v in per_series.values()]
    return float(np.nanmean(vals))


def choose_window(df: pd.DataFrame) -> tuple[int | None, list[dict]]:
    """Pick the trend window on 2019 and 2020, leaving the test years untouched."""
    lo, hi = VALIDATION_YEARS
    inner_train = df[df["year"] < lo]
    inner_val = df[(df["year"] >= lo) & (df["year"] <= hi)]
    trace = []
    for win in WINDOWS:
        vals = []
        for s in series_list(df):
            t = inner_train[(inner_train["category"] == s.category)
                            & (inner_train["kind"] == s.kind)]
            v = inner_val[(inner_val["category"] == s.category) & (inner_val["kind"] == s.kind)]
            m = SeasonalTrend(window=win).fit(t)
            p = m.predict(v["year"].to_numpy(), v["month"].to_numpy())
            a = v["value"].to_numpy(dtype=float)
            vals.append(np.mean(np.abs(p - a)) / a.mean())
        trace.append({"window": win, "validation_scaled_mae": float(np.mean(vals))})
    best = min(trace, key=lambda r: r["validation_scaled_mae"])
    return best["window"], trace


def comparable(payload: dict) -> dict:
    """The payload without the environment block, which varies by machine."""
    return {k: v for k, v in payload.items() if k != "environment"}


#: Floating point results are not bit-identical across platforms, because the
#: linear algebra underneath uses whatever BLAS the platform ships. This is far
#: tighter than any number this repository publishes, which are quoted to four
#: decimals at most, while still tolerating that noise.
TOLERANCE = 1e-9


def differences(stored, fresh, path: str = "") -> list[str]:
    """Every leaf that differs by more than the tolerance, with its path."""
    out: list[str] = []
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in sorted(set(stored) | set(fresh)):
            if key not in stored:
                out.append(f"{path}.{key} only in this run")
            elif key not in fresh:
                out.append(f"{path}.{key} only in the stored run")
            else:
                out += differences(stored[key], fresh[key], f"{path}.{key}")
    elif isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            out.append(f"{path} length {len(stored)} against {len(fresh)}")
        else:
            for i, (a, b) in enumerate(zip(stored, fresh, strict=True)):
                out += differences(a, b, f"{path}[{i}]")
    elif isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        if isinstance(stored, bool) or isinstance(fresh, bool):
            if stored is not fresh:
                out.append(f"{path}: {stored} against {fresh}")
        else:
            a, b = float(stored), float(fresh)
            if a != b and not (a != a and b != b):  # NaN compares unequal to itself
                scale = max(abs(a), abs(b), 1e-300)
                if abs(a - b) / scale > TOLERANCE:
                    out.append(f"{path}: {a!r} against {b!r} "
                               f"(relative {abs(a - b) / scale:.2e})")
    elif stored != fresh:
        out.append(f"{path}: {stored!r} against {fresh!r}")
    return out


def main(check: bool = False) -> int:
    df = load_monthly()
    train, test = split(df, TRAIN_END_YEAR)
    all_series = series_list(df)

    print(f"  {len(df)} monthly observations, {len(all_series)} series, "
          f"{df['year'].min()} to {df['year'].max()}")
    print(f"  train {len(train)} rows (to {TRAIN_END_YEAR}), "
          f"test {len(test)} rows ({test['year'].min()} to {test['year'].max()})\n")

    window, trace = choose_window(df)
    print(f"  trend window chosen on {VALIDATION_YEARS[0]}-{VALIDATION_YEARS[1]}:")
    for row in trace:
        mark = "  <-- chosen" if row["window"] == window else ""
        print(f"    {row['window'] or 'all'!s:>4} years   "
              f"scaled MAE {row['validation_scaled_mae']:.4f}{mark}")
    print()

    methods = {
        "seasonal_trend": lambda: SeasonalTrend(window=window),
        "seasonal_naive": SeasonalNaive,
        "series_mean": SeriesMean,
    }

    fitted_by_method = {name: fit_all(train, factory) for name, factory in methods.items()}
    micro = {}
    for name, fitted in fitted_by_method.items():
        micro[name] = scores(test["value"].to_numpy(), predict_frame(fitted, test))

    per_series: dict[str, dict] = {}
    for s in all_series:
        mask = (test["category"] == s.category) & (test["kind"] == s.kind)
        part = test[mask]
        entry = {
            "category": s.category,
            "kind": s.kind,
            "test_rows": len(part),
            "train_rows": int(((train["category"] == s.category)
                               & (train["kind"] == s.kind)).sum()),
            "test_mean": float(part["value"].mean()),
            "mean_all_years": float(df[(df["category"] == s.category)
                                       & (df["kind"] == s.kind)]["value"].mean()),
            "train_mean": float(train[(train["category"] == s.category)
                                      & (train["kind"] == s.kind)]["value"].mean()),
            "methods": {},
        }
        for name, fitted in fitted_by_method.items():
            p = fitted[s.key].predict(part["year"].to_numpy(), part["month"].to_numpy())
            entry["methods"][name] = scores(part["value"].to_numpy(), p)
        entry["yearly_change"] = fitted_by_method["seasonal_trend"][s.key].yearly_change
        per_series[s.key] = entry

    macro_scores = {
        name: {m: macro(per_series, name, m) for m in ("mae", "rmse", "mape_percent", "scaled_mae")}
        for name in methods
    }

    print("  held-out 2021-2022, macro average over the seven series:")
    for name in methods:
        print(f"    {name:<16} scaled MAE {macro_scores[name]['scaled_mae']:.4f}   "
              f"MAPE {macro_scores[name]['mape_percent']:6.2f}%")
    print("\n  same rows pooled (micro, dominated by the largest series):")
    for name in methods:
        print(f"    {name:<16} MAE {micro[name]['mae']:8.2f}   RMSE {micro[name]['rmse']:8.2f}")

    print("\n  per series, best method by scaled MAE:")
    for s in all_series:
        e = per_series[s.key]
        best = min(e["methods"], key=lambda m: e["methods"][m]["scaled_mae"])
        print(f"    {s!s:<45} mean {e['test_mean']:7.1f}   {best} "
              f"({e['methods'][best]['scaled_mae']:.3f})")

    import joblib
    import sklearn

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": fitted_by_method["seasonal_trend"],
            "series": [{"category": s.category, "kind": s.kind, "key": s.key} for s in all_series],
            "train_end_year": TRAIN_END_YEAR,
            "trend_window_years": window,
            "sklearn": sklearn.__version__,
        },
        MODEL_PATH,
    )

    payload = {
        # Kept out of --check: these say where the run happened, not what it
        # found, and they differ between machines and CI.
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "dataset": {
            "file": "monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv",
            "source": "Landeshauptstadt Muenchen, open data portal",
            "monthly_observations": len(df),
            "series": len(all_series),
            "first_year": int(df["year"].min()),
            "last_year": int(df["year"].max()),
        },
        "split": {
            "strategy": "temporal",
            "train_end_year": TRAIN_END_YEAR,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_years": [int(test["year"].min()), int(test["year"].max())],
            "validation_years": list(VALIDATION_YEARS),
        },
        "model_selection": {
            "parameter": "trend_window_years",
            "chosen": window,
            "selected_on": f"{VALIDATION_YEARS[0]}-{VALIDATION_YEARS[1]}",
            "candidates": trace,
        },
        "scale": {
            "note": "means over the whole record, the basis for the 'no shared scale' claim",
            "smallest": float(min(v["mean_all_years"] for v in per_series.values())),
            "largest": float(max(v["mean_all_years"] for v in per_series.values())),
            "ratio": float(max(v["mean_all_years"] for v in per_series.values())
                           / min(v["mean_all_years"] for v in per_series.values())),
        },
        "macro": macro_scores,
        "micro": micro,
        "per_series": per_series,
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
        print(f"\n  {RESULTS.name} does not match this run "
              f"({len(diffs)} values beyond {TOLERANCE:g} relative):", file=sys.stderr)
        for line in diffs[:25]:
            print(f"    {line}", file=sys.stderr)
        if len(diffs) > 25:
            print(f"    and {len(diffs) - 25} more", file=sys.stderr)
        return 1

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"\n  wrote {MODEL_PATH.relative_to(ROOT).as_posix()}")
    print(f"  wrote {RESULTS.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate and write artifacts.")
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare against the stored run without writing")
    raise SystemExit(main(**vars(parser.parse_args())))
