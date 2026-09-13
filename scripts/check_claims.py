"""Check the numbers in README.md and docs/ against results/reference_run.json.

Documentation drifts away from code quietly. This recomputes each published claim
from the reference run and fails if the text no longer states it.

Patterns carry context rather than matching a bare number, because finding "168"
somewhere in a document proves nothing.

    python scripts/check_claims.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "reference_run.json"


def load() -> dict:
    if not RUN.is_file():
        raise SystemExit("results/reference_run.json is missing. Run: python -m dps.pipeline")
    return json.loads(RUN.read_text(encoding="utf-8"))


def build(run: dict) -> list[tuple[str, Path, str]]:
    """Each claim as (description, file, regex that file must match)."""
    readme = ROOT / "README.md"
    ev = ROOT / "docs" / "evaluation.md"
    prov = ROOT / "docs" / "provenance.md"

    d = run["dataset"]
    s = run["split"]
    macro = run["macro"]
    sel = run["model_selection"]

    claims: list[tuple[str, Path, str]] = [
        ("dataset observations", readme,
         rf"\| Observations \| {d['monthly_observations']:,} monthly counts \|"),
        ("dataset series", readme, rf"\| Series \| {d['series']}, being"),
        ("dataset period", readme, rf"\| Period \| {d['first_year']} to {d['last_year']} \|"),
        ("held-out rows", readme, rf"{s['test_rows']} monthly observations, scored once"),
        # The limitations section states the training size when explaining why this
        # workload has no use for a GPU. It is derived from the split, so it is checked
        # like everything else rather than left as prose that can quietly go stale.
        ("training rows in the limitations", readme,
         rf"fitted on\s+{s['train_rows']:,} monthly observations"),
        ("held-out years", readme, rf"Held out: {s['test_years'][0]} and {s['test_years'][1]},"),
        # The prose may spell the number, so accept either form.
        ("trend window", readme,
         rf"most recent (?:{sel['chosen']}|six) years" if sel['chosen'] == 6
         else rf"most recent {sel['chosen']} years"),
        ("window selected on validation", readme,
         rf"made on {s['validation_years'][0]} and {s['validation_years'][1]}"),
    ]

    for method, label in (("seasonal_trend", "Seasonal trend"),
                          ("seasonal_naive", "Seasonal naive"),
                          ("series_mean", "Series mean")):
        scaled = f"{macro[method]['scaled_mae']:.3f}"
        mape = f"{macro[method]['mape_percent']:.1f}%"
        # Bold markers are allowed around either number in the README table.
        pat = rf"\| {label} \| \*?\*?{re.escape(scaled)}\*?\*? \| \*?\*?{re.escape(mape)}\*?\*? \|"
        claims.append((f"README macro row, {method}", readme, pat))
        claims.append((f"evaluation macro row, {method}", ev,
                       rf"\| {label} \| {re.escape(scaled)} \| {re.escape(mape)} \|"))

    micro = run["micro"]
    for method, label in (("seasonal_trend", "Seasonal trend"),
                          ("seasonal_naive", "Seasonal naive"),
                          ("series_mean", "Series mean")):
        pat = (rf"\| {label} \| {micro[method]['mae']:.1f} \| "
               rf"{micro[method]['rmse']:.1f} \|")
        claims.append((f"evaluation micro row, {method}", ev, pat))

    for row in sel["candidates"]:
        win = "all years" if row["window"] is None else f"{row['window']} years"
        val = f"{row['validation_scaled_mae']:.4f}"
        claims.append((f"window sweep, {win}", ev,
                       rf"\| \*?\*?{re.escape(win)}\*?\*? \| \*?\*?{re.escape(val)}\*?\*? \|"))

    claims += [
        ("split train rows", ev, rf"\| Train, final \| 2000 to {s['train_end_year']} \| "
                                 rf"{s['train_rows']:,} \|"),
        ("split test rows", ev, rf"\| Test \| {s['test_years'][0]} to {s['test_years'][1]} \| "
                                rf"{s['test_rows']} \|"),
        ("original coefficients", prov, r"\[2\.56582329, 11\.29976221\]"),
        ("original intercept", prov, r"-4452\.13126265"),
    ]

    # The scale ratio the README and provenance both cite, from one stored basis.
    sc = run["scale"]
    claims.append(("scale ratio in README", readme, rf"a factor of {sc['ratio']:.0f}"))
    claims.append(("scale ratio in provenance", prov, rf"a factor of {sc['ratio']:.0f}"))
    claims.append(("smallest and largest in README", readme,
                   rf"{sc['smallest']:.0f} to {sc['largest']:,.0f} accidents per month"))

    beats = sum(
        1 for v in run["per_series"].values()
        if (v["methods"]["seasonal_trend"]["scaled_mae"]
            < v["methods"]["seasonal_naive"]["scaled_mae"])
    )
    if beats == len(run["per_series"]):
        claims.append(("beats baseline on all series", readme, r"beats it on all seven series"))

    return claims


def main() -> int:
    run = load()
    claims = build(run)
    cache: dict[Path, str] = {}
    failures = []

    for desc, path, pattern in claims:
        if path not in cache:
            if not path.is_file():
                raise SystemExit(f"missing: {path.relative_to(ROOT).as_posix()}")
            cache[path] = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
        ok = re.search(pattern, cache[path]) is not None
        print(f"  {'ok  ' if ok else 'FAIL'}  {path.name:<16} {desc}")
        if not ok:
            failures.append((desc, path, pattern))

    if failures:
        print(f"\n{len(failures)} of {len(claims)} claims do not match:", file=sys.stderr)
        for desc, path, pattern in failures:
            print(f"  {path.name}: {desc}: no match for /{pattern}/", file=sys.stderr)
        return 1

    print(f"\nAll {len(claims)} published claims match results/reference_run.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
