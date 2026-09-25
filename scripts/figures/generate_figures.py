"""Figures for this repository, drawn from the data and the reference run.

Three figures, each answering one question:

    01  what is in the data, and why one model cannot serve all of it
    02  does the model beat the baselines, and on which series
    03  what the held-out forecast actually looks like against the truth

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import portfolio_style as ps  # noqa: E402

from accidents.data import TRAIN_END_YEAR, load_monthly, series_list, split  # noqa: E402
from accidents.model import SeasonalNaive, SeasonalTrend, fit_all  # noqa: E402

OUT = ROOT / "docs" / "figures"
RUN = ROOT / "results" / "reference_run.json"

SHORT = {
    "Alkoholunfälle": "Alcohol",
    "Fluchtunfälle": "Hit and run",
    "Verkehrsunfälle": "All traffic",
}
KIND = {
    "insgesamt": "total",
    "Verletzte und Getötete": "injured and killed",
    "mit Personenschäden": "with injuries",
}


def label(category: str, kind: str) -> str:
    return f"{SHORT.get(category, category)}, {KIND.get(kind, kind)}"


def load_run() -> dict:
    if not RUN.exists():
        raise SystemExit("results/reference_run.json is missing. Run: python -m accidents.pipeline")
    return json.loads(RUN.read_text(encoding="utf-8"))


def fig_series(df, run):
    """The scale problem, which is the reason the first model could not work."""
    order = sorted(series_list(df), key=lambda s: -df[(df.category == s.category)
                                                      & (df.kind == s.kind)]["value"].mean())
    fig = plt.figure(figsize=(12.6, 7.6))
    ax = fig.add_axes([0.075, 0.325, 0.62, 0.45])
    axR = fig.add_axes([0.775, 0.325, 0.185, 0.45])

    for i, s in enumerate(order):
        part = df[(df.category == s.category) & (df.kind == s.kind)]
        t = part["year"] + (part["month"] - 1) / 12.0
        colour = ps.SEVEN[i]
        ax.plot(t, part["value"], lw=1.15, color=colour, alpha=0.9,
                label=label(s.category, s.kind))
    ax.axvline(TRAIN_END_YEAR + 1, color=ps.INK, lw=1.3, ls="--", zorder=5)
    ax.text(TRAIN_END_YEAR + 1.08, ax.get_ylim()[1] * 0.96, "held out",
            fontsize=9.6, color=ps.INK, va="top")
    ax.set_yscale("log")
    ps.clean(ax)
    ax.set_xlabel("year", fontsize=10.4)
    ax.set_ylabel("accidents per month (log scale)", fontsize=10.4)
    ax.legend(fontsize=8.6, loc="upper left", bbox_to_anchor=(0.0, -0.13),
              ncol=4, handlelength=1.4, columnspacing=1.4)

    means = [df[(df.category == s.category) & (df.kind == s.kind)]["value"].mean()
             for s in order]
    ypos = np.arange(len(order))[::-1]
    for i, (_s, m) in enumerate(zip(order, means, strict=True)):
        axR.barh(ypos[i], m, color=ps.SEVEN[i], height=0.62,
                 alpha=0.85, zorder=3)
        axR.text(m * 1.12, ypos[i], f"{m:,.0f}", va="center", fontsize=9.0,
                 color=ps.SEVEN[i])
    axR.set_xscale("log")
    axR.set_yticks([])
    axR.set_xlim(8, 12000)
    ps.clean(axR, left=False, grid_axis="x")
    axR.set_xlabel("mean per month", fontsize=10.2)
    ratio = run['scale']['ratio']
    axR.text(0, 1.04, f"the largest is {ratio:.0f} times the smallest",
             transform=axR.transAxes, fontsize=10.2, color=ps.INK,
             fontweight="600", va="bottom")
    ps.title_block(
        fig, "One file, seven series, and no shared scale",
        "Munich publishes monthly counts for three accident categories crossed with "
        "an accident type. They share\na file, not a scale.", y=0.955, size=20)
    ps.footnote(fig, [
        f"The largest series averages {max(means):,.0f} accidents a month and the "
        f"smallest {min(means):,.0f}, a factor of {ratio:.0f} apart.",
        "A model given only a year and a month, with no way to tell the series "
        "apart, can do no better than predict their shared average.",
        f"That is what the original model did. Each series is fitted separately "
        f"here, and everything from {TRAIN_END_YEAR + 1} onwards is held out.",
        f"Source: the city's monthly export, "
        f"{run['dataset']['monthly_observations']:,} observations, "
        f"{run['dataset']['first_year']} to {run['dataset']['last_year']}."], y=0.145)
    ps.save(fig, OUT, "01_series")


def fig_benchmark(run):
    """Does the model beat the baselines, and where."""
    per = run["per_series"]
    keys = sorted(per, key=lambda k: -per[k]["test_mean"])
    methods = [("seasonal_trend", "Seasonal trend", ps.BLUE),
               ("seasonal_naive", "Seasonal naive", ps.AMBER),
               ("series_mean", "Series mean", ps.SLATE)]

    fig = plt.figure(figsize=(12.6, 7.8))
    ax = fig.add_axes([0.30, 0.235, 0.42, 0.50])
    axR = fig.add_axes([0.795, 0.235, 0.165, 0.50])

    n = len(keys)
    height = 0.26
    ypos = np.arange(n)[::-1]
    for j, (name, nice, colour) in enumerate(methods):
        vals = [per[k]["methods"][name]["scaled_mae"] for k in keys]
        ax.barh(ypos + (1 - j) * height, vals, height=height, color=colour,
                alpha=0.88, zorder=3, label=nice)
    for i, k in enumerate(keys):
        ax.text(-0.012, ypos[i], label(per[k]["category"], per[k]["kind"]),
                ha="right", va="center", fontsize=9.6, color=ps.INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("mean absolute error, scaled by the series' own level", fontsize=10.4)
    ax.legend(fontsize=9.2, loc="upper left", bbox_to_anchor=(0.62, 1.02))

    macro = run["macro"]
    for j, (name, _nice, colour) in enumerate(methods):
        v = macro[name]["scaled_mae"]
        axR.bar(j, v, color=colour, alpha=0.88, width=0.62, zorder=3)
        axR.text(j, v + 0.008, f"{v:.3f}", ha="center", fontsize=9.6, color=colour,
                 fontweight="600")
    axR.set_xticks(range(len(methods)))
    axR.set_xticklabels([m[1].replace(" ", "\n") for m in methods], fontsize=9.0)
    ps.clean(axR)
    axR.set_ylim(0, max(macro[m[0]]["scaled_mae"] for m in methods) * 1.25)
    axR.text(0, 1.04, "averaged over the seven", transform=axR.transAxes,
             fontsize=10.2, color=ps.INK, fontweight="600", va="bottom")

    trend = macro["seasonal_trend"]["scaled_mae"]
    naive = macro["seasonal_naive"]["scaled_mae"]
    win = sum(1 for k in keys
              if per[k]["methods"]["seasonal_trend"]["scaled_mae"]
              < per[k]["methods"]["seasonal_naive"]["scaled_mae"])
    ps.title_block(
        fig, "The model earns its place against a hard baseline",
        "Held out years 2021 and 2022, never used for fitting or for choosing the "
        "trend window. Errors are scaled by\neach series' own mean so the small "
        "series are not drowned out by the large ones.", y=0.955, size=20)
    ps.footnote(fig, [
        f"Repeating the same calendar month of the previous year is a strong "
        f"baseline for seasonal counts. The model beats it on {win} of {len(keys)} "
        f"series and improves the average scaled error from {naive:.3f} to "
        f"{trend:.3f}.",
        "The series mean is the floor. Anything that cannot beat it has learned "
        "nothing about time at all.",
        "Source: results/reference_run.json."], y=0.105)
    ps.save(fig, OUT, "02_model_vs_baselines")


def fig_forecast(df, run):
    """What the held-out forecast looks like next to the truth."""
    train, test = split(df, TRAIN_END_YEAR)
    window = run["model_selection"]["chosen"]
    fitted = fit_all(train, lambda: SeasonalTrend(window=window))
    naive = fit_all(train, SeasonalNaive)

    chosen = ["Verkehrsunfälle|insgesamt", "Fluchtunfälle|insgesamt",
              "Verkehrsunfälle|Verletzte und Getötete", "Alkoholunfälle|insgesamt"]
    fig = plt.figure(figsize=(12.6, 8.0))
    boxes = [[0.075, 0.505, 0.40, 0.215], [0.565, 0.505, 0.40, 0.215],
             [0.075, 0.165, 0.40, 0.215], [0.565, 0.165, 0.40, 0.215]]

    for key, box in zip(chosen, boxes, strict=True):
        cat, kind = key.split("|")
        ax = fig.add_axes(box)
        recent = df[(df.category == cat) & (df.kind == kind) & (df.year >= 2017)]
        t = recent["year"] + (recent["month"] - 1) / 12.0
        ax.plot(t, recent["value"], color=ps.SLATE, lw=1.3, label="observed", zorder=3)

        part = test[(test.category == cat) & (test.kind == kind)]
        tt = part["year"] + (part["month"] - 1) / 12.0
        ax.plot(tt, fitted[key].predict(part["year"].to_numpy(), part["month"].to_numpy()),
                color=ps.BLUE, lw=1.9, label="seasonal trend", zorder=5)
        ax.plot(tt, naive[key].predict(part["year"].to_numpy(), part["month"].to_numpy()),
                color=ps.AMBER, lw=1.4, ls="--", label="seasonal naive", zorder=4)
        ax.axvspan(TRAIN_END_YEAR + 1, 2023, color=ps.BLUE_SOFT, alpha=0.16, zorder=0)
        ps.clean(ax)
        ax.set_xlim(2017, 2023)
        ax.text(0, 1.06, label(cat, kind), transform=ax.transAxes, fontsize=10.6,
                color=ps.INK, fontweight="600", va="bottom")
        e = run["per_series"][key]["methods"]["seasonal_trend"]
        ax.text(1.0, 1.06, f"MAE {e['mae']:.1f}", transform=ax.transAxes, fontsize=9.4,
                color=ps.BLUE, ha="right", va="bottom")
        if box is boxes[0]:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=9.4, ncol=3, frameon=False,
                       loc="upper left", bbox_to_anchor=(0.073, 0.815))

    ps.title_block(
        fig, "Held-out forecasts against what actually happened",
        "The shaded band is 2021 and 2022, which the model never saw. Four of the "
        "seven series are shown, chosen to\nspan the range of scales.", y=0.962, size=20)
    ps.footnote(fig, [
        "The trend model tracks both the seasonal shape and the level.",
        "The naive baseline carries 2020 forward, so it forecasts low on all seven "
        "series: 2020 was a pandemic year and a poor guide to what followed.",
        "Counts are clipped at zero, since a negative number of accidents is not a "
        "forecast. Source: results/reference_run.json."], y=0.088)
    ps.save(fig, OUT, "03_forecast_vs_actual")


def main() -> int:
    ps.apply()
    run = load_run()
    df = load_monthly()
    print(f"  {len(df)} observations, {len(series_list(df))} series\n")
    fig_series(df, run)
    fig_benchmark(run)
    fig_forecast(df, run)
    print(f"\nfigures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
