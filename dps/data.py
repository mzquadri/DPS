"""Loading and cleaning the Munich monthly road accident export.

The published file is awkward in three ways, and each needs handling before any
model sees it:

  - It is UTF-8 with a byte order mark. Read without accounting for the mark,
    the first column arrives named "﻿MONATSZAHL" and every later reference
    to MONATSZAHL fails.
  - The MONAT column holds YYYYMM, not a month number, and also carries the
    literal value "Summe" on annual subtotal rows that must not be mixed in
    with monthly observations.
  - Rows exist for months that have not been published yet. Their WERT is
    empty, which is not the same thing as zero accidents.

The file covers three accident categories crossed with an accident type,
seven monthly series in total, from 2000 to 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv"

#: The last year used for fitting. The task is framed as a forecast made at the
#: end of 2020, so everything after this is held out and never fitted on.
TRAIN_END_YEAR = 2020


@dataclass(frozen=True)
class Series:
    """One monthly series: an accident category crossed with an accident type."""

    category: str
    kind: str

    @property
    def key(self) -> str:
        return f"{self.category}|{self.kind}"

    def __str__(self) -> str:
        return f"{self.category} / {self.kind}"


def read_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the export as UTF-8, discarding its byte order mark.

    The mark matters: read without accounting for it, the first column arrives
    named "﻿MONATSZAHL" and every lookup of MONATSZAHL fails.
    """
    return pd.read_csv(path or CSV, encoding="utf-8-sig")


def load_monthly(path: Path | None = None) -> pd.DataFrame:
    """Return one tidy row per (category, kind, year, month) that has a value.

    Annual subtotals, unpublished months and the derived columns of the export
    are all dropped, so what comes back is only observed monthly counts.
    """
    df = read_raw(path)
    df = df[df["MONAT"].astype(str).str.fullmatch(r"\d{6}")].copy()
    df["month"] = df["MONAT"].astype(int) % 100
    df["year"] = df["JAHR"].astype(int)
    df["value"] = pd.to_numeric(df["WERT"], errors="coerce")
    # An unpublished month is missing, not zero. Filling it with zero would
    # teach the model that accidents stopped happening.
    df = df.dropna(subset=["value"])
    df = df.rename(columns={"MONATSZAHL": "category", "AUSPRAEGUNG": "kind"})
    out = df[["category", "kind", "year", "month", "value"]].reset_index(drop=True)
    out["value"] = out["value"].astype(float)
    return out.sort_values(["category", "kind", "year", "month"]).reset_index(drop=True)


def series_list(df: pd.DataFrame) -> list[Series]:
    pairs = df[["category", "kind"]].drop_duplicates().itertuples(index=False)
    return sorted((Series(c, k) for c, k in pairs), key=lambda s: s.key)


def split(df: pd.DataFrame, train_end_year: int = TRAIN_END_YEAR):
    """Split in time, never at random.

    A random split over a time series lets the model see 2020 while predicting
    2015, which flatters it and answers a question nobody asked.
    """
    train = df[df["year"] <= train_end_year].reset_index(drop=True)
    test = df[df["year"] > train_end_year].reset_index(drop=True)
    return train, test
