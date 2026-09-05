# Provenance

## What this repository was

This started as a submission to the Digital Product School coding challenge. The
task supplies Munich's published monthly road accident statistics and asks for a
forecast served behind an HTTP endpoint.

The original submission is preserved in `notebooks/`, unmodified:

| File | What it is |
| --- | --- |
| `notebooks/Regression_model.ipynb` | The training code that produced the original model |
| `notebooks/DPS_Model.ipynb` | Exploratory aggregation of the data |
| `notebooks/Test_dps.ipynb` | Exploratory renaming and filtering |

The model artifact those notebooks produced, `Regressionmodel.pkl`, is no longer
in the working tree. It remains in the git history at commit `c4b8391` and
earlier.

## What was wrong with it

The original model was reproduced exactly from `notebooks/Regression_model.ipynb`
before anything was changed, so the following is measured rather than inferred.
Its fitted coefficients are `[2.56582329, 11.29976221]` with intercept
`-4452.13126265`, which the recipe in that notebook reproduces to the last digit.

**It was fitted across all seven series at once.** The notebook applies no filter
on `MONATSZAHL` or `AUSPRAEGUNG`. The file contains three accident categories
crossed with an accident type, seven monthly series whose means range from 21 to
3,512 accidents per month, a factor of 169. Regressing all of them on year and
month alone fits a single line through seven unrelated levels.

**The result had no predictive value.** On its own held-out split the model
scored R² of 0.0009 with a mean absolute error of 824, against a target whose
mean is 780. Predicting the overall average would have done about as well.

**The API could not identify a quantity.** It accepted a year and a month and
nothing else, so no request could distinguish alcohol-related injuries from total
traffic accidents. The endpoint was named `/deaths/` and the README described the
output as a predicted number of deaths. The data contains no death count. The
closest column, `Verletzte und Getötete`, combines injured and killed, and the
model was not fitted to it in isolation.

**The split leaked.** `train_test_split` with a random seed was applied to a time
series, so the model was scored on months surrounded in time by months it had
trained on.

**Missing months were filled with zero.** `df['WERT'].fillna(0)` turns an
unpublished month into a month with no accidents. The 2023 and 2024 rows carry no
values, and 2024 was excluded only as a side effect of the `JAHR <= 2020` filter.

## What was rebuilt

Everything under `dps/`, `app.py`, `tests/`, `scripts/` and `docs/` is new work.

| Area | Change |
| --- | --- |
| Data loading | Reads the file as UTF-8 with a byte order mark, drops annual `Summe` subtotals, converts `MONAT` from YYYYMM to a month number, and drops unpublished months rather than zero-filling them |
| Model | One model per series, linear trend plus monthly seasonality, predictions clipped at zero |
| Model selection | The trend window is chosen on 2019 and 2020, never on the test years |
| Evaluation | Temporal split, 2021 and 2022 held out, scored against a seasonal naive baseline and a series mean floor |
| API | Takes a category and an accident type alongside the year and month, exposes the available series, and loads the artifact once at startup |
| Reproducibility | Pinned dependency versions, since the previous artifact was written by scikit-learn 1.5.2 and loaded under 1.8.0 |

## Attribution

The dataset is published by the Landeshauptstadt München open data portal. It is
included here as a snapshot of the challenge input and is not redistributed under
any claim of ownership. Verify the portal's current terms before reusing it.

The challenge framing is the Digital Product School's. The implementation, the
evaluation design and the analysis above are the author's own.
