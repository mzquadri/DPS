# Evaluation

Every number here is written by `python -m dps.pipeline` into
`results/reference_run.json`. Nothing in this file is typed by hand.

## Split

The task is framed as a forecast made at the end of 2020, so the split follows
that framing rather than being drawn at random.

| Part | Years | Rows | Used for |
| --- | --- | --- | --- |
| Train | 2000 to 2018 | 1,596 | Fitting during model selection |
| Validation | 2019 to 2020 | 168 | Choosing the trend window |
| Train, final | 2000 to 2020 | 1,764 | Refitting the chosen configuration |
| Test | 2021 to 2022 | 168 | Scored once |

A random split would let the model see 2020 while predicting 2015. That flatters
the result and answers a question nobody asks of a forecast.

## Choosing the trend window

Munich's accident counts in 2003 say little about 2021, so the linear trend is
fitted on a recent window rather than the whole record. The window length is a
choice, and it is made on 2019 and 2020 with the test years untouched.

| Window | Validation scaled MAE |
| --- | --- |
| all years | 0.2278 |
| 15 years | 0.2081 |
| 12 years | 0.2214 |
| 10 years | 0.2032 |
| 8 years | 0.1727 |
| **6 years** | **0.1491** |
| 4 years | 0.1673 |

Six years was chosen and then refitted on everything through 2020.

## Metrics

The seven series are not comparable in size, so two aggregates are reported and
they answer different questions.

**Macro** averages a metric across the seven series after scaling each by its own
level. This is the fair comparison and the one to read first.

**Micro** pools every row. It is dominated by the largest series, since total
traffic accidents run near 3,200 a month and alcohol-related injuries near 20.
It is reported because it is what a single pooled error figure would give, and
seeing the gap between the two is the point.

Mean absolute percentage error is computed over rows where the observed count is
non-zero, since the smallest series does reach zero. The row count is recorded
alongside it.

## Results on 2021 and 2022

Macro, averaged over the seven series:

| Method | Scaled MAE | MAPE |
| --- | --- | --- |
| Seasonal trend | 0.153 | 20.5% |
| Seasonal naive | 0.200 | 25.4% |
| Series mean | 0.296 | 42.2% |

Micro, all 168 held-out rows pooled:

| Method | MAE | RMSE |
| --- | --- | --- |
| Seasonal trend | 56.1 | 117.7 |
| Seasonal naive | 86.4 | 187.9 |
| Series mean | 127.3 | 239.6 |

The seasonal trend model beats the seasonal naive baseline on all seven series.

## Why the baseline is the one that matters

Repeating the same calendar month of the previous year is a strong baseline for
seasonal counts, and it is the comparison that decides whether a model is worth
its complexity. Here it is also biased: it carries 2020 forward, and 2020 was a
pandemic year with depressed traffic. Measured on the held-out years, the naive
baseline forecasts low on all seven series, by 0.2 to 40.3 accidents per month
depending on the series.

That is a property of this particular test period, not a general weakness of the
method, and it is the main reason to treat the margin over the baseline here as
smaller than it looks.

## What this evaluation does not establish

The test period is two years and covers a pandemic and its rebound. A model
selected and scored across 2021 and 2022 is being asked about an unusual stretch
of time.

Nothing here supports operational use. These are monthly counts for a city,
forecast from their own history, with no covariates for weather, traffic volume,
policy, road works or reporting changes. The model has no way to anticipate a
change in any of them.
