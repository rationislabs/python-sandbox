# red-tide

Two questions, one experiment, on Python 3.15t (the free-threaded build):

1. **Can weather and recent counts predict a Florida red tide bloom a week ahead?** Yes, better
   than "it was a bloom last time", though not well enough to call the *start* of a bloom.
2. **Does Optuna tuning get faster with threads when the GIL is off?** Yes: 2.7× with 8 threads.
   With the GIL on, the same threads make it *slower* than one thread.

## The prediction

A water sample counts as a **bloom** when the alga *Karenia brevis* is at or above
100,000 cells/L. For a sample on day *t* in a 0.25° cell of the southwest Florida shelf (Tampa Bay
to Naples), the model sees only data up to **t − 7 days**:

- 30-day air temperature mean and trend (no sea temperature exists in the free archive before
  2023; coastal air temperature stands in for it);
- 30-day wind speed and the share of days with winds from the NNW;
- 30-day rainfall;
- the season, as the day of the year;
- the latest *K. brevis* count within 20 km, at least 7 days old.

Split by time, never at random: train 2000–2015 (18,720 samples), validate 2016–2017, test
2018 to March 2024 (9,489 samples, 1,305 blooms). Two baselines and two scikit-learn models,
each tuned with 100 Optuna trials on the validation years, then refit on 2000–2017:

| Model | Test PR-AUC | Recall at 50 % precision | Onset PR-AUC |
|---|---|---|---|
| persistence: "a bloom if the last nearby count was one" | 0.420 | 0.688 | 0.052 |
| season: bloom rate of the month in the training years | 0.232 | 0.000 | 0.088 |
| logistic regression | **0.558** | **0.754** | **0.173** |
| gradient boosting (`HistGradientBoostingClassifier`) | 0.553 | 0.734 | 0.160 |

PR-AUC is the main score because blooms are rare; guessing scores the bloom rate (0.138 on the
test years). Accuracy is not reported: "never a bloom" would be 86 % accurate.

What it means:

- **Both models beat persistence clearly** (0.55 against 0.42). Blooms last for weeks, so the
  last count carries most of the signal; weather and season add a real amount on top.
- **The simple model is as good as the boosted one.** The logistic regression even edges ahead,
  and its tuning settled on strong regularization: with these features there is little
  non-linear structure for boosting to find.
- **The start of a bloom stays hard.** On "onset" samples (no bloom nearby a week before; 407
  blooms among 7,850), the models score about 3× the bloom rate, but at 50 % precision they catch
  none. They can rank a risky week higher; they cannot raise a reliable alarm.

Sea temperature, salinity, river discharge and satellite chlorophyll would be the next features
to try; the free NOAA servers for them timed out from the dev machine when checked (2026-10-28).

## The threads

The same Optuna study (64 trials of the gradient boosting model, fixed seed) on the same 3.15t
interpreter, once with the GIL off (default) and once forced on with `-X gil=1`. scikit-learn's own
OpenMP/BLAS threads are pinned to 1, so only Optuna's threads count. Each run is a fresh process.

![Speedup against threads, GIL off and on](results/bench.png)

| Threads | GIL off: seconds | speedup | GIL on: seconds | speedup |
|---|---|---|---|---|
| 1 | 118.9 | 1.00 | 121.4 | 1.00 |
| 2 | 86.4 | 1.38 | 208.4 | 0.58 |
| 4 | 52.7 | 2.26 | 186.5 | 0.65 |
| 8 (SMT) | 44.5 | 2.68 | 213.8 | 0.57 |

Machine: AMD Ryzen 5 2500U, 4 cores / 8 threads, a desktop session with a browser open (load
about 1.7 before the run), so the numbers are indicative. The 8-thread point uses SMT, where the
curve flattens as expected. With the GIL on, the threads take turns holding it, and the switching
costs more than the work they could share, so 2 to 8 threads end up slower than 1 (likely
explanation, not profiled).

## Lazy imports (PEP 810)

`cli.py` imports pandas, scikit-learn, Optuna and its own modules with `lazy import`, so
`bloom --help` does not load them:

| `bloom --help` | median of 15 runs |
|---|---|
| lazy imports (as shipped) | 0.078 s |
| everything eager | 1.891 s |

This 3.15 build accepts only `-X lazy_imports=all|normal`, with no `none`, so the eager run turns
the lazy imports off with `sys.set_lazy_imports_filter(lambda *a: False)`.

## Run it

```sh
cd red-tide
uv run bloom fetch      # HABSOS + Open-Meteo into data/ (about an hour: the free API is paced)
uv run bloom train      # features, tuning, evaluation on the test years -> results/metrics.csv
uv run bloom bench      # the threading grid and startup times -> results/bench.csv, bench.png
uv run bloom report     # prints the tables above
uv run pytest           # offline; tests never touch the network
```

Weather comes on a 0.5° grid (20 cells): 25 years of daily data per cell counts as hundreds of
calls against Open-Meteo's free limits, so `fetch` waits between requests and when rate-limited.

## Data and licences

- *K. brevis* cell counts: **HABSOS**, NOAA NCEI accession 0120767, version 8.8 (1953 to
  March 2024; 104,932 samples in the region since 2000). Public domain.
- Weather: **Open-Meteo** historical archive (ERA5 reanalysis), CC BY 4.0. Weather data by
  Open-Meteo.com; contains modified Copernicus Climate Change Service information.
