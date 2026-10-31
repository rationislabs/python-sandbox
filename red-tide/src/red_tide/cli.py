"""`bloom`: fetch the data, train and evaluate, benchmark, report."""

import argparse
from datetime import date
from pathlib import Path

# PEP 810: the heavy modules load on first use, so `bloom --help` starts fast.
lazy import pandas as pd

lazy from red_tide import features, habsos, models, weather

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
# Weather starts before 2000 so the first samples get a full window.
WEATHER_START = date(1999, 11, 1)


def fetch() -> None:
    raw = DATA / "habsos.csv"
    if not raw.exists():
        print(f"downloading HABSOS to {raw}")
        habsos.download(raw)
    samples = habsos.load(raw)
    cells = sorted(
        {
            weather.cell_of(a, b, weather.WX_DEG)
            for a, b in zip(samples.lat, samples.lon)
        }
    )
    print(f"{len(samples)} samples in {len(cells)} cells; fetching weather")
    end = samples["date"].max().date()
    weather.load_all(cells, WEATHER_START, end, DATA / "weather")
    print("done")


def build_features() -> pd.DataFrame:
    path = DATA / "features.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    samples = habsos.load(DATA / "habsos.csv")
    cells = sorted(
        {
            weather.cell_of(a, b, weather.WX_DEG)
            for a, b in zip(samples.lat, samples.lon)
        }
    )
    end = samples["date"].max().date()
    wx = weather.load_all(cells, WEATHER_START, end, DATA / "weather")
    df = features.build(samples, wx)
    df.to_csv(path, index=False)
    return df


def train(n_trials: int, n_jobs: int) -> None:
    parts = features.split(build_features())
    train_df, val, test = parts["train"], parts["val"], parts["test"]
    known = pd.concat([train_df, val], ignore_index=True)
    onset = (
        test["last_bloom"] != 1
    )  # no bloom nearby a week before: persistence says no

    scores = {
        "persistence": models.persistence(test),
        "season": models.season_rate(known, test),
    }
    for kind in ("logreg", "hgb"):
        study = models.tune(kind, train_df, val, n_trials, n_jobs, seed=0)
        print(
            f"{kind}: best validation PR-AUC {study.best_value:.3f} {study.best_params}"
        )
        model = models.make_model(kind, study.best_params)
        model.fit(known[features.FEATURES], known["y"])
        scores[kind] = model.predict_proba(test[features.FEATURES])[:, 1]

    rows = []
    for name, score in scores.items():
        for subset, mask in (("test", slice(None)), ("onset", onset.to_numpy())):
            metrics = models.evaluate(test["y"].to_numpy()[mask], score[mask])
            rows.append({"model": name, "subset": subset, **metrics})
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "metrics.csv", index=False)
    print(markdown(table))


def bench(n_trials: int) -> None:
    from red_tide import bench as harness

    features_csv = DATA / "features.csv"
    if not features_csv.exists():
        raise SystemExit("no data/features.csv yet: run `bloom train` first")
    harness.run_grid(features_csv, n_trials=n_trials, out_dir=RESULTS)
    print(harness.startup_times(out_dir=RESULTS))


def markdown(df: pd.DataFrame) -> str:
    def cell(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def report() -> None:
    shown = False
    for name, title in (
        ("metrics.csv", "Bloom prediction, test years (2018 on)"),
        ("bench.csv", "Optuna threads, GIL off vs on"),
        ("startup.csv", "`bloom --help` startup, seconds"),
    ):
        path = RESULTS / name
        if path.exists():
            print(f"## {title}\n\n{markdown(pd.read_csv(path))}\n")
            shown = True
    if not shown:
        print(
            "No results yet: run `bloom fetch`, then `bloom train` and `bloom bench`."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bloom", description="Predict Florida red tide blooms a week ahead."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download HABSOS and Open-Meteo data into data/")
    p_train = sub.add_parser("train", help="build features, tune, evaluate on 2018+")
    p_train.add_argument(
        "--trials", type=int, default=100, help="Optuna trials per model"
    )
    p_train.add_argument("--jobs", type=int, default=4, help="Optuna threads")
    p_bench = sub.add_parser("bench", help="Optuna threads with the GIL off and on")
    p_bench.add_argument("--trials", type=int, default=64)
    sub.add_parser("report", help="print the results tables")
    args = parser.parse_args(argv)
    match args.command:
        case "fetch":
            fetch()
        case "train":
            train(args.trials, args.jobs)
        case "bench":
            bench(args.trials)
        case "report":
            report()


if __name__ == "__main__":
    main()
