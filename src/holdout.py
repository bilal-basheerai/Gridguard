"""Build a hold-out sample of London households from the raw Zenodo file.

The raw file (inside data/london_raw.zip) is in the Monash .ts format: a header, then one
line per household: name:start_timestamp:comma-separated half-hourly readings.

Two commands (run from the project root):

    python -m src.holdout scan
        Reads the file once and prints how many households have a complete 365-day
        window starting each month, so you can choose a hold-out period.

    python -m src.holdout build --start 2012-01-01 --n 300 --seed 2012
        Builds a (time x household) table of households with complete readings for the
        whole window, excluding every household used in the 2013 sample, and saves it to
        data/meters_holdout.parquet.

Households are read one line at a time, so the 1 GB file is never fully loaded into memory.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_PATH = Path("data/london_raw.zip")
SAMPLE_2013 = Path("data/meters_2013.parquet")
OUT_PATH = Path("data/meters_holdout.parquet")
HALF_HOUR = pd.Timedelta("30min")


def iter_series(zip_path=ZIP_PATH):
    """Yield (name, start Timestamp, values string) for each household in the .ts file."""
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(".ts") or n.endswith(".tsf"))
        with z.open(member) as raw:
            lines = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=None)
            in_data = False
            for line in lines:
                if not in_data:
                    in_data = line.strip() == "@data"
                    continue
                line = line.strip()
                if not line:
                    continue
                name, ts, vals = line.split(":", 2)
                start = pd.to_datetime(ts, format="%Y-%m-%d %H-%M-%S").floor("30min")
                yield name, start, vals


def excluded_names(path=SAMPLE_2013):
    """Households used in the 2013 sample (they must not appear in the hold-out)."""
    return set(pd.read_parquet(path).columns) if Path(path).exists() else set()


def scan(zip_path=ZIP_PATH):
    rows = []
    for name, start, vals in iter_series(zip_path):
        n = vals.count(",") + 1
        rows.append((name, start, n, start + n * HALF_HOUR))
    idx = pd.DataFrame(rows, columns=["name", "start", "n_values", "end"])
    skip = excluded_names()
    idx["in_2013_sample"] = idx["name"].isin(skip)
    print(f"Households in file: {len(idx)} (of which {int(idx.in_2013_sample.sum())} are in the 2013 sample)")
    print(f"Data runs from {idx.start.min()} to {idx.end.max()}\n")

    steps = 365 * 48
    out = []
    for m in pd.date_range("2011-11-01", "2013-02-01", freq="MS"):
        end = m + steps * HALF_HOUR
        ok = (idx.start <= m) & (idx.end >= end)
        out.append({"window start": m.date(), "window end": end.date(),
                    "complete households": int(ok.sum()),
                    "excluding 2013 sample": int((ok & ~idx.in_2013_sample).sum())})
    table = pd.DataFrame(out)
    print("Households with complete readings for a 365-day window starting on each date:")
    print(table.to_string(index=False))

    cal = pd.Timestamp("2012-01-01")
    ok = (idx.start <= cal) & (idx.end >= cal + 366 * 48 * HALF_HOUR)
    print(f"\nCalendar year 2012: {int(ok.sum())} complete, "
          f"{int((ok & ~idx.in_2013_sample).sum())} not in the 2013 sample")


def build(start, n_meters, seed, zip_path=ZIP_PATH, out_path=OUT_PATH, days=365):
    start = pd.Timestamp(start)
    steps = days * 48
    index = pd.date_range(start, periods=steps, freq="30min", name="timestamp")
    skip = excluded_names()

    eligible = {}
    for name, first, vals in iter_series(zip_path):
        if name in skip:
            continue
        offset = int((start - first) / HALF_HOUR)
        if offset < 0:
            continue                         # starts after the window begins
        values = vals.split(",")
        if len(values) < offset + steps:
            continue                         # ends before the window ends
        eligible[name] = np.array(values[offset:offset + steps], dtype="float32")

    if len(eligible) < 30:
        raise SystemExit(f"Only {len(eligible)} eligible households for a window starting {start.date()}; "
                         "pick another start date (see `scan`).")
    names = sorted(eligible)
    order = np.random.default_rng(seed).permutation(len(names))
    chosen = [names[i] for i in order[:n_meters]]
    df = pd.DataFrame({n: eligible[n] for n in chosen}, index=index)
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)
    df.to_parquet(out_path)
    print(f"Eligible households: {len(eligible)}; saved {df.shape[1]} meters x {df.shape[0]} readings "
          f"({index[0]} to {index[-1]}) to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    b = sub.add_parser("build")
    b.add_argument("--start", default="2012-01-01")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--seed", type=int, default=2012)
    b.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    if args.cmd == "scan":
        scan()
    else:
        build(args.start, args.n, args.seed, days=args.days)


if __name__ == "__main__":
    main()