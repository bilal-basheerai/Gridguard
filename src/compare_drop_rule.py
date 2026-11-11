"""Does the sustained-drop rule help? Compare GridGuard with and without it.

Uses the London sample (first METERS households) and GridGuard's self-benchmark at
several false-alarm budgets. Saves results/drop_rule_comparison.csv.

Run from the project root (takes several minutes):
    python -m src.compare_drop_rule
"""

import pandas as pd

from src.data import load_sample
from src.engine import prepare, self_benchmark
from src.run_baselines import DATA_PATH, OUT_DIR

METERS = 40
BUDGETS = [0.25, 0.5, 1.0]
TYPES = ["drift", "scale_down", "spike", "zero_out", "flatline"]


def main():
    df = prepare(load_sample(DATA_PATH)).iloc[:, :METERS]
    rows = []
    for budget in BUDGETS:
        for use_drop_rule in (False, True):
            print(f"budget {budget}, drop rule {use_drop_rule} ...", flush=True)
            res = self_benchmark(df, model_budget=budget, use_drop_rule=use_drop_rule)
            recall = res["by_type"]["event_recall"]
            row = {
                "budget": budget,
                "drop_rule": use_drop_rule,
                "event_recall": res["event_recall"],
                "false_alarms_clean": res["false_alarms_per_meter_week_on_clean"],
            }
            row.update({t: float(recall.get(t, float("nan"))) for t in TYPES})
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "drop_rule_comparison.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'drop_rule_comparison.csv'}")


if __name__ == "__main__":
    main()