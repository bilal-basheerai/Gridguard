# GridGuard [Day 1]

AI-based anomaly detection for smart meter data, with physics-informed checks. A portfolio project at the intersection of renewable energy, machine learning and cybersecurity.

**Status:** work in progress. Day 3 of 5 complete (LSTM autoencoder and data diagnostics).
## Why this matters

Modern grids rely on smart meters and sensors that report over networks. Tampered or faulty readings (energy theft, false data injection, sensor failures) can mislead operators and cause financial loss or instability. GridGuard explores how to detect such readings automatically, and whether physics-based rules can reduce false alarms compared with a purely statistical model.

## Data

London Smart Meters dataset (half-hourly readings, 5,560 households), from Zenodo: https://zenodo.org/records/3898331 (originally from Kaggle).

Sample used: 300 households with complete readings for all of 2013 (17,520 readings each), chosen with a fixed random seed (42).

## Plan

- [x] Day 1: environment, data pipeline, exploratory analysis
- [x] Day 2: anomaly simulator, baseline detectors (rolling z-score, Isolation Forest), evaluation harness
- [x] Day 3: LSTM autoencoder detector and data diagnostics
- [ ] Day 4: physics-informed rules, hybrid detector, ablation study, explainability
- [ ] Day 5: Streamlit dashboard, write-up, demo

## Results so far: baseline detectors (Day 2)

Setup: detectors are fitted on 210 households and tested on 90 unseen households. Normal behaviour is learned from January to September; 540 synthetic anomalies (6 per meter) are injected only in October to December. Each detector's alarm threshold is the 99th percentile of its scores on clean training data, so every detector has the same false-alarm budget and no labels are used for tuning.

| Detector | Event recall | Median delay (h) | False alarms per meter-week | PR-AUC |
|---|---|---|---|---|
| Rolling z-score | 31.5% | 0.5 | 1.81 | 0.191 |
| Isolation Forest | 21.3% | 0.0 | 1.34 | 0.222 |

Event recall by anomaly type:

| Anomaly type | Events | Rolling z-score | Isolation Forest |
|---|---|---|---|
| Drift | 55 | 67.3% | 40.0% |
| Spike | 137 | 62.8% | 49.6% |
| Scale-down | 93 | 48.4% | 15.1% |
| Flat-line | 129 | 1.6% | 8.5% |
| Zero-out | 126 | 0.0% | 0.0% |

Finding: both baselines catch gradual and extreme changes but miss zero-outs and flat-lines almost entirely, even though these events change the readings substantially (see the data diagnostics below). Both detectors score readings one at a time, so they likely miss the persistence of a pattern (hours of zeros or repeated values).

![Example detections](results/example_detections.png)
## Day 3: LSTM autoencoder and data diagnostics

An LSTM autoencoder (one day of readings in, one day reconstructed) was trained on clean history and evaluated on the same unseen meters and injected anomalies as the baselines. Its threshold comes from held-out clean meters.

| Detector | Event recall | Median delay (h) | False alarms per meter-week | PR-AUC |
|---|---|---|---|---|
| Rolling z-score | 31.5% | 0.5 | 1.81 | 0.191 |
| Isolation Forest | 21.3% | 0.0 | 1.34 | 0.222 |
| LSTM autoencoder | 11.3% | 22.5 | 1.15 | 0.146 |

Event recall by anomaly type:

| Anomaly type | Rolling z-score | Isolation Forest | LSTM autoencoder |
|---|---|---|---|
| Drift | 67.3% | 40.0% | 40.0% |
| Spike | 62.8% | 49.6% | 5.1% |
| Scale-down | 48.4% | 15.1% | 18.3% |
| Flat-line | 1.6% | 8.5% | 11.6% |
| Zero-out | 0.0% | 0.0% | 0.0% |

Checks on the benchmark itself:
- 11 automated tests pass (labels match injected changes, no look-ahead, perfect detector scores 100%, random detector scores at chance).
- Injected anomalies are visible: 98.3% change the readings by at least 10% of the household's usual usage.
- In the clean data, zero readings lasting 6 hours or more occur in about 5% of households (447 episodes; the longest lasted about 126 days). Identical non-zero readings lasting 12 hours or more occurred twice in 300 household-years.

Finding: the first LSTM version did not beat the simple baselines, and its false-alarm rate (about 1 per meter per week) is too high for practical use for every detector. Zero-outs and flat-lines are visible and essentially never occur naturally, yet all three detectors miss them, which motivates simple rule-based checks (Day 4).
```
src/        data loading and feature code
notebooks/  exploration (01_eda.ipynb)
app/        dashboard (coming)
tests/
```

## Setup

```
git clone https://github.com/bilal-basheerai/Gridguard.git
cd Gridguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from Zenodo, then run `notebooks/01_eda.ipynb` (written for Google Colab) to build the 300-household sample and save it as `data/meters_2013.parquet`.

## Limitations

- Anomalies are simulated, not real attacks.
- Residential data from one city and one year.