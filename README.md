# GridGuard

AI-based anomaly detection for smart meter data, with physics-informed checks. A portfolio project at the intersection of renewable energy, machine learning and cybersecurity.

**Status:** work in progress. Day 1 of 5 complete (data pipeline and exploration).

## Why this matters

Modern grids rely on smart meters and sensors that report over networks. Tampered or faulty readings (energy theft, false data injection, sensor failures) can mislead operators and cause financial loss or instability. GridGuard explores how to detect such readings automatically, and whether physics-based rules can reduce false alarms compared with a purely statistical model.

## Data

London Smart Meters dataset (half-hourly readings, 5,560 households), from Zenodo: https://zenodo.org/records/3898331 (originally from Kaggle).

Sample used: 300 households with complete readings for all of 2013 (17,520 readings each), chosen with a fixed random seed (42).

## Plan

- [x] Day 1: environment, data pipeline, exploratory analysis
- [ ] Day 2: anomaly simulator, baseline detectors (rolling z-score, Isolation Forest), evaluation harness
- [ ] Day 3: LSTM autoencoder detector
- [ ] Day 4: physics-informed rules, hybrid detector, ablation study, explainability
- [ ] Day 5: Streamlit dashboard, write-up, demo

## Project structure

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