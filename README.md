# Data-Driven Incentive Design

### Optimizing incentives for taxi drivers using causal estimates of supply elasticity

#### Julian Streyczek, June 2026

This GitHub repo is intended to show how user data, causal inference, and optimization can be combined to design an incentive scheme. The example uses public taxi trip data from San Francisco and designs a simple tiered rewards scheme to increase the number of trips completed by drivers.

The [Jupyter notebook](code/taxi_incentive_design.ipynb) proceeds as follows. First, it loads and cleans the data. Second, it estimates individual labor supply, i.e. how drivers adjust the number of trips they provide when revenue per trip changes, using a leave-one-out instrument for causal identification. Third, it explains how to use this estimate in a tiered incentive scheme. Fourth, for a given reward structure, it uses a back-of-the-envelope simulation to estimate how many additional trips would be completed, along with the expected cost. Finally, it applies a simple optimization over potential reward levels to find the best design under a predefined budget.

The goal is not to claim that the result is the final optimal policy, but to demonstrate a practical data science workflow for solving a real-world business objective.

## Contents

1. Setting and Data
2. Estimating Drivers' Supply Elasticity
3. Tiered Incentive Scheme
4. Optimization

Main code: [notebook](code/taxi_incentive_design.ipynb) or [Python script](code/taxi_incentive_design.py).

## Data

Source: DataSF Taxi Trips, https://data.sfgov.org/Transportation/Taxi-Trips/m8hk-2ipk

The local run expects monthly CSV files in `data/raw/`. Raw data is intentionally ignored by Git.

## Reproduce

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Add the raw data

Download monthly CSV files from the DataSF Taxi Trips dataset and place them in `data/raw/`. The checked-in outputs were generated from December 2022 through May 2024, with files named like:

```text
data/raw/sf_taxi_trips_2022_12.csv
...
data/raw/sf_taxi_trips_2024_05.csv
```

Raw data is not committed to this repo.

### 3. Run the code

Run [code/taxi_incentive_design.ipynb](code/taxi_incentive_design.ipynb) for the notebook or [code/taxi_incentive_design.py](code/taxi_incentive_design.py) for the raw Python script. The IV calculation is a bit costly due to the high-dimensional fixed effects, and can take 1-3 minutes.

## File Structure

```text
.
├── code/        # main analysis notebook and Python script
├── outputs/     # generated figures and summary CSVs
├── data/        # data (raw DataSF CSVs are ignored by Git)
├── docs/        # methodology notes
├── README.md
├── AGENTS.md
└── requirements.txt
```

## Outputs

Figures and summary CSVs are written to `outputs/`:

- `01_driver_distributions.svg`
- `02_active_days_distribution.svg`
- `03_binscatter_functional_form.svg`
- `04_optimized_trip_tier_simulation.svg`
