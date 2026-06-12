# Data-Driven Incentive Design

### Optimizing incentives for taxi drivers using causal estimates of supply elasticity

#### Julian Streyczek, June 2026

This case study is intended to show how user data, causal inference, and optimization can be combined to design a user incentive scheme. I use public taxi trip data from San Francisco and design a simple tiered rewards scheme to increase the number of trips completed by drivers.

The [Jupyter notebook](code/taxi_incentive_design.ipynb) contains both the code and explanations. First, we load, clean, and show descriptives for the data. Second, we estimate individual labor supply, i.e. how drivers adjust the number of trips they provide when revenue per trip changes, using a leave-one-out instrument for causal identification. Third, we introduce a simple incentive scheme that provides cash rewards upon completing a certain number of trips, and use our labor supply estimates in a back-of-envelope simulation to estimate the effect on additional trips, along with the expected cost. Finally, we run an optimization loop over potential reward tiers and levels to find the most effective design under a pre-defined budget constraint.

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
