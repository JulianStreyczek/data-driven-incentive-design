# Agent Notes

Keep this project notebook-first. The main artifact is `notebooks/taxi_supply_elasticity_public_data.ipynb`; top-level docs should stay short and GitHub-readable.

## Current Design

- Current analysis is driver-day IV estimation plus driver-week tier simulation.
- Do not revive the older spatial hourly plan unless explicitly requested.
- Use DataSF Taxi Trips as the public data.
- Generated SVGs and summary CSVs belong in `outputs/`.

## Empirical Choices

- Clean source trips before aggregation. Treat missing driver IDs and the literal placeholder `driver_id == "-"` as invalid.
- Estimate `log(daily_trips)` on `log(revenue_per_trip)` with driver, weekday, and year-month fixed effects.
- Instrument own revenue per trip with same-day equal-weight leave-one-out revenue per trip among other active drivers.
- Cluster standard errors by date because the instrument varies at the day level.

## Incentive Simulation

- The tier table can report total bonus and total earnings increase at each threshold.
- Response windows should use the marginal reward at the threshold, not the cumulative total reward.
- The simulation reports net program cost as total bonus paid minus incremental company revenue from additional trips.

## Working Notes

- Raw taxi CSVs stay local under `data/raw/` and ignored by Git.
- Avoid stale output paths, old trip-by-hour SVG names, monthly tier settings, or older simulation numbers.
- If notebook logic changes, regenerate the notebook outputs and `outputs/*.csv` together.
