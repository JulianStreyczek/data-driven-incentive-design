# Data-Driven Incentive Design

Case study on using causal estimates of labor supply to design a simple incentive scheme.

The analysis uses San Francisco taxi trip records to build driver-day and driver-week panels, estimate how completed trips respond to revenue per trip, and simulate a two-tier weekly trip bonus.

## Main Deliverable

- `notebooks/taxi_supply_elasticity_public_data.ipynb`

The notebook is the source of truth for data cleaning, estimation, figures, and simulation outputs.

## Data

Source: DataSF Taxi Trips

- Dataset page: https://data.sfgov.org/Transportation/Taxi-Trips/m8hk-2ipk
- API endpoint: https://data.sfgov.org/resource/m8hk-2ipk.csv

The local run expects monthly CSV files in `data/raw/`. Raw data is intentionally ignored by Git.

## Reproduce

Using the vendored local dependencies in `.codex_deps`:

```bash
PYTHONPATH=.codex_deps MPLCONFIGDIR=/tmp MPLBACKEND=Agg python3 - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebooks/taxi_supply_elasticity_public_data.ipynb").read_text())
ns = {"__name__": "__notebook_exec__"}
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), "notebook", "exec"), ns)
PY
```

## Outputs

Generated artifacts are written to `outputs/`:

- `01_driver_distributions.svg`
- `02_active_days_distribution.svg`
- `03_binscatter_functional_form.svg`
- `04_trip_tier_simulation.svg`
- `driver_day_model_summary.csv`
- `incentive_simulation_summary.csv`


