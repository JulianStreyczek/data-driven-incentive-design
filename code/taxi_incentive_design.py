# Generated from notebooks/taxi_supply_elasticity_public_data.ipynb.
# The notebook remains the main project artifact; this script is a plain-Python runner.

# %% Notebook code cell 1
import os
from pathlib import Path

import re
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

cwd = Path.cwd()
PROJECT_ROOT = cwd if (cwd / 'data' / 'raw').exists() else cwd.parent
RAW = PROJECT_ROOT / 'data' / 'raw'
OUTPUTS = PROJECT_ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)


pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 140)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

#print(f'Project root: {PROJECT_ROOT}')

# %% Notebook code cell 2
# Define columns to be loaded
cols = [
    'driver_id',
    'start_time_local', 'end_time_local',
    'pickup_location_latitude', 'pickup_location_longitude',
    'fare_time_milliseconds', 'trip_distance_meters',
    'total_fare_amount'
]


# Load each month and concatenate into a pandas DataFrame
files_raw = sorted(f for f in os.listdir(RAW) if re.match('sf_taxi_trips_.+\\.csv', f))
parts = []
for fn in files_raw:
    df = pd.read_csv(RAW / fn, usecols=lambda c: c in cols, low_memory=False)
    df['source_file'] = fn
    parts.append(df)
src = pd.concat(parts, ignore_index=True);
del cols, files_raw, parts, fn


# Fix data types and create features
src['start']            = pd.to_datetime(src['start_time_local'], errors='coerce')
src['end']              = pd.to_datetime(src['end_time_local'], errors='coerce')
src['date']             = src['start'].dt.normalize()
src['pickup_hour']      = src['start'].dt.hour

src['trip_hours']       = (src['end'] - src['start']).dt.total_seconds() / 3600
fallback_hours          = pd.to_numeric(src['fare_time_milliseconds'], errors='coerce') / 3_600_000
src['trip_hours']       = src['trip_hours'].where(src['trip_hours'].between(1 / 60, 3), fallback_hours)

src['distance_km']      = pd.to_numeric(src['trip_distance_meters'], errors='coerce') / 1000
src['lat']              = pd.to_numeric(src['pickup_location_latitude'], errors='coerce')
src['lon']              = pd.to_numeric(src['pickup_location_longitude'], errors='coerce')
src['speed_kmh']        = src['distance_km'] / src['trip_hours']

src['fare']             = pd.to_numeric(src['total_fare_amount'], errors='coerce')
src['revenue_per_hour'] = src['fare'] / src['trip_hours']
src['revenue_per_km']   = src['fare'] / src['distance_km']


# ── Trip-level cleaning ───────────────────────────────────────────────────
# Layer 1: validity filters removing missing or physically impossible records
valid_driver_id = src['driver_id'].notna() & src['driver_id'].astype(str).str.strip().ne('-')
valid = (
    valid_driver_id
    & src['start'].notna()
    & (src['fare'] > 0)
    & (src['trip_hours'] >= 1/60) & (src['trip_hours'] <= 3)
    & (src['distance_km'] >= 0)
    & (src['speed_kmh'] <= 120)
)

# Layer 2: symmetric quantile trim of the analysis variables among valid trips
TRIM_Q = 0.001
trip_trim_cols = ['fare', 'trip_hours', 'distance_km']
trip_trim_bounds = src.loc[valid, trip_trim_cols].quantile([TRIM_Q, 1 - TRIM_Q])
keep = valid.copy()
for col in trip_trim_cols:
    keep &= src[col].between(trip_trim_bounds.loc[TRIM_Q, col], trip_trim_bounds.loc[1 - TRIM_Q, col])

basic = src[keep].copy()
basic['driver_trip_count'] = basic.groupby('driver_id').transform('size')

# %% Notebook code cell 3
# ── Driver-day panel (for IV estimation) ─────────────────────────────────
driver_day = (
    basic.groupby(['driver_id', 'date'])
    .agg(
        daily_trips   =('fare',      'count'),
        daily_hours   =('trip_hours','sum'),
        daily_earnings=('fare',      'sum'),
    )
    .reset_index()
)
driver_day['revenue_per_trip']  = driver_day['daily_earnings'] / driver_day['daily_trips']
driver_day['weekday'] = driver_day['date'].dt.dayofweek
driver_day['ym']      = driver_day['date'].dt.to_period('M').astype(str)
# Exclude major US holidays: federal holidays plus Christmas Eve and New Year's Eve.
holiday_dates = (
    USFederalHolidayCalendar()
    .holidays(start=driver_day['date'].min(), end=driver_day['date'].max())
    .union(pd.to_datetime([
        f'{year}-12-{day}'
        for year in range(driver_day['date'].dt.year.min(), driver_day['date'].dt.year.max() + 1)
        for day in (24, 31)
    ]))
)

# Active days only: At least 3 trips, maximum 20 hours
keep_day = (driver_day['daily_trips'] >= 3) & (driver_day['daily_hours'] <= 20)
on_holiday = driver_day['date'].isin(holiday_dates)
driver_day = driver_day[keep_day & ~on_holiday].reset_index(drop=True)

driver_day['log_trips'] = np.log(driver_day['daily_trips'])
driver_day['log_rpt']   = np.log(driver_day['revenue_per_trip'])


# Week fields for driver-week aggregation
basic['year_week']  = basic['start'].dt.strftime('%G-W%V')
basic['week_start'] = pd.to_datetime(basic['year_week'] + '-1', format='%G-W%V-%u')


# ── Driver-week panel (for tier calibration and simulation) ───────────────
dw_raw = (
    basic.groupby(['driver_id', 'year_week'])
    .agg(
        weekly_trips   =('fare',      'count'),
        weekly_hours   =('trip_hours','sum'),
        weekly_earnings=('fare',      'sum'),
        active_days    =('date',      'nunique'),
    )
    .reset_index()
)
week_to_ym = (
    basic.groupby('year_week')['week_start'].first()
    .dt.to_period('M').astype(str)
    .rename('ym')
)
dw_raw = dw_raw.join(week_to_ym, on='year_week'); del week_to_ym
dw_raw['revenue_per_trip']  = dw_raw['weekly_earnings'] / dw_raw['weekly_trips']
dw_raw['earnings_per_hour'] = dw_raw['weekly_earnings'] / dw_raw['weekly_hours']
dw_raw['trips_per_hour']    = dw_raw['weekly_trips'] / dw_raw['weekly_hours']

# Active weeks only, then trim above the 99.9th percentile of the analysis
# variables; trips per hour catches weeks whose trips and hours are
# individually plausible but mutually inconsistent.
dw_active = dw_raw[dw_raw['weekly_trips'] >= 3]
week_trim_cols = ['weekly_trips', 'weekly_hours', 'weekly_earnings', 'trips_per_hour']
week_trim_caps = dw_active[week_trim_cols].quantile(1 - TRIM_Q)
driver_week = dw_active[(dw_active[week_trim_cols] <= week_trim_caps).all(axis=1)].reset_index(drop=True)

# %% Notebook code cell 4
# Print descriptive statistics
print(f"Number of raw trips:        {len(src):,}")
print(f"Number of clean trips:      {len(basic):,}")
print(f"Period:                     {min(basic['date']):%Y-%m-%d} - {max(basic['date']):%Y-%m-%d}")
print(f"Drivers:                    {basic['driver_id'].nunique():,}")
print(f"Days:                       {(max(basic['date']) - min(basic['date'])).days + 1}")
print(f"Driver-day observations:    {len(driver_day):,}")
print(f"Driver-week observations:   {len(driver_week):,}")

# %% Notebook code cell 5
summary_cols = [
    'weekly_trips', 'weekly_hours', 'weekly_earnings',
    'active_days',
     'revenue_per_trip', 'earnings_per_hour'
]
print(driver_week[summary_cols].describe().round(2).to_string())

# %% Notebook code cell 6
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

vars_cfg = [
    ('weekly_hours',  'Total active hours',  'Hours per driver-week'),
    ('weekly_trips',  'Completed trips',      'Trips per driver-week'),
    ('revenue_per_trip', 'Revenue per trip',  '$ per trip'),
]

for ax, (col, title, xlabel) in zip(axes, vars_cfg):
    p60 = driver_week[col].quantile(0.60)
    p80 = driver_week[col].quantile(0.80)
    plot_cap = driver_week[col].quantile(0.99)
    plot_values = driver_week[col].clip(upper=plot_cap)
    sns.histplot(plot_values, bins=35, color='steelblue', edgecolor='white', linewidth=0.3, ax=ax)
    ax.axvline(p60, color='dimgray', lw=1.6, linestyle='--', label=f'P60 = {p60:.1f}')
    ax.axvline(p80, color='black', lw=1.6, linestyle=':', label=f'P80 = {p80:.1f}')
    ax.set(title=title, xlabel=f'{xlabel} (top 1% clipped)', ylabel='Driver-weeks')
    ax.legend(fontsize=8)

fig.suptitle('Figure 1: Driver-week distributions', y=1.03)
fig.tight_layout()
fig.savefig(OUTPUTS / '01_driver_distributions.svg', bbox_inches='tight')
plt.close(fig)
print('Figure saved: outputs/01_driver_distributions.svg')

# %% Notebook code cell 7
fig, ax = plt.subplots(figsize=(4, 3))
active_day_order = np.arange(1, int(driver_week['active_days'].max()) + 1)
sns.countplot(data=driver_week, x='active_days', order=active_day_order,
              color='steelblue', linewidth=0, ax=ax)
ax.set(
    title='Figure 2: Active days per driver-week',
    xlabel='Number of active days in week',
    ylabel='Number of driver-weeks',
)
fig.tight_layout()
fig.savefig(OUTPUTS / '02_active_days_distribution.svg', bbox_inches='tight')
plt.close(fig)
print('Figure saved: outputs/02_active_days_distribution.svg')

# %% Notebook code cell 8
# Calculate equal-weight LOO-IV: average revenue per trip among other active drivers, then log.
day_sum = driver_day.groupby('date')['revenue_per_trip'].transform('sum')
day_cnt = driver_day.groupby('date')['log_rpt'].transform('count')
loo_eq_rpt = (day_sum - driver_day['revenue_per_trip']) / (day_cnt - 1)
driver_day['log_loo_eq_rpt'] = np.log(loo_eq_rpt)

dd = (
    driver_day[day_cnt >= 2]
    .dropna(subset=['log_trips', 'log_rpt', 'log_loo_eq_rpt'])
    .pipe(lambda df: df[np.isfinite(df['log_loo_eq_rpt'])])
    .copy()
    .reset_index(drop=True)
)

# Drop the most atypical market days: residualize the day-level mean of log
# revenue per trip on the model's calendar fixed effects and trim the extreme
# 1% tails. City-wide shocks of this size (convention arrival weekends,
# holiday-adjacent travel lulls) are directly observable to drivers, which is
# when the exclusion restriction is least credible. The rule conditions only
# on instrument-side market conditions, so it trims the support of the
# instrument rather than selecting on outcomes.
DAY_TRIM_Q = 0.01
day_conditions = (
    dd.groupby('date')
    .agg(day_log_rpt=('log_rpt', 'mean'), weekday=('weekday', 'first'), ym=('ym', 'first'))
    .reset_index()
)
day_conditions['resid'] = smf.ols('day_log_rpt ~ C(weekday) + C(ym)', data=day_conditions).fit().resid
typical_days = day_conditions.loc[
    day_conditions['resid'].between(
        day_conditions['resid'].quantile(DAY_TRIM_Q),
        day_conditions['resid'].quantile(1 - DAY_TRIM_Q),
    ),
    'date',
]
dd = dd[dd['date'].isin(typical_days)].reset_index(drop=True)


# Within-transform: absorb driver FEs via demeaning (avoids ~1,338-column design matrix)
for col in ['log_trips', 'log_rpt', 'log_loo_eq_rpt']:
    dd[f'{col}_w'] = dd[col] - dd.groupby('driver_id')[col].transform('mean')

clust = dd['date']

# ── helpers ───────────────────────────────────────────────────────────────────
def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

def fmt(coef, se, pval):
    return f'{coef:.3f}{stars(pval)}', f'({se:.3f})'

def run_iv(fe_terms, data, clusters):
    fe = (f' + {fe_terms}') if fe_terms else ''
    iv = IV2SLS.from_formula(
        f'log_trips_w ~ 1{fe} + [log_rpt_w ~ log_loo_eq_rpt_w]', data=data
    ).fit(cov_type='clustered', clusters=clusters)
    first_stage = iv.first_stage.diagnostics.loc['log_rpt_w']
    return iv, first_stage['f.stat']

# ── OLS ───────────────────────────────────────────────────────────────────────
ols1 = smf.ols('log_trips_w ~ log_rpt_w',                      data=dd).fit(cov_type='cluster', cov_kwds={'groups': clust})
ols2 = smf.ols('log_trips_w ~ log_rpt_w + C(weekday)',         data=dd).fit(cov_type='cluster', cov_kwds={'groups': clust})
ols3 = smf.ols('log_trips_w ~ log_rpt_w + C(weekday) + C(ym)', data=dd).fit(cov_type='cluster', cov_kwds={'groups': clust})

# ── IV ────────────────────────────────────────────────────────────────────────
iv1, iv1_fs = run_iv('',                   dd, clust)
iv2, iv2_fs = run_iv('C(weekday)',         dd, clust)
iv3, iv3_fs = run_iv('C(weekday) + C(ym)', dd, clust)

ELASTICITY = float(iv3.params['log_rpt_w'])

# ── FE-residualized arrays for binscatter (Frisch-Waugh) ─────────────────────
_fe = 'C(weekday) + C(ym)'
_y  = smf.ols(f'log_trips_w      ~ {_fe}', data=dd).fit().resid.values
_x  = smf.ols(f'log_rpt_w        ~ {_fe}', data=dd).fit().resid.values
_z  = smf.ols(f'log_loo_eq_rpt_w ~ {_fe}', data=dd).fit().resid.values
_b1 = float(np.dot(_z, _x) / np.dot(_z, _z))
full_spec = {
    'y': _y, 'x': _x, 'z': _z, 'x_hat': _z * _b1,
    'ols_coef':      float(ols3.params['log_rpt_w']),
    'iv_coef':       ELASTICITY,
    'first_stage_stat': iv3_fs,
    'nobs':          int(iv3.nobs),
}

# %% Notebook code cell 9
# ── results table ─────────────────────────────────────────────────────────────
Y, N = 'Yes', ''

def row_ols(m, fe_flags):
    c, s = fmt(m.params['log_rpt_w'], m.bse['log_rpt_w'], m.pvalues['log_rpt_w'])
    return [c, s, '', f'{int(m.nobs):,}', f'{m.rsquared_adj:.3f}'] + fe_flags

def row_iv(m, first_stage_stat, fe_flags):
    c, s = fmt(float(m.params['log_rpt_w']), float(m.std_errors['log_rpt_w']), float(m.pvalues['log_rpt_w']))
    return [c, s, f'{first_stage_stat:.0f}', f'{int(m.nobs):,}', ''] + fe_flags

table = pd.DataFrame({
    '':          ['Elasticity', '(SE)', 'First-stage stat.', 'N', 'Adj. R2', 'Driver FE', 'Weekday FE', 'Year-month FE'],
    'OLS (1)': row_ols(ols1, [Y, N, N]),
    'OLS (2)': row_ols(ols2, [Y, Y, N]),
    'OLS (3)': row_ols(ols3, [Y, Y, Y]),
    'IV (4)':  row_iv(iv1, iv1_fs, [Y, N, N]),
    'IV (5)':  row_iv(iv2, iv2_fs, [Y, Y, N]),
    'IV (6)':  row_iv(iv3, iv3_fs, [Y, Y, Y]),
})

print('Table 1: Driver-Day Supply Elasticity Estimates')
print()
print('Dependent Variable: Log(Number of Trips)')
print(table.to_string(index=False))
print()
print('Notes: SE clustered by date. Significance: * p<0.10,  ** p<0.05,  *** p<0.01')
print('First-stage stat. is linearmodels first_stage.diagnostics f.stat.')

# ── save ──────────────────────────────────────────────────────────────────────
model_summary = pd.DataFrame([
    {'spec': n, 'estimator': est, 'Log(Revenue per Trip)': e, 'se': s, 'p_value': p, 'first_stage_stat': fs, 'adj_r2': r2, 'nobs': nobs}
    for n, est, e, s, p, fs, r2, nobs in [
        ('OLS (1)', 'OLS', ols1.params['log_rpt_w'], ols1.bse['log_rpt_w'],       ols1.pvalues['log_rpt_w'], np.nan, ols1.rsquared_adj, int(ols1.nobs)),
        ('OLS (2)', 'OLS', ols2.params['log_rpt_w'], ols2.bse['log_rpt_w'],       ols2.pvalues['log_rpt_w'], np.nan, ols2.rsquared_adj, int(ols2.nobs)),
        ('OLS (3)', 'OLS', ols3.params['log_rpt_w'], ols3.bse['log_rpt_w'],       ols3.pvalues['log_rpt_w'], np.nan, ols3.rsquared_adj, int(ols3.nobs)),
        ('IV (4)',  'IV',  iv1.params['log_rpt_w'],  iv1.std_errors['log_rpt_w'], iv1.pvalues['log_rpt_w'], iv1_fs, np.nan, int(iv1.nobs)),
        ('IV (5)',  'IV',  iv2.params['log_rpt_w'],  iv2.std_errors['log_rpt_w'], iv2.pvalues['log_rpt_w'], iv2_fs, np.nan, int(iv2.nobs)),
        ('IV (6)',  'IV',  iv3.params['log_rpt_w'],  iv3.std_errors['log_rpt_w'], iv3.pvalues['log_rpt_w'], iv3_fs, np.nan, int(iv3.nobs)),
    ]
])
model_summary.to_csv(OUTPUTS / 'driver_day_model_summary.csv', index=False)

# %% Notebook code cell 10
def binscatter(x_values, y_values, ax, q=40, title='', xlabel='', ylabel='', color='steelblue'):
    plot_df = pd.DataFrame({'x': x_values, 'y': y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    plot_df['bin'] = pd.qcut(plot_df['x'], q=q, duplicates='drop')
    binned = (
        plot_df.groupby('bin', observed=True)
        .agg(x=('x', 'mean'), y=('y', 'mean'))
        .reset_index(drop=True)
    )
    slope = float(np.polyfit(binned['x'], binned['y'], 1)[0])
    sns.scatterplot(data=binned, x='x', y='y', color=color, s=22, edgecolor=None, alpha=0.9, ax=ax)
    sns.regplot(
        data=binned, x='x', y='y', scatter=False, ci=None, color='crimson', ax=ax,
        line_kws={'lw': 1.6, 'ls': '--', 'label': f'Slope = {slope:.3f}'},
    )
    ax.axhline(0, color='#bbbbbb', lw=0.7, ls=':')
    ax.axvline(0, color='#bbbbbb', lw=0.7, ls=':')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    return slope

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
binscatter(
    full_spec['z'], full_spec['x'], ax1,
    title='First stage',
    xlabel='Log leave-one-out revenue/trip (demeaned)',
    ylabel='Log own revenue/trip (demeaned)',
)
binscatter(
    full_spec['x_hat'], full_spec['y'], ax2,
    title='Second stage',
    xlabel='Instrumented log revenue/trip (demeaned)',
    ylabel='Log completed trips (demeaned)',
)
fig.suptitle('Figure 3: Functional form checks')
fig.tight_layout()
fig.savefig(OUTPUTS / '03_binscatter_functional_form.svg', bbox_inches='tight')
plt.close(fig)
print('Figure saved: outputs/03_binscatter_functional_form.svg')

# %% Notebook code cell 11
# Helper function: round to the closest multiple of 5
def round_to_5(value):
    return int(5 * np.round(value / 5))

# Setup
TIER1_EARNINGS_INCREASE = 0.05
TIER2_TOTAL_EARNINGS_INCREASE = 0.10

typical_revenue_per_trip = driver_week['weekly_earnings'].sum() / driver_week['weekly_trips'].sum()
driver_revenue_share = 0.5
driver_revenue_per_trip = typical_revenue_per_trip * driver_revenue_share

# Calculate trip thresholds: 60th and 80th quantile of weekly trips
threshold_1 = round_to_5(driver_week['weekly_trips'].quantile(0.60))
threshold_2 = round_to_5(driver_week['weekly_trips'].quantile(0.80))

# Calibrate marginal rewards: Tier 1 raises retained earnings at its threshold by 5%,
# Tier 2's marginal reward is set so total retained earnings at its threshold rise by 10%.
tier1_reward_pct = TIER1_EARNINGS_INCREASE
tier1_reward = round_to_5(threshold_1 * driver_revenue_per_trip * tier1_reward_pct)
tier2_reward_pct = (
    (threshold_2 * driver_revenue_per_trip * TIER2_TOTAL_EARNINGS_INCREASE) - tier1_reward
) / (threshold_2 * driver_revenue_per_trip)
tier2_reward = round_to_5(threshold_2 * driver_revenue_per_trip * tier2_reward_pct)

tier_table = pd.DataFrame({
    'Tier': ['Tier 1', 'Tier 2'],
    'Completed trips per week': [threshold_1, threshold_2],
    'Marginal bonus ($)': [tier1_reward, tier2_reward],
    'Total bonus ($)': [tier1_reward, tier1_reward + tier2_reward],
    'Total earnings increase at tier': [
        tier1_reward / (threshold_1 * driver_revenue_per_trip),
        (tier1_reward + tier2_reward) / (threshold_2 * driver_revenue_per_trip),
    ],
})
print(tier_table.to_string(index=False, formatters={'Total earnings increase at tier': '{:.1%}'.format}))

# %% Notebook code cell 12
def simulate_two_tier_driver_week_frame(tiers, *, driver_week=driver_week):
    sim = driver_week.copy()
    sim['trips_post'] = sim['weekly_trips'].astype(float)

    # Apply the higher tier first so drivers close to Tier 2 are moved to the higher milestone.
    for t in sorted(tiers, key=lambda d: d['threshold'], reverse=True):
        mask = (sim['weekly_trips'] >= t['lower_bound']) & (sim['weekly_trips'] < t['threshold'])
        sim.loc[mask, 'trips_post'] = np.maximum(sim.loc[mask, 'trips_post'], t['threshold'])

    sim['bonus'] = 0.0
    for t in tiers:
        sim.loc[sim['trips_post'] >= t['threshold'], 'bonus'] += t['marginal_reward']

    return sim


def simulate_two_tier_incentive(
    tier1_threshold,
    tier2_threshold,
    tier1_reward_pct,
    tier2_reward_pct,
    *,
    driver_week=driver_week,
    elasticity=ELASTICITY,
    typical_revenue_per_trip=typical_revenue_per_trip,
    driver_revenue_share=driver_revenue_share,
):
    if tier2_threshold <= tier1_threshold:
        raise ValueError('tier2_threshold must be greater than tier1_threshold')

    driver_revenue_per_trip = typical_revenue_per_trip * driver_revenue_share
    tier1_reward = round_to_5(tier1_threshold * driver_revenue_per_trip * tier1_reward_pct)
    tier2_reward = round_to_5(tier2_threshold * driver_revenue_per_trip * tier2_reward_pct)

    tiers = [
        {
            'name': 'Tier 1',
            'threshold': tier1_threshold,
            'reward_pct': tier1_reward_pct,
            'marginal_reward': tier1_reward,
            'total_reward': tier1_reward,
        },
        {
            'name': 'Tier 2',
            'threshold': tier2_threshold,
            'reward_pct': tier2_reward_pct,
            'marginal_reward': tier2_reward,
            'total_reward': tier1_reward + tier2_reward,
        },
    ]

    for t in tiers:
        threshold_driver_earnings = t['threshold'] * driver_revenue_per_trip
        t['total_earnings_increase_at_threshold'] = t['total_reward'] / threshold_driver_earnings
        t['marginal_incentive_at_threshold'] = t['marginal_reward'] / threshold_driver_earnings
        t['lower_bound'] = t['threshold'] / (1 + elasticity * t['marginal_incentive_at_threshold'])

    sim = simulate_two_tier_driver_week_frame(tiers, driver_week=driver_week)

    base_trips = sim['weekly_trips'].sum()
    additional_trips = sim['trips_post'].sum() - base_trips
    additional_revenue = additional_trips * typical_revenue_per_trip * (1 - driver_revenue_share)
    total_bonus = sim['bonus'].sum()
    total_cost = total_bonus - additional_revenue
    total_cost_per_additional_trip = total_cost / additional_trips if additional_trips else np.nan

    return pd.Series({
        'base_trips': base_trips,
        'additional_trips': additional_trips,
        'additional_trips_pct': additional_trips / base_trips,
        'typical_revenue_per_trip': typical_revenue_per_trip,
        'driver_revenue_share': driver_revenue_share,
        'additional_company_revenue': additional_revenue,
        'total_bonus_paid': total_bonus,
        'net_program_cost': total_cost,
        'net_cost_per_additional_trip': total_cost_per_additional_trip,
        'tier1_threshold': tier1_threshold,
        'tier1_reward_pct': tier1_reward_pct,
        'tier1_marginal_reward': tier1_reward,
        'tier1_lower_bound': tiers[0]['lower_bound'],
        'tier1_total_reward': tiers[0]['total_reward'],
        'tier1_total_earnings_increase_at_threshold': tiers[0]['total_earnings_increase_at_threshold'],
        'tier2_threshold': tier2_threshold,
        'tier2_reward_pct': tier2_reward_pct,
        'tier2_marginal_reward': tier2_reward,
        'tier2_total_reward': tiers[1]['total_reward'],
        'tier2_lower_bound': tiers[1]['lower_bound'],
        'tier2_total_earnings_increase_at_threshold': tiers[1]['total_earnings_increase_at_threshold'],
    })


def tiers_from_simulation_result(result):
    return [
        {
            'name': 'Tier 1',
            'threshold': result['tier1_threshold'],
            'reward_pct': result['tier1_reward_pct'],
            'marginal_reward': result['tier1_marginal_reward'],
            'total_reward': result['tier1_total_reward'],
            'lower_bound': result['tier1_lower_bound'],
            'total_earnings_increase_at_threshold': result['tier1_total_earnings_increase_at_threshold'],
        },
        {
            'name': 'Tier 2',
            'threshold': result['tier2_threshold'],
            'reward_pct': result['tier2_reward_pct'],
            'marginal_reward': result['tier2_marginal_reward'],
            'total_reward': result['tier2_total_reward'],
            'lower_bound': result['tier2_lower_bound'],
            'total_earnings_increase_at_threshold': result['tier2_total_earnings_increase_at_threshold'],
        },
    ]

# %% Notebook code cell 13
baseline_incentive_result = simulate_two_tier_incentive(
    threshold_1,
    threshold_2,
    tier1_reward_pct,
    tier2_reward_pct,
)
tiers = tiers_from_simulation_result(baseline_incentive_result)

print(f"Elasticity used for simulation: {ELASTICITY:.3f}")
print(f"Baseline wage per trip used for calibration: ${typical_revenue_per_trip * driver_revenue_share:.2f}")
print()
print(f"{'Tier':<8} {'Response window':<34}  {'Driver-weeks':>14}")
print('-' * 81)
for t in tiers:
    n = ((driver_week['weekly_trips'] >= t['lower_bound']) & (driver_week['weekly_trips'] < t['threshold'])).sum()
    print(
        f"{t['name']:<8} "
        f"[{t['lower_bound']:6.1f}, {t['threshold']:6.0f}) trips -> {t['threshold']:.0f} {n:>14,}"
    )

# %% Notebook code cell 14
base_trips = baseline_incentive_result['base_trips']
additional_trips = baseline_incentive_result['additional_trips']
base_revenue = base_trips * typical_revenue_per_trip
additional_revenue = baseline_incentive_result['additional_company_revenue']
total_bonus = baseline_incentive_result['total_bonus_paid']
total_cost = baseline_incentive_result['net_program_cost']
total_cost_per_additional_trip = baseline_incentive_result['net_cost_per_additional_trip']

# Print results
print(f'Additional trips:            {additional_trips:,.0f}     (+{additional_trips / base_trips:.2%})')
print(f'Additional revenue:          {additional_revenue:,.1f} (+{additional_revenue / base_revenue:.2%})')
print(f'Total bonus paid:           ${total_bonus:,.0f}')
print(f'Total program cost:         ${total_cost:,.0f}')
print(f'Cost per additional trip:   ${total_cost_per_additional_trip:,.2f}')

incentive_summary = baseline_incentive_result.rename_axis('metric').reset_index(name='value')
incentive_summary.to_csv(OUTPUTS / 'incentive_simulation_summary.csv', index=False)

# %% Notebook code cell 15
threshold_grid = [50, 75, 100, 125, 150]
reward_pct_grid = [0.05, 0.10, 0.15]

optimization_records = []
for tier1_threshold in threshold_grid:
    for tier2_threshold in threshold_grid:
        if tier2_threshold <= tier1_threshold:
            continue
        for tier1_reward_pct in reward_pct_grid:
            for tier2_reward_pct in reward_pct_grid:
                optimization_records.append(
                    simulate_two_tier_incentive(
                        tier1_threshold,
                        tier2_threshold,
                        tier1_reward_pct,
                        tier2_reward_pct,
                    )
                )

optimization_results = pd.DataFrame(optimization_records)
optimization_results.to_csv(OUTPUTS / 'incentive_optimization_results.csv', index=False)

feasible_results = optimization_results[
    optimization_results['net_program_cost'] < 2_000_000
]
best_design = feasible_results.sort_values('additional_trips', ascending=False).iloc[0]
best_design.to_frame().T.to_csv(OUTPUTS / 'incentive_optimization_best_design.csv', index=False)

assert (optimization_results['tier2_threshold'] > optimization_results['tier1_threshold']).all()
assert best_design['net_program_cost'] < 2_000_000
assert feasible_results['additional_trips'].max() == best_design['additional_trips']
for fn in [
    'incentive_simulation_summary.csv',
    'incentive_optimization_results.csv',
    'incentive_optimization_best_design.csv',
]:
    assert (OUTPUTS / fn).exists()

print('Best feasible two-tier incentive design')
print('-' * 41)
print(f"Tier 1 threshold:           {best_design['tier1_threshold']:,.0f} trips")
print(f"Tier 2 threshold:           {best_design['tier2_threshold']:,.0f} trips")
print(f"Tier 1 reward percentage:   {best_design['tier1_reward_pct']:.0%}")
print(f"Tier 2 reward percentage:   {best_design['tier2_reward_pct']:.0%}")
print(f"Tier 1 marginal reward:    ${best_design['tier1_marginal_reward']:,.0f}")
print(f"Tier 2 marginal reward:    ${best_design['tier2_marginal_reward']:,.0f}")
print()
print(f"Additional trips:            {best_design['additional_trips']:,.0f}     (+{best_design['additional_trips_pct']:.2%})")
print(f"Additional revenue:          {best_design['additional_company_revenue']:,.1f} (+{best_design['additional_company_revenue'] / (best_design['base_trips'] * best_design['typical_revenue_per_trip']):.2%})")
print(f"Total bonus paid:           ${best_design['total_bonus_paid']:,.0f}")
print(f"Total program cost:         ${best_design['net_program_cost']:,.0f}")
print(f"Cost per additional trip:   ${best_design['net_cost_per_additional_trip']:,.2f}")

# %% Notebook code cell 16
best_tiers = tiers_from_simulation_result(best_design)
sim_best = simulate_two_tier_driver_week_frame(best_tiers, driver_week=driver_week)

plot_cap = sim_best[['weekly_trips', 'trips_post']].to_numpy().ravel()
plot_cap = np.nanquantile(plot_cap, 0.99)
bins = np.linspace(0, plot_cap, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, col, title, color in [
    (ax1, 'weekly_trips', 'Baseline', 'steelblue'),
    (ax2, 'trips_post', 'With optimized trip-tier bonus', 'tomato'),
]:
    plot_values = sim_best[col].clip(upper=plot_cap)
    ax.hist(plot_values, bins=bins, color=color, edgecolor='white', linewidth=0.25)
    for t, c in zip(best_tiers, ['#555555', '#111111']):
        ax.axvline(t['threshold'], color=c, lw=1.5, linestyle='--', label=f"{t['name']} ({t['threshold']:.0f} trips)")
    ax.set(xlabel='Completed trips per driver-week (top 1% clipped)', title=title)
ax1.set_ylabel('Driver-weeks')
fig.suptitle('Figure 4: Completed trips without and with optimized trip-tier bonus', y=1.03)
fig.tight_layout()
fig.savefig(OUTPUTS / '04_optimized_trip_tier_simulation.svg', bbox_inches='tight')
plt.close(fig)
print('Figure saved: outputs/04_optimized_trip_tier_simulation.svg')

