# Methodology Notes

These notes preserve the durable lessons from earlier project drafts. They are reference material, not the main README.

## Identification

Realized revenue per trip is endogenous. On high-demand days, drivers may complete more trips and earn different revenue per trip for reasons unrelated to a clean labor-supply response. OLS is therefore useful as a benchmark, but not as the preferred interpretation.

The current notebook uses a Hausman-style leave-one-out instrument: a driver's own revenue per trip is instrumented by same-day equal-weight average revenue per trip among other active drivers. Equal weighting matters because volume-weighted averages can mechanically put supply in the denominator.

## Why Driver-Day

Earlier spatial hourly variants made spillovers hard to defend: drivers can reposition across small urban zones, and demand shocks are correlated across the city. The driver-day design avoids zone-boundary sorting and focuses on within-driver variation across dates.

The design identifies an intensive-margin response among drivers who are active on a date. It does not estimate whether inactive drivers decide to work.

## Inference

The instrument mostly varies by date, so standard errors should be clustered by date. More driver-day rows within a date do not create independent instrument variation. This is why the multi-month panel is important.

## Wage Measure

Revenue per trip is the preferred decision-relevant measure in the current notebook. Revenue per hour is more of an ex-post accounting measure, while revenue per trip is closer to the value of the next completed fare.

All wage measures constructed from trip data are ratios. Be careful about mechanical relationships between the outcome and denominator, especially when interpreting OLS.

## Interpretation

The analysis is a public-data analogue for an incentive-design workflow: clean panel construction, IV estimation, diagnostics, and a tier simulation. It should be described as observational evidence for simulation, not as an experiment or final rollout recommendation.

The best next step for a real operator would be an A/B test or randomized incentive experiment with pre-announced bonuses and richer driver availability data.
