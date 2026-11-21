# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hmtmm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hmtmm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI trophy earn rates on Exophase: given the house's reads of the game's PS5 trophy page, it predicts every trophy's earn rate one week after launch as five quantiles (5/25/50/75/95%), modelling the logit rate as a curve in log-time.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Put the case files (`<id>.in`) in `input/`; the forecast for each lands in `out/<id>.out`. The program is standard library only, runs offline, and is deterministic: two runs write identical bytes.

## How it forecasts

An earn rate is a share of the players Exophase tracks, so it is modelled on the logit scale as a smooth curve in log-time, `logit(rate) ~ a + b * (x - x_last) + c * (x - x_last)^2`, fitted by recency-weighted least squares over the case's reads. Three guards keep the extrapolation honest: the slope shrinks toward the cross-trophy median slope when its own standard error is large; the curvature term is kept only when it cuts residual variance by 5% or more; and the extrapolated move is capped at 3 logit units. Two facts about the data sharpen it further: earned counts never fall, so the forecast rate cannot drop below what pure dilution of the last earned count by the target player pool implies; and a fast-growing tracked pool widens the spread, because the next refresh can dilute every rate at once. The spread itself combines a random-walk term that grows with the horizon, the fit's residual scatter, and the slope's standard error times the extrapolation distance — on a case whose history never moves, it reduces to the house baseline's width.

---

<sub>Entered on [make.fun](https://make.fun/?ref=f6d62qsk&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
