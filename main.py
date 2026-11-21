"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only, no network). Reads each case file /input/<id>.in and
writes /output/<id>.out with five quantiles (5/25/50/75/95%) of every
trophy's earn rate at the case's target.

Model (per trophy, on the logit scale)
--------------------------------------
An earn rate is a share of the players Exophase tracks, so it lives on a
logit scale: logit(r) = ln(r / (100 - r)).  Two forces move it: players
earn the trophy (pushes the rate up) and new players join the tracked pool
(dilutes the rate down).  Both forces slow as the game ages, so the logit
rate is modelled as a smooth curve in log-time, x = ln(1 + hours since
release):

    logit(rate) ~ a + b * (x - x_last) + c * (x - x_last)^2

fitted by recency-weighted least squares (half-life 24 h) over the case's
reads, with three guards against the ways a raw extrapolation goes wrong:

* the slope is shrunk toward the cross-trophy median slope when its own
  standard error is large (empirical-Bayes weight b^2 / (b^2 + se^2)), so
  a noisy trophy borrows the typical trophy's drift instead of chasing
  its own noise;
* the curvature term is kept only when it cuts the weighted residual
  variance by at least 5%, and is then halved, so the forecast bends only
  on real evidence;
* the extrapolated move is capped at 3 logit units from the last read.

The spread is honest, not a fixed number: a random-walk term that grows
with the horizon (0.5 * sqrt(days), the house baseline's width), plus the
fit's residual scatter, plus the slope's standard error times how far the
target sits past the last read.  On a case whose history never moves (the
practice cases, frozen launch-night reads) the residual and slope terms
vanish and the forecast reduces to the house baseline: last rate, spread
0.5 * sqrt(days).  On the final case the spread widens with the data's
real volatility.

Output is deterministic: no randomness, no dict-order dependence, rows in
trophy order, numbers rounded half-up to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file,
# but the line keeps the layout the kit documents.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_RW = 0.5          # logit units per sqrt(day): the random-walk width
MAX_MOVE = 3.0          # cap on |forecast logit - last logit|
SIGMA_FLOOR = 0.40      # never claim to know the rate better than this
SIGMA_CAP = 2.50        # never spread wider than this
HALF_LIFE_H = 24.0      # recency weight half-life, in hours
CURVE_KEEP = 0.05       # curvature must cut residual variance by 5%


def clamp_rate(pct: float) -> float:
    """An earn rate as printed (0..100) onto the logit scale's range."""
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round half-up to 4 decimals (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_time(case: dict) -> datetime:
    raw = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        return parse_time(raw) if "T" in raw else datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime(2026, 11, 19, tzinfo=timezone.utc)


def trophy_series(case: dict, index: int):
    """(x, y, w) triples for one trophy: log-time, logit rate, recency weight."""
    release = release_time(case)
    history = case.get("history", [])
    if not history:
        return [], None
    last_at = parse_time(history[-1]["at"])
    points = []
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        at = parse_time(read["at"])
        hours = (at - release).total_seconds() / 3600.0
        age_h = (last_at - at).total_seconds() / 3600.0
        weight = 0.5 ** (age_h / HALF_LIFE_H)
        points.append((math.log1p(max(0.0, hours)), logit(rates[index]), weight))
    return points, last_at


def weighted_linear(points):
    """Weighted least squares y ~ a + b * t, t centred on the last point.

    Returns (level_at_last, slope, slope_se, residual_std, n, x_last).
    """
    if not points:
        return None
    x_last = points[-1][0]
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    mx = sum(w * (x - x_last) for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - x_last - mx) ** 2 for x, _, w in points)
    sxy = sum(w * (x - x_last - mx) * (y - my) for x, y, w in points)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0, len(points), x_last)
    slope = sxy / sxx
    level = my + slope * (0.0 - mx)  # fitted value at t = x_last
    # Weights normalised to mean 1, so the residual variance and the
    # slope's standard error are on the usual least-squares scale.
    n = len(points)
    wscale = n / sw
    sse = sum(wscale * w * (y - (level + slope * (x - x_last))) ** 2 for x, y, w in points)
    resid_std = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    slope_se = math.sqrt(sse / (n - 2) / (wscale * sxx)) if n > 2 and sxx > 0 else 0.0
    return (level, slope, slope_se, resid_std, n, x_last)


def fit_curvature(points, level, slope, x_last):
    """One-parameter quadratic term on the linear fit's residuals.

    Returns (curve, kept): the coefficient on (x - x_last)^2, halved, and
    whether it earned its keep by cutting residual variance by CURVE_KEEP.
    """
    if len(points) < 5:
        return 0.0, False
    base_sse = 0.0
    num = 0.0
    den = 0.0
    for x, y, w in points:
        t = x - x_last
        resid = y - (level + slope * t)
        base_sse += w * resid * resid
        num += w * (t * t) * resid
        den += w * (t * t) ** 2
    if den <= 1e-12 or base_sse <= 0:
        return 0.0, False
    curve = num / den
    new_sse = 0.0
    for x, y, w in points:
        t = x - x_last
        resid = y - (level + slope * t + curve * t * t)
        new_sse += w * resid * resid
    if new_sse < base_sse * (1.0 - CURVE_KEEP):
        return 0.5 * curve, True
    return 0.0, False


def forecast_tracked(case: dict, target_x: float):
    """Players tracked at the target, from a damped log-time fit.

    The tracked pool only grows. Its log count is fitted linearly in
    log-time over the case's reads (recency-weighted, the same fit as a
    trophy's), then the implied growth past the last read is damped by
    30%: launch-week growth decelerates, and an undamped line overshoots.
    """
    release = release_time(case)
    history = case.get("history", [])
    if not history:
        return None
    last_at = parse_time(history[-1]["at"])
    points = []
    for read in history:
        count = read.get("players_tracked")
        if not count or count <= 0:
            continue
        at = parse_time(read["at"])
        hours = (at - release).total_seconds() / 3600.0
        age_h = (last_at - at).total_seconds() / 3600.0
        points.append((math.log1p(max(0.0, hours)), math.log(count), 0.5 ** (age_h / HALF_LIFE_H)))
    if not points:
        return None
    n_last = math.exp(points[-1][1])
    if len(points) < 3:
        return n_last
    fit = weighted_linear(points)
    if fit is None:
        return n_last
    level, slope, _, _, _, x_last = fit
    log_n = level + 0.7 * slope * max(0.0, target_x - x_last)
    return max(n_last, math.exp(log_n))


def tracked_growth_24h(case: dict) -> float:
    """How much the tracked pool grew over the last 24 h of reads (a ratio).

    A fast-growing pool dilutes every rate at the next refresh, so the
    ratio feeds the spread: a pool that doubled in a day is a rate that
    can halve overnight.
    """
    history = case.get("history", [])
    if len(history) < 2:
        return 1.0
    last_at = parse_time(history[-1]["at"])
    n_last = history[-1].get("players_tracked")
    if not n_last or n_last <= 0:
        return 1.0
    n_24h = None
    for read in history:
        if (last_at - parse_time(read["at"])).total_seconds() <= 24 * 3600:
            n_24h = read.get("players_tracked")
            break
    if not n_24h or n_24h <= 0:
        return 1.0
    return max(1.0, n_last / n_24h)


def earned_floor(last_y: float, n_last, n_target):
    """The logit floor dilution alone implies: earned counts never fall.

    If the tracked pool grows to n_target while nobody new earns the
    trophy, the rate falls to 100 * earned_last / n_target and no lower.
    Returns that floor as a logit, or None when the pool is unknown.
    """
    if not n_last or not n_target or n_target <= 0:
        return None
    earned_last = inv_logit(last_y) * n_last / 100.0
    return logit(100.0 * earned_last / n_target)


def forecast_trophy(points, horizon_days, target_x, median_slope,
                    n_last=None, n_target=None, refresh_sigma=0.0):
    """Five quantiles for one trophy from its (x, y, w) history."""
    if not points:
        return [0.01, 0.01, 0.01, 0.01, 0.01]
    last_y = points[-1][1]
    floor = earned_floor(last_y, n_last, n_target)
    if len(points) < 3:
        # Too thin to fit: hold the last rate, widen the spread.
        centre = last_y if floor is None else max(last_y, floor)
        sigma = max(SIGMA_FLOOR, SIGMA_RW * math.sqrt(horizon_days) * 1.5)
        return quantiles(centre, sigma)
    fit = weighted_linear(points)
    if fit is None:
        return quantiles(last_y, SIGMA_RW * math.sqrt(horizon_days))
    level, slope, slope_se, resid_std, n, x_last = fit
    # Shrink the slope toward the typical trophy's drift when it is noisy.
    shrink = (slope * slope) / (slope * slope + slope_se * slope_se) if slope_se > 0 else 1.0
    slope = shrink * slope + (1.0 - shrink) * median_slope
    curve, _ = fit_curvature(points, level, slope, x_last)
    dx = target_x - x_last
    move = slope * dx + curve * dx * dx
    move = max(-MAX_MOVE, min(MAX_MOVE, move))
    centre = level + move
    centre = max(logit(0.01), min(logit(99.99), centre))
    if floor is not None:
        centre = max(centre, floor)
    sigma = math.sqrt(
        (SIGMA_RW * math.sqrt(horizon_days)) ** 2
        + (1.5 * resid_std) ** 2
        + (slope_se * abs(dx)) ** 2
        + refresh_sigma ** 2
    )
    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
    return quantiles(centre, sigma)


def quantiles(centre: float, sigma: float):
    out = []
    for z in Z:
        value = inv_logit(centre + z * sigma)
        value = min(100.0, max(0.01, value))
        out.append(round4(value))
    # Keep the row non-decreasing after rounding.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    if not trophies:
        raise ValueError(f"{name} lists no trophies")
    horizon_days = case["horizon_minutes"] / 1440.0
    release = release_time(case)
    target = parse_time(case["target"])
    target_x = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    # Pass 1: every trophy's own linear fit, for the cross-trophy median slope.
    series = []
    slopes = []
    for i in range(len(trophies)):
        points, _ = trophy_series(case, i)
        series.append(points)
        if points and len(points) >= 3:
            fit = weighted_linear(points)
            if fit is not None:
                slopes.append(fit[1])
    slopes.sort()
    median_slope = slopes[len(slopes) // 2] if slopes else 0.0
    # The tracked pool at the target, and how fast it grew lately: the
    # earned-count floor and the dilution-risk spread both hang off it.
    n_last = history[-1].get("players_tracked") if history else None
    n_target = forecast_tracked(case, target_x)
    growth = tracked_growth_24h(case)
    refresh_sigma = 0.5 * math.log(growth) if growth > 1.0 else 0.0
    # Pass 2: the forecast itself.
    rows = []
    for i, trophy in enumerate(trophies):
        points = series[i]
        if not points:
            # No read ever listed this trophy: hold the floor, wide spread.
            rows.append({"id": trophy["id"], "q": quantiles(logit(0.01), SIGMA_CAP)})
            continue
        rows.append(
            {"id": trophy["id"], "q": forecast_trophy(
                points, horizon_days, target_x, median_slope,
                n_last=n_last, n_target=n_target, refresh_sigma=refresh_sigma)}
        )
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit(f"no .in files in {INPUT_DIR}")
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise SystemExit(f"{path.stem}.out would exceed 65536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
