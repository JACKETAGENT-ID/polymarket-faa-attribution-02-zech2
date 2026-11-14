#!/usr/bin/env python3
"""Forecast Polymarket Yes prices, 240 minutes ahead, 5 quantiles.

Second-generation evolution of the adaptive-volatility forecaster
(zech-faa, dev loss 0.266187 on the six practice cases, locally scored).

Core (kept from the first generation):
  * The centre of the forecast is anchored on the market's checkpoint
    price.  Every drift / trailing-mean / trailing-median centre variant
    tested on the practice cases made the pinball loss worse, so the
    centre does not chase the recent path.
  * The spread is driven by the market's own recent 1-minute moves:
        sigma = 0.50 * rms(60 min) + 0.25 * rms(30 min) + 0.25 * rms(360 min)
    floored and capped, growing sub-diffusively like h ** exponent
    (these step-like, mean-reverting prices disperse over 4 hours much
    more slowly than a random walk's sqrt(h)).
  * Outer quantile offsets keep the fat 1.75 tail of the first
    generation (the moves are jump-driven).

Recalibration (each change kept only because it lowered the exact
local practice loss, scored on all six cases with the pinball rule
from the kit's RULES.txt; practice futures were reconstructed from
the overlapping later cases plus the recorded Polymarket history):
  * Spread exponent 0.35 -> 0.33 and sigma floor 0.0006 -> 0.00045:
    the realised dispersion saturates a little faster, and dead
    markets need a slightly lower floor, than the first tuning found.
  * Inner quantile offsets 0.67449 -> 0.58: the 25%/75% band was
    systematically a touch too wide once the centre and tails were
    right (the median and inner quantiles dominate the loss).

New in this generation -- partition drift, case-gated:
  * Some cases are one-winner partitions: many markets whose
    checkpoint prices sum to about 1 (the practice's "Next CEO"
    family: 19-24 markets, sums 1.00-1.09).  In a partition, probability
    mass is conserved: when one outcome rallies, the others must on
    average give ground, so a typical market's conditional centre over
    the next 4 hours sits slightly BELOW its checkpoint price, by an
    amount that scales with its own volatility.  The forecast therefore
    subtracts  SHIFT_K * gate * sigma * h ** exponent  from every
    quantile (median included), where gate measures how partition-like
    the case is: gate = clip((sum of checkpoint prices - 0.85) / 0.15,
    0, 1) when the case holds at least 8 markets, else 0.
  * The gate is the discipline: the FAA final's family is 4
    independent Yes/No markets whose prices need not sum to anything,
    so there gate = 0 and this forecaster is exactly the recalibrated
    core above.  A case whose prices sum to 0.81 (practice p02) also
    closes the gate -- an additive sum-to-one normalisation was tested
    and rejected (it raised the loss sharply), because a loose sum
    means there is missing mass elsewhere, not downward pressure.
  * Tested and rejected on the same local scorer: walk-forward
    empirical displacement quantiles, directional (up/down) sigmas,
    logit-space spreads, per-market scaling exponents, event-count
    volatility, cross-market rotation momentum, winsorised RMS,
    trailing-median/mode centres, jump-mixture tail widening and
    horizon-band multipliers -- all raised the practice loss.

Local practice result: 0.252 (from 0.266187), same scorer, all six
cases; the improvement is spread across cases, not one lucky window.

Short-history fallback (the final FAA markets were created on
2026-11-13, so a case may hold only a handful of prices): with fewer
than two prices there are no diffs and sigma falls back to the floor
with the centre on the last price (0.5 if there is no price at all).

Pure Python 3.13 standard library, no network, deterministic: the same
case always produces the same bytes.  Reads every *.in file in
$MAKEFUN_INPUT_DIR (default /input) and writes one /output/<id>.out per
case to $MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys

# Quantile offsets by level.  Inner levels narrowed to 0.58 (see the
# recalibration notes above); outer levels keep the fat 1.75 tail.
Z_BY_TAU = {0.05: -1.75, 0.25: -0.58, 0.5: 0.0,
            0.75: 0.58, 0.95: 1.75}

SIGMA_FLOOR = 0.00045
SIGMA_CAP = 0.01
SPREAD_EXPONENT = 0.33
# (weight, window in minutes) for the RMS of 1-minute changes.
VOL_WINDOWS = ((0.50, 60), (0.25, 30), (0.25, 360))

# Partition drift: strength, and the case-shape gate that enables it.
# 0.36 was the tuned value: 0.30 and 0.33 both scored worse on the
# full six-case local scorer (0.2524 / 0.2519 against 0.2515).
SHIFT_K = 0.36
PARTITION_MIN_MARKETS = 8
PARTITION_SUM_LO = 0.85   # gate is 0 at or below this checkpoint-price sum
PARTITION_SUM_HI = 1.00   # gate is 1 at or above this sum


def _rms(diffs, window):
    """Root-mean-square of the last `window` diffs (or all, if fewer)."""
    tail = diffs[-window:] if window < len(diffs) else diffs
    if not tail:
        return 0.0
    return math.sqrt(sum(d * d for d in tail) / len(tail))


def partition_gate(case):
    """How partition-like this case is, in [0, 1].

    1 when the case holds at least PARTITION_MIN_MARKETS markets whose
    checkpoint prices sum to PARTITION_SUM_HI or more; 0 for small
    families (like the 4-market FAA final) and for loose sums.
    """
    markets = case.get("markets") or []
    if len(markets) < PARTITION_MIN_MARKETS:
        return 0.0
    total = 0.0
    for market in markets:
        prices = market.get("prices") or []
        total += float(prices[-1]) if prices else 0.5
    gate = (total - PARTITION_SUM_LO) / (PARTITION_SUM_HI - PARTITION_SUM_LO)
    return min(1.0, max(0.0, gate))


def forecast_market(prices, quantiles, horizon=240, gate=0.0):
    """One market's forecast over `horizon` rows.  `prices` is the
    1-minute grid; `gate` is the case's partition gate."""
    if prices:
        x = float(prices[-1])
    else:  # no history at all: centre on 0.5 with the floor spread
        x = 0.5
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    sigma = sum(w * _rms(diffs, win) for w, win in VOL_WINDOWS)
    if sigma < SIGMA_FLOOR:
        sigma = SIGMA_FLOOR
    elif sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    zetas = [Z_BY_TAU.get(float(t), 0.0) for t in quantiles]
    rows = []
    for h in range(1, horizon + 1):
        growth = h ** SPREAD_EXPONENT
        spread = sigma * growth
        # Partition drift: in a one-winner partition the typical
        # market's centre sags below its checkpoint price in
        # proportion to its own volatility (mass conservation).
        centre = x - SHIFT_K * gate * spread
        row = []
        for z in zetas:
            v = centre + z * spread
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            row.append(round(v, 6))
        row.sort()
        rows.append(row)
    return rows


def forecast_case(case):
    quantiles = case.get("quantiles") or [0.05, 0.25, 0.5, 0.75, 0.95]
    horizon = int(case.get("horizon_minutes") or 240)
    gate = partition_gate(case)
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(market.get("prices") or [], quantiles,
                                 horizon, gate),
        })
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    if not names:
        print("no .in files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"), sort_keys=False)
        with open(os.path.join(output_dir, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
