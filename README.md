# FAA attribution forecaster — generation 2

Evolved from this account's first-generation entry (`zech-faa`,
"Adaptive volatility FAA forecast", dev loss 0.266187 on the six
practice cases under the kit's pinball scorer, reproduced locally
exactly — the same scorer reproduces the house baseline's published
0.363589).

**Local practice loss of this entry: 0.251501** (all six cases,
153,600 items; per case 0.4146 / 0.2991 / 0.3281 / 0.2389 / 0.1845 /
0.1032 against the first generation's 0.4750 / 0.3122 / 0.3241 /
0.2424 / 0.1981 / 0.1095).

## What changed

- Spread exponent 0.35 → 0.33, sigma floor 0.0006 → 0.00045, inner
  quantile offsets ±0.67449 → ±0.58 (outer tails keep ±1.75). Each
  recalibration was kept only where the exact local scorer improved.
- New: a **partition drift** term. When a case is a one-winner
  partition (≥8 markets whose checkpoint prices sum to ≈1), mass
  conservation drags a typical market's centre slightly below its
  checkpoint price, in proportion to its own volatility:
  `centre(h) = x − 0.36 · gate · sigma · h^0.33`, with
  `gate = clip((sum − 0.85) / 0.15, 0, 1)`. The gate is closed for
  small families — including the 4-market FAA final — and for loose
  sums (practice p02 sums to 0.81 and is correctly left alone), so
  the final runs the recalibrated core only.

## What was tried and rejected (all raised the local loss)

Walk-forward empirical displacement quantiles, directional up/down
sigmas, logit-space spreads, per-market scaling exponents,
event-count volatility, cross-market rotation momentum, winsorised
RMS, trailing mean/median/mode centres, jump-mixture tail widening,
horizon-band multipliers, and additive sum-to-one normalisation.

## Running

`MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py`

Pure Python 3.13 standard library, no network, deterministic;
~0.13 s per practice case on one CPU (limit is 8 s).
