# Phase 2 Strategy Study: Momentum and its Alternatives

A paper study comparing long-only equity strategies for the Bloomberg Global
Trading Challenge, run over rolling 5-week windows on US large/mid/small-cap
stocks (S&P 500/400/600 constituents), 2016-2026, price-and-volume data only.
The competition is long-only, no shorting, no ETFs, no derivatives, scored on
5-week relative return, so the study is confined to what that ruleset allows.

## What I set out to answer

Which strategy to run, and at what risk posture. I framed three archetypes up
front: a diversified core (steady, beat-the-benchmark), a concentrated satellite
(a high-variance bet on a few names), and an 80/20 blend of the two. The aim was
to see how they trade off return against risk, and whether any refinement beats a
simple baseline.

## Data pipeline

- Universe: 1504 S&P 500/400/600 constituents scraped from Wikipedia, mapped to
  Stooq filenames. 1500 matched in the Stooq US daily archive.
- Cleaning: trimmed to 2016 on, split-adjusted (auto-adjust clean canonical
  ratios, flag ambiguous jumps rather than risk erasing real crashes like GME or
  the COVID crash), screened for history and liquidity. 1466 names survived.
- Known biases, stated honestly: current constituents only, so survivorship-
  biased (understates the small-cap downside tail); Stooq gives no dividend
  adjustment, which happens to match the competition's price-return benchmark;
  the liquidity screen tilts toward liquid names.

## The signal

Cross-sectional momentum, grounded in Jegadeesh & Titman (1993): rank stocks by
trailing return over roughly 6 months (126 days), skipping the most recent week
to avoid short-term reversal, and buy the winners (long-only, the short-loser leg
of the academic strategy is not available). Holding-horizon support from J&T
(2001); factor status from Carhart (1997); breadth from Asness-Moskowitz-Pedersen
(2013).

## Method

Non-overlapping 5-week windows (~102) as the primary comparison, overlapping
windows (~509) as a robustness diagnostic. Look-ahead prevented by construction:
each window forms its portfolio using only prices up to the formation date, then
is scored on prices after. Mid-window delisting treated as a total loss
(conservative, keeps the downside tail honest). Reported per-window return
distribution plus VaR/CVaR tail metrics, three views (primary, overlapping,
ex-COVID), and a chained equity curve (reinvest every 5 weeks) for the compounded
total-return picture.

## Findings

### 1. The four constructions lie on a single risk-return frontier

Core, satellite, blend, and a volatility-sized core all sit on one upward
risk-return line: more risk, more return, in lockstep, no strategy dominates on
risk-adjusted terms. This is the DeMiguel-Garlappi-Uppal (2009) naive-
diversification result reproduced on my own data.

### 2. Concentration buys return only with disproportionate tail risk

The satellite (top 4 names) had a marginally higher mean than the core (0.047 vs
0.043) but a LOWER median (0.026 vs 0.043), 60% more volatility, and a worse CVaR
(0.213 vs 0.156). Its positive mean is a few right-tail wins sitting on a worse
typical outcome. On the compounded equity curve it is starker: the satellite's
higher mean produced the LOWEST compounded return of the momentum group, because
volatility drag punishes its fat left tail under compounding. Concentration is
even worse lived-through than the per-window distribution suggests.

### 3. Volatility-sizing tightens risk but does not improve risk-adjusted return

Inverse-volatility weighting the core narrowed the distribution (lowest std and
CVaR) but also lowered the mean, so the Sharpe-like ratio barely moved (0.39 vs
0.40). Same frontier, a lower-risk lower-return point, not a free lunch.
Consistent with Barroso & Santa-Clara (2015) helping most where the tail is worst,
and the long-only tail here is milder than their long-short setting.

### 4. Scheduled rebalancing of the momentum book does not help

Re-ranking momentum weekly or fortnightly within the window produced identical
gross returns to buy-and-hold (means within a thousandth) while adding 35-42%
turnover. The 6-month signal barely refreshes over 5 weeks, so rebalancing just
churns the basket edges. Hold, do not churn.

### 5. Short-term reversal is real but untradeable

Reversal (buy last week's losers) is a genuine but weaker signal, dominated by
momentum on every metric in the liquid universe (as the literature predicts,
reversal is strongest in illiquid names the screen removed). Crucially, held for
5 weeks it decays badly; its edge only recovers with frequent rebalancing, and
the recovery is monotonic in frequency (reversal-core mean 0.028 held, rising to
0.047 at twice-weekly). But that improvement costs enormous turnover (6.6x the
book per window at twice-weekly, vs 0.6x for momentum). Even gross it barely edges
momentum, and any transaction cost or spread obliterates the margin, so it is
cost-fatal and manually impossible to trade. A real gross edge that is
untradeable in practice, which is exactly the academic verdict on short-term
reversal, now demonstrated on my own data.

### 6. The momentum-reversal combination underperformed

A conditional combo (momentum selects a wide pool, reversal picks the dips within
it, held) underperformed pure momentum (blend mean 0.029 vs 0.044; compounded
13.8x vs 45.4x). Buying the dips among momentum winners did worse than buying the
winners, because the reversal refinement carries the same decay problem when held.

## Conclusion

Momentum-blend, held, is the strategy. Four refinements were tested against the
simple momentum baseline and all four were rejected for evidenced reasons:
scheduled rebalancing (no benefit, adds turnover), volatility-sizing (same
frontier, lower return), the momentum-reversal combo (underperformed), and
frequent-rebalancing reversal (gross edge exists but is cost-fatal). Simple
momentum survived every challenge. The intellectual result of the study is that
convergence: I did not merely pick momentum, I stress-tested it against every
plausible price-and-volume alternative the ruleset allows, and it won on
practical, cost-and-execution-aware grounds.

## Honest caveats

- All backtests are gross of transaction costs; the equity-curve magnitudes
  (thousands of percent over 10 years) are idealised and inflated by survivorship
  bias and frictionless reinvestment. The RELATIVE ordering of strategies is the
  finding, not the absolute levels.
- Long-only: the academic momentum-crash risk (Daniel-Moskowitz 2016) is driven
  by the short-loser leg, not held here, so the tail is milder in mechanism than
  the long-short literature, though concentration's tail is still real.
- The competition is a 5-week ranked-return race across thousands of teams, whose
  outright winner is variance-dominated (a concentrated bet that happened to pay
  off). None of these strategies meaningfully raises the odds of winning outright;
  the value is a well-understood, well-tested, defensible process aimed at a
  respectable finish, and the study itself.

## References

- Jegadeesh & Titman (1993, 2001) - cross-sectional momentum
- Carhart (1997) - momentum as a factor
- Asness, Moskowitz & Pedersen (2013) - momentum across markets
- Daniel & Moskowitz (2016) - momentum crashes and negative skew
- Barroso & Santa-Clara (2015) - volatility-managed momentum
- DeMiguel, Garlappi & Uppal (2009) - naive diversification vs optimisation