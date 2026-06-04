# Quant Project

A personal project on developing financial models, portfolio optimisation, and trading strategy.

Specifically involves stochastic models for derivatives pricing, fund construction,
and systematic strategies. Following a structured curriculum from Black-Scholes
to rough volatility, with a multi-asset fund case study and backtested strategies.

## Setup 

Requires miniforge/mamba. To recreate the environment:

mamba env create -f environment.yml
conda activate quant

## Structure

- `models/` - stochastic models (BSM, Heston, SABR, jumps, rough vol)
- `pricing/` - Monte Carlo, Fourier, PDE pricing engines
- `calibration/` - surface calibrators
- `portfolio/` - portfolio construction methods
- `strategies/` - trading strategies
- `backtest/` - backtesting infrastructure
- `reporting/` - tear sheets, attribution
- `notebooks/` - phase-by-phase exploratory notebooks
- `data/` - price and option data (gitignored)