# Macro System: Trend + Carry + Vol (Proxy)

A simple monthly-rebalanced backtest using ETF proxies:
- Trend sleeve: time-series momentum across diversified ETFs
- Carry sleeve: DBV as FX carry proxy with a risk filter
- Vol sleeve: SVXY as short-vol proxy with a risk filter
- Defensive: SHY

## Setup
```bash
pip install yfinance pandas numpy matplotlib
python macro_system.py
clear
cat > README.md << 'EOF'
# Trading: Macro System Backtest

This project implements a simple systematic macro strategy combining:

- Trend (time-series momentum)
- Carry (FX carry proxy via DBV)
- Volatility income (short-vol proxy via SVXY)

The system rebalances monthly and uses real ETF data from Yahoo Finance.

---

## Strategies Used

### Trend
Follows momentum across:
- SPY (Equities)
- TLT (Bonds)
- GLD (Gold)
- DBC (Commodities)
- EFA (Global equities)

### Carry
Uses DBV as a proxy for currency carry trades.

### Volatility
Uses SVXY to capture volatility risk premium.

---

## Setup

Install required libraries:

pip install yfinance pandas numpy matplotlib

---

## Run Backtest

python macro_system.py

---

## Notes

- Uses real market data
- Simulated execution
- Monthly rebalancing
- Intended for research & learning
