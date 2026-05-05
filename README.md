# Crypto Market & Liquidity Risk Dashboard

A Streamlit dashboard that monitors **market risk** and **liquidity risk** for crypto
portfolios. Designed for crypto exchanges, funds, and institutional digital-asset holders.

## Research questions
1. How much would the portfolio lose in a market crash? → VaR / ES / Max Drawdown.
2. How much **extra loss** comes from liquidity drying up? → Liquidity Cost / **L-VaR**.

## Modules
1. **Portfolio Input** — sidebar; positions in USD per asset + risk parameters.
2. **Market Data** — prices, normalized prices, returns, rolling vol, drawdowns.
3. **VaR & ES** — historical method at 95% / 99% with full return distribution.
4. **Liquidity Risk** — Liq Ratio, Days-to-Liquidate, Liquidity Cost, **L-VaR**, λ sensitivity.
5. **Crash Scenarios** — Crypto Winter, Exchange Collapse, Stablecoin Panic, Regulatory, Custom.

## Liquidity model
