# app.py — Crypto Market & Liquidity Risk Dashboard
# Run: streamlit run app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Risk Dashboard", layout="wide", page_icon="📊")

# ============================================================
# Data layer
# ============================================================
DEFAULT_ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"]

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbols, start, end):
    df = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
        volume_usd = (df["Volume"] * df["Close"]).copy()
    else:
        # single asset case
        close = df[["Close"]].rename(columns={"Close": symbols[0]})
        volume_usd = (df["Volume"] * df["Close"]).to_frame(symbols[0])
    close = close.ffill().dropna(how="all")
    volume_usd = volume_usd.ffill().dropna(how="all")
    return close, volume_usd

# ============================================================
# Risk metrics  (defensive: return NaN on empty input)
# ============================================================
def log_returns(p): return np.log(p / p.shift(1)).dropna(how="all")

def _clean(s):
    s = pd.Series(s).dropna()
    return s

def hist_var(r, c=0.95):
    r = _clean(r)
    if len(r) == 0: return float("nan")
    return float(-np.quantile(r, 1 - c))

def expected_shortfall(r, c=0.95):
    r = _clean(r)
    if len(r) == 0: return float("nan")
    v = hist_var(r, c); tail = r[r <= -v]
    return float(-tail.mean()) if len(tail) else v

def max_drawdown(p):
    p = _clean(p)
    if len(p) == 0: return float("nan")
    nav = p / p.iloc[0]; dd = nav / nav.cummax() - 1
    return float(dd.min())

def ann_vol(r):
    r = _clean(r)
    if len(r) < 2: return float("nan")
    return float(r.std() * np.sqrt(365))

# ============================================================
# Liquidity model — the core of the project
#   LiquidityCost = lambda * (Position / ADV) * sigma * Position
#   L-VaR         = Std VaR + Liquidity Cost
# ============================================================
def liquidity_cost(position_usd, adv_usd, sigma, lam=0.5):
    if adv_usd is None or adv_usd <= 0 or np.isnan(adv_usd):
        return float("inf")
    return lam * (position_usd / adv_usd) * sigma * position_usd

# ============================================================
# Sidebar — Module 1: Portfolio Input + parameters
# ============================================================
st.sidebar.title("📥 Module 1 · Portfolio Input")
preset = st.sidebar.selectbox("Preset", ["Custom", "Balanced", "BTC-heavy", "Altcoin-heavy"])
PRESETS = {
    "Balanced":      {"BTC-USD": 25000, "ETH-USD": 20000, "SOL-USD": 10000, "BNB-USD": 10000, "XRP-USD": 5000,  "ADA-USD": 5000},
    "BTC-heavy":     {"BTC-USD": 60000, "ETH-USD": 20000, "SOL-USD": 5000,  "BNB-USD": 5000,  "XRP-USD": 5000,  "ADA-USD": 5000},
    "Altcoin-heavy": {"BTC-USD": 5000,  "ETH-USD": 10000, "SOL-USD": 25000, "BNB-USD": 15000, "XRP-USD": 20000, "ADA-USD": 25000},
}
base = PRESETS.get(preset, {a: 10000.0 for a in DEFAULT_ASSETS})

positions = {}
for a in DEFAULT_ASSETS:
    positions[a] = st.sidebar.number_input(f"{a} (USD)", min_value=0.0,
                                            value=float(base.get(a, 10000.0)), step=500.0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Risk Parameters")
conf = st.sidebar.selectbox("Confidence level", [0.95, 0.99], index=0)
lam = st.sidebar.slider("λ (market-impact coefficient)", 0.1, 2.0, 0.5, 0.1)
participation = st.sidebar.slider("Participation rate (% of ADV/day)", 0.01, 1.0, 0.20, 0.01)
window = st.sidebar.selectbox("Rolling vol window (days)", [30, 60, 90], index=0)
adv_window = st.sidebar.slider("ADV window (days)", 7, 90, 30)
end_date = st.sidebar.date_input("End date", datetime.utcnow().date())
start_date = st.sidebar.date_input("Start date", end_date - timedelta(days=365 * 3))

# ============================================================
# Load market data
# ============================================================
active = [a for a, v in positions.items() if v > 0]
if not active:
    st.warning("Enter at least one positive position in the sidebar to begin.")
    st.stop()

with st.spinner("Loading market data from Yahoo Finance…"):
    close, vol_usd = load_data(active, start_date, end_date)

if close.empty:
    st.error("No data returned. Try a different date range.")
    st.stop()

# Drop assets that returned no usable data (yfinance can fail per-symbol)
missing = [a for a in active if a not in close.columns or close[a].notna().sum() < 2]
if missing:
    st.warning(f"No data returned for: {', '.join(missing)}. Skipping these assets.")
active = [a for a in active if a not in missing]
if not active:
    st.error("None of the selected assets returned usable data. Try a different date range or fewer assets.")
    st.stop()

rets = log_returns(close[active])
total_pos = sum(positions[a] for a in active)
weights = np.array([positions[a] / total_pos for a in active])
port_ret = (rets[active] * weights).sum(axis=1)

# ============================================================
# Header KPIs
# ============================================================
st.title("📊 Crypto Market & Liquidity Risk Dashboard")
st.caption("Designed for crypto exchanges, funds, and institutional digital-asset holders")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Portfolio Value", f"${total_pos:,.0f}")
k2.metric(f"Portfolio VaR ({int(conf*100)}%)", f"{hist_var(port_ret, conf)*100:.2f}%")
k3.metric(f"Portfolio ES ({int(conf*100)}%)", f"{expected_shortfall(port_ret, conf)*100:.2f}%")
k4.metric("Portfolio Max Drawdown", f"{max_drawdown((1 + port_ret).cumprod())*100:.2f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 2 · Market Data",
    "📉 3 · VaR & ES",
    "💧 4 · Liquidity Risk",
    "💥 5 · Crash Scenarios",
    "ℹ️ About / Methodology",
])

# ============================================================
# Module 2 — Market Data Dashboard
# ============================================================
with tab1:
    st.subheader("Market Data Dashboard")

    norm = close[active] / close[active].iloc[0] * 100
    st.plotly_chart(px.line(norm, title="Normalized Prices (Base = 100)",
                            labels={"value": "Index", "index": "Date"}),
                    use_container_width=True)

    fig_p = px.line(close[active], title="Raw Price Trends (log scale)")
    fig_p.update_yaxes(type="log")
    st.plotly_chart(fig_p, use_container_width=True)

    st.plotly_chart(px.line(rets[active], title="Daily Log Returns"),
                    use_container_width=True)

    rv = rets[active].rolling(window).std() * np.sqrt(365)
    st.plotly_chart(px.line(rv, title=f"{window}-Day Rolling Annualized Volatility"),
                    use_container_width=True)

    nav = (1 + rets[active]).cumprod()
    dd = nav / nav.cummax() - 1
    st.plotly_chart(px.area(dd, title="Drawdown by Asset"),
                    use_container_width=True)

# ============================================================
# Module 3 — VaR & Expected Shortfall
# ============================================================
with tab2:
    st.subheader("VaR & Expected Shortfall")

    rows = []
    for s in active:
        r = rets[s].dropna()
        rows.append({
            "Asset": s,
            "Position": positions[s],
            "Weight": positions[s] / total_pos,
            "Ann Vol": ann_vol(r),
            "VaR 95%": hist_var(r, 0.95),
            "VaR 99%": hist_var(r, 0.99),
            "ES 95%":  expected_shortfall(r, 0.95),
            "ES 99%":  expected_shortfall(r, 0.99),
            "Max DD":  max_drawdown(close[s].dropna()),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "Position": "${:,.0f}", "Weight": "{:.1%}", "Ann Vol": "{:.2%}",
            "VaR 95%": "{:.2%}", "VaR 99%": "{:.2%}",
            "ES 95%":  "{:.2%}", "ES 99%":  "{:.2%}", "Max DD":  "{:.2%}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Portfolio daily-return distribution**")
    var_x = -hist_var(port_ret, conf)
    es_x  = -expected_shortfall(port_ret, conf)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_ret, nbinsx=80, name="Daily returns"))
    fig.add_vline(x=var_x, line_color="orange",
                  annotation_text=f"VaR {int(conf*100)}%", annotation_position="top")
    fig.add_vline(x=es_x, line_color="red",
                  annotation_text=f"ES {int(conf*100)}%", annotation_position="bottom")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Module 4 — Liquidity Risk (THE CORE)
# ============================================================
with tab3:
    st.subheader("💧 Liquidity Risk — Liquidity-Adjusted VaR")
    st.markdown(r"""
**Liquidity-cost model** (the soul of this project):
$$
\text{Liquidity Cost} = \lambda \times \frac{\text{Position}}{\text{ADV}} \times \sigma \times \text{Position}
$$
$$
\text{L-VaR} = \text{Std VaR} + \text{Liquidity Cost}
$$
Two assets with similar volatility can have very different total risk if one is hard to liquidate.
    """)

    rows = []
    total_sv = 0.0; total_lc = 0.0
    for s in active:
        r = rets[s].dropna()
        adv_s = float(vol_usd[s].dropna().tail(adv_window).mean())
        sigma = ann_vol(r)
        pos = positions[s]
        sv_usd = pos * hist_var(r, conf)
        lc = liquidity_cost(pos, adv_s, sigma, lam)
        rows.append({
            "Asset": s,
            "Position": pos,
            "ADV (USD)": adv_s,
            "Liq Ratio": (pos / adv_s) if adv_s else np.nan,
            "Days to Liquidate": (pos / (participation * adv_s)) if adv_s else np.nan,
            "Std VaR ($)": sv_usd,
            "Liquidity Cost ($)": lc,
            "L-VaR ($)": sv_usd + lc,
            "L-VaR / Std VaR": (sv_usd + lc) / sv_usd if sv_usd else np.nan,
        })
        total_sv += sv_usd; total_lc += lc
    df_liq = pd.DataFrame(rows)

    a, b, c = st.columns(3)
    a.metric("Standard VaR", f"${total_sv:,.0f}")
    delta_pct = f"+{total_lc/total_sv*100:.1f}%" if total_sv else None
    b.metric("Liquidity Cost (extra loss)", f"${total_lc:,.0f}", delta=delta_pct)
    c.metric("Liquidity-Adjusted VaR", f"${total_sv + total_lc:,.0f}")

    st.dataframe(
        df_liq.style.format({
            "Position": "${:,.0f}", "ADV (USD)": "${:,.0f}",
            "Liq Ratio": "{:.4f}", "Days to Liquidate": "{:.2f}",
            "Std VaR ($)": "${:,.0f}", "Liquidity Cost ($)": "${:,.0f}",
            "L-VaR ($)": "${:,.0f}", "L-VaR / Std VaR": "{:.2f}x",
        }),
        use_container_width=True, hide_index=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_liq["Asset"], y=df_liq["Std VaR ($)"], name="Std VaR"))
    fig.add_trace(go.Bar(x=df_liq["Asset"], y=df_liq["Liquidity Cost ($)"], name="Liquidity Cost"))
    fig.update_layout(barmode="stack", title="Std VaR vs Liquidity Cost (stacked, by asset)")
    st.plotly_chart(fig, use_container_width=True)

    # λ sensitivity
    lams = np.linspace(0.1, 2.0, 25)
    lvars = []
    advs = {s: float(vol_usd[s].dropna().tail(adv_window).mean()) for s in active}
    sigmas = {s: ann_vol(rets[s].dropna()) for s in active}
    for l in lams:
        tlc = sum(liquidity_cost(positions[s], advs[s], sigmas[s], l) for s in active)
        lvars.append(total_sv + tlc)
    fig_l = px.line(x=lams, y=lvars,
                    labels={"x": "λ", "y": "Portfolio L-VaR ($)"},
                    title="λ Sensitivity — Portfolio L-VaR vs Market-Impact Coefficient")
    st.plotly_chart(fig_l, use_container_width=True)

    # Auto-insight
    if total_sv > 0 and df_liq["Days to Liquidate"].notna().any():
        worst = df_liq.loc[df_liq["Days to Liquidate"].idxmax()]
        st.info(
            f"💡 At a {participation*100:.0f}% participation rate, **{worst['Asset']}** would take "
            f"**{worst['Days to Liquidate']:.1f} days** to liquidate. "
            f"Liquidity adds **{total_lc/total_sv*100:.1f}%** on top of standard VaR for this portfolio."
        )

# ============================================================
# Module 5 — Crash Scenarios
# ============================================================
with tab4:
    st.subheader("💥 Crash Scenario Stress Testing")
    SCENARIOS = {
        "Crypto Winter":        {"BTC-USD": -0.50, "ETH-USD": -0.60, "_alt": -0.75, "adv": 0.40, "lam": 1.5},
        "Exchange Collapse":    {"BTC-USD": -0.25, "ETH-USD": -0.30, "_alt": -0.50, "adv": 0.20, "lam": 3.0},
        "Stablecoin Panic":     {"BTC-USD": -0.15, "ETH-USD": -0.20, "_alt": -0.35, "adv": 0.50, "lam": 2.0},
        "Regulatory Crackdown": {"BTC-USD": -0.20, "ETH-USD": -0.25, "_alt": -0.40, "adv": 0.60, "lam": 1.8},
    }
    sc_name = st.selectbox("Scenario", list(SCENARIOS.keys()) + ["Custom"])
    if sc_name == "Custom":
        st.markdown("**Custom shocks (per asset, % return)**")
        cols = st.columns(len(active))
        shocks = {}
        for i, s in enumerate(active):
            shocks[s] = cols[i].slider(s, -0.95, 0.50, -0.30, 0.05)
        adv_mult = st.slider("ADV multiplier (lower = thinner market)", 0.05, 1.0, 0.50)
        lam_mult = st.slider("λ multiplier (higher = more market impact)", 1.0, 5.0, 2.0)
    else:
        sc = SCENARIOS[sc_name]
        shocks = {s: sc.get(s, sc["_alt"]) for s in active}
        adv_mult = sc["adv"]; lam_mult = sc["lam"]

    rows = []; sc_pnl = 0.0; sc_sv = 0.0; sc_lc = 0.0
    for s in active:
        pos = positions[s]
        r = rets[s].dropna()
        adv_s = float(vol_usd[s].dropna().tail(adv_window).mean()) * adv_mult
        sigma = ann_vol(r)
        sv = pos * hist_var(r, conf)
        lc = liquidity_cost(pos, adv_s, sigma, lam * lam_mult)
        pnl = pos * shocks[s]
        sc_pnl += pnl; sc_sv += sv; sc_lc += lc
        rows.append({"Asset": s, "Shock": shocks[s], "P&L": pnl,
                     "Std VaR": sv, "Liquidity Cost": lc, "L-VaR": sv + lc})

    a, b, c, d = st.columns(4)
    a.metric("Scenario P&L", f"${sc_pnl:,.0f}")
    b.metric("Std VaR (stressed)", f"${sc_sv:,.0f}")
    c.metric("Liquidity Cost (stressed)", f"${sc_lc:,.0f}")
    d.metric("L-VaR (stressed)", f"${sc_sv + sc_lc:,.0f}")

    df_sc = pd.DataFrame(rows)
    st.dataframe(df_sc.style.format({
        "Shock": "{:.0%}", "P&L": "${:,.0f}", "Std VaR": "${:,.0f}",
        "Liquidity Cost": "${:,.0f}", "L-VaR": "${:,.0f}",
    }), use_container_width=True, hide_index=True)

    base_lvar = total_sv + total_lc
    stressed_lvar = sc_sv + sc_lc
    fig = go.Figure(go.Waterfall(
        x=["Baseline L-VaR", "Δ Std VaR", "Δ Liquidity Cost", "Stressed L-VaR"],
        y=[base_lvar, sc_sv - total_sv, sc_lc - total_lc, stressed_lvar],
        measure=["absolute", "relative", "relative", "total"],
    ))
    fig.update_layout(title=f"Scenario Waterfall — Baseline → {sc_name}")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# About / Methodology
# ============================================================
with tab5:
    st.markdown(r"""
### Why Liquidity-Adjusted VaR matters
Standard VaR assumes positions can be exited at observed market prices. In reality, large
positions in thinly-traded crypto cannot be unwound without market impact. This dashboard
quantifies that **extra loss** with a transparent, parameterised model:

$$
\text{Liquidity Cost} = \lambda \times \frac{\text{Position}}{\text{ADV}} \times \sigma \times \text{Position}
$$

Two assets with nearly identical volatility can have very different total risk if their
average daily volume differs. Try moving allocation from BTC to SOL or ADA in the sidebar
and watch the **L-VaR / Std VaR** ratio in Module 4.

### Methodology in one page
- **Returns**: daily log returns from Yahoo Finance close prices.
- **VaR**: historical method, 95% / 99%.
- **ES**: average loss conditional on losses worse than VaR.
- **Max Drawdown**: peak-to-trough on the cumulative NAV.
- **Volatility**: rolling std × √365 (crypto trades 24/7).
- **ADV**: trailing N-day average of `Volume × Close` (USD).
- **Days to Liquidate**: Position / (participation × ADV).
- **L-VaR**: closed-form additive model above.

### Limitations
- Linear market-impact model; reality is concave for very large orders.
- Historical VaR assumes the past distribution repeats; supplement with stress tests.
- Cross-asset liquidity correlations are ignored; in panics they spike together.
    """)
