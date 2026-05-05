# app.py — Crypto Market & Liquidity Risk Analytics (Institutional-Grade)
# Université de Genève · MSc Finance · Global Asset Allocation · Spring 2026
# Author: Liyan Zeng
# ----------------------------------------------------------------
# Implements:
#   • Historical / Parametric / Cornish-Fisher VaR & Expected Shortfall
#   • Kupiec (1995) Proportion-of-Failures backtest with rolling window
#   • Component VaR via Euler decomposition (risk-budgeting view)
#   • Bootstrap 95% CI for portfolio VaR
#   • Bangia et al. (1999) / Almgren-Chriss (2000) Liquidity-Adjusted VaR
#   • Four named crash scenarios + custom stress designer
# ----------------------------------------------------------------
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto L-VaR Analytics", layout="wide",
                   page_icon="📊", initial_sidebar_state="expanded")

# === Institutional CSS ===
st.markdown("""<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    font-size: 0.76rem !important; color: #475569 !important;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] {
    font-size: 1.55rem !important; font-weight: 700; color: #0f172a !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { font-size: 0.94rem; font-weight: 500; padding: 10px 18px; }
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 100%);
    color: white; padding: 22px 28px; border-radius: 12px; margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.15);
}
.header-banner h1 { color: white; margin: 0; font-size: 1.65rem; font-weight: 700; }
.header-banner .subtitle { color: rgba(255,255,255,0.88); font-size: 0.95rem; margin-top: 4px; }
.header-banner .meta { color: rgba(255,255,255,0.65); font-size: 0.78rem; margin-top: 10px; }
</style>""", unsafe_allow_html=True)

# === Constants ===
DEFAULT_ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
TRADING_DAYS = 365  # crypto markets trade 24/7

import time

def _fetch_one(symbol, start, end, attempts=3, sleep=1.5):
    """Fetch a single ticker with retry. Yahoo Finance occasionally returns empty
    data on batch requests; per-symbol retries dramatically improve reliability."""
    for i in range(attempts):
        try:
            df = yf.download(symbol, start=start, end=end,
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass
        if i < attempts - 1:
            time.sleep(sleep)
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbols, start, end):
    # Step 1: try batch download (fast path)
    df = yf.download(list(symbols), start=start, end=end,
                     progress=False, auto_adjust=False, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
        volume_usd = df["Volume"].copy()  # Yahoo crypto Volume is already USD turnover
    elif df is not None and not df.empty:
        close = df[["Close"]].rename(columns={"Close": symbols[0]})
        volume_usd = df[["Volume"]].rename(columns={"Volume": symbols[0]})
    else:
        close = pd.DataFrame()
        volume_usd = pd.DataFrame()

    # Step 2: detect any symbols that came back empty and retry individually
    failed = [s for s in symbols if s not in close.columns or close[s].notna().sum() < 2]
    for s in failed:
        retry = _fetch_one(s, start, end)
        if retry is not None and not retry.empty:
            close[s] = retry["Close"]
            volume_usd[s] = retry["Volume"]  # already USD turnover for crypto

    close = close.ffill().dropna(how="all")
    volume_usd = volume_usd.ffill().dropna(how="all")
    return close, volume_usd

# === Risk metrics (defensive against empty input) ===
def log_returns(p): return np.log(p / p.shift(1)).dropna(how="all")

def _clean(s): return pd.Series(s).dropna()

def hist_var(r, c=0.95):
    r = _clean(r)
    if len(r) == 0: return float("nan")
    return float(-np.quantile(r, 1 - c))

def parametric_var(r, c=0.95):
    """Gaussian VaR (Risk-Metrics-style): VaR = -(mu + z_(1-c) * sigma)."""
    r = _clean(r)
    if len(r) < 2: return float("nan")
    return float(-(r.mean() + stats.norm.ppf(1 - c) * r.std()))

def cornish_fisher_var(r, c=0.95):
    """Cornish-Fisher (1937) VaR — corrects the Gaussian quantile for empirical skew & excess kurtosis (fat tails)."""
    r = _clean(r)
    if len(r) < 4: return float("nan")
    mu, sg = r.mean(), r.std()
    s = float(stats.skew(r)); k = float(stats.kurtosis(r))
    z = stats.norm.ppf(1 - c)
    z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*(s**2)/36
    return float(-(mu + z_cf * sg))

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
    return float(r.std() * np.sqrt(TRADING_DAYS))

def kupiec_pof(returns, var_func, c=0.95, window=250):
    """Kupiec (1995) POF backtest with rolling-window VaR forecasts.
    H0: empirical violation rate equals (1 - c). Reject if p-value < 5%."""
    r = _clean(returns); n = len(r) - window
    if n <= 50: return None
    viol = 0
    for i in range(window, len(r)):
        v = var_func(r.iloc[i-window:i], c)
        if r.iloc[i] < -v: viol += 1
    p = 1 - c; pi_hat = viol / n if n else 0
    if 0 < pi_hat < 1:
        lr = -2 * (np.log((1-p)**(n-viol) * p**viol)
                   - np.log((1-pi_hat)**(n-viol) * pi_hat**viol))
        p_val = 1 - stats.chi2.cdf(lr, df=1)
    else:
        lr, p_val = np.nan, np.nan
    return {"n": n, "violations": viol, "expected": n*p,
            "rate": pi_hat, "lr": lr, "p_value": p_val}

def component_var_normal(rets_df, w, c=0.95):
    """Euler decomposition of portfolio VaR under Gaussian assumption.
    Returns each asset's contribution to portfolio VaR in return units."""
    cov = rets_df.cov().values
    w = np.asarray(w, dtype=float)
    var_p = float(w @ cov @ w)
    sg = np.sqrt(var_p) if var_p > 0 else 0.0
    if sg == 0: return np.zeros_like(w)
    marginal = (cov @ w) / sg
    return -stats.norm.ppf(1 - c) * (w * marginal)

def bootstrap_var_ci(returns, c=0.95, n_boot=1000, seed=42):
    """Bootstrap 95% CI for the historical VaR statistic."""
    r = _clean(returns).values
    if len(r) < 30: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        boots.append(-np.quantile(sample, 1 - c))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# ============================================================
# Liquidity model — the core of the project
#   LiquidityCost = lambda * (Position / ADV) * sigma * Position
#   L-VaR         = Std VaR + Liquidity Cost
# ============================================================
def liquidity_cost(position_usd, adv_usd, sigma, lam=0.5):
    if adv_usd is None or adv_usd <= 0 or np.isnan(adv_usd):
        return float("inf")
    return lam * (position_usd / adv_usd) * sigma * position_usd

# === Sidebar (Module 1) ===
st.sidebar.markdown("### 📥 Portfolio Input")
preset = st.sidebar.selectbox("Preset", ["Custom", "Balanced", "BTC-heavy", "Altcoin-heavy"])
# Institutional-scale defaults so liquidity cost is visible against deep crypto markets
PRESETS = {
    "Balanced":      {"BTC-USD": 25_000_000, "ETH-USD": 20_000_000, "SOL-USD": 10_000_000, "BNB-USD": 10_000_000, "XRP-USD": 5_000_000,  "ADA-USD": 5_000_000},
    "BTC-heavy":     {"BTC-USD": 60_000_000, "ETH-USD": 20_000_000, "SOL-USD": 5_000_000,  "BNB-USD": 5_000_000,  "XRP-USD": 5_000_000,  "ADA-USD": 5_000_000},
    "Altcoin-heavy": {"BTC-USD": 5_000_000,  "ETH-USD": 10_000_000, "SOL-USD": 25_000_000, "BNB-USD": 15_000_000, "XRP-USD": 20_000_000, "ADA-USD": 25_000_000},
}
base = PRESETS.get(preset, {a: 10_000_000.0 for a in DEFAULT_ASSETS})

positions = {}
for a in DEFAULT_ASSETS:
    positions[a] = st.sidebar.number_input(f"{a} (USD)", min_value=0.0,
                                            value=float(base.get(a, 10_000_000.0)),
                                            step=500_000.0, format="%.0f")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk Parameters")
conf = st.sidebar.selectbox("Confidence level α", [0.95, 0.99], index=0)
lam = st.sidebar.slider("λ (market-impact coefficient)", 0.1, 2.0, 0.5, 0.1)
participation = st.sidebar.slider("Participation rate (% ADV/day)", 0.01, 1.0, 0.20, 0.01)
window = st.sidebar.selectbox("Rolling vol window (days)", [30, 60, 90], index=0)
adv_window = st.sidebar.slider("ADV window (days)", 7, 90, 30)
end_date = st.sidebar.date_input("End date", datetime.utcnow().date())
start_date = st.sidebar.date_input("Start date", end_date - timedelta(days=365 * 3))
run_backtest = st.sidebar.checkbox("Run Kupiec POF backtest (slower)", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Université de Genève · MSc Finance · Spring 2026")

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

# === Header banner ===
st.markdown(f"""<div class="header-banner">
<h1>📊 Crypto Market & Liquidity Risk Analytics</h1>
<div class="subtitle">Liquidity-Adjusted Value-at-Risk Dashboard for Institutional Digital-Asset Portfolios</div>
<div class="meta">Université de Genève · MSc Finance · Global Asset Allocation (Spring 2026) · Liyan Zeng</div>
</div>""", unsafe_allow_html=True)

# Pre-compute portfolio L-VaR for header KPIs
total_sv_pre = sum(positions[s] * hist_var(rets[s].dropna(), conf) for s in active)
total_lc_pre = sum(
    (lambda pos, adv, sg: lam * (pos/adv) * sg * pos if adv and adv > 0 else 0)(
        positions[s],
        float(vol_usd[s].dropna().tail(adv_window).mean()) if s in vol_usd.columns else 0,
        ann_vol(rets[s].dropna())
    )
    for s in active
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Portfolio AUM", f"${total_pos/1e6:,.1f}M")
k2.metric(f"Std VaR ({int(conf*100)}%)", f"${total_sv_pre/1e6:,.2f}M")
k3.metric("Liquidity Cost", f"${total_lc_pre/1e6:,.2f}M",
          delta=f"+{total_lc_pre/total_sv_pre*100:.0f}%" if total_sv_pre else None)
k4.metric("Liquidity-Adjusted VaR", f"${(total_sv_pre+total_lc_pre)/1e6:,.2f}M")
k5.metric("Annualized Vol", f"{ann_vol(port_ret)*100:.1f}%")

st.caption(":blue[**👈 Adjust positions and risk parameters in the sidebar (Module 1).**] All analytical modules below recompute live.")

tabs = st.tabs([
    "📈 Market Data",
    "📉 VaR & ES",
    "🧮 Component & Diversification",
    "💧 Liquidity Risk",
    "💥 Crash Scenarios",
    "ℹ️ Methodology & References",
])

# === Tab 1 — Market Data ===
with tabs[0]:
    st.subheader("Market Data Dashboard")

    norm = close[active] / close[active].iloc[0] * 100
    st.plotly_chart(px.line(norm, title="Normalized Prices (Base = 100)"),
                    use_container_width=True)

    fig_p = px.line(close[active], title="Raw Price Trends (log scale)")
    fig_p.update_yaxes(type="log")
    st.plotly_chart(fig_p, use_container_width=True)

    rv = rets[active].rolling(window).std() * np.sqrt(TRADING_DAYS)
    st.plotly_chart(px.line(rv, title=f"{window}-Day Rolling Annualized Volatility"),
                    use_container_width=True)

    nav = (1 + rets[active]).cumprod()
    dd = nav / nav.cummax() - 1
    st.plotly_chart(px.area(dd, title="Drawdown by Asset"), use_container_width=True)

    # NEW: Cross-asset correlation heatmap
    st.markdown("#### Cross-Asset Return Correlation")
    corr = rets[active].corr()
    fig_c = px.imshow(corr, text_auto=".2f", aspect="auto",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      title="Pearson Correlation of Daily Log Returns")
    st.plotly_chart(fig_c, use_container_width=True)
    st.caption("High pairwise correlations (typically 0.7–0.9 across crypto majors) imply limited "
               "diversification benefit and dominant common-factor exposure — a defining feature of the "
               "asset class compared to traditional multi-asset portfolios.")

# === Tab 2 — VaR & ES (3 methods + Kupiec backtest) ===
with tabs[1]:
    st.subheader("Value-at-Risk & Expected Shortfall")
    st.markdown("Three estimators are compared and validated with the **Kupiec (1995) Proportion-of-Failures backtest**.")

    rows = []
    for s in active:
        r = rets[s].dropna()
        rows.append({
            "Asset": s,
            "Position": positions[s],
            "Weight": positions[s] / total_pos,
            "Ann Vol": ann_vol(r),
            "Skew": float(stats.skew(r)),
            "Excess Kurt": float(stats.kurtosis(r)),
            "Hist VaR": hist_var(r, conf),
            "Param VaR": parametric_var(r, conf),
            "CF VaR": cornish_fisher_var(r, conf),
            "ES": expected_shortfall(r, conf),
            "Max DD": max_drawdown(close[s].dropna()),
        })
    df_v = pd.DataFrame(rows)
    st.dataframe(df_v.style.format({
        "Position":"${:,.0f}", "Weight":"{:.1%}", "Ann Vol":"{:.2%}",
        "Skew":"{:+.2f}", "Excess Kurt":"{:+.2f}",
        "Hist VaR":"{:.2%}","Param VaR":"{:.2%}","CF VaR":"{:.2%}",
        "ES":"{:.2%}","Max DD":"{:.2%}",
    }), use_container_width=True, hide_index=True)
    st.download_button("📥 Download asset-level VaR table (CSV)",
                       df_v.to_csv(index=False).encode(),
                       "asset_var.csv", "text/csv")

    # Portfolio return distribution + all VaR lines
    st.markdown("#### Portfolio Daily Return Distribution")
    hv = hist_var(port_ret, conf)
    pv = parametric_var(port_ret, conf)
    cf = cornish_fisher_var(port_ret, conf)
    es_p = expected_shortfall(port_ret, conf)
    boot_lo, boot_hi = bootstrap_var_ci(port_ret, conf)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_ret, nbinsx=80, name="Returns", marker_color="#94a3b8"))
    for label, val, color in [("Hist VaR", -hv, "#f97316"),
                              ("Param VaR", -pv, "#3b82f6"),
                              ("CF VaR", -cf, "#dc2626"),
                              ("ES", -es_p, "#7c3aed")]:
        fig.add_vline(x=val, line_color=color, line_width=2,
                      annotation_text=label, annotation_position="top")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Historical VaR", f"{hv*100:.2f}%")
    cB.metric("Parametric VaR", f"{pv*100:.2f}%")
    cC.metric("Cornish-Fisher VaR", f"{cf*100:.2f}%")
    cD.metric("Expected Shortfall", f"{es_p*100:.2f}%")
    if not np.isnan(boot_lo):
        st.caption(f"**Bootstrap 95% CI for Historical VaR**: [{boot_lo*100:.2f}%, {boot_hi*100:.2f}%]  "
                   f"— sampling uncertainty from 1,000 resamples of {len(port_ret)} daily returns.")

    # Kupiec POF backtest
    if run_backtest:
        st.markdown("#### Kupiec (1995) Proportion-of-Failures Backtest")
        st.caption("250-day rolling-window backtest. H₀: empirical violation rate equals (1 − α). "
                   "p-value > 5% means we cannot reject H₀ — the model is statistically valid.")
        bt_rows = []
        with st.spinner("Running rolling-window backtests…"):
            for name, fn in [("Historical", hist_var),
                             ("Parametric (Normal)", parametric_var),
                             ("Cornish-Fisher", cornish_fisher_var)]:
                res = kupiec_pof(port_ret, fn, conf, 250)
                if res:
                    pass_ = res["p_value"] > 0.05 if not np.isnan(res["p_value"]) else False
                    bt_rows.append({
                        "Method": name,
                        "N (test)": res["n"],
                        "Violations": res["violations"],
                        "Expected": f"{res['expected']:.1f}",
                        "Empirical Rate": f"{res['rate']*100:.2f}%",
                        "Expected Rate": f"{(1-conf)*100:.2f}%",
                        "LR stat": f"{res['lr']:.3f}" if not np.isnan(res["lr"]) else "n/a",
                        "p-value": f"{res['p_value']:.3f}" if not np.isnan(res["p_value"]) else "n/a",
                        "Verdict": "✅ Pass" if pass_ else "❌ Reject",
                    })
        if bt_rows:
            st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)
            st.caption("Under Basel III's traffic-light framework, fewer-than-expected violations are conservative; "
                       "more-than-expected violations indicate the model under-estimates tail risk.")

# === Tab 3 — Component VaR & Diversification ===
with tabs[2]:
    st.subheader("Component VaR & Diversification Analysis")
    st.markdown(r"""
**Euler decomposition** of portfolio VaR under the Gaussian assumption:
$$
\text{VaR}_p = \sum_i w_i \cdot \frac{\partial \text{VaR}_p}{\partial w_i}, \qquad
\text{Component VaR}_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p} \cdot z_{1-\alpha}
$$
This answers **which assets actually drive portfolio risk** — often very different from their capital weights.
""")
    contrib = component_var_normal(rets[active], weights, conf)
    contrib_dollar = contrib * total_pos
    standalone_dollar = np.array([positions[s] * hist_var(rets[s].dropna(), conf) for s in active])
    risk_share = (contrib / contrib.sum() * 100) if contrib.sum() else np.zeros_like(contrib)
    df_c = pd.DataFrame({
        "Asset": active,
        "Weight (%)": weights * 100,
        "Standalone VaR ($)": standalone_dollar,
        "Component VaR ($)": contrib_dollar,
        "Risk Share (%)": risk_share,
    })
    st.dataframe(df_c.style.format({
        "Weight (%)": "{:.1f}%", "Standalone VaR ($)": "${:,.0f}",
        "Component VaR ($)": "${:,.0f}", "Risk Share (%)": "{:.1f}%",
    }), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=df_c["Asset"], y=df_c["Weight (%)"], marker_color="#3b82f6"))
        fig.update_layout(title="Capital Allocation (Weight %)", yaxis_title="%", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=df_c["Asset"], y=df_c["Risk Share (%)"], marker_color="#dc2626"))
        fig.update_layout(title="Risk Allocation (Component VaR %)", yaxis_title="%", height=340)
        st.plotly_chart(fig, use_container_width=True)

    sum_standalone = float(standalone_dollar.sum())
    portfolio_var_usd = total_pos * hist_var(port_ret, conf)
    div_benefit = (sum_standalone - portfolio_var_usd) / sum_standalone if sum_standalone else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Σ Standalone VaR", f"${sum_standalone/1e6:,.2f}M")
    m2.metric("Portfolio VaR", f"${portfolio_var_usd/1e6:,.2f}M")
    m3.metric("Diversification Benefit", f"{div_benefit*100:.1f}%")
    st.caption("Diversification benefit = 1 − Portfolio VaR ÷ Σ Standalone VaR. Crypto's high pairwise correlations "
               "keep this typically modest (10–25%) — many assets ≠ diversified portfolio.")

# === Tab 4 — Liquidity Risk ===
with tabs[3]:
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

# === Tab 5 — Crash Scenarios ===
with tabs[4]:
    st.subheader("💥 Crash Scenario Stress Testing")
    st.markdown("Each scenario applies a **price shock** (mark-to-market loss) plus a **liquidity stress** "
                "(reduced ADV, amplified λ). Total stressed loss = |MTM P&L| + Stressed Liquidity Cost.")
    SCENARIOS = {
        "Crypto Winter":        {"BTC-USD": -0.50, "ETH-USD": -0.60, "_alt": -0.75, "adv": 0.40, "lam": 1.5,
                                  "desc": "Multi-month structural bear market (cf. 2018, 2022). Sentiment collapse compounded by macro tightening."},
        "Exchange Collapse":    {"BTC-USD": -0.25, "ETH-USD": -0.30, "_alt": -0.50, "adv": 0.20, "lam": 3.0,
                                  "desc": "Major venue insolvency (cf. FTX, November 2022). Severe liquidity withdrawal and custody freeze."},
        "Stablecoin Panic":     {"BTC-USD": -0.15, "ETH-USD": -0.20, "_alt": -0.35, "adv": 0.50, "lam": 2.0,
                                  "desc": "USDT/USDC depeg episode (cf. UST/Luna, May 2022). Cross-asset basis blowup."},
        "Regulatory Crackdown": {"BTC-USD": -0.20, "ETH-USD": -0.25, "_alt": -0.40, "adv": 0.60, "lam": 1.8,
                                  "desc": "Coordinated G7 enforcement action (cf. PRC ban, 2021). Forced exchange off-boarding."},
    }
    sc_name = st.selectbox("Scenario", list(SCENARIOS.keys()) + ["Custom"])
    if sc_name != "Custom":
        st.info(f"**{sc_name}** — {SCENARIOS[sc_name]['desc']}")
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

    # Total scenario loss = |MTM loss from price shock| + Stressed Liquidity Cost
    rows = []; sc_pnl = 0.0; sc_lc = 0.0
    for s in active:
        pos = positions[s]
        r = rets[s].dropna()
        adv_stressed = float(vol_usd[s].dropna().tail(adv_window).mean()) * adv_mult
        sigma = ann_vol(r)
        lc = liquidity_cost(pos, adv_stressed, sigma, lam * lam_mult)
        pnl = pos * shocks[s]
        sc_pnl += pnl; sc_lc += lc
        rows.append({"Asset": s, "Shock": shocks[s], "MTM P&L": pnl,
                     "Liq Cost (stressed)": lc,
                     "Total Loss": abs(pnl) + lc})

    realized_loss = abs(sc_pnl)
    total_stressed_loss = realized_loss + sc_lc
    base_lvar = total_sv + total_lc

    a, b, c, d = st.columns(4)
    a.metric("Scenario P&L (MTM)", f"${sc_pnl:,.0f}")
    b.metric("Realized Loss", f"${realized_loss:,.0f}")
    c.metric("Liquidity Cost (stressed)", f"${sc_lc:,.0f}")
    delta_txt = f"{(total_stressed_loss/base_lvar - 1)*100:+.0f}% vs baseline L-VaR" if base_lvar else None
    d.metric("Total Stressed Loss", f"${total_stressed_loss:,.0f}", delta=delta_txt)

    df_sc = pd.DataFrame(rows)
    st.dataframe(df_sc.style.format({
        "Shock": "{:.0%}", "MTM P&L": "${:,.0f}",
        "Liq Cost (stressed)": "${:,.0f}", "Total Loss": "${:,.0f}",
    }), use_container_width=True, hide_index=True)

    fig = go.Figure(go.Waterfall(
        x=["Baseline L-VaR", "+ MTM shock loss", "+ Extra liquidity stress", "Total Stressed Loss"],
        y=[base_lvar, realized_loss - total_sv, sc_lc - total_lc, total_stressed_loss],
        measure=["absolute", "relative", "relative", "total"],
    ))
    fig.update_layout(title=f"Scenario Waterfall — Baseline → {sc_name}")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Under **{sc_name}**: ${realized_loss:,.0f} mark-to-market loss from the price shock "
        f"+ ${sc_lc:,.0f} extra to liquidate in a thinner market "
        f"(ADV × {adv_mult:.0%}, λ × {lam_mult:.1f}) = **${total_stressed_loss:,.0f} total**."
    )

# === Tab 6 — Methodology & References ===
with tabs[5]:
    st.subheader("Methodology & Academic References")
    st.markdown(r"""
### 1. Returns
Daily continuously-compounded log returns from Yahoo Finance close prices: $r_t = \ln(P_t / P_{t-1})$. Crypto markets trade 24/7, so we annualize volatility with $\sqrt{365}$ rather than $\sqrt{252}$.

### 2. Three Value-at-Risk Estimators (compared in Tab 2)
- **Historical Simulation**: $\text{VaR}_\alpha = -Q_{1-\alpha}(\{r_t\})$ — non-parametric, fully data-driven.
- **Parametric (Gaussian)**: $\text{VaR}_\alpha = -(\mu + z_{1-\alpha}\sigma)$ — closed-form but assumes normality.
- **Cornish-Fisher (1937)**: corrects the Gaussian quantile for empirical skewness $s$ and excess kurtosis $k$:
$$
z_{CF} = z + \tfrac{(z^2-1)s}{6} + \tfrac{(z^3-3z)k}{24} - \tfrac{(2z^3-5z)s^2}{36}
$$
Captures crypto's well-documented **fat left tails** while remaining parametric. Favoured in practice when historical samples are short.

### 3. Expected Shortfall (Conditional VaR)
$$\text{ES}_\alpha = -\mathbb{E}[\,r \mid r \le -\text{VaR}_\alpha\,]$$
A **coherent** risk measure (Artzner et al., 1999) — satisfies sub-additivity, unlike VaR. Adopted by Basel III's FRTB as the principal market-risk metric.

### 4. Backtesting — Kupiec (1995) Proportion-of-Failures
Likelihood-ratio test of unconditional coverage. Under H₀, the violation count follows a binomial distribution with parameter $1-\alpha$. We reject if
$$
LR_{POF} = -2 \ln\!\left[\frac{(1-p)^{n-x}\,p^x}{(1-\hat\pi)^{n-x}\,\hat\pi^x}\right] > \chi^2_1(0.95) = 3.84.
$$
Implemented with a 250-day rolling forecast window.

### 5. Component VaR (Euler Decomposition)
Under Gaussian assumptions, portfolio VaR is **homogeneous of degree 1** in weights, so by Euler's theorem:
$$
\text{VaR}_p = \sum_i w_i \cdot \frac{\partial \text{VaR}_p}{\partial w_i}, \qquad
\text{Component VaR}_i = -z_{1-\alpha} \cdot \frac{w_i\,(\Sigma w)_i}{\sigma_p}.
$$
Surfaces **risk concentration** independent of capital allocation.

### 6. Liquidity-Adjusted VaR — Bangia et al. (1999), Almgren-Chriss (2000)
The project's flagship contribution. Standard VaR assumes positions can be exited at the prevailing mid-price; for institutional-scale crypto positions this is unrealistic. We add a closed-form **endogenous liquidity premium**:
$$
\text{Liquidity Cost}_i = \lambda \cdot \frac{P_i}{\text{ADV}_i} \cdot \sigma_i \cdot P_i, \qquad
\text{L-VaR}_i = \text{VaR}_i + \text{Liquidity Cost}_i,
$$
where $P_i$ is position size in USD, $\text{ADV}_i$ is the trailing 30-day average daily volume, $\sigma_i$ is annualized volatility, and $\lambda \in [0.1, 2.0]$ is the **market-impact coefficient** (a Kyle-1985-style price-pressure parameter).

The formulation collapses to the **Almgren-Chriss linear temporary-impact model** when $\lambda$ is interpreted as twice the impact coefficient.

### 7. Stress Testing
Each scenario applies (a) a price shock vector $\{\delta_i\}$, (b) an ADV haircut $a \in (0,1]$ proxying market thinning, and (c) a $\lambda$ multiplier $\mu_\lambda \ge 1$ proxying impact amplification. Total stressed loss:
$$
\mathcal{L}_{stressed} = \Big|\sum_i P_i \delta_i\Big| + \sum_i \mu_\lambda \lambda \cdot \frac{P_i}{a\,\text{ADV}_i} \cdot \sigma_i \cdot P_i.
$$

---

### References
1. **Bangia, A., Diebold, F. X., Schuermann, T., & Stroughair, J.** (1999). *Modeling liquidity risk, with implications for traditional market risk measurement and management*. The Wharton Financial Institutions Center, Working Paper 99-06.
2. **Almgren, R., & Chriss, N.** (2000). *Optimal execution of portfolio transactions*. Journal of Risk, 3(2), 5–39.
3. **Kupiec, P. H.** (1995). *Techniques for verifying the accuracy of risk measurement models*. Journal of Derivatives, 3(2), 73–84.
4. **Artzner, P., Delbaen, F., Eber, J. M., & Heath, D.** (1999). *Coherent measures of risk*. Mathematical Finance, 9(3), 203–228.
5. **Cornish, E. A., & Fisher, R. A.** (1937). *Moments and cumulants in the specification of distributions*. Revue de l'Institut International de Statistique, 5(4), 307–320.
6. **Kyle, A. S.** (1985). *Continuous auctions and insider trading*. Econometrica, 53(6), 1315–1335.
7. **Brunnermeier, M. K., & Pedersen, L. H.** (2009). *Market liquidity and funding liquidity*. Review of Financial Studies, 22(6), 2201–2238.
8. **Basel Committee on Banking Supervision** (2019). *Minimum capital requirements for market risk*. BCBS d457.

### Limitations & Caveats
- **Linear** market-impact model; reality is concave for very large orders (Almgren et al., 2005).
- Historical / Cornish-Fisher VaR assume the past distribution repeats — supplement with stress tests.
- Cross-asset liquidity correlations ignored; in panics they spike together (Brunnermeier & Pedersen, 2009).
- ADV proxied by 30-day average dollar volume from Yahoo Finance; institutional flows often clear off-screen (OTC / RFQ).
- Single-day horizon; multi-day VaR via $\sqrt{T}$ scaling is fragile under volatility clustering and crisis-time autocorrelation.
    """)

st.markdown("---")
st.caption("Built with Streamlit · Data: Yahoo Finance via yfinance · © 2026 Liyan Zeng · Université de Genève")
