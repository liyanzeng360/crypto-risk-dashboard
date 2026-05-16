"""
Crypto Market & Liquidity Risk Dashboard
Université de Genève · Quantitative Risk Management (Spring 26) · Group 3
Streamlit deployment
"""

import time, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ──────────────── Constants ────────────────
TRADING_DAYS      = 365
BRAND             = "#0B3D91"
DEFAULT_TICKERS   = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"]
DEFAULT_POSITIONS = {"BTC":5_000_000,"ETH":3_000_000,"SOL":1_000_000,
                      "BNB":800_000,"XRP":200_000}
LIQ_PARTICIPATION = 0.25

SCENARIOS = {
    "Crypto Winter (2022-style)": {
        "shocks":{"BTC":-0.50,"ETH":-0.60,"SOL":-0.75,"BNB":-0.55,"XRP":-0.55},
        "adv_mult":0.40,"lambda_mult":1.5,"vol_mult":1.5,
        "description":"Prolonged bear market: large drops, volumes contract ~60%."},
    "Exchange Collapse (FTX-style)": {
        "shocks":{"BTC":-0.25,"ETH":-0.30,"SOL":-0.50,"BNB":-0.55,"XRP":-0.30},
        "adv_mult":0.20,"lambda_mult":3.0,"vol_mult":2.5,
        "description":"Sudden venue failure, liquidity evaporates."},
    "Stablecoin Panic": {
        "shocks":{"BTC":-0.15,"ETH":-0.20,"SOL":-0.35,"BNB":-0.25,"XRP":-0.20},
        "adv_mult":0.50,"lambda_mult":2.0,"vol_mult":1.8,
        "description":"Loss of confidence in USD stablecoin."},
    "Regulatory Shock": {
        "shocks":{"BTC":-0.20,"ETH":-0.25,"SOL":-0.40,"BNB":-0.45,"XRP":-0.50},
        "adv_mult":0.60,"lambda_mult":1.8,"vol_mult":2.2,
        "description":"Major-jurisdiction enforcement action."},
}

# Risk Regime (Fix #2: 6 levels)
REGIME_RULES = [
    (-0.05, "Crisis",   "#7b241c"),
    (-0.03, "High Risk","#c0392b"),
    (-0.015,"Elevated", "#e67e22"),
    ( 0.00, "Moderate", "#f0c040"),
    ( 0.015,"Stable",   "#27ae60"),
]
REGIME_BULL = ("Bull", "#1a5276")


# ──────────────── Helper Functions ────────────────
def classify_regime(ret):
    if pd.isna(ret): return "N/A", "#999"
    for threshold, label, color in REGIME_RULES:
        if ret < threshold: return label, color
    return REGIME_BULL

def _fetch_one(symbol, start, end, attempts=3, sleep=1.5):
    for k in range(attempts):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(sleep*(k+1))
    return None

@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_features(tickers, start_str, end_str):
    start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
    frames, missing = [], []
    for s in tickers:
        df = _fetch_one(s, start, end)
        if df is None or df.empty:
            missing.append(s); continue
        sub = df.reset_index()
        sub.columns = [c if not isinstance(c, tuple) else c[0] for c in sub.columns]
        sub = sub.rename(columns={"Date":"date","Close":"close","Volume":"volume"})
        sub["asset"]      = s.split("-")[0]
        sub["volume_usd"] = sub["volume"]
        frames.append(sub[["date","asset","close","volume","volume_usd"]])
    if not frames:
        return None, missing
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["asset","date"]).reset_index(drop=True)
    g = raw.groupby("asset", group_keys=False)
    raw["return"]         = g["close"].pct_change()
    raw["norm_price"]     = g["close"].transform(lambda s: s/s.iloc[0])
    raw["rolling_vol_30"] = (g["return"].rolling(30).std()
                             .reset_index(level=0, drop=True) * np.sqrt(TRADING_DAYS))
    raw["cum_max"]    = g["close"].cummax()
    raw["drawdown"]   = raw["close"] / raw["cum_max"] - 1.0
    raw["adv_30_usd"] = g["volume_usd"].rolling(30).mean().reset_index(level=0, drop=True)
    return raw, missing

def _arr(x):
    r = np.asarray(pd.Series(x), dtype=float); return r[~np.isnan(r)]
def hist_var(x, c=0.95):
    r = _arr(x); return float(-np.quantile(r, 1-c)) if r.size else np.nan
def parametric_var(x, c=0.95):
    r = _arr(x)
    return np.nan if r.size<2 else float(-(r.mean()+r.std(ddof=1)*stats.norm.ppf(1-c)))
def cornish_fisher_var(x, c=0.95):
    r = _arr(x)
    if r.size < 4: return np.nan
    s = float(stats.skew(r, bias=False))
    k = float(stats.kurtosis(r, fisher=True, bias=False))
    z = stats.norm.ppf(1-c)
    z_cf = z+(z**2-1)*s/6+(z**3-3*z)*k/24-(2*z**3-5*z)*(s**2)/36
    return float(-(r.mean()+r.std(ddof=1)*z_cf))
def expected_shortfall(x, c=0.95):
    r = _arr(x)
    if r.size == 0: return np.nan
    q = np.quantile(r, 1-c); tail = r[r<=q]
    return float(-tail.mean()) if tail.size else np.nan
def ann_vol(x):
    r = _arr(x)
    return float(r.std(ddof=1)*np.sqrt(TRADING_DAYS)) if r.size>1 else np.nan
def max_drawdown(prices):
    p = pd.Series(prices).dropna()
    return float((p/p.cummax()-1).min()) if not p.empty else np.nan
def kupiec_pof(returns, c=0.95, window=250):
    r = _arr(returns)
    if r.size <= window+5: return np.nan, np.nan, np.nan
    breaches = n = 0
    for i in range(window, len(r)):
        v = hist_var(r[i-window:i], c)
        if not np.isnan(v) and r[i] < -v: breaches += 1
        n += 1
    if n == 0 or breaches in (0, n):
        return (breaches/n if n else np.nan), np.nan, np.nan
    p_hat, p = breaches/n, 1-c
    lr = -2*(np.log(((1-p)**(n-breaches))*(p**breaches))
             - np.log(((1-p_hat)**(n-breaches))*(p_hat**breaches)))
    return p_hat, float(lr), float(1-stats.chi2.cdf(lr, df=1))
def component_var_normal(rets_df, weights, c=0.95):
    rets = rets_df.dropna()
    if rets.empty: return None
    cov = rets.cov().values; w = np.asarray(weights, dtype=float); pv = float(w@cov@w)
    if pv <= 0: return None
    cvar = w*((cov@w)/np.sqrt(pv))*(-stats.norm.ppf(1-c))
    denom = np.abs(cvar).sum()
    pct = cvar/denom*100 if denom>1e-12 else np.zeros_like(cvar)
    return pd.DataFrame({"Asset":rets.columns,
                          "Component VaR ($)":cvar,
                          "% of Portfolio VaR":pct})
def bootstrap_var_ci(returns, c=0.95, n_boot=1000, seed=42):
    r = _arr(returns)
    if r.size < 30: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(r, size=(n_boot, r.size), replace=True)
    vars_ = -np.quantile(boot, 1-c, axis=1)
    return float(np.quantile(vars_, 0.025)), float(np.quantile(vars_, 0.975))
def liquidity_cost(position_usd, adv_usd, sigma, lam=0.5):
    if adv_usd is None or adv_usd<=0 or np.isnan(adv_usd) or np.isnan(sigma): return 0.0
    return float(lam*(position_usd/adv_usd)*sigma*position_usd)
def portfolio_returns_from(df, holdings):
    total = sum(holdings.values()) or 1.0
    weights = {a: v/total for a, v in holdings.items()}
    pivot = df.pivot_table(index="date", columns="asset", values="return")
    common = [a for a in weights if a in pivot.columns]
    if not common: return pd.Series(dtype=float)
    w = np.array([weights[a] for a in common])
    return (pivot[common]*w).sum(axis=1).dropna()
def rolling_metric(returns, window=250, alpha=0.95, kind="var"):
    fn = hist_var if kind=="var" else expected_shortfall
    return returns.rolling(window).apply(lambda x: fn(x, alpha), raw=True)
def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    a = abs(x)
    if a >= 1e9: return f"${x/1e9:.2f}B"
    if a >= 1e6: return f"${x/1e6:.2f}M"
    if a >= 1e3: return f"${x/1e3:.1f}K"
    return f"${x:,.0f}"
def pctf(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x*100:.{d}f}%"

# Fix #1: Liq. time human-readable
def fmt_liq_time(position_usd, adv_usd):
    if adv_usd <= 0 or np.isnan(adv_usd): return "∞"
    days = position_usd / (adv_usd * LIQ_PARTICIPATION)
    secs = days * 86400
    if secs < 60:   return f"{secs:.0f}s"
    mins = secs / 60
    if mins < 60:   return f"{mins:.1f}min"
    hrs  = mins / 60
    if hrs  < 24:   return f"{hrs:.2f}h"
    return f"{days:.3f}d"


# ──────────────── Streamlit App ────────────────
st.set_page_config(
    page_title="Crypto Risk Dashboard — UNIGE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div style="background:{BRAND};padding:16px 24px;border-radius:8px;margin-bottom:16px">
  <div style="color:#bbd;font-size:0.8rem">
    Université de Genève · Geneva School of Economics and Management
  </div>
  <div style="color:white;font-size:1.6rem;font-weight:700;margin:4px 0">
    Crypto Market &amp; Liquidity Risk Dashboard
  </div>
  <div style="color:#dde6f5;font-size:0.9rem">
    Quantitative Risk Management (Spring 26) · <b>Group 3</b>
  </div>
</div>
""", unsafe_allow_html=True)

# Load data
END   = pd.Timestamp.today().normalize()
START = END - pd.Timedelta(days=365*3)
FEATURES, MISSING = load_features(
    tuple(DEFAULT_TICKERS),
    START.strftime("%Y-%m-%d"),
    END.strftime("%Y-%m-%d")
)
if FEATURES is None:
    st.error("All data downloads failed. Check internet connection.")
    st.stop()

ASSETS = sorted(FEATURES["asset"].unique().tolist())

# ──────────────── Sidebar Controls ────────────────
with st.sidebar:
    st.markdown(f"### 📦 Portfolio holdings (USD)")
    st.caption("Edit any value — all charts re-compute live.")
    holdings = {}
    for a in ASSETS:
        holdings[a] = st.number_input(
            a, min_value=0, step=10_000,
            value=int(DEFAULT_POSITIONS.get(a, 0)),
            key=f"hold_{a}"
        )
    st.divider()
    alpha = st.slider("VaR / ES confidence (α)",
                      min_value=0.90, max_value=0.99,
                      value=0.95, step=0.01,
                      format="%.2f")
    lam = st.slider("Liquidity coefficient (λ)",
                    min_value=0.0, max_value=2.0,
                    value=0.5, step=0.05,
                    format="%.2f")
    scenario = st.selectbox("Stress scenario", list(SCENARIOS.keys()))
    st.divider()
    AUM = sum(holdings.values()) or 1.0
    port_ret = portfolio_returns_from(FEATURES, holdings).dropna()
    last_r = port_ret.iloc[-1] if not port_ret.empty else np.nan
    regime_label, regime_color = classify_regime(last_r)
    st.markdown(f"**AUM:** {money(AUM)}")
    st.markdown(f"**α:** {int(alpha*100)}%")
    st.markdown(f"**λ:** {lam:.2f}")
    st.markdown(
        f"**Regime:** <span style='color:{regime_color};font-weight:700'>{regime_label}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**Scenario:** {scenario}")

# ──────────────── KPI Row ────────────────
std_var_pct = hist_var(port_ret, alpha) if not port_ret.empty else np.nan
std_var_usd = std_var_pct * AUM if not np.isnan(std_var_pct) else np.nan
latest = FEATURES.sort_values("date").groupby("asset").tail(1).set_index("asset")
lc_total = sum(
    liquidity_cost(
        holdings.get(a,0),
        float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,
        FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1), lam
    ) for a in ASSETS if a in latest.index
)
lvar = (std_var_usd + lc_total) if not np.isnan(std_var_usd) else np.nan
a_vol = ann_vol(port_ret) if not port_ret.empty else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Portfolio AUM",          money(AUM),          f"{len(ASSETS)} assets")
k2.metric(f"Std VaR {int(alpha*100)}%", money(std_var_usd),  pctf(std_var_pct))
k3.metric(f"Liq. cost (λ={lam:.2f})",  money(lc_total),     f"{lc_total/AUM*100:.4f}% AUM")
k4.metric(f"L-VaR {int(alpha*100)}%",   money(lvar),         pctf(lvar/AUM) if not np.isnan(lvar) else "—")
k5.metric("Ann. vol (√365)",          pctf(a_vol,1),       "Crypto 365d/yr")

st.divider()

# ──────────────── Tabs ────────────────
tab_market, tab_var, tab_liq, tab_scen, tab_meth = st.tabs([
    "📊 Market Data",
    "📉 VaR & ES",
    "💧 Liquidity",
    "🌪 Crash Scenarios",
    "📚 Methodology",
])


# ──────────── TAB 1: Market Data ────────────
with tab_market:
    df = FEATURES.copy()
    vol_30 = port_ret.tail(30).std(ddof=1)*np.sqrt(TRADING_DAYS) if len(port_ret)>=30 else np.nan
    ret_7d = (1+port_ret.tail(7)).prod()-1 if len(port_ret)>=7 else np.nan
    avg_p  = df.pivot_table(index="date",columns="asset",values="close").mean(axis=1).dropna()
    mdd    = max_drawdown(avg_p)
    senti  = ("Risk-off" if not pd.isna(ret_7d) and ret_7d<-0.10
              else "Risk-on" if not pd.isna(ret_7d) and ret_7d>0.10
              else "Neutral" if not pd.isna(ret_7d) else "N/A")

    # Regime legend (Fix #2)
    st.info(
        "🟤 **Crisis** <−5% │ "
        "🔴 **High Risk** <−3% │ "
        "🟠 **Elevated** <−1.5% │ "
        "🟡 **Moderate** −1.5% to 0% │ "
        "🟢 **Stable** 0% to +1.5% │ "
        "🔵 **Bull** >+1.5%"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Risk regime",  regime_label, "1-day signal")
    m2.metric("Vol (30d)",    pctf(vol_30,1), "Annualised (√365)")
    m3.metric("7-day return", pctf(ret_7d,2), "Trailing")
    m4.metric("Max drawdown", pctf(mdd,1),    "Equal-weight")
    m5.metric("Sentiment",    senti,          "7d return")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df, x="date", y="norm_price", color="asset",
                      title="Normalised price (start=1.0)", height=320)
        fig.update_layout(template="plotly_white", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(df, x="date", y="rolling_vol_30", color="asset",
                      title="30d rolling volatility (ann. √365)", height=320)
        fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(df, x="date", y="drawdown", color="asset",
                      title="Drawdown from peak", height=320)
        fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        corr = df.pivot_table(index="date",columns="asset",values="return").corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text:.2f}"))
        fig.update_layout(title="Return correlation matrix",
                          template="plotly_white", height=320)
        st.plotly_chart(fig, use_container_width=True)


# ──────────── TAB 2: VaR & ES ────────────
with tab_var:
    if port_ret.empty:
        st.warning("No return data.")
    else:
        methods = {
            "Historical Simulation": hist_var(port_ret, alpha),
            "Parametric (Normal)":   parametric_var(port_ret, alpha),
            "Cornish-Fisher":        cornish_fisher_var(port_ret, alpha),
        }
        es     = expected_shortfall(port_ret, alpha)
        ci_lo, ci_hi = bootstrap_var_ci(port_ret.values, alpha)
        phat, lr_, pval = kupiec_pof(port_ret.values, alpha)
        ci_str = (f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]"
                  if not np.isnan(ci_lo) else "—")
        rows = []
        for name, v in methods.items():
            rows.append({
                "Method": name,
                f"VaR {int(alpha*100)}% (%)": f"{v*100:.2f}" if not np.isnan(v) else "—",
                f"VaR {int(alpha*100)}% ($)": money(v*AUM)   if not np.isnan(v) else "—",
                "Bootstrap 95% CI": ci_str,
            })
        rows.append({
            "Method": f"ES {int(alpha*100)}%",
            f"VaR {int(alpha*100)}% (%)": f"{es*100:.2f}" if not np.isnan(es) else "—",
            f"VaR {int(alpha*100)}% ($)": money(es*AUM)    if not np.isnan(es) else "—",
            "Bootstrap 95% CI": ci_str,
        })

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"VaR comparison @ {int(alpha*100)}%")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=port_ret*100, nbinsx=60, marker_color="#9aa9c2"))
            v0 = methods["Historical Simulation"]
            if not np.isnan(v0):
                fig.add_vline(x=-v0*100, line_color="red", line_dash="dash",
                              annotation_text=f"VaR {int(alpha*100)}%")
            if not np.isnan(es):
                fig.add_vline(x=-es*100, line_color="darkred", line_dash="dot",
                              annotation_text=f"ES {int(alpha*100)}%")
            fig.update_layout(title="Return distribution", template="plotly_white",
                              height=320, xaxis_title="Daily return (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Kupiec POF backtest")
        kp = []
        if not (phat is None or (isinstance(phat,float) and np.isnan(phat))):
            kp.append(f"Breach rate: {phat*100:.2f}% (target {(1-alpha)*100:.0f}%)")
        if pval is not None and not np.isnan(pval):
            kp.append(f"p-value: {pval:.3f} → "
                      f"{'adequate ✓' if pval>0.05 else 'under-estimates risk ✗'}")
        if not kp:
            kp = ["Need > 255 days for Kupiec backtest"]
        for item in kp:
            st.write(f"• {item}")

        st.subheader(f"Per-asset @ α={int(alpha*100)}%")
        per = []
        for a in ASSETS:
            sub = FEATURES[FEATURES["asset"]==a].sort_values("date")
            r   = sub["return"].dropna()
            if r.empty: continue
            per.append({"Asset":a,
                        "Last price": f"${sub['close'].iloc[-1]:,.2f}",
                        f"VaR {int(alpha*100)}% (%)": f"{hist_var(r,alpha)*100:.2f}",
                        f"ES {int(alpha*100)}% (%)": f"{expected_shortfall(r,alpha)*100:.2f}",
                        "Max DD":   f"{max_drawdown(sub['close'])*100:.1f}%",
                        "Ann. vol (√365)": f"{ann_vol(r)*100:.0f}%"})
        st.dataframe(pd.DataFrame(per), hide_index=True, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            pivot = FEATURES.pivot_table(index="date",columns="asset",values="return").dropna()
            w = np.array([holdings.get(a,0)/AUM for a in pivot.columns])
            cvar_df = component_var_normal(pivot, w, alpha)
            if cvar_df is not None:
                fig = px.bar(cvar_df, x="Asset", y="% of Portfolio VaR", color="Asset",
                             title="Component VaR (Euler decomposition)")
                fig.update_layout(template="plotly_white", height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            rv  = rolling_metric(port_ret, 250, alpha, "var")*100
            re_ = rolling_metric(port_ret, 250, alpha, "es")*100
            roll_df = pd.DataFrame({"date":rv.index,"Rolling VaR":rv.values,
                                     "Rolling ES":re_.values}).dropna()
            if roll_df.empty:
                st.info("Rolling VaR & ES: need 250+ days")
            else:
                fig = px.line(roll_df, x="date", y=["Rolling VaR","Rolling ES"],
                              title=f"Rolling VaR & ES (250d, {int(alpha*100)}%)")
                fig.update_layout(template="plotly_white", height=320,
                                  yaxis_title="Loss (%)", legend_title="")
                st.plotly_chart(fig, use_container_width=True)


# ──────────── TAB 3: Liquidity ────────────
with tab_liq:
    st.info(
        "**Liquidity Cost:** LC = λ · (Position/ADV) · σ · Position  "
        "[Kyle/Almgren-Chriss adapted; λ = price-impact coeff.; ADV = 30d avg. daily volume]  "
        "\n**L-VaR** = Std VaR + LC  │  "
        "**Liq. time** = time to liquidate at 25% daily volume participation"
    )
    rows_liq = []; total_lc = 0.0
    for a in ASSETS:
        if a not in latest.index: continue
        h     = float(holdings.get(a, 0))
        adv   = float(latest.loc[a,"adv_30_usd"])
        sd    = FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1)
        var_a = hist_var(FEATURES[FEATURES["asset"]==a]["return"], alpha)
        lc    = liquidity_cost(h, adv, sd, lam); total_lc += lc
        lavar = (var_a*h+lc) if not np.isnan(var_a) else np.nan
        rows_liq.append({
            "Asset":         a,
            "Position":      money(h),
            "30-day ADV":    money(adv),
            "Position/ADV":  f"{(h/adv*100 if adv>0 else 0):.4f}%",
            "Liq. time":     fmt_liq_time(h, adv),   # Fix #1
            "σ (daily)":     f"{sd*100:.2f}%",
            "Liq. cost":     money(lc),
            "Std VaR ($)":   money(var_a*h) if not np.isnan(var_a) else "—",
            "L-VaR ($)":     money(lavar)   if not np.isnan(lavar)  else "—",
        })

    st.subheader(f"Per-asset liquidity (λ={lam:.2f}, α={int(alpha*100)}%)")
    st.dataframe(pd.DataFrame(rows_liq), hide_index=True, use_container_width=True)
    st.markdown(f"**Total LC:** {money(total_lc)} &nbsp;&nbsp; "
                f"**Std VaR:** {money(std_var_usd)} &nbsp;&nbsp; "
                f"**L-VaR:** {money((std_var_usd or 0)+total_lc)}")

    # Lambda sensitivity
    lam_grid = np.linspace(0.0, 2.0, 21); sens = []
    for L in lam_grid:
        tot = sum(liquidity_cost(
                    float(holdings.get(a,0)),
                    float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,
                    FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1), L)
                  for a in ASSETS if a in latest.index)
        sens.append({"lambda":L, "Liquidity cost":tot,
                     "L-VaR":(std_var_usd or 0)+tot})
    sens_df = pd.DataFrame(sens)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sens_df["lambda"], y=sens_df["Liquidity cost"],
                              name="Liquidity cost", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sens_df["lambda"], y=sens_df["L-VaR"],
                              name="L-VaR", mode="lines+markers"))
    if std_var_usd and not np.isnan(std_var_usd):
        fig.add_hline(y=std_var_usd, line_dash="dash", line_color="gray",
                      annotation_text=f"Std VaR={money(std_var_usd)}")
    fig.add_vline(x=lam, line_color="black", line_dash="dot",
                  annotation_text=f"λ={lam:.2f}")
    fig.update_layout(title="L-VaR vs λ (price-impact sensitivity)",
                      template="plotly_white", height=380,
                      xaxis_title="λ", yaxis_title="USD")
    st.subheader("λ sensitivity")
    st.plotly_chart(fig, use_container_width=True)


# ──────────── TAB 4: Crash Scenarios ────────────
with tab_scen:
    sc = SCENARIOS[scenario]
    st.warning(f"**{scenario}** — {sc['description']}")
    rows_sc = []; total_mtm = 0.0; total_stress_lc = 0.0
    for a in ASSETS:
        if a not in latest.index: continue
        h     = float(holdings.get(a, 0))
        shock = sc["shocks"].get(a, -0.30)
        adv   = float(latest.loc[a,"adv_30_usd"])
        sd    = FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1)
        mtm   = h*shock
        slc   = liquidity_cost(h, adv*sc["adv_mult"], sd*sc["vol_mult"],
                                lam*sc["lambda_mult"])
        total_mtm += mtm; total_stress_lc += slc
        rows_sc.append({
            "Asset":a, "Position":money(h),
            "Shock":f"{shock*100:+.0f}%", "MTM P&L":money(mtm),
            "Stressed ADV":money(adv*sc["adv_mult"]),
            "Stressed σ":f"{sd*sc['vol_mult']*100:.2f}%",
            "Stressed λ":f"{lam*sc['lambda_mult']:.2f}",
            "Stressed LC":money(slc),
            "Total loss":money(mtm-slc),
        })
    total_loss = abs(total_mtm)+total_stress_lc

    s1, s2, s3 = st.columns(3)
    s1.metric("MTM P&L",    money(total_mtm),       "Shock leg")
    s2.metric("Stressed LC",money(total_stress_lc),  "Fire-sale leg")
    s3.metric("Total loss", money(-total_loss),       f"{total_loss/AUM*100:.1f}% AUM")

    cumulative = total_mtm - total_stress_lc
    fig_w = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative","relative","total"],
        x=["MTM P&L","Stressed LC","Total stressed loss"],
        text=[money(total_mtm), money(-total_stress_lc), money(cumulative)],
        y=[total_mtm, -total_stress_lc, cumulative],
        textposition="outside",
        connector={"line":{"color":"#888"}},
    ))
    fig_w.update_layout(title=f"{scenario} — loss decomposition",
                        template="plotly_white", height=420, yaxis_title="USD",
                        margin=dict(t=60, b=80))
    st.plotly_chart(fig_w, use_container_width=True)
    st.dataframe(pd.DataFrame(rows_sc), hide_index=True, use_container_width=True)

    base_var = hist_var(port_ret, alpha)*AUM if not port_ret.empty else 0
    base_lc  = sum(liquidity_cost(
                    holdings.get(a,0),
                    float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,
                    FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1), lam)
                   for a in ASSETS if a in latest.index)
    fig_cmp = go.Figure([
        go.Bar(name="Std VaR/shock", x=["Base","Stressed"],
               y=[base_var, abs(total_mtm)], marker_color="#c0392b"),
        go.Bar(name="Liq. cost",     x=["Base","Stressed"],
               y=[base_lc, total_stress_lc], marker_color="#e67e22"),
    ])
    fig_cmp.update_layout(barmode="stack", title="Base vs stressed",
                          template="plotly_white", height=320, yaxis_title="USD")
    st.plotly_chart(fig_cmp, use_container_width=True)


# ──────────── TAB 5: Methodology ────────────
with tab_meth:
    st.subheader("🔬 Risk Methodology")
    st.markdown("""
Three VaR estimators (Historical Simulation, Parametric Normal, Cornish-Fisher),
Expected Shortfall (CVaR), Bootstrap CI (n=1,000), Kupiec POF backtest,
Component VaR (Euler allocation), Liquidity-adjusted VaR (L-VaR), 4 stress scenarios.

**Annualisation:** σ\_annual = σ\_daily × √365  
*(crypto markets trade 24/7, 365 days/year; cf. √252 for equities)*

**Cornish-Fisher VaR:**  
z\* = z + (z²−1)·γ₁/6 + (z³−3z)·γ₂/24 − (2z³−5z)·γ₁²/36  
where γ₁ = skewness, γ₂ = excess kurtosis, z = Φ⁻¹(α)

**Liquidity Cost (Kyle 1985 / Almgren-Chriss 2000, simplified):**  
LC = λ · (Position / ADV) · σ · Position = λ · σ · Position² / ADV  
*[quadratic in position size; λ = price-impact coefficient; ADV = 30-day average daily volume in USD]*

**L-VaR (Liquidity-adjusted VaR):** L-VaR = Std VaR + LC

**Risk Regime classification (1-day portfolio return r):**  
Crisis r < −5% │ High Risk r < −3% │ Elevated r < −1.5% │ Moderate −1.5% ≤ r < 0% │ Stable 0% ≤ r < +1.5% │ Bull r ≥ +1.5%
""")
    st.subheader("📚 References")
    refs = [
        "Bangia, Diebold, Schuermann & Stroughair (1999). Modeling Liquidity Risk. Wharton FIC Working Paper 99-06.",
        "Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions. Journal of Risk, 3(2), 5–39.",
        "Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica, 53(6), 1315–1335.",
        "Kupiec, P. H. (1995). Techniques for verifying VaR models. Journal of Derivatives, 3(2), 73–84.",
        "Artzner et al. (1999). Coherent measures of risk. Mathematical Finance, 9(3), 203–228.",
        "Cornish & Fisher (1937). Moments and cumulants. Revue de l'Institut International de Statistique.",
        "Brunnermeier & Pedersen (2009). Market liquidity and funding liquidity. RFS, 22(6), 2201–2238.",
        "BCBS (2019). Minimum capital requirements for market risk. BIS document d457.",
    ]
    for i, r in enumerate(refs, 1):
        st.markdown(f"{i}. {r}")

st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.78rem;margin-top:24px'>"
    "© 2026 Group 3 · UNIGE · Quantitative Risk Management (Spring 26)"
    "</div>",
    unsafe_allow_html=True
)
