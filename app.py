"""Réplica de las pantallas de métricas (estilo Nevometrics): rentabilidad, EPS (real y estimado) y FCF.

Uso:  pip install -r requirements.txt  &&  streamlit run app.py
Datos: sin clave usa Yahoo Finance (yfinance, ~4-5 años anuales).
       Con la variable FMP_API_KEY usa Financial Modeling Prep (10+ años según plan).
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Métricas financieras", layout="wide")

METRICS = {
    "Return on Capital Employed": "roce",
    "Gross Margin": "gross",
    "Operating Margin": "operating",
    "Profit Margin": "profit",
}
COLORS = {"roce": "#4f83e8", "gross": "#1f9d6b", "operating": "#f59e0b", "profit": "#8b5cf6"}
RANGES = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Max": None}
FMP_KEY = os.getenv("FMP_API_KEY", "")
BLUE, LIGHT_BLUE, RED = "#4f83e8", "#a9c1f5", "#e5484d"


def _fmp(path, symbol, period, limit=60):
    r = requests.get(
        f"https://financialmodelingprep.com/stable/{path}",
        params={"symbol": symbol, "period": period, "limit": limit, "apikey": FMP_KEY},
        timeout=30,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=3600, show_spinner="Descargando datos…")
def load(symbol: str, mode: str) -> pd.DataFrame:
    period = "annual" if mode == "Annual" else "quarter"  # TTM parte de trimestrales
    if FMP_KEY:
        inc = _fmp("income-statement", symbol, period)
        bal = _fmp("balance-sheet-statement", symbol, period)
        cf = _fmp("cash-flow-statement", symbol, period)
        m = inc.merge(bal, on="date", suffixes=("", "_b")).merge(cf[["date", "freeCashFlow"]], on="date", how="left")
        eps = m["epsDiluted"] if "epsDiluted" in m else m.get("eps")
        df = pd.DataFrame({
            "date": pd.to_datetime(m["date"]), "revenue": m["revenue"],
            "gross": m["grossProfit"], "ebit": m["operatingIncome"],
            "net": m["netIncome"], "assets": m["totalAssets"],
            "curliab": m["totalCurrentLiabilities"], "eps": eps, "fcf": m["freeCashFlow"],
        })
    else:
        t = yf.Ticker(symbol)
        annual = period == "annual"
        inc = t.income_stmt if annual else t.quarterly_income_stmt
        bal = t.balance_sheet if annual else t.quarterly_balance_sheet
        cf = t.cashflow if annual else t.quarterly_cashflow
        row = lambda f, k: f.loc[k] if k in f.index else pd.Series(dtype=float)
        df = pd.DataFrame({
            "revenue": row(inc, "Total Revenue"), "gross": row(inc, "Gross Profit"),
            "ebit": row(inc, "Operating Income"), "net": row(inc, "Net Income"),
            "assets": row(bal, "Total Assets"), "curliab": row(bal, "Current Liabilities"),
            "eps": row(inc, "Diluted EPS"), "fcf": row(cf, "Free Cash Flow"),
        })
        df.index = pd.to_datetime(df.index)
        df = df.rename_axis("date").reset_index()
    return df.dropna(subset=["revenue"]).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_forward(symbol: str, mode: str, last_date: pd.Timestamp) -> pd.DataFrame:
    """EPS estimado por consenso de analistas: 2 periodos futuros (años o trimestres)."""
    annual = mode == "Annual"
    labels = []
    try:
        if FMP_KEY:
            est = _fmp("analyst-estimates", symbol, "annual" if annual else "quarter", limit=20)
            est["date"] = pd.to_datetime(est["date"])
            est = est[est["date"] > last_date].sort_values("date").head(2)
            vals = est["epsAvg"].tolist()
            labels = [str(d.year) if annual else str(d.to_period("Q")) for d in est["date"]]
        else:
            e = yf.Ticker(symbol).earnings_estimate
            keys = ["0y", "+1y"] if annual else ["0q", "+1q"]
            vals = [e.loc[k, "avg"] for k in keys if k in e.index]
            base = last_date.year if annual else last_date.to_period("Q")
            labels = [str(base + i) for i in range(1, len(vals) + 1)]
    except Exception:
        return pd.DataFrame(columns=["label", "eps"])
    return pd.DataFrame({"label": [f"{l} (est.)" for l in labels], "eps": vals})


def compute(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    if mode == "TTM":
        for c in ["revenue", "gross", "ebit", "net", "eps", "fcf"]:
            df[c] = df[c].rolling(4).sum()
        df = df.dropna(subset=["revenue"])
    out = pd.DataFrame({"date": df["date"], "eps": df["eps"], "fcf": df["fcf"]})
    out["gross"] = df["gross"] / df["revenue"] * 100
    out["operating"] = df["ebit"] / df["revenue"] * 100
    out["profit"] = df["net"] / df["revenue"] * 100
    out["roce"] = df["ebit"] / (df["assets"] - df["curliab"]) * 100  # EBIT / (Activos - Pasivo corriente)
    out["label"] = out["date"].dt.year.astype(str) if mode == "Annual" else out["date"].dt.to_period("Q").astype(str)
    return out


def style(fig, **yaxis):
    fig.update_layout(barmode="group", height=460, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(gridcolor="rgba(128,128,128,.25)", **yaxis),
                      xaxis=dict(type="category"),
                      legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------- Controles ----------
st.markdown("### Métricas financieras")
c1, c2, c3 = st.columns([2, 1, 3])
mode = c1.segmented_control("Periodo", ["Annual", "Quarterly", "TTM"], default="Annual", label_visibility="collapsed") or "Annual"
symbol = c2.text_input("Ticker", "GOOG", label_visibility="collapsed").strip().upper()
rng = c3.segmented_control("Rango", list(RANGES), default="10Y", label_visibility="collapsed") or "10Y"

try:
    data = compute(load(symbol, mode), mode)
except Exception as e:
    st.error(f"No se pudieron obtener datos de {symbol}: {e}")
    st.stop()

per_year = 1 if mode == "Annual" else 4
years = RANGES[rng]
view = data if years is None else data.tail(years * per_year)

tab_rent, tab_eps, tab_fcf = st.tabs(["Rentabilidad", "Ganancia por acción", "Flujo de caja libre (FCF)"])

# ---------- Rentabilidad ----------
with tab_rent:
    picked = st.pills("Series", list(METRICS), selection_mode="multi", default=["Return on Capital Employed"], label_visibility="collapsed")
    fig = go.Figure()
    for name in picked:
        k = METRICS[name]
        fig.add_bar(x=view["label"], y=view[k], name=name, marker_color=COLORS[k])
    st.plotly_chart(style(fig, ticksuffix="%"), use_container_width=True)
    if picked:  # Promedio de la primera serie en los últimos 1/3/5/10 años (N/A si faltan datos)
        k = METRICS[picked[0]]
        for col, n in zip(st.columns(4), [1, 3, 5, 10]):
            tail = data[k].dropna().tail(n * per_year)
            col.metric(f"{n}Y (promedio)", f"{tail.mean():.1f}%" if len(tail) >= n * per_year else "N/A")

# ---------- EPS real + estimado ----------
with tab_eps:
    if view["eps"].notna().any():
        fig = go.Figure()
        fig.add_bar(x=view["label"], y=view["eps"], name="EPS diluido", marker_color=BLUE,
                    texttemplate="%{y:,.2f}", textposition="outside")
        fwd = load_forward(symbol, mode, data["date"].iloc[-1]) if mode != "TTM" else pd.DataFrame()
        if not fwd.empty:
            fig.add_bar(x=fwd["label"], y=fwd["eps"], name="EPS estimado (consenso)",
                        marker_color=LIGHT_BLUE, marker_pattern_shape="/",
                        texttemplate="%{y:,.2f}", textposition="outside")
        st.plotly_chart(style(fig), use_container_width=True)
        if mode == "TTM":
            st.caption("Las estimaciones futuras se muestran solo en Annual y Quarterly.")
        elif fwd.empty:
            st.caption("No se encontraron estimaciones de analistas para este ticker.")
    else:
        st.info("Sin datos de EPS para este ticker.")

# ---------- FCF ----------
with tab_fcf:
    if view["fcf"].notna().any():
        div, unit = (1e9, "B") if view["fcf"].abs().max() >= 1e9 else (1e6, "M")
        y = view["fcf"] / div
        fig = go.Figure(go.Bar(
            x=view["label"], y=y, name="Free Cash Flow",
            marker_color=[BLUE if v >= 0 else RED for v in y.fillna(0)],
            texttemplate=f"$%{{y:,.2f}}{unit}", textposition="outside",
            customdata=view["fcf"], hovertemplate="%{x}<br>FCF: $%{customdata:,.0f}<extra></extra>"))
        st.plotly_chart(style(fig, ticksuffix=unit, tickprefix="$", tickformat=",.0f"), use_container_width=True)
    else:
        st.info("Sin datos de FCF para este ticker.")

with st.expander("Ver datos"):
    t = view.drop(columns="date").set_index("label").rename_axis("Periodo").rename(columns={
        "eps": "EPS", "fcf": "FCF (USD)", "gross": "Margen bruto (%)",
        "operating": "Margen operativo (%)", "profit": "Margen de beneficio (%)", "roce": "ROCE (%)"})
    fmt = {c: "{:,.2f}" for c in t.columns}
    fmt["FCF (USD)"] = "{:,.0f}"
    st.dataframe(t.style.format(fmt, na_rep="—"), use_container_width=True)
