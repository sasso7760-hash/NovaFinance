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


SEC_UA = os.getenv("SEC_USER_AGENT", "NovaFinance contacto@novafinance.app")  # la SEC pide nombre + correo
# metrica: (tags XBRL por prioridad, es_saldo_a_fecha, unidad)
SEC_TAGS = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], False, "USD"),
    "cogs": (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], False, "USD"),
    "gross": (["GrossProfit"], False, "USD"),
    "ebit": (["OperatingIncomeLoss"], False, "USD"),
    "net": (["NetIncomeLoss"], False, "USD"),
    "assets": (["Assets"], True, "USD"),
    "curliab": (["LiabilitiesCurrent"], True, "USD"),
    "eps": (["EarningsPerShareDiluted"], False, "USD/shares"),
    "ocf": (["NetCashProvidedByUsedInOperatingActivities"], False, "USD"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], False, "USD"),
}


def _sec_series(gaap, tags, instant, unit):
    """Serie anual (fin de ejercicio -> valor) de los 10-K; usa el dato mas reciente y varios tags."""
    out = {}
    for tag in tags:
        best = {}
        for e in gaap.get(tag, {}).get("units", {}).get(unit, []):
            if not str(e.get("form", "")).startswith("10-K") or (("start" not in e) != instant):
                continue
            if not instant and not 340 <= (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days <= 380:
                continue
            if e["end"] not in best or e["filed"] > best[e["end"]][0]:
                best[e["end"]] = (e["filed"], e["val"])
        for k, (_, v) in best.items():
            out.setdefault(k, v)
    return pd.Series(out, dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def _sec_cik(symbol: str):
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": SEC_UA}, timeout=30)
    r.raise_for_status()
    t = symbol.upper().replace(".", "-")
    return next((str(v["cik_str"]).zfill(10) for v in r.json().values() if v["ticker"].upper() == t), None)


def _sec_annual(symbol: str) -> pd.DataFrame:
    cik = _sec_cik(symbol)
    if not cik:
        raise ValueError("ticker no encontrado en la SEC")
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                     headers={"User-Agent": SEC_UA}, timeout=60)
    r.raise_for_status()
    gaap = r.json().get("facts", {}).get("us-gaap", {})
    df = pd.DataFrame({k: _sec_series(gaap, *v) for k, v in SEC_TAGS.items()})
    if df["revenue"].dropna().empty:  # p. ej. empresas extranjeras (20-F/IFRS): sin datos us-gaap
        raise ValueError("la SEC no tiene estados financieros us-gaap para este ticker")
    df["gross"] = df["gross"].fillna(df["revenue"] - df["cogs"])
    df["fcf"] = df["ocf"] - df["capex"]
    df.index = pd.to_datetime(df.index)
    cols = ["revenue", "gross", "ebit", "net", "assets", "curliab", "eps", "fcf"]
    return df[cols].rename_axis("date").reset_index()


def _yahoo(symbol: str, annual: bool) -> pd.DataFrame:
    t = yf.Ticker(symbol)
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
    return df.rename_axis("date").reset_index()


def _merge_free(symbol: str) -> pd.DataFrame:
    """Anual gratis: la SEC manda (historia larga); Yahoo rellena huecos y el ejercicio mas reciente."""
    out = None
    for fn in (_sec_annual, lambda sym: _yahoo(sym, True)):
        try:
            p = fn(symbol)
        except Exception:
            continue
        p = p.assign(key=p["date"].dt.to_period("M")).drop_duplicates("key").set_index("key")
        out = p if out is None else out.combine_first(p)
    if out is None:
        raise ValueError("ninguna fuente gratuita devolvio datos")
    out["date"] = out.index.to_timestamp(how="end").normalize()
    return out.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="Descargando datos…")
def load(symbol: str, mode: str) -> pd.DataFrame:
    period = "annual" if mode == "Annual" else "quarter"  # TTM parte de trimestrales
    if FMP_KEY:  # con clave: Financial Modeling Prep
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
    elif period == "annual":
        df = _merge_free(symbol)
    else:
        df = _yahoo(symbol, False)
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


# ---------- Moneda y precio en vivo ----------
GREEN, GREEN_BG, RED_BG = "#0f9d58", "rgba(15,157,88,0.15)", "rgba(229,72,77,0.15)"
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "TWD": "NT$", "CAD": "C$",
                    "MXN": "MX$", "CRC": "₡", "KRW": "₩", "INR": "₹", "BRL": "R$", "CHF": "CHF "}
# rango: (period, interval) de Yahoo
PRICE_RANGES = {"1D": ("1d", "1m"), "5D": ("5d", "5m"), "1M": ("1mo", "30m"), "6M": ("6mo", "1d"),
                "YTD": ("ytd", "1d"), "1Y": ("1y", "1d"), "5Y": ("5y", "1wk"), "Max": ("max", "1mo")}


def cur_symbol(cur):
    return CURRENCY_SYMBOLS.get(cur, f"{cur} " if cur else "")


@st.cache_data(ttl=3600, show_spinner=False)
def info(symbol: str) -> dict:
    """Datos casi estáticos de Yahoo: moneda de reporte, moneda del precio, beta, nombre."""
    try:
        i = yf.Ticker(symbol).get_info()
    except Exception:
        i = {}
    return {k: i.get(k) for k in ("financialCurrency", "currency", "beta", "shortName")}


@st.cache_data(ttl=15, show_spinner=False)
def quote(symbol: str) -> dict:
    fi = yf.Ticker(symbol).fast_info
    out = {}
    for k in ("last_price", "previous_close", "day_high", "day_low", "year_high", "year_low"):
        try:
            out[k] = float(getattr(fi, k))
        except Exception:
            out[k] = None
    return out


@st.cache_data(ttl=15, show_spinner=False)
def price_history(symbol: str, rng: str) -> pd.Series:
    period, interval = PRICE_RANGES[rng]
    h = yf.Ticker(symbol).history(period=period, interval=interval)
    return h["Close"].dropna() if not h.empty else pd.Series(dtype=float)


def _num(v):
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "N/D"


def render_price(symbol: str, rng: str):
    try:
        h, q = price_history(symbol, rng), quote(symbol)
    except Exception as e:
        st.error(f"No se pudo obtener el precio de {symbol}: {e}")
        return
    if h.empty:
        st.warning("Sin datos de precio para este ticker y rango.")
        return
    meta = info(symbol)
    sym = cur_symbol(meta.get("currency"))
    last = q.get("last_price") or float(h.iloc[-1])
    prev = q.get("previous_close")
    base = prev if (rng == "1D" and prev) else float(h.iloc[0])  # como Yahoo: 1D se compara con el cierre anterior
    up = last >= base
    color, bg = (GREEN, GREEN_BG) if up else (RED, RED_BG)

    m = st.columns(5)
    delta = f"{last - prev:+,.2f} ({(last / prev - 1) * 100:+.2f}%)" if prev else None
    m[0].metric(meta.get("shortName") or symbol, f"{sym}{last:,.2f}", delta)
    m[1].metric("Beta", _num(meta.get("beta")),
                help="Beta de Yahoo Finance (5 años, mensual, frente al S&P 500). "
                     "Mayor que 1: se mueve más que el mercado. Menor que 1: se mueve menos.")
    m[2].metric("Cierre anterior", f"{sym}{prev:,.2f}" if prev else "N/D")
    m[3].metric("Rango del día", f"{_num(q.get('day_low'))} – {_num(q.get('day_high'))}")
    m[4].metric("Rango 52 semanas", f"{_num(q.get('year_low'))} – {_num(q.get('year_high'))}")

    last_ts = h.index[-1]
    if rng in ("1D", "5D", "1M"):  # cortar la línea en las pausas (noche / fin de semana)
        gaps = h.index.to_series().diff() > pd.Timedelta(hours=3)
        br = h.index[gaps.values] - pd.Timedelta(seconds=1)
        if len(br):
            h = pd.concat([h, pd.Series(float("nan"), index=br)]).sort_index()
    lo, hi = float(h.min()), float(h.max())
    if rng == "1D" and prev:
        lo, hi = min(lo, prev), max(hi, prev)
    pad = (hi - lo) * 0.08 or hi * 0.01
    fig = go.Figure(go.Scatter(
        x=h.index, y=h.values, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=bg, connectgaps=False, hovertemplate=f"%{{x}}<br>{sym}%{{y:,.2f}}<extra></extra>"))
    if rng == "1D" and prev:
        fig.add_hline(y=prev, line_dash="dot", line_color="gray")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      yaxis=dict(range=[lo - pad, hi + pad], side="right", gridcolor="rgba(128,128,128,.25)"),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Último dato: {last_ts:%d/%m/%Y %H:%M} ({last_ts.tzname()}). Fuente: Yahoo Finance; puede tener "
               "hasta 15 minutos de retraso según la bolsa. El rango de arriba (1Y, 3Y…) no aplica a esta pestaña.")


# ---------- Controles ----------
st.markdown("### Métricas financieras")
c1, c2, c3 = st.columns([2, 1, 3])
mode = c1.segmented_control("Periodo", ["Annual", "Quarterly", "TTM"], default="Annual", label_visibility="collapsed") or "Annual"
symbol = c2.text_input("Ticker", "GOOG", label_visibility="collapsed").strip().upper()
rng = c3.segmented_control("Rango", list(RANGES), default="10Y", label_visibility="collapsed") or "10Y"

tab_rent, tab_eps, tab_fcf, tab_price = st.tabs(
    ["Rentabilidad", "Ganancia por acción", "Flujo de caja libre (FCF)", "Precio en vivo"])

# ---------- Precio en vivo (independiente de los datos financieros) ----------
with tab_price:
    p1, p2 = st.columns([4, 2])
    prng = p1.segmented_control("Rango del precio", list(PRICE_RANGES), default="1D", label_visibility="collapsed") or "1D"
    live = p2.toggle("Actualizar cada 30 s", value=True)
    st.fragment(run_every=30 if live else None)(render_price)(symbol, prng)

# ---------- Datos financieros ----------
try:
    data = compute(load(symbol, mode), mode)
except Exception as e:
    msg = str(e).replace(FMP_KEY, "***") if FMP_KEY else str(e)  # nunca mostrar la clave en pantalla
    st.error(f"No se pudieron obtener datos financieros de {symbol}: {msg}")
    st.stop()

per_year = 1 if mode == "Annual" else 4
years = RANGES[rng]
view = data if years is None else data.tail(years * per_year)
cur = info(symbol).get("financialCurrency")  # moneda en que la empresa reporta (puede no ser USD)
csym = cur_symbol(cur)
cur_note = (f"Cifras en {cur} (moneda de reporte de la empresa)." if cur else "")
if cur and cur != "USD":
    cur_note += " El EPS estimado puede venir en otra moneda o por ADR: compáralo con cuidado."

# ---------- Rentabilidad ----------
with tab_rent:
    picked = st.pills("Series", list(METRICS), selection_mode="multi", default=["Return on Capital Employed"], label_visibility="collapsed")
    fig = go.Figure()
    for name in picked:
        k = METRICS[name]
        fig.add_bar(x=view["label"], y=view[k], name=name, marker_color=COLORS[k])
    st.plotly_chart(style(fig, ticksuffix="%"), width="stretch")
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
        st.plotly_chart(style(fig, title_text=f"EPS ({cur})" if cur else None), width="stretch")
        if mode == "TTM":
            st.caption("Las estimaciones futuras se muestran solo en Annual y Quarterly.")
        elif fwd.empty:
            st.caption("No se encontraron estimaciones de analistas para este ticker.")
        if cur_note:
            st.caption(cur_note)
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
            texttemplate=f"{csym}%{{y:,.2f}}{unit}", textposition="outside",
            customdata=view["fcf"], hovertemplate=f"%{{x}}<br>FCF: {csym}%{{customdata:,.0f}}<extra></extra>"))
        st.plotly_chart(style(fig, ticksuffix=unit, tickprefix=csym, tickformat=",.0f"), width="stretch")
        if cur_note:
            st.caption(cur_note)
    else:
        st.info("Sin datos de FCF para este ticker.")

with st.expander("Ver datos"):
    fcf_col = f"FCF ({cur})" if cur else "FCF"
    t = view.drop(columns="date").set_index("label").rename_axis("Periodo").rename(columns={
        "eps": "EPS", "fcf": fcf_col, "gross": "Margen bruto (%)",
        "operating": "Margen operativo (%)", "profit": "Margen de beneficio (%)", "roce": "ROCE (%)"})
    fmt = {c: "{:,.2f}" for c in t.columns}
    fmt[fcf_col] = "{:,.0f}"
    st.dataframe(t.style.format(fmt, na_rep="—"), width="stretch")
