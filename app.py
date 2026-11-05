"""Réplica de las pantallas de métricas (estilo Nevometrics): rentabilidad, EPS (real y estimado) y FCF.

Uso:  pip install -r requirements.txt  &&  streamlit run app.py
Datos: sin clave usa Yahoo Finance (yfinance, ~4-5 años anuales).
       Con la variable FMP_API_KEY usa Financial Modeling Prep (10+ años según plan).
"""
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Métricas financieras", layout="wide")

METRICS = {
    "Return on Capital Employed": "roce",
    "Operating Margin": "operating",
    "Net Margin": "profit",
}
COLORS = {"roce": "#4f83e8", "operating": "#f59e0b", "profit": "#1f9d6b"}
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
    """Serie anual (fin de ejercicio -> valor) de los 10-K, con la fecha de presentación del dato usado."""
    out, filed = {}, {}
    for tag in tags:
        best = {}
        for e in gaap.get(tag, {}).get("units", {}).get(unit, []):
            if not str(e.get("form", "")).startswith("10-K") or (("start" not in e) != instant):
                continue
            if not instant and not 340 <= (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days <= 380:
                continue
            if e["end"] not in best or e["filed"] > best[e["end"]][0]:
                best[e["end"]] = (e["filed"], e["val"])
        for k, (f, v) in best.items():
            if k not in out:
                out[k], filed[k] = v, f
    return pd.Series(out, dtype=float), pd.Series(filed, dtype=object)


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
    res = {k: _sec_series(gaap, *v) for k, v in SEC_TAGS.items()}
    df = pd.DataFrame({k: r[0] for k, r in res.items()})
    if df["revenue"].dropna().empty:  # p. ej. empresas extranjeras (20-F/IFRS): sin datos us-gaap
        raise ValueError("la SEC no tiene estados financieros us-gaap para este ticker")
    df["gross"] = df["gross"].fillna(df["revenue"] - df["cogs"])
    df["fcf"] = df["ocf"] - df["capex"]
    try:  # el EPS de un 10-K antiguo no refleja splits posteriores: se ajusta para compararlo con el precio actual
        sp = yf.Ticker(symbol).splits
        sp.index = sp.index.tz_localize(None)
        for end, f in res["eps"][1].items():
            fac = float(sp[sp.index > pd.Timestamp(f)].prod())
            if fac not in (1.0, 0.0) and end in df.index:
                df.loc[end, "eps"] = df.loc[end, "eps"] / fac
    except Exception:
        pass
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
PRICE_LABELS = {"1 d": "1D", "5 d": "5D", "1 m": "1M", "6 m": "6M", "AIF": "YTD", "1 a": "1Y", "5 a": "5Y", "Todos": "Max"}
PRICE_RANGES = {"1D": ("1d", "1m"), "5D": ("5d", "5m"), "1M": ("1mo", "30m"), "6M": ("6mo", "1d"),
                "YTD": ("ytd", "1d"), "1Y": ("1y", "1d"), "5Y": ("5y", "1wk"), "Max": ("max", "1mo")}
STATE_TEXT = {"REGULAR": "Mercado abierto", "PRE": "Pre-mercado", "PREPRE": "Pre-mercado",
              "POST": "Post-mercado", "POSTPOST": "Post-mercado", "CLOSED": "Mercado cerrado"}
INFO_KEYS = ("financialCurrency", "currency", "beta", "shortName", "longName", "quoteType", "fullExchangeName",
             "marketState", "bid", "ask", "bidSize", "askSize", "open", "regularMarketOpen", "previousClose",
             "regularMarketPreviousClose", "dayLow", "dayHigh", "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "volume",
             "averageVolume", "marketCap", "trailingPE", "trailingEps", "dividendRate", "targetMeanPrice",
             "totalAssets", "navPrice", "regularMarketPrice", "regularMarketTime", "postMarketPrice",
             "postMarketTime", "preMarketPrice", "preMarketTime", "forwardPE", "trailingPegRatio", "pegRatio")


def cur_symbol(cur):
    return CURRENCY_SYMBOLS.get(cur, f"{cur} " if cur else "")


@st.cache_data(ttl=30, show_spinner=False)
def info(symbol: str) -> dict:
    """Ficha de Yahoo (moneda, beta, bid/ask, volumen, capitalización, etc.)."""
    try:
        i = yf.Ticker(symbol).get_info()
    except Exception:
        i = {}
    return {k: i.get(k) for k in INFO_KEYS}


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
def price_history(symbol: str, rng: str) -> pd.DataFrame:
    period, interval = PRICE_RANGES[rng]
    h = yf.Ticker(symbol).history(period=period, interval=interval, prepost=rng in ("1D", "5D"))
    return h.reindex(columns=["Close", "Volume"]).dropna(subset=["Close"])


@st.cache_data(ttl=300, show_spinner=False)
def trading_periods(symbol: str):
    """Horas de pre-mercado, mercado regular y post-mercado de la sesión actual (epoch)."""
    try:
        cp = yf.Ticker(symbol).get_history_metadata()["currentTradingPeriod"]
        return {k: (cp[k]["start"], cp[k]["end"]) for k in ("pre", "regular", "post")}
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def calc_beta(symbol: str):
    """Beta mensual a 5 años frente al S&P 500 (como Yahoo), por si Yahoo no la entrega."""
    try:
        def monthly(sym):
            c = yf.Ticker(sym).history(period="5y", interval="1mo")["Close"]
            c.index = c.index.tz_localize(None).to_period("M")
            return c.pct_change()
        r = pd.concat([monthly(symbol), monthly("^GSPC")], axis=1, keys=["a", "b"]).dropna()
        return float(r["a"].cov(r["b"]) / r["b"].var()) if len(r) >= 24 else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def prev_year_close(symbol: str):
    try:
        h = yf.Ticker(symbol).history(period="2y", interval="1d")["Close"].dropna()
        return float(h[h.index.year < h.index[-1].year].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def close_before(symbol: str, day: str):
    """Último cierre diario anterior a `day` (referencia del cambio en 5 d y 1 m)."""
    try:
        h = yf.Ticker(symbol).history(period="3mo", interval="1d")["Close"].dropna()
        h.index = h.index.tz_localize(None).normalize()
        h = h[h.index < pd.Timestamp(day)]
        return float(h.iloc[-1]) if len(h) else None
    except Exception:
        return None


def _n(v):
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "N/D"


def _big(v):
    if not isinstance(v, (int, float)):
        return "N/D"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "MM")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"


def _lohi(lo, hi):
    return f"{_n(lo)} - {_n(hi)}" if lo is not None and hi is not None else "N/D"


TIPS = {
    "Return on Capital Employed": "ROCE: utilidad operativa ÷ (activos totales − pasivo corriente). Cuánto gana la empresa por cada dólar de capital que usa en su negocio.",
    "Operating Margin": "Margen operativo: utilidad operativa ÷ ingresos. Qué parte de cada dólar vendido queda después de los costos de operar, antes de intereses e impuestos.",
    "Net Margin": "Margen neto: utilidad neta ÷ ingresos. Qué parte de cada dólar vendido queda como ganancia final, después de todos los costos, intereses e impuestos.",
    "Promedio": "Promedio de la primera métrica elegida en los últimos N años. Aparece N/A si no hay datos suficientes para ese plazo.",
    "EPS diluido": "Ganancia por acción diluida: utilidad neta ÷ acciones en circulación, contando también opciones y bonos convertibles.",
    "EPS estimado": "Promedio de lo que esperan los analistas que gane la empresa por acción en los próximos periodos (consenso de Yahoo Finance).",
    "Proyección": "Escenario simple: EPS del último año × (1 + crecimiento anual)^años. Es un ejercicio con el supuesto que elijas, no una predicción.",
    "P/E": "Precio ÷ ganancia por acción. Cuántos dólares pagan los inversionistas por cada dólar de ganancia anual. Alto: se espera crecimiento (o está cara). Bajo: barata o con problemas. Se compara mejor con empresas del mismo sector.",
    "P/E adelante": "Precio actual ÷ EPS estimado para el próximo año. Un P/E menor al actual sugiere que se espera que las ganancias crezcan.",
    "Crecimiento EPS (CAGR)": "Tasa de crecimiento anual compuesta del EPS en los últimos años (hasta 5). Solo se calcula si el EPS inicial y el final son positivos.",
    "FCF": "Flujo de caja libre: efectivo que generan las operaciones − inversiones en activos fijos (capex). Es el dinero que queda para dividendos, recompras, deuda o crecimiento.",
    "PEG": "PEG = P/E ÷ crecimiento anual del EPS (%). Ajusta el P/E por el crecimiento: cerca de 1 suele verse como valoración razonable, menos de 1 como barata respecto a su crecimiento y más de 2 como cara. Es una guía, no una recomendación.",
    "Precio de cierre del día anterior": "Último precio al cierre de la sesión anterior. Es la base del cambio diario.",
    "Apertura": "Precio de la primera operación de la sesión de hoy.",
    "Oferta": "Precio más bajo al que hay vendedores dispuestos a vender ahora (ask), y la cantidad ofrecida.",
    "Demanda": "Precio más alto que ofrecen los compradores ahora (bid), y la cantidad demandada.",
    "Rango diario": "Precio más bajo y más alto de la sesión de hoy.",
    "Rango de 52 semanas": "Precio más bajo y más alto del último año.",
    "Volumen": "Número de acciones negociadas en la sesión de hoy.",
    "Volumen prom.": "Promedio reciente de acciones negociadas por día. Sirve para saber si el volumen de hoy es normal o inusual.",
    "Cap. de mercado": "Precio × acciones en circulación: el valor total de la empresa en bolsa.",
    "Relación precio/utilidad (TTM)": "P/E con la ganancia de los últimos 12 meses (TTM). Cuántos dólares se pagan por cada dólar de ganancia anual.",
    "Relación precio/utilidad": "P/E: precio ÷ ganancia por acción. Cuántos dólares se pagan por cada dólar de ganancia anual.",
    "BPA (TTM)": "Beneficio por acción de los últimos 12 meses (TTM).",
    "Dividendo anual": "Dividendo por acción que la empresa paga en un año.",
    "Beta (mensual por 5 años)": "Sensibilidad frente al S&P 500 con 5 años de datos mensuales. 1: se mueve igual que el mercado. Mayor que 1: más volátil. Menor que 1: menos volátil.",
    "Variación en el año (precio)": "Cambio del precio desde el último cierre del año pasado, sin contar dividendos.",
    "Precio objetivo a 1 año": "Promedio de los precios objetivo que dan los analistas a 12 meses.",
    "Activos netos": "Patrimonio total que administra el ETF.",
    "Valor de activo neto": "NAV: valor por participación de los activos que tiene el ETF.",
}
CSS = ("<style>.tip{cursor:help;border-bottom:1px dotted rgba(128,128,128,.9)}"
       ".chips{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:.85rem;margin:4px 0 10px}</style>")


def tip(label):
    t = TIPS.get(label)
    return f'<span class="tip" title="{html.escape(t, quote=True)}">{label}</span>' if t else label


def glosario(*keys):
    """Fila de nombres de métricas; al pasar el cursor se ve qué significan."""
    st.markdown('<div class="chips">' + "".join(f"<span>{tip(k)} ⓘ</span>" for k in keys) + "</div>",
                unsafe_allow_html=True)


def stat_grid(columns):
    """Cuadrícula de datos (etiqueta a la izquierda, valor a la derecha), una lista por columna."""
    rows = max(len(c) for c in columns)
    cells = "".join(
        f'<div class="sg"><span>{tip(c[r][0])}</span><b>{c[r][1]}</b></div>'
        for r in range(rows) for c in columns if r < len(c))
    st.markdown(
        "<style>.sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:0 32px}"
        ".sg{display:flex;justify-content:space-between;gap:12px;padding:8px 0;"
        "border-bottom:1px solid rgba(128,128,128,.25);font-size:.88rem}.sg b{font-weight:600;text-align:right}</style>"
        f'<div class="sgrid">{cells}</div>', unsafe_allow_html=True)


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _hora(h, m=0):
    return f"{(h % 12) or 12}:{m:02d} {'a.m.' if h < 12 else 'p.m.'}"


def _tiempo(t):
    return f"{(t.hour % 12) or 12}:{t:%M:%S} {'a.m.' if t.hour < 12 else 'p.m.'} {t.tzname() or ''}".strip()


def _fecha(d, nuevo_mes=False):
    return f"{MESES[d.month - 1]} '{d:%y}" if nuevo_mes else f"{d.day}/{d.month}"


def _epoch(e, tz):
    """Segundos epoch o datetime (con o sin zona) -> Timestamp en la zona `tz`."""
    if e is None or (not isinstance(e, datetime) and not e):
        return None
    t = pd.Timestamp(e) if isinstance(e, datetime) else pd.Timestamp(float(e), unit="s", tz="UTC")
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(tz)


def _intraday_axis(idx, rng, xaxis, tp, tz):
    """Eje X como Yahoo: sin huecos de noche/fin de semana, horas con la fecha debajo, pre/post-mercado sombreados."""
    days = sorted(set(idx.normalize()))
    hrs = pd.Series(idx.hour + idx.minute / 60)
    start_h, end_h = int(hrs.min()), int(-(-hrs.max() // 1))
    ticks, texts, shapes = [], [], []
    if rng != "1D":
        breaks = [dict(bounds=["sat", "mon"])]
        if end_h - start_h < 23:
            breaks.append(dict(bounds=[end_h % 24, start_h], pattern="hour"))
        have = set(days)
        missing = [d.strftime("%Y-%m-%d") for d in pd.date_range(days[0], days[-1], freq="D")
                   if d.weekday() < 5 and d not in have]
        if missing:
            breaks.append(dict(values=missing))
        xaxis["rangebreaks"] = breaks
    if rng == "1D":
        d = days[-1]
        if tp and tz is not None:  # el eje cubre pre-mercado y post-mercado completos
            start_h = min(start_h, _epoch(tp["pre"][0], tz).hour)
            end_h = max(end_h, _epoch(tp["post"][1], tz).hour)
        for h in range(start_h, end_h + 1):
            if h % 2 == 1:
                ticks.append(d + pd.Timedelta(hours=h)); texts.append(_hora(h))
    elif rng == "5D":
        hours = [h for h in (8, 14) if start_h <= h < end_h] or [start_h]
        for i, d in enumerate(days):
            for j, h in enumerate(hours):
                t = _hora(h) + (f"<br>{_fecha(d, i > 0 and d.month != days[i - 1].month)}" if j == 0 else "")
                ticks.append(d + pd.Timedelta(hours=h)); texts.append(t)
    else:  # 1M: una fecha cada pocos días
        step = max(1, len(days) // 8)
        for i in range(0, len(days), step):
            d = days[i]
            ticks.append(d + pd.Timedelta(hours=start_h)); texts.append(_fecha(d, i > 0 and d.month != days[i - 1].month))
    if ticks:
        xaxis.update(tickmode="array", tickvals=ticks, ticktext=texts)
    if tp and rng in ("1D", "5D") and tz is not None:
        tod = lambda e: _epoch(e, tz).tz_localize(None)
        off = lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute)
        pre, post = [(off(tod(tp[k][0])), off(tod(tp[k][1]))) for k in ("pre", "post")]
        for d in days:
            for a, b in (pre, post):
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=d + a, x1=d + b, y0=0, y1=1,
                                   fillcolor="rgba(128,128,128,.15)", line_width=0, layer="below"))
        if rng == "1D":
            xaxis["range"] = [days[-1] + pre[0], days[-1] + post[1]]
    return shapes


def render_price(symbol: str, rng: str):
    try:
        df, q = price_history(symbol, rng), quote(symbol)
    except Exception as e:
        st.error(f"No se pudo obtener el precio de {symbol}: {e}")
        return
    if df.empty:
        st.warning("Sin datos de precio para este ticker y rango.")
        return
    meta = info(symbol)
    sym = cur_symbol(meta.get("currency"))
    tz = df.index.tz
    last_ts, last_bar = df.index[-1], float(df["Close"].iloc[-1])  # último dato (incluye pre/post-mercado)
    prev = q.get("previous_close") or meta.get("previousClose")
    reg = meta.get("regularMarketPrice") or q.get("last_price") or last_bar
    state = meta.get("marketState")

    # ----- Encabezado (como Yahoo): precio regular y, si aplica, post/pre-mercado -----
    def block(col, price, base, caption):
        d = price - base if base else None
        c = (GREEN if d >= 0 else RED) if d is not None else "inherit"
        txt = f"{d:+,.2f} ({d / base * 100:+.2f}%)" if d is not None else ""
        col.markdown(
            f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap'>"
            f"<span style='font-size:2.4rem;font-weight:700'>{price:,.2f}</span>"
            f"<span style='font-size:1.15rem;font-weight:600;color:{c}'>{txt}</span></div>"
            f"<div style='opacity:.75;font-size:.85rem'>{caption}</div>", unsafe_allow_html=True)

    st.caption(f"{meta.get('fullExchangeName') or 'Bolsa'} • {meta.get('currency') or ''}")
    st.markdown(f"### {meta.get('longName') or meta.get('shortName') or symbol} ({symbol})")
    h1, h2 = st.columns(2)
    rt = _epoch(meta.get("regularMarketTime"), tz)
    block(h1, reg, prev, f"A partir del {_tiempo(last_ts)}. Mercado abierto." if state == "REGULAR"
          else (f"Al cierre: {_tiempo(rt)}" if rt else STATE_TEXT.get(state, "")))
    if state in ("POST", "POSTPOST", "CLOSED") and meta.get("postMarketPrice"):
        pt = _epoch(meta.get("postMarketTime"), tz)
        block(h2, meta["postMarketPrice"], reg, f"Poscierre: {_tiempo(pt)}" if pt else "Poscierre")
    elif state in ("PRE", "PREPRE") and meta.get("preMarketPrice"):
        pt = _epoch(meta.get("preMarketTime"), tz)
        block(h2, meta["preMarketPrice"], reg, f"Pre-mercado: {_tiempo(pt)}" if pt else "Pre-mercado")

    # ----- Variación del rango elegido (cambia con 1 d / 5 d / 1 m ...) -----
    py = prev_year_close(symbol)
    if rng == "1D" and prev:
        ref = prev
    elif rng == "YTD" and py:
        ref = py
    elif rng in ("5D", "1M"):
        ref = close_before(symbol, str(df.index[0].tz_localize(None).date())) or float(df["Close"].iloc[0])
    else:
        ref = float(df["Close"].iloc[0])
    pchg = reg - ref  # precio regular (igual que el encabezado), no el post-mercado
    up = pchg >= 0
    color, bg = (GREEN, GREEN_BG) if up else (RED, RED_BG)
    name = {v: k for k, v in PRICE_LABELS.items()}[rng]
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 12px;border-radius:6px;font-weight:700'>"
        f"{pchg / ref * 100:+.2f}%</span> <span style='opacity:.8'>&nbsp;{pchg:+,.2f} en el rango ({name})</span>",
        unsafe_allow_html=True)

    # ----- Gráfico -----
    df = df.copy()
    df.index = df.index.tz_localize(None)  # hora local de la bolsa, sin zona horaria
    intraday = rng in ("1D", "5D", "1M")
    if rng == "1D":  # cortar la línea si hay pausas largas (pre-mercado poco líquido)
        gaps = df.index.to_series().diff() > pd.Timedelta(hours=3)
        br = df.index[gaps.values] - pd.Timedelta(seconds=1)
        if len(br):
            df = pd.concat([df, pd.DataFrame(float("nan"), index=br, columns=df.columns)]).sort_index()
    c = df["Close"]
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=c, mode="lines", line=dict(width=0), fill="tozeroy", fillcolor=bg,
                    connectgaps=False, hoverinfo="skip")
    fmt_x = "%d/%m %I:%M %p" if intraday else "%d %b %Y"
    hover = f"%{{x|{fmt_x}}}<br>{sym}%{{y:,.2f}}<extra></extra>"
    near = lambda m: m | m.shift(1, fill_value=False) | m.shift(-1, fill_value=False)
    for mask, col in ((near(c >= ref), GREEN), (near(c < ref), RED)):  # verde sobre la referencia, rojo debajo
        fig.add_scatter(x=df.index, y=c.where(mask), mode="lines", line=dict(color=col, width=2),
                        connectgaps=False, hovertemplate=hover)
    fig.add_scatter(x=[df.index[-1]], y=[last_bar], mode="markers", marker=dict(color=color, size=8), hoverinfo="skip")
    vmax = df["Volume"].max()
    if pd.notna(vmax) and vmax > 0:  # volumen abajo
        fig.add_bar(x=df.index, y=df["Volume"], yaxis="y2", marker_color="rgba(128,128,128,.5)", hoverinfo="skip")
        fig.update_layout(yaxis2=dict(overlaying="y", range=[0, vmax * 4], visible=False))
    lo, hi = min(float(c.min()), ref), max(float(c.max()), ref)
    fig.add_hline(y=ref, line_dash="dash", line_color="rgba(200,200,200,.8)", annotation_text=f"{ref:,.2f}",
                  annotation_position="right", annotation_bgcolor="white", annotation_font_color="black")
    fig.add_hline(y=last_bar, line_dash="dot", line_color=color, annotation_text=f"{last_bar:,.2f}",
                  annotation_position="right", annotation_bgcolor=color, annotation_font_color="white")
    xaxis = dict(showgrid=False)
    if intraday:
        try:
            shapes = _intraday_axis(df.index, rng, xaxis, trading_periods(symbol) if rng != "1M" else None, tz)
        except Exception:
            xaxis, shapes = dict(showgrid=False), []
        for sh in shapes:
            fig.add_shape(**sh)
    pad = (hi - lo) * 0.08 or hi * 0.01
    fig.update_layout(height=460, margin=dict(l=10, r=70, t=10, b=10), showlegend=False, xaxis=xaxis,
                      yaxis=dict(range=[lo - pad, hi + pad], side="right", gridcolor="rgba(128,128,128,.25)",
                                 tickformat=",.2f"), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # ----- Datos (como Yahoo) -----
    beta = meta.get("beta")
    beta = beta if isinstance(beta, (int, float)) else calc_beta(symbol)
    ytd = f"{(reg / py - 1) * 100:+.2f}%" if py else "N/D"
    bid, ask = meta.get("bid"), meta.get("ask")
    size = lambda k: f" x {meta[k]}" if meta.get(k) else ""
    col1 = [("Precio de cierre del día anterior", _n(prev)),
            ("Apertura", _n(meta.get("open") or meta.get("regularMarketOpen"))),
            ("Oferta", f"{_n(ask)}{size('askSize')}" if ask else "N/D"),
            ("Demanda", f"{_n(bid)}{size('bidSize')}" if bid else "N/D")]
    col2 = [("Rango diario", _lohi(q.get("day_low"), q.get("day_high"))),
            ("Rango de 52 semanas", _lohi(q.get("year_low"), q.get("year_high"))),
            ("Volumen", f"{meta['volume']:,.0f}" if meta.get("volume") else "N/D"),
            ("Volumen prom.", f"{meta['averageVolume']:,.0f}" if meta.get("averageVolume") else "N/D")]
    if meta.get("quoteType") == "ETF":
        col3 = [("Activos netos", _big(meta.get("totalAssets"))), ("Valor de activo neto", _n(meta.get("navPrice"))),
                ("Relación precio/utilidad", _n(meta.get("trailingPE")))]
    else:
        col3 = [("Cap. de mercado", _big(meta.get("marketCap"))), ("Relación precio/utilidad (TTM)", _n(meta.get("trailingPE"))),
                ("BPA (TTM)", _n(meta.get("trailingEps"))), ("Dividendo anual", _n(meta.get("dividendRate")))]
    col4 = [("Beta (mensual por 5 años)", _n(beta)), ("Variación en el año (precio)", ytd)]
    if meta.get("quoteType") != "ETF":
        col4.append(("Precio objetivo a 1 año", _n(meta.get("targetMeanPrice"))))
    stat_grid([col1, col2, col3, col4])
    st.caption("Fuente: Yahoo Finance; puede tener hasta 15 minutos de retraso según la bolsa.")


# ---------- P/E, PEG y proyección ----------
ORANGE = "#f59e0b"


@st.cache_data(ttl=3600, show_spinner=False)
def daily_close(symbol: str) -> pd.DataFrame:
    """Cierres diarios (ajustados solo por splits) para calcular el P/E histórico."""
    try:
        h = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)["Close"].dropna()
        return pd.DataFrame({"date": h.index.tz_localize(None).normalize().astype("datetime64[ns]"), "px": h.values})
    except Exception:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "px": pd.Series(dtype=float)})


def same_currency(meta: dict) -> bool:
    """El P/E solo es válido si el precio y las cifras están en la misma moneda (ej. no en ADR de Taiwán)."""
    a, b = meta.get("currency"), meta.get("financialCurrency")
    return not (a and b and a != b)


def pe_series(symbol: str, data: pd.DataFrame, mode: str) -> pd.Series:
    """P/E = precio al cierre de cada periodo ÷ EPS (de 12 meses en Quarterly y TTM)."""
    eps = data["eps"].rolling(4).sum() if mode == "Quarterly" else data["eps"]
    px = daily_close(symbol)
    if px.empty:
        return pd.Series(np.nan, index=data.index)
    left = pd.DataFrame({"date": pd.to_datetime(data["date"]).astype("datetime64[ns]")})
    price = pd.merge_asof(left, px, on="date", direction="backward")["px"].values
    return pd.Series(price / eps.values, index=data.index).where(eps.values > 0)


def eps_cagr(d: pd.DataFrame, max_years: int = 5):
    e = d["eps"].dropna()
    k = min(max_years, len(e) - 1)
    if k < 1:
        return None
    e0, e1 = float(e.iloc[-1 - k]), float(e.iloc[-1])
    return ((e1 / e0) ** (1 / k) - 1) * 100 if e0 > 0 and e1 > 0 else None


def pe_peg(symbol: str, d: pd.DataFrame) -> pd.DataFrame:
    """Por año: P/E, crecimiento del EPS (CAGR de hasta 3 años) y PEG = P/E ÷ crecimiento."""
    d = d.dropna(subset=["eps"]).reset_index(drop=True)
    out = pd.DataFrame({"label": d["label"], "eps": d["eps"], "pe": pe_series(symbol, d, "Annual")})
    g = []
    for i in range(len(d)):
        k = min(3, i)
        e0, e1 = (float(d["eps"].iloc[i - k]), float(d["eps"].iloc[i])) if k else (0.0, 0.0)
        g.append(((e1 / e0) ** (1 / k) - 1) * 100 if k and e0 > 0 and e1 > 0 else np.nan)
    out["growth"] = g
    out["peg"] = out["pe"] / out["growth"].where(out["growth"] > 0)
    return out


def annual(sym: str):
    try:
        return compute(load(sym, "Annual"), "Annual")
    except Exception:
        return None


# ---------- Controles ----------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("### Métricas financieras")
for k_, d_ in (("mode", "Annual"), ("rng", "10Y"), ("cmp", "")):
    st.session_state.setdefault(k_, d_)
tc, _ = st.columns([1, 3])
symbol = tc.text_input("Ticker", "GOOG", label_visibility="collapsed", max_chars=12).strip().upper()


def _sync(k, w):  # copia el valor de un control a la variable compartida
    st.session_state[k] = st.session_state.get(w) or st.session_state[k]


def _sync_cmp(w):
    st.session_state["cmp"] = (st.session_state.get(w) or "").strip().upper()


def fin_controls(tab, mode=True, rng=True, cmp=False):
    """Controles dentro de cada pestaña financiera (no aparecen en 'Precio en vivo'); se mantienen sincronizados."""
    cols = st.columns([2, 3, 2])
    for col, k, opts, on in ((cols[0], "mode", ["Annual", "Quarterly", "TTM"], mode),
                             (cols[1], "rng", list(RANGES), rng)):
        if on:
            w = f"{k}_{tab}"
            st.session_state[w] = st.session_state[k]
            col.segmented_control(k, opts, key=w, label_visibility="collapsed", on_change=_sync, args=(k, w))
    if cmp:
        w = f"cmp_{tab}"
        st.session_state[w] = st.session_state["cmp"]
        cols[2].text_input("Comparar con", key=w, placeholder="Comparar con (ej. MSFT)", label_visibility="collapsed",
                           on_change=_sync_cmp, args=(w,))


tab_rent, tab_eps, tab_fcf, tab_peg, tab_price = st.tabs(
    ["Rentabilidad", "Ganancia por acción", "Flujo de caja libre (FCF)", "PEG ratio", "Precio en vivo"])

# ---------- Precio en vivo (independiente de los datos financieros) ----------
with tab_price:
    p1, p2 = st.columns([4, 2])
    plabel = p1.segmented_control("Rango del precio", list(PRICE_LABELS), default="1 d", label_visibility="collapsed") or "1 d"
    live = p2.toggle("Actualizar cada 30 s", value=True)

    def safe_price(sym_, rng_):
        try:
            render_price(sym_, rng_)
        except Exception as e:
            st.error(f"No se pudo mostrar el precio de {sym_}: {type(e).__name__}: {e}")
    st.fragment(run_every=30 if live else None)(safe_price)(symbol, PRICE_LABELS[plabel])

# ---------- Datos financieros ----------
mode, rng = st.session_state["mode"], st.session_state["rng"]
per_year = 1 if mode == "Annual" else 4
years = RANGES[rng]
try:
    data = compute(load(symbol, mode), mode)
    view = data if years is None else data.tail(years * per_year)
    err = None
except Exception as e:
    data = view = None
    err = f"No se pudieron obtener datos financieros de {symbol}: " + (str(e).replace(FMP_KEY, "***") if FMP_KEY else str(e))


def has_data():
    if data is None:
        st.error(err)
    return data is not None


meta = info(symbol)
cur = meta.get("financialCurrency")  # moneda en que la empresa reporta (puede no ser USD)
csym = cur_symbol(cur)
cur_note = f"Cifras en {cur} (moneda de reporte de la empresa)." if cur else ""
if cur and cur != "USD":
    cur_note += " El EPS estimado puede venir en otra moneda o por ADR: compáralo con cuidado."

# ---------- Rentabilidad ----------
with tab_rent:
    fin_controls("rent")
    if has_data():
        picked = st.pills("Series", list(METRICS), selection_mode="multi", default=["Return on Capital Employed"], label_visibility="collapsed")
        fig = go.Figure()
        for name in picked:
            k = METRICS[name]
            fig.add_bar(x=view["label"], y=view[k], name=name, marker_color=COLORS[k])
        st.plotly_chart(style(fig, ticksuffix="%"), width="stretch")
        glosario(*METRICS)
        if picked:  # Promedio de la primera serie en los últimos 1/3/5/10 años (N/A si faltan datos)
            k = METRICS[picked[0]]
            for col, n in zip(st.columns(4), [1, 3, 5, 10]):
                tail = data[k].dropna().tail(n * per_year)
                col.metric(f"{n}Y (promedio)", f"{tail.mean():.1f}%" if len(tail) >= n * per_year else "N/A", help=TIPS["Promedio"])
        with st.expander("Ver datos"):
            fcf_col = f"FCF ({cur})" if cur else "FCF"
            t = view.drop(columns=["date", "gross"]).set_index("label").rename_axis("Periodo").rename(columns={
                "eps": "EPS", "fcf": fcf_col, "operating": "Margen operativo (%)",
                "profit": "Margen neto (%)", "roce": "ROCE (%)"})
            fmt = {c: "{:,.2f}" for c in t.columns}
            fmt[fcf_col] = "{:,.0f}"
            st.dataframe(t.style.format(fmt, na_rep="—"), width="stretch")

# ---------- EPS: actual, estimado, P/E, comparativo y proyección ----------
with tab_eps:
    fin_controls("eps", cmp=True)
    if has_data():
        cmp_sym = st.session_state["cmp"] if st.session_state["cmp"] != symbol else ""
        cdata, cmeta, cview = None, {}, None
        if cmp_sym:
            try:
                cdata, cmeta = compute(load(cmp_sym, mode), mode), info(cmp_sym)
                cview = cdata if years is None else cdata.tail(years * per_year)
            except Exception as e:
                st.warning(f"No se pudo cargar {cmp_sym} para comparar: {e}")
        ad = annual(symbol)
        cagr = eps_cagr(ad) if ad is not None else None
        m = st.columns(4)
        m[0].metric("P/E (TTM)", _n(meta.get("trailingPE")), help=TIPS["P/E"])
        m[1].metric("P/E adelante", _n(meta.get("forwardPE")), help=TIPS["P/E adelante"])
        m[2].metric("BPA (TTM)", _n(meta.get("trailingEps")), help=TIPS["BPA (TTM)"])
        m[3].metric("Crecimiento EPS (CAGR)", f"{cagr:.1f}%" if cagr is not None else "N/D", help=TIPS["Crecimiento EPS (CAGR)"])

        st.markdown("**Ganancia por acción (EPS)**")
        if view["eps"].notna().any():
            fig = go.Figure()
            fig.add_bar(x=view["label"], y=view["eps"], name=f"EPS diluido {symbol}" if cview is not None else "EPS diluido",
                        marker_color=BLUE, texttemplate="%{y:,.2f}", textposition="outside")
            if cview is not None:
                fig.add_bar(x=cview["label"], y=cview["eps"], name=f"EPS diluido {cmp_sym}", marker_color=ORANGE,
                            texttemplate="%{y:,.2f}", textposition="outside")
            fwd = load_forward(symbol, mode, data["date"].iloc[-1]) if mode != "TTM" else pd.DataFrame()
            if not fwd.empty:
                fig.add_bar(x=fwd["label"], y=fwd["eps"], name="EPS estimado (consenso)",
                            marker_color=LIGHT_BLUE, marker_pattern_shape="/",
                            texttemplate="%{y:,.2f}", textposition="outside")
            st.plotly_chart(style(fig, title_text=f"EPS ({cur})" if cur else None), width="stretch")
            glosario("EPS diluido", "EPS estimado")
            if mode == "TTM":
                st.caption("Las estimaciones futuras se muestran solo en Annual y Quarterly.")
            elif fwd.empty:
                st.caption("No se encontraron estimaciones de analistas para este ticker.")
            if cur_note:
                st.caption(cur_note)
        else:
            st.info("Sin datos de EPS para este ticker.")

        st.markdown("**P/E histórico (precio al cierre de cada periodo ÷ EPS)**")
        if not same_currency(meta):
            st.info(f"No se calcula el P/E histórico: el precio está en {meta.get('currency')} y las cifras en {cur}. "
                    "Arriba se muestra el P/E actual de Yahoo.")
        else:
            fig = go.Figure()
            fig.add_scatter(x=view["label"], y=pe_series(symbol, data, mode).loc[view.index], mode="lines+markers+text",
                            name=f"P/E {symbol}" if cview is not None else "P/E", line=dict(color=BLUE),
                            texttemplate="%{y:.1f}", textposition="top center")
            if cview is not None and same_currency(cmeta):
                fig.add_scatter(x=cview["label"], y=pe_series(cmp_sym, cdata, mode).loc[cview.index], mode="lines+markers+text",
                                name=f"P/E {cmp_sym}", line=dict(color=ORANGE), texttemplate="%{y:.1f}", textposition="top center")
            st.plotly_chart(style(fig, title_text="P/E (veces)"), width="stretch")
            glosario("P/E", "P/E adelante")

        st.markdown("**Proyección del EPS**")
        base = ad.dropna(subset=["eps"]).tail(5) if ad is not None else None
        if base is None or base.empty:
            st.info("No hay EPS anual suficiente para proyectar.")
        else:
            last_eps, last_year = float(base["eps"].iloc[-1]), int(base["date"].iloc[-1].year)
            g_def = cagr if cagr is not None else 8.0
            c1, c2 = st.columns(2)
            g = c1.number_input("Crecimiento anual supuesto (%)", value=float(round(max(-50.0, min(100.0, g_def)), 1)),
                                step=0.5, help="Por defecto es el crecimiento histórico del EPS (CAGR). Cámbialo para probar otros escenarios.")
            n = c2.slider("Años a proyectar", 1, 5, 3)
            fig = go.Figure()
            fig.add_bar(x=base["label"], y=base["eps"], name="EPS real", marker_color=BLUE,
                        texttemplate="%{y:,.2f}", textposition="outside")
            fig.add_bar(x=[str(last_year + i) for i in range(1, n + 1)], y=[last_eps * (1 + g / 100) ** i for i in range(1, n + 1)],
                        name=f"Proyección ({g:+.1f}% anual)", marker_color=ORANGE, marker_pattern_shape="x",
                        texttemplate="%{y:,.2f}", textposition="outside")
            fa = load_forward(symbol, "Annual", ad["date"].iloc[-1])
            if not fa.empty:
                fig.add_bar(x=fa["label"].str.replace(" (est.)", "", regex=False), y=fa["eps"], name="Estimado de analistas",
                            marker_color=LIGHT_BLUE, marker_pattern_shape="/", texttemplate="%{y:,.2f}", textposition="outside")
            st.plotly_chart(style(fig, title_text=f"EPS ({cur})" if cur else None), width="stretch")
            glosario("Proyección", "EPS estimado", "Crecimiento EPS (CAGR)")
            if cagr is None:
                st.caption("No se pudo calcular el crecimiento histórico (EPS negativo o pocos años): usa un supuesto manual.")

        if cdata is not None:
            st.markdown(f"**Comparativo: {symbol} vs {cmp_sym}**")
            cad = annual(cmp_sym)
            rows = [{"Ticker": s_, "BPA (TTM)": m_.get("trailingEps"), "P/E (TTM)": m_.get("trailingPE"),
                     "P/E adelante": m_.get("forwardPE"),
                     "Crecimiento EPS (CAGR, %)": eps_cagr(d_) if d_ is not None else None}
                    for s_, m_, d_ in ((symbol, meta, ad), (cmp_sym, cmeta, cad))]
            st.dataframe(pd.DataFrame(rows).set_index("Ticker").astype(float).style.format("{:,.2f}", na_rep="—"), width="stretch")

# ---------- FCF ----------
with tab_fcf:
    fin_controls("fcf")
    if has_data():
        if view["fcf"].notna().any():
            div, unit = (1e9, "B") if view["fcf"].abs().max() >= 1e9 else (1e6, "M")
            y = view["fcf"] / div
            fig = go.Figure(go.Bar(
                x=view["label"], y=y, name="Free Cash Flow",
                marker_color=[BLUE if v >= 0 else RED for v in y.fillna(0)],
                texttemplate=f"{csym}%{{y:,.2f}}{unit}", textposition="outside",
                customdata=view["fcf"], hovertemplate=f"%{{x}}<br>FCF: {csym}%{{customdata:,.0f}}<extra></extra>"))
            st.plotly_chart(style(fig, ticksuffix=unit, tickprefix=csym, tickformat=",.0f"), width="stretch")
            glosario("FCF")
            if cur_note:
                st.caption(cur_note)
        else:
            st.info("Sin datos de FCF para este ticker.")

# ---------- PEG ratio ----------
with tab_peg:
    fin_controls("peg", mode=False, cmp=True)
    if has_data():
        cmp_sym = st.session_state["cmp"] if st.session_state["cmp"] != symbol else ""
        ad = annual(symbol)
        if ad is None or ad["eps"].dropna().empty:
            st.info("Sin EPS anual para calcular el PEG de este ticker.")
        elif not same_currency(meta):
            st.info(f"No se calcula el PEG: el precio está en {meta.get('currency')} y las cifras en {cur}.")
        else:
            pg = pe_peg(symbol, ad)
            pg = pg.tail(years) if years else pg
            last = pg.dropna(subset=["peg"]).tail(1)
            m = st.columns(4)
            m[0].metric("P/E (TTM)", _n(meta.get("trailingPE")), help=TIPS["P/E"])
            m[1].metric("Crecimiento EPS (CAGR)", f"{last['growth'].iloc[0]:.1f}%" if len(last) else "N/D", help=TIPS["Crecimiento EPS (CAGR)"])
            m[2].metric("PEG calculado", _n(last["peg"].iloc[0]) if len(last) else "N/D", help=TIPS["PEG"])
            m[3].metric("PEG (Yahoo)", _n(meta.get("trailingPegRatio") or meta.get("pegRatio")),
                        help="PEG que publica Yahoo; usa otra estimación de crecimiento, por eso puede diferir del calculado.")
            cg = None
            if cmp_sym:
                cad = annual(cmp_sym)
                if cad is not None and same_currency(info(cmp_sym)):
                    cg = pe_peg(cmp_sym, cad)
                    cg = cg.tail(years) if years else cg
                else:
                    st.warning(f"No se pudo calcular el PEG de {cmp_sym}.")
            if pg["peg"].notna().any() or (cg is not None and cg["peg"].notna().any()):
                fig = go.Figure()
                if cg is None:  # verde < 1, azul 1-2, rojo > 2
                    cols = [("#9aa0a6" if pd.isna(v) else GREEN if v < 1 else BLUE if v <= 2 else RED) for v in pg["peg"]]
                    fig.add_bar(x=pg["label"], y=pg["peg"], name="PEG", marker_color=cols, texttemplate="%{y:.2f}", textposition="outside")
                else:
                    fig.add_bar(x=pg["label"], y=pg["peg"], name=f"PEG {symbol}", marker_color=BLUE, texttemplate="%{y:.2f}", textposition="outside")
                    fig.add_bar(x=cg["label"], y=cg["peg"], name=f"PEG {cmp_sym}", marker_color=ORANGE, texttemplate="%{y:.2f}", textposition="outside")
                fig.add_hline(y=1, line_dash="dash", line_color="rgba(200,200,200,.8)", annotation_text="PEG = 1", annotation_position="top left")
                st.plotly_chart(style(fig, title_text="PEG (veces)"), width="stretch")
                glosario("PEG", "P/E", "Crecimiento EPS (CAGR)")
                st.caption("PEG = P/E ÷ crecimiento anual del EPS (%). El P/E usa el precio al cierre de cada ejercicio fiscal y el "
                           "crecimiento es el CAGR del EPS de hasta 3 años. Verde: menor a 1. Azul: entre 1 y 2. Rojo: mayor a 2. "
                           "Solo se calcula si el crecimiento es positivo. Es una guía de valoración, no una recomendación.")
            else:
                st.info("No hay suficientes años con EPS positivo y creciente para calcular el PEG (necesita crecimiento mayor a 0).")
            with st.expander("Ver datos del PEG"):
                t = pg.set_index("label").rename_axis("Periodo").rename(columns={
                    "eps": "EPS", "pe": "P/E", "growth": "Crecimiento EPS (%)", "peg": "PEG"})
                st.dataframe(t.style.format("{:,.2f}", na_rep="—"), width="stretch")
