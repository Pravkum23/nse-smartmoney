"""nse-smartmoney — Streamlit dashboard.

    streamlit run dashboard/app.py

Reads the same SQLite warehouse the pipeline populates, so notebook,
pipeline and dashboard always stay in sync.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_smartmoney import storage                              # noqa: E402
from nse_smartmoney.analysis import (delivery_spike_study,      # noqa: E402
                                     event_study_curve, granger_flows,
                                     granger_stock_flow)
from nse_smartmoney.config import (DII_PROFILES,                # noqa: E402
                                   SMART_MONEY_PROFILES, WATCHLIST)

st.set_page_config(page_title="NSE Smart-Money Tracker", page_icon="📈",
                   layout="wide")

PROFILE_COLORS = {
    "DII — Mutual Fund": "#1f77b4", "DII — Insurance": "#17becf",
    "DII — AIF / PMS / NBFC": "#2ca02c", "FII / FPI": "#9467bd",
    "Prop / HFT Desk": "#ff7f0e", "HNI / Family Office": "#d62728",
    "Other / Retail-HNI": "#7f7f7f",
}


@st.cache_data(ttl=600)
def load():
    return {
        "prices": storage.read("SELECT * FROM prices"),
        "flows": storage.read("SELECT * FROM fii_dii_flows"),
        "deals": storage.read("SELECT * FROM deals"),
        "features": storage.read("SELECT * FROM features"),
        "validation": storage.read("SELECT * FROM validation",
                                   parse_dates=()),
        "source": storage.get_meta("source") or "unknown",
        "last_run": storage.get_meta("last_run") or "never",
    }


try:
    D = load()
except Exception:
    st.error("Warehouse not found. Run the pipeline first:  "
             "`python -m nse_smartmoney.pipeline`")
    st.stop()

prices, flows, deals = D["prices"], D["flows"], D["deals"]
features, validation = D["features"], D["validation"]
deals["signed_qty"] = np.where(deals["side"].str.upper() == "BUY",
                               deals["qty"], -deals["qty"])
deals["value_cr"] = deals["qty"] * deals["price"] / 1e7

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 NSE Smart-Money")
st.sidebar.caption(f"source: **{D['source']}** · last run: "
                   f"{str(D['last_run'])[:16]}")
symbol = st.sidebar.selectbox("Focused ticker", WATCHLIST,
                              index=WATCHLIST.index("BEL"))
horizon = st.sidebar.radio("Validation horizon (days)", [5, 10], index=1,
                           horizontal=True)
lookback = st.sidebar.slider("Lookback (trading days)", 60, 320, 180, 20)
min_events = st.sidebar.slider("Min events for validation", 3, 15, 5)
if D["source"] == "sample":
    st.sidebar.warning("Running on bundled **sample data** (fictional "
                       "participants). Run `python -m nse_smartmoney."
                       "pipeline --live` on a machine with NSE access "
                       "for real data.")

sym_px = prices[prices.symbol == symbol].sort_values("date").tail(lookback)
sym_ft = features[features.symbol == symbol].sort_values("date") \
    .tail(lookback)
sym_deals = deals[deals.symbol == symbol]

# ---------------------------------------------------------------- header
st.title(f"{symbol} — institutional smart-money view")
last = sym_ft.iloc[-1] if len(sym_ft) else None
c1, c2, c3, c4, c5 = st.columns(5)
if last is not None:
    c1.metric("Accumulation score", f"{last.accum_score:.0f}/100")
    sig = "Accumulation" if last.accum_score >= 60 else \
          "Distribution" if last.accum_score <= 40 else "Neutral"
    c2.metric("Signal", sig)
    r5 = sym_ft.ret_1d.tail(5).add(1).prod() - 1
    c3.metric("5D return", f"{r5:+.2%}")
    c4.metric("Delivery z-score", f"{(last.deliv_z or 0):+.2f}")
    smart_net = sym_deals[sym_deals.profile.isin(SMART_MONEY_PROFILES)] \
        ["signed_qty"].sum()
    c5.metric("Smart-money net (shares)", f"{smart_net:,.0f}")

tabs = st.tabs(["Overview", "FII/DII Flows", "Bulk & Block Deals",
                "Delivery Analysis", "Validation", "Screener",
                "Raw Tables"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    lcol, rcol = st.columns([2, 1])
    with lcol:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=sym_px.date, open=sym_px.open, high=sym_px.high,
            low=sym_px.low, close=sym_px.close, name=symbol))
        ev = sym_deals[(sym_deals.profile.isin(DII_PROFILES))
                       & (sym_deals.side == "BUY")]
        ev = ev[ev.date.isin(sym_px.date)]
        if len(ev):
            ev_px = ev.merge(sym_px[["date", "low"]], on="date")
            fig.add_trace(go.Scatter(
                x=ev_px.date, y=ev_px.low * 0.985, mode="markers",
                marker=dict(symbol="triangle-up", size=11, color="#2ca02c"),
                name="DII bulk/block buy"))
        fig.update_layout(height=420, xaxis_rangeslider_visible=False,
                          margin=dict(t=30, b=10),
                          title="Price with DII deal markers")
        st.plotly_chart(fig, use_container_width=True)
    with rcol:
        prof = sym_deals.groupby("profile")["signed_qty"].sum() \
            .sort_values()
        figp = px.bar(prof, orientation="h",
                      color=prof.index.map(PROFILE_COLORS),
                      color_discrete_map="identity",
                      title="Net deal flow by participant profile")
        figp.update_layout(height=420, showlegend=False,
                           yaxis_title="", xaxis_title="net shares")
        st.plotly_chart(figp, use_container_width=True)

    st.subheader("Top participants in this stock")
    top = (sym_deals.groupby(["client", "profile"])
           .agg(net_qty=("signed_qty", "sum"),
                gross_value_cr=("value_cr", "sum"),
                n_deals=("qty", "count")).reset_index()
           .sort_values("net_qty", ascending=False))
    st.dataframe(top, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- flows
with tabs[1]:
    piv = flows.pivot_table(index="date", columns="category",
                            values="net_cr", aggfunc="sum").tail(lookback)
    figf = go.Figure()
    for col, color in (("DII", "#1f77b4"), ("FII/FPI", "#9467bd")):
        if col in piv:
            figf.add_trace(go.Bar(x=piv.index, y=piv[col], name=col,
                                  marker_color=color))
    figf.update_layout(barmode="relative", height=380,
                       title="Daily FII vs DII net flow (₹ crore, cash "
                             "market)")
    st.plotly_chart(figf, use_container_width=True)

    cum = piv.cumsum()
    figc = px.line(cum, title="Cumulative net flow (₹ crore)")
    figc.update_layout(height=320)
    st.plotly_chart(figc, use_container_width=True)

    st.subheader("Do flows lead the market? (Granger causality)")
    with st.spinner("running Granger tests..."):
        g = granger_flows(features, flows)
    st.dataframe(g, use_container_width=True, hide_index=True)
    st.caption("p < 0.05 ⇒ lagged flow adds predictive information about "
               "next-day market returns beyond the market's own history.")

# ---------------------------------------------------------------- deals
with tabs[2]:
    st.subheader(f"Bulk & block deals — {symbol}")
    show = sym_deals.sort_values("date", ascending=False)
    st.dataframe(show[["date", "client", "profile", "side", "qty",
                       "price", "value_cr", "kind"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Cumulative net buying — top participants")
    top5 = (sym_deals.groupby("client")["signed_qty"].sum()
            .abs().sort_values(ascending=False).head(6).index)
    cum_df = []
    for cl in top5:
        d = sym_deals[sym_deals.client == cl].sort_values("date")
        d = d.groupby("date")["signed_qty"].sum().cumsum().rename(cl)
        cum_df.append(d)
    if cum_df:
        figd = px.line(pd.concat(cum_df, axis=1).ffill(),
                       title="Cumulative net position (shares) — the "
                             "'who keeps buying?' chart")
        figd.update_layout(height=400)
        st.plotly_chart(figd, use_container_width=True)

    st.subheader("Market-wide smart-money deal flow by profile")
    mw = (deals.groupby(["profile"])
          .agg(net_qty=("signed_qty", "sum"),
               gross_value_cr=("value_cr", "sum"),
               n=("qty", "count")).reset_index())
    st.dataframe(mw, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- delivery
with tabs[3]:
    figv = go.Figure()
    figv.add_trace(go.Scatter(x=sym_ft.date, y=sym_ft.deliv_z,
                              name="delivery z-score",
                              line=dict(color="#1f77b4")))
    figv.add_trace(go.Scatter(x=sym_ft.date, y=sym_ft.volume_z,
                              name="volume z-score",
                              line=dict(color="#ff7f0e", dash="dot")))
    spikes = sym_ft[sym_ft.deliv_spike == 1]
    figv.add_trace(go.Scatter(x=spikes.date, y=spikes.deliv_z,
                              mode="markers", name="accumulation day",
                              marker=dict(size=10, color="#2ca02c")))
    figv.add_hline(y=1.5, line_dash="dash", line_color="gray")
    figv.update_layout(height=380, title="Delivery & volume z-scores — "
                       "spikes = volume-confirmed conviction days")
    st.plotly_chart(figv, use_container_width=True)

    st.subheader("Delivery-spike edge across the universe")
    ds = delivery_spike_study(features)
    st.dataframe(ds.round(4), use_container_width=True, hide_index=True)
    st.caption("edge = mean 5-day forward return on spike days minus "
               "all other days.")

# ---------------------------------------------------------------- validation
with tabs[4]:
    st.subheader(f"Participant validation — {horizon}-day forward "
                 f"returns after repeated net buying")
    v = validation[(validation.horizon == horizon)
                   & (validation.n_events >= min_events)].copy()
    v = v.sort_values("p_value")
    st.dataframe(
        v[["symbol", "client", "profile", "n_events", "mean_fwd_ret",
           "median_fwd_ret", "win_rate", "p_value", "significant"]]
        .style.format({"mean_fwd_ret": "{:+.2%}",
                       "median_fwd_ret": "{:+.2%}",
                       "win_rate": "{:.0%}", "p_value": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    n_sig = int(v.significant.sum())
    st.caption(f"{n_sig} of {len(v)} participant-symbol pairs pass the "
               f"one-sided test at α=5 %. With {len(v)} tests, ≈"
               f"{len(v) * 0.05:.1f} false positives are expected by "
               "chance — treat single hits as leads, not proof.")

    st.subheader(f"Event study — {symbol}")
    sel_clients = v[v.symbol == symbol].client.tolist()
    if sel_clients:
        cl = st.selectbox("Participant", sel_clients)
        ev = deals[(deals.symbol == symbol) & (deals.client == cl)
                   & (deals.side == "BUY")][["date", "symbol"]] \
            .drop_duplicates()
        curve = event_study_curve(prices, ev)
        if len(curve):
            fige = px.line(curve, x="rel_day", y=["mean_cumret",
                                                  "median_cumret"],
                           title=f"Average price path around {cl} "
                                 f"net-buy days (n={curve.n_events.iloc[0]})")
            fige.add_vline(x=0, line_dash="dash")
            fige.update_layout(height=380,
                               yaxis_tickformat=".1%")
            st.plotly_chart(fige, use_container_width=True)
    else:
        st.info("No participant on this ticker clears the minimum-event "
                "threshold — lower it in the sidebar or pick another "
                "ticker.")

    st.subheader(f"Does smart-money flow *lead* {symbol}? (Granger)")
    gs = granger_stock_flow(features, symbol)
    if len(gs):
        st.dataframe(gs, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough deal-flow variation for this ticker.")

# ---------------------------------------------------------------- screener
with tabs[5]:
    st.subheader("Cross-universe accumulation screener")
    latest = features.sort_values("date").groupby("symbol").tail(1)
    val_best = (validation[validation.horizon == horizon]
                .sort_values("p_value").groupby("symbol").head(1)
                [["symbol", "client", "p_value", "win_rate",
                  "mean_fwd_ret"]]
                .rename(columns={"client": "best_participant"}))
    scr = latest[["symbol", "accum_score", "deliv_z", "volume_z",
                  "smart_deal_net_qty"]].merge(val_best, on="symbol",
                                               how="left")
    scr = scr.sort_values(["accum_score"], ascending=False)
    st.dataframe(
        scr.style.format({"accum_score": "{:.0f}", "deliv_z": "{:+.2f}",
                          "volume_z": "{:+.2f}",
                          "smart_deal_net_qty": "{:,.0f}",
                          "mean_fwd_ret": "{:+.2%}",
                          "win_rate": "{:.0%}", "p_value": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.caption("Ranked by composite accumulation score. best_participant "
               "= lowest-p validated accumulator on the ticker.")

# ---------------------------------------------------------------- raw
with tabs[6]:
    t = st.selectbox("Table", ["features", "deals", "prices",
                               "fii_dii_flows", "validation"])
    st.dataframe(D[{"features": "features", "deals": "deals",
                    "prices": "prices", "fii_dii_flows": "flows",
                    "validation": "validation"}[t]].tail(500),
                 use_container_width=True)

st.markdown("---")
st.caption("Education & research only — not investment advice. Bulk/block "
           "deal disclosures identify counterparties, not intent; past "
           "patterns do not guarantee future returns.")
