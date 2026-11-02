"""Cottonil BI & AI — مركز قيادة الأعمال.  التشغيل:  streamlit run app.py"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import bootstrap, data, insights, inventory as inv, metrics as M, models, ui
from src.assistant import Assistant, Ctx
from src.data import CH_AR
from src.ui import AMBER, BLUE, MINT, RED, esc

st.set_page_config(page_title="Cottonil | مركز قيادة الأعمال", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")
PAGES = ["نظرة عامة", "المبيعات", "الفروع", "المخزون", "التوقعات", "العملاء", "التوصيات", "المساعد الذكي", "صحة البيانات"]
DIMS = {"الفروع": "branch", "المحافظات": "governorate", "الأنواع": "category", "المنتجات": "product", "القنوات": "channel"}
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
USER = os.getenv("BI_USER_NAME", "مراد")


@st.cache_data
def deployed_revision():
    """Read the actual checkout, never infer deployed code from remote HEAD."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
            cwd=data.ROOT, stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "غير متاح"


def H(x: str):
    """HTML للصفحة. الأرقام والتواريخ بتتعزل (bidi) عشان تتعرض بالترتيب الصح جوه الجمل العربي."""
    st.markdown(x if x.lstrip().startswith("<style") else ui.bidi_html(x), unsafe_allow_html=True)


def mlabel(ym: str, short=True) -> str:
    y, m = ym.split("-")
    return f"{AR_MONTHS[int(m) - 1]}" + (f" {y[2:]}" if short else f" {y}")


# ---------------------------------------------------------------- البيانات والكاش
def stamp() -> float:
    p = data.PROC / "meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="بحمّل البيانات...")
def _load(s: float):
    return data.load()


def DD():
    return _load(stamp())


def sel(key):
    s, e, b, c, p, ch = key
    return data.apply_filters(DD().sales, s, e, list(b) or None, list(c) or None, list(p) or None, list(ch) or None)


@st.cache_data(show_spinner=False, max_entries=200)
def c_kpis(s, key):
    return M.kpis(sel(key))


@st.cache_data(show_spinner=False, max_entries=200)
def c_dim(s, key, dim):
    return M.by_dim(sel(key), DD(), dim)


@st.cache_data(show_spinner=False, max_entries=100)
def c_series(s, key, freq):
    df = sel(key)
    d = df.date
    k = d.dt.normalize() if freq == "D" else d.dt.to_period("W-SAT").dt.start_time if freq == "W" else d.dt.to_period("M").dt.start_time
    return df.groupby(k).agg(net=("net", "sum"), margin=("margin", "sum"), orders=("order", "nunique"))


@st.cache_data(show_spinner=False, max_entries=50)
def c_ontime(s, start, end, freq):
    d = DD().deliveries
    if not len(d):
        return pd.Series(dtype=float)
    x = d[(d.ship_date >= pd.Timestamp(start)) & (d.ship_date <= pd.Timestamp(end))]
    k = x.ship_date.dt.normalize() if freq == "D" else x.ship_date.dt.to_period("W-SAT").dt.start_time if freq == "W" else x.ship_date.dt.to_period("M").dt.start_time
    return (1 - x.groupby(k).late.mean()) * 100


@st.cache_data(show_spinner="بحسب المخزون...", max_entries=20)
def c_inv(s, params):
    df = inv.intel(DD(), **dict(params))
    return df, inv.transfers(DD(), df)


@st.cache_data(show_spinner="بدوّر على المشاكل والفرص...")
def c_insights(s):
    return insights.generate(DD())


def split_big(big: str):
    p = big.split(" ", 1)
    return (p[0], p[1]) if len(p) == 2 and p[0][:1].isdigit() else (big, "")


def go_to(page: str):
    st.session_state["nav"] = page


def link_button(label: str, page: str, key: str):
    st.button(label, key=f"link_{key}", on_click=go_to, args=(page,))


def table(df, money=(), pct=(), ints=(), height=None):
    df = df.copy()
    for k in list(money) + list(ints):
        if k in df.columns:
            df[k] = df[k].round(0)
    df = df[list(df.columns)[::-1]]            # RTL: أول عمود (الاسم) يظهر على اليمين
    cfg = {k: st.column_config.NumberColumn(k, format="localized") for k in money}
    cfg.update({k: st.column_config.NumberColumn(k, format="%.1f%%") for k in pct})
    cfg.update({k: st.column_config.NumberColumn(k, format="%d") for k in ints})
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg, **({"height": height} if height else {}))


def thin_axis(fig, xs, step=2):
    fig.update_xaxes(type="category", tickmode="array", tickvals=xs[::step] if len(xs) > 8 else xs, ticktext=xs[::step] if len(xs) > 8 else xs)
    return fig


def chart(fig, h=340, legend=True, cap: str | None = None, key=None):
    st.plotly_chart(ui.fig_style(fig, h, legend), width="stretch", config={"displayModeBar": False}, key=key)
    if cap:
        H(f'<div class="cap">💡 {esc(cap)}</div>')


# ---------------------------------------------------------------- الشريط الجانبي والفلاتر
def sidebar(D, rec_count: int):
    H(ui.css(rec_count))
    with st.sidebar:
        H('<div class="brand"><div class="logo">C</div><div><b>COTTONIL</b><small>مركز قيادة الأعمال</small></div></div>')
        st.radio("الصفحة", PAGES, key="nav", label_visibility="collapsed")
        st.caption("نسخة التشغيل: " + deployed_revision())
        reason = data.needs_rebuild()
        age = (time.time() - stamp()) / 3600
        when = "النهارده" if age < 24 else f"من {int(age // 24)} يوم"
        H(f'<div class="fresh {"warn" if reason else ""}"><b>{"في بيانات جديدة" if reason else "البيانات محدّثة"}</b>'
          f'آخر يوم في الداتا: {D.last:%Y-%m-%d}<br>{D.meta["rows"] / 1e6:.1f} مليون سطر · اتبنت {when}' + (f"<br>{esc(reason)}" if reason else "") + "</div>")


def _ask_from_top():
    q = st.session_state.get("topq", "").strip()
    if q:
        st.session_state["pending_q"] = q
        st.session_state["nav"] = "المساعد الذكي"
        st.session_state["topq"] = ""


def _reset_filters():
    for k in ("f_gov", "f_br", "f_cat", "f_prod", "f_ch"):
        st.session_state[k] = []


def controls(D):
    """الفترة + الفلاتر + البحث. بيرجع F اللي كل الصفحات بتستخدمه."""
    L = D.last.normalize()
    presets = {"آخر 30 يوم": (L - pd.Timedelta(days=29), L), "آخر 3 شهور": (L - pd.Timedelta(days=89), L), "آخر 12 شهر": (L - pd.Timedelta(days=364), L),
               "الشهر اللي فات": ((L.to_period("M") - 1).start_time, (L.to_period("M") - 1).end_time.normalize()),
               "السنة دي حتى الآن": (pd.Timestamp(L.year, 1, 1), L), "السنة اللي فاتت": (pd.Timestamp(L.year - 1, 1, 1), pd.Timestamp(L.year - 1, 12, 31)),
               "كل الفترة": (D.first.normalize(), L), "فترة مخصصة": None}
    c1, c2, c3 = st.columns([1.15, 1, 1.7])
    with c3:
        st.text_input("بحث", key="topq", placeholder="اسأل عن أي رقم...", label_visibility="collapsed", on_change=_ask_from_top)
    with c2:
        pop = st.popover("الفلاتر", width="stretch")
    with c1:
        p = st.selectbox("الفترة", list(presets), index=2, label_visibility="collapsed", key="period")
    if presets[p] is None:
        r = pop.date_input("من - إلى", (L - pd.Timedelta(days=89), L), min_value=D.first.date(), max_value=L.date())
        start, end = (pd.Timestamp(r[0]), pd.Timestamp(r[-1])) if len(r) else (L - pd.Timedelta(days=89), L)
    else:
        start, end = presets[p]
    with pop:
        mode = "yoy" if st.selectbox("قارن بـ", ["الفترة اللي قبلها مباشرة", "نفس الفترة السنة اللي فاتت"], key="f_cmp").startswith("نفس") else "prev"
        govs = st.multiselect("المحافظة", sorted(D.branches.governorate.unique()), key="f_gov", placeholder="كل المحافظات")
        bo = D.branches[D.branches.governorate.isin(govs)] if govs else D.branches
        bsel = st.multiselect("الفرع", bo.sort_values("label").label.tolist(), key="f_br", placeholder="كل الفروع")
        cats_ar = st.multiselect("نوع المنتج", sorted(D.cat_ar.values()), key="f_cat", placeholder="كل الأنواع")
        cats = [k for k, v in D.cat_ar.items() if v in cats_ar]
        po = D.products[D.products.category.isin(cats)] if cats else D.products
        psel = st.multiselect("المنتج", sorted(po.product_name_ar.tolist()), key="f_prod", placeholder="كل المنتجات")
        chs_ar = st.multiselect("قناة البيع", list(CH_AR.values()), key="f_ch", placeholder="كل القنوات")
        st.button("مسح الفلاتر", on_click=_reset_filters)
    b_ids = D.branches[D.branches.label.isin(bsel)].branch_id.tolist() or (bo.branch_id.tolist() if govs else [])
    prods = D.products[D.products.product_name_ar.isin(psel)].product_id.tolist()
    chs = [k for k, v in CH_AR.items() if v in chs_ar]
    F = dict(start=start, end=end, mode=mode, key=(str(start.date()), str(end.date()), tuple(b_ids), tuple(cats), tuple(prods), tuple(chs)))
    F["ps"], F["pe"] = M.prev_range(start, end, mode)
    F["pkey"] = (str(F["ps"].date()), str(F["pe"].date())) + F["key"][2:]
    F["label"] = f"{start:%Y-%m-%d} ← {end:%Y-%m-%d}"
    parts = [f"{len(b_ids)} فرع" if b_ids else "كل الفروع", f"{len(cats)} نوع" if cats else "كل الأنواع"] + ([f"{len(prods)} منتج"] if prods else []) + ([" / ".join(chs_ar)] if chs else [])
    F["scope"] = " · ".join(parts)
    F["filtered"] = bool(b_ids or cats or prods or chs)
    return F


def title(txt: str, sub: str = ""):
    H(f'<div class="ptitle">{esc(txt)}</div><div class="psub">{esc(sub)}</div>')


PAGE_SUB = {"المبيعات": lambda F: f"{F['label']} · {F['scope']}", "الفروع": lambda F: f"{F['label']} · ربحية كل فرع بعد المرتبات والمصروفات",
            "المخزون": lambda F: "إيه اللي هيخلص؟ وإيه اللي واقف؟ ونعمل إيه؟", "التوقعات": lambda F: "الطلب المتوقع في الشهور الجاية، ومدى دقة التوقع",
            "العملاء": lambda F: f"{F['label']} · شرائح، قيمة، وفرص", "التوصيات": lambda F: "مشاكل وفرص طلعت من حسابات فعلية على الداتا، مرتبة بالأولوية",
            "المساعد الذكي": lambda F: "اسأل بالعربي. كل رقم من الداتا، ومعاه مصدره والفترة", "صحة البيانات": lambda F: "إحنا واثقين في الأرقام قد إيه؟ وإزاي نحدّثها؟"}


# ---------------------------------------------------------------- عرض ردود المساعد
def df_html(df: pd.DataFrame, max_rows: int | None = 12) -> str:
    return ui.dataframe_html(df, max_rows)


def render_answer(a, compact=False, top=True, chart_key=None):
    """رد المساعد: ملخص، حقائق، توقع، استنتاج، اقتراحات، جدول، رسم، مصدر، ومين اللي رد (Groq ولا المحرك المحلي)."""
    if a.parts:
        if a.summary:
            H(f'<div class="sumbox"><span class="lbl-fact">ملخص</span> {esc(a.summary)}</div>')
        for i, p in enumerate(a.parts):
            H(f'<div class="part"><div class="part-t">{i + 1}. {esc(p.title)}</div></div>')
            render_answer(p, compact, top=False, chart_key=f"{chart_key}_part_{i}" if chart_key is not None else None)
    else:
        if a.summary:
            H(f'<div class="sumbox"><span class="lbl-fact">ملخص</span> {esc(a.summary)}</div>')
        H(f'<div style="font-size:16px;line-height:1.9">{esc(a.headline or a.text)}</div>')
        for lbl, cls, items in (("حقائق من الداتا", "lbl-fact", a.facts), ("توقع", "lbl-fc", a.forecasts), ("استنتاج محتمل (مش مثبت)", "lbl-inf", a.inferences), ("اقتراحات مبنية على الأرقام", "lbl-fact", a.actions)):
            if items and not (compact and cls not in ("lbl-fact", "lbl-fc")):
                H(f'<div style="margin-top:10px"><span class="{cls}">{lbl}</span></div>' + '<ul class="answer-list">' + "".join(f'<li>{esc(x)}</li>' for x in (items[:2] if compact else items)) + '</ul>' )
        for x in a.notes:
            H(f'<div class="cap">ℹ️ {esc(x)}</div>')
        if not compact:
            if a.table is not None and len(a.table):
                H(df_html(a.table, max_rows=None))
            if a.fig is not None:
                fig = go.Figure(a.fig)
                fig.update_layout(title=None)
                chart(fig, max(300, fig.layout.height or 300), key=chart_key)
        if a.source:
            H('<div class="src">📎 المصدر: ' + " · ".join(f"<b>{esc(k)}</b>: {esc(v)}" for k, v in a.source.items()) + "</div>")
    if top and (a.mode or a.llm_note):
        chips = f'<span class="chip {"ai" if a.mode.startswith("Groq") else ""}">الرد من: {esc(a.mode)}</span>'
        H(f'<div class="chip-row">{chips}</div>')
        if a.llm_note:
            H(ui.note(esc(a.llm_note), "warn"))


# ---------------------------------------------------------------- 1) نظرة عامة
def kpi_row(F):
    cur, prv = c_kpis(stamp(), F["key"]), c_kpis(stamp(), F["pkey"])
    n = (F["end"] - F["start"]).days + 1
    freq = "D" if n <= 45 else "W" if n <= 240 else "M"
    g = c_series(stamp(), F["key"], freq)
    D = DD()
    ot, otp = M.on_time_rate(D, F["start"], F["end"]), M.on_time_rate(D, F["ps"], F["pe"])
    cols = st.columns(4)
    v, u = ui.money_parts(cur["net"])
    cols[0].markdown(ui.kpi_card("صافي المبيعات", v, u, ui.delta_pill(cur["net"], prv["net"]), g.net.values), unsafe_allow_html=True)
    cols[1].markdown(ui.kpi_card("هامش الربح", f"{cur['margin_pct']:.1f}", "%", ui.delta_pill(cur["margin_pct"], prv["margin_pct"], points=True), (g.margin / g.net * 100).values, AMBER), unsafe_allow_html=True)
    v, u = ui.money_parts(cur["basket"])
    cols[2].markdown(ui.kpi_card("متوسط الفاتورة", v, u, ui.delta_pill(cur["basket"], prv["basket"]), (g.net / g.orders).values, MINT), unsafe_allow_html=True)
    if ot is not None:
        cols[3].markdown(ui.kpi_card("التوصيل في الميعاد", f"{ot:.0f}", "%", ui.delta_pill(ot, otp or ot, points=True), c_ontime(stamp(), F["start"], F["end"], freq).values, BLUE), unsafe_allow_html=True)
    return cur, prv


def overview_body(D, F, rec):
    kpi_row(F)
    if F["filtered"]:
        H(ui.note("الأرقام والمقارنات فوق متفلترة حسب اختيارك. المشاكل والتوصيات والتوقعات على مستوى الشركة كلها.", ""))
    st.write("")
    top = rec.head(3)
    cols = st.columns(3)
    for i, (col, x) in enumerate(zip(cols, top.itertuples())):
        with col:
            with st.container(key=f"card_p{i}"):
                big, unit = split_big(x.big)
                H(ui.problem_card(i + 1, x.priority, x.title, big, unit, x.label, x.items[:3], x.evidence.split(". ")[0] + ".", x.ring, x.kind))
                link_button("شوف الحل ›", x.page, f"p{i}")
    st.write("")
    a, b = st.columns([1.55, 1])
    with a:
        with st.container(key="card_fc"):
            H(ui.sect("المبيعات مقابل التوقع", "بالمليون جنيه · شهريًا · كل الشركة"))
            hist = models.monthly(D.sales, "net")
            f = D.forecast[(D.forecast.level == "total") & (D.forecast.target == "net")].sort_values("month")
            xs = [mlabel(str(p)) for p in hist.index[-12:]]
            fig = go.Figure()
            fig.add_scatter(x=xs, y=hist.values[-12:] / 1e6, name="فعلي", mode="lines", line=dict(color=MINT, width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(94,234,176,.10)")
            fx = [xs[-1]] + [mlabel(m) for m in f.month]
            fig.add_scatter(x=fx, y=[hist.values[-1] / 1e6] + list(f.hi / 1e6), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=fx, y=[hist.values[-1] / 1e6] + list(f.lo / 1e6), line=dict(width=0), mode="lines", fill="tonexty", fillcolor="rgba(246,185,77,.14)", name="المدى المحتمل", hoverinfo="skip")
            fig.add_scatter(x=fx, y=[hist.values[-1] / 1e6] + list(f.forecast / 1e6), name="توقع", mode="lines+markers", line=dict(color=AMBER, width=2.6, dash="dash"))
            fig.update_yaxes(ticksuffix=" م")
            thin_axis(fig, xs + [mlabel(m) for m in f.month])
            chart(fig, 300, True)
            nxt = f.iloc[0]
            H(f'<div class="cap">الشهر الجاي ({mlabel(nxt.month, False)}): المتوقع <b>{M.money(nxt.forecast)}</b> (من {M.money(nxt.lo)} لـ {M.money(nxt.hi)}). الخط المتقطع توقع والمنطقة حواليه هي المدى المحتمل (80%). الإيراد بيفترض ثبات الأسعار (الأسعار بتزيد كل يناير)، وتقدر تجرّب زيادة من صفحة التوقعات.</div>')
    with b:
        with st.container(key="card_top"):
            H(ui.sect("أعلى الفروع", "صافي المبيعات · م ج.م · الفترة المختارة"))
            t = c_dim(stamp(), F["key"], "branch").sort_values("net", ascending=False).head(7)
            H(ui.ranked([(r.label, r.net, f"{r.net / 1e6:.2f} م") for r in t.itertuples()]))
            link_button("كل الفروع ›", "الفروع", "tb")
    st.write("")
    a, b = st.columns([1.55, 1])
    with a:
        assistant_panel(D, F)
    with b:
        with st.container(key="card_quick"):
            H(ui.sect("توصيات سريعة", "الأثر المقدر بالجنيه"))
            icons = {"حرجة": ("🔥", RED), "مهمة": ("⚡", AMBER), "فرصة": ("💡", MINT), "معلومة": ("ℹ️", BLUE)}
            for x in rec.iloc[3:6].itertuples():
                ic, c = icons[x.priority]
                H(ui.quick_row(ic, c, x.title, (f"{x.big} · " if x.big else "") + x.label))
            link_button("كل التوصيات ›", "التوصيات", "qr")


def assistant_panel(D, F):
    with st.container(key="card_ai"):
        H(ui.sect("المساعد الذكي", "اسأل بالعربي، والإجابة من أرقام الشركة الفعلية"))
        q = st.session_state.get("quick")
        if q:
            H(f'<div style="background:#141f1c;border-radius:18px;padding:10px 16px;width:fit-content;margin:6px 0">{esc(q["q"])}</div>')
            with st.container(key="card_ai_ans"):
                render_answer(q["a"], compact=True)
            link_button("افتح الرد كامل مع الجداول ›", "المساعد الذكي", "ai")
        c1, c2 = st.columns([5, 1])
        txt = c1.text_input("سؤال سريع", key="quickq", placeholder="اسأل بالعربي عن مبيعاتك...", label_visibility="collapsed")
        go_ = c2.button("ابعت", type="primary", width="stretch")
        chips = st.columns(3)
        for i, s in enumerate(["إيه أعلى 5 فروع الشهر اللي فات؟", "ليه المبيعات اتغيرت الشهر ده؟", "إيه اللي هيخلص من المخزون؟"]):
            if chips[i].button(s, key=f"chip{i}", width="stretch"):
                txt, go_ = s, True
        if go_ and txt:
            asst = get_assistant(D, F)
            st.session_state["quick"] = {"q": txt, "a": asst.ask(txt)}
            st.rerun()


# ---------------------------------------------------------------- 2) المبيعات
def drill_key(F, dim, k):
    b, c, p, ch = list(F["key"][2]), list(F["key"][3]), list(F["key"][4]), list(F["key"][5])
    if dim == "branch":
        b = [k]
    elif dim == "governorate":
        b = [x for x, g in DD().branch_gov.items() if g == k]
    elif dim == "category":
        c = [k]
    elif dim == "product":
        p = [k]
    elif dim == "channel":
        ch = [k]
    return (F["key"][0], F["key"][1], tuple(b), tuple(c), tuple(p), tuple(ch))


def page_sales(D, F):
    tabs = st.tabs(list(DIMS))
    for tab, (name, dim) in zip(tabs, DIMS.items()):
        with tab:
            cur, prv = c_dim(stamp(), F["key"], dim), c_dim(stamp(), F["pkey"], dim).set_index("key").net
            if not len(cur):
                st.info("مفيش بيانات بالفلاتر دي.")
                continue
            cur = cur.assign(growth=(cur.net / cur.key.map(prv) - 1) * 100).sort_values("net", ascending=False)
            n = min(st.slider("عدد العناصر", 5, min(25, len(cur)), min(10, len(cur)), key=f"n_{dim}") if len(cur) > 5 else len(cur), len(cur))
            with st.container(key=f"card_s_{dim}"):
                H(ui.sect(f"أعلى {n} من {name} في المبيعات", "الرقم = المبيعات · هامش الربح · النمو عن فترة المقارنة"))
                H(ui.ranked([(r.label, r.net, f"{M.money(r.net)} · {r.margin_pct:.0f}% · " + (f"{r.growth:+.0f}%" if r.growth == r.growth else "جديد")) for r in cur.head(n).itertuples()]))
            with st.expander("عرض الجدول الكامل"):
                t = cur.rename(columns={"label": "الاسم", "net": "المبيعات", "margin": "الربح", "margin_pct": "هامش الربح %", "qty": "القطع", "orders": "الفواتير", "disc_pct": "الخصم %", "basket": "متوسط الفاتورة", "growth": "النمو %"})
                table(t[["الاسم", "المبيعات", "الربح", "هامش الربح %", "الخصم %", "القطع", "الفواتير", "متوسط الفاتورة", "النمو %"]], money=("المبيعات", "الربح", "القطع", "الفواتير", "متوسط الفاتورة"), pct=("هامش الربح %", "الخصم %", "النمو %"), height=320)
            pick = st.selectbox("تعمّق في عنصر واحد", ["—"] + cur.label.tolist(), key=f"d_{dim}")
            if pick != "—":
                dk = drill_key(F, dim, cur[cur.label == pick].key.iloc[0])
                x = c_kpis(stamp(), dk)
                cols = st.columns(4)
                for c_, (lb, vv) in zip(cols, [("المبيعات", M.money(x["net"])), ("الربح", M.money(x["margin"])), ("هامش الربح", M.pct(x["margin_pct"])), ("متوسط الفاتورة", M.money(x["basket"]))]):
                    c_.markdown(ui.stat(lb, vv, pick), unsafe_allow_html=True)
                s = c_series(stamp(), dk, "M")
                l, r = st.columns(2)
                with l:
                    with st.container(key=f"card_dr1_{dim}"):
                        H(ui.sect(f"مبيعات {pick} شهريًا"))
                        fig = go.Figure(go.Scatter(x=[mlabel(f"{d:%Y-%m}") for d in s.index], y=s.net / 1e6, mode="lines", line=dict(color=MINT, width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(94,234,176,.10)"))
                        fig.update_yaxes(ticksuffix=" م")
                        chart(fig, 260, False)
                other = "category" if dim != "category" else "branch"
                o = c_dim(stamp(), dk, other).sort_values("net", ascending=False).head(8)
                with r:
                    with st.container(key=f"card_dr2_{dim}"):
                        H(ui.sect(f"{pick}: التوزيع حسب {'النوع' if other == 'category' else 'الفرع'}"))
                        H(ui.ranked([(q.label, q.net, M.money(q.net)) for q in o.itertuples()]))
    st.write("")
    H(ui.sect("ليه الأرقام اتغيرت؟", f"مقارنة {F['label']} بـ {F['ps']:%Y-%m-%d} ← {F['pe']:%Y-%m-%d}"))
    a, b = sel(F["pkey"]), sel(F["key"])
    if not len(a) or not len(b):
        st.info("مفيش بيانات كفاية في فترة المقارنة.")
        return
    br = M.bridge(a, b)
    word = "زادت" if br["delta"] >= 0 else "نزلت"
    big = max(("الكمية", br["volume"]), ("السعر والخصم", br["price"]), ("مزيج المنتجات", br["mix"]), key=lambda z: abs(z[1]))
    H(ui.note(f"المبيعات <b>{word} {abs(M.growth(br['net_b'], br['net_a']) or 0):.1f}%</b> ({M.money(br['net_a'])} ← {M.money(br['net_b'])}). أكبر عامل: <b>{big[0]}</b> ({M.money(big[1])}). "
              "«الكمية» = بعنا قطع أكتر أو أقل، «السعر والخصم» = اتغير سعر أو خصم نفس المنتج، «المزيج» = اتجهنا لمنتجات أغلى أو أرخص."))
    l, r = st.columns([1, 1])
    with l:
        with st.container(key="card_wf"):
            wf = go.Figure(go.Waterfall(x=["قبل", "الكمية", "السعر والخصم", "المزيج", "بعد"], measure=["absolute", "relative", "relative", "relative", "total"],
                                        y=[br["net_a"] / 1e6, br["volume"] / 1e6, br["price"] / 1e6, br["mix"] / 1e6, br["net_b"] / 1e6], connector=dict(line=dict(color="#2a3a35")),
                                        increasing=dict(marker=dict(color=MINT)), decreasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color="#51635c")), texttemplate="%{y:.1f}", textposition="outside"))
            wf.update_yaxes(ticksuffix=" م", range=[min(br["net_a"], br["net_b"]) / 1e6 * 0.7, max(br["net_a"], br["net_b"]) / 1e6 * 1.08])
            chart(wf, 330, False, "الأخضر زيادة والأحمر نقص (بالمليون جنيه). المحور مش بيبدأ من الصفر عشان الفروق تبان.")
    with r:
        with st.container(key="card_dv"):
            dn = st.radio("أكتر عناصر أثّرت حسب:", list(DIMS), horizontal=True, key="why_dim")
            d = M.drivers(a, b, D, DIMS[dn])
            tp = pd.concat([d.head(5), d.tail(5)]).drop_duplicates("key").sort_values("delta")
            H(ui.diverge([(q.label, q.delta, M.money(q.delta)) for q in tp.itertuples()]))
            H('<div class="cap">أخضر = زوّد المبيعات، أحمر = نقّصها مقارنة بفترة المقارنة.</div>')


# ---------------------------------------------------------------- 3) الفروع
def page_branches(D, F):
    df = sel(F["key"])
    if not len(df):
        st.warning("مفيش بيانات بالفلاتر دي.")
        return
    pnl = M.branch_pnl(D, df, F["start"], F["end"])
    prev = c_dim(stamp(), F["pkey"], "branch").set_index("key").net
    pnl["growth"] = (pnl.net / pnl.branch.map(prev) - 1) * 100
    fl = M.branch_flags(D)
    ok = pnl[pnl.cost_ok]
    mg, mp = ok.growth.median(), ok.profit_pct.median()

    def klass(r):
        hi_g, hi_p = (r.growth >= mg) or r.growth != r.growth, r.profit_pct >= mp
        return "نجوم" if hi_g and hi_p else "ربحه حلو ونموه ضعيف" if hi_p else "بينمو بس ربحه ضعيف" if hi_g else "محتاج تدخل"
    pnl["class"] = pnl.apply(lambda r: klass(r) if r.cost_ok else "تكلفة ناقصة", axis=1)
    groups = [("نجوم", "نمو وربحية فوق المتوسط", MINT), ("ربحه حلو ونموه ضعيف", "بيكسب كويس لكن مبيعاته مش بتكبر", BLUE), ("بينمو بس ربحه ضعيف", "مبيعات بتزيد لكن الربح قليل", AMBER), ("محتاج تدخل", "نمو وربحية تحت المتوسط", RED)]
    cols = st.columns(4)
    for col, (g, sub, c) in zip(cols, groups):
        x = pnl[pnl["class"] == g]
        col.markdown(ui.stat(g, f"{len(x)} فرع", sub, c), unsafe_allow_html=True)
    nbad = int((~pnl.cost_ok).sum())
    if nbad:
        H(ui.note(f"<b>تنبيه:</b> في الفترة المختارة ({F['label']}) {nbad} فرع مفيش ليه مرتبات مسجلة في الداتا، فصافي ربحه عن الفترة دي مش دقيق وبنستبعده من الترتيب والتقسيم (موجود في الجدول تحت بتقييم «تكلفة ناقصة»). لو غيّرت الفترة العدد ممكن يتغير.", "warn"))
    H('<div class="cap">التقسيم بالنسبة لوسيط الفروع: النمو مقارنة بفترة المقارنة، والربحية = صافي الربح بعد مرتبات الفرع ومصروفاته (من غير مصروفات المخازن والإدارة).</div>')
    st.write("")
    a, b = st.columns(2)
    with a:
        with st.container(key="card_bp1"):
            H(ui.sect("أعلى 8 فروع ربحًا", "صافي الربح بعد المصروفات"))
            H(ui.ranked([(r.label, r.profit, M.money(r.profit)) for r in ok.head(8).itertuples()]))
    with b:
        with st.container(key="card_bp2"):
            H(ui.sect("أضعف 8 فروع ربحًا", "الأقل ربحًا في الفترة"))
            H(ui.ranked([(r.label, max(r.profit, 0.0001), M.money(r.profit)) for r in ok.sort_values("profit").head(8).itertuples()], RED, False))
    bad = fl[fl.gap <= -15]
    if len(bad):
        st.write("")
        with st.container(key="card_bad"):
            H(ui.sect(f"{len(bad)} فرع نموه أقل من الشركة بكتير", f"آخر 13 أسبوع مقابل نفس الفترة السنة اللي فاتت. الشركة {'بتنمو' if bad.network_growth.iloc[0] >= 0 else 'بتنكمش'} {abs(bad.network_growth.iloc[0]):.0f}%"))
            H(ui.diverge([(r.label, r.gap, f"{r.gap:+.0f} نقطة") for r in bad.head(8).itertuples()]))
    st.write("")
    H(ui.sect("كل الفروع"))
    t = pnl.rename(columns={"label": "الفرع", "governorate": "المحافظة", "net": "المبيعات", "margin": "ربح البيع", "payroll": "المرتبات", "opex": "المصروفات", "profit": "صافي الربح", "profit_pct": "هامش صافي الربح %", "growth": "النمو %", "class": "التقييم"})
    table(t[["الفرع", "المحافظة", "المبيعات", "ربح البيع", "المرتبات", "المصروفات", "صافي الربح", "هامش صافي الربح %", "النمو %", "التقييم"]], money=("المبيعات", "ربح البيع", "المرتبات", "المصروفات", "صافي الربح"), pct=("هامش صافي الربح %", "النمو %"), height=380)
    st.download_button("تحميل جدول الفروع", t.to_csv(index=False).encode("utf-8-sig"), "branches.csv", "text/csv")
    pick = st.selectbox("تفاصيل فرع", ["—"] + pnl.label.tolist())
    if pick != "—":
        bid = pnl[pnl.label == pick].branch.iloc[0]
        dk = drill_key(F, "branch", bid)
        s = c_series(stamp(), dk, "M")
        l, r = st.columns(2)
        with l:
            with st.container(key="card_bd1"):
                H(ui.sect(f"مبيعات {pick} شهريًا"))
                fig = go.Figure(go.Scatter(x=[mlabel(f"{d:%Y-%m}") for d in s.index], y=s.net / 1e6, mode="lines", line=dict(color=MINT, width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(94,234,176,.10)"))
                fig.update_yaxes(ticksuffix=" م")
                chart(fig, 260, False)
        with r:
            with st.container(key="card_bd2"):
                H(ui.sect(f"مخزون {pick}", "كام نوع في كل حالة"))
                idf, _ = c_inv(stamp(), tuple(inv.DEFAULTS.items()))
                z = idf[idf.branch == bid].groupby("status").size().reindex(inv.STATUS_ORDER).dropna()
                H(ui.ranked([(k, v, f"{int(v)} نوع") for k, v in z.items()]))


# ---------------------------------------------------------------- 4) المخزون
def transfer_text(srcs, need) -> str:
    """القرار لكل حالة نقص: حوّل من مصدر موجود، ولو مفيش مصدر يكفي نطلب الباقي."""
    if not srcs:
        return f"اطلب توريد {need:.0f} قطعة (مفيش مصدر تحويل)"
    got = sum(x["units"] for x in srcs)
    t = f"حوّل {got:.0f} قطعة من {srcs[0]['from']}" + (f" و{len(srcs) - 1} مصدر تاني" if len(srcs) > 1 else "")
    return t + (f" واطلب الباقي {need - got:.0f}" if need - got > 1 else "")


def inv_rows(df, params, by="category"):
    rows = []
    for r in df.itertuples():
        act = (f"محتاج {r.need:.0f} قطعة (شوف التحويلات المقترحة)") if r.need > 0 else ("زيادة: وقف الشراء/حوّل" if r.status == "زيادة" else "راكد: مفيش مبيعات" if r.status == "راكد" else "تمام")
        rows.append(dict(name=r.category_ar, sub=r.branch_name, stock=r.stock, demand=r.weekly_demand, cover=r.cover_weeks, status=r.status, action=act))
    return rows


def page_inventory(D, F):
    with st.expander("إعدادات الحساب (اختياري)"):
        a, b, c, d = st.columns(4)
        lead = a.number_input("مدة توصيل البضاعة (أسابيع)", 0.5, 8.0, inv.DEFAULTS["lead"], 0.5)
        tgt = b.number_input("المخزون المثالي (أسابيع بيع)", 2.0, 20.0, inv.DEFAULTS["target"], 1.0)
        ov = c.number_input("زيادة لو أكتر من (أسابيع)", 12.0, 104.0, inv.DEFAULTS["over"], 2.0)
        sf = d.number_input("أمان (أسابيع)", 0.0, 4.0, inv.DEFAULTS["safety"], 0.5)
    p = {**inv.DEFAULTS, "lead": lead, "target": tgt, "over": ov, "safety": sf}
    full, tr = c_inv(stamp(), tuple(p.items()))
    df = full
    b_ids, cats = list(F["key"][2]), list(F["key"][3])
    if b_ids:
        df = df[df.branch.isin(b_ids)]
    if cats:
        df = df[df.category.isin(cats)]
    if not len(df):
        st.warning("مفيش بيانات مخزون للفلاتر دي.")
        return
    s = inv.summary(df)
    ls = p["lead"] + p["safety"]
    H(ui.note(f"<b>إزاي بنحسب:</b> «بيكفي كام أسبوع» = المخزون الحالي ÷ المبيعات المتوقعة في الأسبوع (آخر 13 أسبوع مضروبة في اتجاه الطلب المتوقع). "
              f"أقل من <b>{ls:.0f} أسابيع</b> = حرج (البضاعة بتاخد {p['lead']:.0f} أسبوع توصل)، وأكتر من <b>{p['over']:.0f} أسبوع</b> = زيادة. المنطقة الخضراء المتقطعة في الأشرطة هي المدى السليم."))
    cols = st.columns(4)
    cols[0].markdown(ui.stat("مبيعات ممكن تضيع (تقدير)", M.money(s["exposure"]), f"لو {s['critical'] + s['low']} حالة قرّبت تخلص البضاعة متحولتش في ميعادها", RED), unsafe_allow_html=True)
    cols[1].markdown(ui.stat("تكلفة المخزون الزيادة (تقدير)", M.money(s["tied"]), f"فلوس متجمدة في {s['over']} حالة مخزونها أكتر من الحد", BLUE), unsafe_allow_html=True)
    cols[2].markdown(ui.stat("بيكفي كام أسبوع (الفروع)", f"{s['cover']:.0f}", "إجمالي مخزون الفروع ÷ الطلب", MINT), unsafe_allow_html=True)
    cols[3].markdown(ui.stat("قيمة مخزون الفروع", M.money(s["value"]), f"{M.num(s['units'])} قطعة", None), unsafe_allow_html=True)
    st.write("")
    cp = inv.category_plan(D, df, p)
    src_of: dict = {}
    for r in tr.to_dict("records"):
        src_of.setdefault((r["to_id"], r["category"]), []).append(r)
    n_tr = tr.groupby("category").to_id.nunique().to_dict()
    with st.container(key="card_cat"):
        H(ui.sect("الوضع لكل نوع منتج", "مخزون الفروع مقابل بيعها، والقرار بيراعي مخزون المخازن كمان"))
        rows = [dict(name=r.category_ar, sub=f"المخازن: بتكفي {min(r.wh_cover, 999):.0f} أسبوع إضافي" if r.wh_cover == r.wh_cover else "", stock=r.branch_stock, demand=r.demand, cover=r.branch_cover, status=r.status,
                     action=r.action + (f" · وحوّل لـ {n_tr[r.Index]} فرع من المخازن" if n_tr.get(r.Index) else "")) for r in cp.set_index("category").itertuples()]
        H(ui.coverage_table(rows, ls, p["over"]))
        H(f'<div class="cap">مقياس الشريط من 0 لـ 52 أسبوع. «وقف الشراء» معناها إن مخزون الفروع + المخازن يكفي أكتر من {p["over"] * 2:.0f} أسبوع، فمفيش داعي نشتري شراء جديد. ده مش بيتعارض مع «حوّل لـ N فرع»: النقص في فروع معينة بيتغطى من مخزون موجود فعلًا في المخازن والفروع التانية، مش بشراء.</div>')
    st.write("")
    crit = df[df.status.isin(["حرج", "منخفض"])].sort_values(["exposure", "cover_weeks"], ascending=[False, True]).head(10)
    with st.container(key="card_crit"):
        H(ui.sect("الفروع اللي محتاجة بضاعة دلوقتي", f"أهم 10 حالات حسب حجم المبيعات المعرضة للضياع · من إجمالي {s['critical'] + s['low']}"))
        if len(crit):
            rows = [dict(name=f"{r.branch_name} · {r.category_ar}", sub=f"معرض للضياع {M.money(r.exposure)}", stock=r.stock, demand=r.weekly_demand, cover=r.cover_weeks, status=r.status,
                         action=transfer_text(src_of.get((r.branch, r.category)), r.need)) for r in crit.itertuples()]
            H(ui.coverage_table(rows, ls, p["over"], "الفرع · النوع"))
        else:
            H(ui.note("مفيش حالات نقص حاليًا بالفلاتر دي. 👌"))
    st.write("")
    tt = tr
    if b_ids:
        tt = tt[tt.to_id.isin(b_ids)]
    if cats:
        tt = tt[tt.category.isin(cats)]
    with st.container(key="card_tr"):
        H(ui.sect("التحويلات المقترحة", "الفرع الناقص بياخد من فرع زيادة في نفس المحافظة، وبعدين من مخزن المنطقة"))
        if len(tt):
            recs = tt.head(8).to_dict("records")
            H(ui.html_table(["النوع", "إلى فرع", "من", "الكمية", "الفرع بيكفي", "مبيعات محمية"],
                            [[esc(r["category_ar"]), esc(r["to"]), f'{esc(r["from_kind"])}: {esc(r["from"])}', f'{r["units"]:,.0f}', f'{r["cover_weeks"]:.1f} أسبوع' if r["cover_weeks"] < 99 else "-", M.money(r["exposure"])] for r in recs], numeric=(3,)))
            out = tt[["category_ar", "to", "from_kind", "from", "units", "status", "cover_weeks", "exposure"]].rename(columns={"category_ar": "النوع", "to": "إلى فرع", "from_kind": "نوع المصدر", "from": "من", "units": "الكمية", "status": "حالة الفرع", "cover_weeks": "بيكفي (أسبوع)", "exposure": "مبيعات معرضة (ج)"})
            with st.expander(f"كل التحويلات ({len(out)})"):
                table(out.round(1), money=("الكمية", "مبيعات معرضة (ج)"), height=320)
            st.download_button("تحميل التحويلات", out.to_csv(index=False).encode("utf-8-sig"), "transfers.csv", "text/csv")
        else:
            H(ui.note("مفيش تحويلات مطلوبة بالفلاتر دي."))
    st.write("")
    pick = st.selectbox("شوف مخزون فرع معين", ["—"] + sorted(df.branch_name.unique()))
    if pick != "—":
        x = df[df.branch_name == pick].sort_values("cover_weeks")
        with st.container(key="card_bsel"):
            H(ui.sect(f"مخزون {pick}", "كل الأنواع مرتبة من الأقرب للنفاد"))
            H(ui.coverage_table(inv_rows(x, p), ls, p["over"]))
    warn = [h for h in D.health if "المخزون" in h["check"] or "مخزون" in h["check"]]
    if warn:
        H(ui.note("<b>ملحوظة عن البيانات:</b> " + esc(" · ".join(h["detail"] for h in warn[:2])), "warn"))


# ---------------------------------------------------------------- 5) التوقعات
LEVELS = {"كل الشركة": "total", "نوع منتج": "category", "فرع": "branch", "منتج": "product"}


def fc_names(D, level):
    if level == "category":
        return {k: D.cat_ar.get(k, k) for k in D.fq[D.fq.level == "category"].series}
    if level == "branch":
        return {k: D.branch_name.get(k, k) for k in D.fq[D.fq.level == "branch"].series}
    if level == "product":
        return {k: D.prod_name.get(k, k) for k in D.fq[D.fq.level == "product"].series}
    return {"__total__": "كل الشركة"}


def price_factor(months, step: float, last_actual: pd.Period) -> list[float]:
    """زيادة الأسعار المفترضة بتتطبق من أول يناير بعد آخر شهر فعلي."""
    jan = pd.Period(f"{last_actual.year + 1}-01", "M")
    return [1 + step / 100 if pd.Period(m, "M") >= jan else 1.0 for m in months]


def page_forecast(D, F):
    if not len(D.fq):
        st.warning("نماذج التوقع لسه متجهزتش. روح «صحة البيانات» وادوس «إعادة بناء».")
        return
    c1, c2, c3 = st.columns([1, 1.4, 1.2])
    lv_name = c1.radio("التوقع على مستوى", list(LEVELS), horizontal=True, key="fc_lv")
    level = LEVELS[lv_name]
    names = fc_names(D, level)
    pick = c2.selectbox("العنصر", list(names.values()), key=f"fc_pick_{level}") if level != "total" else "كل الشركة"
    series = next((k for k, v in names.items() if v == pick), "__total__")
    metric = c3.radio("بالـ", ["المبيعات (ج)", "القطع"], horizontal=True, key="fc_metric")
    target = "net" if metric.startswith("المبيعات") else "qty"
    step = 0.0
    if target == "net":
        step = st.slider("افتراض زيادة أسعار يناير القادم (%)", 0, 40, 0, help="أسعار الشركة بتزيد مرة في السنة في يناير (آخر زيادتين +38% و+23%). الصفر معناه إن الأسعار ثابتة.", key="fc_step")
    f = D.forecast[(D.forecast.level == level) & (D.forecast.series == series) & (D.forecast.target == "qty")].sort_values("h")
    q = D.fq[(D.fq.level == level) & (D.fq.series == series) & (D.fq.target == "qty")]
    if not len(f) or not len(q):
        st.warning("مفيش توقع موثوق للعنصر ده (التاريخ قصير أو المبيعات صغيرة).")
        return
    q = q.iloc[0]
    hist = models.monthly(D.sales, "net" if target == "net" else "qty", by=None if level == "total" else level)
    hist = hist if level == "total" else hist[series]
    qhist = models.monthly(D.sales, "qty", by=None if level == "total" else level)
    qhist = qhist if level == "total" else qhist[series]
    last_p = hist.index[-1]
    mult = price_factor(f.month, step, last_p) if target == "net" else [1.0] * len(f)
    unit = q.price if target == "net" else 1.0
    f = f.assign(forecast=f.forecast.values * unit * np.array(mult), lo=f.lo.values * unit * np.array(mult), hi=f.hi.values * unit * np.array(mult))
    fmtv = M.money if target == "net" else (lambda v: f"{v:,.0f} قطعة")
    ly = qhist.reindex(pd.PeriodIndex(f.month.head(3), freq="M") - 12)
    qf = D.forecast[(D.forecast.level == level) & (D.forecast.series == series) & (D.forecast.target == "qty")].sort_values("h").head(3)
    chg = (qf.forecast.sum() / ly.sum() - 1) * 100 if ly.notna().all() and ly.sum() else None
    n0 = f.iloc[0]
    rel_color = {"عالية": MINT, "متوسطة": AMBER, "ضعيفة": RED}[q.reliability]
    st.write("")
    ui_cols = st.columns(4)
    ui_cols[0].markdown(ui.stat(f"المتوقع في {mlabel(n0.month, False)}", fmtv(n0.forecast), f"المدى المحتمل: من {fmtv(n0.lo)} إلى {fmtv(n0.hi)}", MINT), unsafe_allow_html=True)
    ui_cols[1].markdown(ui.stat("القطع في أول 3 شهور مقابل نفس الشهور السنة اللي فاتت", f"{chg:+.0f}%" if chg is not None else "-", "نمو متوقع في الكمية" if (chg or 0) >= 0 else "نقص متوقع في الكمية", MINT if (chg or 0) >= 0 else RED), unsafe_allow_html=True)
    ui_cols[2].markdown(ui.stat("متوسط الغلط في الاختبار", f"{q.wape_test:.1f}%", f"بنغلط في المتوسط {q.wape_test:.0f}% من الفعلي (الطريقة البسيطة {q.baseline_test:.0f}%)", AMBER), unsafe_allow_html=True)
    ui_cols[3].markdown(ui.stat("موثوقية التوقع", q.reliability, (q.note or f"المدى المحتمل وقع جواه الفعلي في {q.coverage_test:.0f}% من شهور الاختبار"), rel_color), unsafe_allow_html=True)
    if q.reliability == "ضعيفة":
        H(ui.note(f"<b>تنبيه:</b> توقع {esc(pick)} ضعيف الموثوقية ({esc(q.note) or 'خطأ كبير'}). استخدمه كاتجاه عام، متبنيش عليه قرار شراء أو مخزون دقيق.", "bad"))
    if target == "net":
        base_txt = "الإيراد = الكمية المتوقعة × سعر القطعة الحالي، وبيفترض " + ("ثبات الأسعار." if step == 0 else f"زيادة أسعار {step}% من يناير القادم.") + " الغلط المعروض محسوب على الكمية (القطع)."
        H(ui.note(esc(base_txt), "warn" if step == 0 else ""))
    st.write("")
    with st.container(key="card_fcc"):
        H(ui.sect(f"{'المبيعات' if target == 'net' else 'القطع'} - {pick}", "الفعلي والمتوقع · شهريًا · المنطقة المظللة = المدى المحتمل (80%)"))
        xs = [mlabel(str(p)) for p in hist.index[-24:]]
        sc = 1e6 if target == "net" else 1
        last = hist.values[-1] / sc
        fx = [xs[-1]] + [mlabel(m) for m in f.month]
        fig = go.Figure()
        fig.add_scatter(x=xs, y=hist.values[-24:] / sc, name="فعلي", mode="lines", line=dict(color=MINT, width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(94,234,176,.10)")
        fig.add_scatter(x=fx, y=[last] + list(f.hi / sc), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fx, y=[last] + list(f.lo / sc), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(246,185,77,.15)", name="المدى المحتمل (80%)", hoverinfo="skip")
        fig.add_scatter(x=fx, y=[last] + list(f.forecast / sc), name="توقع", mode="lines+markers", line=dict(color=AMBER, width=2.6, dash="dash"))
        if level == "total" and target == "net":
            tr = D.forecast[(D.forecast.level == "total") & (D.forecast.target == "net_trend")].sort_values("h")
            if len(tr):
                fig.add_scatter(x=fx, y=[last] + list(tr.forecast / sc), name="لو نمط الأسعار السابق اتكرر", mode="lines", line=dict(color="#7aa7ff", width=1.8, dash="dot"))
        fig.update_yaxes(ticksuffix=" م" if target == "net" else "")
        thin_axis(fig, xs + [mlabel(m) for m in f.month])
        chart(fig, 340, True)
        H('<div class="cap"><b>إزاي تقرأ الرسمة:</b> الأخضر اللي حصل فعلًا، والبرتقالي المتقطع المتوقع. <b>متوسط الغلط</b> بيقول التوقع بيبعد عن الفعلي بكام في المتوسط، أما <b>المدى المحتمل</b> فهو نطاق بنتوقع إن الرقم الفعلي يقع جواه في 80% من الحالات، وهما حاجتين مختلفتين.</div>')
        if f.iloc[0].month == f"{D.last:%Y-%m}" and D.last.day < D.last.days_in_month:
            nc = models.mtd_nowcast(D, series, level, "net" if target == "net" else "qty")
            if nc and nc["projected"]:
                H(ui.note(f"<b>الشهر الحالي لسه ماخلصش:</b> اتباع منه {fmtv(nc['so_far'])} حتى يوم {nc['day']} من {nc['days']}. في الشهور اللي فاتت كان بيتباع في الفترة دي حوالي {nc['typical_share'] * 100:.0f}% من الشهر، فالتقدير بنمط الشهور السابقة حوالي <b>{fmtv(nc['projected'])}</b> للشهر كله (والتوقع بالنموذج {fmtv(n0.forecast)})."))
    st.write("")
    t = f.rename(columns={"month": "الشهر", "lo": "أقل تقدير", "forecast": "المتوقع", "hi": "أعلى تقدير"})
    with st.expander("جدول الشهور المتوقعة"):
        table(t[["الشهر", "أقل تقدير", "المتوقع", "أعلى تقدير"]], money=("أقل تقدير", "المتوقع", "أعلى تقدير"))
    a, b2 = st.columns(2)
    with a:
        with st.container(key="card_fcg"):
            H(ui.sect("أنواع هيزيد أو يقل عليها الطلب", "القطع المتوقعة في أول 3 شهور مقابل نفس الشهور السنة اللي فاتت"))
            rows = []
            qm = models.monthly(D.sales, "qty", by="category")
            for c in qm.columns:
                fc = D.forecast[(D.forecast.level == "category") & (D.forecast.series == c) & (D.forecast.target == "qty")].sort_values("h").head(3)
                if len(fc) == 3:
                    old = qm[c].reindex(pd.PeriodIndex(fc.month, freq="M") - 12)
                    if old.notna().all() and old.sum():
                        rows.append((D.cat_ar.get(c, c), (fc.forecast.sum() / old.sum() - 1) * 100))
            rows.sort(key=lambda z: z[1])
            H(ui.diverge([(n, v, f"{v:+.0f}%") for n, v in rows]))
    with b2:
        with st.container(key="card_fcm"):
            H(ui.sect("موثوقية التوقع حسب المستوى", "متوسط الغلط في الاختبار وعدد العناصر الموثوقة"))
            for lv, nm in (("total", "كل الشركة"), ("category", "الأنواع"), ("branch", "الفروع"), ("product", "المنتجات")):
                x = D.fq[(D.fq.level == lv) & (D.fq.target == "qty")]
                if len(x):
                    good = int((x.reliability != "ضعيفة").sum())
                    H(ui.stat(f"{nm} ({len(x)})", f"{x.wape_test.median():.1f}%", f"وسيط متوسط الغلط · {good} من {len(x)} موثوق (عالي أو متوسط)", MINT if x.wape_test.median() <= 8 else AMBER))
    with st.expander("إزاي اتختار النموذج وقيست الدقة؟"):
        H(ui.note("<b>الطريقة:</b> كل نموذج اتجرّب على 12 نقطة بداية قديمة (بنقص الداتا لحد نقطة، نتوقع 6 شهور، ونقارن بالفعلي من غير ما النموذج يشوف أي حاجة بعدها). الاختيار بيتم على أقدم 6 نقاط، والدقة المعروضة على أحدث 6 نقاط (شهور محجوزة للاختبار). "
                  "للشركة والأنواع بنجرّب: نفس الشهر السنة اللي فاتت، × معدل النمو، ETS، Theta، SARIMA، ومتوسطهم. للفروع والمنتجات (بيانات أصغر وأكتر تقلب): نموذج مستقل، أو نصيب العنصر من توقع الأعلى، أو دمجهم، والأحسن بيتختار للمستوى كله. "
                  "اتجرّبت كمان مواسم رمضان والعيد كعوامل إضافية ومحسّنتش النتيجة فمش مستخدمة."))
        bt = D.backtest[(D.backtest.level == level) & (D.backtest.series == series)] if level in ("total", "category") else pd.DataFrame()
        if len(bt):
            rows = []
            for mname, d in bt.groupby("model"):
                v, tt = d[(d.origin >= models.SPLIT) & (d.h <= 3)], d[(d.origin < models.SPLIT) & (d.h <= 3)]
                rows.append({"الطريقة": models.MODEL_AR.get(mname, mname), "غلط الاختيار %": models._wape(v), "غلط الاختبار %": models._wape(tt), "الانحياز %": models._bias(tt)})
            table(pd.DataFrame(rows).sort_values("غلط الاختبار %").round(1), pct=("غلط الاختيار %", "غلط الاختبار %", "الانحياز %"))
            st.caption(f"الطريقة المختارة: {models.MODEL_AR.get(q.model, q.model)}. الانحياز الموجب = التوقع بيطلع أعلى من الفعلي.")
        else:
            st.caption(f"الطريقة المختارة للمستوى: {models.MODEL_AR.get(q.model, q.model)}.")


# ---------------------------------------------------------------- 6) العملاء
def page_customers(D, F):
    c = D.churn
    df = sel(F["key"])
    if not len(c) or not len(df):
        st.warning("مفيش بيانات عملاء كفاية.")
        return
    unk = df[df.customer.isna()].net.sum() / df.net.sum() * 100
    top10 = c.spend12m.sort_values(ascending=False).head(int(len(c) * 0.1)).sum() / c.spend12m.sum() * 100
    cols = st.columns(4)
    cols[0].markdown(ui.stat("عملاء مسجلين اشتروا", M.num(df.customer.nunique()), "في الفترة المختارة", MINT), unsafe_allow_html=True)
    cols[1].markdown(ui.stat("مبيعات من غير حساب عميل", f"{unk:.0f}%", "مش بنقدر نتواصل معاهم", AMBER), unsafe_allow_html=True)
    cols[2].markdown(ui.stat("أفضل 10% عملاء بيجيبوا", f"{top10:.0f}%", "من إنفاق آخر 12 شهر", None), unsafe_allow_html=True)
    cols[3].markdown(ui.stat("اشتروا أكتر من مرة", f"{(c.freq >= 2).mean() * 100:.0f}%", "من العملاء المسجلين", None), unsafe_allow_html=True)
    st.write("")
    seg = c.groupby("segment").agg(n=("customer_id", "size"), spend=("spend12m", "sum"), freq=("freq", "mean")).reset_index().sort_values("spend", ascending=False)
    l, r = st.columns(2)
    with l:
        with st.container(key="card_seg"):
            H(ui.sect("إنفاق كل شريحة عملاء", "آخر 12 شهر · الشريحة حسب آخر شراء وعدد المرات"))
            H(ui.ranked([(q.segment, q.spend, f"{M.money(q.spend)} · {q.n:,} عميل") for q in seg.itertuples()]))
            H('<div class="cap">«أبطال» بيشتروا كتير ومؤخرًا، «كانوا مهمين وبعدوا» كانوا بيشتروا كتير لكن بقالهم فترة.</div>')
    ab = c.groupby("absence").size().reindex(["نشيط", "بعيد 3-6 شهور", "بعيد 6-12 شهر", "بعيد أكتر من سنة"])
    with r:
        with st.container(key="card_abs"):
            H(ui.sect("العملاء حسب مدة الغياب", "من آخر مرة اشتروا فيها"))
            H(ui.ranked([(k, v, f"{v:,} عميل") for k, v in ab.items()], BLUE, False))
    cm = D.model_metrics.get("churn", {})
    if cm:
        if cm.get("predictive"):
            H(ui.note(f"نموذج توقع رجوع العميل خلال {cm['window_days']} يوم دقته (AUC) <b>{cm['auc']}</b> مقابل {cm['auc_baseline']} للقاعدة البسيطة."))
        else:
            H(ui.note(f"<b>مهم:</b> جرّبنا نموذج يتوقع مين هيرجع يشتري خلال {cm['window_days']} يوم، لكن دقته ({cm['auc']}) زي التخمين (0.5). نسبة الرجوع حوالي <b>{cm['base_rate'] * 100:.0f}%</b> تقريبًا لكل العملاء سواء لسه مشتريين أو غايبين من سنة. "
                      "يعني الغياب لوحده مش مؤشر ضياع. جرّب أي حملة استرجاع على مجموعة تجريبية الأول وقارنها بمجموعة من غير حملة.", "warn"))
    cu = c.merge(D.customers[["customer_id", "customer_name_ar", "governorate_ar", "age_band"]], on="customer_id", how="left")
    l, r = st.columns(2)
    with l:
        with st.container(key="card_age"):
            H(ui.sect("الإنفاق حسب السن", "آخر 12 شهر"))
            ag = cu.groupby("age_band").spend12m.sum().sort_index()
            H(ui.ranked([(k, v, M.money(v)) for k, v in ag.items()], MINT, False))
    with r:
        with st.container(key="card_gov"):
            H(ui.sect("أعلى 7 محافظات في إنفاق العملاء", "آخر 12 شهر"))
            gv = cu.groupby("governorate_ar").spend12m.sum().sort_values(ascending=False).head(7)
            H(ui.ranked([(k, v, M.money(v)) for k, v in gv.items()]))
    st.write("")
    H(ui.sect("أفضل العملاء", "أعلى 200 عميل إنفاقًا"))
    top = cu.sort_values("spend12m", ascending=False).head(200)
    t = top[["customer_name_ar", "governorate_ar", "segment", "freq", "spend12m", "last_purchase"]].rename(columns={"customer_name_ar": "العميل", "governorate_ar": "المحافظة", "segment": "الشريحة", "freq": "عدد الطلبات", "spend12m": "إنفاق آخر 12 شهر", "last_purchase": "آخر شراء"})
    table(t.assign(**{"آخر شراء": t["آخر شراء"].dt.strftime("%Y-%m-%d")}), money=("إنفاق آخر 12 شهر", "عدد الطلبات"), height=320)
    st.download_button("تحميل أفضل 200 عميل", t.to_csv(index=False).encode("utf-8-sig"), "top_customers.csv", "text/csv")


# ---------------------------------------------------------------- 7) التوصيات
def page_recs(D, F):
    r = c_insights(stamp())
    allp, alla = list(insights.PRIORITY), sorted(r.area.unique())
    pri = st.pills("الأولوية", allp, selection_mode="multi", default=allp, key="rec_pri") or allp
    area = st.pills("المجال", alla, selection_mode="multi", default=alla, key="rec_area") or alla
    r = r[r.priority.isin(pri) & r.area.isin(area)]
    items = list(r.itertuples())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (j, x) in zip(cols, [(i, items[i])] + ([(i + 1, items[i + 1])] if i + 1 < len(items) else [])):
            with col:
                with st.container(key=f"card_r{j}"):
                    big, unit = split_big(x.big)
                    H(ui.problem_card(j + 1, x.priority, x.title, big, unit, x.label, x.items[:4], x.evidence, x.ring, x.kind))
                    H(ui.note(f"<b>الإجراء المقترح:</b> {esc(x.action)}" + (f"<br><small>الأثر المقدر: {M.money(x.impact)} ({esc(x.impact_note)})</small>" if x.impact == x.impact else "")))
                    link_button("افتح التفاصيل ›", x.page, f"r{j}")
    st.download_button("تحميل التوصيات", r.drop(columns=["items"]).to_csv(index=False).encode("utf-8-sig"), "recommendations.csv", "text/csv")


# ---------------------------------------------------------------- 8) المساعد
SUGGEST = ["إيه أعلى 5 فروع مبيعًا الشهر اللي فات؟", "ليه المبيعات اتغيرت الشهر ده؟", "قارن بين مبيعات القاهرة والإسكندرية", "توقع مبيعات الشهر الجاي",
           "إيه اللي هيخلص من المخزون؟", "عايز تحويلات مقترحة", "ليه الربح اتغير السنة دي؟", "إيه توصياتك؟"]


@st.cache_resource(show_spinner=False)
def llm_client():
    """عميل Groq واحد مشترك (عشان حالة الانتظار عند حد الاستخدام تفضل ثابتة)، وإعداداته من .env بس."""
    from src.llm import GroqClient
    return GroqClient()


def get_assistant(D, F):
    ctx = Ctx(F["start"], F["end"])
    current_stamp = stamp()
    previous = st.session_state.get("asst")
    same_data = st.session_state.get("_asst_stamp") == current_stamp
    # A Streamlit session can outlive the imported Assistant class after edits.
    if not same_data or type(previous) is not Assistant:
        replacement = Assistant(D, ctx, llm_client())
        if same_data and previous is not None:
            from copy import deepcopy
            replacement.state = deepcopy(getattr(previous, "state", {}))
            replacement.turns = deepcopy(getattr(previous, "turns", []))
            replacement.last_answer = deepcopy(getattr(previous, "last_answer", None))
            if replacement.last_answer is None:
                # Migrate existing same-dataset sessions after this capability is
                # deployed; never search past the most recent assistant reply.
                latest = next((m for m in reversed(st.session_state.get("chat", []))
                               if m.get("role") == "assistant"), None)
                if latest is not None:
                    replacement.last_answer = deepcopy(latest.get("a"))
        st.session_state.asst = replacement
        st.session_state._asst_stamp = current_stamp
    st.session_state.asst.ctx = ctx
    return st.session_state.asst


def page_assistant(D, F):
    asst = get_assistant(D, F)
    s = asst.status()
    if not s["configured"]:
        eng = '<span class="chip">المحرك المحلي</span>'
    elif s["available"]:
        eng = '<span class="chip ai">Groq · مهيّأ</span>'
    else:
        eng = f'<span class="chip warn">المحرك المحلي · انتظار Groq ({s["cooldown"]} ثانية)</span>'
    H(f'<div class="chip-row">{eng}</div>')
    H(f'<div class="cap">الفترة الافتراضية: <b>{F["label"]}</b> (لو سألت عن فترة تانية زي «الشهر اللي فات» بيستخدمها). والإجابة على مستوى الشركة إلا لو ذكرت فرع أو نوع. بيفتكر سياق المحادثة: «طب والربح؟» و«وفي القاهرة؟» و«ليه؟» بيكملوا على نفس الموضوع. تقدر تسأل سؤال من أكتر من جزء.</div>')
    st.session_state.setdefault("chat", [])
    clicked = None
    with st.expander("أسئلة مقترحة", expanded=not st.session_state.chat):
        with st.container(key="chat_suggestions"):
            for start in range(0, len(SUGGEST), 4):
                cols = st.columns(4)
                for i, q in enumerate(SUGGEST[start:start + 4], start):
                    if cols[i - start].button(q, key=f"sg{i}", width="stretch"):
                        clicked = q
    for message_index, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🤖"):
            H(f'<span class="who-{"user" if m["role"] == "user" else "bot"}"></span>')
            if m["role"] == "assistant" and m.get("a") is not None:
                render_answer(m["a"], chart_key=f"chat_{message_index}_part_root")
            else:
                H(f'<div style="line-height:1.8">{esc(m["text"])}</div>')
    q = st.chat_input("اكتب سؤالك هنا...") or clicked or st.session_state.pop("pending_q", None)
    if q:
        st.session_state.chat.append({"role": "user", "text": q})
        with st.spinner("بحلل البيانات..."):
            try:
                a = asst.ask(q)
                st.session_state.chat.append({"role": "assistant", "a": a, "text": a.text})
            except Exception:
                import logging
                logging.getLogger("cottonil.llm").exception("assistant failure")
                st.session_state.chat.append({"role": "assistant", "text": "حصلت مشكلة وأنا بحلل السؤال. جرّب تصيغه بشكل تاني أو اسأل عن جزء واحد."})
        st.rerun()
    if st.session_state.chat and st.button("مسح المحادثة"):
        st.session_state.chat = []
        asst.state, asst.turns = {}, []
        asst.last_answer = None
        st.rerun()


# ---------------------------------------------------------------- 9) صحة البيانات
def page_health(D, F):
    ok = sum(h["status"] == "ok" for h in D.health)
    cols = st.columns(4)
    cols[0].markdown(ui.stat("آخر يوم في البيانات", f"{D.last:%Y-%m-%d}", f"من {D.first:%Y-%m-%d}", MINT), unsafe_allow_html=True)
    cols[1].markdown(ui.stat("سطور المبيعات", M.num(D.meta["rows"]), "سطر فاتورة", None), unsafe_allow_html=True)
    cols[2].markdown(ui.stat("فحوصات سليمة", f"{ok} من {len(D.health)}", "فحص جودة", MINT if ok == len(D.health) else AMBER), unsafe_allow_html=True)
    cols[3].markdown(ui.stat("آخر تحديث للبيانات", D.meta["built_at"], "", None), unsafe_allow_html=True)
    st.write("")
    H(ui.sect("فحوصات جودة البيانات", "الملاحظات مش معناها إن الأرقام غلط، لكن معناها نكون حذرين في الجزء ده"))
    stc = {"ok": ("سليم", MINT), "warn": ("ملاحظة", AMBER), "error": ("مشكلة", RED)}
    for i in range(0, len(D.health), 2):
        cols = st.columns(2)
        for j, (col, h) in enumerate(zip(cols, D.health[i:i + 2])):
            lab, c = stc[h["status"]]
            with col:
                with st.container(key=f"card_hc{i + j}"):
                    H(f'<div>{ui.tag(lab, c)} <b>{esc(h["check"])}</b></div><div class="cap">{esc(h["detail"]) or "-"}</div>')
    st.write("")
    H(ui.sect("بيانات مش بنعتمد عليها في أي قرار", "حسب توضيح الجهة اللي وفّرت الداتا"))
    a, b = st.columns(2)
    with a:
        with st.container(key="card_cw1"):
            H("<b>كميات المشتريات</b> مش مرتبطة بشكل موثوق بالطلب الفعلي، فمش بنستخدمها في حساب المخزون ولا في توصيات الشراء. بنعتمد على المبيعات والمخزون المسجل والتوقعات.")
    with b:
        with st.container(key="card_cw2"):
            H("<b>الطلبات المنسوبة للحملات الإعلانية</b> مش طلبات بيع مؤكدة، فمش بنحسب منها عائد تسويقي ولا بنبني عليها توصيات.")
    st.write("")
    H(ui.sect("جودة التوقعات والنماذج"))
    q = D.fq[D.fq.target == "qty"]
    if len(q):
        med = q.groupby("level").wape_test.median()
        H(ui.note("<b>التوقعات:</b> متوسط الغلط على شهور محجوزة للاختبار (وسيط): " + "، ".join(f"{nm} <b>{med[lv]:.1f}%</b>" for lv, nm in (("total", "الشركة"), ("category", "الأنواع"), ("branch", "الفروع"), ("product", "المنتجات")) if lv in med)
                  + ". كل ما المستوى أصغر (فرع أو منتج) الغلط أكبر، وده طبيعي لأن الأرقام الصغيرة بتتقلب."))
    cm = D.model_metrics.get("churn", {})
    if cm:
        H(ui.note(f"<b>نموذج العملاء:</b> AUC = {cm['auc']} (التخمين 0.5). " + ("مفيد." if cm.get("predictive") else "مش بيتنبأ أحسن من التخمين فمش بنستخدمه في القرارات."), "" if cm.get("predictive") else "warn"))
    st.write("")
    H(ui.sect("تحديث البيانات"))
    if bootstrap.settings(st.secrets).get("BI_DATA_URL"):
        st.caption("نسخة نشر محفوظة: تحديث البيانات والتوقعات بيتم بنشر حزمة جديدة، بدون إعادة تدريب على السيرفر.")
        return
    reason = data.needs_rebuild()
    if reason:
        H(ui.note(esc(reason), "warn"))
    st.caption("ضيف ملفات الـ CSV الجديدة (أو zip) جوه فولدر data/raw أو ارفعها هنا، وبعدين دوس «إعادة بناء» (حوالي 4 دقايق لأن التوقعات بتتجرّب على مئات السلاسل). الملفات الأساسية: sales_orders و sales_order_items و branches و products_catalog.")
    up = st.file_uploader("ملفات البيانات", type=["csv", "zip"], accept_multiple_files=True)
    a, b = st.columns(2)
    if a.button("احفظ الملفات المرفوعة", disabled=not up):
        st.success(f"اتحفظ {data.save_upload(up)} ملف. دوس «إعادة بناء» دلوقتي.")
    if b.button("إعادة بناء البيانات والنماذج", type="primary"):
        with st.status("جاري التحديث...", expanded=True) as s:
            try:
                data.build(log=st.write)
                s.update(label="تم التحديث", state="complete")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                s.update(label="فشل التحديث", state="error")
                st.error(str(e))


# ---------------------------------------------------------------- الحالة الفاضية والتشغيل
def empty_state():
    H(ui.css(0))
    H('<div class="brand"><div class="logo">C</div><div><b>COTTONIL</b><small>مركز قيادة الأعمال</small></div></div>')
    raw = data.find_raw()
    H('<div class="ptitle">لسه مفيش بيانات</div><div class="psub">مش هنعرض أي أرقام لحد ما تضيف ملفات الداتا الأصلية، عشان منعرضش حاجة وهمية.</div>')
    H(ui.note("<b>الخطوات:</b><br>1) حط ملفات الـ CSV (أو فولدر COTTONIL_Data كامل) جوه فولدر <code>data/raw</code>، أو ارفعها من تحت.<br>2) دوس «ابني البيانات» (حوالي نص دقيقة)، وبعدها هتفتح لوحة القيادة.", "warn"))
    up = st.file_uploader("ارفع ملفات الـ CSV أو ملف zip", type=["csv", "zip"], accept_multiple_files=True)
    if up and st.button("احفظ الملفات"):
        st.success(f"اتحفظ {data.save_upload(up)} ملف")
        st.rerun()
    if raw is not None:
        H(ui.note(f"لقينا ملفات بيانات في: <code>{esc(raw)}</code>"))
        if st.button("ابني البيانات", type="primary"):
            with st.status("جاري البناء...", expanded=True):
                try:
                    data.build(log=st.write)
                except Exception as e:
                    st.error(str(e))
                    return
            st.rerun()


def main():
    try:
        with st.spinner("بجهّز نسخة البيانات المحفوظة..."):
            data.PROC = bootstrap.ensure(data.PROC, bootstrap.settings(st.secrets))
    except bootstrap.BundleError as exc:
        st.error("تعذر تجهيز بيانات النشر. راجع إعدادات مصدر البيانات في Secrets.")
        st.code(str(exc))
        return
    if not (data.PROC / "sales.parquet").exists():
        empty_state()
        return
    D = DD()
    rec = c_insights(stamp())
    st.session_state.setdefault("nav", PAGES[0])
    sidebar(D, int(rec.priority.isin(["حرجة", "مهمة"]).sum()))
    page = st.session_state["nav"]
    left, right = st.columns([1.15, 1])
    F = None
    with right:
        F = controls(D)
    with left:
        if page == "نظرة عامة":
            probs = int(rec.priority.isin(["حرجة", "مهمة"]).sum())
            greet = "صباح الخير" if pd.Timestamp.now().hour < 12 else "مساء الخير"
            hl = f"لقينا <em>{probs}</em> مشاكل — ابدأ بالأولى." if probs else "كل حاجة ماشية كويس."
            H(f'<div class="hello">{greet} {esc(USER)} 👋</div><div class="headline">{hl}</div>')
        else:
            title(page, PAGE_SUB[page](F))
    if page == "نظرة عامة":
        overview_body(D, F, rec)
    else:
        {"المبيعات": page_sales, "الفروع": page_branches, "المخزون": page_inventory, "التوقعات": page_forecast, "العملاء": page_customers,
         "التوصيات": page_recs, "المساعد الذكي": page_assistant, "صحة البيانات": page_health}[page](D, F)


main()
