"""حسابات الأرقام الأساسية: المؤشرات، التجميع، أسباب التغيير، وربحية الفروع."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import apply_filters


# ---------------------------------------------------------------- تنسيق الأرقام
def money(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f} مليار ج"
    if a >= 1e6:
        return f"{x / 1e6:,.1f} مليون ج"
    if a >= 1e4:
        return f"{x / 1e3:,.0f} ألف ج"
    return f"{x:,.0f} ج"


def num(x: float) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.{d}f}%"


def growth(a: float, b: float) -> float | None:
    """نسبة التغير من b إلى a بالمئة."""
    return None if not b else (a / b - 1) * 100


def prev_range(start, end, mode: str):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if mode == "yoy":
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    n = (end - start).days + 1
    return start - pd.Timedelta(days=n), start - pd.Timedelta(days=1)


# ---------------------------------------------------------------- المؤشرات
def kpis(df: pd.DataFrame) -> dict:
    net, margin, gross, disc = df.net.sum(), df.margin.sum(), df.gross.sum(), df.disc.sum()
    orders, qty = df.order.nunique(), int(df.qty.sum())
    return {"net": net, "margin": margin, "margin_pct": margin / net * 100 if net else 0.0, "qty": qty, "orders": orders,
            "basket": net / orders if orders else 0.0, "disc_pct": disc / gross * 100 if gross else 0.0,
            "customers": df.customer.nunique(), "price": net / qty if qty else 0.0, "gross": gross, "disc": disc}


def return_stats(D, start, end, branches=None, categories=None, channels=None) -> dict:
    r = D.returns
    if not len(r):
        return {"refund": 0.0, "qty_ret": 0}
    m = (r.date >= pd.Timestamp(start)) & (r.date <= pd.Timestamp(end))
    for col, vals in (("branch", branches), ("category", categories), ("channel", channels)):
        if vals:
            m &= r[col].isin(vals)
    x = r[m]
    return {"refund": float(x.refund.sum()), "qty_ret": int(x.qty_ret.sum())}


# ---------------------------------------------------------------- التجميع
def add_dim(df: pd.DataFrame, D, dim: str) -> pd.Series:
    if dim == "governorate":
        return df.branch.astype(str).map(D.branch_gov)
    if dim == "month":
        return df.date.dt.to_period("M").astype(str)
    if dim == "week":
        return df.date.dt.to_period("W-SAT").dt.start_time.dt.strftime("%Y-%m-%d")
    if dim == "day":
        return df.date.dt.strftime("%Y-%m-%d")
    if dim == "year":
        return df.date.dt.year.astype(str)
    return df[dim].astype(str)


def label_of(D, dim: str, key: str) -> str:
    if dim == "branch":
        return D.branch_name.get(key, key)
    if dim == "category":
        return D.cat_ar.get(key, key)
    if dim == "product":
        return D.prod_name.get(key, key)
    if dim == "channel":
        from .data import CH_AR
        return CH_AR.get(key, key)
    return key


def by_dim(df: pd.DataFrame, D, dim: str) -> pd.DataFrame:
    key = add_dim(df, D, dim)
    g = df.assign(_k=key.values).groupby("_k")
    out = g.agg(net=("net", "sum"), margin=("margin", "sum"), qty=("qty", "sum"), gross=("gross", "sum"),
                disc=("disc", "sum"), orders=("order", "nunique"))
    out["margin_pct"] = np.where(out.net > 0, out.margin / out.net * 100, 0.0)
    out["disc_pct"] = np.where(out.gross > 0, out.disc / out.gross * 100, 0.0)
    out["basket"] = np.where(out.orders > 0, out.net / out.orders, 0.0)
    out = out.reset_index().rename(columns={"_k": "key"})
    out["label"] = out.key.map(lambda k: label_of(D, dim, k))
    return out


# ---------------------------------------------------------------- أسباب التغيير
def bridge(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """يفسر الفرق في المبيعات بين فترتين: كمية، سعر/خصم، تغير في مزيج المنتجات. (a = قبل، b = بعد)"""
    pa = a.groupby("product", observed=True).agg(n=("net", "sum"), q=("qty", "sum"))
    pb = b.groupby("product", observed=True).agg(n=("net", "sum"), q=("qty", "sum"))
    delta = pb.n.sum() - pa.n.sum()
    qa, qb = pa.q.sum(), pb.q.sum()
    avg_a = pa.n.sum() / qa if qa else 0.0
    volume = (qb - qa) * avg_a
    both = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
    price = float(((both.n_b / both.q_b - both.n_a / both.q_a) * both.q_b).sum())
    return {"delta": float(delta), "volume": float(volume), "price": price, "mix": float(delta - volume - price),
            "net_a": float(pa.n.sum()), "net_b": float(pb.n.sum()), "qty_a": int(qa), "qty_b": int(qb)}


def drivers(a: pd.DataFrame, b: pd.DataFrame, D, dim: str) -> pd.DataFrame:
    """مساهمة كل عنصر (فرع/نوع/قناة...) في التغير، مع فصل أثر الكمية عن أثر السعر."""
    ka, kb = by_dim(a, D, dim).set_index("key"), by_dim(b, D, dim).set_index("key")
    j = ka[["net", "qty"]].join(kb[["net", "qty"]], lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
    pa = np.where(j.qty_a > 0, j.net_a / j.qty_a.replace(0, np.nan), 0)
    pb = np.where(j.qty_b > 0, j.net_b / j.qty_b.replace(0, np.nan), 0)
    both = (j.qty_a > 0) & (j.qty_b > 0)
    j["volume"] = np.where(both, (j.qty_b - j.qty_a) * pa, j.net_b - j.net_a)
    j["price"] = np.where(both, (pb - pa) * j.qty_b, 0.0)
    j["delta"] = j.net_b - j.net_a
    j["growth"] = np.where(j.net_a > 0, j.delta / j.net_a.replace(0, np.nan) * 100, np.nan)
    j = j.reset_index().rename(columns={"index": "key"})
    j["label"] = j.key.map(lambda k: label_of(D, dim, k))
    return j.sort_values("delta")


# ---------------------------------------------------------------- ربحية الفروع
def branch_pnl(D, df: pd.DataFrame, start, end) -> pd.DataFrame:
    t = by_dim(df, D, "branch")[["key", "label", "net", "margin", "margin_pct", "qty", "orders"]].rename(columns={"key": "branch"})
    t["governorate"] = t.branch.map(D.branch_gov)
    t["payroll"], t["opex"] = 0.0, 0.0
    if len(D.costs):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        w = {}
        for m in pd.period_range(start, end, freq="M"):
            ov = (min(end, m.end_time.normalize()) - max(start, m.start_time)).days + 1
            w[str(m)] = max(ov, 0) / m.days_in_month
        c = D.costs[D.costs.month.isin(w)].copy()
        c["w"] = c.month.map(w)
        s = c.assign(payroll=c.payroll * c.w, opex=c.opex * c.w).groupby("branch")[["payroll", "opex"]].sum()
        t = t.drop(columns=["payroll", "opex"]).merge(s, left_on="branch", right_index=True, how="left").fillna({"payroll": 0, "opex": 0})
    t["cost_ok"] = t.payroll > 0          # الفروع اللي مفيش ليها مرتبات مسجلة ربحها الصافي مش دقيق
    t["profit"] = t.margin - t.payroll - t.opex
    t["profit_pct"] = np.where(t.net > 0, t.profit / t.net * 100, 0.0)
    return t.sort_values("profit", ascending=False)


def branch_flags(D, end=None, weeks: int = 13) -> pd.DataFrame:
    """مقارنة نمو كل فرع بنمو الشركة كلها (آخر 13 أسبوع مقابل نفس الفترة السنة اللي فاتت)."""
    end = pd.Timestamp(end or D.last)
    s, e = end - pd.Timedelta(days=weeks * 7 - 1), end
    ps, pe = s - pd.DateOffset(years=1), e - pd.DateOffset(years=1)
    cur = apply_filters(D.sales, s, e).groupby("branch", observed=True).net.sum()
    old = apply_filters(D.sales, ps, pe).groupby("branch", observed=True).net.sum()
    j = pd.concat([cur.rename("cur"), old.rename("old")], axis=1).fillna(0)
    opened = D.branches.set_index("branch_id").opening_date
    j = j[(j.old > 0) & j.index.map(lambda b: opened.get(b, pd.NaT) < ps)]
    net_g = (cur.sum() / old.sum() - 1) * 100
    j["growth"] = (j.cur / j.old - 1) * 100
    j["gap"] = j.growth - net_g
    j["lost"] = np.maximum(0, -(j.gap / 100) * j.old)
    j["label"] = j.index.map(D.branch_name)
    j["network_growth"] = net_g
    return j.sort_values("gap")


def peer_outliers(D, df: pd.DataFrame, col: str, dim: str = "branch", z: float = 2.0) -> pd.DataFrame:
    t = by_dim(df, D, dim)
    t = t[t.net > t.net.median() * 0.2]
    sd = t[col].std()
    t["z"] = (t[col] - t[col].mean()) / sd if sd else 0.0
    return t[t.z.abs() >= z].sort_values("z", ascending=False)


def margin_bridge(a: pd.DataFrame, b: pd.DataFrame, D, dim: str) -> pd.DataFrame:
    """بيفسر تغير هامش الربح (بالنقاط) بين فترتين: جزء من تغير هامش كل عنصر (اللي هو أسعار/خصومات/تكلفة)، وجزء من تغير وزنه في المبيعات (مزيج)."""
    ka, kb = by_dim(a, D, dim).set_index("key"), by_dim(b, D, dim).set_index("key")
    j = ka[["net", "margin"]].join(kb[["net", "margin"]], lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
    ta, tb = j.net_a.sum(), j.net_b.sum()
    sa, sb = j.net_a / ta, j.net_b / tb
    ma = np.where(j.net_a > 0, j.margin_a / j.net_a.replace(0, np.nan), np.nan)
    mb = np.where(j.net_b > 0, j.margin_b / j.net_b.replace(0, np.nan), np.nan)
    base = j.margin_a.sum() / ta
    both = (j.net_a > 0) & (j.net_b > 0)
    j["rate"] = np.where(both, (mb - ma) * sb * 100, 0.0)                    # أثر تغير هامش العنصر نفسه
    j["mix"] = np.where(j.net_b > 0, (sb - sa) * (np.nan_to_num(ma, nan=base) - base) * 100, np.where(j.net_a > 0, (0 - sa) * (np.nan_to_num(ma, nan=base) - base) * 100, 0.0))
    j["total"] = j.rate + j.mix
    j["margin_pct_a"], j["margin_pct_b"] = ma * 100, mb * 100
    j = j.reset_index().rename(columns={"index": "key"})
    j["label"] = j.key.map(lambda k: label_of(D, dim, k))
    return j.sort_values("total")


def on_time_rate(D, start, end) -> float | None:
    d = D.deliveries
    if not len(d):
        return None
    x = d[(d.ship_date >= pd.Timestamp(start)) & (d.ship_date <= pd.Timestamp(end))]
    return float((~x.late).mean() * 100) if len(x) else None


def profit_bridge(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """بيفسر تغير **قيمة الربح** (مش النسبة) بين فترتين: كمية، سعر، تكلفة القطعة، ومنتجات دخلت أو اتوقفت. (a = قبل، b = بعد)"""
    def g(x):
        return x.groupby("product", observed=True).agg(n=("net", "sum"), q=("qty", "sum"), c=("cost", "sum"))
    j = g(a).join(g(b), lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
    both = (j.q_a > 0) & (j.q_b > 0)
    qa, qb = j.q_a.where(j.q_a > 0, 1), j.q_b.where(j.q_b > 0, 1)
    pa, pb, ca, cb = j.n_a / qa, j.n_b / qb, j.c_a / qa, j.c_b / qb
    volume = ((j.q_b - j.q_a) * (pa - ca))[both].sum()
    price = ((pb - pa) * j.q_b)[both].sum()
    cost = (-(cb - ca) * j.q_b)[both].sum()
    ma, mb = j.n_a - j.c_a, j.n_b - j.c_b
    mix = (mb - ma)[~both].sum()
    return {"a": float(ma.sum()), "b": float(mb.sum()), "delta": float(mb.sum() - ma.sum()), "volume": float(volume), "price": float(price), "cost": float(cost), "mix": float(mix)}


def delta_by(a: pd.DataFrame, b: pd.DataFrame, D, dim: str, col: str = "margin") -> pd.DataFrame:
    """التغير في عمود (الربح مثلًا) لكل عنصر في بُعد معين."""
    ka, kb = by_dim(a, D, dim).set_index("key")[col], by_dim(b, D, dim).set_index("key")[col]
    j = pd.concat([ka.rename("a"), kb.rename("b")], axis=1).fillna(0)
    j["delta"] = j.b - j.a
    j = j.reset_index().rename(columns={"index": "key"})
    j["label"] = j.key.map(lambda k: label_of(D, dim, k))
    return j.sort_values("delta")
