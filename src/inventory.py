"""ذكاء المخزون: بنربط مخزون كل فرع بمعدل بيعه والطلب المتوقع، ونطلع تحويلات مقترحة بين الفروع والمخازن."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import apply_filters
from .models import monthly

DEFAULTS = {"lead": 2.0, "safety": 1.0, "target": 8.0, "over": 26.0, "weeks": 13}   # كلها بالأسابيع
STATUS_ORDER = ["حرج", "منخفض", "سليم", "زيادة", "راكد"]
WAREHOUSES = {"WH01": "مخزن القاهرة المركزي", "WH02": "مخزن الإسكندرية", "WH03": "مخزن الجيزة", "WH04": "مخزن الدلتا",
              "WH05": "مخزن الصعيد", "WH07": "مخزن المنتج التام", "WH09": "مخزن الشرقية", "WH10": "مخزن البحيرة",
              "WH11": "مخزن الغربية", "WH12": "مخزن القناة"}
WH_OF = {"القاهرة": "WH01", "الإسكندرية": "WH02", "مطروح": "WH02", "الجيزة": "WH03", "الفيوم": "WH03", "بني سويف": "WH03",
         "الدقهلية": "WH04", "دمياط": "WH04", "المنيا": "WH05", "أسيوط": "WH05", "سوهاج": "WH05", "قنا": "WH05", "أسوان": "WH05",
         "الشرقية": "WH09", "البحيرة": "WH10", "الغربية": "WH11", "المنوفية": "WH11", "كفر الشيخ": "WH11",
         "الإسماعيلية": "WH12", "بورسعيد": "WH12", "البحر الأحمر": "WH01"}


def growth_factor(D) -> pd.Series:
    """نسبة الطلب المتوقع في الـ3 شهور الجاية على آخر 3 شهور فعلية، لكل نوع (بتشمل المواسم زي نوفمبر وديسمبر).
    لو توقع النوع ضعيف الموثوقية، بنقرّب النسبة لـ1 (نصها) عشان منبنيش قرار مخزون على توقع مش ثابت."""
    fc = D.forecast[(D.forecast.level == "category") & (D.forecast.target == "qty")]
    if not len(fc):
        return pd.Series(dtype=float)
    nxt = fc.sort_values("h").groupby("series").head(3).groupby("series").forecast.mean()
    act = monthly(D.sales, "qty", by="category").iloc[-3:].mean()
    gf = (nxt / act.reindex(nxt.index)).clip(0.6, 1.6)
    if len(D.fq):
        rel = D.fq[(D.fq.level == "category") & (D.fq.target == "qty")].set_index("series").reliability.reindex(gf.index)
        w = np.where(rel == "ضعيفة", 0.5, 1.0)
        gf = 1 + (gf - 1) * w
    return gf


def intel(D, **kw) -> pd.DataFrame:
    p = {**DEFAULTS, **kw}
    end = D.last.normalize()
    s = apply_filters(D.sales, end - pd.Timedelta(days=int(p["weeks"] * 7) - 1), end)
    u = s.groupby(["branch", "category"], observed=True).agg(units=("qty", "sum"), net=("net", "sum")).reset_index()
    u["branch"], u["category"] = u.branch.astype(str), u.category.astype(str)
    st = D.stock[D.stock.kind == "Branch"]
    st = st[st.year_month == st.year_month.max()][["location", "category", "closing_units", "closing_stock_value_EGP", "stockout_risk_score"]]
    st = st.rename(columns={"location": "branch", "closing_units": "stock", "closing_stock_value_EGP": "value", "stockout_risk_score": "reported_risk"})
    st["category"] = st.category.astype(str)
    df = st.merge(u, on=["branch", "category"], how="left").fillna({"units": 0, "net": 0})
    gf = growth_factor(D)
    df["growth"] = df.category.map(gf).fillna(1.0)
    df["weekly_demand"] = df.units / p["weeks"] * df.growth
    df["cover_weeks"] = np.where(df.weekly_demand > 0, df.stock / df.weekly_demand.replace(0, np.nan), np.inf)
    df["unit_cost"] = df.value / df.stock.replace(0, np.nan)
    df["price"] = np.where(df.units > 0, df.net / df.units.replace(0, np.nan), np.nan)
    df["price"] = df.price.fillna(df.groupby("category").price.transform("median"))
    c = df.cover_weeks
    slow = (df.weekly_demand < 0.5) & (c <= p["over"])      # مبيعات بطيئة جدًا: مش بنعتبرها نقص حتى لو المخزون قليل (الأثر ضعيف)
    df["status"] = np.select([df.units == 0, slow, c < p["lead"] + p["safety"], c < p["target"] * 0.75, c <= p["over"]],
                             ["راكد", "سليم", "حرج", "منخفض", "سليم"], default="زيادة")
    df["need"] = np.where(df.status.isin(["حرج", "منخفض"]), np.maximum(0, p["target"] * df.weekly_demand - df.stock), 0).round()
    df["excess"] = np.where(df.status.isin(["زيادة", "راكد"]), np.maximum(0, df.stock - p["over"] * df.weekly_demand), 0).round()
    df["exposure"] = np.where(df.status.isin(["حرج", "منخفض"]), np.maximum(0, p["lead"] - c) * df.weekly_demand * df.price, 0)
    df["tied_capital"] = df.excess * df.unit_cost
    df["governorate"] = df.branch.map(D.branch_gov)
    df["branch_name"] = df.branch.map(D.branch_name)
    df["category_ar"] = df.category.map(D.cat_ar)
    return df


def warehouse_stock(D) -> pd.DataFrame:
    w = D.stock[D.stock.kind == "Warehouse"]
    w = w[w.year_month == w.year_month.max()]
    w = w[w.location.isin(WAREHOUSES)]
    return w.assign(category=w.category.astype(str)).groupby(["location", "category"]).closing_units.sum().reset_index()


def transfers(D, df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    """مطابقة بسيطة: الفرع الناقص بياخد من فرع زيادة في نفس المحافظة، وبعدين من مخزن المنطقة."""
    wh = warehouse_stock(D).set_index(["location", "category"]).closing_units.to_dict()
    rows = []
    for cat, g in df.groupby("category"):
        rec = g[g.need > 0].sort_values("cover_weeks")
        donors = g[g.excess > 0].copy()
        left = donors.set_index("branch").excess.to_dict()
        for _, r in rec.iterrows():
            need = need0 = float(r.need)
            same = [b for b in donors[donors.governorate == r.governorate].sort_values("excess", ascending=False).branch if left.get(b, 0) > 0 and b != r.branch]
            for b in same:
                q = min(need, left[b])
                if q >= 1:
                    rows.append((cat, r.branch, r.branch_name, "فرع", D.branch_name.get(b, b), round(q), r.status, r.exposure * q / need0, r.cover_weeks))
                    left[b] -= q
                    need -= q
                if need < 1:
                    break
            if need >= 1:
                home = WH_OF.get(r.governorate, "WH01")
                for w in [home] + [x for x in WAREHOUSES if x != home]:
                    avail = wh.get((w, cat), 0)
                    q = min(need, avail * 0.5)        # مش بنفضي المخزن: أقصى 50% منه للنوع الواحد
                    if q >= 1:
                        rows.append((cat, r.branch, r.branch_name, "مخزن", WAREHOUSES[w], round(q), r.status, r.exposure * q / need0, r.cover_weeks))
                        wh[(w, cat)] = avail - q
                        need -= q
                    if need < 1:
                        break
    t = pd.DataFrame(rows, columns=["category", "to_id", "to", "from_kind", "from", "units", "status", "exposure", "cover_weeks"])
    if not len(t):
        return t
    t["category_ar"] = t.category.map(D.cat_ar)
    return t.sort_values("exposure", ascending=False).head(max_rows).reset_index(drop=True)


def category_plan(D, df: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
    """لكل نوع: المخزون الكلي (فروع + مخازن) مقارنة بالطلب المتوقع، والقرار المقترح (يشتري ولا يوقف)."""
    p = {**DEFAULTS, **(p or {})}
    g = df.groupby("category").agg(branch_stock=("stock", "sum"), demand=("weekly_demand", "sum"), tied=("tied_capital", "sum"),
                                   critical=("status", lambda s: int(s.isin(["حرج", "منخفض"]).sum())), exposure=("exposure", "sum"))
    w = warehouse_stock(D).groupby("category").closing_units.sum()
    g["wh_stock"] = w.reindex(g.index).fillna(0)
    g["total_stock"] = g.branch_stock + g.wh_stock
    g["cover_weeks"] = g.total_stock / g.demand.replace(0, np.nan)
    g["branch_cover"] = g.branch_stock / g.demand.replace(0, np.nan)
    g["wh_cover"] = g.wh_stock / g.demand.replace(0, np.nan)
    bc = g.branch_cover
    g["status"] = np.select([bc < p["lead"] + p["safety"], bc < p["target"] * 0.75, bc <= p["over"]], ["حرج", "منخفض", "سليم"], default="زيادة")
    g["growth"] = g.index.map(growth_factor(D)).fillna(1.0)
    g["action"] = np.select([g.cover_weeks > p["over"] * 2, g.cover_weeks > p["over"], g.cover_weeks < p["target"]],
                            ["وقف الشراء تمامًا", "قلل الشراء", "اطلب توريد"], default="كمّل بالمعدل العادي")
    g["category_ar"] = g.index.map(D.cat_ar)
    return g.reset_index().sort_values("cover_weeks")


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    return {"units": float(df.stock.sum()), "value": float(df.value.sum()), "cover": float(df.stock.sum() / max(df.weekly_demand.sum(), 1e-9)),
            "critical": int((df.status == "حرج").sum()), "low": int((df.status == "منخفض").sum()), "over": int((df.status == "زيادة").sum()),
            "dead": int((df.status == "راكد").sum()), "n": n, "exposure": float(df.exposure.sum()), "tied": float(df.tied_capital.sum())}
