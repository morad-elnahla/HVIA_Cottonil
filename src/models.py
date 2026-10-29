"""التوقعات (باختبار زمني صحيح) + توقع رجوع العملاء.

الفكرة:
- بنتوقع **الكمية** (القطع) لأنها هي اللي ليها موسمية واضحة، وبعدين الإيراد = الكمية المتوقعة × سعر القطعة الحالي.
  (الأسعار في الداتا بتتغير مرة في السنة في يناير، فتوقع الإيراد مباشرة بيخلط الموسم بزيادة الأسعار).
- كل نموذج بيتجرّب على 12 نقطة بداية قديمة (Rolling-origin): بنقص الداتا لحد نقطة البداية ونتوقع 6 شهور ونقارن بالفعلي.
- بنختار النموذج بأقدم 6 نقاط، ونقيس دقته على أحدث 6 نقاط (شهور محجوزة للاختبار ومالهاش دخل في الاختيار).
- المدى المحتمل (80%) محسوب من أخطاء الاختبار القديمة لكل أفق، ونقيس كمان نسبة اللي وقع فعلًا جواه.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import PROC, _rd

warnings.filterwarnings("ignore")
HORIZON, ORIGINS, SPLIT, MIN_TRAIN = 6, 12, 6, 36   # توقع 6 شهور، 12 نقطة بداية، أحدث 6 = اختبار، أقدم 6 = اختيار
MODEL_AR = {"snaive": "نفس الشهر السنة اللي فاتت", "sgrow": "السنة اللي فاتت × معدل النمو", "ets": "اتجاه + مواسم (ETS)", "theta": "Theta",
            "sarima": "SARIMA", "ensemble": "متوسط نماذج", "topdown": "من الشركة ككل × نصيب العنصر", "blend": "دمج: عنصر + نصيبه من الأعلى",
            "last_year": "نفس الشهر السنة اللي فاتت", "seasonal_growth": "السنة اللي فاتت × معدل النمو"}
COLS = ["level", "series", "target", "month", "forecast", "lo", "hi", "model", "h"]
LEVEL_AR = {"total": "كل الشركة", "category": "نوع منتج", "branch": "فرع", "product": "منتج"}


# ---------------------------------------------------------------- السلاسل الشهرية
def monthly(sales: pd.DataFrame, col: str, by: str | None = None) -> pd.DataFrame | pd.Series:
    """سلسلة شهرية للشهور الكاملة بس (بيتجاهل أول شهر وآخر شهر لو ناقصين)."""
    first, last = sales.date.min(), sales.date.max()
    start = first.to_period("M") + (1 if first.day > 1 else 0)
    end = last.to_period("M") - (0 if last.normalize() == (last + pd.offsets.MonthEnd(0)).normalize() else 1)
    idx = pd.period_range(start, end, freq="M")
    m = sales.date.dt.to_period("M")
    if by:
        return sales.groupby([m, by], observed=True)[col].sum().unstack(fill_value=0).reindex(idx, fill_value=0)
    return sales.groupby(m)[col].sum().reindex(idx, fill_value=0)


# ---------------------------------------------------------------- النماذج
def _lg(y):
    return np.log(np.maximum(np.asarray(y, dtype=float), 1.0))


def _snaive(y, h):
    return np.array([y[len(y) - 12 + (k % 12)] for k in range(h)], dtype=float)


def _sgrow(y, h):
    return _snaive(y, h) * np.clip(y[-6:].sum() / max(y[-18:-12].sum(), 1.0), 0.6, 1.8)


def _ets(y, h):
    m = ExponentialSmoothing(_lg(y), trend="add", damped_trend=True, seasonal="add", seasonal_periods=12, initialization_method="estimated").fit()
    return np.exp(m.forecast(h))


def _theta(y, h):
    idx = pd.period_range("2000-01", periods=len(y), freq="M")
    return np.maximum(np.asarray(ThetaModel(pd.Series(np.maximum(y, 1.0), index=idx), period=12, deseasonalize=True).fit().forecast(h)), 1.0)


def _sarima(y, h):
    r = SARIMAX(_lg(y), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
    return np.exp(r.forecast(h))


BASE = {"snaive": _snaive, "sgrow": _sgrow, "ets": _ets, "theta": _theta, "sarima": _sarima}
SETS = {"total": ["snaive", "sgrow", "ets", "theta", "sarima"], "category": ["snaive", "sgrow", "ets", "theta"]}


def predict_all(y, h: int, names: list[str]) -> dict[str, np.ndarray]:
    y = np.maximum(np.asarray(y, dtype=float), 1.0)
    out = {}
    for k in names:
        try:
            p = BASE[k](y, h)
            if np.all(np.isfinite(p)) and np.all(p > 0):
                out[k] = p
        except Exception:
            pass
    ens = [out[k] for k in ("ets", "theta", "sarima") if k in out]
    if len(ens) >= 2:
        out["ensemble"] = np.mean(ens, axis=0)
    return out


def backtest(y: np.ndarray, idx: pd.PeriodIndex, names: list[str], origins: int = ORIGINS) -> pd.DataFrame:
    """نقاط بداية قديمة: بنقص الداتا عند cut ونتوقع 6 شهور قدام، ونقارن بالفعلي (من غير ما نبص على أي بيانات بعد cut)."""
    rows, n = [], len(y)
    for o in range(origins):
        cut = n - HORIZON - o
        if cut < MIN_TRAIN:
            break
        for k, p in predict_all(y[:cut], HORIZON, names).items():
            for h in range(HORIZON):
                rows.append((k, o, h + 1, str(idx[cut + h]), float(y[cut + h]), float(p[h])))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "month", "actual", "pred"])


def _wape(d: pd.DataFrame) -> float:
    return float((d.actual - d.pred).abs().sum() / d.actual.sum() * 100) if len(d) and d.actual.sum() else np.nan


def _bias(d: pd.DataFrame) -> float:
    return float((d.pred - d.actual).sum() / d.actual.sum() * 100) if len(d) and d.actual.sum() else np.nan


def choose(bt: pd.DataFrame) -> tuple[str, float]:
    """النموذج الأحسن على أقدم نقاط البداية (الاختيار)، من غير ما نبص على نقاط الاختبار الأحدث."""
    val = bt[(bt.origin >= SPLIT) & (bt.h <= 3)]
    scores = {k: _wape(val[val.model == k]) for k in val.model.unique()}
    scores = {k: v for k, v in scores.items() if v == v}
    best = min(scores, key=scores.get)
    if "ensemble" in scores and scores["ensemble"] <= scores[best] * 1.05:      # لو المتوسط قريب من الأحسن، المتوسط أثبت
        best = "ensemble"
    return best, scores[best]


QPAIRS = [(0.10, 0.90), (0.05, 0.95), (0.025, 0.975), (0.01, 0.99)]


def _arrays(bt: pd.DataFrame, model: str):
    d = bt[bt.model == model]
    return (d.origin.to_numpy(), d.h.to_numpy(), np.log(d.actual.clip(lower=1).to_numpy() / d.pred.clip(lower=1).to_numpy()), d.actual.to_numpy(), d.pred.to_numpy())


def _q(o, hh, r, h, q, older_than=None):
    m = np.abs(hh - h) <= 1
    if older_than is not None:
        m &= o > older_than
    x = r[m]
    return (float(np.exp(np.quantile(x, q[0]))), float(np.exp(np.quantile(x, q[1])))) if len(x) >= 4 else (np.nan, np.nan)


def interval_factors(bt: pd.DataFrame, model: str, q=(0.10, 0.90), older_than: int | None = None) -> pd.DataFrame:
    """نسب الخطأ (الفعلي ÷ المتوقع) لكل أفق. older_than=o معناها: نستخدم أخطاء نقاط بداية أقدم من o بس. بنضم أفق ±1 عشان العينة تبقى أكبر."""
    o, hh, r, _, _ = _arrays(bt, model)
    rows = [(h, *_q(o, hh, r, h, q, older_than)) for h in range(1, HORIZON + 1)]
    return pd.DataFrame(rows, columns=["h", "q10", "q90"]).ffill().bfill()


def _coverage(bt: pd.DataFrame, model: str, q) -> tuple[int, int]:
    """كام نقطة من نقاط الاختبار (أحدث 6 نقاط بداية، أفق 1-3) وقعت جوه المدى، والمدى بيتحسب من أخطاء أقدم بس."""
    o, hh, r, act, pred = _arrays(bt, model)
    hit = tot = 0
    for og in range(SPLIT):
        for h in (1, 2, 3):
            lo, hi = _q(o, hh, r, h, q, older_than=og)
            if lo != lo:
                lo, hi = (np.nan, np.nan)
            m = (o == og) & (hh == h)
            if m.any() and lo == lo:
                hit += int(((act[m] >= pred[m] * lo) & (act[m] <= pred[m] * hi)).sum())
                tot += int(m.sum())
    return hit, tot


def calibrate(items: list[dict], target: float = 0.78) -> tuple[float, float]:
    """أضيق زوج كميات بيحقق تغطية ≈80% على شهور الاختبار (مجمّعة لكل المستوى)."""
    for q in QPAIRS:
        h = t = 0
        for it in items:
            a, b = _coverage(it["bt"], it["model"], q)
            h, t = h + a, t + b
        if t and h / t >= target:
            return q
    return QPAIRS[-1]


def _price(sales: pd.DataFrame, mask=None) -> float:
    """سعر القطعة المتوسط في آخر 3 شهور كاملة."""
    end = sales.date.max()
    last_full = end.to_period("M") - (0 if end.normalize() == (end + pd.offsets.MonthEnd(0)).normalize() else 1)
    s = sales[(sales.date >= (last_full - 2).start_time) & (sales.date <= last_full.end_time)]
    if mask is not None:
        s = s[mask(s)]
    return float(s.net.sum() / max(s.qty.sum(), 1))


def _reliability(wape_t, base_t, vol) -> tuple[str, str]:
    if not wape_t == wape_t or vol < 5:
        return "ضعيفة", "حجم المبيعات صغير أو التاريخ قصير"
    if base_t == base_t and wape_t > base_t * 1.05:
        return "ضعيفة", "مش أحسن من الطريقة البسيطة (نفس الشهر السنة اللي فاتت)"
    if wape_t <= 8:
        return "عالية", ""
    if wape_t <= 15:
        return "متوسطة", ""
    return "ضعيفة", "الخطأ كبير، استخدمه كاتجاه عام مش كرقم دقيق"


def _finish(it: dict, q) -> tuple[list, dict]:
    """بيطلّع صفوف التوقع النهائي (مدى من كل الأخطاء) وملخص الجودة (تغطية من الأخطاء الأقدم)."""
    bt, model, y, idx = it["bt"], it["model"], it["y"], it["idx"]
    hit, tot = _coverage(bt, model, q)
    test = bt[(bt.model == model) & (bt.origin < SPLIT)]
    t13, t46 = test[test.h <= 3], test[test.h > 3]
    base = bt[(bt.model == "snaive") & (bt.origin < SPLIT) & (bt.h <= 3)]
    w, b = _wape(t13), _wape(base)
    rel, note = _reliability(w, b, it["vol"])
    s = dict(level=it["level"], series=it["name"], target=it["target"], model=model, wape_test=w, wape_val=it["val_w"], wape_h46=_wape(t46), baseline_test=b, bias_test=_bias(t13),
             coverage_test=hit / tot * 100 if tot else np.nan, n_test=int(len(t13)), reliability=rel, note=note, price=it["price"], volume=it["vol"])
    f = interval_factors(bt, model, q)
    if model == "topdown":
        p = it["td"]
    elif model == "blend":
        p = 0.5 * predict_all(y, HORIZON, ["ets"])["ets"] + 0.5 * it["td"]
    else:
        p = predict_all(y, HORIZON, it["names"])[model]
    months = pd.period_range(idx[-1] + 1, periods=HORIZON, freq="M")
    rows = [(it["level"], it["name"], it["target"], str(m), float(p[i]), float(p[i] * min(f.q10[i], 1.0)), float(p[i] * max(f.q90[i], 1.0)), model, i + 1) for i, m in enumerate(months)]
    return rows, s


def run_forecast(D, log=print) -> None:
    t0 = time.time()
    sales = D.sales
    levels: dict[str, list[dict]] = {"total": [], "category": [], "branch": [], "product": []}
    tot_q = monthly(sales, "qty")
    idx, y = tot_q.index, tot_q.values.astype(float)
    # ---- الشركة ككل (كمية)
    bt_t = backtest(y, idx, SETS["total"])
    best, vw = choose(bt_t)
    price_t = _price(sales)
    levels["total"].append(dict(level="total", name="__total__", target="qty", bt=bt_t, model=best, y=y, idx=idx, vol=y[-3:].mean(), price=price_t, val_w=vw, names=SETS["total"]))
    # ---- أنواع المنتجات
    qc = monthly(sales, "qty", by="category")
    cat_bt = {}
    for c in qc.columns:
        yc = qc[c].values.astype(float)
        bt = backtest(yc, idx, SETS["category"])
        best, vw = choose(bt)
        cat_bt[c] = bt
        levels["category"].append(dict(level="category", name=c, target="qty", bt=bt, model=best, y=yc, idx=idx, vol=yc[-3:].mean(), price=_price(sales, lambda s, c=c: s.category == c), val_w=vw, names=SETS["category"]))
    log(f"  الشركة والأنواع جاهزين ({time.time() - t0:.0f} ثانية)")
    # ---- الفروع والمنتجات: بنقارن (1) نموذج مستقل (2) نصيب من الأعلى (3) دمج الاتنين، ونختار طريقة واحدة للمستوى كله
    ref_t = bt_t[bt_t.model == "ets"].set_index(["origin", "h"]).pred
    tot_ets_final = predict_all(y, HORIZON, ["ets"])["ets"]
    cat_ets_final = {c: predict_all(qc[c].values.astype(float), HORIZON, ["ets"])["ets"] for c in qc.columns}
    for level, parent in (("branch", None), ("product", "category")):
        q = monthly(sales, "qty", by=level)
        q = q.loc[:, [c for c in q.columns if (q[c].iloc[-24:] > 0).sum() >= 24 and q[c].iloc[-12:].sum() >= 30]]
        pmap = {c: D.prod_cat.get(c) for c in q.columns} if parent else {}
        per = {}
        for c in q.columns:
            yc = q[c].values.astype(float)
            n = len(yc)
            bt = backtest(yc, idx, ["ets", "snaive"], origins=ORIGINS - 3)
            rows = []
            for o in sorted(bt.origin.unique()):
                cut = n - HORIZON - o
                if parent:
                    cb = cat_bt.get(pmap[c])
                    ref = cb[(cb.model == "ets") & (cb.origin == o)].set_index("h").pred if cb is not None else None
                    par = qc[pmap[c]].values.astype(float)
                else:
                    ref = ref_t.xs(o, level="origin") if o in ref_t.index.get_level_values(0) else None
                    par = y
                if ref is None or len(ref) < HORIZON:
                    continue
                share = yc[cut - 12:cut].sum() / max(par[cut - 12:cut].sum(), 1.0)
                for h in range(1, HORIZON + 1):
                    own = bt[(bt.model == "ets") & (bt.origin == o) & (bt.h == h)].pred
                    if not len(own):
                        continue
                    td = float(ref.loc[h] * share)
                    a = yc[cut + h - 1]
                    rows += [("topdown", o, h, str(idx[cut + h - 1]), a, td), ("blend", o, h, str(idx[cut + h - 1]), a, 0.5 * (float(own.iloc[0]) + td))]
            per[c] = pd.concat([bt, pd.DataFrame(rows, columns=bt.columns)])
        pooled = pd.concat(per.values())
        val = pooled[(pooled.origin >= SPLIT) & (pooled.h <= 3)]
        scores = {k: _wape(val[val.model == k]) for k in ("ets", "topdown", "blend")}
        best = min(scores, key=scores.get)
        log(f"  {LEVEL_AR[level]}: {len(q.columns)} سلسلة. الطريقة المختارة: {MODEL_AR[best]} ({', '.join(f'{MODEL_AR[k]} {v:.1f}%' for k, v in scores.items())})  ({time.time() - t0:.0f} ثانية)")
        for c in q.columns:
            yc = q[c].values.astype(float)
            td = None
            if best in ("topdown", "blend"):
                par_y, par_f = (qc[pmap[c]].values.astype(float), cat_ets_final[pmap[c]]) if parent else (y, tot_ets_final)
                td = par_f * (yc[-12:].sum() / max(par_y[-12:].sum(), 1.0))
            levels[level].append(dict(level=level, name=c, target="qty", bt=per[c], model=best, y=yc, idx=idx, vol=yc[-3:].mean(), price=_price(sales, lambda s, c=c, lv=level: s[lv] == c),
                                      val_w=scores[best], names=["ets"], td=td))
    # ---- معايرة المدى المحتمل لكل مستوى، وبعدين التوقع النهائي وملخص الجودة
    fc_rows, sm_rows, bt_all = [], [], []
    for lv, items in levels.items():
        qpair = calibrate(items)
        for it in items:
            rows, s = _finish(it, qpair)
            fc_rows += rows
            s["interval_q"] = f"{int((qpair[1] - qpair[0]) * 100)}%"
            sm_rows.append(s)
        log(f"  مدى {LEVEL_AR[lv]}: كميات {qpair} → تغطية الاختبار {np.nanmedian([r['coverage_test'] for r in sm_rows if r['level'] == lv]):.0f}%")
        if lv in ("total", "category"):
            bt_all += [it["bt"].assign(level=lv, series=it["name"], target="qty") for it in items]
    # ---- الإيراد = الكمية × السعر الحالي (مع توقع الإيراد المباشر للمقارنة)
    fc = pd.DataFrame(fc_rows, columns=COLS)
    tq = fc[(fc.level == "total") & (fc.target == "qty")]
    net = tq.assign(target="net", forecast=tq.forecast * price_t, lo=tq.lo * price_t, hi=tq.hi * price_t)
    tot_n = monthly(sales, "net").values.astype(float)
    bt_n = backtest(tot_n, idx, ["ets"])
    it_n = dict(level="total", name="__total__", target="net_trend", bt=bt_n, model="ets", y=tot_n, idx=idx, vol=tot_n[-3:].mean(), price=price_t, val_w=np.nan, names=["ets"])
    rn, sn = _finish(it_n, calibrate([it_n]))
    sn["interval_q"] = "80%"
    sm_rows.append(sn)
    fc = pd.concat([fc, net, pd.DataFrame(rn, columns=COLS)], ignore_index=True)
    fc.to_parquet(PROC / "forecast.parquet", index=False)
    pd.DataFrame(sm_rows).to_parquet(PROC / "fquality.parquet", index=False)
    pd.concat(bt_all).to_parquet(PROC / "backtest.parquet", index=False)
    log(f"  التوقعات خلصت في {time.time() - t0:.0f} ثانية")


def accuracy(D=None) -> pd.DataFrame:
    """جدول جودة التوقعات (كل مستوى/سلسلة): الخطأ على شهور الاختبار، والمدى، والموثوقية."""
    return _rd("fquality")


def mtd_nowcast(D, series: str = "__total__", level: str = "total", target: str = "qty") -> dict | None:
    """لو الشهر الحالي ماخلصش: بنقدّر ناتج الشهر من اللي اتباع لحد دلوقتي ونمط الشهور اللي فاتت (نسبة اللي بيتباع لحد نفس اليوم)."""
    last = D.last
    if last.day == last.days_in_month:
        return None
    s = D.sales
    if level == "category":
        s = s[s.category == series]
    elif level == "branch":
        s = s[s.branch == series]
    elif level == "product":
        s = s[s["product"] == series]
    col = "qty" if target == "qty" else "net"
    cur = float(s[(s.date >= last.replace(day=1)) & (s.date <= last)][col].sum())
    shares = []
    for k in range(1, 13):
        m = (last.to_period("M") - k)
        full = s[(s.date >= m.start_time) & (s.date <= m.end_time)][col].sum()
        part = s[(s.date >= m.start_time) & (s.date <= m.start_time + pd.Timedelta(days=last.day - 1, hours=23, minutes=59))][col].sum()
        if full > 0:
            shares.append(part / full)
    if len(shares) < 6:
        return None
    sh = float(np.median(shares))
    return {"so_far": cur, "typical_share": sh, "projected": cur / sh if sh else None, "day": int(last.day), "days": int(last.days_in_month)}


# ---------------------------------------------------------------- العملاء
def _features(o: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    o = o[o.date <= cut]
    g = o.groupby("customer", observed=True)
    f = pd.DataFrame({"recency": (cut - g.date.max()).dt.days, "freq": g.size(), "monetary": g.net.sum(),
                      "tenure": (cut - g.date.min()).dt.days})
    f["avg_gap"] = f.tenure / np.maximum(f.freq - 1, 1)
    f["avg_order"] = f.monetary / f.freq
    f["recency_ratio"] = f.recency / (f.avg_gap + 1)
    for d in (90, 180, 365):
        f[f"n{d}"] = o[o.date > cut - pd.Timedelta(days=d)].groupby("customer", observed=True).size().reindex(f.index).fillna(0)
    f["spend12m"] = o[o.date > cut - pd.Timedelta(days=365)].groupby("customer", observed=True).net.sum().reindex(f.index).fillna(0)
    return f


def _label(o: pd.DataFrame, f: pd.DataFrame, cut: pd.Timestamp, days: int) -> pd.Series:
    buyers = o[(o.date > cut) & (o.date <= cut + pd.Timedelta(days=days))].customer.unique()
    return f.index.isin(buyers).astype(int)


def _rfm(f: pd.DataFrame) -> pd.Series:
    r = pd.qcut(f.recency.rank(method="first", ascending=False), 5, labels=False) + 1      # 5 = اشترى حديثًا
    fq = pd.qcut(f.freq.rank(method="first"), 5, labels=False) + 1
    seg = np.select(
        [(r >= 4) & (fq >= 4), (r >= 3) & (fq >= 4), (r >= 4) & (fq <= 3), (r == 3) & (fq <= 3), (r <= 2) & (fq >= 3)],
        ["أبطال", "أوفياء", "واعدين", "محتاجين اهتمام", "كانوا مهمين وبعدوا"], default="ضايعين أو شبه ضايعين")
    return pd.Series(seg, index=f.index)


def run_churn(D, log=print, days: int = 120) -> dict:
    s = D.sales[D.sales.customer.notna()]
    o = s.groupby("order", observed=True).agg(customer=("customer", "first"), date=("date", "first"), net=("net", "sum"))
    o["customer"] = o.customer.astype(str)
    T = D.last.normalize()
    cuts = [T - pd.Timedelta(days=days * k) for k in (1, 2, 3, 4)]
    sets = []
    for c in cuts:
        f = _features(o, c)
        f["y"] = _label(o, f, c, days)
        f["cut"] = c
        sets.append(f)
    X = [c for c in sets[0].columns if c not in ("y", "cut")]
    train, valid = pd.concat(sets[1:]), sets[0]
    mdl = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0).fit(train[X], train.y)
    auc = roc_auc_score(valid.y, mdl.predict_proba(valid[X])[:, 1])
    base = roc_auc_score(valid.y, -valid.recency_ratio)
    full = pd.concat(sets)
    mdl = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0).fit(full[X], full.y)
    now = _features(o, T)
    base_rate = float(valid.y.mean())
    predictive = bool(auc >= 0.58 and auc > base + 0.03)      # لو النموذج مش أحسن من التخمين، مش هنخدع المستخدم بأرقام وهمية
    now["p_return"] = mdl.predict_proba(now[X])[:, 1] if predictive else base_rate
    now["segment"] = _rfm(now)
    now["last_purchase"] = T - pd.to_timedelta(now.recency, unit="D")
    now["absence"] = pd.cut(now.recency, [-1, 90, 180, 365, 1e9], labels=["نشيط", "بعيد 3-6 شهور", "بعيد 6-12 شهر", "بعيد أكتر من سنة"]).astype(str)
    now["risk"] = pd.cut(now.p_return, [-1, 0.25, 0.5, 2], labels=["عالية", "متوسطة", "منخفضة"]).astype(str) if predictive else "غير متاح"
    out = now.reset_index().rename(columns={"customer": "customer_id"})[
        ["customer_id", "last_purchase", "recency", "freq", "monetary", "spend12m", "avg_order", "avg_gap", "p_return", "segment", "absence", "risk"]]
    out.to_parquet(PROC / "churn.parquet", index=False)
    m = {"auc": round(float(auc), 3), "auc_baseline": round(float(base), 3), "base_rate": round(float(valid.y.mean()), 3),
         "predictive": predictive, "window_days": days, "n_train": int(len(train)), "n_valid": int(len(valid))}
    log(f"  نموذج العملاء: AUC={m['auc']} مقابل {m['auc_baseline']} للقاعدة البسيطة")
    return m


def run_all(D, log=print) -> None:
    log("التوقعات...")
    run_forecast(D, log)
    log("العملاء...")
    m = run_churn(D, log)
    (PROC / "model_metrics.json").write_text(json.dumps({"churn": m}, ensure_ascii=False, indent=1), encoding="utf-8")
