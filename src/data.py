"""تجهيز البيانات: قراءة ملفات CSV، التحقق من صحتها، وتخزينها بصيغة سريعة (parquet).

التشغيل من سطر الأوامر:  python -m src.data
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
CORE = ["sales_orders.csv", "sales_order_items.csv", "branches.csv", "products_catalog.csv"]
OPTIONAL = ["customers.csv", "monthly_stock.csv", "sales_returns.csv", "delivery_shipments.csv", "employees.csv",
            "payroll.csv", "operating_expenses.csv", "warehouses.csv", "marketing_campaigns.csv",
            "purchase_order_items.csv", "inventory_transfer_items.csv", "inventory_transfers.csv"]

CAT_AR = {"Accessories": "إكسسوارات", "Boys pajamas": "بيجامات أولاد", "Girls pajamas": "بيجامات بنات",
          "Girls underwear": "داخلي بنات", "Home textiles": "مفروشات منزلية", "Kids underwear": "داخلي أطفال",
          "Men bottoms": "بناطيل وشورتات رجالي", "Men boxers": "بوكسرات رجالي", "Men tops": "تيشيرتات رجالي",
          "Men underwear": "داخلي رجالي", "Socks": "شرابات", "Women bottoms": "بناطيل حريمي",
          "Women bras": "حمالات", "Women pajamas": "بيجامات حريمي", "Women tops": "بلوزات حريمي",
          "Women underwear": "داخلي حريمي"}
CH_AR = {"Retail POS": "الفروع", "E-commerce": "أونلاين", "Wholesale / reseller": "الجملة"}
VAT = 0.14


# ---------------------------------------------------------------- أدوات الملفات
def find_raw(base: Path | None = None) -> Path | None:
    base = base or RAW
    hit = next(iter(base.rglob("sales_orders.csv")), None) if base.exists() else None
    return hit.parent if hit else None


def fingerprint(d: Path | None) -> dict:
    if d is None:
        return {}
    return {f: [(d / f).stat().st_size, int((d / f).stat().st_mtime)] for f in CORE + OPTIONAL if (d / f).exists()}


def save_upload(files) -> int:
    """يحفظ ملفات مرفوعة (zip أو csv) داخل data/raw ويرجع عدد الملفات."""
    RAW.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for m in z.namelist():
                    if m.lower().endswith(".csv"):
                        (RAW / Path(m).name).write_bytes(z.read(m))
                        n += 1
        elif f.name.lower().endswith(".csv"):
            (RAW / f.name).write_bytes(f.getvalue())
            n += 1
    return n


def needs_rebuild() -> str | None:
    """يرجع سبب لو الداتا الخام اتغيرت عن آخر بناء، أو None."""
    raw = find_raw()
    if raw is None:
        return None
    meta = PROC / "meta.json"
    if not meta.exists():
        return "لسه محدش بنى البيانات"
    old = json.loads(meta.read_text(encoding="utf-8")).get("fingerprint", {})
    return "فيه ملفات بيانات جديدة أو متعدلة" if old != fingerprint(raw) else None


# ---------------------------------------------------------------- بناء البيانات
def build(raw_dir: Path | None = None, log=print, with_models: bool = True) -> dict:
    t0 = time.time()
    raw_dir = raw_dir or find_raw()
    if raw_dir is None:
        raise FileNotFoundError("حط ملفات الـ CSV (أو فولدر COTTONIL_Data) جوه data/raw")
    missing = [f for f in CORE if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"ملفات أساسية ناقصة: {', '.join(missing)}")
    PROC.mkdir(parents=True, exist_ok=True)
    rd = lambda n, **k: pd.read_csv(raw_dir / n, **k)
    has = lambda n: (raw_dir / n).exists()
    health: list[dict] = []

    def chk(name, ok, detail, warn=False):
        health.append({"check": name, "status": "ok" if ok else ("warn" if warn else "error"), "detail": detail})

    log("قراءة المبيعات...")
    so, it = rd("sales_orders.csv"), rd("sales_order_items.csv")
    br, pr = rd("branches.csv"), rd("products_catalog.csv")
    pr = pr.drop_duplicates("product_id")
    pr["category"] = pr.category_en
    pr["category_ar"] = pr.category_en.map(CAT_AR).fillna(pr.category_en)

    # ---- فحوصات الجودة
    chk("مفيش فواتير أو سطور مكررة", so.sales_order_id.is_unique and it.sales_line_id.is_unique,
        f"{so.sales_order_id.duplicated().sum()} فاتورة مكررة، {it.sales_line_id.duplicated().sum()} سطر مكرر")
    orphan = int((~it.sales_order_id.isin(so.sales_order_id)).sum() + (~it.branch_id.isin(br.branch_id)).sum()
                 + (~it.product_id.isin(pr.product_id)).sum())
    chk("كل سطور البيع مربوطة بفاتورة وفرع ومنتج موجودين", orphan == 0, f"{orphan} سطر مربوط بحاجة مش موجودة")
    agg = it.groupby("sales_order_id").net_line_EGP.sum()
    diff = (so.set_index("sales_order_id").net_before_tax_EGP - agg).abs().max()
    chk("إجمالي كل فاتورة = مجموع سطورها", diff < 1, f"أكبر فرق: {diff:,.2f} ج")
    bad_m = int(((it.net_line_EGP - it.quantity * it.unit_cost_EGP - it.margin_EGP).abs() > 0.5).sum())
    chk("الربح = سعر البيع - التكلفة", bad_m == 0, f"{bad_m} سطر الربح فيه مش مظبوط")
    neg = int(((it.quantity <= 0) | (it.net_line_EGP <= 0)).sum())
    chk("مفيش كميات أو مبالغ صفر/سالبة", neg == 0, f"{neg} سطر فيه قيمة غير منطقية")
    miss = so.customer_id.isna().mean()
    chk("كل المبيعات مربوطة بعميل", miss < 0.02, f"{miss:.0%} من الفواتير من غير حساب عميل (بنحلل العملاء من الباقي)", warn=True)

    sales = pd.DataFrame({
        "date": pd.to_datetime(it.order_date), "order": it.sales_order_id.factorize()[0].astype("int32"),
        "branch": it.branch_id.astype("category"), "product": it.product_id.astype("category"),
        "qty": it.quantity.astype("int32"), "gross": it.gross_line_EGP.astype("float64"),
        "disc": it.discount_line_EGP.astype("float64"), "net": it.net_line_EGP.astype("float64"),
        "cost": (it.quantity * it.unit_cost_EGP).astype("float64"), "margin": it.margin_EGP.astype("float64"),
        "line": it.sales_line_id})
    omap = so.set_index("sales_order_id")
    sales["channel"] = it.sales_order_id.map(omap.sales_channel).astype("category")
    sales["customer"] = it.sales_order_id.map(omap.customer_id).astype("category")
    sales["category"] = it.product_id.map(pr.set_index("product_id").category).astype("category")
    sales = sales.sort_values("date", kind="stable").reset_index(drop=True)
    last, first = sales.date.max(), sales.date.min()
    days = pd.date_range(first.normalize(), last.normalize())
    gaps = len(days.difference(sales.date.dt.normalize().unique()))
    chk("مبيعات كل يوم في الفترة متسجلة", gaps == 0, f"{gaps} يوم من غير أي مبيعات", warn=True)
    last_month_full = last.normalize() == (last + pd.offsets.MonthEnd(0)).normalize()
    chk("آخر شهر في الداتا مكتمل", last_month_full,
        f"آخر يوم في الداتا {last:%Y-%m-%d}، فشهر {last:%Y-%m} ناقص ومش بيدخل في التوقعات", warn=True)

    # ---- الفروع (اسم واضح غير مكرر)
    br = br.rename(columns={"branch_name_ar": "name", "governorate_ar": "governorate", "city_ar": "city"})
    br["opening_date"] = pd.to_datetime(br.opening_date, errors="coerce", format="mixed")
    dup = br.name.duplicated(keep=False)
    br["label"] = np.where(dup, br.name + " (" + br.governorate + ")", br.name)
    first_sale = sales.groupby("branch", observed=True).date.min()
    early = int((first_sale.reindex(br.branch_id).values < br.opening_date.values - np.timedelta64(30, "D")).sum())
    chk("مبيعات الفروع بتبدأ بعد تاريخ الافتتاح", early == 0, f"{early} فرع ليه مبيعات قبل تاريخ افتتاحه بأكتر من 30 يوم", warn=True)

    # ---- المرتجعات
    returns = pd.DataFrame()
    if has("sales_returns.csv"):
        r = rd("sales_returns.csv").merge(sales[["line", "branch", "category", "net", "qty", "channel"]], left_on="sales_line_id",
                                          right_on="line", how="left")
        too_much = int((r.refund_amount_EGP > r.net * (1 + VAT) + 1).sum())
        chk("المبلغ المسترد مش أكبر من سعر السطر (بالضريبة)", too_much == 0, f"{too_much} مرتجع مبلغه أكبر من اللي اتباع", warn=True)
        chk("الكمية المرتجعة مش أكبر من المباعة", int((r.return_quantity > r.qty).sum()) == 0, "")
        returns = r.rename(columns={"return_date": "date", "return_quantity": "qty_ret", "refund_amount_EGP": "refund"})
        returns["date"] = pd.to_datetime(returns.date)
        returns = returns[["date", "branch", "category", "channel", "qty_ret", "refund", "reason"]]
        for c in ("branch", "category", "channel", "reason"):
            returns[c] = returns[c].astype("category")
    sales = sales.drop(columns=["line"])

    # ---- العملاء
    cust = pd.DataFrame()
    if has("customers.csv"):
        cust = rd("customers.csv")
        cust["joined_date"] = pd.to_datetime(cust.joined_date, errors="coerce", format="mixed")
        unk = int((~sales.customer.dropna().isin(cust.customer_id)).sum())
        chk("كل عملاء الفواتير موجودين في ملف العملاء", unk == 0, f"{unk} سطر لعميل مش موجود", warn=True)

    # ---- المخزون
    stock = pd.DataFrame()
    if has("monthly_stock.csv"):
        stock = rd("monthly_stock.csv")
        stock = stock.rename(columns={"stock_location_id": "location", "stock_location_type": "kind", "product_category": "category"})
        sm = stock[(stock.year_month == last.strftime("%Y-%m")) | (stock.year_month == (last - pd.offsets.MonthBegin(1)).strftime("%Y-%m"))]
        sm = sm[sm.kind == "Branch"].groupby(["location", "category"]).agg(stock=("closing_units", "mean"), risk=("stockout_risk_score", "mean")).reset_index()
        rate = sales[sales.date >= last - pd.Timedelta(days=90)].groupby(["branch", "category"], observed=True).qty.sum().div(3).rename("m").reset_index()
        j = sm.merge(rate, left_on=["location", "category"], right_on=["branch", "category"], how="left").fillna({"m": 0})
        c1, c2 = j[["stock", "m"]].corr().iloc[0, 1], j[["risk", "m"]].corr().iloc[0, 1]
        chk("المخزون مرتبط بمعدل البيع", abs(c1) > 0.2, f"الارتباط بين مخزون الفرع وبيعه {c1:.2f} (قريب من صفر). اعتمدنا على حساب أيام التغطية بدل الاعتماد على المخزون لوحده", warn=True)
        chk("درجة خطر النفاد المسجلة منطقية", abs(c2) > 0.2, f"الارتباط بين الدرجة المسجلة والبيع {c2:.2f}، فبنحسب الخطر بنفسنا من التغطية والتوقعات", warn=True)
        tot = stock[(stock.year_month == last.strftime("%Y-%m")) & (stock.kind == "Warehouse")].closing_units.sum()
        sold = sales[sales.date >= last - pd.Timedelta(days=90)].qty.sum() / 3
        chk("مخزون المخازن متناسب مع المبيعات", tot / max(sold, 1) < 12, f"مخزون المخازن يكفي حوالي {tot / max(sold, 1):.0f} شهر من المبيعات الحالية", warn=True)
        stock["category"] = stock.category.astype("category")

    # ---- تكاليف الفروع (مرتبات + مصروفات)
    costs = pd.DataFrame()
    if has("employees.csv") and has("payroll.csv") and has("operating_expenses.csv"):
        emp, pay, ox = rd("employees.csv"), rd("payroll.csv"), rd("operating_expenses.csv")
        p = pay.merge(emp[["employee_id", "branch_id"]], on="employee_id").groupby(["pay_month", "branch_id"]).net_paid_EGP.sum().rename("payroll")
        ox["month"] = ox.expense_date.str[:7]
        ox = ox[ox.branch_or_department_id.str.startswith("BR")]
        o = ox.assign(rent=np.where(ox.expense_category == "Rent", ox.amount_EGP, 0), util=np.where(ox.expense_category == "Utilities", ox.amount_EGP, 0)) \
            .groupby(["month", "branch_or_department_id"]).agg(opex=("amount_EGP", "sum"), rent=("rent", "sum"), util=("util", "sum"))
        p.index.names, o.index.names = ["month", "branch"], ["month", "branch"]
        costs = pd.concat([p, o], axis=1).fillna(0).reset_index()
        nopay = sorted(set(br.branch_id) - set(emp.branch_id.dropna()))
        chk("كل فرع له موظفين ومرتبات مسجلة", not nopay, f"{len(nopay)} فرع من {len(br)} مفيش ليه مرتبات في الداتا، فصافي ربحه مش دقيق (مش بنرتبه مع الباقي)", warn=True)

    # ---- التوصيل
    deliv = pd.DataFrame()
    if has("delivery_shipments.csv"):
        deliv = rd("delivery_shipments.csv")
        deliv["days"] = (pd.to_datetime(deliv.delivery_date) - pd.to_datetime(deliv.ship_date)).dt.days
        deliv["late"] = deliv.fulfillment_status.eq("Delayed")
        deliv["ship_date"] = pd.to_datetime(deliv.ship_date)
        deliv = deliv[["ship_date", "carrier", "ship_to_governorate", "shipping_fee_EGP", "days", "late"]]

    # ---- فحوصات تشغيلية إضافية
    if has("marketing_campaigns.csv"):
        mk = rd("marketing_campaigns.csv")
        chk("الطلبات المنسوبة للحملات منطقية", mk.orders_attributed.sum() <= len(so) * 1.2,
            f"الحملات بتنسب {mk.orders_attributed.sum():,.0f} طلب، بينما إجمالي الفواتير الفعلية {len(so):,.0f}. فمش بنعتمد على أرقام العائد التسويقي", warn=True)
    if has("purchase_order_items.csv"):
        po = rd("purchase_order_items.csv")
        yr = po.groupby(po.delivery_date.str[:4]).quantity.sum()
        ys = sales.groupby(sales.date.dt.year).qty.sum()
        yr.index, ys.index = yr.index.astype(str), ys.index.astype(str)
        both = (yr / ys.rename(index=str)).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = both.median() if len(both) else yr.sum() / max(float(ys.sum()), 1.0)
        chk("كميات الشراء قريبة من كميات البيع", ratio < 3, f"المشتريات المسجلة حوالي {ratio:.0f} ضعف المبيعات (غالبًا وحدات مختلفة زي متر قماش). مش بنقارنهم", warn=True)

    # ---- حفظ
    log("حفظ الملفات...")
    sales.to_parquet(PROC / "sales.parquet", index=False)
    br.to_parquet(PROC / "branches.parquet", index=False)
    pr[["product_id", "product_name_ar", "category", "category_ar", "audience", "listed_price_EGP"]].to_parquet(PROC / "products.parquet", index=False)
    for name, df in (("customers", cust), ("stock", stock), ("returns", returns), ("costs", costs), ("deliveries", deliv)):
        if len(df):
            df.to_parquet(PROC / f"{name}.parquet", index=False)
    meta = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "fingerprint": fingerprint(raw_dir), "rows": len(sales),
            "first_date": str(first.date()), "last_date": str(last.date()), "raw_dir": str(raw_dir)}
    (PROC / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=1), encoding="utf-8")
    (PROC / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if with_models:
        from . import models
        models.run_all(load(), log=log)
    meta["seconds"] = round(time.time() - t0)
    log(f"تم في {meta['seconds']} ثانية")
    return meta


# ---------------------------------------------------------------- التحميل
def _rd(name):
    p = PROC / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load() -> SimpleNamespace:
    if not (PROC / "sales.parquet").exists():
        raise FileNotFoundError("البيانات لسه متجهزتش")
    D = SimpleNamespace(**{n: _rd(n) for n in ("sales", "branches", "products", "customers", "stock", "returns", "costs", "deliveries",
                                               "forecast", "backtest", "churn", "fquality")})
    D.fq = D.fquality
    D.meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    D.health = json.loads((PROC / "health.json").read_text(encoding="utf-8"))
    mm = PROC / "model_metrics.json"
    D.model_metrics = json.loads(mm.read_text(encoding="utf-8")) if mm.exists() else {}
    D.first, D.last = D.sales.date.min(), D.sales.date.max()
    D.branch_name = D.branches.set_index("branch_id").label.to_dict()
    D.branch_gov = D.branches.set_index("branch_id").governorate.to_dict()
    D.cat_ar = D.products.drop_duplicates("category").set_index("category").category_ar.to_dict()
    D.prod_name = D.products.set_index("product_id").product_name_ar.to_dict()
    D.prod_cat = D.products.set_index("product_id").category.to_dict()
    return D


def apply_filters(sales: pd.DataFrame, start=None, end=None, branches=None, categories=None, products=None, channels=None) -> pd.DataFrame:
    lo = sales.date.searchsorted(pd.Timestamp(start)) if start is not None else 0
    hi = sales.date.searchsorted(pd.Timestamp(end) + pd.Timedelta(days=1)) if end is not None else len(sales)
    df = sales.iloc[lo:hi]
    for col, vals in (("branch", branches), ("category", categories), ("product", products), ("channel", channels)):
        if vals:
            df = df[df[col].isin(vals)]
    return df


if __name__ == "__main__":
    build()
