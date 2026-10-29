"""محرك التوصيات: بيدور في الداتا على مشاكل وفرص، ويطلّع لكل واحدة الدليل والإجراء وتقدير الأثر بالجنيه."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import inventory as inv
from . import metrics as M
from .data import VAT, apply_filters

PRIORITY = {"حرجة": 0, "مهمة": 1, "فرصة": 2, "معلومة": 3}


def _names(labels, n=3):
    labels = list(labels)
    return "، ".join(labels[:n]) + (f" و{len(labels) - n} غيرهم" if len(labels) > n else "")


def generate(D) -> pd.DataFrame:
    end = D.last.normalize()
    start = end - pd.Timedelta(days=364)
    cur = apply_filters(D.sales, start, end)
    k = M.kpis(cur)
    out: list[dict] = []

    def add(priority, area, title, evidence, action, impact=np.nan, note="", page="", items=None, big=None, ring=None, label="", kind="تقدير"):
        if big is None and impact == impact:
            big = M.money(impact)
        out.append({"priority": priority, "area": area, "title": title, "evidence": evidence, "action": action, "impact": impact,
                    "impact_note": note, "page": page, "items": items or [], "big": big or "", "ring": ring, "label": label or note, "kind": kind})

    # 1) فروع بتنمو أقل من الشركة
    fl = M.branch_flags(D)
    bad = fl[fl.gap <= -15]
    if len(bad):
        lost = bad.lost.sum()
        add("حرجة" if lost > 0.03 * cur.net.sum() / 4 else "مهمة", "الفروع", f"{len(bad)} فرع نموه أقل بكتير من الشركة",
            f"الشركة {'بتنمو' if fl.network_growth.iloc[0] >= 0 else 'بتنكمش'} {abs(fl.network_growth.iloc[0]):.0f}% (آخر 13 أسبوع مقابل نفس الفترة السنة اللي فاتت)، لكن {_names(bad.label)} أقل منها بأكتر من 15 نقطة. الفرق عن نمو الشركة بيساوي تقريبًا {M.money(lost)} مبيعات إضافية كانت ممكن تتحقق في 13 أسبوع لو نموهم زي الشركة (تقدير، مش خسارة مؤكدة).",
            "راجع الفروع دي: الموظفين، توفر المنتجات (شوف صفحة المخزون)، الخصومات، والمنافسين حواليها.",
            lost * 4, "تقدير لو نفس الفجوة استمرت سنة، مش خسارة مؤكدة", "الفروع",
            [(r.label, r.gap, f"{r.gap:+.0f} نقطة") for r in bad.head(4).itertuples()], label="فرق محتمل في المبيعات سنويًا لو نموهم زي الشركة", kind="تقدير")

    # 2) تسرب الخصومات
    bd = M.by_dim(cur, D, "branch")
    avg_d = cur.disc.sum() / cur.gross.sum() * 100
    hi = bd[bd.disc_pct > avg_d + 2 * bd.disc_pct.std()]
    if len(hi):
        leak = float(((hi.disc_pct - avg_d) / 100 * hi.gross).sum())
        add("مهمة", "التسعير", f"{len(hi)} فرع بيدي خصومات أعلى من الباقي",
            f"متوسط الخصم في الشركة {avg_d:.1f}% بينما {_names(hi.label)} عندهم {hi.disc_pct.max():.1f}% كحد أقصى.",
            "اطلب من مديري الفروع دي تبرير الخصومات، وحدد سقف للخصم من غير موافقة.", leak, "تقدير: الفرق عن متوسط الخصم في آخر 12 شهر", "المبيعات",
            [(r.label, r.disc_pct, f"{r.disc_pct:.1f}%") for r in hi.sort_values("disc_pct", ascending=False).head(4).itertuples()], label="خصومات زيادة عن المتوسط (آخر 12 شهر)", kind="تقدير")
    ch = M.by_dim(cur, D, "channel").set_index("key")
    if "Wholesale / reseller" in ch.index and "Retail POS" in ch.index and ch.disc_pct["Wholesale / reseller"] > ch.disc_pct["Retail POS"] + 1:
        w, r = ch.loc["Wholesale / reseller"], ch.loc["Retail POS"]
        extra = (w.disc_pct - r.disc_pct) / 100 * w.gross
        add("فرصة", "التسعير", "خصم الجملة أعلى من الفروع",
            f"الجملة بتاخد خصم {w.disc_pct:.1f}% مقابل {r.disc_pct:.1f}% في الفروع، وهامش ربحها {w.margin_pct:.1f}% مقابل {r.margin_pct:.1f}%.",
            "حط جدول خصم بالكمية للجملة، وراجع الصفقات الكبيرة قبل الموافقة.", extra, "لو خصم الجملة اتساوى بالفروع", "المبيعات",
            [("الجملة", w.disc_pct, f"{w.disc_pct:.1f}%"), ("الفروع", r.disc_pct, f"{r.disc_pct:.1f}%")], label="خصم الجملة الزيادة عن الفروع", kind="تقدير")

    # 3) منتجات هامشها ضعيف
    pd_ = M.by_dim(cur, D, "product")
    low = pd_[(pd_.margin_pct < k["margin_pct"] - 8) & (pd_.net > pd_.net.median())]
    if len(low):
        gain = float((0.5 * (k["margin_pct"] - low.margin_pct) / 100 * low.net).sum())
        add("مهمة", "الربحية", f"{len(low)} منتج هامش ربحه أضعف من المتوسط",
            f"متوسط هامش الربح {k['margin_pct']:.1f}%، لكن منتجات زي {_names(low.sort_values('margin_pct').label, 2)} هامشها أقل من {low.margin_pct.max():.0f}% وهي منتجات بتتباع بكميات كبيرة.",
            "ارفع السعر شوية أو فاوض المورد على التكلفة، أو قلل الخصم على المنتجات دي.", gain, "لو قفلنا نص الفجوة مع المتوسط", "المبيعات",
            [(r.label, r.margin_pct, f"{r.margin_pct:.0f}%") for r in low.sort_values("margin_pct").head(4).itertuples()], label="ربح إضافي محتمل لو الهامش اتحسن", kind="تقدير")

    # 4+5) المخزون
    df = inv.intel(D)
    s = inv.summary(df)
    cp = inv.category_plan(D, df)
    if s["critical"] + s["low"]:
        need_tot = float(df.need.sum())
        tr = inv.transfers(D, df, max_rows=100000)
        cover_pct = float(tr.units.sum() / need_tot * 100) if need_tot else 100.0
        by_cat = df[df.exposure > 0].groupby("category_ar").exposure.sum().sort_values(ascending=False).head(4)
        add("حرجة", "المخزون", f"{s['critical'] + s['low']} حالة (فرع × نوع) المخزون فيها قرّب يخلص",
            f"{s['critical']} حالة حرجة (التغطية أقل من 3 أسابيع) و{s['low']} منخفضة. المبيعات المعرضة للضياع في فترة التوريد حوالي {M.money(s['exposure'])}.",
            f"نفّذ التحويلات المقترحة من صفحة المخزون (من فروع الزيادة ومخازن المنطقة)، وده بيغطي حوالي {cover_pct:.0f}% من النقص من غير شراء جديد.", s["exposure"], "تقدير للمبيعات اللي ممكن تضيع في فترة التوريد لو البضاعة متحولتش", "المخزون",
            [(c, v, M.money(v)) for c, v in by_cat.items()], label="مبيعات ممكن تضيع لو البضاعة متحولتش (تقدير)", kind="تقدير")
    stop = cp[cp.action == "وقف الشراء تمامًا"]
    if s["tied"] > 0:
        add("مهمة", "المخزون", "فلوس متجمدة في مخزون زيادة",
            f"في {s['over']} حالة (فرع × نوع) المخزون يكفي أكتر من 26 أسبوع. تكلفة الزيادة في الفروع حوالي {M.money(s['tied'])}، والمخازن مليانة كمان" +
            (f" ({len(stop)} نوع مخزونه الكلي يكفي أكتر من سنة، منهم {_names(stop.category_ar, 3)})." if len(stop) else "."),
            "وقف الشراء للأنواع اللي مخزونها يكفي سنة، وسوّق الزيادة بعروض مدروسة أو حوّلها للفروع الأسرع في البيع.", s["tied"], "تقدير لقيمة المخزون الزيادة عن الحد (فلوس ممكن تتحرر)", "المخزون",
            [(r.category_ar, min(r.cover_weeks, 400), (f"{r.cover_weeks / 52:.0f} سنة" if r.cover_weeks > 104 else f"{r.cover_weeks:.0f} أسبوع")) for r in cp.sort_values("cover_weeks", ascending=False).head(4).itertuples()], label="تكلفة المخزون الزيادة عن الحد (فلوس متجمدة)", kind="تقدير")

    # 6) التوقعات: الموسم الجاي
    fc = D.forecast[(D.forecast.level == "total") & (D.forecast.target == "qty")].sort_values("month")
    if len(fc) >= 3:
        from .models import monthly
        base = monthly(D.sales, "qty").iloc[-3:].mean()
        nxt = fc.forecast.iloc[:3].mean()
        pk = fc.loc[fc.forecast.idxmax()]
        gf = inv.growth_factor(D).sort_values(ascending=False)
        add("معلومة", "التوقعات", f"الطلب المتوقع في الـ3 شهور الجاية {((nxt / base) - 1) * 100:+.0f}% عن آخر 3 شهور",
            f"أعلى شهر متوقع {pk.month} بحوالي {pk.forecast:,.0f} قطعة. أكتر أنواع هيزيد عليها الطلب: {_names(gf.index.map(D.cat_ar)[:3])}.",
            "جهّز المخزون في الفروع قبل الموسم، وخطط للعروض والموظفين.", np.nan, "", "التوقعات",
            [(D.cat_ar.get(c, c), (v - 1) * 100, f"{(v - 1) * 100:+.0f}%") for c, v in gf.head(4).items()], big=f"{(nxt / base - 1) * 100:+.0f}%", label="الطلب المتوقع في الـ3 شهور الجاية", kind="توقع")

    # 7) العملاء
    unk = cur[cur.customer.isna()].net.sum() / cur.net.sum() * 100
    if unk > 5:
        add("فرصة", "العملاء", f"{unk:.0f}% من المبيعات من غير بيانات عميل",
            f"{M.money(cur[cur.customer.isna()].net.sum())} اتباعت لعملاء مش متسجلين، يعني مش قادرين نتواصل معاهم أو نعرف بيرجعوا ولا لأ.",
            "خلي الكاشير يطلب رقم التليفون (مثلًا مقابل نقاط أو خصم صغير)، وبكده نقدر نحلل ونكلم العملاء.", np.nan, "", "العملاء", big=f"{unk:.0f}%", ring=unk, label="مبيعات من غير حساب عميل", kind="حقيقة")
    cm = D.model_metrics.get("churn", {})
    if cm and not cm.get("predictive", True):
        add("معلومة", "العملاء", "غياب العميل مش بيقول إنه هيبطل يشتري",
            f"اختبرنا نموذج يتوقع مين هيرجع يشتري خلال 4 شهور، ودقته ({cm['auc']}) مجرد تخمين. نسبة اللي بيرجعوا حوالي {cm['base_rate'] * 100:.0f}% سواء العميل لسه مشتري أو غايب من سنة.",
            "متصرفش ميزانية كبيرة على حملات استرجاع من غير اختبار: ابعت العرض لمجموعة وسيب مجموعة بدون، وقارن النتيجة.", np.nan, "", "العملاء",
            big=f"{cm['base_rate'] * 100:.0f}%", ring=cm["base_rate"] * 100, label="نسبة الرجوع خلال 4 شهور لأي عميل", kind="حقيقة")

    # 8) المرتجعات
    if len(D.returns):
        rr = D.returns[(D.returns.date >= start) & (D.returns.date <= end)].groupby("branch", observed=True).refund.sum()
        rate = (rr / (bd.set_index("key").net * (1 + VAT))).dropna() * 100
        z = (rate - rate.mean()) / rate.std()
        hi = rate[z >= 2]
        if len(hi):
            excess = float(((hi - rate.mean()) / 100 * bd.set_index("key").net.reindex(hi.index) * (1 + VAT)).sum())
            add("مهمة", "الجودة", f"{len(hi)} فرع مرتجعاته أعلى من الباقي",
                f"متوسط المرتجعات {rate.mean():.1f}% من المبيعات، لكن {_names(hi.index.map(D.branch_name))} وصلوا {hi.max():.1f}%.",
                "راجع أسباب المرتجعات في الفروع دي: خطأ في المقاس، عيوب، أو سياسة استرجاع متساهلة.", excess, "المبالغ الزيادة عن المتوسط", "الفروع",
                [(D.branch_name.get(b, b), v, f"{v:.1f}%") for b, v in hi.sort_values(ascending=False).head(4).items()], label="مرتجعات زيادة عن المتوسط", kind="حقيقة")

    # 9) جودة البيانات
    warn = [h for h in D.health if h["status"] != "ok"]
    if warn:
        add("معلومة", "البيانات", f"{len(warn)} ملاحظة على جودة البيانات",
            "؛ ".join(h["detail"] for h in warn[:3]) + ("..." if len(warn) > 3 else ""),
            "راجع صفحة «صحة البيانات» قبل ما تعتمد على أرقام المخزون أو التسويق في قرارات كبيرة.", np.nan, "", "صحة البيانات", big=str(len(warn)), label="ملاحظة على الداتا", kind="حقيقة")

    r = pd.DataFrame(out)
    r["rank"] = r.priority.map(PRIORITY)
    return r.sort_values(["rank", "impact"], ascending=[True, False]).drop(columns="rank").reset_index(drop=True)
