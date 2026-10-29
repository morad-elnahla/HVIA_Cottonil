"""المساعد الذكي.

طبقتين:
1) محرك محلي بيفهم السؤال العربي (فرع/نوع/فترة/مقارنة/ليه/توقع/مخزون...) ويحسب الإجابة من الداتا الفعلية.
2) لو فيه مفتاح LLM (Groq أو أي خدمة بنفس شكل OpenAI) بيستخدم نفس الحسابات كأدوات، والنموذج بيفهم ويشرح بحرية أكبر.
الأرقام دايمًا من الأدوات، مش من النموذج.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import insights as ins
from . import inventory as inv
from . import metrics as M
from .data import CH_AR, apply_filters
from .llm import GroqClient, LLMError, qid
from .models import MODEL_AR, monthly, mtd_nowcast

MINT, RED = "#5eeab0", "#ff6b6b"
LLM_BUSY = "خدمة الذكاء الاصطناعي (Groq) مشغولة مؤقتًا"
METRIC_AR = {"net": "المبيعات", "margin": "الربح", "margin_pct": "هامش الربح", "qty": "القطع المباعة", "orders": "عدد الفواتير",
             "basket": "متوسط الفاتورة", "disc_pct": "نسبة الخصم", "customers": "عدد العملاء", "refund": "المرتجعات"}
DIM_AR = {"branch": "الفرع", "governorate": "المحافظة", "category": "النوع", "product": "المنتج", "channel": "القناة",
          "month": "الشهر", "week": "الأسبوع", "day": "اليوم", "year": "السنة"}
DIM_PL = {"branch": "فروع", "governorate": "محافظات", "category": "أنواع", "product": "منتجات", "channel": "قنوات"}
MONTHS = {"يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "أبريل": 4, "مايو": 5, "يونيو": 6, "يونيه": 6, "يوليو": 7, "يوليه": 7,
          "اغسطس": 8, "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}
GENDER = {"Men": ["رجالي", "رجال", "رجاليه"], "Women": ["حريمي", "حريم", "نسائي", "نسائيه", "ستاتي"], "Girls": ["بناتي", "بنات", "بنوتات"],
          "Boys": ["اولاد", "ولادي", "ولاد"], "Kids": ["اطفال", "طفل"]}
TYPES = {"boxers": (["بوكسر", "بوكسرات"], ["Men boxers"]), "socks": (["شراب", "شرابات", "جوارب", "شرابات"], ["Socks"]),
         "bras": (["حمالات", "حماله", "برا", "صدريه"], ["Women bras"]), "home": (["مفروشات", "مفروش"], ["Home textiles"]),
         "acc": (["اكسسوار", "اكسسوارات"], ["Accessories"]),
         "pj": (["بيجاما", "بيجامات"], ["Boys pajamas", "Girls pajamas", "Women pajamas"]),
         "under": (["داخلي", "ملابس داخليه", "لانجري"], ["Men underwear", "Women underwear", "Kids underwear", "Girls underwear"]),
         "tops": (["تيشيرت", "تيشيرتات", "تيشرت", "بلوزه", "بلوزات", "قمصان"], ["Men tops", "Women tops"]),
         "bottoms": (["بنطلون", "بناطيل", "شورت", "شورتات"], ["Men bottoms", "Women bottoms"])}
CHANNELS = {"E-commerce": ["اونلاين", "اون لاين", "الموقع", "ايكومرس", "تجاره الكترونيه", "الكتروني"], "Wholesale / reseller": ["جمله", "تجار", "موزعين"],
            "Retail POS": ["كاشير", "محلات"]}


STOP = {"مبيعات", "مخزون", "فرع", "فروع", "منتج", "نوع", "ربح", "عميل", "عملاء", "جديد", "جديده", "مول", "سيتي", "سنتر", "ميدان", "شارع", "مركز", "داون", "تاون"}


# ---------------------------------------------------------------- فهم النص العربي
def norm(s: str) -> str:
    s = str(s).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    s = re.sub("[\u064B-\u0652\u0640]", "", s).translate(str.maketrans("أإآىة", "ااايه")).lower()
    return re.sub(r"[^\w\s%]", " ", s)


def stem(t: str) -> str:
    return re.sub(r"^(وال|بال|لل|ال)", "", t) if len(t) > 4 else t


def toks(s: str) -> list[str]:
    return [stem(t) for t in norm(s).split()]


def kw(tokens, *words) -> bool:
    ws = [stem(norm(w)) for w in words]
    tk = set(tokens) | {t[1:] for t in tokens if len(t) > 3 and t[0] in "وبلفك"}      # «وهامش» «بالربح» «لقطع»
    return any(t.startswith(w) if len(w) >= 3 else t == w for t in tk for w in ws)


def phrase(text: str, *ps) -> bool:
    return any(norm(p) in text for p in ps)


@dataclass
class Ctx:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class Answer:
    """رد المساعد: ملخص + حقائق من الداتا + توقعات + استنتاجات محتملة + اقتراحات + مصدر. (text = كل ده كنص واحد)"""
    text: str = ""
    table: pd.DataFrame | None = None
    fig: go.Figure | None = None
    mode: str = "المحرك المحلي"
    headline: str = ""
    facts: list = field(default_factory=list)
    forecasts: list = field(default_factory=list)
    inferences: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    source: dict = field(default_factory=dict)
    parts: list = field(default_factory=list)          # لو السؤال فيه أكتر من جزء
    title: str = ""
    summary: str = ""                                   # ملخص صاغه النموذج اللغوي (لو اتفعّل واتحقق)
    engine_detail: str = ""
    llm_note: str = ""                                  # رسالة مفهومة لو الذكاء الاصطناعي ماردش

    @classmethod
    def make(cls, headline, facts=(), forecasts=(), inferences=(), source=None, table=None, fig=None, notes=(), actions=(), mode="المحرك المحلي"):
        a = cls(headline=headline, facts=list(facts), forecasts=list(forecasts), inferences=list(inferences), actions=list(actions), notes=list(notes),
                source=source or {}, table=table, fig=fig, mode=mode)
        a.text = a._compose()
        return a

    def _compose(self) -> str:
        parts = [self.headline]
        for title, items in (("حقائق من الداتا", self.facts), ("توقعات", self.forecasts), ("استنتاجات محتملة (مش مثبتة)", self.inferences), ("اقتراحات مبنية على الأرقام", self.actions), ("ملحوظات", self.notes)):
            if items:
                parts.append(title + ":\n" + "\n".join("- " + x for x in items))
        return "\n".join(parts)

    @classmethod
    def combine(cls, answers: list["Answer"], titles: list[str]) -> "Answer":
        if len(answers) == 1:
            return answers[0]
        for a, t in zip(answers, titles):
            a.title = t
        c = cls(headline=f"رديت على {len(answers)} أجزاء من سؤالك:", parts=answers, mode=answers[0].mode)
        c.text = c.headline + "\n" + "\n\n".join(f"[{a.title}]\n{a.text}" for a in answers)
        return c


class Assistant:
    def __init__(self, D, ctx: Ctx | None = None, client: GroqClient | None = None):
        self.D = D
        self.ctx = ctx or Ctx(D.last - pd.Timedelta(days=364), D.last)
        self.client = client or GroqClient()
        br = D.branches
        self._br = [(r.branch_id, set(toks(r["name"]))) for _, r in br.iterrows()]
        self._gov = [(g, set(toks(g))) for g in br.governorate.unique()]
        self._inv = None
        self._ins = None
        self.state: dict = {}
        self.turns: list[dict] = []

    def status(self) -> dict:
        """حالة الذكاء الاصطناعي للواجهة (من غير أي تفاصيل إعدادات)."""
        c = self.client
        return {"configured": c.configured, "available": c.available, "cooldown": int(c.cooldown_left())}

    @property
    def inv_df(self):
        if self._inv is None:
            self._inv = inv.intel(self.D)
        return self._inv

    @property
    def insights(self):
        if self._ins is None:
            self._ins = ins.generate(self.D)
        return self._ins

    # ---------- استخراج الكيانات
    def entities(self, text: str) -> dict:
        t = toks(text)
        ts = set(t)
        used: set[str] = set()
        brs = []
        for bid, bt in sorted(self._br, key=lambda x: -sum(len(w) for w in x[1])):
            if bt and bt <= ts and sum(len(w) for w in bt) >= 4 and not bt <= used:
                brs.append(bid)
                used |= bt
        govs = [g for g, gt in self._gov if gt <= ts and not gt <= used and (len(gt) > 1 or not (gt & used))]
        if not brs:    # مطابقة جزئية: كلمة مميزة من اسم الفرع (مثلًا «الرحاب» لفرع «الرحاب مول»)
            gov_toks = set().union(*[g for _, g in self._gov])
            for w in ts - used - gov_toks:
                hit = [b for b, bt in self._br if w in bt and len(w) >= 4]
                if 0 < len(hit) <= 2 and w not in STOP:
                    brs += [b for b in hit if b not in brs]
        cats: list[str] = []
        for key, (words, cs) in TYPES.items():
            if kw(t, *words):
                sel = list(cs)
                gs = [g for g, ws in GENDER.items() if kw(t, *ws)]
                if gs:
                    narrowed = [c for c in sel if any(c.startswith(g) for g in gs)]
                    sel = narrowed or sel
                cats += sel
        if not cats:
            gs = [g for g, ws in GENDER.items() if kw(t, *ws)]
            if gs:
                cats = [c for c in self.D.cat_ar if any(c.startswith(g) for g in gs)]
        chans = [c for c, ws in CHANNELS.items() if any(norm(w) in norm(text) for w in ws)]
        return {"branches": brs, "governorates": govs, "categories": list(dict.fromkeys(cats)), "channels": chans}

    def parse_period(self, text: str):
        n, end = norm(text), self.D.last.normalize()
        t = toks(text)
        m = re.search(r"(?:اخر|اخري|ال\s*)\s*(\d+|اسبوعين|شهرين|سنتين)?\s*(يوم|ايام|اسبوع|اسابيع|شهر|شهور|اشهر|سنه|سنين|سنوات)", n)
        if m and phrase(n, "اخر"):
            k = m.group(1)
            k = 2 if k in ("اسبوعين", "شهرين", "سنتين") else (int(k) if k else 1)
            u = m.group(2)
            days = k * (1 if u.startswith(("يوم", "ايام")) else 7 if u.startswith("اسبو") or u.startswith("اسابيع") else 30 if u.startswith(("شهر", "شهور", "اشهر")) else 365)
            return end - pd.Timedelta(days=days - 1), end, f"آخر {days} يوم"
        if phrase(n, "امبارح", "امس", "اليوم", "النهارده", "النهاردة"):
            d = end if phrase(n, "اليوم", "النهارده") else end - pd.Timedelta(days=1)
            return d, d, f"{d:%Y-%m-%d}"
        if phrase(n, "الشهر اللي فات", "الشهر الماضي", "الشهر السابق", "الشهر الي فات", "الشهر اللى فات"):
            p = (end.to_period("M") - 1)
            return p.start_time, p.end_time.normalize(), f"شهر {p}"
        if phrase(n, "الشهر ده", "الشهر دا", "هذا الشهر", "الشهر الحالي", "الشهر دة"):
            return end.replace(day=1), end, f"{end:%Y-%m} حتى {end:%d}"
        if phrase(n, "الاسبوع ده", "هذا الاسبوع", "الاسبوع الحالي", "الاسبوع دا"):
            return end - pd.Timedelta(days=6), end, "آخر 7 أيام"
        if phrase(n, "السنه اللي فاتت", "السنه الماضيه", "العام الماضي", "السنه الي فاتت", "العام اللي فات"):
            y = end.year - 1
            return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), f"سنة {y}"
        if phrase(n, "السنه دي", "السنه دي", "هذه السنه", "العام ده", "السنه الحاليه", "العام الحالي", "السنه دى"):
            return pd.Timestamp(end.year, 1, 1), end, f"سنة {end.year} حتى الآن"
        if phrase(n, "الربع الاخير"):
            return end - pd.Timedelta(days=89), end, "آخر 3 شهور"
        year = re.search(r"\b(20\d\d)\b", n)
        for name, mo in MONTHS.items():
            if norm(name) in t or any(x.startswith(norm(name)) for x in t):
                y = int(year.group(1)) if year else (end.year if mo <= end.month else end.year - 1)
                p = pd.Period(f"{y}-{mo:02d}", "M")
                return p.start_time, min(p.end_time.normalize(), end), f"{name} {y}"
        if year:
            y = int(year.group(1))
            return pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 31), end), f"سنة {y}"
        return None


    # ---------- فهم السؤال
    @staticmethod
    def requested_text(text: str) -> str:
        """Remove negated metric mentions before keyword matching or decomposition."""
        return re.sub(
            r"(?:و)?(?:متجيبليش|ما\s*تجيبليش|متعرضش|ما\s*تعرضش|مش\s*عايز|لا\s*اريد|بدون|باستثناء|استبعد)\s+"
            r"(?:ال)?(?:هامش\s+(?:ال)?ربح|نسبه\s+(?:ال)?ربح|ارباح|ربح|مبيعات)",
            "", norm(text))

    def metric_of(self, t, n):
        n = self.requested_text(n)
        t = toks(n)
        if kw(t, "هامش"):
            return "margin_pct"
        if kw(t, "ربح", "ارباح", "مكسب", "بنكسب", "مكاسب"):
            return "margin"
        if phrase(n, "متوسط الفاتوره", "متوسط السله", "متوسط البيع"):
            return "basket"
        if kw(t, "خصم", "خصومات", "تخفيض"):
            return "disc_pct"
        if kw(t, "مرتجع", "مرتجعات", "ارجاع"):
            return "refund"
        if kw(t, "قطع", "قطعه", "كميه", "كميات", "وحدات"):
            return "qty"
        if kw(t, "فواتير", "فاتوره", "طلبات", "اوردر", "عمليات"):
            return "orders"
        if kw(t, "عملاء", "عميل") and phrase(n, "كام", "عدد"):
            return "customers"
        if kw(t, "مبيعات", "ايراد", "ايرادات", "دخل", "بعنا", "اتباع", "بنبيع"):
            return "net"
        return None

    DIM_WORDS = [("month", ("شهري", "شهور", "اتجاه", "تطور")), ("week", ("اسبوعي", "اسبوعيا")), ("day", ("يومي", "يوميا")), ("governorate", ("محافظه", "محافظات", "منطقه", "مناطق")),
                 ("branch", ("فرع", "فروع")), ("product", ("منتج", "منتجات", "صنف", "اصناف")), ("category", ("نوع", "انواع", "فئه", "فئات", "قسم", "اقسام", "تصنيف", "كاتيجوري")), ("channel", ("قناه", "قنوات"))]

    def dim_of(self, t, ent=None):
        j = " ".join(t)
        if phrase(j, "شهر بشهر", "كل شهر"):
            return "month"
        if phrase(j, "كل اسبوع"):
            return "week"
        if phrase(j, "كل يوم"):
            return "day"
        found = [(i, d) for i, w in enumerate(t) for d, ws in self.DIM_WORDS if kw([w], *ws)]
        # «حسب النوع» / «لكل فرع» / «موزعة على القنوات»: البعد اللي بعد الكلمة دي هو التقسيم المطلوب
        for i, w in enumerate(t):
            if w in ("حسب", "لكل", "موزع", "موزعه", "بالنسبه") or (w == "على" and i + 1 < len(t) and t[i + 1] in ("حسب",)):
                nxt = [d for k, d in found if k > i]
                if nxt:
                    return nxt[0]
        dims = [d for _, d in found]
        if ent and ent.get("branches") and dims:       # «فرع جليم»: كلمة «فرع» هنا بتعرّف الفرع، مش بتطلب تقسيم بالفروع
            plural = any(w.startswith("فروع") for w in t)
            if not plural:
                dims = [d for d in dims if d != "branch"]
        if ent and ent.get("governorates") and not kw(t, "محافظات", "مناطق"):
            dims = [d for d in dims if d != "governorate"]
        return dims[0] if dims else None

    def intent_of(self, t, n):
        if kw(t, "خصم", "خصومات", "عرض", "عروض", "تخفيض") and (kw(t, "اقترح", "اقتراح", "انصح", "اعمل", "احط", "نزود", "نعمل", "ينفع", "يفيد") or phrase(n, "هل اعمل", "هل نعمل")):
            return "discount"
        if kw(t, "توصي", "اقتراح", "اقترح", "انصح", "نصيح", "فرص", "فرصه", "مشاكل", "مشكله", "حلول", "تحسين") or phrase(n, "اعمل ايه", "اتصرف"):
            return "recommend"
        if kw(t, "ليه", "لماذا", "سبب", "اسباب") or phrase(n, "بسبب ايه", "ايه السبب"):
            return "why"
        if kw(t, "توقع", "متوقع", "هتبقي", "هيبقي", "تنبو", "القادم", "الجاي") or phrase(n, "الشهر الجاي", "الشهور الجايه"):
            return "forecast"
        if kw(t, "مخزون", "يخلص", "هيخلص", "نفاد", "راكد", "ركود", "بضاعه", "تحويل", "تحويلات", "نقص", "تغطيه"):
            return "inventory"
        if kw(t, "ضايعين", "ابطال", "اوفياء", "شريحه", "شرائح", "rfm", "churn", "استرجاع") or phrase(n, "احسن عملاء", "اكبر عملاء", "اهم عملاء", "عملاء ضايعين"):
            return "customers"
        if phrase(n, "صحه البيانات", "جوده البيانات", "دقه البيانات", "مشاكل البيانات"):
            return "health"
        return None

    def parse(self, text: str) -> dict:
        n = self.requested_text(text)
        t = toks(n)
        ent, per = self.entities(n), self.parse_period(n)
        asc = kw(t, "اقل", "اضعف", "اسوا", "اوطي", "ادني", "اصغر")
        rank = asc or kw(t, "اعلي", "اكتر", "اكبر", "احسن", "افضل", "اقوي", "اول", "top")
        k = re.search(r"\b(\d{1,2})\b", n)
        sp = {"text": text, "intent": self.intent_of(t, n), "ent": ent, "per": per, "metric": self.metric_of(t, n), "dim": self.dim_of(t, ent), "rank": rank, "asc": asc,
              "topn": int(k.group(1)) if k and rank and int(k.group(1)) <= 50 else None,
              "yoy": phrase(n, "السنه اللي فاتت", "السنه الماضيه", "العام الماضي", "السنه الي فاتت") and (per is None or phrase(n, "مقارنه", "مقابل", "بالسنه", "عن ")),
              "claim_drop": kw(t, "نزل", "انخفض", "قلت", "هبط", "تراجع"), "claim_rise": kw(t, "زاد", "ارتفع", "طلع"), "inherited": [],
              "comparative": phrase(n, "ليه", "لماذا") and kw(t, "اقل", "اضعف", "اسوا", "ادني", "اعلي", "اكبر", "احسن", "افضل"), "view": None, "months": None, "unresolved": []}
        explicit = phrase(n, "طب", "طيب", "وبالنسبه", "كمان", "برضو", "بردو", "ماذا عن", "نفس الفتره", "نفس الفرع", "نفس اخر", "نفس المحافظه", "نفس الفلاتر") or (bool(t) and t[0].startswith("و") and len(t) <= 7)
        short = len(t) <= 3 and not any(ent.values()) and per is None and sp["metric"] is None and sp["dim"] is None
        sp["follow"] = explicit or short or (bool(self.state) and sp["metric"] is not None and not phrase(n, "كل الشركه", "الشركه كلها", "كل الفروع", "من جديد"))
        return sp

    def merge(self, sp: dict) -> dict:
        """سؤال المتابعة بيورث الفترة والفرع والنوع والمقياس من السؤال اللي قبله (لو مقالش حاجة جديدة)."""
        st = self.state
        if sp["follow"] and st:
            for key in ("branches", "governorates", "categories", "channels"):
                changed_location = ((key == "branches" and sp["ent"]["governorates"] and sp["ent"]["governorates"] != st["ent"]["governorates"])
                                    or (key == "governorates" and sp["ent"]["branches"] and sp["ent"]["branches"] != st["ent"]["branches"]))
                if not changed_location and not sp["ent"][key] and st["ent"][key]:
                    sp["ent"][key] = st["ent"][key]
                    sp["inherited"].append({"branches": "الفرع", "governorates": "المحافظة", "categories": "النوع", "channels": "القناة"}[key])
            if sp["per"] is None and st.get("start") is not None:
                sp["per"] = (st["start"], st["end"], st["plabel"])
                sp["inherited"].append("الفترة")
            if sp["metric"] is None and st.get("metric"):
                sp["metric"] = st["metric"]
                sp["inherited"].append("المقياس")
            if sp["metric"] == st.get("metric") and sp["dim"] is None and not sp["rank"] and st.get("dim") and sp["intent"] in (None, "query"):
                sp["dim"], sp["rank"], sp["asc"], sp["topn"] = st["dim"], st["rank"], st["asc"], st["topn"]
            if sp["intent"] is None and st.get("intent") not in (None, "recommend"):
                # نورّث نوع السؤال (توقع/ليه/مخزون...) إلا لو المتابعة بتطلب مقياس جديد مش مدعوم فيه (زي «والربح؟» بعد توقع)
                if sp["metric"] is None or (sp["metric"] == st.get("metric") and st["intent"] == "why") or (st["intent"] == "forecast" and sp["metric"] in ("net", "qty")):
                    sp["intent"] = st["intent"]
        if sp.get("comparative") and st and not any(len(v) >= 2 for v in sp["ent"].values()):
            for key in ("branches", "governorates", "categories", "channels"):
                if len(st["ent"][key]) >= 2 and (not sp["ent"][key] or sp["ent"][key][0] in st["ent"][key]):
                    sp["ent"][key] = list(st["ent"][key])
                    sp["inherited"].append("المقارنة بين العناصر اللي قبلها")
        per = sp["per"]
        if per:
            sp["start"], sp["end"], sp["plabel"] = per
        else:
            sp["start"], sp["end"] = self.ctx.start, self.ctx.end
            sp["plabel"] = f"{self.ctx.start:%Y-%m-%d} إلى {self.ctx.end:%Y-%m-%d}"
        sp["metric_final"] = sp["metric"] or "net"
        sp["intent_final"] = sp["intent"] or "query"
        return sp

    @staticmethod
    def prev_for(start, end, yoy):
        """الفترة المقارنة. لو الفترة شهر لسه ماخلصش، بنقارنه بنفس عدد أيام الشهر اللي قبله."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if not yoy and start.day == 1:
            month_end = start + pd.offsets.MonthEnd(0)
            if end < month_end:                                   # شهر لسه ماخلصش
                ps = start - pd.offsets.MonthBegin(1)
                return ps, ps + (end - start)
            if end == month_end:                                  # شهر كامل → الشهر اللي قبله
                ps = start - pd.offsets.MonthBegin(1)
                return ps, start - pd.Timedelta(days=1)
            if start.month == 1 and end < pd.Timestamp(end.year, 12, 31):   # السنة حتى الآن → نفس الفترة السنة اللي فاتت
                return M.prev_range(start, end, "yoy")
        return M.prev_range(start, end, "yoy" if yoy else "prev")

    def data(self, start, end, ent: dict) -> pd.DataFrame:
        brs = ent["branches"] or [b for b, g in self.D.branch_gov.items() if g in ent["governorates"]]
        return apply_filters(self.D.sales, start, end, branches=brs or None, categories=ent["categories"] or None, channels=ent["channels"] or None)

    def value(self, df, metric, start=None, end=None, ent=None):
        if metric == "refund":
            e = ent or {"branches": [], "categories": [], "channels": []}
            return M.return_stats(self.D, start, end, e["branches"] or None, e["categories"] or None, e["channels"] or None)["refund"]
        return M.kpis(df)[metric]

    def fmt(self, metric, v):
        return M.pct(v) if metric in ("margin_pct", "disc_pct") else M.num(v) if metric in ("qty", "orders", "customers") else M.money(v)

    def scope_text(self, ent) -> str:
        parts = [self.D.branch_name.get(b, b) for b in ent["branches"]] + ent["governorates"] + [self.D.cat_ar.get(c, c) for c in ent["categories"]] + [CH_AR.get(c, c) for c in ent["channels"]]
        return " - ".join(parts) if parts else "كل الشركة"

    def source(self, sp, what, scope=None, origin="مبيعات الشركة الفعلية (ملف sales_order_items)") -> dict:
        s = {"المقياس": what, "الفترة": sp["plabel"], "النطاق": scope or self.scope_text(sp["ent"]), "المصدر": origin}
        if sp["inherited"]:
            s["كملت على السؤال اللي قبله في"] = "، ".join(sp["inherited"])
        return s

    def _cols(self):
        return {"label": "الاسم", "net": "المبيعات", "margin": "الربح", "margin_pct": "هامش الربح %", "qty": "القطع", "orders": "الفواتير", "basket": "متوسط الفاتورة", "disc_pct": "الخصم %"}

    # ---------- التنفيذ
    def dispatch(self, sp: dict) -> Answer:
        i = sp["intent_final"]
        h = {"recommend": self.h_recommend, "why": self.h_why, "forecast": self.h_forecast, "inventory": self.h_inventory, "customers": self.h_customers,
             "health": self.h_health, "discount": self.h_discount}.get(i, self.h_query)
        return h(sp)

    @staticmethod
    def split_parts(text: str) -> list[str]:
        """بيقسّم السؤال المركب لأجزاء (علامة استفهام، فاصلة، «وكمان»، «وبعدين»، «وقولي»...)."""
        text = re.sub(r"(?:و)?(?:متجيبليش|ما\s*تجيبليش|متعرضش|ما\s*تعرضش|مش\s*عايز|لا\s*أ?ريد|بدون|باستثناء|استبعد)\s+(?:ال)?(?:هامش\s+(?:ال)?ربح|نسب[ةه]\s+(?:ال)?ربح|[أا]رباح|ربح|مبيعات)", "", text)
        pieces = re.split(r"[؟?\n;؛]|\s+(?:وكمان|وبعدين|وبعد كده|وبعد كدة|ثم|وقولي|وقوللي|وعايز|وإيه|وايه)\s+", text)
        pieces = [part for piece in pieces for part in re.split(r"\s+(?=و(?:هامش|الربح|ربح)(?:\s|$))", piece)]
        fillers = {"طيب", "طب", "تمام", "حسنا", "و", "عايز", "عاوز"}
        parts = [p.strip(" ،,.") for p in pieces if p and set(toks(p)) - fillers]
        return (parts or [text])[:4]

    def run_specs(self, specs: list[dict], titles: list[str]) -> Answer:
        answers, kept_titles, seen = [], [], set()
        for index, sp in enumerate(specs):
            sp = self.merge(sp)
            signature = repr((sp["intent_final"], sp["metric_final"], sp["ent"], sp["start"], sp["end"],
                              sp["dim"], sp["rank"], sp["asc"], sp["topn"], sp["yoy"], sp.get("view"),
                              sp.get("months"), sp.get("driver_count"), sp.get("unresolved")))
            if signature in seen:
                continue
            seen.add(signature)
            kept_titles.append(titles[index])
            miss = sp.get("unresolved") or []
            if miss and not any(sp["ent"].values()):
                answers.append(Answer.make("مقدرتش ألاقي في الداتا: " + "، ".join(miss) + ". اكتب الاسم زي ما هو في الفروع أو الأنواع.", notes=["مش هحلل الشركة كلها بدل اللي طلبته."]))
                continue
            a = self.dispatch(sp)
            if sp["inherited"] and "كملت على السؤال اللي قبله في" not in a.source:
                a.source["كملت على السؤال اللي قبله في"] = "، ".join(sp["inherited"])
            if miss:
                a.notes.append("مقدرتش ألاقي: " + "، ".join(miss) + " (اتجاهلته).")
            answers.append(a)
            self.state = sp
        return Answer.combine(answers, kept_titles)

    def comparison_spec(self, text: str):
        """مقارنة زمنية مع مساهمة الفروع: نطاق واحد لكل الأرقام المطلوبة."""
        sp = self.parse(text)
        n = self.requested_text(text)
        t = toks(n)
        # Quantities/discounts mentioned as sales drivers are not a new primary metric.
        if kw(t, "مبيعات", "ايراد", "ايرادات") and not kw(t, "ربح", "ارباح", "هامش"):
            sp["metric"] = "net"
        if (sp["intent"] not in (None, "why") or sp["metric"] not in ("net", "margin")
                or (sp["per"] is None and not (sp["follow"] and self.state)) or not kw(t, "فروع", "فرع")
                or not kw(t, "اثر", "اثرت", "تاثير", "ساهم", "مساهم")
                or not phrase(n, "قبل", "السابق", "مقابل", "مقارنه", "قارن", "اثر", "ساهم")):
            return None
        # Don't fold unrelated requests (profit, stock, forecasts...) into sales.
        if any(self.intent_of(toks(p), norm(p)) not in (None, "why")
               or self.metric_of(toks(p), norm(p)) not in (None, sp["metric"], "qty", "disc_pct")
               for p in self.split_parts(text)):
            return None
        count = re.search(r"(?:اكتر|اعلي|اكثر|اهم|اكبر)\s+(\d{1,2})\s+فروع", n)
        sp.update(intent="why", dim="branch", rank=False,
                  driver_count=min(50, max(1, int(count.group(1)))) if count else 3)
        return sp

    def ask_local(self, text: str) -> Answer:
        comparison = self.comparison_spec(text)
        if comparison is not None:
            return self.run_specs([comparison], [text])
        parts = self.split_parts(text)
        specs = []
        for i, p in enumerate(parts):
            sp = self.parse(p)
            if i > 0:
                sp["follow"] = True
            specs.append(sp)
        return self.run_specs(specs, parts)

    def ask(self, text: str) -> Answer:
        """الترتيب: (1) Groq يفهم السؤال ويقسمه (2) النظام يحسب كل جزء من الداتا (3) Groq يلخّص لو لازم. أي فشل → المحرك المحلي برسالة بسيطة."""
        note, engine, detail, plan = "", "المحرك المحلي", "كل الحسابات من النظام", None
        if self.client.configured:
            if self.client.available:
                try:
                    plan = self.plan(text)
                    engine, detail = "Groq", "فهم السؤال بـ Groq + حسابات النظام"
                except LLMError as e:
                    note = f"{e.user_msg}، فاستخدمنا المحرك المحلي."
            else:
                note = f"{LLM_BUSY}، فاستخدمنا المحرك المحلي."
        comparison = self.comparison_spec(text)
        if comparison is not None and plan is not None and plan.get("calls"):
            # Keep the successful planner; enforce the explicit scope in execution below.
            detail = "فهم السؤال بـ Groq؛ النطاق والمقارنة والأرقام تحقق منها النظام"
        if plan is not None and plan.get("clarify") and not plan["calls"]:
            a = Answer.make(plan["clarify"], mode="Groq", notes=["محتاج توضيح بسيط عشان أجاوب بدقة."])
            a.engine_detail = "Groq"
            self.turns += [{"role": "user", "content": text}, {"role": "assistant", "content": a.headline}]
            return a
        if comparison is not None:
            ans = self.run_specs([comparison], [text])
        elif plan is not None:
            specs, titles = [], []
            for i, c in enumerate(plan["calls"]):
                sp = self.spec_from_call(c, text)
                if i > 0:
                    sp["follow"] = True
                specs.append(sp)
                titles.append(c.get("title") or f"الجزء {i + 1}")
            requested = [self.parse(part) for part in self.split_parts(text)]
            requested_metrics = {part["metric"] for part in requested if part["metric"]}
            if (len(requested_metrics) > 1 or self.requested_text(text) != norm(text)) and all(part["intent"] in (None, "why") for part in requested):
                # Match each explicit metric to its own clause; don't repeat a full-question plan.
                specs, titles = [], self.split_parts(text)
                for i, part in enumerate(requested):
                    part["follow"] = part["follow"] or i > 0
                    specs.append(part)
                detail = "فهم السؤال بـ Groq؛ الأجزاء الصريحة وحساباتها تحقق منها النظام"
            ans = self.run_specs(specs, titles)
        else:
            ans = self.ask_local(text)
        if engine == "Groq" and self.should_narrate(ans):
            try:
                s = self.narrate(ans, text)
                if s:
                    ans.summary, detail = s, "فهم السؤال + ملخص بـ Groq + حسابات النظام"
            except LLMError as e:
                note = f"{e.user_msg}، فعرضنا الأرقام المحسوبة من غير الملخص."
        source_mode = ("Groq لتوجيه السؤال والتلخيص؛ الأرقام من المحرك المحلي" if ans.summary else "Groq لتوجيه السؤال؛ الأرقام من المحرك المحلي") if engine == "Groq" else "المحرك المحلي"
        ans.mode, ans.engine_detail, ans.llm_note = source_mode, detail, note
        self.turns = (self.turns + [{"role": "user", "content": text}, {"role": "assistant", "content": ans.headline[:240]}])[-6:]
        return ans

    # ---------- سؤال عام
    def h_query(self, sp) -> Answer:
        D, ent, metric, dim = self.D, sp["ent"], sp["metric_final"], sp["dim"]
        start, end, plabel, yoy = sp["start"], sp["end"], sp["plabel"], sp["yoy"]
        scope = self.scope_text(ent)
        compare_ents = [k for k in ("branches", "governorates", "categories", "channels") if len(ent[k]) >= 2]
        topn = sp["topn"] or 5
        if compare_ents and (dim is None or dim in ("branch", "governorate", "category", "channel")):
            dim = {"branches": "branch", "governorates": "governorate", "categories": "category", "channels": "channel"}[compare_ents[0]]
            sel = {"branch": ent["branches"], "governorate": ent["governorates"], "category": ent["categories"], "channel": ent["channels"]}[dim]
            tab = M.by_dim(apply_filters(D.sales, start, end), D, dim)
            tab = tab[tab.key.isin(sel)] if dim != "governorate" else tab[tab.label.isin(sel)]
            return self._rank_answer(sp, tab, dim, metric, f"مقارنة {DIM_AR[dim]} ({plabel})", False, len(tab), compare=True)
        df = self.data(start, end, ent)
        if dim in ("month", "week", "day", "year"):
            tab = M.by_dim(df, D, dim).sort_values("key")
            if not len(tab):
                return Answer.make("مفيش مبيعات في الفترة دي.")
            col = "net" if metric not in tab.columns else metric
            fig = px.line(tab, x="label", y=col, markers=True, color_discrete_sequence=[MINT], labels={"label": DIM_AR[dim], col: METRIC_AR.get(metric, "")})
            hi = tab.loc[tab[col].idxmax()]
            head = f"{METRIC_AR[metric]} ({scope}) خلال {plabel}: بدأ بـ {self.fmt(metric, tab[col].iloc[0])} وآخر فترة {self.fmt(metric, tab[col].iloc[-1])}. أعلى {DIM_AR[dim]} كان {hi.label} بـ {self.fmt(metric, hi[col])}."
            return Answer.make(head, source=self.source(sp, METRIC_AR[metric]), table=tab[["label", "net", "margin", "qty", "orders"]].rename(columns=self._cols()), fig=fig)
        if dim and (sp["rank"] or dim != "branch" or not ent["branches"]):
            brs = ent["branches"] or [b for b, g in D.branch_gov.items() if g in ent["governorates"]] or None
            tab = M.by_dim(apply_filters(D.sales, start, end, categories=ent["categories"] or None, channels=ent["channels"] or None, branches=brs), D, dim)
            return self._rank_answer(sp, tab, dim, metric, plabel + (f" - {scope}" if scope != "كل الشركة" else ""), sp["asc"], topn if sp["rank"] else 10)
        if not len(df):
            return Answer.make(f"مفيش مبيعات متسجلة لـ {scope} في {plabel}.")
        ps, pe = self.prev_for(start, end, yoy)
        prev = self.data(ps, pe, ent)
        cur_v, prev_v = self.value(df, metric, start, end, ent), self.value(prev, metric, ps, pe, ent)
        pname = "نفس الفترة السنة اللي فاتت" if yoy else f"الفترة {ps:%Y-%m-%d} إلى {pe:%Y-%m-%d}"
        if metric in ("margin_pct", "disc_pct"):
            cmp_txt = f" (كانت {self.fmt(metric, prev_v)} في {pname}، يعني {cur_v - prev_v:+.1f} نقطة)" if len(prev) else ""
        else:
            g = M.growth(cur_v, prev_v)
            cmp_txt = f" ({g:+.1f}% عن {pname}: {self.fmt(metric, prev_v)})" if g is not None and prev_v else ""
        head = f"{METRIC_AR[metric]} ({scope}) في {plabel}: {self.fmt(metric, cur_v)}{cmp_txt}."
        facts = []
        if metric == "net":
            kp = M.kpis(df)
            facts.append(f"منها ربح {M.money(kp['margin'])} (هامش {kp['margin_pct']:.1f}%) من {M.num(kp['orders'])} فاتورة، ومتوسط الفاتورة {M.money(kp['basket'])}.")
            if scope != "كل الشركة":
                facts.append(f"ده {kp['net'] / apply_filters(D.sales, start, end).net.sum() * 100:.1f}% من مبيعات الشركة في نفس الفترة.")
        return Answer.make(head, facts, source=self.source(sp, METRIC_AR[metric]))

    def _rank_answer(self, sp, tab, dim, metric, title, asc, n, compare=False) -> Answer:
        col = metric if metric in tab.columns and metric != "refund" else "net"
        tab = tab.sort_values(col, ascending=asc).head(n)
        if not len(tab):
            return Answer.make("مفيش بيانات مطابقة.")
        if compare and len(tab) >= 2:
            a, b = tab.iloc[0], tab.iloc[1]
            gap = M.growth(a[col], b[col]) if col not in ("margin_pct", "disc_pct") else a[col] - b[col]
            head = f"{METRIC_AR[col]} - {title}: {a.label} {self.fmt(col, a[col])} مقابل {b.label} {self.fmt(col, b[col])}. {a.label} أعلى بـ {abs(gap):.1f}{'%' if col not in ('margin_pct', 'disc_pct') else ' نقطة'}."
        else:
            lines = "، ".join(f"{r.label} ({self.fmt(col, r[col])})" for _, r in tab.head(5).iterrows())
            head = f"{'أقل' if asc else 'أعلى'} {n if n > 1 else ''} {DIM_PL.get(dim, DIM_AR[dim]) if n > 1 else DIM_AR[dim]} في {METRIC_AR[col]} ({title}): {lines}."
        fig = px.bar(tab.sort_values(col), x=col, y="label", orientation="h", color_discrete_sequence=[MINT], labels={col: METRIC_AR[col], "label": DIM_AR[dim]})
        show = tab[["label", "net", "margin", "margin_pct", "qty", "orders", "disc_pct"]].rename(columns=self._cols())
        return Answer.make(head, source=self.source(sp, METRIC_AR[col]), table=show, fig=fig)

    # ---------- ليه حصل التغيير؟
    def _slice(self, ent: dict, key: str, val) -> dict:
        e = {k: list(v) for k, v in ent.items()}
        e[key] = [val]
        return e

    def h_gap(self, sp) -> Answer:
        """«ليه X أقل من Y؟»: مقارنة العنصرين في نفس الفترة (مش تغير عبر الزمن)."""
        D, ent = self.D, sp["ent"]
        key = next(k for k in ("branches", "governorates", "categories", "channels") if len(ent[k]) >= 2)
        nm = lambda v: {"branches": D.branch_name.get(v, v), "categories": D.cat_ar.get(v, v), "channels": CH_AR.get(v, v)}.get(key, v)
        x = {v: self.data(sp["start"], sp["end"], self._slice(ent, key, v)) for v in ent[key][:2]}
        kp = {v: M.kpis(d) for v, d in x.items()}
        hi, lo = sorted(x, key=lambda v: -kp[v]["net"])
        kh, kl = kp[hi], kp[lo]
        gap = kh["net"] - kl["net"]
        facts = [f"المبيعات: {nm(lo)} {M.money(kl['net'])} مقابل {nm(hi)} {M.money(kh['net'])} (الفرق {M.money(gap)}، يعني {nm(lo)} أقل بـ {gap / kh['net'] * 100:.0f}%)."]
        effects = []
        if key == "governorates":
            nb = {v: int(x[v].branch.nunique()) for v in x}
            pb = {v: kp[v]["net"] / max(nb[v], 1) for v in x}
            facts.append(f"عدد الفروع اللي بتبيع: {nm(lo)} {nb[lo]} مقابل {nm(hi)} {nb[hi]}، ومتوسط مبيعات الفرع {M.money(pb[lo])} مقابل {M.money(pb[hi])}.")
            effects = [("عدد الفروع", (nb[hi] - nb[lo]) * pb[lo]), ("أداء الفرع الواحد", nb[hi] * (pb[hi] - pb[lo]))]
        else:
            effects = [("عدد الفواتير", (kh["orders"] - kl["orders"]) * kl["basket"]), ("متوسط الفاتورة", kh["orders"] * (kh["basket"] - kl["basket"]))]
            facts.append(f"الفواتير: {M.num(kl['orders'])} مقابل {M.num(kh['orders'])}، ومتوسط الفاتورة {M.money(kl['basket'])} مقابل {M.money(kh['basket'])}.")
        br = M.bridge(x[lo], x[hi])
        facts.append(f"القطع {M.num(kl['qty'])} مقابل {M.num(kh['qty'])}، وسعر القطعة المتوسط {M.money(kl['price'])} مقابل {M.money(kh['price'])}، والخصم {kl['disc_pct']:.1f}% مقابل {kh['disc_pct']:.1f}%، وهامش الربح {kl['margin_pct']:.1f}% مقابل {kh['margin_pct']:.1f}%.")
        facts.append(f"تفسير الفرق بالكمية والسعر والمزيج: أثر الكمية {M.money(br['volume'])}، أثر السعر والخصم {M.money(br['price'])}، أثر مزيج المنتجات {M.money(br['mix'])}.")
        top = max(effects, key=lambda e: abs(e[1]))
        head = f"{nm(lo)} أقل من {nm(hi)} في المبيعات بحوالي {M.money(gap)}. أكبر سبب ظاهر في الأرقام: «{top[0]}» (حوالي {min(abs(top[1]) / abs(gap) * 100, 100):.0f}% من الفرق)."
        tbl = pd.DataFrame({"المؤشر": ["المبيعات", "الربح", "القطع", "الفواتير", "متوسط الفاتورة", "سعر القطعة"], nm(lo): [kl["net"], kl["margin"], kl["qty"], kl["orders"], kl["basket"], kl["price"]],
                            nm(hi): [kh["net"], kh["margin"], kh["qty"], kh["orders"], kh["basket"], kh["price"]]})
        src = self.source(sp, "مقارنة عنصرين في نفس الفترة", origin="مبيعات الشركة الفعلية")
        return Answer.make(head, facts, [], [], source=src, notes=["ده فرق بين عنصرين في نفس الفترة، مش تغير عبر الزمن. الداتا بتوضح الفرق لكن مش بتقول ليه (موقع، منافسين، حجم السوق)."], table=tbl)

    def h_why(self, sp) -> Answer:
        D, ent = self.D, sp["ent"]
        if sp.get("comparative") and any(len(v) >= 2 for v in ent.values()):
            return self.h_gap(sp)
        start, end, yoy, metric = sp["start"], sp["end"], sp["yoy"], sp["metric_final"]
        metric = metric if metric in ("net", "margin", "margin_pct") else "net"
        ps, pe = (M.prev_range(start, end, "prev") if sp.get("driver_count") and not yoy
                  else self.prev_for(start, end, yoy))
        b, a = self.data(start, end, ent), self.data(ps, pe, ent)
        if not len(a) or not len(b):
            return Answer.make("مفيش بيانات كفاية في الفترتين للمقارنة.")
        scope = self.scope_text(ent)
        if sp.get("driver_count") and metric == "margin":
            before, after = float(a.margin.sum()), float(b.margin.sum())
            delta = after - before
            change = "زاد" if delta > 0 else "قل" if delta < 0 else "لم يتغير"
            percent = f"{delta / abs(before) * 100:+.2f}%" if before else "غير محسوبة لأن ربح الفترة السابقة صفر"
            drivers = M.delta_by(a, b, D, "branch", "margin")
            drivers = drivers.loc[drivers.delta.abs().sort_values(ascending=False).index].head(sp["driver_count"])
            table = drivers[["label", "a", "b", "delta"]].rename(columns={
                "label": "الفرع", "a": "الربح السابق (جنيه)", "b": "الربح الحالي (جنيه)",
                "delta": "المساهمة في تغير الربح (جنيه)"})
            return Answer.make(
                f"ربح {scope} {change}؛ الفرق {delta:+,.2f} جنيه. نسبة تغير قيمة الربح: {percent}.",
                facts=[f"ربح الفترة الحالية {start:%Y-%m-%d} إلى {end:%Y-%m-%d}: {after:,.2f} جنيه.",
                       f"ربح الفترة السابقة {ps:%Y-%m-%d} إلى {pe:%Y-%m-%d}: {before:,.2f} جنيه."],
                table=table.round(2), source=self.source(sp, "قيمة الربح ومساهمة الفروع في تغيره"),
                notes=["الربح هنا ربح البيع المسجل، قبل المرتبات والمصروفات.",
                       "تأثير الفرع هو فرق ربحه عن الفترة السابقة المساوية؛ الترتيب حسب القيمة المطلقة للفرق بالجنيه."])
        if sp.get("driver_count"):
            before, after = float(a.net.sum()), float(b.net.sum())
            delta = after - before
            pct = M.growth(after, before)
            pct_text = f"{pct:+.2f}%" if pct is not None else "غير محسوبة لأن مبيعات الفترة السابقة صفر"
            drivers = M.drivers(a, b, D, "branch")
            drivers = drivers.loc[drivers.delta.abs().sort_values(ascending=False).index].head(sp["driver_count"])
            table = drivers[["label", "net_a", "net_b", "delta", "growth"]].rename(columns={
                "label": "الفرع", "net_a": "المبيعات السابقة (جنيه)", "net_b": "المبيعات الحالية (جنيه)",
                "delta": "المساهمة في التغير (جنيه)", "growth": "التغير %"})
            bridge = M.bridge(a, b)
            ka, kb = M.kpis(a), M.kpis(b)
            factors = [
                f"الكمية: من {bridge['qty_a']:,.0f} إلى {bridge['qty_b']:,.0f} قطعة؛ أثرها المحسوب {bridge['volume']:+,.2f} جنيه.",
                f"تغير سعر البيع الصافي بعد الخصم للمنتجات المشتركة: أثره {bridge['price']:+,.2f} جنيه.",
                f"تغير مزيج المنتجات (الباقي في تفكيك الفرق): أثره {bridge['mix']:+,.2f} جنيه.",
                f"نسبة الخصم من إجمالي المبيعات قبل الخصم: من {ka['disc_pct']:.2f}% إلى {kb['disc_pct']:.2f}%، بفارق {kb['disc_pct'] - ka['disc_pct']:+.2f} نقطة مئوية. أثر الخصم داخل أثر السعر الصافي، ومش بند إضافي نجمعه عليه."]
            return Answer.make(
                f"مقارنة مبيعات {scope}: الفرق {delta:+,.2f} جنيه ({pct_text}).",
                facts=[f"الحالية: {start:%Y-%m-%d} إلى {end:%Y-%m-%d}: {after:,.2f} جنيه.",
                       f"السابقة: {ps:%Y-%m-%d} إلى {pe:%Y-%m-%d}: {before:,.2f} جنيه."] + factors,
                table=table.round(2), source=self.source(sp, "مقارنة المبيعات ومساهمة الفروع"),
                notes=["ده تفسير حسابي للتغير في المبيعات، مش إثبات لأسباب خارجية زي الحملات أو المنافسة أو الموسم. الأثر السالب خفّض المبيعات والموجب زوّدها.",
                       "الفروع مرتبة حسب القيمة المطلقة للفرق بالجنيه؛ الإشارة توضح زيادة أو انخفاض المبيعات.",
                       f"الفترة المستخدمة: {sp['plabel']}. راجع التواريخ أعلاه؛ طلب آخر عدد من الشهور يُحسب كفترة متحركة بواقع 30 يومًا للشهر، وليس شهورًا تقويمية كاملة."])
        pcmp = "نفس الفترة السنة اللي فاتت" if yoy else f"الفترة {ps:%Y-%m-%d} إلى {pe:%Y-%m-%d}"
        label = {"net": "المبيعات", "margin": "الربح (القيمة)", "margin_pct": "هامش الربح (النسبة)"}[metric]
        src = self.source(sp, "تغير " + label, origin=f"مقارنة فترتين من مبيعات الشركة (الحالية مقابل {pcmp})")
        facts, infer = [], []
        notes = ["الداتا بتوضح إيه اللي اتغير ومين اتغير، لكن مش بتقول ليه حصل بره الأرقام (حملات، منافسين، موسم)."]
        # الأبعاد اللي هنحلل بيها: بنتجنب البعد اللي المستخدم ثبّته بالفعل (لو سأل عن فروع معينة نحلل الفروع دي وأنواعها، مش القنوات)
        if ent["branches"]:
            rows_key, dims = "branches", ["category", "product"]
        elif ent["categories"]:
            rows_key, dims = "categories", ["branch", "product"]
        elif ent["channels"]:
            rows_key, dims = "channels", ["branch", "category"]
        elif ent["governorates"]:
            rows_key, dims = "governorates", ["branch", "category"]
        else:
            rows_key, dims = None, ["branch", "category", "channel"]
        rows = []
        if rows_key:
            for v in ent[rows_key]:
                e2 = self._slice(ent, rows_key, v)
                ra, rb = self.data(ps, pe, e2), self.data(start, end, e2)
                nm = {"branches": D.branch_name.get(v, v), "categories": D.cat_ar.get(v, v), "channels": CH_AR.get(v, v)}.get(rows_key, v)
                if not len(ra) or not len(rb):
                    rows.append({"label": nm, "قبل": np.nan, "بعد": np.nan, "التغير": np.nan, "%": np.nan})
                    continue
                if metric == "net":
                    br_ = M.bridge(ra, rb)
                    rows.append({"label": nm, "قبل": br_["net_a"], "بعد": br_["net_b"], "التغير": br_["delta"], "%": M.growth(br_["net_b"], br_["net_a"]),
                                 "أثر الكمية": br_["volume"], "أثر السعر والخصم": br_["price"], "أثر المزيج": br_["mix"]})
                elif metric == "margin":
                    pb_ = M.profit_bridge(ra, rb)
                    rows.append({"label": nm, "قبل": pb_["a"], "بعد": pb_["b"], "التغير": pb_["delta"], "%": M.growth(pb_["b"], pb_["a"]),
                                 "أثر الكمية": pb_["volume"], "أثر السعر": pb_["price"], "أثر التكلفة": pb_["cost"], "أثر منتجات جديدة/موقوفة": pb_["mix"]})
                else:
                    ka, kb = M.kpis(ra), M.kpis(rb)
                    rows.append({"label": nm, "قبل": ka["margin_pct"], "بعد": kb["margin_pct"], "التغير": kb["margin_pct"] - ka["margin_pct"], "%": np.nan})
        ka, kb = M.kpis(a), M.kpis(b)
        for r_ in rows:
            if r_["قبل"] == r_["قبل"]:
                u = (lambda v: f"{v:.1f}%") if metric == "margin_pct" else M.money
                facts.append(f"{r_['label']}: {u(r_['قبل'])} ← {u(r_['بعد'])}" + (f" ({r_['%']:+.1f}%)" if r_["%"] == r_["%"] else ""))
            else:
                facts.append(f"{r_['label']}: مفيش بيانات كفاية في إحدى الفترتين.")
        if metric == "net":
            br = M.bridge(a, b)
            d, ga = br["delta"], M.growth(br["net_b"], br["net_a"])
            pre = "ملحوظة: الأرقام بتقول إنها زادت مش نزلت. " if sp["claim_drop"] and d > 0 else ("ملحوظة: الأرقام بتقول إنها نزلت مش زادت. " if sp["claim_rise"] and d < 0 else "")
            head = f"{pre}مبيعات {scope} {'زادت' if d >= 0 else 'نزلت'} {abs(ga or 0):.1f}% ({M.money(br['net_a'])} ← {M.money(br['net_b'])}) مقارنة ب{pcmp}."
            facts.append(f"أثر الكمية {M.money(br['volume'])} (القطع من {M.num(br['qty_a'])} لـ {M.num(br['qty_b'])})، وأثر السعر والخصم {M.money(br['price'])}، وأثر تغير مزيج المنتجات {M.money(br['mix'])}.")
        elif metric == "margin":
            pb = M.profit_bridge(a, b)
            d, ga = pb["delta"], M.growth(pb["b"], pb["a"])
            pre = "ملحوظة: الأرقام بتقول إن الربح زاد مش نزل. " if sp["claim_drop"] and d > 0 else ("ملحوظة: الأرقام بتقول إن الربح نزل مش زاد. " if sp["claim_rise"] and d < 0 else "")
            head = f"{pre}ربح {scope} (القيمة بالجنيه) {'زاد' if d >= 0 else 'نزل'} {abs(ga or 0):.1f}% ({M.money(pb['a'])} ← {M.money(pb['b'])}) مقارنة ب{pcmp}."
            facts.append(f"أثر الكمية {M.money(pb['volume'])}، وأثر السعر {M.money(pb['price'])}، وأثر تكلفة القطعة {M.money(pb['cost'])}، وأثر منتجات دخلت أو اتوقفت {M.money(pb['mix'])}. (ده تغير في قيمة الربح، مش في نسبة الهامش.)")
            facts.append(f"هامش الربح (النسبة) كان {ka['margin_pct']:.1f}% وبقى {kb['margin_pct']:.1f}%.")
        else:
            d = kb["margin_pct"] - ka["margin_pct"]
            pre = "ملحوظة: الهامش تقريبًا ثابت (الفرق أقل من 0.1 نقطة). " if abs(d) < 0.1 and (sp["claim_drop"] or sp["claim_rise"]) else ""
            move = f"ثابت تقريبًا ({ka['margin_pct']:.1f}% ← {kb['margin_pct']:.1f}%)" if abs(d) < 0.05 else f"{'زاد' if d >= 0 else 'نزل'} {abs(d):.1f} نقطة ({ka['margin_pct']:.1f}% ← {kb['margin_pct']:.1f}%)"
            head = f"{pre}هامش ربح {scope} (النسبة) {move} مقارنة ب{pcmp}."
        tabs = []
        for dim in dims:
            if metric == "net":
                t = M.drivers(a, b, D, dim)
                pick = t.head(3) if d < 0 else t.tail(3).iloc[::-1]
                facts.append(f"أكتر {DIM_PL[dim]} {'اتسببت في النزول' if d < 0 else 'ساهمت في الزيادة'}: " + "، ".join(f"{r.label} ({M.money(r.delta)})" for r in pick.itertuples()))
                tabs.append((dim, t.rename(columns={"delta": "delta"})))
            elif metric == "margin":
                t = M.delta_by(a, b, D, dim, "margin")
                pick = t.head(3) if d < 0 else t.tail(3).iloc[::-1]
                facts.append(f"أكتر {DIM_PL[dim]} {'نزّلت' if d < 0 else 'رفعت'} الربح: " + "، ".join(f"{r.label} ({M.money(r.delta)})" for r in pick.itertuples()))
                tabs.append((dim, t))
            else:
                t = M.margin_bridge(a, b, D, dim).rename(columns={"total": "delta"})
                pick = t.head(2) if d < 0 else t.tail(2).iloc[::-1]
                facts.append(f"أكتر {DIM_PL[dim]} {'نزّلت' if d < 0 else 'رفعت'} الهامش: " + "، ".join(f"{r.label} ({r.delta:+.2f} نقطة؛ منها {r.rate:+.2f} من تغير هامشه هو و{r.mix:+.2f} من تغير وزنه)" for r in pick.itertuples()))
                tabs.append((dim, t))
        if abs(kb["disc_pct"] - ka["disc_pct"]) >= 0.3:
            facts.append(f"الخصم الإجمالي اتغير من {ka['disc_pct']:.1f}% لـ {kb['disc_pct']:.1f}%.")
        if metric != "margin_pct" and d < 0:
            idf = self.inv_df
            scope_b = ent["branches"] or [r.key for r in (M.drivers(a, b, D, "branch").head(3).itertuples())]
            for bid in scope_b[:3]:
                x = idf[(idf.branch == bid) & idf.status.isin(["حرج", "منخفض"])]
                if len(x):
                    infer.append(f"فرع {D.branch_name.get(bid, bid)} فيه {len(x)} نوع مخزونه حرج أو منخفض ({', '.join(x.category_ar.head(3))})، فممكن يكون ده من أسباب النزول (لازم تتأكد).")
        fig = None
        if tabs:
            t0 = tabs[0][1]
            top = pd.concat([t0.head(5), t0.tail(5)]).drop_duplicates("key").sort_values("delta")
            fig = px.bar(top, x="delta", y="label", orientation="h", color="delta", color_continuous_scale=[RED, "#2a3a35", MINT], labels={"delta": "التغير", "label": ""})
            fig.update_layout(coloraxis_showscale=False)
        if rows:
            tbl = pd.DataFrame(rows).rename(columns={"label": {"branches": "الفرع", "categories": "النوع", "channels": "القناة", "governorates": "المحافظة"}[rows_key]})
        elif metric == "net":
            tbl = pd.DataFrame({"أثر": ["الكمية", "السعر والخصم", "مزيج المنتجات", "الإجمالي"], "القيمة (ج)": [br["volume"], br["price"], br["mix"], br["delta"]]})
        elif metric == "margin":
            tbl = pd.DataFrame({"أثر": ["الكمية", "السعر", "تكلفة القطعة", "منتجات جديدة/موقوفة", "الإجمالي"], "القيمة (ج)": [pb["volume"], pb["price"], pb["cost"], pb["mix"], pb["delta"]]})
        else:
            tbl = tabs[0][1][["label", "margin_pct_a", "margin_pct_b", "rate", "mix", "delta"]].rename(columns={"label": "الاسم", "margin_pct_a": "الهامش قبل %", "margin_pct_b": "الهامش بعد %", "rate": "أثر تغير هامشه", "mix": "أثر تغير وزنه", "delta": "الإجمالي (نقطة)"})
        return Answer.make(head, facts, [], infer, source=src, notes=notes, table=tbl, fig=fig)

    # ---------- التوقعات
    def h_forecast(self, sp) -> Answer:
        D, ent = self.D, sp["ent"]
        t, n = toks(sp["text"]), norm(sp["text"])
        if ent["branches"]:
            level, keys = "branch", ent["branches"]
        elif ent["categories"]:
            level, keys = "category", ent["categories"]
        else:
            level, keys = "total", ["__total__"]
        qty_wanted = kw(t, "قطع", "كميه", "وحدات") or sp["metric"] == "qty"
        target = "qty" if qty_wanted or level != "total" and not kw(t, "مبيعات", "ايراد", "فلوس") else "net"
        months = sp.get("months") or (1 if phrase(n, "الشهر الجاي", "الشهر القادم") else 3)
        lines, notes, out = [], [], []
        fig = go.Figure()
        name_of = lambda lv, k: "كل الشركة" if k == "__total__" else (D.cat_ar.get(k, k) if lv == "category" else D.branch_name.get(k, k))
        for k in keys:
            f = D.forecast[(D.forecast.level == level) & (D.forecast.series == k) & (D.forecast.target == "qty")].sort_values("h")
            q = D.fq[(D.fq.level == level) & (D.fq.series == k) & (D.fq.target == "qty")]
            nm = name_of(level, k)
            if not len(f) or not len(q):
                notes.append(f"مفيش توقع موثوق لـ {nm}: تاريخ المبيعات قصير أو حجمها صغير.")
                continue
            q = q.iloc[0]
            price = q.price if target == "net" else 1.0
            val = (lambda v: f"{v:,.0f} قطعة") if target == "qty" else M.money
            f = f.assign(forecast=f.forecast * price, lo=f.lo * price, hi=f.hi * price)
            lines.append(f"{nm}: " + "، ".join(f"{r.month}: {val(r.forecast)} (المدى المحتمل {val(r.lo)} إلى {val(r.hi)})" for r in f.head(months).itertuples()))
            lines.append(f"{nm}: الطريقة «{MODEL_AR.get(q.model, q.model)}». في اختبار على شهور حُجزت (الأحدث) كان متوسط الغلط {q.wape_test:.1f}% في الشهور 1-3 (الطريقة البسيطة {q.baseline_test:.1f}%). الموثوقية: {q.reliability}." + (f" {q.note}." if q.note else ""))
            if q.reliability == "ضعيفة":
                notes.append(f"توقع {nm} ضعيف الموثوقية، استخدمه كاتجاه عام مش كرقم للقرار.")
            hist = monthly(D.sales, target if target in ("qty", "net") else "qty", by=None if level == "total" else level)
            hist = hist if level == "total" else hist[k]
            fig.add_scatter(x=[str(p) for p in hist.index[-18:]], y=hist.values[-18:], name=f"{nm} (فعلي)", mode="lines")
            fig.add_scatter(x=f.month, y=f.forecast, name=f"{nm} (توقع)", mode="lines+markers", line=dict(dash="dash"))
            out.append(f.head(months).assign(الاسم=nm)[["الاسم", "month", "lo", "forecast", "hi"]])
            if f.iloc[0].month == f"{D.last:%Y-%m}":
                nc = mtd_nowcast(D, k, level, target if target in ("qty", "net") else "qty")
                if nc and nc["projected"]:
                    notes.append(f"{nm}: الشهر الحالي لسه ماخلصش. اتباع منه {val(nc['so_far'] * (price if target == 'net' and False else 1))} حتى يوم {nc['day']}، وبنمط الشهور اللي فاتت ده بيوصل لحوالي {val(nc['projected'])} للشهر كله.")
        if not out:
            return Answer.make("مفيش توقعات موثوقة للاختيار ده.", notes=notes)
        if target == "net":
            notes.append("الإيراد المتوقع = الكمية المتوقعة × سعر القطعة الحالي، يعني بيفترض إن الأسعار ثابتة. أسعار الشركة بتزيد مرة في السنة في يناير (آخر زيادتين: +38% و+23%)، فأي شهر بعد يناير محتاج افتراض لزيادة الأسعار (في صفحة التوقعات).")
        notes.append("متوسط الغلط ≠ المدى المحتمل: الأول بيقيس ابتعاد التوقع عن الفعلي في المتوسط، والتاني نطاق بنتوقع إن الرقم الفعلي يقع جواه في 80% من الحالات.")
        tbl = pd.concat(out).rename(columns={"month": "الشهر", "lo": "أقل تقدير", "forecast": "المتوقع", "hi": "أعلى تقدير"})
        sp_ = {"المقياس": "توقع " + ("القطع" if target == "qty" else "المبيعات"), "الفترة": "الشهور الجاية", "النطاق": ("، ".join(name_of(level, k) for k in keys)),
               "المصدر": "نماذج توقع متجرّبة على شهور فاتت (صفحة التوقعات)"}
        return Answer.make(f"توقع {'القطع' if target == 'qty' else 'المبيعات'} للشهور الجاية.", [], lines, [], source=sp_, notes=notes, table=tbl, fig=fig)

    # ---------- المخزون
    def h_inventory(self, sp) -> Answer:
        D, ent = self.D, sp["ent"]
        t = toks(sp["text"])
        df = self.inv_df
        if ent["branches"]:
            df = df[df.branch.isin(ent["branches"])]
        elif ent["governorates"]:
            df = df[df.governorate.isin(ent["governorates"])]
        if ent["categories"]:
            df = df[df.category.isin(ent["categories"])]
        if not len(df):
            return Answer.make("مفيش بيانات مخزون للاختيار ده.")
        scope, s = self.scope_text(ent), inv.summary(df)
        src = {"المقياس": "حالة المخزون", "الفترة": f"آخر سجل مخزون + مبيعات آخر 13 أسبوع حتى {D.last:%Y-%m-%d}", "النطاق": scope, "المصدر": "ملف المخزون الشهري + المبيعات + التوقعات"}
        head = f"مخزون {scope}: {M.num(s['units'])} قطعة بقيمة {M.money(s['value'])}، بيكفي حوالي {s['cover']:.0f} أسبوع من البيع المتوقع. حالات حرجة: {s['critical']}، منخفضة: {s['low']}، زيادة: {s['over']}."
        view = sp.get("view") or ("transfers" if kw(t, "تحويل", "تحويلات") else "overstock" if kw(t, "راكد", "ركود", "زياده") else "critical")
        if view == "transfers":
            tr = inv.transfers(D, self.inv_df)
            if ent["branches"]:
                tr = tr[tr.to_id.isin(ent["branches"])]
            if ent["categories"]:
                tr = tr[tr.category.isin(ent["categories"])]
            if not len(tr):
                return Answer.make("مفيش تحويلات مطلوبة للاختيار ده.", source=src)
            facts = [f"{r.units:.0f} {r.category_ar} من {r['from']} إلى {r.to} (الفرع بيكفي {r.cover_weeks:.1f} أسبوع)" for _, r in tr.head(5).iterrows()]
            return Answer.make(f"{len(tr)} تحويلة مقترحة بإجمالي {M.num(tr.units.sum())} قطعة.", facts, source=src,
                               table=tr[["category_ar", "to", "from", "units", "status"]].rename(columns={"category_ar": "النوع", "to": "إلى فرع", "from": "من", "units": "الكمية", "status": "الحالة"}))
        if view == "overstock":
            pick = df[df.status.isin(["زيادة", "راكد"])].sort_values("tied_capital", ascending=False).head(8)
            facts = [f"الفلوس المتجمدة في الزيادة حوالي {M.money(s['tied'])}."] + [f"{r.category_ar} في {r.branch_name}: متجمد {M.money(r.tied_capital)} (يكفي {min(r.cover_weeks, 999):.0f} أسبوع)" for _, r in pick.head(4).iterrows()]
        else:
            pick = df[df.status.isin(["حرج", "منخفض"])].sort_values("cover_weeks").head(8)
            facts = [f"{r.category_ar} في {r.branch_name}: بيكفي {r.cover_weeks:.1f} أسبوع (بيبيع {r.weekly_demand:.1f} في الأسبوع)" for _, r in pick.head(4).iterrows()]
        tbl = pick[["branch_name", "category_ar", "stock", "weekly_demand", "cover_weeks", "status"]].rename(columns={"branch_name": "الفرع", "category_ar": "النوع", "stock": "المخزون", "weekly_demand": "الطلب الأسبوعي المتوقع", "cover_weeks": "يكفي (أسبوع)", "status": "الحالة"}).round(1)
        return Answer.make(head, facts, source=src, table=tbl)

    # ---------- العملاء
    def h_customers(self, sp) -> Answer:
        D = self.D
        c = D.churn
        if not len(c):
            return Answer.make("بيانات العملاء مش متاحة.")
        t = toks(sp["text"])
        seg = c.groupby("segment").agg(عدد=("customer_id", "size"), إنفاق_آخر_12_شهر=("spend12m", "sum"), متوسط_الطلبات=("freq", "mean")).sort_values("إنفاق_آخر_12_شهر", ascending=False)
        top10 = c.spend12m.sort_values(ascending=False).head(int(len(c) * 0.1)).sum() / c.spend12m.sum() * 100
        cm = D.model_metrics.get("churn", {})
        head = f"عندنا {M.num(len(c))} عميل مسجل. أكبر شريحة إنفاقًا: {seg.index[0]} ({M.num(seg.iloc[0].عدد)} عميل، {M.money(seg.iloc[0].إنفاق_آخر_12_شهر)} آخر 12 شهر)."
        facts = [f"أفضل 10% من العملاء بيجيبوا {top10:.0f}% من إنفاق آخر 12 شهر."]
        notes = []
        if cm and not cm.get("predictive", True):
            notes.append(f"اختبرنا توقع رجوع العميل ودقة النموذج ({cm['auc']}) زي التخمين. نسبة الرجوع خلال 4 شهور حوالي {cm['base_rate'] * 100:.0f}% في كل الشرائح، فالغياب لوحده مش مؤشر على الضياع.")
        src = {"المقياس": "شرائح العملاء", "الفترة": "آخر 12 شهر حتى " + f"{D.last:%Y-%m-%d}", "النطاق": "العملاء المسجلين فقط", "المصدر": "مبيعات العملاء المسجلين (نموذج RFM)"}
        if sp.get("view") == "top" or (sp.get("view") is None and kw(t, "احسن", "اكبر", "اهم")):
            top = c.sort_values("spend12m", ascending=False).head(10).merge(D.customers[["customer_id", "customer_name_ar", "governorate_ar"]], on="customer_id", how="left")
            return Answer.make(head, facts, source=src, notes=notes, table=top[["customer_name_ar", "governorate_ar", "freq", "spend12m", "segment"]].rename(
                columns={"customer_name_ar": "العميل", "governorate_ar": "المحافظة", "freq": "عدد الطلبات", "spend12m": "إنفاق آخر 12 شهر", "segment": "الشريحة"}).round(0))
        fig = px.bar(seg.reset_index(), x="segment", y="إنفاق_آخر_12_شهر", color_discrete_sequence=[MINT], labels={"segment": "الشريحة", "إنفاق_آخر_12_شهر": "إنفاق آخر 12 شهر (ج)"})
        return Answer.make(head, facts, source=src, notes=notes, fig=fig,
                           table=seg.reset_index().rename(columns={"segment": "الشريحة", "إنفاق_آخر_12_شهر": "إنفاق آخر 12 شهر", "متوسط_الطلبات": "متوسط الطلبات"}).round(1))

    # ---------- التوصيات
    def h_recommend(self, sp) -> Answer:
        t = toks(sp["text"])
        r = self.insights
        areas = {"الفروع": ("فرع", "فروع"), "المخزون": ("مخزون", "بضاعه"), "العملاء": ("عملاء", "عميل"), "التسعير": ("تسعير", "خصم", "خصومات", "سعر"), "الربحية": ("ربح", "هامش"), "الجودة": ("مرتجع", "جوده")}
        for a, ws in areas.items():
            if kw(t, *ws) and len(r[r.area == a]):
                r = r[r.area == a]
                break
        r = r.head(5)
        facts = [f"[{x.priority}] {x.title}: {x.evidence}" + (f" (أثر مقدر {M.money(x.impact)}: {x.impact_note})" if x.impact == x.impact else "") for x in r.itertuples()]
        actions = [f"{x.title}: {x.action}" for x in r.itertuples()]
        src = {"المقياس": "محرك التوصيات", "الفترة": "آخر 12 شهر (والمخزون الحالي)", "النطاق": "كل الشركة", "المصدر": "قواعد محسوبة على الداتا (صفحة التوصيات)"}
        return Answer.make(f"أهم {len(r)} حاجات الداتا بتقول إنها محتاجة تدخل:", facts, [], [], actions=actions, source=src,
                           table=r[["priority", "area", "title", "impact"]].rename(columns={"priority": "الأولوية", "area": "المجال", "title": "التوصية", "impact": "الأثر المقدر (ج)"}))

    def h_discount(self, sp) -> Answer:
        D = self.D
        df = apply_filters(D.sales, D.last - pd.Timedelta(days=364), D.last)
        ch = M.by_dim(df, D, "channel").sort_values("disc_pct", ascending=False)
        facts = [f"آخر 12 شهر: خصم {CH_AR.get(r.key, r.key)} {r.disc_pct:.1f}% وهامش ربحها {r.margin_pct:.1f}%." for r in ch.itertuples()]
        return Answer.make("مقدرش أحسب أثر خصم جديد على المبيعات من الداتا الحالية، فمش هقترح نسبة خصم معينة.", facts, [], [],
                           notes=["الداتا مفيهاش تجارب خصم مقارنة، وأي رقم هقوله عن «هيزود المبيعات كام» هيبقى تخمين.", "الأنسب: جرّب الخصم في عدد محدود من الفروع مع فروع شبيهة من غير خصم، وقارن النتيجة بعد 4 أسابيع."],
                           source={"المقياس": "الخصومات الحالية", "الفترة": "آخر 12 شهر", "النطاق": "كل الشركة", "المصدر": "مبيعات الشركة الفعلية"})

    def h_health(self, sp) -> Answer:
        h = pd.DataFrame(self.D.health)
        bad = h[h.status != "ok"]
        head = f"عملنا {len(h)} فحص على البيانات: {len(h) - len(bad)} سليم و{len(bad)} فيهم ملاحظات."
        return Answer.make(head, list(bad.detail.head(4)), source={"المقياس": "جودة البيانات", "الفترة": "آخر بناء للبيانات", "النطاق": "كل الجداول", "المصدر": "فحوصات data.py"},
                           table=h.rename(columns={"check": "الفحص", "status": "النتيجة", "detail": "التفاصيل"}))

    # ---------- الذكاء الاصطناعي (Groq): فهم السؤال وتلخيص بس، والأرقام كلها من حسابات النظام
    PLAN_SYS = ("You convert a manager's Arabic question about a clothing retailer into JSON calls. Output ONLY JSON: "
                '{"calls":[{"title":"","tool":"","metric":null,"dim":null,"rank":null,"top_n":null,"start":null,"end":null,"branches":[],"governorates":[],"categories":[],"channels":[],"compare":null,"view":null,"months":null,"inherit":false}],"clarify":""}. '
                "tool: query|why|forecast|inventory|customers|recommend|discount. metric: net (sales/مبيعات), margin (profit value/ربح), margin_pct (profit margin %/هامش الربح), qty, orders, basket, disc_pct, refund, customers. "
                "dim: branch|governorate|category|product|channel|month|week|day|year|null. rank: top|bottom|null. compare: prev|yoy|null. view: inventory(summary|critical|overstock|transfers), customers(segments|top). "
                "Dates YYYY-MM-DD. Omit unused optional fields; use short titles and at most four calls. Names (branches/categories/...) exactly as the user wrote, in Arabic. Split independent requests into calls in order. "
                "A period comparison with branch contributions and reasons is one why call, not repeated rankings. A named governorate is a filter, not a request to rank governorates. Keep the same explicit scope and dates across related calls. "
                "ربح=margin, هامش=margin_pct, never confuse them with sales. Keep every branch/category/channel the user names. inherit=true only if the question depends on the previous turn. "
                "If the question is too vague, return no calls and a short Arabic question in clarify. Never answer the question yourself.")
    TOOLS_OK = {"query", "why", "forecast", "inventory", "customers", "recommend", "discount"}

    def state_summary(self) -> str:
        st = self.state
        if not st:
            return ""
        return f"آخر سؤال: الفترة {st.get('plabel', '')} | النطاق {self.scope_text(st['ent'])} | المقياس {METRIC_AR.get(st.get('metric_final', 'net'), '')} | النوع {st.get('intent_final', '')}"

    def plan(self, text: str) -> dict:
        user = json.dumps({"q": text, "today": f"{self.D.last:%Y-%m-%d}", "default_period": [f"{self.ctx.start:%Y-%m-%d}", f"{self.ctx.end:%Y-%m-%d}"], "context": self.state_summary(),
                           "previous": [m["content"][:120] for m in self.turns[-2:] if m["role"] == "user"]}, ensure_ascii=False)
        out = self.client.chat_json([{"role": "system", "content": self.PLAN_SYS}, {"role": "user", "content": user}], model=self.client.cfg.planner_model, max_tokens=900,
                                    purpose="plan", qkey=qid(text), cache_key=f"{norm(text)}|{self.state_summary()}|{self.ctx.start:%Y%m%d}{self.ctx.end:%Y%m%d}")
        raw_calls = out.get("calls") or []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list) or any(not isinstance(c, dict) or not isinstance(c.get("tool"), str) or c.get("tool") not in self.TOOLS_OK for c in raw_calls):
            raise LLMError("invalid", "invalid plan calls")
        calls = raw_calls[:4]
        if not calls and not str(out.get("clarify") or "").strip():
            raise LLMError("invalid", "empty plan")
        return {"calls": calls, "clarify": str(out.get("clarify") or "").strip()[:300]}

    def _resolve(self, kind: str, names) -> tuple[list[str], list[str]]:
        """أسماء من النموذج → معرّفات حقيقية. أي اسم مش موجود بيتسجل كـ«مش لاقيه» بدل ما نحلل حاجة تانية."""
        D, ok, bad = self.D, [], []
        for nm in [str(x) for x in (names or [])][:6]:
            n = norm(nm).strip()
            hit: list[str] = []
            if kind == "branches":
                hit = [r.branch_id for r in D.branches.itertuples() if norm(r.label).strip() == n or norm(r.name).strip() == n] or self.entities(nm)["branches"]
            elif kind == "governorates":
                hit = [g for g in D.branches.governorate.unique() if norm(g).strip() == n or stem(norm(g).strip()) == stem(n)] or self.entities(nm)["governorates"]
            elif kind == "categories":
                hit = [k for k, v in D.cat_ar.items() if norm(v).strip() == n or k.lower() == nm.lower()] or self.entities(nm)["categories"]
            elif kind == "channels":
                hit = self.entities(nm)["channels"]
            ok += [h for h in hit if h not in ok]
            if not hit:
                bad.append(nm)
        return ok, bad

    def spec_from_call(self, c: dict, text: str) -> dict:
        t = toks(text)
        ent, missing = {}, []
        for kind in ("branches", "governorates", "categories", "channels"):
            ent[kind], bad = self._resolve(kind, c.get(kind))
            missing += bad
        explicit = self.entities(text)
        for kind in ent:
            if not ent[kind] and not c.get(kind):
                ent[kind] = explicit[kind]
        per = self.parse_period(text)
        try:
            if c.get("start") or c.get("end"):
                s = pd.Timestamp(c.get("start") or c.get("end"))
                e = min(pd.Timestamp(c.get("end") or c.get("start")), self.D.last.normalize())
                per = (s, e, f"{s:%Y-%m-%d} إلى {e:%Y-%m-%d}")
        except (ValueError, TypeError):
            per = None
        metric = c.get("metric") if c.get("metric") in METRIC_AR else None
        dim = c.get("dim") if c.get("dim") in DIM_AR else None
        topn = c.get("top_n") if isinstance(c.get("top_n"), int) and 0 < c["top_n"] <= 50 else None
        rank = c.get("rank") in ("top", "bottom")
        sp = {"text": f"{c.get('title') or ''} {text}", "intent": {"query": None, "why": "why", "forecast": "forecast", "inventory": "inventory", "customers": "customers", "recommend": "recommend", "discount": "discount"}[c["tool"]],
              "ent": ent, "per": per, "metric": metric, "dim": dim, "rank": rank, "asc": c.get("rank") == "bottom", "topn": topn, "yoy": c.get("compare") == "yoy",
              "claim_drop": kw(t, "نزل", "انخفض", "قلت", "هبط", "تراجع"), "claim_rise": kw(t, "زاد", "ارتفع", "طلع"), "inherited": [], "follow": bool(c.get("inherit")),
              "view": c.get("view") if c.get("view") in ("summary", "critical", "overstock", "transfers", "segments", "top") else None,
              "months": c.get("months") if isinstance(c.get("months"), int) and 1 <= c["months"] <= 6 else None, "unresolved": missing}
        local = self.parse(text)
        if phrase(norm(text), "كل الشركه", "الشركه كلها", "كل الفروع", "من جديد"):
            sp["follow"] = False
        if local["follow"]:
            sp["follow"] = True
            if self.state and len(self.split_parts(text)) == 1:
                for kind in sp["ent"]:
                    sp["ent"][kind] = list(local["ent"][kind])
                sp["unresolved"] = [name for name in missing if norm(name).strip() in norm(text)]
            sp["per"] = local["per"]  # None means inherit actual previous dates, not model defaults.
        if len(self.split_parts(text)) == 1:
            if local["per"] is not None:
                sp["per"] = local["per"]
            if local["metric"] is not None:
                sp["metric"] = local["metric"]
            for kind, values in local["ent"].items():
                if values:
                    sp["ent"][kind] = values
            if local["ent"]["governorates"] and local["dim"] is None:
                sp["dim"], sp["rank"] = None, False
        return sp

    def should_narrate(self, ans: Answer) -> bool:
        mode = self.client.cfg.narrate
        if mode in ("off", "false", "0") or not self.client.available:
            return False
        if mode == "on":
            return True
        parts = ans.parts or [ans]
        return len(parts) > 1 or any(p.inferences or p.forecasts for p in parts)

    def narrate(self, ans: Answer, text: str) -> str:
        """ملخص تنفيذي قصير. بنبعت للنموذج نص النتايج المحسوبة بس، وأي رقم في ملخصه مش موجود فيها بيترفض."""
        parts = ans.parts or [ans]
        facts = [{"title": p.title, "headline": p.headline[:260], "facts": [x[:240] for x in p.facts[:4]], "forecast": [x[:240] for x in p.forecasts[:2]],
                  "inference": [x[:200] for x in p.inferences[:2]], "notes": [x[:160] for x in p.notes[:1]]} for p in parts]
        src = json.dumps(facts, ensure_ascii=False)
        sysm = ("اكتب ملخص تنفيذي من 3 لـ 4 جمل بالعربي البسيط لمدير غير تقني، من النتايج المرفقة فقط. ممنوع تضيف أي رقم أو سبب أو توصية مش موجودة فيها. "
                "ميّز بين «الداتا بتقول» (حقائق) و«توقع» و«استنتاج محتمل (مش مثبت)». لو النتايج فيها تحفظ أو ملحوظة مهمة اذكرها. "
                'رد JSON بس: {"summary": "..."}')
        out = self.client.chat_json([{"role": "system", "content": sysm}, {"role": "user", "content": f"السؤال: {text}\nالنتايج: {src}"}], max_tokens=300, purpose="narrate", qkey=qid(text))
        s = str(out.get("summary") or "").strip()
        if not s or len(s) > 700:
            return ""
        bad = self.grounded(s, [src])
        if bad:
            self.client.log.warning("narration rejected: numbers not in results q=%s n=%d", qid(text), len(bad))
            return ""
        return s

    @staticmethod
    def _numbers(s: str) -> list[float]:
        return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", str(s)) if x.replace(",", "").replace(".", "").isdigit()]

    def grounded(self, text: str, bodies: list[str]) -> list[str]:
        """بترجع أي أرقام في النص مش موجودة في نتايج الحسابات (بسماح 1.5%)."""
        pool = []
        for b in bodies:
            for v in self._numbers(b):
                pool += [v, v * 1e3, v * 1e6, v / 100]
        bad = []
        for m in re.finditer(r"(\d[\d,]*\.?\d*)\s*(%?)", text):
            raw = m.group(1).replace(",", "")
            if not raw.replace(".", "").isdigit():
                continue
            v = float(raw)
            if (v <= 10 and not m.group(2) and "." not in raw) or 1900 <= v <= 2100:
                continue
            if not any(abs(v - p) <= max(0.015 * abs(p), 0.06) for p in pool):
                bad.append(m.group(0).strip())
        return bad
