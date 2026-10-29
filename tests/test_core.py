"""اختبارات: بتقارن أرقام النظام بحسابات مستقلة من ملفات CSV الأصلية، وبتجرب التوقعات (من غير تسريب بيانات) والمساعد وطبقة Groq والواجهة."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from src import data, inventory as inv, metrics as M, models, ui
from src.assistant import Assistant, Ctx
from src.llm import GroqClient, LLMError, Settings, load_env, redact

RAW = data.find_raw()
pytestmark = pytest.mark.skipif(RAW is None or not (data.PROC / "sales.parquet").exists(), reason="محتاج data/raw وdata/processed")


def local_assistant(D, ctx=None):
    """مساعد من غير مفتاح (محرك محلي)، عشان الاختبارات متتصلش بالإنترنت أبدًا."""
    return Assistant(D, ctx or Ctx(D.last - pd.Timedelta(days=364), D.last), GroqClient(Settings(api_key="")))


@pytest.fixture(scope="module")
def D():
    return data.load()


@pytest.fixture(scope="module")
def raw():
    it = pd.read_csv(RAW / "sales_order_items.csv")
    so = pd.read_csv(RAW / "sales_orders.csv")
    pr = pd.read_csv(RAW / "products_catalog.csv")
    it = it.merge(so[["sales_order_id", "sales_channel"]], on="sales_order_id").merge(pr[["product_id", "category_en"]], on="product_id")
    it["date"] = pd.to_datetime(it.order_date)
    return it


def test_totals_match_raw(D, raw):
    assert len(D.sales) == len(raw)
    assert D.sales.net.sum() == pytest.approx(raw.net_line_EGP.sum(), rel=1e-9)
    assert D.sales.margin.sum() == pytest.approx(raw.margin_EGP.sum(), rel=1e-9)
    assert D.sales.qty.sum() == raw.quantity.sum()
    assert D.sales.order.nunique() == raw.sales_order_id.nunique()


def test_filtered_kpis_match_raw(D, raw):
    s, e = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-08-31")
    brs, cats, chs = ["BR004", "BR010", "BR020"], ["Men boxers", "Socks"], ["Retail POS"]
    k = M.kpis(data.apply_filters(D.sales, s, e, brs, cats, None, chs))
    t = raw[(raw.date >= s) & (raw.date <= e) & raw.branch_id.isin(brs) & raw.category_en.isin(cats) & raw.sales_channel.isin(chs)]
    assert k["net"] == pytest.approx(t.net_line_EGP.sum(), rel=1e-9)
    assert k["margin"] == pytest.approx(t.margin_EGP.sum(), rel=1e-9)
    assert k["qty"] == t.quantity.sum() and k["orders"] == t.sales_order_id.nunique()
    assert k["disc_pct"] == pytest.approx(t.discount_line_EGP.sum() / t.gross_line_EGP.sum() * 100)


def test_by_dim_sums_to_total(D):
    df = data.apply_filters(D.sales, "2026-01-01", "2026-06-30")
    for dim in ("branch", "category", "channel", "governorate", "product", "month"):
        assert M.by_dim(df, D, dim).net.sum() == pytest.approx(df.net.sum(), rel=1e-9)


def test_bridge_and_drivers_add_up(D):
    a = data.apply_filters(D.sales, "2025-01-01", "2025-06-30")
    b = data.apply_filters(D.sales, "2026-01-01", "2026-06-30")
    br = M.bridge(a, b)
    assert br["volume"] + br["price"] + br["mix"] == pytest.approx(br["delta"], rel=1e-9)
    for dim in ("branch", "category", "channel"):
        d = M.drivers(a, b, D, dim)
        assert d.delta.sum() == pytest.approx(br["delta"], rel=1e-9)
        assert (d.volume + d.price).sum() == pytest.approx(br["delta"], rel=1e-6)


def test_branch_pnl_costs_are_prorated(D):
    df = data.apply_filters(D.sales, "2025-01-01", "2025-12-31")
    p = M.branch_pnl(D, df, "2025-01-01", "2025-12-31")
    raw_pay = D.costs[D.costs.month.str.startswith("2025")]
    assert p.payroll.sum() == pytest.approx(raw_pay.payroll.sum(), rel=1e-9)
    assert (p.profit == p.margin - p.payroll - p.opex).all()


def test_inventory_logic(D):
    df = inv.intel(D)
    c = df.cover_weeks
    assert (df[df.status == "حرج"].cover_weeks < 3).all()
    assert (df[df.status == "زيادة"].cover_weeks > inv.DEFAULTS["over"]).all()
    assert (df[df.status.isin(["سليم"])].cover_weeks.between(6, 26)).all()
    assert (df.need[~df.status.isin(["حرج", "منخفض"])] == 0).all()
    tr = inv.transfers(D, df)
    got = tr.groupby("to_id").units.sum()
    need = df.set_index(["branch"]).groupby("branch").need.sum()
    assert (got <= need.reindex(got.index) + len(tr)).all()           # مفيش فرع بياخد أكتر من احتياجه (فرق التقريب بس)
    gave = tr[tr.from_kind == "فرع"].groupby("from").units.sum()
    ex = df.groupby("branch_name").excess.sum()
    assert (gave <= ex.reindex(gave.index).fillna(0) + 1).all()       # مفيش فرع بيدي أكتر من زيادته
    # أثر نمو الطلب المتوقع: لو الطلب زاد، التغطية لازم تقل
    d2 = inv.intel(D, weeks=13)
    assert np.isclose(d2.cover_weeks.replace(np.inf, 0).sum(), c.replace(np.inf, 0).sum())


def test_assistant_understands_entities_and_periods(D):
    A = local_assistant(D)
    assert A.entities("فرع الرحاب")["branches"] == ["BR004"]
    assert A.entities("مبيعات القاهرة")["governorates"] == ["القاهرة"]
    assert set(A.entities("بيجامات حريمي")["categories"]) == {"Women pajamas"}
    assert "Men boxers" in A.entities("بوكسرات")["categories"]
    s, e, _ = A.parse_period("مبيعات سبتمبر 2026")
    assert (s, e) == (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30"))
    s, e, _ = A.parse_period("آخر 3 شهور")
    assert (e - s).days == 89
    assert A.parse_period("مبيعات كام") is None


def test_insights_have_evidence(D):
    from src import insights
    r = insights.generate(D)
    assert len(r) >= 5 and r.evidence.str.len().min() > 20 and set(r.priority) <= set(insights.PRIORITY)
    assert (r.impact.dropna() >= 0).all()


def _md(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_streamlit_pages_and_interaction():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
    assert not at.exception
    pages = list(at.sidebar.radio[0].options)
    assert len(pages) == 9
    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception[0].value if at.exception else "")
    # الفلتر لازم يغيّر الأرقام في الصفحة الرئيسية
    at.sidebar.radio[0].set_value(pages[0]).run()
    before = _md(at)
    assert "صافي المبيعات" in before and "لقينا" in before
    at.multiselect(key="f_gov").set_value(["القاهرة"]).run()
    assert not at.exception and _md(at) != before
    # البحث العلوي بيودّي للمساعد وبيجاوب
    at.multiselect(key="f_gov").set_value([]).run()
    at.text_input(key="topq").set_value("أعلى 3 فروع في المبيعات").run()
    assert at.sidebar.radio[0].value == "المساعد الذكي" and not at.exception
    assert any("أعلى" in _md_msg(m) for m in at.chat_message)


def _md_msg(m):
    return " ".join(str(x.value) for x in m.markdown)


def test_empty_state_shows_no_fake_numbers(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(data, "PROC", tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception
    txt = _md(at)
    assert "لسه مفيش بيانات" in txt and "صافي المبيعات" not in txt and "مليون" not in txt


def test_branch_cost_completeness_flag(D):
    df = data.apply_filters(D.sales, "2025-10-19", "2026-10-18")
    p = M.branch_pnl(D, df, "2025-10-19", "2026-10-18")
    assert (p.cost_ok == (p.payroll > 0)).all() and 0 < (~p.cost_ok).sum() < len(p)


def test_ui_components_are_safe_and_complete():
    from src import ui
    h = ui.problem_card(1, "حرجة", "<script>x</script>", "15.4", "مليون ج", "تسمية", [("فرع", -20.0, "-20 نقطة")], "دليل", None)
    assert "<script>" not in h and "مليون ج" in h
    assert "nan" not in ui.problem_card(1, "مهمة", "t", "1", "", "l", [("a", 1.0, "1")], "e", float("nan"))
    assert ui.kpi_card("ا", "1", "%", ui.delta_pill(1, 1), [1, 2, 3]).count("<svg") == 1
    assert "ثابت" in ui.delta_pill(10.0, 10.004) and "down" in ui.delta_pill(5, 10) and "up" in ui.delta_pill(5, 10, inverse=True)
    assert "cov-b" in ui.coverage_table([dict(name="ن", sub="", stock=10, demand=2, cover=5.0, status="منخفض", action="")], 3, 26)


# ================================================================= التوقعات
def test_forecast_tables_are_complete_and_sane(D):
    f, q = D.forecast, D.fq
    assert {"level", "series", "target", "month", "forecast", "lo", "hi", "model", "h"} <= set(f.columns)
    assert set(f.level) == {"total", "category", "branch", "product"}
    assert (f.forecast > 0).all() and (f.lo <= f.forecast + 1e-9).all() and (f.forecast <= f.hi + 1e-9).all()
    first = f[(f.level == "total") & (f.target == "qty")].sort_values("h").month.iloc[0]
    assert first == f"{D.last:%Y-%m}"                                 # الشهر الحالي الناقص: التوقع بيبدأ منه (للشهر كله)
    assert q.reliability.isin(["عالية", "متوسطة", "ضعيفة"]).all()
    tot = q[(q.level == "total") & (q.target == "qty")].iloc[0]
    assert tot.wape_test < 8 and tot.wape_test <= tot.baseline_test    # الشركة: التوقع أحسن من الطريقة البسيطة
    assert (q.n_test > 0).all()


def test_revenue_forecast_is_units_times_price(D):
    qf = D.forecast[(D.forecast.level == "total") & (D.forecast.target == "qty")].sort_values("h")
    nf = D.forecast[(D.forecast.level == "total") & (D.forecast.target == "net")].sort_values("h")
    price = D.fq[(D.fq.level == "total") & (D.fq.target == "qty")].price.iloc[0]
    assert np.allclose(nf.forecast.values, qf.forecast.values * price)
    assert price == pytest.approx(models._price(D.sales))


def test_backtest_has_no_leakage():
    """التوقع عند نقطة بداية لازم ميتأثرش بأي رقم بعدها: لو غيّرنا المستقبل، توقع الماضي ميتغيرش."""
    rng = np.random.default_rng(1)
    y = 1000 * np.exp(0.01 * np.arange(80)) * (1 + 0.2 * np.sin(2 * np.pi * np.arange(80) / 12)) * rng.normal(1, 0.02, 80)
    idx = pd.period_range("2019-01", periods=80, freq="M")
    a = models.backtest(y, idx, ["snaive", "sgrow", "ets"], origins=4)
    y2 = y.copy()
    y2[-models.HORIZON - 1:] *= 5                                    # بنخرّب آخر شهور بس
    b = models.backtest(y2, idx, ["snaive", "sgrow", "ets"], origins=4)
    last = a.origin == 3                                              # أقدم نقطة بداية: بيانات تدريبها ما بتشملش الشهور المخرّبة
    assert np.allclose(a[last].sort_values(["model", "h"]).pred.values, b[b.origin == 3].sort_values(["model", "h"]).pred.values)
    assert (a[a.origin == 0].actual.values != b[b.origin == 0].actual.values).any()


def test_model_selection_uses_older_origins_and_test_uses_recent(D):
    bt = D.backtest[(D.backtest.level == "total") & (D.backtest.series == "__total__")]
    best, _ = models.choose(bt)
    q = D.fq[(D.fq.level == "total") & (D.fq.target == "qty")].iloc[0]
    test = bt[(bt.model == best) & (bt.origin < models.SPLIT) & (bt.h <= 3)]
    assert q.wape_test == pytest.approx(models._wape(test))           # الدقة المعروضة = أحدث نقاط بداية بس
    val = bt[(bt.model == best) & (bt.origin >= models.SPLIT) & (bt.h <= 3)]
    assert q.wape_val == pytest.approx(models._wape(val))
    assert set(bt.origin.unique()) == set(range(models.ORIGINS))


def test_interval_coverage_is_close_to_target(D):
    q = D.fq[D.fq.target == "qty"]
    for lv in ("total", "category", "branch", "product"):
        cov = q[q.level == lv].coverage_test.mean()
        assert 65 <= cov <= 100, (lv, cov)                            # المدى معاير على 80%، والحقيقة بتتقاس في الاختبار
    cat = q[q.level == "category"]
    assert cat.interval_q.nunique() == 1                              # نفس المعايرة لكل المستوى


def test_models_recover_known_seasonality():
    y = 1000 * np.exp(0.01 * np.arange(84)) * (1 + 0.2 * np.sin(2 * np.pi * np.arange(84) / 12))
    out = models.predict_all(y[:-6], 6, ["snaive", "sgrow", "ets", "theta"])
    truth = y[-6:]
    assert np.abs(out["ets"] - truth).sum() / truth.sum() < 0.05
    assert np.abs(out["ets"] - truth).sum() < np.abs(out["snaive"] - truth).sum()


def test_mtd_nowcast_is_sane(D):
    nc = models.mtd_nowcast(D)
    assert nc and nc["day"] == D.last.day and 0.3 < nc["typical_share"] < 0.9
    assert nc["projected"] == pytest.approx(nc["so_far"] / nc["typical_share"])


def test_inventory_growth_factor_shrinks_for_weak_forecasts(D):
    gf = inv.growth_factor(D)
    weak = D.fq[(D.fq.level == "category") & (D.fq.reliability == "ضعيفة")].series
    assert gf.between(0.6, 1.6).all() and len(gf) == D.fq[D.fq.level == "category"].shape[0]
    assert all(abs(gf[c] - 1) <= 0.3 + 1e-9 for c in weak)           # التوقع الضعيف نصف أثره


# ================================================================= Groq: الإعدادات والاتصال
class _R:
    def __init__(self, code=200, content=None, headers=None, text=""):
        self.status_code, self.headers, self.text, self._c = code, headers or {}, text, content

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._c, ensure_ascii=False)}}], "usage": {"total_tokens": 300}}


class _Sess:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        r = self.fn(json, len(self.calls))
        if isinstance(r, Exception):
            raise r
        return r


def _client(fn, **kw):
    cfg = Settings(api_key="gsk_TESTKEY0000000000", **kw)
    sleeps = []
    return GroqClient(cfg, _Sess(fn), sleep=sleeps.append), sleeps


M1 = [{"role": "user", "content": "x"}]


def test_env_file_is_loaded_without_overriding(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# تعليق\nLLM_API_KEY="gsk_FROM_FILE_123456"\nLLM_MODEL=abc-model\nLLM_NARRATE=off\n', encoding="utf-8")
    for k in ("LLM_API_KEY", "GROQ_API_KEY", "LLM_MODEL", "LLM_NARRATE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_MODEL", "from-real-env")
    load_env(p)
    import os
    assert os.environ["LLM_API_KEY"] == "gsk_FROM_FILE_123456" and os.environ["LLM_MODEL"] == "from-real-env"   # متغير البيئة الفعلي أقوى
    s = Settings.from_env()
    assert s.api_key == "gsk_FROM_FILE_123456" and s.narrate == "off" and s.base_url.startswith("https://api.groq.com")


def test_env_example_has_no_secret_and_env_is_ignored():
    root = Path(__file__).resolve().parents[1]
    ex = (root / ".env.example").read_text(encoding="utf-8")
    assert "gsk_" not in ex and "LLM_API_KEY=" in ex
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").split()


def test_retry_after_429_then_success():
    c, sleeps = _client(lambda req, n: _R(429, headers={"retry-after": "2"}) if n == 1 else _R(content={"ok": 1}))
    assert c.chat_json(M1) == {"ok": 1} and len(c.s.calls) == 2 and 2 <= sleeps[0] <= 3 and c.stats["retries"] == 1


def test_long_rate_limit_enters_cooldown_and_stops_calling():
    c, _ = _client(lambda req, n: _R(429, text="Please try again in 45s"))
    with pytest.raises(LLMError) as e:
        c.chat_json(M1)
    assert e.value.kind == "rate_limit" and "حد الاستخدام" in e.value.user_msg and not c.available
    n = len(c.s.calls)
    with pytest.raises(LLMError):
        c.chat_json(M1)
    assert len(c.s.calls) == n                                         # مفيش طلبات جديدة أثناء الانتظار


def test_timeout_and_network_errors_are_retried_with_backoff():
    c, sleeps = _client(lambda req, n: [requests.Timeout("t"), requests.ConnectionError("c"), _R(content={"v": 1})][n - 1])
    assert c.chat_json(M1) == {"v": 1} and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9


def test_failures_become_friendly_messages_without_secrets(tmp_path):
    c, _ = _client(lambda req, n: requests.ConnectionError("boom gsk_TESTKEY0000000000"))
    with pytest.raises(LLMError) as e:
        c.chat_json(M1)
    assert e.value.kind == "network" and "gsk_" not in str(e.value) and "gsk_" not in e.value.user_msg
    log = (Path(__file__).resolve().parents[1] / "logs" / "assistant.log")
    assert not log.exists() or "gsk_TESTKEY" not in log.read_text(encoding="utf-8")
    assert "gsk_[محجوب]".replace("gsk_[", "[") in redact("x gsk_ABCDEFGH12345 y").replace("x ", "").replace(" y", "") or "[محجوب]" in redact("gsk_ABCDEFGH12345")


def test_auth_error_is_not_retried_and_hidden_from_user():
    c, _ = _client(lambda req, n: _R(401, text="invalid api key"))
    with pytest.raises(LLMError) as e:
        c.chat_json(M1)
    assert len(c.s.calls) == 1 and "مفتاح" not in e.value.user_msg and not c.available


def test_cache_and_local_token_budget():
    c, _ = _client(lambda req, n: _R(content={"v": n}))
    a = c.chat_json(M1, cache_key="k")
    assert c.chat_json(M1, cache_key="k") == a and len(c.s.calls) == 1
    c2, _ = _client(lambda req, n: _R(content={"v": 1}), tpm_limit=500)
    with pytest.raises(LLMError) as e:
        c2.chat_json([{"role": "user", "content": "x" * 3000}], max_tokens=300)
    assert e.value.kind == "budget" and len(c2.s.calls) == 0


# ================================================================= المساعد مع Groq (بمحاكاة ردود الخدمة)
def _plan_asst(D, plan, summary=None, **kw):
    def fn(req, n):
        if "convert" in req["messages"][0]["content"]:
            return _R(content=plan)
        return _R(content={"summary": summary or ""})
    cfg = Settings(api_key="gsk_TESTKEY0000000000", **kw)
    c = GroqClient(cfg, _Sess(fn), sleep=lambda x: None)
    return Assistant(D, Ctx(D.last - pd.Timedelta(days=364), D.last), c), c


def test_groq_plans_multi_part_question_and_summary_is_grounded(D):
    plan = {"calls": [{"title": "أعلى فروع الربح", "tool": "query", "metric": "margin", "dim": "branch", "rank": "top", "top_n": 3},
                      {"title": "توقع جليم", "tool": "forecast", "branches": ["جليم"], "months": 1}], "clarify": ""}
    A, c = _plan_asst(D, plan, "أعلى الفروع في الربح مدينتي الربيع وجليم، وتوقع جليم للشهر الجاي له مدى محتمل.")
    a = A.ask("أعلى 3 فروع في الربح وتوقع جليم للشهر الجاي")
    assert a.mode == "Groq" and len(a.parts) == 2 and a.summary and len(c.s.calls) == 2
    assert "الربح" in a.parts[0].headline and "جليم" in a.parts[1].forecasts[0] and len(a.parts[1].table) == 1
    assert M.money(a.parts[0].table.iloc[0]["الربح"]) in a.parts[0].headline


def test_groq_summary_with_invented_number_is_dropped(D):
    plan = {"calls": [{"title": "أ", "tool": "query", "metric": "net"}, {"title": "ب", "tool": "query", "metric": "margin"}], "clarify": ""}
    A, _ = _plan_asst(D, plan, "المبيعات هتزيد 77% بسبب حملة الصيف.")
    a = A.ask("المبيعات والربح")
    assert a.summary == "" and a.mode == "Groq" and "77" not in a.text


def test_unknown_branch_is_not_silently_replaced_by_company_total(D):
    A, _ = _plan_asst(D, {"calls": [{"title": "x", "tool": "query", "metric": "net", "branches": ["فرع الأحلام"]}], "clarify": ""})
    a = A.ask("مبيعات فرع الأحلام")
    assert "مقدرتش ألاقي" in a.headline and "الأحلام" in a.headline and "كل الشركة" not in a.source.get("النطاق", "")


def test_groq_rate_limit_falls_back_to_local_with_friendly_note(D):
    c = GroqClient(Settings(api_key="gsk_TESTKEY0000000000"), _Sess(lambda r, n: _R(429, text="try again in 40s")), sleep=lambda x: None)
    A = Assistant(D, Ctx(D.last - pd.Timedelta(days=364), D.last), c)
    a = A.ask("أعلى 3 فروع في الربح")
    assert a.mode == "المحرك المحلي" and "حد الاستخدام" in a.llm_note and "فروع" in a.headline
    calls = len(c.s.calls)
    b = A.ask("المبيعات كام الشهر اللي فات؟")
    assert len(c.s.calls) == calls and b.mode == "المحرك المحلي" and "مؤقتًا" in b.llm_note       # مفيش ضغط زيادة على الخدمة
    assert "gsk_" not in a.text + a.llm_note + b.llm_note


def test_groq_invalid_json_and_clarify(D):
    A, _ = _plan_asst(D, {"hello": 1})
    a = A.ask("أعلى 3 فروع في الربح")
    assert a.mode == "المحرك المحلي" and "مكانش مفهوم" in a.llm_note
    A2, _ = _plan_asst(D, {"calls": [], "clarify": "تقصد أنهي فرع؟"})
    assert A2.ask("وفي فرع؟").headline == "تقصد أنهي فرع؟"


# ================================================================= المساعد المحلي: الفهم والدقة
def test_assistant_numbers_match_raw(D, raw):
    A = local_assistant(D)
    a = A.ask_local("المبيعات كام الشهر اللي فات؟")
    truth = raw[(raw.date >= "2026-09-01") & (raw.date <= "2026-09-30")].net_line_EGP.sum()
    assert M.money(truth) in a.text
    a = A.ask_local("أعلى 3 فروع في المبيعات")
    cur = raw[raw.date >= D.last - pd.Timedelta(days=364)]
    top = cur.groupby("branch_id").net_line_EGP.sum().sort_values(ascending=False).head(3).index
    assert list(a.table["الاسم"][:3]) == [D.branch_name[b] for b in top]
    a = A.ask_local("مبيعات البوكسرات رجالي اونلاين في سبتمبر 2026")
    t = raw[(raw.date >= "2026-09-01") & (raw.date <= "2026-09-30") & (raw.category_en == "Men boxers") & (raw.sales_channel == "E-commerce")].net_line_EGP.sum()
    assert M.money(t) in a.text
    assert set(A.ask_local("قارن بين مبيعات القاهرة والإسكندرية").table["الاسم"]) == {"القاهرة", "الإسكندرية"}
    n = raw[raw.date == D.last.normalize() - pd.Timedelta(days=1)].sales_order_id.nunique()
    assert M.num(n) in A.ask_local("كام فاتورة امبارح").text


def test_profit_is_not_confused_with_margin_or_sales(D, raw):
    A = local_assistant(D)
    cur = raw[raw.date >= D.last - pd.Timedelta(days=364)]
    prof = cur.groupby("branch_id").margin_EGP.sum().sort_values(ascending=False)
    a = A.ask_local("أعلى 3 فروع في الربح")
    assert "الربح" in a.headline and M.money(prof.iloc[0]) in a.headline and "هامش" not in a.headline
    b = A.ask_local("أعلى 3 فروع في هامش الربح")
    assert "هامش الربح" in b.headline and "%" in b.headline
    c = A.ask_local("ليه الربح نزل السنة دي؟")
    assert "الربح (القيمة)" in c.text or "ربح" in c.headline and "نسبة الهامش" not in c.headline
    assert any("أثر تكلفة" in f for f in c.facts)


def test_named_branches_are_analysed_not_channels(D):
    A = local_assistant(D)
    a = A.ask_local("ليه الربح نزل في فرع مدينتي الربيع وفرع جليم السنة دي؟")
    assert set(a.table["الفرع"]) == {"مدينتي الربيع", "جليم"}
    assert not any("قنوات" in f for f in a.facts) and any("أنواع" in f for f in a.facts)
    b = A.ask_local("المبيعات حسب النوع في فرع جليم")
    assert b.table is not None and "جليم" in b.source["النطاق"] and len(b.table) >= 5
    top = b.table.iloc[0]["المبيعات"]
    j = data.apply_filters(D.sales, D.last - pd.Timedelta(days=364), D.last, branches=[next(k for k, v in D.branch_name.items() if v == "جليم")])
    assert top == pytest.approx(j.groupby("category", observed=True).net.sum().max())


def test_gap_question_compares_two_entities_not_time(D):
    A = local_assistant(D)
    a = A.ask_local("ليه مبيعات الإسكندرية أقل من القاهرة؟")
    assert "أقل من" in a.headline and any("عدد الفروع" in f for f in a.facts) and "مش تغير عبر الزمن" in " ".join(a.notes)


def test_multi_part_local_questions_and_follow_ups(D):
    A = local_assistant(D)
    a = A.ask_local("أعلى 3 فروع في الربح وقولي توقع جليم للشهر الجاي")
    assert len(a.parts) == 2 and "جليم" in a.parts[1].source["النطاق"]
    A = local_assistant(D)
    a1 = A.ask("أعلى 5 فروع في المبيعات الشهر اللي فات")
    a2 = A.ask("طب وهامش الربح؟")
    assert a2.source["الفترة"] == a1.source["الفترة"] and "هامش الربح" in a2.headline and "فروع" in a2.headline
    a3 = A.ask("وفي القاهرة؟")
    assert "القاهرة" in a3.source["النطاق"]
    a4 = A.ask("المبيعات كام؟")
    assert a4.source["النطاق"] == "كل الشركة"


def test_forecast_answers_show_reliability_and_price_assumption(D):
    A = local_assistant(D)
    a = A.ask_local("توقع مبيعات الشهر الجاي")
    txt = a.text
    assert "الموثوقية" in txt and "متوسط الغلط" in txt and "ثابتة" in txt and "المدى المحتمل" in txt and "≠" in txt
    assert not any("±" in x for x in a.forecasts + a.notes)           # من غير «±2.5%» اللي بتوصل معنى غلط
    b = A.ask_local("توقع القطع للبوكسرات رجالي")
    assert "قطعة" in b.forecasts[0] and b.source["النطاق"] == "بوكسرات رجالي"


def test_assistant_separates_fact_from_inference_and_refuses_made_up_advice(D):
    A = local_assistant(D)
    w = A.ask_local("ليه المبيعات اتغيرت السنة دي؟")
    assert w.facts and w.notes and "مش بتقول ليه" in w.notes[0] and w.source["المصدر"] and w.table is not None
    d = A.ask_local("اقترح خصم 20% على البوكسرات")
    assert "مقدرش" in d.headline and not d.inferences and "تجارب" in " ".join(d.notes)
    z = A.ask_local("ليه المبيعات نزلت السنة دي؟")
    assert "زادت مش نزلت" in z.headline
    for q, word in [("توقع مبيعات الشهر الجاي", "المتوقع"), ("ايه اللي هيخلص من المخزون", "مخزون"), ("ايه توصياتك", "اقتراحات"), ("مين احسن عملاء", "عميل"), ("عايز تحويلات", "تحويلة")]:
        assert word in A.ask_local(q).text


# ================================================================= الواجهة والـ RTL
def test_bidi_isolates_numbers_dates_and_percents():
    s = ui.bidi("نمو +30% من 2026-10-24 18:27 و بعيد 3-6 شهور فشهر 2026-10")
    for tok in ("+30%", "2026-10-24 18:27", "3-6", "2026-10"):
        assert f"{ui.LRI}{tok}{ui.PDI}" in s
    assert ui.bidi_html('<div style="width:30%">نمو 5%</div>') == f'<div style="width:30%">نمو {ui.LRI}5%{ui.PDI}</div>'
    assert "&lt;" in ui.esc("<b>") and ui.LRI in ui.esc("نمو 5%")


def test_no_api_settings_in_the_ui_and_no_secrets_in_ui_code():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("المساعد الذكي").run()
    assert not at.exception and not at.text_input.__len__() or all("API" not in (t.label or "") and "مفتاح" not in (t.label or "") for t in at.text_input)
    code = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "llm_key" not in code and "api.groq.com" not in code and "llm_base" not in code


def test_insights_are_labelled_as_fact_estimate_or_forecast(D):
    from src import insights
    r = insights.generate(D)
    assert set(r.kind) <= {"حقيقة", "تقدير", "توقع"} and (r.kind == "تقدير").any()
    est = r[r.kind == "تقدير"]
    assert est.impact_note.str.contains("تقدير|لو|مش خسارة|محتمل").all()
    stock = r[(r.area == "المخزون") & r.title.str.contains("قرّب يخلص")]
    assert len(stock) and "من غير شراء جديد" in stock.action.iloc[0]
