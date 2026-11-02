"""Offline assistant contracts. No training, network, .env or production writes."""
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import requests

from src.assistant import Assistant, Answer, Ctx, result_count
from src.llm import GroqClient, Settings, LLMError
from src import data, ui
from src.health import analytical_effect, CHECK_EFFECTS


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden in assistant regression tests")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)


@pytest.fixture
def D():
    branches = pd.DataFrame([
        {"branch_id": f"B{i}", "name": name, "label": name, "governorate": gov}
        for i, (name, gov) in enumerate([("جليم", "الإسكندرية"), ("الرحاب", "القاهرة"),
                                       ("الدقي", "الجيزة"), ("مدينة نصر", "القاهرة")])])
    rows = []
    for m in pd.period_range("2025-01", "2026-10", freq="M"):
        for i in range(4):
            for j, cat in enumerate(["Men boxers", "Socks"]):
                net = float(1000 * (i + 1) * (j + 1))
                margin = net * .3
                if str(m) == "2026-10":
                    net *= .925
                    margin *= .936
                rows.append(dict(date=m.start_time + pd.Timedelta(days=4), branch=f"B{i}", category=cat,
                                 product=f"P{j}", channel="Retail POS", customer=None if i == 3 else f"C{i}",
                                 order=f"{m}-{i}-{j}", qty=100, net=net, margin=margin, gross=net / .9, disc=net / 9))
    sales = pd.DataFrame(rows)
    series = [("total", "__total__")] + [("branch", b) for b in branches.branch_id] + [("category", c) for c in sales.category.unique()]
    forecast = pd.DataFrame([dict(level=level, series=key, target="qty", month=str(m), h=h,
                                 forecast=1000., lo=800., hi=1200., model="ets")
                             for level, key in series for h, m in enumerate(pd.period_range("2026-11", periods=6, freq="M"), 1)])
    fq = pd.DataFrame([dict(level=level, series=key, target="qty", model="ets", wape_test=10., baseline_test=15.,
                            wape_h46=20., coverage_test=78., reliability="متوسطة", note="", price=10.) for level, key in series])
    stock = pd.DataFrame([dict(kind="Branch", location=f"B{i}", category=c, year_month="2026-10",
                               closing_units=3 + i, closing_stock_value_EGP=30., stockout_risk_score=.9)
                          for i in range(4) for c in sales.category.unique()])
    return SimpleNamespace(last=pd.Timestamp("2026-10-30"), sales=sales, branches=branches,
        branch_name=branches.set_index("branch_id")["name"].to_dict(),
        branch_gov=branches.set_index("branch_id").governorate.to_dict(),
        cat_ar={"Men boxers": "بوكسرات رجالي", "Socks": "شرابات"}, prod_name={"P0": "بوكسر", "P1": "شراب"},
        forecast=forecast, fq=fq, stock=stock, churn=pd.DataFrame(), customers=pd.DataFrame(), returns=pd.DataFrame(),
        model_metrics={"churn": {"auc": .51, "auc_baseline": .50, "predictive": False, "base_rate": .2, "window_days": 120}},
        health=[dict(check="ids", status="ok", detail="المعرفات سليمة"), dict(check="missing", status="warn", detail="عميل ناقص")])


@pytest.fixture
def assistant(D):
    a = Assistant(D, Ctx(pd.Timestamp("2026-10-01"), D.last), GroqClient(Settings(api_key="", enabled=False)))
    a._ins = pd.DataFrame([dict(priority=p, area="الفروع", title=f"مشكلة {i}", evidence="دليل محسوب",
                               action="راجع الفرع", impact=value, impact_note="آخر 12 شهر", kind=kind)
                           for i, (p, value, kind) in enumerate([
                               ("حرجة", 100., "تقدير"), ("مهمة", 50., "حقيقة"),
                               ("مهمة", np.nan, "تقدير"), ("معلومة", np.nan, "حقيقة")])])
    return a


@pytest.mark.parametrize("question,intents", [
    ("المبيعات كام في سبتمبر 2026؟", ["query"]),
    ("الربح وهامش الربح", ["query", "query"]),
    ("أعلى 3 فروع في الربح", ["query"]),
    ("ليه المبيعات انخفضت؟", ["why"]),
    ("أكتر 3 فروع عندها نقص مخزون، والمنتجات المتأثرة وأسابيع التغطية والتوصية لكل فرع", ["inventory"]),
    ("توقعات نوفمبر وديسمبر، ودقة النموذج، وهل ينفع نعتمد عليها في المخزون والتوريد؟", ["forecast"]),
    ("حلل العملاء آخر 12 شهر، ونسبة المبيعات بدون بيانات عميل، ومساهمة أكبر 10% وموثوقية Churn", ["customers"]),
    ("أهم 3 مشاكل محتاجة تدخل عاجل، وتأثيرها المالي والإجراء المناسب", ["recommend"]),
    ("راجع صحة البيانات", ["health"]),
    ("أهم 3 مشاكل وبعدين راجع صحة البيانات", ["recommend", "health"]),
    ("توقعات نوفمبر وديسمبر وحلل العملاء آخر 12 شهر", ["forecast", "customers"]),
    ("أعلى 3 فروع في الربح وتوقع جليم للشهر الجاي", ["query", "forecast"]),
    ("المبيعات والربح، بدون هامش الربح وبدون توقعات", ["query", "query"]),
    ("هل انخفاض الربح أكبر من انخفاض المبيعات؟", ["metric_comparison"]),
])
def test_plan_coverage(assistant, question, intents):
    specs, _ = assistant.response_specs(question)
    assert [s["intent"] for s in specs] == intents


def test_forecast_to_customers_to_health(assistant):
    a = assistant.ask("توقعات نوفمبر وديسمبر، ودقة النموذج وهل نعتمد عليها في المخزون؟")
    assert set(a.table["الشهر"]) == {"2026-11", "2026-12"}
    assert "دقة إجمالي الشركة" in a.text and "مدة التوريد" in a.text
    b = assistant.ask("حلل العملاء آخر 12 شهر وموثوقية النموذج")
    assert assistant.state["intent_final"] == "customers"
    assert "AUC=0.510" in b.text and "40.00%" in b.text and "3 عميل" in b.text
    c = assistant.ask("أهم 3 مشاكل محتاجة تدخل عاجل وتأثيرها المالي والإجراء")
    assert len(c.table) == 3 and "غير مقدّر ماليًا" in c.text
    d = assistant.ask("وصحة البيانات؟")
    assert assistant.state["intent_final"] == "health" and "2 فحص" in d.text


def test_followup_scope_and_reset(assistant):
    a = assistant.ask("مبيعات القاهرة في سبتمبر 2026")
    b = assistant.ask("والربح؟")
    assert b.source["النطاق"] == "القاهرة" and b.source["الفترة"] == a.source["الفترة"]
    c = assistant.ask("وفي الإسكندرية؟")
    assert c.source["النطاق"] == "الإسكندرية"
    d = assistant.ask("حلل العملاء آخر 12 شهر")
    assert d.source["النطاق"] == "كل الشركة"
    assert assistant.state["start"] == pd.Timestamp("2025-10-31")


def test_compound_scope_and_no_extra_tools(assistant, monkeypatch):
    calls = []
    original = assistant.dispatch
    def dispatch(spec):
        calls.append(spec["intent_final"])
        return original(spec)
    monkeypatch.setattr(assistant, "dispatch", dispatch)
    a = assistant.ask("مبيعات القاهرة سبتمبر 2026 والربح، وحلل العملاء آخر 12 شهر")
    assert calls == ["query", "query", "customers"]
    assert len(a.parts) == 3
    assert a.parts[0].source["النطاق"] == a.parts[1].source["النطاق"] == "القاهرة"
    assert a.parts[2].source["النطاق"] == "كل الشركة"


def test_declines_calculated_not_displayed_only(assistant):
    assistant.ctx = Ctx(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-31"))
    a = assistant.ask("هل انخفاض الربح أكبر من انخفاض المبيعات؟")
    assert a.headline.startswith("لا") and "1.10 نقطة مئوية" in a.headline
    assert "-7.50%" in a.text and "-6.40%" in a.text


def test_inventory_distinct_branches(assistant):
    a = assistant.ask("أكتر 3 فروع عندها نقص مخزون والمنتجات المتأثرة وأسابيع التغطية والتوصية لكل فرع")
    assert len(a.facts) == 3
    assert len({f.split(" — ")[0] for f in a.facts}) == 3
    assert all("طلب تقديري" in f and "التوصية" in f for f in a.facts)
    assert "SKU" in a.text


def test_negation_and_profit_margin(assistant):
    a = assistant.ask("المبيعات والربح بدون هامش الربح وبدون توقعات")
    assert len(a.parts) == 2 and not a.forecasts
    assert a.parts[0].source["المقياس"] == "المبيعات"
    assert a.parts[1].source["المقياس"] == "الربح"
    assert "هامش الربح" not in a.text


def test_unknown_branch_does_not_broaden(assistant):
    state = assistant.state.copy()
    a = assistant.ask("مبيعات فرع الأحلام")
    assert "مقدرتش ألاقي" in a.text
    assert assistant.state == state


def test_empty_customer_data_still_explains_churn(assistant):
    assistant.D.sales = assistant.D.sales.iloc[:0]
    a = assistant.ask("حلل العملاء آخر 12 شهر")
    assert "0 عميل" in a.text and "AUC=" in a.text and "غير قابلة للحساب" in a.text


def test_no_data_mutation(assistant):
    original = assistant.D.sales.copy(deep=True)
    assistant.ask("مبيعات القاهرة والربح")
    assistant.ask("حلل العملاء آخر 12 شهر")
    pd.testing.assert_frame_equal(original, assistant.D.sales)


def test_safe_ui_bidi():
    assert ui.bidi(ui.bidi("فرق -7.50% في 2026-10")) == ui.bidi("فرق -7.50% في 2026-10")
    assert "&lt;script&gt;" in ui.esc("<script>").replace(ui.LRI, "").replace(ui.PDI, "")


def test_table_complete_precise_and_escaped():
    frame = pd.DataFrame({"مبلغ (جنيه)": [1234.56] * 15, "اسم": ["<img src=x onerror=alert(1)>"] * 15})
    rendered = ui.dataframe_html(frame)
    assert rendered.count("<tr>") == 16 and 'dir="rtl"' in rendered
    assert "1,234.56" in rendered and "<img" not in rendered
    assert "معروض" in ui.dataframe_html(frame, 3)


def test_actual_answer_renderer_does_not_hide_requested_results():
    # Exercise the production renderer without booting all Streamlit pages/data.
    import ast
    app = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    renderer = next(n for n in app.body if isinstance(n, ast.FunctionDef) and n.name == "render_answer")
    rendered = []
    namespace = {"H": rendered.append, "esc": ui.esc, "ui": ui, "df_html": ui.dataframe_html}
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), "app.py", "exec"), namespace)
    answer = Answer.make("نتائج", facts=[f"دليل {i}" for i in range(10)],
                         table=pd.DataFrame({"مبلغ": [1234.56] * 15}))
    namespace["render_answer"](answer)
    markup = "".join(rendered)
    assert markup.count("<li>") == 10 and markup.count("<tr>") == 16
    assert "1,234.56" in markup and 'dir="rtl"' in markup


@pytest.mark.parametrize("first,second,intent", [
    ("أهم 3 مشاكل", "صحة البيانات", "health"),
    ("توقعات نوفمبر وديسمبر", "تحليل العملاء آخر 12 شهر", "customers"),
    ("تحليل العملاء", "المبيعات في سبتمبر 2026", "query"),
    ("المبيعات كام", "أهم 3 تدخلات إدارية", "recommend"),
    ("أهم 3 مشاكل", "هامش الربح كام", "query"),
])
def test_topic_switch_matrix(assistant, first, second, intent):
    assistant.ask(first)
    assistant.ask(second)
    assert assistant.state["intent_final"] == intent


def test_health_not_recommendation_or_forecast(assistant):
    assistant.ask("أهم 3 مشاكل")
    a = assistant.ask("مشاكل جودة البيانات ودقتها")
    assert assistant.state["intent_final"] == "health" and a.source["المصدر"] == "فحوصات data.py"


def test_explicit_period_comparison(assistant):
    a = assistant.ask("قارن مبيعات سبتمبر وأكتوبر 2026")
    assert assistant.state["start"] == pd.Timestamp("2026-10-01")
    assert "-7.50%" in a.text and "2026-09-30" in a.text


def test_entity_comparison_honours_category(assistant):
    a = assistant.ask("قارن مبيعات القاهرة والإسكندرية للبوكسرات رجالي سبتمبر 2026")
    assert set(a.table["الاسم"]) == {"القاهرة", "الإسكندرية"}
    assert sorted(a.table["المبيعات"]) == [1000., 6000.]


def test_profit_entity_comparison_not_sales(assistant):
    a = assistant.ask("ليه ربح الإسكندرية أقل من القاهرة في سبتمبر 2026؟")
    assert "الربح" in a.headline and "900" in a.text and "5,400" in a.text


def test_explicit_scope_reference_cross_topic(assistant):
    a = assistant.ask("مبيعات القاهرة في سبتمبر 2026")
    b = assistant.ask("حلل العملاء لنفس الفترة ونفس المحافظة")
    assert b.source["النطاق"] == "القاهرة" and b.source["الفترة"] == a.source["الفترة"]
    assert assistant.state["intent_final"] == "customers"


def test_duplicate_requests_once(assistant):
    a = assistant.ask("المبيعات كام؟ المبيعات كام؟")
    assert not a.parts


def test_excluded_only_no_default_sales(assistant):
    a = assistant.ask("بدون توقعات وبدون مبيعات")
    assert "مفيش طلب" in a.text and not assistant.state


class Response:
    def __init__(self, status=200, content=None, headers=None):
        self.status_code, self.headers, self.text = status, headers or {}, "redacted test response"
        self.content = content
    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.content)}, "finish_reason": "stop"}], "usage": {"total_tokens": 10}}


def synthesis_output():
    return {"analysis": [{"text": "قد يرتبط تغير البيع بالتوافر والخصومات، لكن اللقطة الحالية لا تثبت السبب؛ نحتاج مقارنة تاريخية متزامنة.",
                          "evidence": ["sales", "inventory", "discount"]}],
            "steps": [{"text": "راجع سجل نفاد الأصناف في الفروع المتراجعة وتحقق من الرصيد الفعلي قبل التوريد.", "evidence": ["sales", "inventory"]},
                      {"text": "اختبر الخصم في فروع متشابهة وقارن المبيعات والهامش قبل التوسع.", "evidence": ["sales", "discount"]},
                      {"text": "راجع النتائج بعد التحقق من توافر الأصناف وحدد هل يستمر التدخل.", "evidence": ["inventory", "discount"]}]}


def test_cross_domain_synthesis_receives_actual_evidence(assistant, monkeypatch):
    session = configure_mock(assistant, [Response(content=synthesis_output()), Response(content=synthesis_output())])
    monkeypatch.setattr(assistant, "h_recommend", lambda sp: pytest.fail("No canned recommendations"))
    answer = assistant.ask("اربط انخفاض المبيعات بالمخزون والخصومات في القاهرة في سبتمبر 2026 واقترح خطة من 3 خطوات")
    assert not answer.parts and len(answer.actions) == 3 and len(answer.inferences) == 1
    assert answer.mode.startswith("Groq") and len(session.calls) == 2
    packet = json.loads(session.calls[0]["messages"][1]["content"])
    evidence = {e["id"]: e for e in packet["evidence"]}
    assert set(evidence) == {"sales", "inventory", "discount"}
    assert evidence["sales"]["source"]["النطاق"] == "القاهرة"
    assert evidence["discount"]["source"]["النطاق"] == "القاهرة"
    assert evidence["discount"]["source"]["الفترة"] == "سبتمبر 2026"
    assert evidence["inventory"]["table"]
    assert "طلب تقديري" in answer.text
    assert assistant.ask_local("راجع صحة البيانات").source
    assert assistant.state["intent_final"] == "health"


@pytest.mark.parametrize("mutation", ["number", "refs", "count", "causal", "omission", "duplicate"])
def test_reject_ungrounded_synthesis(assistant, mutation):
    out = synthesis_output()
    if mutation == "number":
        out["steps"][0]["text"] += " وسيكون العائد 987654321 جنيه"
    elif mutation == "refs":
        out["analysis"][0]["evidence"] = ["unknown"]
    elif mutation == "count":
        out["steps"].pop()
    elif mutation == "causal":
        out["analysis"][0]["text"] = "انخفضت المبيعات بسبب المخزون بالتأكيد"
    elif mutation == "omission":
        for item in out["analysis"] + out["steps"]:
            item["evidence"] = ["sales", "discount"]
    else:
        out["steps"][1] = out["steps"][0].copy()
    configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات واقترح خطة من 3 خطوات")
    if mutation == "number":
        assert answer.mode.startswith("Groq") and len(answer.actions) == 3
        assert "987654321" not in answer.text and "بديل محلي" in answer.actions[0]
    else:
        assert answer.mode == "المحرك المحلي" and not answer.actions and not answer.inferences
        assert "لم يكتمل" in answer.llm_note and answer.facts


@pytest.mark.parametrize("result,reason", [(requests.Timeout(), "بطيئة"), (Response(401), "رفضت")])
def test_synthesis_transport_failure(assistant, result, reason):
    configure_mock(assistant, [result])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات واقترح خطة من 3 خطوات")
    assert reason in answer.llm_note and "أدلة محلية فقط" in answer.llm_note
    assert not answer.actions and answer.mode == "المحرك المحلي"


@pytest.mark.parametrize("question,domains", [
    ("اربط الربح بالخصومات وخطة من خطوتين", ["profit", "discount"]),
    ("ما علاقة العملاء بالمبيعات؟", ["sales", "customers"]),
    ("اربط المبيعات بالمخزون بدون توقعات", ["sales", "inventory"]),
    ("اربط المبيعات بالمخزون بدون خصومات", ["sales", "inventory"]),
    ("اعرض المبيعات والمخزون", []),
    ("جودة البيانات وتأثيرها على المبيعات", []),
])
def test_relationship_planning_not_keyword_specific(question, domains):
    assert Assistant.analysis_domains(question) == domains


def test_local_synthesis_is_honest(assistant):
    answer = assistant.ask_local("اربط المبيعات بالمخزون والخصومات واقترح خطة من 3 خطوات")
    assert not answer.parts and not answer.actions
    assert "ليس تحليلاً سببياً أو خطة مكتملة" in answer.llm_note


def test_synthesis_two_steps_and_followup_scope(assistant):
    assistant.ask_local("مبيعات القاهرة في سبتمبر 2026")
    out = synthesis_output()
    out["steps"] = out["steps"][:2]
    session = configure_mock(assistant, [Response(content=out), Response(content=out)])
    question = "اربط المبيعات بالمخزون والخصومات لنفس الفترة ونفس المحافظة واقترح خطة من خطوتين"
    assert len(assistant.split_parts(question)) == 1
    answer = assistant.ask(question)
    assert len(answer.actions) == 2 and len(session.calls) == 2
    assert assistant.state["ent"]["governorates"] == ["القاهرة"]
    assert assistant.state["start"] == pd.Timestamp("2026-09-01")


@pytest.mark.parametrize("branch_count,step_count", [(3, 3), (2, 3), (3, 2)])
def test_rank_branches_before_collecting_each_domain(assistant, branch_count, step_count):
    current = assistant.D.sales.date.dt.to_period("M") == pd.Period("2026-10")
    for branch, factor in zip(["B0", "B1", "B2", "B3"], [.9, .7, .5, .3]):
        assistant.D.sales.loc[current & assistant.D.sales.branch.eq(branch), "net"] *= factor
    out = synthesis_output()
    out["steps"] = out["steps"][:step_count]
    session = configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask(f"حلل {branch_count} فروع ضعيفة واربط مبيعاتها بالمخزون والخصومات وخطة من {step_count} خطوات")
    assert answer.mode.startswith("Groq"), answer.llm_note
    assert len(answer.actions) == step_count and not answer.parts and len(session.calls) == 2
    packet = json.loads(session.calls[0]["messages"][1]["content"])
    expected = ["B3", "B2", "B1"][:branch_count]
    assert assistant.state["ent"]["branches"] == expected
    assert "نسبة تغير" in packet["selection"]
    for item in packet["evidence"]:
        assert [r["branch_id"] for r in item["branches"]] == expected
        for record in item["branches"]:
            assert record["branch"] in record["source"]["النطاق"]
            assert "كل الشركة" not in record["source"]["النطاق"]
            if item["id"] == "inventory":
                assert all(row["الفرع"] == record["branch"] for row in record["table"])
    assert all(assistant.D.branch_name[b] in answer.text for b in expected)


def test_branch_selection_honors_governorate_and_shortfall(assistant):
    answer = assistant.ask_local("حلل 3 فروع ضعيفة في القاهرة واربط المبيعات بالمخزون والخصومات")
    assert set(assistant.state["ent"]["branches"]) == {"B1", "B3"}
    assert "المتاح ببيانات كافية 2" in answer.text
    assert "جليم" not in answer.text and "الدقي" not in answer.text


@pytest.mark.parametrize("wording", ["لا يمكن القول إن المبيعات انخفضت بسبب المخزون", "قد يكون التغير بسبب التوافر ونحتاج تحققاً"])
def test_synthesis_allows_explicit_causal_uncertainty_and_numbered_steps(assistant, wording):
    out = synthesis_output()
    out["analysis"] = {"text": wording, "evidence": ["sales", "inventory", "discount"]}
    for index, item in enumerate(out["steps"], 1):
        item["text"] = f"{index}. " + item["text"]
    configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert answer.mode.startswith("Groq") and len(answer.actions) == 3


def test_synthesis_rejection_logged_without_response_or_credentials(assistant, caplog):
    out = synthesis_output()
    out["steps"][0]["text"] = "secret-response-token عائد 987654321 جنيه"
    configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert "synthesis:quantitative_prose_use_computed_facts" in caplog.text
    assert "secret-response-token" not in caplog.text and "test-only" not in caplog.text
    assert "secret-response-token" not in answer.text and "بديل محلي" in answer.actions[0]


@pytest.mark.parametrize("claim,evidence,accepted", [
    ("المخزون بقيمة 648000 جنيه", "بقيمة 648 ألف ج", True),
    ("المبيعات حوالي 3.47 مليون جنيه", "المبيعات 3,474,578.08 جنيه", True),
    ("النمو ١٢٫٨٧٪", "النمو +12.87%", True),
    ("مبيعات ٣٬٤٧٤٬٥٧٨٫٠٨ جنيه", "مبيعات 3,474,578.08 جنيه", True),
    ("حوالي 13%", "12.87%", True),
    ("13%", "12.87%", False),
    ("حوالي 4 مليون جنيه", "3,474,578.08 جنيه", False),
    ("648 مليون جنيه", "648 ألف ج", False),
    ("26%", "26 أسبوع", False),
    ("-12.87%", "+12.87%", False),
    ("عائد 987654321 جنيه", "مبيعات 30000 جنيه", False),
])
def test_synthesis_numeric_formatting_is_unit_aware(claim, evidence, accepted):
    assert (not Assistant.unsupported_synthesis_numbers(claim, [evidence])) == accepted


def test_synthesis_repairs_numeric_prose_without_changing_computed_facts(assistant):
    out = synthesis_output()
    out["analysis"][0]["text"] = "المبيعات الحالية حوالي 28 ألف جنيه؛ قد يرتبط التغير بالتوافر، لكننا نحتاج تحققًا تاريخيًا."
    session = configure_mock(assistant, [Response(content=out), Response(content=synthesis_output())])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert answer.mode.startswith("Groq"), answer.llm_note
    assert len(answer.actions) == 3 and len(session.calls) == 2
    assert "27,750.00" in answer.text


@pytest.mark.parametrize("text", ["حدد ثلاثة أسابيع للتغطية", "نفذ خصماً خمسة بالمائة", "متوسط التغطية أسبوع واحد", "التغطية أسبوعين"])
def test_written_numbers_cannot_bypass_synthesis_contract(assistant, text):
    out = synthesis_output()
    out["steps"][0]["text"] = text
    configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert answer.mode.startswith("Groq") and text not in answer.text
    assert "بديل محلي" in answer.actions[0]


def test_reviewed_answer_replaces_draft_and_keeps_branch_claims_local(assistant):
    draft = synthesis_output()
    draft["analysis"][0]["text"] = "جليم يمتلك أعلى نمو مبيعات"
    reviewed = synthesis_output()
    session = configure_mock(assistant, [Response(content=draft), Response(content=reviewed)])
    answer = assistant.ask("حلل 3 فروع ضعيفة واربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert answer.mode.startswith("Groq") and len(session.calls) == 2
    assert "جليم يمتلك" not in " ".join(answer.inferences)
    assert len(answer.table) == 3 and "أنواع حرجة" in answer.table
    assert "موجب" not in answer.text  # synthetic current period actually declined


def test_failed_review_never_publishes_unreviewed_draft(assistant):
    configure_mock(assistant, [Response(content=synthesis_output()), requests.Timeout()])
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert not answer.actions and not answer.inferences and "بطيئة" in answer.llm_note


def test_narration_off_does_not_call_provider(assistant):
    session = configure_mock(assistant, [], narrate="off")
    answer = assistant.ask("اربط المبيعات بالمخزون والخصومات وخطة من 3 خطوات")
    assert not session.calls and answer.mode == "المحرك المحلي"


MANAGER_BRANCH_REQUEST = (
    "أنا مدير Cottonil وعندي ميزانية محدودة لتحسين أداء الفروع. "
    "اختار أضعف فرعين من حيث نمو المبيعات خلال آخر 12 شهر، "
    "وقارن أداء كل فرع بمخزونه وخصوماته وهامش ربحه. "
    "حدد المشكلة الأهم في كل فرع بناءً على الأدلة، ووضح إيه المؤكد وإيه مجرد احتمال. "
    "في النهاية اقترح خطتين عمليتين مختلفتين، واحدة لكل فرع، وحدد أولوية التنفيذ وليه. "
    "استخدم أرقام فعلية ومصادرها، ومش عايز مجرد عرض جداول."
)


@pytest.mark.parametrize("question", [MANAGER_BRANCH_REQUEST,
    "اختر أقل فرعين في نمو المبيعات وقارن مخزون كل فرع وخصوماته وهامش ربحه ثم اقترح خطة لكل فرع",
    "حدد أضعف 2 فروع في نمو المبيعات. حلل أداء كل فرع ومخزونه وخصوماته. عايز خطتين واحدة لكل فرع",
])
def test_entity_workflow_keeps_selector_scope_and_branch_plans(assistant, monkeypatch, question):
    # Make sales-growth ranking disagree with both absolute sales and profit ranking.
    current = assistant.D.sales.date.dt.to_period("M") == pd.Period("2026-10")
    for branch, factor in zip(["B0", "B1", "B2", "B3"], [.9, .8, .7, .6]):
        mask = current & assistant.D.sales.branch.eq(branch)
        assistant.D.sales.loc[mask, "net"] *= factor
        assistant.D.sales.loc[mask, "margin"] *= 1 - factor
    assistant._inv = assistant.inv_df.copy()
    assistant._inv.loc[assistant._inv.branch.eq("B2"), "status"] = "زيادة"
    monkeypatch.setattr(assistant, "h_recommend", lambda sp: pytest.fail("Intro is not a recommendations request"))
    answer = assistant.ask_local(question)
    assert len(assistant.split_parts(question)) == 1 and not answer.parts
    assert assistant.state["ent"]["branches"] == ["B3", "B2"]
    assert len(answer.table) == 2 and len(answer.actions) == 2
    assert "مدينة نصر" in answer.actions[0] and "الدقي" in answer.actions[1]
    assert "أولوية 1" in answer.actions[0] and "أولوية 2" in answer.actions[1]
    assert "المؤكد" in answer.actions[0] and "الاحتمال" in answer.actions[0]
    assert "المصنفة زيادة" in answer.actions[1]
    if "هامش" in question:
        assert "هامش الربح الحالية (%)" in answer.table
        assert "فرق الهامش (نقطة مئوية)" in answer.table
    assert "الخطط من المحرك المحلي" in answer.llm_note


def test_calendar_months_do_not_become_360_days(assistant):
    start, end, label = assistant.parse_period("آخر 12 شهر")
    assert start == end - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    assert (end - start).days + 1 == 365 and label == "آخر 12 شهر"


def test_branch_plan_contract_with_groq_is_not_a_flat_step_list(assistant):
    out = synthesis_output()
    out["steps"] = []
    session = configure_mock(assistant, [Response(content=out), Response(content=out)])
    answer = assistant.ask(MANAGER_BRANCH_REQUEST)
    assert answer.mode.startswith("Groq") and len(answer.actions) == 2 and len(answer.table) == 2
    assert "exactly 0 steps" in session.calls[0]["messages"][0]["content"]
    assert "المؤكد" in answer.actions[0] and "سبب الأولوية" in answer.actions[1]
    followup = assistant.ask_local("طب والربح؟")
    assert assistant.state["ent"]["branches"] and "كل الشركة" not in followup.source["النطاق"]
    assistant.ask_local("سيبنا من التوصيات. عايز أخطر 3 مشاكل في جودة البيانات")
    assert assistant.state["intent_final"] == "health" and not assistant.state["ent"]["branches"]


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"])
def test_reasoning_budget_parameter_only_sent_to_supported_models(assistant, model):
    session = configure_mock(assistant, [Response(content={})], model=model)
    assistant.client.chat_json([{"role": "user", "content": "JSON"}], purpose="synthesis")
    assert session.calls[0].get("reasoning_effort") == ("low" if model.startswith("openai/gpt-oss-") else None)


class Session:
    def __init__(self, results):
        self.results, self.calls = list(results), []
    def post(self, *args, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["json"])))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.parametrize("finish,content,reason", [
    ("stop", "{bad", "parse:json_syntax"),
    ("stop", "[]", "parse:json_not_object"),
    ("tool_calls", None, "parse:unexpected_finish_reason"),
    ("stop", None, "parse:missing_content"),
    ("length", "{", "parse:output_truncated"),
])
def test_response_parse_failure_has_specific_safe_reason(finish, content, reason, caplog):
    response = Response()
    response.json = lambda: {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"total_tokens": 1}}
    client = GroqClient(Settings(api_key="test-only", retries=1), Session([response]), sleep=lambda _: None)
    with pytest.raises(LLMError, match=reason):
        client.chat_json([{"role": "user", "content": "private question"}], max_tokens=3200, purpose="synthesis")
    assert reason in caplog.text and "private question" not in caplog.text


def test_synthesis_truncation_retries_with_larger_budget_offline():
    truncated = Response()
    truncated.json = lambda: {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}], "usage": {"total_tokens": 1}}
    session = Session([truncated, Response(content=synthesis_output())])
    client = GroqClient(Settings(api_key="test-only", retries=2), session, sleep=lambda _: None)
    out = client.chat_json([{"role": "user", "content": "x"}], max_tokens=3200, purpose="synthesis")
    assert len(out["steps"]) == 3
    assert [call["max_tokens"] for call in session.calls] == [3200, 6400]


def configure_mock(assistant, results, **kwargs):
    session = Session(results)
    assistant.client = GroqClient(Settings(api_key="test-only", retries=1, **kwargs), session, sleep=lambda _: None)
    return session


@pytest.mark.parametrize("result,reason", [
    (Response(429, headers={"retry-after": "40"}), "حد الاستخدام"),
    (Response(401), "رفضت بيانات الاتصال"),
    (requests.Timeout(), "بطيئة"),
    (requests.ConnectionError(), "مش قادرين نوصل"),
    (Response(content={"wrong": True}), "مكانش مفهوم"),
    (Response(503), "غير متاحة"),
])
def test_groq_failure_is_local_and_honest(assistant, result, reason):
    session = configure_mock(assistant, [result])
    a = assistant.ask("وريني أداء الشركة")
    assert a.mode == "المحرك المحلي" and reason in a.llm_note
    assert len(session.calls) == 1 and not assistant.state


def test_mock_plan_success_has_provenance(assistant):
    session = configure_mock(assistant, [Response(content={"calls": [{"tool": "query", "metric": "net"}]})])
    a = assistant.ask("وريني أداء الشركة")
    assert a.mode.startswith("Groq لفهم") and "الحسابات" in a.mode
    assert len(session.calls) == 1 and "27,750" in a.text


@pytest.mark.parametrize("call", [
    {"tool": "forecast"}, {"tool": "query", "branches": ["جليم"]},
    {"tool": "query", "branches": "جليم"}, {"tool": "query", "metric": "fake"},
    {"tool": "query", "start": "not-a-date"}, {"tool": "query", "top_n": True},
])
def test_unrequested_or_invalid_model_call_rejected(assistant, call):
    configure_mock(assistant, [Response(content={"calls": [call]})])
    a = assistant.ask("وريني أداء الشركة")
    assert a.mode == "المحرك المحلي" and a.llm_note and not assistant.state


def test_known_routes_do_not_consult_model(assistant):
    session = configure_mock(assistant, [])
    assistant.ask("توقعات نوفمبر وديسمبر ودقة النموذج")
    assistant.ask("حلل العملاء آخر 12 شهر")
    assistant.ask("أهم 3 مشاكل")
    assistant.ask("صحة البيانات")
    assert session.calls == []


def test_auth_cooldown_keeps_real_reason(assistant):
    session = configure_mock(assistant, [Response(401)])
    a = assistant.ask("وريني أداء الشركة")
    b = assistant.ask("وريني نتائج الشركة")
    assert "رفضت بيانات الاتصال" in a.llm_note and "رفضت بيانات الاتصال" in b.llm_note
    assert len(session.calls) == 1


def test_cache_does_not_share_mutable_plans(assistant):
    session = configure_mock(assistant, [Response(content={"calls": [{"tool": "query", "metric": "net"}]})])
    one = assistant.plan("وريني أداء الشركة")
    one["calls"][0]["metric"] = "margin"
    two = assistant.plan("وريني أداء الشركة")
    assert two["calls"][0]["metric"] == "net" and len(session.calls) == 1


@pytest.mark.parametrize("text,count", [
    ("أهم مشكلتين محتاجين تدخل عاجل", 2), ("أول مشكلتان", 2),
    ("أهم توصيتين", 2), ("أفضل فرعين", 2), ("أكبر عميلين", 2),
    ("أخطر ٣ مشاكل في جودة البيانات", 3), ("أول ثلاث ملاحظات", 3),
    ("أعلى خمسة منتجات", 5), ("أهم عشر توصيات", 10),
    ("المبيعات آخر 12 شهر", None), ("مساهمة أكبر 10% من العملاء", None),
    ("صحة البيانات 2026", None),
])
def test_general_result_cardinality(text, count):
    assert result_count(text) == count


def test_dual_recommendations_then_leave_topic_for_health(assistant, monkeypatch):
    assistant.D.health = [
        dict(check="valid", status="ok", detail="سليم"),
        dict(check="warning1", status="warn", detail="تنبيه 1"),
        dict(check="failed", status="error", detail="فشل الربط"),
        dict(check="warning2", status="warn", detail="تنبيه 2"),
        dict(check="warning3", status="warn", detail="تنبيه 3"),
    ]
    calls = []
    dispatch = assistant.dispatch
    def record(sp):
        calls.append(sp["intent_final"])
        return dispatch(sp)
    monkeypatch.setattr(assistant, "dispatch", record)
    one = assistant.ask("إيه أهم مشكلتين محتاجين تدخل عاجل؟")
    assert len(one.table) == 2
    two = assistant.ask("سيبنا من التوصيات، عايز أخطر 3 مشاكل في جودة البيانات")
    assert calls == ["recommend", "health"] and not two.parts
    assert len(two.table) == 3 and two.table.iloc[0]["الفحص"] == "failed"
    assert two.source["المصدر"] == "فحوصات data.py"
    assert "مرتجعات" not in two.text and "حرجة" not in two.text


def test_health_reports_shortfall_without_inventing_checks(assistant):
    a = assistant.ask("أخطر 3 مشاكل في صحة البيانات")
    assert len(a.table) == 1 and "من أصل 3" in a.headline
    assert a.table.iloc[0]["النتيجة"] != "ok"


@pytest.mark.parametrize("before", ["المبيعات كام", "الربح كام", "أعلى 3 فروع في الربح", "المخزون", "توقعات نوفمبر",
                                    "حلل العملاء", "أهم مشكلتين", "صحة البيانات"])
def test_all_domains_can_be_left_for_health(assistant, before):
    assistant.ask(before)
    a = assistant.ask("سيبنا من الموضوع ده، عايز أخطر ثلاث ملاحظات على صحة البيانات")
    assert assistant.state["intent_final"] == "health" and not a.parts


@pytest.mark.parametrize("after,intent", [("المبيعات كام", "query"), ("الربح كام", "query"),
    ("أعلى فرعين في الربح", "query"), ("أكتر فرعين عندهم نقص مخزون", "inventory"),
    ("توقعات نوفمبر", "forecast"), ("حلل العملاء", "customers"), ("أهم توصيتين", "recommend")])
def test_leave_health_for_other_domains(assistant, after, intent):
    assistant.ask("صحة البيانات")
    assistant.ask(after)
    assert assistant.state["intent_final"] == intent


def test_customer_ranking_honours_dual(assistant):
    a = assistant.ask("أهم عميلين آخر 12 شهر")
    assert len(a.table) == 2


def test_next_month_not_first_saved_month(assistant):
    # Some saved forecasts include the current partial month.
    current = assistant.D.forecast.iloc[[0]].copy()
    current["month"], current["h"] = "2026-10", 0
    assistant.D.forecast = pd.concat([current, assistant.D.forecast], ignore_index=True)
    a = assistant.ask("توقع مبيعات للشهر الجاي")
    assert set(a.table["الشهر"]) == {"2026-11"}


@pytest.mark.parametrize("question", [
    "سيبنا من التوصيات. عايز أخطر 3 مشاكل في جودة البيانات وتأثيرها على التحليلات",
    "سيبنا من التوصيات، عايز أخطر ثلاث مشاكل في جودة البيانات، وتأثيرها على المبيعات والربح",
    "سيبنا من التوصيات؛ عايز أخطر ٣ مشاكل في صحة البيانات. وقولي تأثيرها على التحليلات والمؤشرات غير الموثوقة",
    "عايز أخطر 3 مشاكل في جودة بيانات العملاء وتأثيرها على التحليلات بدون توصيات",
])
def test_health_details_are_one_tool_and_one_answer(assistant, monkeypatch, question):
    assistant.D.health = [dict(check=key, status="warn", detail=f"دليل {i}")
                          for i, key in enumerate(list(CHECK_EFFECTS)[5:10])]
    assistant.ask("أهم مشكلتين محتاجين تدخل عاجل")
    session = configure_mock(assistant, [])
    calls = []
    original = assistant.dispatch
    def tracked(spec):
        calls.append(spec["intent_final"])
        return original(spec)
    monkeypatch.setattr(assistant, "dispatch", tracked)
    a = assistant.ask(question)
    assert calls == ["health"] and not a.parts and len(a.table) == 3
    assert set(a.table["الفحص"]) <= {row["check"] for row in assistant.D.health}
    assert a.table["الأثر على التحليلات"].str.len().min() > 20
    assert a.table["المؤشرات المتأثرة وحدود الثقة"].str.len().min() > 20
    assert a.source["المصدر"] == "فحوصات data.py" and session.calls == []


def test_health_effects_do_not_invent_numbers_or_global_invalidity():
    effect, indicators = analytical_effect("كل المبيعات مربوطة بعميل", "warn")
    assert "لا يبطل إجمالي المبيعات" in indicators
    assert not any(ch.isdigit() for ch in effect + indicators)
    effect, indicators = analytical_effect("فحص جديد غير معروف", "error")
    assert "غير موصوف" in effect and "غير محددة" in indicators
    effect, _ = analytical_effect("كل المبيعات مربوطة بعميل", "ok")
    assert "لا توجد مشكلة" in effect


def test_health_tool_schema_and_wrong_fallback(assistant):
    assert "health" in Assistant.TOOLS_OK and "|health" in Assistant.PLAN_SYS
    question = "سيبنا من التوصيات. عايز أخطر 3 مشاكل في جودة البيانات وتأثيرها"
    for tool in ("recommend", "query", "forecast"):
        with pytest.raises(LLMError):
            assistant.spec_from_call({"tool": tool}, question)
    spec = assistant.spec_from_call({"tool": "health"}, question)
    assert spec["intent"] == "health" and spec["topn"] == 3
    a = assistant.ask_local(question)
    assert a.source["المصدر"] == "فحوصات data.py" and not a.parts


def test_negated_topics_not_reintroduced():
    for phrase in ("سيبنا من التوصيات", "تجاهل التوصيات", "بدون توصيات"):
        assert Assistant.topic_of(phrase) is None
        assert not Assistant.recommendation_request(phrase)
    parts = Assistant.split_parts("سيبنا من التوصيات. جودة البيانات وتأثيرها على المبيعات والربح")
    assert len(parts) == 1 and "توصيات" not in parts[0]


def test_session_replaces_old_assistant_class_without_losing_memory():
    import ast
    from copy import deepcopy
    app = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    function = next(n for n in app.body if isinstance(n, ast.FunctionDef) and n.name == "get_assistant")
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    class Current:
        def __init__(self, D, ctx, client):
            self.state, self.turns, self.ctx = {}, [], ctx
    old = SimpleNamespace(state={"ent": {"governorates": ["القاهرة"]}}, turns=[{"role": "user", "content": "سؤال"}])
    memory = deepcopy(old.state)
    state = State(asst=old, _asst_stamp="same-data")
    namespace = {"Ctx": Ctx, "Assistant": Current, "st": SimpleNamespace(session_state=state),
                 "stamp": lambda: "same-data", "llm_client": lambda: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
    filters = {"start": pd.Timestamp("2026-01-01"), "end": pd.Timestamp("2026-10-30")}
    current = namespace["get_assistant"](None, filters)
    assert type(current) is Current and current.state == memory and current.state is not old.state
    assert namespace["get_assistant"](None, filters) is current
    state._asst_stamp = "old-data"
    assert namespace["get_assistant"](None, filters).state == {}


def test_reported_question_against_saved_health_snapshot(assistant):
    path = data.PROC / "health.json"
    if not path.exists():
        pytest.skip("Saved health snapshot unavailable")
    assistant.D.health = json.loads(path.read_text(encoding="utf-8"))
    assistant.ask("أهم مشكلتين محتاجين تدخل عاجل")
    session = configure_mock(assistant, [])
    a = assistant.ask("سيبنا من التوصيات. عايز أخطر 3 مشاكل في جودة البيانات وتأثيرها على التحليلات")
    recorded = {row["check"]: row for row in assistant.D.health}
    assert len(a.table) == min(3, sum(row["status"] != "ok" for row in recorded.values()))
    for _, row in a.table.iterrows():
        assert row["التفاصيل"] == recorded[row["الفحص"]]["detail"]
        assert row["النتيجة"] == recorded[row["الفحص"]]["status"] != "ok"
        assert row["الأثر على التحليلات"] and row["المؤشرات المتأثرة وحدود الثقة"]
    assert session.calls == [] and not a.parts


DOMAIN_REQUESTS = [
    ("المبيعات كام", "query"), ("الربح كام", "query"), ("أعلى فرعين في المبيعات", "query"),
    ("أكتر فرعين عندهم نقص مخزون", "inventory"), ("توقعات نوفمبر وديسمبر", "forecast"),
    ("حلل العملاء آخر 12 شهر", "customers"), ("أهم مشكلتين", "recommend"), ("صحة البيانات", "health"),
]


@pytest.mark.parametrize("before,_", DOMAIN_REQUESTS)
@pytest.mark.parametrize("after,intent", DOMAIN_REQUESTS)
def test_pairwise_topic_routing_contract(assistant, before, _, after, intent):
    prior, _titles = assistant.response_specs(before)
    assistant.state = assistant.merge(prior[0])
    current, _titles = assistant.response_specs(after)
    assert len(current) == 1
    resolved = assistant.merge(current[0])
    assert resolved["intent_final"] == intent
    assert resolved["ent"] == {"branches": [], "governorates": [], "categories": [], "channels": []}


def test_real_processed_data_smoke():
    if not (data.PROC / "sales.parquet").exists():
        pytest.skip("Local processed data unavailable")
    D = data.load()
    a = Assistant(D, client=GroqClient(Settings(api_key="", enabled=False)))
    for question, intent in [("توقعات نوفمبر وديسمبر ودقة النموذج وهل نعتمد عليها في المخزون؟", "forecast"),
                             ("حلل العملاء آخر 12 شهر وموثوقية النموذج", "customers"),
                             ("راجع صحة البيانات", "health")]:
        result = a.ask(question)
        assert result.text and a.state["intent_final"] == intent
    recommendations = a.ask("إيه أهم مشكلتين محتاجين تدخل عاجل؟")
    assert len(recommendations.table) == 2 and a.state["intent_final"] == "recommend"
    checks = a.ask("سيبنا من التوصيات، عايز أخطر 3 مشاكل في جودة البيانات")
    available = sum(h["status"] != "ok" for h in D.health)
    assert len(checks.table) == min(3, available) and a.state["intent_final"] == "health"
    assert checks.source["المصدر"] == "فحوصات data.py"
    session = configure_mock(a, [Response(content=synthesis_output()), Response(content=synthesis_output())])
    analysis = a.ask("حلل 3 فروع ضعيفة واربط مبيعاتها بالمخزون والخصومات وخطة من 3 خطوات")
    assert analysis.mode.startswith("Groq"), analysis.llm_note
    assert len(a.state["ent"]["branches"]) == 3 and len(analysis.actions) == 3
    assert len(session.calls) == 2
    packet = json.loads(session.calls[0]["messages"][1]["content"])
    assert all(len(item["branches"]) == 3 for item in packet["evidence"])
    decision = a.ask_local(MANAGER_BRANCH_REQUEST)
    start, end = D.last.normalize() - pd.DateOffset(months=12) + pd.Timedelta(days=1), D.last.normalize()
    previous_end = start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    current = D.sales.loc[D.sales.date.between(start, end)].groupby("branch").net.sum()
    previous = D.sales.loc[D.sales.date.between(previous_start, previous_end)].groupby("branch").net.sum()
    growth = ((current - previous) / previous.abs().replace(0, np.nan)).dropna().sort_index().sort_values(kind="stable")
    assert a.state["ent"]["branches"] == growth.head(2).index.tolist()
    assert len(decision.table) == len(decision.actions) == 2 and not decision.parts
    assert "هامش الربح الحالية (%)" in decision.table

@pytest.mark.parametrize('visual_request', ['اعملها علي هيأة جراف او داش بورد يوضح اكتر', 'حولها لرسم بياني', 'اعرضها في داشبورد', 'show as charts'])
def test_visual_followup_uses_saved_numeric_result(assistant, monkeypatch, visual_request):
    table = pd.DataFrame({'الفرع': ['أ', 'ب'], 'المبيعات (جنيه)': [120., 90.], 'التغطية المقدرة (أسابيع)': [2., 3.], 'هامش الربح %': [30., 25.]})
    original = Answer.make('تحليل', table=table, source={'الفترة': '2026-10', 'النطاق': 'أ - ب'}, notes=['التغطية تقديرية'])
    assistant.last_answer = original
    def forbidden(*a, **k):
        raise AssertionError('Formatting must not route or call tools/Groq')
    monkeypatch.setattr(assistant, '_answer', forbidden)
    result = assistant.ask(visual_request)
    assert result.fig is not None and len(result.fig.data) == 3
    for trace, col in zip(result.fig.data, table.columns[1:]):
        assert list(trace.y) == table[col].tolist()
        assert list(trace.customdata) == ['أ', 'ب']
    assert result.source == original.source and result.notes == original.notes
    assert result.table is None
    assert original.fig is None
    assert assistant.last_answer is original


def test_visual_followup_no_history_or_non_numeric(assistant):
    assert assistant.ask_local('اعملها جراف').fig is None
    assistant.last_answer = Answer.make('صحة البيانات', table=pd.DataFrame({'الفحص': ['المبيعات'], 'الحالة': ['سليم']}))
    result = assistant.ask_local('اعملها جراف')
    assert result.fig is None and 'النتيجة الأخيرة' in result.text


def test_visual_new_topic_does_not_replay_previous(assistant, monkeypatch):
    old = Answer.make('قديم', table=pd.DataFrame({'الفرع': ['أ'], 'المبيعات': [12]}))
    assistant.last_answer = old
    fresh = Answer.make('عملاء', table=pd.DataFrame({'الشريحة': ['جديد'], 'العدد': [7]}))
    seen = []
    monkeypatch.setattr(assistant, '_answer', lambda text, use_llm: seen.append(text) or fresh)
    result = assistant.ask_local('اعرض العملاء في رسم بياني')
    assert seen and list(result.fig.data[0].y) == [7]
    monkeypatch.setattr(assistant, '_answer', lambda text, use_llm: Answer.make('فحوصات جودة البيانات سليمة'))
    assistant.ask_local('صحة البيانات')
    assert assistant.ask_local('اعملها جراف').fig is None


def test_visual_real_branch_analysis_without_groq():
    a = Assistant(data.load(), client=GroqClient(Settings(enabled=False)))
    answer = a.ask_local('اختار أضعف فرعين من حيث نمو المبيعات وحلل أداء كل فرع باستخدام المبيعات والمخزون والخصومات وهامش الربح واقترح خطة')
    assert answer.table is not None and len(answer.table) == 2
    result = a.ask_local('اعملها علي هيأة جراف او داش بورد يوضح اكتر')
    assert result.fig is not None
    for trace in result.fig.data:
        assert list(trace.y) == answer.table[trace.name].tolist()
        assert list(trace.customdata) == answer.table['الفرع'].tolist()
    assert result.source == answer.source
    assert a.ask_local('حولها لرسم بياني').fig.to_json() == result.fig.to_json()


def test_visual_followup_rendered_in_streamlit():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / 'app.py'), default_timeout=180).run()
    at.sidebar.radio[0].set_value('المساعد الذكي').run()
    a = at.session_state.asst
    a.last_answer = Answer.make('تحليل', table=pd.DataFrame({'الفرع': ['أ', 'ب'], 'المبيعات': [12., 9.], 'التغطية المقدرة (أسابيع)': [2., 3.]}))
    at.chat_input[0].set_value('اعملها جراف').run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 1
    at.run()
    assert not at.exception and len(at.get('plotly_chart')) == 1
    next(b for b in at.button if b.label == 'مسح المحادثة').click().run()
    assert not at.exception and at.session_state.asst.last_answer is None
    assert not at.get('plotly_chart')


def test_visual_negation_does_not_activate_chart_mode(assistant, monkeypatch):
    seen = []
    answer = Answer.make('نتيجة محلية')
    monkeypatch.setattr(assistant, '_answer', lambda text, use_llm: seen.append(text) or answer)
    result = assistant.ask_local('مش عايز جراف، عايز تحليل العملاء')
    assert seen and result is answer and result.fig is None

CUSTOMER_ACTION_QUESTION = 'حلل أداء العملاء في Cottonil خلال آخر 12 شهر. عايز أعرف عدد العملاء النشطين، ونسبة المبيعات اللي ملهاش بيانات عملاء، وقد إيه أكبر 10% من العملاء بيساهموا في المبيعات. بناءً على الأرقام دي، اقترح 3 إجراءات عملية لتحسين الاحتفاظ بالعملاء، ووضح إيه اللي نقدر نستنتجه فعلًا وإيه اللي محتاج بيانات إضافية.'


def test_customer_actions_and_priority_use_same_evidence(assistant, monkeypatch):
    first = assistant.ask_local(CUSTOMER_ACTION_QUESTION)
    assert not first.parts
    assert len(first.actions) == 3
    assert first.inferences
    assert first.decision_topic == 'customers'
    def blocked(*args, **kwargs):
        raise AssertionError('Priority followup must reuse saved evidence')
    monkeypatch.setattr(assistant, '_answer', blocked)
    follow = assistant.ask('طيب من الإجراءات دي، اختار أهم إجراء نبدأ بيه لو الميزانية محدودة، وفسر اختيارك بناءً على نتائج تحليل العملاء اللي لسه عملته.')
    assert not follow.parts and len(follow.actions) == 1
    assert follow.actions[0].startswith(first.decision_options[0]['action'])
    assert follow.source == first.source
    assert not follow.facts and 'حسّن تسجيل' in follow.actions[0]
    assert 'تكلفة' in follow.notes[0]


def test_priority_followup_does_not_hijack_topic_change(assistant):
    assistant.ask_local(CUSTOMER_ACTION_QUESTION)
    changed = assistant.ask_local('سيبنا من العملاء عايز أهم 3 مشاكل في جودة البيانات')
    assert changed.decision_topic != 'customers'


def test_customer_action_candidates_depend_on_available_evidence(assistant):
    assistant.D.sales['customer'] = 'C0'
    first = assistant.ask_local(CUSTOMER_ACTION_QUESTION)
    assert len(first.actions) == 2
    assert not any('حسّن تسجيل' in x for x in first.actions)
    assert any('المتاح 2' in x for x in first.notes)

@pytest.mark.parametrize('joiner', ['، ', '. ', '\n', '؛ '])
def test_narrative_evidence_is_one_request_across_punctuation(assistant, joiner):
    question = joiner.join(['عندنا مبيعات كتير من غير بيانات عملاء', 'وفي نفس الوقت جزء كبير من الإيرادات جاي من عدد قليل من العملاء', 'حلل الوضع خلال آخر 12 شهر', 'واقترح 3 إجراءات واقعية لتحسين الاحتفاظ بالعملاء بأقل تكلفة مع توضيح الأرقام اللي بنيت عليها قراراتك'])
    result = assistant.ask_local(question)
    assert not result.parts and len(result.actions) == 3
    assert result.decision_topic == 'customers'
    assert 'آخر 12 شهر' in result.source['الفترة']
    follow = assistant.ask_local('طيب لو الميزانية تسمح بتنفيذ إجراء واحد بس من التلاتة اللي اقترحتهم، تختار أنهي واحد وليه؟ وإزاي نقيس نجاحه بعد 3 شهور؟')
    assert not follow.parts and len(follow.actions) == 1
    assert len(follow.facts) == 1 and 'بعد 3 شهور' in follow.facts[0]
    assert 'خط الأساس' in follow.facts[0] and 'الفواتير' in follow.facts[0]
    assert follow.source == result.source
    assert not follow.table


@pytest.mark.parametrize('topic', ['المبيعات', 'المخزون', 'التوقعات', 'صحة البيانات'])
def test_referential_analysis_clause_keeps_its_owner(topic):
    parts = Assistant.split_parts(f'اعرض {topic}، حلل الوضع ووضح النتائج')
    assert len(parts) == 1


def test_independent_commands_remain_separate():
    parts = Assistant.split_parts('حلل العملاء. اعرض صحة البيانات')
    assert len(parts) == 2


@pytest.mark.parametrize('text,expected', [('إجراء واحد بس بعد 3 شهور', 1), ('خطوة واحدة', 1), ('3 إجراءات', 3), ('بعد 3 شهور', None)])
def test_action_cardinality_is_not_evaluation_horizon(text, expected):
    assert result_count(text) == expected

def test_real_narrative_and_evaluation_horizon():
    a = Assistant(data.load(), client=GroqClient(Settings(enabled=False)))
    first = a.ask_local('عندنا مبيعات كتير من غير بيانات عملاء، وفي نفس الوقت جزء كبير من الإيرادات جاي من عدد قليل من العملاء. حلل الوضع خلال آخر 12 شهر، واقترح 3 إجراءات واقعية لتحسين الاحتفاظ بالعملاء بأقل تكلفة، مع توضيح الأرقام اللي بنيت عليها قراراتك.')
    assert not first.parts and len(first.actions) == 3
    follow = a.ask_local('طيب لو الميزانية تسمح بتنفيذ إجراء واحد بس من التلاتة اللي اقترحتهم، تختار أنهي واحد وليه؟ وإزاي نقيس نجاحه بعد 3 شهور؟')
    assert not follow.parts and len(follow.actions) == 1
    assert '19.23%' in follow.facts[0]
    assert 'بعد 3 شهور' in follow.facts[0]
    assert first.source == follow.source
    assert '2025-10-19' in follow.source['الفترة']

CLOUD_CUSTOMER_Q = 'عندنا مشكلة إن جزء من مبيعات Cottonil مش مرتبط بعملاء معروفين. حلل تأثير ده على فهم سلوك العملاء، واقترح 3 حلول عملية قليلة التكلفة، ورتبهم حسب الأولوية.'
CLOUD_FOLLOW_Q = 'طيب من التلات حلول دول، لو هننفذ واحد بس، تختار إيه وليه؟ وحطلي 3 مؤشرات نقيس بيها نجاحه خلال 90 يوم.'


def test_reviewed_synthesis_actions_survive_real_data_followup(monkeypatch):
    a = Assistant(data.load(), client=GroqClient(Settings(enabled=False)))
    draft = {'analysis': [{'text': 'قد يحد غياب تعريف العميل من تتبع سلوكه', 'evidence': ['sales', 'customers']}],
             'steps': [{'text': text, 'evidence': ['customers']} for text in (
                 'راجع ربط المعاملات بمعرف العميل قبل تخصيص إنفاق جديد',
                 'دقق جودة معرّفات العملاء المسجلة',
                 'سجل الموافقات وتكلفة تحسين التسجيل') ]}
    session = configure_mock(a, [Response(content=draft), Response(content=draft)])
    first = a.ask(CLOUD_CUSTOMER_Q)
    assert first.mode.startswith('Groq لصياغة'), first.llm_note
    assert len(first.actions) == 3 and len(first.decision_options) == 3
    assert '19.23%' in first.text
    def blocked(*args, **kwargs):
        raise AssertionError('No routing/tool/LLM on decision reference')
    monkeypatch.setattr(a, '_answer', blocked)
    result = a.ask(CLOUD_FOLLOW_Q)
    assert len(session.calls) == 2
    assert not result.parts and len(result.actions) == 1
    assert first.actions[0] in result.actions[0]
    assert len(result.table) == 3 and '90 يوم' in result.text
    assert result.source == first.source


def test_old_generated_action_snapshot_is_hydrated(assistant, monkeypatch):
    assistant.last_answer = Answer.make('تحليل', actions=['تحسين تعريف العملاء', 'تدقيق الربط', 'قياس الجودة'], source={'customers': {'الفترة': '2026'}})
    monkeypatch.setattr(assistant, '_answer', lambda *a, **k: pytest.fail('must reuse saved actions'))
    answer = assistant.ask_local(CLOUD_FOLLOW_Q)
    assert len(answer.actions) == 1 and len(answer.table) == 3
    assert 'تحسين تعريف العملاء' in answer.actions[0]

@pytest.mark.parametrize('noun', ['مشكلة نقص', 'عدم اكتمال', 'ضعف جودة'])
@pytest.mark.parametrize('actions', ['حلول', 'إجراءات', 'خطوات'])
def test_customer_subject_owns_problem_and_actions(assistant, monkeypatch, noun, actions):
    q = f'حلل {noun} بيانات العملاء في Cottonil خلال آخر 12 شهر، واقترح 3 {actions} عملية قليلة التكلفة لتحسين جودة البيانات والاحتفاظ بالعملاء، ورتبهم حسب الأولوية.'
    monkeypatch.setattr(assistant, 'h_recommend', lambda *a: pytest.fail('wrong recommendations tool'))
    monkeypatch.setattr(assistant, 'h_health', lambda *a: pytest.fail('quality outcome is not an audit request'))
    first = assistant.ask_local(q)
    assert not first.parts and len(first.actions) == 3
    assert len(first.decision_options) == 3
    follow = assistant.ask_local('طيب من التلات حلول اللي اقترحتهم، اختار حل واحد نبدأ بيه لو الميزانية محدودة، واشرح ليه اخترته، وحدد 3 مؤشرات نقيس بيها نجاحه خلال 90 يوم.')
    assert not follow.parts and len(follow.actions) == 1 and len(follow.table) == 3
    assert follow.source == first.source


def test_missing_decision_reference_fails_closed(assistant, monkeypatch):
    assistant.last_answer = Answer.make('رد قديم بدون خيارات')
    monkeypatch.setattr(assistant, '_answer', lambda *a: pytest.fail('unresolved reference must not route'))
    result = assistant.ask_local('طيب من التلات حلول اختار حل واحد وليه')
    assert 'مفيهوش خيارات' in result.text


def test_explicit_customer_audit_and_engine_still_route(assistant):
    assert assistant.response_specs('اعرض فحوص جودة بيانات العملاء')[0][0]['intent'] == 'health'
    assert assistant.response_specs('اعرض محرك التوصيات عن العملاء')[0][0]['intent'] == 'recommend'


def test_real_customer_quality_scenario_no_groq():
    a = Assistant(data.load(), client=GroqClient(Settings(enabled=False)))
    first = a.ask_local('حلل مشكلة نقص بيانات العملاء في Cottonil خلال آخر 12 شهر، واقترح 3 حلول عملية قليلة التكلفة لتحسين جودة البيانات والاحتفاظ بالعملاء، ورتبهم حسب الأولوية.')
    assert not first.parts and len(first.actions) == 3 and '19.23%' in first.text
    second = a.ask_local('طيب من التلات حلول اللي اقترحتهم، اختار حل واحد نبدأ بيه لو الميزانية محدودة، واشرح ليه اخترته، وحدد 3 مؤشرات نقيس بيها نجاحه خلال 90 يوم.')
    assert not second.parts and len(second.actions) == 1 and len(second.table) == 3
    assert first.source == second.source
