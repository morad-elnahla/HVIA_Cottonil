"""Explicit opt-in live Groq acceptance; aggregate evidence only, no training."""
import argparse
import json
from pathlib import Path
from src import data
from src.assistant import Assistant
from src.llm import GroqClient, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize the external Groq requests for this check")
    parser.add_argument("--scenario", choices=("ranked-comparison", "branch-plans"), default="ranked-comparison")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live explicitly; this command consumes Groq tokens.")
    cfg = Settings.from_env()
    cfg.retries = min(cfg.retries, 2)
    client = GroqClient(cfg)
    assistant = Assistant(data.load(), client=client)
    drafts = []
    validate = assistant.validate_synthesis
    def capture(output, evidence, steps):
        drafts.append(output)
        return validate(output, evidence, steps)
    assistant.validate_synthesis = capture
    question = "اختار أضعف 3 فروع من حيث نمو المبيعات، وقارن مبيعات ومخزون وخصومات كل فرع. وضح هل فيه علاقة بينهم، وبعدها اقترح خطة من 3 خطوات لتحسين الأداء، باستخدام أرقام فعلية من الداتا."
    expected = 3
    if args.scenario == "branch-plans":
        expected = 2
        question = ("أنا مدير Cottonil وعندي ميزانية محدودة لتحسين أداء الفروع. اختار أضعف فرعين من حيث نمو المبيعات خلال آخر 12 شهر، "
                    "وقارن أداء كل فرع بمخزونه وخصوماته وهامش ربحه. حدد المشكلة الأهم في كل فرع بناءً على الأدلة، ووضح إيه المؤكد وإيه مجرد احتمال. "
                    "في النهاية اقترح خطتين عمليتين مختلفتين، واحدة لكل فرع، وحدد أولوية التنفيذ وليه. استخدم أرقام فعلية ومصادرها، ومش عايز مجرد عرض جداول.")
    answer = assistant.ask(question)
    checks = {"reviewed_synthesis": answer.mode.startswith("Groq") and not answer.llm_note,
              "requested_action_count": len(answer.actions) == expected,
              "requested_distinct_branches": answer.table is not None and answer.table["الفرع"].nunique() == expected}
    calls_after_synthesis = client.stats["calls"]
    for question, intent in [("حلل العملاء آخر 12 شهر", "customers"),
                             ("إيه أهم مشكلتين محتاجين تدخل عاجل؟", "recommend"),
                             ("سيبنا من التوصيات. عايز أخطر 3 مشاكل في جودة البيانات وتأثيرها على التحليلات", "health")]:
        response = assistant.ask(question)
        checks["switch_to_" + intent] = assistant.state.get("intent_final") == intent
        if intent in ("recommend", "health"):
            checks["count_" + intent] = response.table is not None and len(response.table) == (2 if intent == "recommend" else 3)
    checks["no_unnecessary_followup_api"] = client.stats["calls"] == calls_after_synthesis
    result = {"checks": checks, "passed": all(checks.values()), "scenario": args.scenario, "model": cfg.model,
              "successful_requests": client.stats["calls"], "tokens": client.stats["tokens"],
              "mode": answer.mode, "limitation": "One live conversation; not a universal quality or availability guarantee."}
    root = Path(__file__).resolve().parent
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs/live_acceptance_answer.txt").write_text(answer.text, encoding="utf-8")
    (root / "logs/live_validation_drafts.json").write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "logs/release_live_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
