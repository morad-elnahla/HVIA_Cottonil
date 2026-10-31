"""اتصال Groq: إعدادات من ملف .env، إعادة محاولة ذكية، حماية من حدود الاستخدام، وتسجيل تقني آمن.

القواعد:
- المفتاح بيتقري من .env أو متغيرات البيئة بس، ومبيظهرش في الواجهة ولا في اللوج ولا في رسائل الأخطاء.
- أي فشل بيتحول لرسالة عربي بسيطة للمستخدم (LLMError.user_msg)، والتفاصيل التقنية بتتسجل في logs/assistant.log.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FRIENDLY = {"rate_limit": "خدمة الذكاء الاصطناعي (Groq) مشغولة دلوقتي بسبب حد الاستخدام",
            "timeout": "خدمة الذكاء الاصطناعي (Groq) بطيئة في الرد",
            "network": "مش قادرين نوصل لخدمة الذكاء الاصطناعي (Groq)",
            "unavailable": "خدمة الذكاء الاصطناعي (Groq) غير متاحة دلوقتي",
            "invalid": "رد خدمة الذكاء الاصطناعي (Groq) مكانش مفهوم",
            "auth": "خدمة Groq رفضت بيانات الاتصال؛ إعداد الاتصال محتاج مراجعة",
            "budget": "وفّرنا استهلاك الذكاء الاصطناعي (Groq) عشان نتجنب حد الاستخدام"}


# ---------------------------------------------------------------- الإعدادات
def load_env(path: Path | None = None) -> None:
    """بيقرا .env (KEY=VALUE) وبيحطه في os.environ من غير ما يكتب فوق متغير موجود فعلًا."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass
class Settings:
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    planner_model: str = ""
    timeout: float = 25.0
    retries: int = 3
    max_wait: float = 8.0
    narrate: str = "auto"
    tpm_limit: int = 9000
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        g = os.environ.get
        model = g("LLM_MODEL") or cls.model
        return cls(api_key=(g("LLM_API_KEY") or g("GROQ_API_KEY") or "").strip(), base_url=(g("LLM_BASE_URL") or cls.base_url).rstrip("/"), model=model,
                   planner_model=g("LLM_PLANNER_MODEL") or model, timeout=float(g("LLM_TIMEOUT") or cls.timeout), retries=int(g("LLM_MAX_RETRIES") or cls.retries),
                   max_wait=float(g("LLM_MAX_WAIT") or cls.max_wait), narrate=(g("LLM_NARRATE") or cls.narrate).lower(), tpm_limit=int(g("LLM_TPM_LIMIT") or cls.tpm_limit),
                   enabled=(g("LLM_ENABLED") or "true").lower() not in ("0", "false", "no", "off"))


# ---------------------------------------------------------------- تسجيل آمن
_SECRET = re.compile(r"(gsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9\-_]{8,}|Bearer\s+[A-Za-z0-9\-_\.]{8,})")


def redact(s: str) -> str:
    return _SECRET.sub("[محجوب]", str(s))


def _logger() -> logging.Logger:
    lg = logging.getLogger("cottonil.llm")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        lg.propagate = False
        try:
            (ROOT / "logs").mkdir(exist_ok=True)
            h = RotatingFileHandler(ROOT / "logs" / "assistant.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        except OSError:
            h = logging.NullHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(h)
    return lg


def qid(text: str) -> str:
    """معرّف مختصر للسؤال (من غير نصه) للتتبع في اللوج."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


class LLMError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(redact(detail) or kind)
        self.kind, self.user_msg = kind, FRIENDLY.get(kind, FRIENDLY["unavailable"])


# ---------------------------------------------------------------- العميل
class GroqClient:
    def __init__(self, cfg: Settings | None = None, session: requests.Session | None = None, sleep=time.sleep, clock=time.monotonic):
        self.cfg = cfg or Settings.from_env()
        self.s = session or requests.Session()
        self.sleep, self.clock = sleep, clock
        self.cool_until = 0.0
        self.cool_kind = "unavailable"
        self.fails = 0
        self.window: deque[tuple[float, int]] = deque()      # (وقت، توكنز) في آخر 60 ثانية
        self.cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.stats = {"calls": 0, "tokens": 0, "retries": 0, "errors": 0}
        self.log = _logger()

    # ---- حالة
    @property
    def configured(self) -> bool:
        return bool(self.cfg.api_key) and self.cfg.enabled

    def cooldown_left(self) -> float:
        return max(0.0, self.cool_until - self.clock())

    @property
    def available(self) -> bool:
        return self.configured and self.cooldown_left() <= 0

    def _cool(self, secs: float, why: str):
        self.cool_until = self.clock() + secs
        self.cool_kind = why if why in FRIENDLY else "unavailable"
        self.log.warning("cooldown %.0fs (%s)", secs, why)

    # ---- حد الاستهلاك المحلي (قبل ما Groq يرفض)
    def _budget_ok(self, est: int) -> bool:
        now = self.clock()
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()
        return sum(t for _, t in self.window) + est <= self.cfg.tpm_limit

    @staticmethod
    def _wait_from(resp) -> float:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        m = re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)", resp.text or "")
        if m:
            v = float(m.group(1))
            return v / 1000 if m.group(2) == "ms" else v * 60 if m.group(2) == "m" else v
        return 2.0

    # ---- طلب JSON واحد (مع إعادة محاولة)
    def chat_json(self, messages: list[dict], model: str | None = None, max_tokens: int = 300, purpose: str = "plan", qkey: str = "", cache_key: str | None = None) -> dict:
        if not self.configured:
            raise LLMError("unavailable", "no key")
        left = self.cooldown_left()
        if left > 0:
            raise LLMError(self.cool_kind, f"cooldown {left:.0f}s")
        if cache_key and cache_key in self.cache and self.clock() - self.cache[cache_key][0] < 3600:
            self.cache.move_to_end(cache_key)
            return deepcopy(self.cache[cache_key][1])
        est = sum(len(m["content"]) for m in messages) // 3 + max_tokens
        if not self._budget_ok(est):
            raise LLMError("budget", f"est {est} tokens over local TPM budget")
        body = {"model": model or self.cfg.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
        # GPT-OSS reasoning consumes the same completion budget as the JSON answer.
        # Low effort is documented for these two models; don't send it to others.
        if body["model"] in ("openai/gpt-oss-20b", "openai/gpt-oss-120b") and purpose in ("synthesis", "synthesis_review"):
            body["reasoning_effort"] = "low"
        last: LLMError | None = None
        for attempt in range(1, self.cfg.retries + 1):
            est = sum(len(m["content"]) for m in body["messages"]) // 3 + body["max_tokens"]
            if not self._budget_ok(est):
                raise LLMError("budget", "retry exceeds local TPM budget")
            t0 = self.clock()
            try:
                r = self.s.post(f"{self.cfg.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.cfg.api_key}"}, json=body, timeout=(5, self.cfg.timeout))
            except requests.Timeout as e:
                last = LLMError("timeout", f"timeout: {type(e).__name__}")
            except requests.RequestException as e:
                last = LLMError("network", f"{type(e).__name__}")
            else:
                dt = self.clock() - t0
                if r.status_code == 200:
                    parse_reason = "response_envelope"
                    try:
                        data = r.json()
                        choice = data["choices"][0]
                        if choice.get("finish_reason") == "length":
                            # Retry with room for the complete object, within the existing local budget.
                            used = int((data.get("usage") or {}).get("total_tokens") or est)
                            self.window.append((self.clock(), used))
                            larger = min(body["max_tokens"] * 2, max(1800, max_tokens * 2))
                            next_est = sum(len(m["content"]) for m in body["messages"]) // 3 + larger
                            if larger > body["max_tokens"] and self._budget_ok(next_est):
                                body["max_tokens"] = larger
                            else:
                                raise LLMError("budget", "truncated response; retry exceeds local budget")
                            self.log.warning("truncated JSON q=%s %s attempt=%d", qkey, purpose, attempt)
                            parse_reason = "output_truncated"
                            raise ValueError("truncated JSON")
                        if choice.get("finish_reason") not in (None, "stop"):
                            parse_reason = "unexpected_finish_reason"
                            raise ValueError("unexpected finish reason")
                        content = choice["message"]["content"]
                        if not isinstance(content, str):
                            parse_reason = "missing_content"
                            raise ValueError("missing JSON text")
                        # Some models wrap otherwise valid JSON in a Markdown fence.
                        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
                        parse_reason = "json_syntax"
                        out = json.loads(content)
                        if not isinstance(out, dict):
                            parse_reason = "json_not_object"
                            raise ValueError("expected JSON object")
                        parse_reason = "usage_metadata"
                        used = int((data.get("usage") or {}).get("total_tokens") or est)
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        last = LLMError("invalid", "parse:" + parse_reason)
                        body["messages"] = messages + [{"role": "user", "content": "Return a complete JSON object matching the requested schema. No explanation, reasoning or Markdown; keep optional fields omitted and text short."}]
                    else:
                        self.fails = 0
                        self.window.append((self.clock(), used))
                        self.stats["calls"] += 1
                        self.stats["tokens"] += used
                        self.log.info("ok q=%s %s model=%s attempt=%d %.1fs tokens=%d", qkey, purpose, body["model"], attempt, dt, used)
                        if cache_key:
                            self.cache[cache_key] = (self.clock(), deepcopy(out))
                            while len(self.cache) > 128:
                                self.cache.popitem(last=False)
                        return out
                elif r.status_code == 429:
                    wait = self._wait_from(r)
                    self.log.warning("429 q=%s %s attempt=%d wait=%.1fs body=%s", qkey, purpose, attempt, wait, redact((r.text or "")[:160]))
                    if wait > self.cfg.max_wait or attempt == self.cfg.retries:
                        self._cool(min(max(wait, 5.0), 120.0), "rate_limit")
                        self.stats["errors"] += 1
                        raise LLMError("rate_limit", f"429 wait {wait:.1f}s")
                    self.stats["retries"] += 1
                    self.sleep(wait + random.uniform(0.1, 0.5))
                    continue
                elif r.status_code in (401, 403):
                    self._cool(600, "auth")
                    self.stats["errors"] += 1
                    self.log.error("auth failure status=%s", r.status_code)
                    raise LLMError("auth", f"auth {r.status_code}")
                elif r.status_code == 400:
                    try:
                        error = r.json().get("error", {})
                        code = error.get("code", "") if isinstance(error, dict) else ""
                    except (ValueError, AttributeError):
                        code = ""
                    json_failure = code in ("json_validate_failed", "json_generation_failed")
                    last = LLMError("invalid", "400 JSON validation failed" if json_failure else "400 invalid request")
                    self.log.warning("400 q=%s %s attempt=%d json_validation=%s", qkey, purpose, attempt, json_failure)
                    if json_failure and "response_format" in body and attempt < self.cfg.retries:
                        # Keep strict local validation, but don't repeat the rejected JSON-mode request.
                        body.pop("response_format")
                        body["messages"] = messages + [{"role": "user", "content": "Return one complete JSON object only. No reasoning, prose or Markdown. Omit unused optional fields and keep titles short."}]
                    else:
                        self.stats["errors"] += 1
                        raise last
                else:
                    last = LLMError("unavailable", f"http {r.status_code} {redact((r.text or '')[:160])}")
            self.stats["errors"] += 1
            self.fails += 1
            self.log.warning("fail q=%s %s attempt=%d kind=%s %s", qkey, purpose, attempt, last.kind, last)
            if self.fails >= 4:
                self._cool(60, "consecutive failures")
                break
            if attempt < self.cfg.retries:
                self.stats["retries"] += 1
                self.sleep(min(2 ** (attempt - 1), 4) + random.uniform(0.1, 0.4))
        raise last or LLMError("unavailable")
