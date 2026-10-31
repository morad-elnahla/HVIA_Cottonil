<div align="center">

<br>

[![COTTONIL BI & AI — animated title](https://readme-typing-svg.demolab.com?font=Syne&weight=750&size=36&duration=3000&pause=1500&color=5EEAB0&center=true&vCenter=true&width=850&lines=COTTONIL+BI+%26+AI;Arabic+Business+Intelligence;Data+%C2%B7+Evidence+%C2%B7+Decisions)](https://github.com/morad-elnahla/HVIA_Cottonil)

### مركز قيادة الأعمال · Arabic Business Intelligence

**Streamlit Community Cloud:** the owner-approved processed dataset and saved
forecasts are included in this repository. Deploy `main` → `app.py` with Python
3.13; no data upload or rebuild is needed. Configure `LLM_API_KEY` and `LLM_MODEL`
in Streamlit Secrets for Groq. See [DEPLOYMENT.md](DEPLOYMENT.md).


![Python](https://img.shields.io/badge/Python-3.13-5eeab0?style=for-the-badge&logo=python&logoColor=white&labelColor=111b18)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-5eeab0?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=111b18)
![Groq](https://img.shields.io/badge/Groq-Optional_AI-5eeab0?style=for-the-badge&labelColor=111b18)
![Status](https://img.shields.io/badge/Status-Private_Deployment_Candidate-f6b94d?style=for-the-badge&labelColor=111b18)

> **From business records to measurable decisions — in Arabic.**
>
> Sales, profitability, inventory, forecasting, and a conversational assistant grounded in local calculations.

| Interface | Analytics | AI | Data | Release |
|:---:|:---:|:---:|:---:|:---:|
| Arabic · RTL · Dark theme | Pandas · Plotly | Groq + evidence review | Authorized local datasets | Tested private candidate |

**Private deployment candidate. See [deployment instructions](DEPLOYMENT.md) and the [verification report](RELEASE_REPORT.md). No public instance has been published.**

<br>
</div>

---

## ✦ Dashboard Preview

![Cottonil BI dashboard — overview, KPIs, and recommendations](photo/1.png)

*Project screenshot provided for the repository. Displayed figures are a snapshot, not live results or performance guarantees.*

## ✦ What Is Cottonil BI?

Cottonil BI is a local Streamlit application for exploring retail performance. It connects sales, branch profitability, inventory coverage, customer analysis, and recommendations in an Arabic dashboard.

The assistant combines Arabic parsing rules, conversation context, optional Groq planning, and deterministic calculations. **Groq does not supply the business numbers:** results are calculated from the selected dataset and shown with their period and scope.

## ✦ What's Inside

| Page | Purpose |
|:---|:---|
| Overview | Sales KPIs, trends, and prioritized business issues |
| Sales | Branch, governorate, category, product, and channel analysis |
| Branches | Branch comparisons and profitability where expense data is available |
| Inventory | Stock coverage, shortage/overstock signals, suggested transfers |
| Forecasting | Company, category, branch, and product forecasts with measured uncertainty |
| Customers | RFM segments, purchasing history, and model-quality disclosures |
| Recommendations | Evidence, proposed actions, and clearly labeled estimated impact |
| AI Assistant | Arabic questions, period comparisons, profit analysis, and follow-up context |
| Data Health | Data-quality findings, model diagnostics, and explicit rebuild controls |

Filters apply according to each page's stated scope. Recommendations and forecasting are not automatically restricted by every sales-page filter.

## ✦ How the Assistant Works

| Step | What happens |
|:---|:---|
| **1. Understand** | Identify metric, entities, dates and dependencies; a selected branch set stays bound to its comparisons and per-branch plans |
| **2. Keep context** | Supported follow-ups inherit the previous governorate, period, and filters; explicit new instructions take precedence |
| **3. Calculate** | Local modules calculate totals, period changes, and branch contributions from filtered records |
| **4. Explain** | Locally calculated facts are separate from Groq's conditional interpretations; generated analysis is reviewed against the evidence |
| **5. Decide and disclose** | Per-branch plans explain the recorded issue, uncertainty and operational priority. Local alternatives and unavailable AI interpretation are explicitly labelled |

Profit value in assistant answers means **sales profit before payroll and operating expenses**, unless another measure is explicitly stated. It is different from profit-margin percentage.

Example prompts — actual results depend on the available data:

```text
قارن مبيعات محافظة الإسكندرية آخر 60 يوم بالـ60 يوم اللي قبلهم.
عايز القيمتين والفرق بالجنيه والنسبة وأكتر 3 فروع أثرت على التغيير.

طيب والربح في نفس الفترة؟ عايز قيمة الربح في الفترتين والفرق
بالجنيه والنسبة وأكتر 3 فروع أثرت عليه. متجيبليش هامش الربح.

اختار أضعف فرعين من حيث نمو المبيعات خلال آخر 12 شهر،
وقارن أداء كل فرع بمخزونه وخصوماته وهامش ربحه.
اقترح خطتين مختلفتين، واحدة لكل فرع، وحدد أولوية التنفيذ وليه.
```

The assistant has specific parsing rules and is still being improved; arbitrary wording is not guaranteed. Review the displayed metric, dates, scope, and source. Relative dates use the **last date in the dataset**, not necessarily today. Requests for the last N months use calendar-month offsets ending at the last recorded date.

## ✦ Forecasting

- Forecasts are available at company, category, branch, and product levels.
- Temporal backtesting uses 12 origins, selecting on older periods and evaluating on later ones.
- Revenue is derived from expected quantity and price, with a configurable January price-increase assumption.
- Forecast intervals, historical average error, and reliability labels are separate indicators; none guarantees future accuracy.
- The current incomplete month can be estimated explicitly. Results change when the authorized dataset is rebuilt.

## ✦ Quick Start

Python 3.13 is the validated release interpreter. Use `requirements.lock` in an isolated environment; `requirements.txt` retains the development minimum versions.

```bash
git clone https://github.com/morad-elnahla/HVIA_Cottonil.git
cd HVIA_Cottonil
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and start:

```bash
python -m pip install -r requirements.lock
streamlit run app.py
```

Open [localhost:8501](http://localhost:8501).

**The repository includes the owner-approved processed dataset and saved forecasts.** A Git checkout runs without uploading or rebuilding data. Raw source files and credentials remain excluded. Source-only delivery ZIPs still omit data unless the explicit data-inclusion option is used.

## ✦ Environment & Groq

Groq is optional. Local analysis works without an API key.

Copy `.env.example` to `.env` only if `.env` does not already exist. Add your own key locally; never commit it.

```dotenv
LLM_API_KEY=
LLM_MODEL=openai/gpt-oss-120b
LLM_NARRATE=auto
LLM_TPM_LIMIT=7000
```

| Setting | Purpose |
|:---|:---|
| `LLM_API_KEY` | Your private Groq key; leave empty for local-only operation |
| `LLM_MODEL` | Model identifier; must be available to your account |
| `LLM_NARRATE` | `auto`, `on`, or `off` for optional narrative summaries |
| `LLM_TPM_LIMIT` | Local token-budget guard; does not override provider limits |
| `LLM_ENABLED` | Set `false` to disable external LLM calls |
| `LLM_BASE_URL` | Optional endpoint override; default `https://api.groq.com/openai/v1` |
| `LLM_PLANNER_MODEL` | Optional separate planning model |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_MAX_WAIT` | Request timeout and retry controls |

`src/llm.py` reads `.env`; existing environment variables take precedence. Restart Streamlit after configuration or assistant-code changes to clear old cached objects. `BI_USER_NAME`, set in the launch environment, changes the greeting.

When enabled, Groq receives the question, relevant conversation context, and selected calculated results for optional summaries. Use only data permitted for that service. Local diagnostics are written under `logs/`; logs are excluded from GitHub and delivery archives.

## ✦ Data & Rebuilding

There are two supported local setups:

1. Place a complete, authorized processed bundle in `data/processed/` to open the dashboard directly.
2. Supply authorized source CSV files under `data/raw/`, then rebuild explicitly:

```bash
python -m src.data
```

The same rebuild is available on **Data Health**. It rebuilds derived data and models and may take around four minutes, depending on the machine and dataset. **It is not required for every startup.**

Core source tables are `sales_orders`, `sales_order_items`, `branches`, and `products_catalog`. Additional source tables support inventory, customers, returns, payroll, and deliveries. Processed forecasts require `forecast.parquet`, `fquality.parquet`, and `backtest.parquet`; the delivery script validates the complete processed bundle when inclusion is requested.

Raw data, processed data, customer records, local metadata, credentials, and logs are not published in this repository. Missing payroll or unreliable source fields can limit profitability and other analyses; read the in-app quality notes. Customer-model quality and forecast uncertainty should not be treated as guaranteed predictions.

## ✦ Private Delivery ZIP

The script uses only Python's standard library and a fixed file allowlist. It preserves the folder layout, validates the selected files, checks for common credential patterns, and verifies ZIP integrity. It does not rebuild data, train models, call APIs, or change application code.

```bash
# Source-only ZIP on your Desktop (safe default; business data supplied separately)
python package_delivery.py

# Approved private delivery: includes the complete existing processed bundle
python package_delivery.py --include-processed --confirm-data-sharing

# Inspect the selection without creating a ZIP
python package_delivery.py --include-processed --confirm-data-sharing --dry-run
```

Windows Desktop redirection, including OneDrive, is respected. `--output-dir PATH` optionally overrides the destination. Archives include a `Cottonil_BI/` top-level folder and a delivery note. Raw data, `.env`, logs, caches, virtual environments, and ad hoc API utilities are never included. Offline regression tests and release checks are included. No ZIP is uploaded to GitHub.

**Only use the data-inclusion option when all processed files are approved for the specific recipient.** Processed data can still contain private customer and commercial information. A source-only ZIP is not a populated, data-ready dashboard.

## ✦ Project Structure

```text
HVIA_Cottonil/
├── app.py                    # Streamlit entry point and pages
├── requirements.txt          # Runtime dependencies and existing test dependency
├── requirements.lock         # Exact versions validated in isolated Python 3.13
├── README.md
├── DEPLOYMENT.md             # Private-host setup, container option and rollback
├── RELEASE_REPORT.md         # Test evidence and remaining limits
├── release_check.py          # Offline startup preflight
├── live_acceptance.py        # Real Groq check, explicit --live opt-in
├── Dockerfile                # Optional Linux container; see verification limits
├── .env.example              # Empty key; safe configuration template
├── .gitignore
├── package_delivery.py       # Allowlisted private-delivery ZIP builder
├── .streamlit/config.toml    # Dark theme and Streamlit settings
├── photo/1.png               # Dashboard preview
├── src/
│   ├── assistant.py          # Intent parsing, context, and grounded answers
│   ├── llm.py                # Groq client, retry handling, and local budgets
│   ├── health.py             # Analytical impact of existing data-health checks
│   ├── data.py               # Loading, source processing, quality checks
│   ├── metrics.py            # Business metrics and change attribution
│   ├── models.py             # Forecasting and customer models
│   ├── inventory.py          # Coverage and transfer logic
│   ├── insights.py           # Recommendations
│   └── ui.py                 # Shared presentation helpers
└── tests/                   # Offline regression, app, transport and release checks
```

`data/`, `logs/`, and `.env` remain local and are ignored by Git.

## ✦ Release Status & Validation

The release candidate includes an exact dependency lock, offline regression tests,
a read-only preflight, a complete source archive, and deployment instructions.
See [RELEASE_REPORT.md](RELEASE_REPORT.md) for measured results and outstanding host checks.

**Verified on 2026-10-31:** 260 tests passed in an isolated Python 3.13.13
environment; two model-training/selection tests were deliberately excluded.
Checks include all nine Streamlit pages, raw-data financial reconciliation,
topic transitions, negation, branch selection, calendar periods and independent
branch plans. A real Groq run of the two-branch manager question completed with
two selected branches, two locally evidenced plans and no split answer parts.
Passing mocked tests is not treated as proof of a live connection.

For compound analysis, factual comparisons are computed locally. Groq generates
conditional interpretations and a proposed plan, followed by an independent
review against the same evidence. Generated prose may not add quantities,
including quantities written as Arabic words, or restate ranked branch claims.
The local comparison table owns those facts. If either stage fails, the response
states that only local evidence is available; it never presents an unchecked draft.
For per-branch decisions, evidence-based local plans remain available even when
AI interpretation is unavailable. Priorities are operational, not a guaranteed
ROI ranking or an optimized budget allocation.

Deploy privately or behind an authenticated HTTPS gateway. This repository does
not provide application login or claim that arbitrary questions are always understood.
