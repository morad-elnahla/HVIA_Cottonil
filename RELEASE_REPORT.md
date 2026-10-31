# Release verification — 2026-10-31

Status: tested **private deployment candidate**, not a published production service.

## Verified

- Windows, isolated Python **3.13.13** environment installed from `requirements.lock`;
  **48 pinned distributions**. `pip check`: no broken requirements.
- Final offline command: `python -m pytest tests -q --disable-warnings -k "not model_selection_uses_older_origins and not models_recover_known_seasonality"`.
  **260 passed, 2 deselected, 28 warnings, 105.09 seconds**. The two excluded tests
  exercise model training/selection; the stored forecasting models were not retrained.
- Includes all nine Streamlit pages via AppTest, filter interaction, assistant
  entry, RTL/escaping, independent raw-data financial checks, scope and memory,
  section transitions, cardinality, mocked transport failures, source archive
  integrity and missing-data startup. The run emitted 28 non-failing warnings;
  omit `--disable-warnings` to inspect their details.
- Actual headless Streamlit startup in the isolated environment: application
  root **HTTP 200**, `/_stcore/health` **HTTP 200**, body `ok`. The temporary test
  server was stopped after checking it.
- The generated source ZIP was extracted into a separate directory. Its runtime
  imports passed and its no-data setup screen ran under AppTest without exceptions.
- **Real Groq acceptance** with `openai/gpt-oss-120b`: the reported weakest-three-
  branches question returned three distinct branches and three actions after
  evidence review. The final successful run used **2 successful requests / 4,423
  reported tokens**. Subsequent customer, two-recommendation and three-data-health
  questions switched correctly, with no additional API calls.
- That live response used a **declared local replacement for an unsafe generated
  action**, not an entirely model-authored plan. Its mode and per-action labels
  disclose this. Live intermediate failures exposed hallucinated quantitative
  targets and factual branch comparisons; they were not counted as successes.

## Changes that matter

### Entity-workflow correction (2026-10-31)

The subsequent manager/limited-budget question exposed a gap in the earlier
acceptance set: selection followed by per-branch comparison/plans was being
split when no explicit relationship keyword was present. That prior acceptance
did not establish support for this workflow.

The corrected path binds one selected branch set to all dependent evidence and
decisions. It handles Arabic dual cardinality, ranks by the metric in the
selection clause rather than later requested margin fields, and treats one
plan per branch separately from a flat list of steps. Calendar-month requests
no longer use a 30-day approximation. Discount evidence supplies margin rates
without a duplicate profit-value tool/citation requirement.

The exact new question was also exercised through real Groq: two branches
(`BR010`, `BR078`), two rows, two independent locally evidenced priority plans,
zero split answer parts; **2 successful requests / 3,368 tokens**, with no
fallback warning. Groq supplies conditional interpretation; the app explicitly
labels branch selection, plans and operational priorities as local. These
priorities are not a computed financial optimum: intervention costs are absent.

Regression cases now include introductory manager context, alternate wording,
dual/plural counts, growth-versus-value ranking, requested margin fields,
independent branch plans, retained follow-up scope, and an independent ranking
check against the saved sales data. See `logs/manager_branch_result.json` and
`logs/manager_branch_acceptance.txt` locally for the live check (not archived).

- Rank/select branches before gathering their sales, stock and discount evidence.
  Result cardinality and plan-step cardinality are independent.
- Render one locally calculated comparison table, preserving actual versus
  estimated quantities, periods and the meaning of relative underperformance.
  Descriptive comparisons between coverage and growth are computed locally.
- Groq drafts conditional interpretations and actions; an independent evidence
  review precedes publication. Generated prose cannot add quantities (including
  Arabic number words) or restate branch rankings. Isolated unsupported action
  targets are replaced with explicitly labelled, evidence-related local checks;
  malformed or unsupported analysis is still rejected honestly.
- Compact evidence and bounded response budgets; model-specific reasoning effort
  for GPT-OSS according to [Groq's documented support](https://console.groq.com/docs/reasoning).
  Separate JSON/truncation, transport and semantic validation diagnostics without
  recording credentials or raw model text in the production logger.
- Fix delivery allowlist omission of `src/health.py`; include tests, preflight,
  locked dependencies, opt-in live acceptance, and deployment instructions.
- Offline tests use an isolated logger, so mock errors no longer pollute live logs.

## Reproduce

```powershell
python release_check.py
python -m pip check
python -m pytest tests -q -k "not model_selection_uses_older_origins and not models_recover_known_seasonality"
# Explicit opt-in: this consumes real Groq tokens and sends aggregate evidence.
python live_acceptance.py --live
python live_acceptance.py --live --scenario branch-plans
python package_delivery.py --output-dir dist
```

`logs/release_live_result.json`, `logs/live_acceptance_answer.txt` and diagnostic
drafts remain local and are excluded from Git/delivery. They can contain business
information. `.env` and the source/processed data were not edited.

## Remaining host and product limits

- No target server, domain or identity provider was supplied. No public deployment
  was performed. Application-level authentication is absent: use private access
  or an authenticated HTTPS gateway before exposing business/customer views.
- Docker CLI is installed but its Linux daemon was unavailable. The Dockerfile
  is supplied; an actual Linux image build/start has **not** been verified here.
- Source delivery intentionally excludes business data and secrets. Provision
  the complete authorized processed bundle and host secrets separately.
- No concurrent-user load test, penetration test or all-device visual audit was
  performed. Capacity and access controls need validation on the actual host.
- A successful live conversation does not guarantee future model output or
  availability. Unsupported questions, budget limits and service failures remain
  possible and are disclosed in the response. Model prose remains conditional,
  not proof of causality or a guaranteed financial outcome.
# Community Cloud data restore — 2026-10-31

- Root cause: all `data/` artifacts are intentionally ignored by Git; source-only
  deployments had no data source/bootstrap and entered the empty dashboard state.
- Added private HTTPS snapshot restoration with required SHA-256, bounded download
  and extraction, exact file allowlist, Parquet/JSON checks, immutable versioned
  activation and safe error codes. Local data and saved models remain unchanged.
- Cloud installs the validated lock through `requirements.txt`. Existing root-level
  Streamlit secrets supply Groq environment settings. Snapshot mode avoids UI
  uploads/retraining; all pages read the same complete saved bundle.
- Local verification: **268 passed, 2 training tests deselected, 28 warnings**.
  Includes cold start into an empty directory using actual saved data through a
  mocked download, all nine pages, forecast byte equality, cache reuse, bad checksum,
  HTTP authorization rejection, unsafe/incomplete ZIP and invalid content rejection.
- Private deployment artifact generated locally under ignored `dist/`; not published.
- Still unverified: actual Community Cloud deployment/host memory and external data
  credentials, plus live Groq on that host. Need the app URL and authorized private
  snapshot endpoint/secrets. Offline mocks are not proof of remote connectivity.

## Approved bundled-data deployment — 2026-10-31

The owner explicitly authorized public publication of the existing dataset. The
15 processed runtime files are tracked unchanged; raw inputs, credentials and logs
remain excluded. This supersedes the private-source requirement above for the
Git-based deployment. Tested an isolated export of the staged Git tree without
.env or external data configuration: all nine pages rendered, and saved forecast,
quality and churn results loaded successfully. No Groq calls or training ran.
The actual Cottonil Cloud URL remains unknown: the supplied URL serves FinCon.
