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
# Visualization follow-ups — 2026-11-02

- Root cause: conversational state stored filters and truncated text, not the
  structured last answer. Presentation-only requests reached ambiguous routing.
- Store a copy of only the latest answer per assistant session. Chart follow-ups
  reuse its exact tables/figures, sources and caveats without Groq or data tools.
  Explicit new subjects/periods/entities use normal routing; nonnumeric latest
  results never fall back to an unrelated older analysis. Negated chart requests
  remain normal requests. Clearing chat also clears the saved result.
- Plotly subplots use independent axes for measured columns, preserving units,
  branch labels and missing values. Streamlit renders the actual figure with a
  stable message key and sufficient height, without mutating saved figures.
- Offline suite: 277 passed, 2 training tests deselected, 32 warnings. Includes
  real two-branch analysis followed by the reported chart request, exact plotted
  values, paraphrases, topic change, no history/nonnumeric results, negation,
  repeat rendering and an actual Streamlit chart element. No API calls or training.
- Live Cloud rendering and Groq connectivity were not tested for this change.
- Streamlit chart API: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart

## Customer actions and prioritization — 2026-11-02

Customer overview handlers previously omitted requested actions, and dependent
customer fields could be decomposed into repeated analyses. Customer analysis now
keeps dependent action requirements together and generates evidence-conditioned
options with their rationale and explicit uncertainty. Structured options support
prioritization follow-ups before decomposition, reusing the saved scope and facts
without Groq or data-tool reruns. Explicit topic/period/entity changes continue
through normal routing. Options are operational suggestions, not measured ROI.

Validation: 280 offline tests passed, two training tests deselected, 32 warnings.
The reported pair was also run against the actual saved dataset: three actions,
one selected priority referencing 19.23% unidentified sales, identical source and
scope, no split, no API calls. Regression cases include topic switching and full
customer identification (no unsupported missing-ID action). Cloud execution was
not tested for this change.

## Narrative requests and decision evaluation — 2026-11-02

Separated referential analysis clauses from independent data commands. Customer
narratives now retain their evidence, scope and action requirements as one
operation across punctuation variants. Financial profit/margin requests remain
separate measured quantities. Decision follow-ups recognize selection language
and singular/postfix action counts; future evaluation horizons no longer overwrite
historical filters. Evidence-linked decision options include measurement protocols,
with historical baselines distinguished from future outcomes and unmeasured costs.

Validation: full offline suite 293 passed, 2 training tests deselected, 32 warnings;
one additional real-data regression passed independently after that run. Coverage
includes four punctuation variants, four domain references, independent commands,
cardinality versus horizon, and the exact reported two-turn real-data interaction.
No API calls, data changes or training. Cloud behavior remains unverified. The
planner remains rule-assisted, not a guarantee of unrestricted language understanding.

## Generated-action follow-up persistence — 2026-11-02

Root causes: reviewed synthesis actions lacked the decision metadata created by
local customer handlers; solution synonyms and `during 90 days` bypassed the
follow-up route. Unified action persistence runs for every displayed answer and
hydrates older saved responses. Measurement horizons resolve before decomposition;
action count and KPI count remain separate. Synthesis prompts now condition
inventory/discount advice on the requested domain. Sidebar revision reads actual
checkout HEAD, rather than assuming GitHub HEAD is deployed.

Validation: 296 offline tests passed, 2 training tests deselected, 32 warnings.
Real saved-data tools were exercised with reviewed model responses replayed offline.
Additionally, the exact two-turn scenario was submitted once to the actual Cloud
app: revision 4472f68 was visibly verified; first response was Groq synthesis,
follow-up selected an actual prior action with three KPI rows and 90-day evaluation
without recommendations/sales rerouting. The live test exposed overly generic
mixed-domain topic metadata; a subsequent targeted correction gives customer-source
ownership precedence and preserves the original missing-ID fact in the rationale.
Three affected regressions passed after that correction; no second live Groq call
was made for the correction. No data/model changes or retraining.
