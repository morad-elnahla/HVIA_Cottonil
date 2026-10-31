# Private deployment and handoff

Use Python 3.13 and `requirements.lock` for the validated dependency versions.
The application contains business/customer data views and does **not** implement
user authentication. Keep it on localhost/private infrastructure, or place it
behind the hosting provider's authenticated HTTPS gateway. Do not expose an
unauthenticated public Streamlit port.

## Install in an isolated environment

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip check
.venv/Scripts/python release_check.py
.venv/Scripts/python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true
```

On Linux use `.venv/bin/python` in the commands above. Supply the existing,
authorized `data/processed/` bundle separately. Do not rebuild or retrain at
startup. `release_check.py` checks the bundle and imports; it never calls Groq.

Set `LLM_API_KEY` (or `GROQ_API_KEY`) using the host's secret store/environment.
Other settings are documented in README. Leave the key empty for local mode.
Do not put `.env`, customer records, or diagnostic logs in source archives/images.
The host needs writable `logs/` and enough memory for the processed sales table.
Validate memory usage with the intended concurrent user count before expanding access.

## Optional container

```sh
docker build -t cottonil-bi:release .
docker run --rm -p 127.0.0.1:8501:8501 \
  --mount type=bind,src=/absolute/private/data/processed,dst=/app/data/processed,readonly \
  --env LLM_API_KEY --env LLM_MODEL --env LLM_TPM_LIMIT cottonil-bi:release
```

The data mount is read-only: do not use the UI rebuild controls in this mode.
The image runs as UID 10001; give it read permission on the mounted data.
Docker daemon availability and Linux image execution must be checked on the
target host. See `RELEASE_REPORT.md` for what was actually verified here.

## Acceptance and rollback

1. `GET /_stcore/health` returns HTTP 200.
2. Open each of the nine pages; confirm dates and intended data access.
3. Run the offline suite below (it blocks network calls).
4. With approved Groq access, test one combined analysis and inspect its facts,
   branch scope, explanations and action feasibility. Model responses can vary.
5. Keep the previous source archive and data snapshot. Roll back code and its
   dependency environment together; restart Streamlit to clear cached resources.

```sh
python -m pytest tests -q -k "not model_selection_uses_older_origins and not models_recover_known_seasonality"
```

Synthesis uses at most two logical requests: draft and independent evidence
review, plus the client's bounded transport retries. Both obey the configured
local token budget. Unsupported quantitative action targets may be replaced
individually with labelled local verification actions; this is disclosed in the
answer's mode and notes. If the analysis review fails or the service is unavailable, the UI
explicitly shows local evidence without claiming a completed AI analysis.
This is not a guarantee that every natural-language question is supported.
