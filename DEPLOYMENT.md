# Private deployment and handoff

## Streamlit Community Cloud (automatic data restore)

Git deliberately excludes `data/`, including processed tables and saved forecasts.
Community Cloud clones the repository, not your local disk. A source-only deploy
therefore has no business data until a private snapshot source is configured.

1. Generate the existing data snapshot locally: `python package_cloud_data.py`.
   This creates `dist/cottonil-cloud-data.zip` and its `.sha256` file without
   rebuilding data or retraining. Keep this artifact private; it contains customer
   and business records. Never attach it to a public GitHub release.
2. Store that ZIP at an authorized private HTTPS download endpoint (a direct object
   URL, optionally with a bearer token). The endpoint must return the ZIP, not a
   login/share HTML page. Prefer a durable protected endpoint; signed links must
   remain valid for future cold starts, or be rotated before expiry.
3. In Streamlit app Settings → Secrets, configure these **top-level** entries:

   ```toml
   BI_DATA_URL = "https://your-private-host/cottonil-cloud-data.zip"
   BI_DATA_SHA256 = "paste-the-64-character-hash-from-the-sha256-file"
   # BI_DATA_TOKEN = "private-download-token-if-required"
   LLM_API_KEY = "your-Groq-key"
   LLM_MODEL = "your-tested-Groq-model"
   ```

   Do not put secrets in Git or send them in chat. Streamlit exposes top-level
   secrets as environment variables, which the existing Groq settings read.
4. Deploy `app.py`, branch `main`, with Python **3.13**. `requirements.txt` now
   installs the exact versions in `requirements.lock`. Set authorized viewer
   access in Cloud before making business/customer views available.
5. Restart the app. It downloads once per snapshot, checks SHA-256, validates all
   15 files, then activates a versioned local cache. It restores after a fresh
   host start too. All pages share the same snapshot, including forecast,
   backtest, forecast quality, churn and health results. No training runs.

For updates, publish a new ZIP and change URL/hash together, then reboot. A failed
download or checksum displays a safe diagnostic and does not substitute fake data
or overwrite local data. Hosted snapshot mode disables manual upload/retraining
controls. Existing local mode still works without these settings.

Offline cloud simulation: `python -m pytest tests/test_bootstrap.py -q`. This
uses a mocked download, real saved data and all nine Streamlit pages. It does
not prove hosting credentials, remote access, Cloud memory limits or real Groq
connectivity. Verify those separately on the deployed application.

Official setup: [Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

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
