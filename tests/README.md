# Offline assistant regression

Run the focused suite from the project root:

```powershell
python -m pytest tests/test_assistant_regression.py -q
```

The suite covers local routing, scope inheritance, independent topic changes,
compound requests, negation, result counts (digits, words and Arabic dual forms),
financial comparisons, customer metrics, inventory, forecasts, recommendations,
data-health checks, renderer/RTL contracts, and mocked Groq transport failures.
It includes all 64 ordered transitions between the eight business sections.

Fixtures are synthetic except for one smoke test that reads the existing
`data/processed` snapshot and uses the existing recommendation engine. That test
is skipped if the processed data is absent. No test trains a model or rebuilds
data. `conftest.py` blocks real requests and socket connections; Groq tests use
in-memory transports. Passing mocks does **not** verify a real Groq connection.

The complete historical `test_core.py` suite also contains model-training and
full-app tests; do not run it unfiltered when only lightweight regression is
desired. Its assistant and transport checks can be selected separately:

```powershell
python -m pytest tests/test_core.py -q -k "retry_after or long_rate_limit or timeout_and_network or failures_become or auth_error or cache_and_local or unknown_branch or multi_part_local or gap_question or named_branches or assistant_understands or forecast_answers or bidi_isolates or groq_ or explicit_multi_part"
```

Known limits: local language parsing is intentionally conservative, unsupported
scopes should be reported rather than silently widened, forecast validation is
at its saved aggregation level, and health-check severity is the recorded
status (not an invented financial risk score). DOM/HTML tests verify escaping,
number direction and completeness, not appearance on every browser/device.
