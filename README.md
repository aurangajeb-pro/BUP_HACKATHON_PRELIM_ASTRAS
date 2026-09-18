# GridWise

BUP CSE Fest 2026 preliminary: a FastAPI service that interprets 1-3 operator
notes with a real LLM, validates the extracted directives, and computes a
cost-optimal 24-hour battery schedule with PuLP/CBC.

## Local setup (Python 3.11 or 3.12)

```bash
git clone https://github.com/aurangajeb-pro/BUP_HACKATHON_PRELIM_ASTRAS.git
cd BUP_HACKATHON_PRELIM_ASTRAS
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Choose one model connection below, then start the API:

```bash
python -m uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000
```

On Linux/macOS or Git Bash, `bash run.sh` creates the environment, installs
requirements, reads `.env`, and starts the API. Model setup is still required.

### Option A: existing Ollama installation

Install [Ollama](https://ollama.com/download), start the Ollama app/server,
and authenticate for cloud inference:

```bash
ollama signin
ollama pull gemma4:31b-cloud
ollama run gemma4:31b-cloud "Reply with ready"
```

The default `.env` points to `http://127.0.0.1:11434` and
`gemma4:31b-cloud`. A working cloud login and available quota are required.
To use a local model, pull that model and set `OLLAMA_MODEL` to its exact tag.
Warm up local models before evaluating request latency.

### Option B: direct Ollama cloud API

An Ollama installation is not needed for this path. Create your own API key
and edit the untracked `.env`:

```dotenv
OLLAMA_URL=https://ollama.com
OLLAMA_MODEL=gemma4:31b
OLLAMA_API_KEY=your-own-key
OLLAMA_FORMAT=auto
```

Use the direct API model name rather than the local `-cloud` alias. See the
[official cloud setup](https://docs.ollama.com/cloud). Never commit `.env`.

## Call the API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' --data-binary @sample_input.json
```

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/optimize-energy -Method Post -ContentType 'application/json' -InFile sample_input.json
```

`sample_input.json` contains official SAMPLE-01. With the correct live
interpretation, its expected total grid cost is **38365 BDT**. The complete
10-case organiser fixture, including expected responses, is bundled unchanged
at `tests/data/public_sample_cases.json`.

- `GET /health`: HTTP 200 and exactly `{"status":"ok"}` when the configured
  model is listed by Ollama; HTTP 503 when it is unavailable. This checks model
  discovery, not an inference call; verify authentication/quota with the sample POST.
- `GET /healthz`: process liveness, always `{"status":"ok"}`.
- `POST /optimize-energy`: accepts `scenario_id`, `operator_notes`, `hours`,
  and `battery`. Returns `directive_interpretation`, `hourly_plan`,
  `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`, and
  the original `scenario_id`.
- HTTP 400: malformed JSON or invalid request data. HTTP 422: infeasible
  schedule. HTTP 500: controlled model, solver, or schedule-validation failure.
  An unavailable model never silently turns applicable instructions into `no_op`.

## Tests

From an activated virtual environment:

```bash
python -m pytest tests/ -q
```

The default suite is fully offline. It injects the organiser's expected
interpretations at the model boundary for the 10 public math scenarios, then
checks optimal cost, hourly energy balance, battery limits, directive
application, neutrality, and totals through the actual API/solver. It also
covers fractional battery drift, input ordering, invalid/non-finite inputs,
malformed model output, repair retries, provider failures, and solver errors.
These tests **do not measure LLM interpretation accuracy**.

Run the separate real-model tests after configuring Ollama. Pytest does not
automatically read `.env`; export the variables or use:

```bash
python -m dotenv -f .env run -- python -m pytest tests/ --run-llm-tests -q
```

The 21 live tests are skipped by default. When explicitly enabled, an
unavailable configured model fails the live test gate instead of counting as a
pass. Equivalent optimal schedules can have different peaks; the tests compare
optimal cost and independently recompute peak import from the returned plan.

## Docker

The Dockerfile builds only the API. To use direct cloud inference, configure
`.env` using Option B, then:

```bash
docker build -t gridwise:reviewed .
docker run --rm --env-file .env -p 8000:8000 gridwise:reviewed
```

For the bundled Ollama service, sign in before using the cloud model:

```bash
docker compose up -d ollama
docker compose exec ollama ollama signin
docker compose up --build -d
```

The `model-init` service pulls the selected model before the API starts.
For a local model, set `OLLAMA_MODEL` in `.env`; sign-in is unnecessary for
local inference. The Ollama health check uses its installed CLI, so it does
not depend on `curl` being present in the image. The compose stack always uses
its own Ollama service; use `docker run` for direct-cloud `.env` settings.

Verify both `/health` and a real `/optimize-energy` request after starting the
container. A publicly pullable image tag/digest and deployed public API URL
still need to be created and submitted by the team. Docker was not available
in the review environment, so an actual image build/run remains unverified.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server, without `/api` |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Exact installed model tag |
| `OLLAMA_API_KEY` | empty | Optional direct-cloud bearer token |
| `OLLAMA_TIMEOUT_S` | `8` | Per-attempt read timeout in seconds |
| `OLLAMA_NUM_PREDICT` | `1024` | Maximum generated tokens |
| `OLLAMA_KEEP_ALIVE` | `30m` | Local model retention |
| `OLLAMA_FORMAT` | `auto` | `auto`, `schema`, `json`, or `none` |

Ollama cloud [does not support structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
In `auto` mode, cloud requests use the JSON prompt without a `format` argument;
local models use the generated JSON schema. Both paths receive exactly the
same deterministic validation. One repair retry is allowed. Defaults allocate
two requests (each with a 2-second connect and 8-second read timeout) plus a
5-second solver limit. Actual end-to-end latency, including cold starts and
provider behaviour, must be measured against the judge's 30-second deadline.
Increasing timeouts can exceed that deadline.

## Algorithm and guardrails

1. The LLM extracts exactly one directive per note: solar reduction, reserve,
   no-charge window, no-discharge window, grid cap, or irrelevant-note `no_op`.
2. Strict Pydantic validation rejects coercions, extra fields, invalid hours,
   non-finite numbers, mismatched adjustment types, and invalid note mappings.
   Failed interpretation is retried once, then returned as a controlled error.
3. PuLP/CBC minimizes `sum(tariff[h] * grid[h])`. Binary variables prevent
   simultaneous charge and discharge, so this is a mixed-integer linear model.
4. Constraints enforce hourly demand balance, solar availability, battery
   capacity/reserve/rates, every directive, and final energy equal to initial
   energy. A time-limited incumbent is not presented as a proven optimum.
5. The final serialized schedule is replayed before it is returned. Energy
   values retain eight decimal places; only the monetary total is rounded to
   two. Totals are calculated from returned rows. Directive factors and limits
   retain their original validated precision.

Input hours may arrive in any order and are sorted internally. Time windows
are start-inclusive and end-exclusive. A solar reduction factor represents the
remaining fraction. For overlapping solar reductions, the smallest factor is
used against the original forecast (a documented assumption where organiser
combination semantics are unspecified). Overlapping reserves take the maximum;
overlapping grid caps take the minimum. There is no grid export and battery
round-trip efficiency is 100%, matching the challenge model.

## Review status and remaining checks

The supplied code's nonportable test path, rounding crash, unsorted-hour
handling, battery-range validation, LLM guardrails/failure handling, model
format compatibility, and container startup configuration were corrected.
Offline tests validate scheduling with known directives; the live model,
provider credentials/quota, Docker runtime, public endpoint, and real request
latency must still be checked in the team's deployment environment.
