"""
FastAPI surface for GridWise.

Endpoints:
  GET  /health          — readiness (per Problem Statement §06)
  GET  /healthz         — liveness alias
  POST /optimize-energy — main scoring endpoint
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .llm import LLMInterpretationError, OLLAMA_MODEL, OLLAMA_URL, ollama_headers
from .optimizer import InfeasibleSchedule, SolverFailure
from .response_builder import InvalidSchedule
from .schemas import OptimizeRequest, OptimizeResponse
from .solver import solve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise",
    description="LLM-assisted 24-hour energy scheduler — BUP CSE Fest 2026",
    version="2.0.0",
)


def _ollama_status() -> bool:
    """Quick probe — is the configured Ollama model reachable?"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", headers=ollama_headers(), timeout=2)
        if not r.ok:
            return False
        data = r.json()
        configured = OLLAMA_MODEL if ":" in OLLAMA_MODEL else OLLAMA_MODEL + ":latest"
        return any(m.get("name") == configured or m.get("model") == configured
                   for m in data.get("models", []))
    except Exception:
        return False


@app.get("/health")
def health():
    """Readiness requires the configured model; successful body matches the spec."""
    if not _ollama_status():
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> dict:
    """Liveness alias — always 200 while the process is up."""
    return {"status": "ok"}


@app.exception_handler(InfeasibleSchedule)
async def _infeasible_handler(_request: Request, exc: InfeasibleSchedule) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error": "infeasible_schedule"},
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(_request: Request, exc: RequestValidationError):
    # Do not echo submitted values, exception contexts, NaN or infinity into JSON.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
              for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errors, "error": "invalid_request"})


@app.exception_handler(LLMInterpretationError)
@app.exception_handler(SolverFailure)
@app.exception_handler(InvalidSchedule)
async def _service_failure_handler(_request: Request, exc: Exception):
    log.error("Request failed (%s)", type(exc).__name__)
    return JSONResponse(status_code=500, content={
        "detail": "Unable to produce a validated schedule. Check model and solver availability.",
        "error": "schedule_unavailable",
    })


def _strip_kind(optimize_response: OptimizeResponse) -> Dict[str, Any]:
    """Return a JSON-safe dict that omits the Pydantic `kind` discriminator
    from structured_adjustment (the reference output does not include it)."""
    body = optimize_response.model_dump()
    for entry in body.get("directive_interpretation", []):
        adj = entry.get("structured_adjustment")
        if isinstance(adj, dict):
            adj.pop("kind", None)
    return body


@app.post("/optimize-energy")
def optimize_energy(req: OptimizeRequest) -> Dict[str, Any]:
    """Interpret operator notes, solve, replay-check, and return the schedule."""
    t0 = time.perf_counter()
    result = solve(req)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "scenario=%s parse_solve_ms=%.1f total_cost_bdt=%s total_grid_kwh=%s",
        req.scenario_id,
        elapsed_ms,
        result.total_cost_bdt,
        result.total_grid_kwh,
    )
    return _strip_kind(result)
