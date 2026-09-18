"""FastAPI service exposing GET /health and POST /optimize-energy.

Orchestrates the full pipeline (Problem Statement Section 03):
Energy Data + Operator Notes -> LLM Interpreter -> Guardrail Validator ->
Math Optimizer -> Final Validator -> API Response.

Every stage fails safe: a broken/slow LLM call, malformed LLM JSON, or an
infeasible optimization all resolve to a controlled error response, never
a crash and never a silently invented directive (Section 08 SAFE FAILURE).
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import REQUEST_TIMEOUT_SECONDS
from app.guardrails import apply_guardrails
from app.llm_interpreter import LLMInterpretationError, interpret_notes
from app.optimizer import OptimizationInfeasibleError, solve
from app.replay_validator import replay_validate
from app.schemas import HealthResponse, OptimizeRequest, OptimizeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise.main")

app = FastAPI(title="GridWise LLM Optimizer", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Problem Statement Section 06.1: malformed JSON or structurally invalid
    # request -> 400. FastAPI's default for this is 422; we align with the
    # canonical spec explicitly rather than rely on the framework default.
    return JSONResponse(status_code=400, content={"detail": "malformed or structurally invalid request"})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Controlled internal error: never leak stack traces or secrets
    # (Section 06.1, 500; Participant Guide "Secret handling").
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _run_pipeline(payload: OptimizeRequest) -> OptimizeResponse:
    num_notes = len(payload.operator_notes)

    try:
        raw_llm_output = interpret_notes(payload.operator_notes, payload.battery)
    except LLMInterpretationError as exc:
        logger.warning(
            "scenario %s: LLM interpretation unavailable, falling back to no_op for all notes: %s",
            payload.scenario_id,
            exc,
        )
        raw_llm_output = None  # apply_guardrails safely turns this into all-no_op

    guardrail_result = apply_guardrails(raw_llm_output, num_notes, payload.battery)
    if guardrail_result.fallback_note_indices:
        logger.info(
            "scenario %s: notes %s fell back to no_op after guardrail validation",
            payload.scenario_id,
            guardrail_result.fallback_note_indices,
        )

    optimization = solve(payload.hours, payload.battery, guardrail_result.directives)

    violations = replay_validate(
        payload.hours,
        payload.battery,
        guardrail_result.directives,
        optimization.hourly_plan,
        optimization.total_grid_kwh,
        optimization.total_cost_bdt,
        optimization.peak_grid_kwh,
    )
    if violations:
        # Should never happen if the optimizer is correct -- defense in
        # depth. Never return a plan known to violate constraints.
        logger.error("scenario %s: replay validation failed: %s", payload.scenario_id, violations)
        raise RuntimeError("optimizer produced a plan that failed replay validation")

    applied_types = sorted(
        {interp.directive_type.value for interp in guardrail_result.interpretations if interp.applies}
    )
    if applied_types:
        summary = f"Applied {', '.join(applied_types)} directive(s) and minimized grid electricity cost over 24 hours."
    else:
        summary = "No operator directive affected today's schedule; minimized grid electricity cost over 24 hours."

    return OptimizeResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=guardrail_result.interpretations,
        hourly_plan=optimization.hourly_plan,
        total_grid_kwh=optimization.total_grid_kwh,
        total_cost_bdt=optimization.total_cost_bdt,
        peak_grid_kwh=optimization.peak_grid_kwh,
        plan_summary=summary,
    )


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(payload: OptimizeRequest) -> OptimizeResponse:
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_run_pipeline, payload),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("scenario %s: request exceeded %.1fs timeout", payload.scenario_id, REQUEST_TIMEOUT_SECONDS)
        return JSONResponse(status_code=500, content={"detail": "request timed out"})
    except OptimizationInfeasibleError as exc:
        logger.error("scenario %s: optimization infeasible: %s", payload.scenario_id, exc)
        return JSONResponse(status_code=500, content={"detail": "no feasible schedule for the given constraints"})


if __name__ == "__main__":
    import uvicorn

    from app.config import PORT

    uvicorn.run(app, host="0.0.0.0", port=PORT)
