"""LLM interpreter stage: Energy Data + Operator Notes -> **LLM Interpreter**
-> Guardrail Validator -> Math Optimizer -> Final Validator -> API Response.

Calls a Groq-hosted language model (default: openai/gpt-oss-120b) to turn
1-3 natural-language operator notes into a draft, untrusted
``directive_interpretation`` array. This module's output is never trusted
directly -- app.guardrails validates every field before anything reaches
the optimizer (Problem Statement Section 08, "CORE IDEA").

Latency design: the Participant Guide scores p95 latency in bands
(<=5s: 3/3, 5-15s: 2/3, 15-30s: 1/3, >30s/timeout: 0/3). We therefore use a
short per-call timeout and at most one retry, rather than long/multiple
retries -- a slow-but-eventually-correct response still loses latency-tier
points and risks the hard 30s request cutoff.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_TIMEOUT_SECONDS
from app.schemas import BatterySpec

logger = logging.getLogger("gridwise.llm_interpreter")

_SYSTEM_PROMPT = """You are the operator-note interpretation stage of GridWise, a campus \
energy scheduling system. Your ONLY job is to read short natural-language notes from \
campus operators and convert each one into a machine-checkable directive, OR mark it as \
not relevant to today's 24-hour energy schedule.

You do not see demand, solar, or tariff numbers, and you must never invent any -- a later, \
separate optimization step already has the real hourly data and applies your directives to it.

SUPPORTED DIRECTIVE TYPES (use exactly one of these six for every note):

1. solar_reduction - usable solar generation is reduced during specific hours.
   structured_adjustment: {"hours": [int, ...], "factor": number}
   "factor" is the USABLE FRACTION THAT REMAINS, not the reduction amount.
   Example: a note saying solar drops to about 20% of normal, or "an 80% reduction",
   both mean factor = 0.2.

2. minimum_battery_reserve - the battery must be kept at or above a level during specific hours.
   structured_adjustment: {"hours": [int, ...], "minimum_energy_kwh": number}
   If the note gives a percentage of battery capacity (e.g. "keep at least 50% of capacity"),
   convert it to an absolute kWh number using the battery capacity given below in CONTEXT DATA.
   If the note already gives an absolute kWh number, use it directly.

3. no_charge_window - the battery must not charge during specific hours.
   structured_adjustment: {"hours": [int, ...]}

4. no_discharge_window - the battery must not discharge during specific hours.
   structured_adjustment: {"hours": [int, ...]}

5. max_grid_window - grid electricity import must not exceed a stated amount during specific hours.
   structured_adjustment: {"hours": [int, ...], "max_grid_kwh": number}

6. no_op - the note does not affect today's 24-hour energy schedule (e.g. it is about something
   unrelated to power/battery/solar/grid operations, or about a different day/date).
   structured_adjustment: null

TIME RULES:
- Convert any clock time, hour range, or relative time-of-day phrase into a hours array of
  whole-hour integers from 0 to 23, in ascending order, with no duplicates.
- Time windows are START-INCLUSIVE and END-EXCLUSIVE. "1 PM to 3 PM" means hours [13, 14]
  (hour 15 / 3 PM itself is excluded). "Noon until 2 PM" means hours [12, 13]. Midnight is hour 0.
- Every hour in the array must be a plain integer 0-23. Never wrap past hour 23.

STRICT RULES:
- Every note maps to EXACTLY ONE of the six directive types above. Never invent a new type.
- For every note that maps to a real directive (not no_op): "applies" must be true and
  "structured_adjustment" must be a JSON object with exactly the required shape for that type.
- For no_op: "applies" must be false and "structured_adjustment" must be null.
- Never invent or alter demand, solar, tariff, or battery numbers beyond what CONTEXT DATA gives
  you and what the note itself states. Only use CONTEXT DATA to convert a percentage/relative
  phrase into an absolute number when the directive type requires one.
- If a note is ambiguous or ordinary distractor text unrelated to today's power schedule
  (e.g. an administrative announcement, a booking change, a date far in the future), mark it
  no_op rather than guessing a directive.
- Respond in the exact note order given, one entry per note.

OUTPUT FORMAT:
Respond with ONLY a single JSON object, no markdown fences, no commentary, of this exact shape:
{"directive_interpretation": [
  {"note_index": 0, "applies": true|false, "directive_type": "<one of the six types>",
   "structured_adjustment": {...} or null, "explanation": "short human-readable reason"},
  ...
]}
Return exactly one entry per input note, in note_index order starting at 0.
"""


def _build_user_prompt(operator_notes: list[str], battery: BatterySpec) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))
    return (
        "CONTEXT DATA (use only to convert relative/percentage language to absolute numbers; "
        "never repeat these as if they were a directive on their own):\n"
        f"battery.capacity_kwh = {battery.capacity_kwh}\n"
        f"battery.minimum_energy_kwh = {battery.minimum_energy_kwh}\n"
        f"battery.initial_energy_kwh = {battery.initial_energy_kwh}\n\n"
        f"OPERATOR NOTES ({len(operator_notes)} total, note_index: text):\n"
        f"{notes_block}\n\n"
        "Interpret every note above and return the JSON object described in your instructions."
    )


class LLMInterpretationError(Exception):
    """Raised only when no usable JSON could be obtained from the LLM after
    retries. Callers must treat this as a safe-failure signal (fall back to
    no_op for every note), never as a reason to crash the request."""


def _client() -> Groq:
    if not GROQ_API_KEY:
        raise LLMInterpretationError("GROQ_API_KEY is not configured")
    return Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)


def _call_once(client: Groq, operator_notes: list[str], battery: BatterySpec, *, use_json_mode: bool) -> Any:
    kwargs: dict[str, Any] = dict(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(operator_notes, battery)},
        ],
    )
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("empty completion content")
    return json.loads(content)


def interpret_notes(operator_notes: list[str], battery: BatterySpec) -> Any:
    """Return the raw, still-untrusted parsed JSON from the LLM.

    Tries JSON mode first, retries once (optionally without JSON mode, in
    case the model/provider combination rejects that parameter). Raises
    LLMInterpretationError only if both attempts fail to produce any
    parseable JSON -- callers must treat that as "no directives extracted",
    not as a reason to fail the whole request.
    """
    client = _client()
    last_error: Exception | None = None

    for attempt, use_json_mode in enumerate((True, False)):
        try:
            return _call_once(client, operator_notes, battery, use_json_mode=use_json_mode)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we always fail safe
            last_error = exc
            logger.warning("LLM interpretation attempt %d failed: %s", attempt + 1, exc)

    raise LLMInterpretationError(str(last_error) if last_error else "unknown LLM failure")
