"""Prediction service for Spike Watch.

The serving half of a *batch prediction* system in the Lecture 8 sense:
predictions are generated periodically, before any request arrives, stored,
and retrieved when a request comes in. Nothing here runs a model.

    src/make_warnings.py      prediction service. Retrains on history only,
                              scores every pair, writes the store. Needs the
                              raw file and scikit-learn. Runs monthly, on a
                              schedule (.github/workflows/monthly-warnings.yml).

    data/processed/           the store of precomputed predictions.
      dashboard_warnings.json

    serve/main.py             this. Reads the store, answers HTTP. No model,
                              no pandas, no training data, so the image is
                              small and the cold start is fast.

The store holds two kinds of warning. A *scored* warning is for a month whose
prices have since been published, so it carries the outcome and is evidence.
A *pending* warning is for a month that has not happened: its outcome is null
and it is what the system is actually for. `/warnings` returns the pending
month when there is one, and every scored month stays readable at
`/warnings/{month}` as the record behind it.

Online prediction would be the wrong shape: prices publish monthly, no user
request is pending, latency is irrelevant to correctness, and the input space
is finite and known in advance (584 state-and-commodity pairs). Batch is the
fit, not the shortcut.

No credentials are used or required. The service reads one local JSON file and
talks to nothing else, which is the simplest possible answer to "where are your
secrets": there are none to leak.

Run locally:
    uvicorn serve.main:app --reload --port 8000
    curl http://localhost:8000/warnings
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

STORE = Path(os.environ.get("WARNINGS_PATH", "data/processed/dashboard_warnings.json"))

# The dashboard is served from here rather than from a separate static host,
# and the reason is a browser rule rather than a preference: a page served over
# HTTPS may not call an HTTP service, and this service has no certificate. A
# dashboard on GitHub Pages therefore cannot read it at all. Served from the
# same origin as the API, the page's requests are relative and the question
# does not arise.
DASHBOARD = Path(os.environ.get("DASHBOARD_PATH", "dashboard.html"))

# Zero months behind is the healthy state: the batch job scores the newest
# month in the file and warns about the one after it. One is tolerable, since
# WFP publishes on no fixed day. Two or more means the refresh has not run,
# and the service says so rather than quietly serving a stale shortlist.
STALE_AFTER_MONTHS = 2

MONTH_RE = r"^\d{4}-(0[1-9]|1[0-2])$"

app = FastAPI(
    title="Spike Watch prediction service",
    version="1.3.0",
    description=(
        "Monthly food price spike warnings for Borno, Yobe and Adamawa. "
        "Serves predictions computed ahead of time by the batch job."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # read-only service, no credentials
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- schemas
# Declared response models, so the shape of every reply is part of the
# contract and shows up in /docs rather than being whatever the dict held.

class Warning(BaseModel):
    rank: int = Field(..., ge=1, description="1 is the highest-risk pick")
    state: str
    commodity: str
    score: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability")
    reasons: List[str]
    month: Optional[str] = Field(
        None, description="Month whose prices were read")
    predicts: Optional[str] = Field(
        None, description="Month the warning is about: the month after it")
    status: Optional[str] = Field(
        None, description='"pending" if the month has not happened, else "scored"')
    actual: Optional[int] = Field(
        None, description="1 if the price did rise, 0 if not, null if not yet known")


class Freshness(BaseModel):
    latest_month: Optional[str]
    months_behind: Optional[int]
    stale: bool


class Shortlist(BaseModel):
    month: str
    predicts: Optional[str] = None
    outcome_known: Optional[bool] = None
    caveat: Optional[str] = Field(
        None, description="Set when the month behind this list was thinly reported")
    requested: int
    returned: int
    background_rate: Optional[float]
    expected_hit_rate: Optional[float]
    freshness: Freshness
    warnings: List[Warning]


class Health(BaseModel):
    status: str
    warnings_in_store: Optional[int] = None
    latest_month: Optional[str] = None
    months_behind: Optional[int] = None
    stale: Optional[bool] = None
    pending_for: Optional[str] = None
    reason: Optional[str] = None
    checked_at: str


# ---------------------------------------------------------------- loading

def _load() -> dict:
    if not STORE.exists():
        raise FileNotFoundError(
            f"No prediction store at {STORE}. Run `python -m src.make_warnings`, "
            "or set WARNINGS_PATH."
        )
    data = json.loads(STORE.read_text())
    if isinstance(data, list):          # tolerate the pre-meta shape
        return {"meta": {}, "warnings": data}
    if "warnings" not in data:
        raise ValueError(f"{STORE} has no 'warnings' key")
    data.setdefault("meta", {})
    return data


def _months(data: dict) -> List[str]:
    return sorted({w["month"] for w in data["warnings"] if w.get("month")})


def _freshness(data: dict) -> dict:
    months = _months(data)
    if not months:
        return {"latest_month": None, "months_behind": None, "stale": True}
    latest = months[-1]
    ly, lm = (int(p) for p in latest.split("-"))
    today = date.today()
    behind = max(0, (today.year - ly) * 12 + (today.month - lm))
    return {"latest_month": latest, "months_behind": behind,
            "stale": behind >= STALE_AFTER_MONTHS}


def _caveat(data: dict, month: str) -> Optional[str]:
    """A warning about the warning, when the month behind it was thin.

    Returned in the response rather than logged, because the person deciding
    what to buy is the one who needs to know that five markets had not
    reported when this was computed.
    """
    live = (data.get("meta") or {}).get("live") or {}
    if live.get("scored_from") != month:
        return None
    cov = live.get("coverage") or {}
    if not cov.get("partial"):
        return None
    share = cov.get("share_of_normal")
    pct = f"{share:.0%}" if isinstance(share, (int, float)) else "fewer"
    return (f"{month} was only {pct} of its usual size when this was computed: "
            "some markets had not reported yet. The ranking still holds, but "
            "it rests on thinner data than the months it was measured on.")


def _pending_for(data: dict) -> Optional[str]:
    """The month a live warning is about, or None if the feed is all history."""
    live = (data.get("meta") or {}).get("live") or {}
    if live.get("predicts"):
        return live["predicts"]
    for w in data.get("warnings", []):
        if w.get("status") == "pending" and w.get("predicts"):
            return w["predicts"]
    return None


# Read once at start-up: the store only changes when the batch job runs.
try:
    _DATA: Optional[dict] = _load()
    _ERROR: Optional[str] = None
except Exception as exc:                # surfaced through /health
    _DATA, _ERROR = None, str(exc)


def _data() -> dict:
    if _DATA is None:
        raise HTTPException(status_code=503, detail=_ERROR or "store unavailable")
    return _DATA


# ---------------------------------------------------------------- routes

@app.get("/", summary="What this service is")
def root() -> dict:
    return {
        "service": "Spike Watch prediction service",
        "pattern": "batch prediction: precomputed monthly, served on request",
        "docs": "/docs",
        "endpoints": ["/health", "/warnings", "/warnings/{month}",
                      "/months", "/performance", "/feed", "/dashboard"],
        "for people": "/dashboard",
    }


@app.get("/health", response_model=Health, summary="Liveness and data freshness")
def health() -> dict:
    """Always answers. Reports degraded rather than failing, so a probe can
    tell 'the service is down' apart from 'the batch job has not run'."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if _DATA is None:
        return {"status": "unhealthy", "reason": _ERROR, "checked_at": now}
    fresh = _freshness(_DATA)
    return {"status": "degraded" if fresh["stale"] else "ok",
            "warnings_in_store": len(_DATA["warnings"]), **fresh,
            "pending_for": _pending_for(_DATA), "checked_at": now}


@app.get("/months", summary="Months the store covers")
def months() -> dict:
    data = _data()
    return {"months": _months(data), "pending_for": _pending_for(data),
            **_freshness(data)}


@app.get("/warnings", response_model=Shortlist, summary="The current shortlist")
def warnings(
    k: int = Query(5, ge=1, le=20, description="How many foods to return"),
    state: Optional[str] = Query(None, max_length=40, description="Filter to one state"),
) -> dict:
    """The shortlist for the newest month in the store.

    When the batch job has a month with no outcome yet, that is the month
    returned here, and `predicts` names the month being warned about. The
    answer to "when is this spike" is that field, and `outcome_known` is
    false until the prices for it are published.

    Five is the default deliberately: an alarm tuned to catch most spikes
    flags most of the market, and a buyer holding forty warnings has none.
    """
    data = _data()
    available = _months(data)
    if not available:
        raise HTTPException(status_code=503, detail="store holds no warnings")
    return _for_month(data, available[-1], k, state)


@app.get("/warnings/{month}", response_model=Shortlist,
         summary="The shortlist for one month")
def warnings_for_month(
    month: str = PathParam(..., pattern=MONTH_RE, examples=["2026-06"],
                           description="YYYY-MM"),
    k: int = Query(5, ge=1, le=20),
    state: Optional[str] = Query(None, max_length=40),
) -> dict:
    data = _data()
    if month not in _months(data):
        raise HTTPException(status_code=404,
                            detail=f"no warnings for {month}. Available: {_months(data)}")
    return _for_month(data, month, k, state)


def _for_month(data: dict, month: str, k: int, state: Optional[str]) -> dict:
    rows = [w for w in data["warnings"] if w.get("month") == month]
    if state:
        rows = [w for w in rows if w.get("state", "").lower() == state.lower()]
    rows.sort(key=lambda w: w.get("rank", 10**6))
    meta = data.get("meta", {})
    predicts = next((w.get("predicts") for w in rows if w.get("predicts")), None)
    pending = any(w.get("status") == "pending" for w in rows)
    return {
        "month": month,
        "predicts": predicts,
        "outcome_known": (not pending) if rows else None,
        "caveat": _caveat(data, month),
        "requested": k,
        "returned": len(rows[:k]),
        "background_rate": meta.get("base_rate"),
        "expected_hit_rate": meta.get("precision_at_k", {}).get(str(k)),
        "freshness": _freshness(data),
        "warnings": rows[:k],
    }


@app.get("/feed", summary="Every month at once, for the dashboard")
def feed() -> dict:
    """The whole store in one request, with short keys.

    The dashboard draws a bar per month, so it needs every month at once. A
    request per month would be a dozen round trips for data that changes once
    a month. Same content as /warnings, named shorter because this one is
    read by a page rather than by a person.

    `a` is the outcome: 1 if the price rose, 0 if it held, and null when the
    month has not happened yet. A page must treat null as unknown and not as
    a miss, or a pending month reads as a failed one.
    """
    data = _data()
    return {
        "meta": data.get("meta", {}),
        "freshness": _freshness(data),
        "pending_for": _pending_for(data),
        "warnings": [
            {"s": w.get("state"), "c": w.get("commodity"), "m": w.get("month"),
             "p": w.get("score"), "r": w.get("reasons", []),
             "a": w.get("actual"), "status": w.get("status"),
             "predicts": w.get("predicts"),
             "pairs_this_month": w.get("pairs_this_month")}
            for w in data["warnings"]
        ],
    }


@app.get("/dashboard", response_class=HTMLResponse,
         summary="The page a person actually looks at")
def dashboard() -> HTMLResponse:
    """The dashboard, served from the same origin as the data it reads."""
    if not DASHBOARD.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no dashboard at {DASHBOARD}. Set DASHBOARD_PATH.")
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))


@app.get("/performance", summary="How well the shortlist has done")
def performance() -> dict:
    """The evaluation served alongside the predictions, so a user can see what
    the list is worth before acting on it."""
    data = _data()
    meta = data.get("meta", {})
    base = meta.get("base_rate")
    p_at = meta.get("precision_at_k", {})
    return {
        "tested_on_months": meta.get("months"),
        "pairs_scored": meta.get("pairs_scored"),
        # The month currently being warned about, which by definition is not
        # part of any accuracy figure below.
        "pending": meta.get("live"),
        "background_rate": base,
        "precision_at_k": p_at,
        "lift_over_chance": {k: round(v / base, 2) for k, v in p_at.items()} if base else {},
        "retrained": meta.get("retrained"),
        "train_window_months": meta.get("train_window_months"),
        "purge_months": meta.get("purge_months", 1),
        "note": (
            "Walk-forward: refit on data before each test month, with a "
            "one-month purge because the label looks one month ahead. "
            "Intervals in data/processed/uncertainty.json are month-clustered."
        ),
    }
