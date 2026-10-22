"""
SERVER (Owner: Nishant)
=======================
FastAPI application: REST API for the dashboard, live market ticker, and the
entry point (python run.py).
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from Abhik import ai_models as ai
from Abhik.copilot import HELP_TEXT, agent_answer, current_fusion, rule_copilot
from Nishant.core import (
    ADJ, AIS_ZONES, BASE_DIR, CHOKE_POINTS, CHOKE_PROFILES, EDGE_WAYPOINTS, HEURISTIC_CHECK,
    LAST_ROUTE, LLM, NODES, STARTED_AT, STATE, STRATEGIC_NOTES, TRACKER, live_severity,
    parse_override,
)
from Nishant.routing import COST_MODEL, compute_routes, effective_risk, risk_penalty_days
from Yash.clustering import cluster_commodities
from Yash.data_sources import FEEDS, MARKET_SERIES, PORTWATCH_IDS, fetch_ticker
from Yash.intelligence import intelligence_payload, run_scan


# =============================================================================
# FASTAPI APPLICATION  (no start-up tasks: nothing external runs until asked)
# =============================================================================
app = FastAPI(title="Global Chain AI - Maritime Risk Intelligence & Pathfinding", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def degrade_gracefully(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"})


class ChatRequest(BaseModel):
    command: str
    use_llm: bool = True






@app.get("/", include_in_schema=False)
async def serve_dashboard():
    page = BASE_DIR / "Harsh" / "index.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return JSONResponse({"status": "ok", "message": "Dashboard not found: expected Harsh/index.html."})


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "uptime_s": int(time.time() - STARTED_AT),
        "llm": LLM.info(),
        "intelligence_mode": STATE["mode"],
        "scanned_at": STATE["scanned_at"],
        "graph": {"nodes": len(NODES), "directed_edges": sum(len(v) for v in ADJ.values())},
        "heuristic": {k: HEURISTIC_CHECK[k] for k in ("admissible", "consistent", "directed_edges_checked")},
        "ais": {"running": TRACKER.running, "tracked": len(TRACKER.vessels),
                "providers": {k: {"state": v["state"], "enabled": v["enabled"]} for k, v in TRACKER.providers.items()}},
        "agent": {
            "type": "Model-based, utility-driven rational agent",
            "performance": "Minimise transit time subject to geopolitical risk exposure",
            "environment": "Global maritime network: partially observable, dynamic, stochastic",
            "sensors": {"news": [f["publisher"] for f in FEEDS],
                        "traffic": "IMF PortWatch daily AIS transit counts",
                        "markets": [s["name"] for s in MARKET_SERIES],
                        "live_ais": [p["name"] for p in TRACKER.providers.values()]},
            "actuators": ["Route recommendations (A*)", "Map threat overlays", "Copilot directives"],
        },
    }


@app.get("/api/graph")
async def graph():
    sev = live_severity()
    nodes = [{"id": k, **{x: v[x] for x in ("name", "lat", "lng", "kind", "base_hazard")},
              "live_severity": sev.get(k, 0.0)} for k, v in NODES.items()]
    edges = [{"from": u, "to": v, "nm": round(ADJ[u][v]["nm"])} for (u, v) in EDGE_WAYPOINTS]
    return {"status": "success", "nodes": nodes, "edges": edges, "cost_model": COST_MODEL, "heuristic": HEURISTIC_CHECK}


@app.get("/api/routes")
async def routes(start: str = "SHANGHAI", end: str = "ROTTERDAM", override: str = ""):
    start, end = start.strip().upper(), end.strip().upper()
    if start not in NODES or end not in NODES:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Unknown node.", "valid_nodes": list(NODES)})
    if start == end:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Source and destination must differ."})
    live = live_severity()
    ovr = parse_override(override)
    live.update(ovr)
    result = compute_routes(start, end, live)
    if result["status"] != "success":
        return JSONResponse(status_code=404, content=result)
    LAST_ROUTE.clear()
    LAST_ROUTE.update(start=start, end=end, live=dict(live), override=ovr)
    result.update(threat_snapshot=live, override=ovr, cost_model=COST_MODEL,
                  heuristic={k: HEURISTIC_CHECK[k] for k in ("admissible", "consistent", "directed_edges_checked")})
    return result


@app.get("/api/clusters")
async def clusters(k: int = Query(2, ge=2, le=3)):
    return cluster_commodities(live_severity(), k)


@app.get("/api/intelligence")
async def intelligence():
    """Read-only: returns the current intelligence state. Never scrapes, never calls Gemini."""
    return intelligence_payload()


@app.post("/api/intelligence/ingest")
async def ingest():
    """The only entry point that sweeps the sensors and (at most once) calls Gemini."""
    try:
        await run_scan()
    except Exception as exc:  # last line of defence: keep serving the current state
        STATE["note"] = f"Ingestion failed ({type(exc).__name__}: {str(exc)[:120]}); the previous intelligence state is shown."
    return intelligence_payload()


@app.get("/api/chokepoints")
async def chokepoints():
    sev = live_severity()
    fusion = current_fusion()
    traffic = STATE.get("traffic") or {}
    out = []
    for c in CHOKE_POINTS:
        n = NODES[c]
        r = effective_risk(c, sev)
        t = traffic.get(c)
        out.append({
            "id": c, "name": n["name"], "lat": n["lat"], "lng": n["lng"],
            "base_hazard": n["base_hazard"], "live_severity": sev[c], "effective_risk": r,
            "routing_penalty_days": round(risk_penalty_days(r), 2),
            "status": "CRITICAL" if r >= 8 else "ELEVATED" if r >= 5 else "NORMAL",
            "strategic_relevance": STRATEGIC_NOTES[c],
            "profile": CHOKE_PROFILES.get(c),
            "fusion": {k: fusion[c][k] for k in ("posterior", "risk", "confidence", "signals_observed")},
            "traffic": {k: t.get(k) for k in ("recent_7d_mean", "baseline_median", "pct_vs_baseline", "anomaly",
                                              "direction", "as_of", "change_point")} if t else None,
            "events": [e for e in STATE["events"] if e["choke_point_targeted"] == c],
        })
    return {"status": "success", "mode": STATE["mode"], "chokepoints": out}


@app.get("/api/signals")
async def signals():
    """Evidence fusion per choke point, traffic analytics (anomaly + Holt forecast) and the model spec."""
    fusion = current_fusion()
    traffic = STATE.get("traffic") or {}
    nodes = CHOKE_POINTS + ["CAPE_OF_GOOD_HOPE"]
    return {
        "status": "success",
        "scanned_at": STATE["scanned_at"],
        "fusion_model": ai.FUSION_MODEL,
        "chokepoints": [{"id": c, "name": NODES[c]["name"], "fusion": fusion.get(c), "traffic": traffic.get(c)}
                        for c in nodes],
        "traffic_sources": STATE.get("traffic_sources", []),
    }


@app.get("/api/markets")
async def markets():
    m = STATE.get("markets") or {}
    return {
        "status": "success",
        "scanned_at": STATE["scanned_at"],
        "series": [{"code": s["code"], "symbol": s["symbol"], "name": s["name"], "role": s["role"],
                    "unit": s["unit"], "stats": m.get(s["code"], {}).get("stats")} for s in MARKET_SERIES],
        "sources": STATE.get("market_sources", []),
    }


TICKER_TTL_S = 120
TICKER_CACHE: Dict[str, Any] = {"at": 0.0, "quotes": [], "fetched_at": None}
TICKER_LOCK = asyncio.Lock()


@app.get("/api/ticker")
async def ticker():
    """Live quotes for the dashboard marquee: global indices, commodities, shipping and FX. Cached 2 min."""
    async with TICKER_LOCK:
        if time.time() - TICKER_CACHE["at"] > TICKER_TTL_S:
            quotes = await asyncio.to_thread(fetch_ticker)
            if any(q["price"] is not None for q in quotes) or not TICKER_CACHE["quotes"]:
                TICKER_CACHE.update(at=time.time(), quotes=quotes,
                                    fetched_at=datetime.now(timezone.utc).isoformat())
            else:
                TICKER_CACHE["at"] = time.time() - TICKER_TTL_S + 30  # upstream down: retry in 30 s
    return {"status": "success", "fetched_at": TICKER_CACHE["fetched_at"], "ttl_s": TICKER_TTL_S,
            "quotes": TICKER_CACHE["quotes"]}


@app.get("/api/vessels")
async def vessels(bbox: str = "", limit: int = Query(4000, ge=1, le=12000)):
    """Live AIS positions. bbox = 'south,west,north,east'. The first call starts the tracker."""
    TRACKER.touch()
    box = None
    try:
        parts = [float(x) for x in bbox.split(",")] if bbox else []
        if len(parts) == 4:
            box = parts
    except ValueError:
        box = None
    snap = TRACKER.snapshot(box, limit)
    snap["zones"] = {k: {"name": NODES[k]["name"], **v} for k, v in TRACKER.zone_counts(AIS_ZONES).items()}
    return snap


@app.post("/api/chat")
async def chat(req: ChatRequest):
    cmd = (req.command or "").strip()[:1200]
    if not cmd:
        return {"response": HELP_TEXT, "engine": "RULE_ENGINE", "note": None, "tools_used": []}
    if req.use_llm and LLM.status == "READY":
        result = await agent_answer(cmd)
        if result:
            return {"response": result["text"], "engine": f"GEMINI AGENT ({LLM.active_model})",
                    "note": None, "tools_used": result["trace"]}
    deterministic = rule_copilot(cmd)
    note = None
    if req.use_llm and LLM.status != "NOT_CONFIGURED":
        note = LLM.unavailable_note()
    return {"response": deterministic, "engine": "RULE_ENGINE", "note": note, "tools_used": []}


def main() -> None:
    print("=" * 66)
    print(" GLOBAL CHAIN AI - MARITIME RISK INTELLIGENCE BACKEND")
    print(f" LLM engine : {LLM.status} ({LLM.models[0] if LLM.models else LLM.last_error})")
    print(f" Sensors    : {len(FEEDS)} news feeds, {len(PORTWATCH_IDS)} AIS traffic series, {len(MARKET_SERIES)} market series")
    print(f" Live AIS   : " + ", ".join(f"{p['name']} [{p['state']}]" for p in TRACKER.providers.values()))
    print(f" Heuristic  : admissible={HEURISTIC_CHECK['admissible']} over {HEURISTIC_CHECK['directed_edges_checked']} edges")
    print(" Dashboard  : http://localhost:8000      API docs: http://localhost:8000/docs")
    print("=" * 66)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
