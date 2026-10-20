"""
GLOBAL CHAIN AI - MARITIME RISK INTELLIGENCE & PATHFINDING
==========================================================
FastAPI backend implementing a model-based, utility-driven rational agent:

  SENSORS    : 12 news feeds (trade press + Google News topic searches), IMF PortWatch AIS
               transit counts, Yahoo Finance market series, live AIS vessel positions
               (data_sources.py, ais_live.py)
  INFERENCE  : (a) Gemini LLM threat synthesis with a strict JSON schema
               (b) forward-chaining rule-based expert system (fallback + explainability)
               (c) A* search over a weighted maritime graph (Unit 2)
               (d) K-Means clustering of commodity risk vectors built from market data (Unit 5)
               (e) TF-IDF story clustering + retrieval (ai_models.py)
               (f) traffic anomaly detection + Holt exponential-smoothing forecast
               (h) Naive-Bayes log-odds fusion of news, traffic and market evidence
               (i) noisy-OR supply exposure
               (j) tool-using Gemini agent (function calling) with TF-IDF RAG for the Copilot
  ACTUATORS  : route recommendations, map overlays, Copilot answers (REST API)

Quota policy: the server NEVER scrapes or calls Gemini on start-up or page load.
Ingestion runs only on POST /api/intelligence/ingest. Gemini calls are capped by a local
per-minute budget and paused automatically after any HTTP 429. Live AIS starts only when
the map's live-ships layer requests /api/vessels and stops after 10 idle minutes.

Run:  python backend.py      (serves the dashboard at http://localhost:8000)
"""

import asyncio
import hashlib
import heapq
import json
import math
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import ai_models as ai
from ais_live import AISTracker
from data_sources import FEEDS, MARKET_SERIES, PORTWATCH_IDS, sweep


# =============================================================================
# CONFIGURATION (.env is resolved next to this file, independent of the CWD)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "your_new_api_key_here", "your_key", "changeme"}
PRIMARY_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip()
FALLBACK_MODELS = [
    m.strip()
    for m in (os.getenv("GEMINI_FALLBACK_MODELS") or "gemini-3.5-flash,gemini-flash-latest").split(",")
    if m.strip()
]
GEMINI_MAX_RPM = max(1, int(os.getenv("GEMINI_MAX_RPM", "8")))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SCAN_WALL_CLOCK_S = 14.0        # hard ceiling for one sensor sweep (news + traffic + markets)
LLM_TIMEOUT_S = 60.0            # per LLM call (story assessment prompts are large)
DEFAULT_COOLDOWN_S = 60         # pause after a 429 when the API gives no retry hint
DAILY_COOLDOWN_S = 3600         # pause after a daily-quota 429
SCAN_DEBOUNCE_S = 8             # ignore ingestion requests closer together than this
MAX_STORIES = 40                # stories kept after clustering
MAX_STORIES_FOR_LLM = 14        # stories sent to Gemini in one call
AGENT_MAX_STEPS = 4             # Gemini calls per Copilot answer (tool rounds + final answer)

SHIP_SPEED_KNOTS = 14.0                      # slow-steaming container / tanker speed
NM_PER_DAY = SHIP_SPEED_KNOTS * 24.0         # 336 nm per day
ROUTING_FACTOR = 1.04                        # coastal / traffic-separation allowance
RISK_ALPHA = 0.35                            # P(r) = alpha * (e^(beta*r) - 1)  [days]
RISK_BETA = 0.55
SAFETY_DECAY = 150.0                         # safety = 100 * exp(-sum(r^2) / SAFETY_DECAY)
EXPOSURE_THRESHOLD = 5.0                     # live severity >= this counts as threat exposure

STARTED_AT = time.time()


# =============================================================================
# LLM ENGINE (Gemini REST): local rate budget, 429 cooldown, model fallback, tools
# =============================================================================
class LLMEngine:
    def __init__(self) -> None:
        self.enabled = False
        self.key_invalid = False
        self.models: List[str] = []
        self.active_model: Optional[str] = None
        self.cooldown_until = 0.0
        self.cooldown_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.call_times: deque = deque()
        self.calls = 0

        if API_KEY.lower() in PLACEHOLDER_KEYS:
            self.last_error = "GEMINI_API_KEY not configured"
            print(f"[WARN] GEMINI_API_KEY not found in {env_path} or using placeholder. Running in fallback mode.")
            return
        if not API_KEY.startswith("AIza"):
            print("[WARN] GEMINI_API_KEY does not look like a Google AI Studio key (they start with 'AIza'). "
                  "Get one at https://aistudio.google.com/apikey")
        self.models = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
        self.enabled = True
        print(f"[INIT] Gemini key loaded ({len(API_KEY)} chars), model chain {self.models}")

    def _prune(self) -> None:
        cutoff = time.time() - 60
        while self.call_times and self.call_times[0] < cutoff:
            self.call_times.popleft()

    @property
    def status(self) -> str:
        if not self.enabled:
            return "NOT_CONFIGURED"
        if self.key_invalid:
            return "KEY_INVALID"
        if time.time() < self.cooldown_until:
            return "COOLDOWN"
        self._prune()
        if len(self.call_times) >= GEMINI_MAX_RPM:
            return "THROTTLED"
        return "READY"

    def info(self) -> Dict[str, Any]:
        self._prune()
        status = self.status
        remaining = max(0, int(self.cooldown_until - time.time())) if status == "COOLDOWN" else 0
        if status == "THROTTLED" and self.call_times:
            remaining = max(1, int(self.call_times[0] + 60 - time.time()))
        return {
            "status": status,
            "model": self.active_model or (self.models[0] if self.models else None),
            "cooldown_remaining_s": remaining,
            "cooldown_reason": self.cooldown_reason if status == "COOLDOWN" else None,
            "budget_per_minute": GEMINI_MAX_RPM,
            "used_last_minute": len(self.call_times),
            "last_error": self.last_error,
        }

    def unavailable_note(self) -> str:
        i = self.info()
        if i["status"] == "NOT_CONFIGURED":
            return "Gemini is not configured; the local AI models produced this result."
        if i["status"] == "KEY_INVALID":
            return "Google rejected the Gemini API key (create one at aistudio.google.com/apikey); the local AI models produced this result."
        if i["status"] == "COOLDOWN":
            what = "daily quota" if i["cooldown_reason"] == "daily" else "rate limit"
            return f"Gemini {what} reached (retry in about {i['cooldown_remaining_s']} s); the local AI models produced this result."
        if i["status"] == "THROTTLED":
            return f"Local Gemini budget of {GEMINI_MAX_RPM} calls/min is used up (resets in about {i['cooldown_remaining_s']} s); the local AI models produced this result."
        return f"Gemini call failed ({(self.last_error or 'unknown error')[:120]}); the local AI models produced this result."

    @staticmethod
    def _post(model_name: str, body: Dict[str, Any]) -> requests.Response:
        return requests.post(GEMINI_URL.format(model=model_name), headers={"x-goog-api-key": API_KEY},
                             json=body, timeout=LLM_TIMEOUT_S)

    def _register_429(self, msg: str) -> None:
        daily = bool(re.search(r"per ?day|PerDay|daily", msg, re.I))
        hint = re.search(r"retry in ([\d.]+)\s*s", msg, re.I) or re.search(r'"retryDelay":\s*"(\d+)', msg)
        wait_s = float(hint.group(1)) + 2 if hint else DEFAULT_COOLDOWN_S
        if daily:
            wait_s = max(wait_s, DAILY_COOLDOWN_S)
        self.cooldown_until = time.time() + wait_s
        self.cooldown_reason = "daily" if daily else "per-minute"

    async def call(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """One generateContent request (counted against the budget). Returns the first candidate's
        content dict, or None when Gemini is unavailable, over budget, cooling down or failing.
        Fallback models are tried when a model is not found or overloaded (5xx), never after a quota error."""
        if self.status != "READY":
            return None
        for name in self.models:
            if self.status != "READY":
                return None
            self.call_times.append(time.time())
            self.calls += 1
            try:
                resp = await asyncio.to_thread(self._post, name, body)
            except requests.Timeout:
                self.last_error = f"{name}: timeout after {LLM_TIMEOUT_S:.0f}s"
                return None
            except requests.RequestException as exc:
                self.last_error = f"{name}: network ({type(exc).__name__})"
                continue  # transient connection failure -> next model in the chain
            text = resp.text
            if resp.status_code == 200:
                try:
                    cand = (resp.json().get("candidates") or [{}])[0]
                except ValueError:
                    self.last_error = f"{name}: invalid JSON response"
                    return None
                content = cand.get("content") or {}
                if not content.get("parts"):
                    self.last_error = f"{name}: empty response ({cand.get('finishReason')})"
                    return None
                self.active_model = name
                self.last_error = None
                return content
            self.last_error = f"{name}: HTTP {resp.status_code} {text[:200]}"
            if resp.status_code == 429:
                self._register_429(text)
                return None
            if resp.status_code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
                self.key_invalid = True
                print("[WARN] Gemini rejected the API key; LLM features disabled until restart with a valid key.")
                return None
            if resp.status_code == 404 or resp.status_code >= 500 or re.search(r"not found|is not supported|unknown model", text, re.I):
                continue  # model unavailable or overloaded -> next model in the chain
            return None
        return None

    async def generate(self, prompt: str, json_mode: bool = False, temperature: float = 0.2) -> Optional[str]:
        config: Dict[str, Any] = {"temperature": temperature}
        if json_mode:
            config["responseMimeType"] = "application/json"
        content = await self.call({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                   "generationConfig": config})
        if not content:
            return None
        text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought"))
        return text if text.strip() else None


LLM = LLMEngine()
TRACKER = AISTracker()


# =============================================================================
# MARITIME NETWORK (graph nodes, sea-lane geometry, base hazards)
# =============================================================================
NODES: Dict[str, Dict[str, Any]] = {
    "SHANGHAI":          {"name": "Shanghai",            "lat": 31.00,  "lng": 122.30,  "kind": "PORT",     "base_hazard": 1.0},
    "TOKYO":             {"name": "Tokyo",               "lat": 34.90,  "lng": 139.90,  "kind": "PORT",     "base_hazard": 1.0},
    "SINGAPORE":         {"name": "Singapore",           "lat": 1.20,   "lng": 103.90,  "kind": "PORT",     "base_hazard": 2.0},
    "MALACCA":           {"name": "Strait of Malacca",   "lat": 3.20,   "lng": 100.60,  "kind": "CHOKE",    "base_hazard": 3.0},
    "MUMBAI":            {"name": "Mumbai",              "lat": 18.90,  "lng": 72.75,   "kind": "PORT",     "base_hazard": 2.0},
    "DUBAI":             {"name": "Dubai (Jebel Ali)",   "lat": 25.20,  "lng": 55.15,   "kind": "PORT",     "base_hazard": 2.0},
    "HORMUZ":            {"name": "Strait of Hormuz",    "lat": 26.50,  "lng": 56.45,   "kind": "CHOKE",    "base_hazard": 4.0},
    "BAB_EL_MANDEB":     {"name": "Bab-el-Mandeb",       "lat": 12.60,  "lng": 43.35,   "kind": "CHOKE",    "base_hazard": 4.0},
    "SUEZ":              {"name": "Suez Canal",          "lat": 30.00,  "lng": 32.57,   "kind": "CHOKE",    "base_hazard": 3.0},
    "GIBRALTAR":         {"name": "Strait of Gibraltar", "lat": 35.95,  "lng": -5.60,   "kind": "WAYPOINT", "base_hazard": 1.0},
    "ROTTERDAM":         {"name": "Rotterdam",           "lat": 52.00,  "lng": 3.90,    "kind": "PORT",     "base_hazard": 1.0},
    "LONDON":            {"name": "London (Thames)",     "lat": 51.50,  "lng": 1.20,    "kind": "PORT",     "base_hazard": 1.0},
    "NEW_YORK":          {"name": "New York",            "lat": 40.45,  "lng": -73.80,  "kind": "PORT",     "base_hazard": 1.0},
    "PANAMA":            {"name": "Panama Canal",        "lat": 9.10,   "lng": -79.70,  "kind": "CHOKE",    "base_hazard": 2.0},
    "LOS_ANGELES":       {"name": "Los Angeles",         "lat": 33.60,  "lng": -118.25, "kind": "PORT",     "base_hazard": 1.0},
    "CAPE_OF_GOOD_HOPE": {"name": "Cape of Good Hope",   "lat": -34.80, "lng": 18.30,   "kind": "WAYPOINT", "base_hazard": 3.0},
}

CHOKE_POINTS = ["SUEZ", "HORMUZ", "BAB_EL_MANDEB", "PANAMA", "MALACCA"]
CANAL_DWELL_DAYS = {"SUEZ": 1.0, "PANAMA": 1.5}  # queueing + transit time at canals

# Intermediate sea-lane waypoints (lat, lng). Edge length = polyline length * ROUTING_FACTOR,
# which guarantees edge cost >= great-circle distance (heuristic consistency).
EDGE_WAYPOINTS: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
    ("SHANGHAI", "TOKYO"): [(30.0, 130.5), (30.4, 131.5), (32.6, 135.0), (34.2, 139.0)],
    ("SHANGHAI", "SINGAPORE"): [(27.5, 121.5), (24.5, 119.8), (22.0, 116.5), (15.0, 112.5), (8.0, 108.8), (3.5, 105.5)],
    ("SHANGHAI", "LOS_ANGELES"): [(28.5, 130.0), (31.5, 140.5), (38.0, 160.0), (40.0, -175.0), (38.0, -150.0), (34.5, -124.0)],
    ("TOKYO", "LOS_ANGELES"): [(35.5, 145.0), (42.0, 170.0), (42.0, -165.0), (38.0, -140.0), (34.5, -124.0)],
    ("TOKYO", "PANAMA"): [(33.0, 150.0), (25.0, -170.0), (15.0, -130.0), (10.0, -100.0), (7.0, -80.5)],
    ("LOS_ANGELES", "PANAMA"): [(28.0, -116.0), (20.0, -108.0), (12.0, -95.0), (7.0, -80.5)],
    ("SINGAPORE", "MALACCA"): [(1.25, 103.6), (2.0, 102.1)],
    ("MALACCA", "MUMBAI"): [(4.5, 99.6), (6.3, 96.5), (5.4, 80.6), (7.6, 76.8), (13.5, 73.4)],
    ("MALACCA", "BAB_EL_MANDEB"): [(4.5, 99.6), (6.3, 96.5), (5.4, 80.6), (8.0, 70.0), (11.8, 54.5), (12.0, 47.0), (12.4, 44.5)],
    ("SINGAPORE", "CAPE_OF_GOOD_HOPE"): [(-0.5, 105.5), (-2.5, 107.0), (-6.2, 105.7), (-9.5, 103.0), (-25.0, 70.0), (-35.5, 30.0)],
    ("MUMBAI", "HORMUZ"): [(21.5, 66.0), (24.8, 58.5), (25.8, 57.0)],
    ("HORMUZ", "DUBAI"): [(26.1, 55.9), (25.6, 55.4)],
    ("MUMBAI", "BAB_EL_MANDEB"): [(15.0, 63.0), (12.6, 51.5), (12.3, 45.5)],
    ("HORMUZ", "BAB_EL_MANDEB"): [(25.6, 57.3), (22.8, 60.2), (16.0, 55.5), (12.6, 51.5), (12.3, 45.5)],
    ("HORMUZ", "CAPE_OF_GOOD_HOPE"): [(25.6, 57.3), (22.8, 60.2), (5.0, 55.0), (-11.0, 44.0), (-17.0, 41.0), (-26.0, 35.5), (-35.0, 25.0)],
    ("MUMBAI", "CAPE_OF_GOOD_HOPE"): [(10.0, 65.0), (-5.0, 52.0), (-11.0, 44.0), (-17.0, 41.0), (-26.0, 35.5), (-35.0, 25.0)],
    ("BAB_EL_MANDEB", "SUEZ"): [(14.5, 42.3), (20.0, 38.7), (26.5, 34.6), (29.3, 32.7)],
    ("SUEZ", "GIBRALTAR"): [(31.4, 32.35), (33.8, 25.0), (35.5, 18.0), (37.4, 11.2), (37.8, 6.0), (36.6, 0.0), (36.0, -4.0)],
    ("GIBRALTAR", "ROTTERDAM"): [(36.3, -9.5), (43.3, -9.9), (48.4, -5.6), (49.9, -3.0), (50.6, 0.0), (51.3, 2.0)],
    ("GIBRALTAR", "LONDON"): [(36.3, -9.5), (43.3, -9.9), (48.4, -5.6), (49.9, -3.0), (50.6, 0.0), (51.2, 1.7), (51.45, 1.6)],
    ("GIBRALTAR", "NEW_YORK"): [(36.0, -12.0), (38.5, -30.0), (40.0, -55.0), (40.2, -69.0)],
    ("ROTTERDAM", "LONDON"): [(51.8, 2.6)],
    ("LONDON", "NEW_YORK"): [(51.45, 1.6), (51.2, 1.7), (50.6, 0.0), (49.9, -3.0), (49.4, -6.5), (46.0, -25.0), (42.0, -50.0), (40.2, -69.0)],
    ("ROTTERDAM", "NEW_YORK"): [(51.3, 2.0), (50.6, 0.0), (49.9, -3.0), (49.4, -6.5), (46.0, -25.0), (42.0, -50.0), (40.2, -69.0)],
    ("NEW_YORK", "PANAMA"): [(36.0, -73.0), (28.0, -72.5), (20.0, -73.8), (15.0, -76.5), (10.0, -79.6)],
    ("CAPE_OF_GOOD_HOPE", "ROTTERDAM"): [(-25.0, 10.0), (-5.0, -5.0), (15.0, -21.0), (30.0, -17.0), (36.5, -11.5), (43.3, -9.9), (48.4, -5.6), (49.9, -3.0), (50.6, 0.0), (51.3, 2.0)],
    ("CAPE_OF_GOOD_HOPE", "NEW_YORK"): [(-25.0, 5.0), (0.0, -25.0), (20.0, -50.0), (34.0, -68.0)],
}


def haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in nautical miles."""
    r_nm = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r_nm * math.asin(min(1.0, math.sqrt(h)))


def coord(node: str) -> Tuple[float, float]:
    return NODES[node]["lat"], NODES[node]["lng"]


def build_adjacency() -> Dict[str, Dict[str, Dict[str, Any]]]:
    adj: Dict[str, Dict[str, Dict[str, Any]]] = {n: {} for n in NODES}
    for (u, v), waypoints in EDGE_WAYPOINTS.items():
        pts = [coord(u)] + list(waypoints) + [coord(v)]
        length = sum(haversine_nm(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) * ROUTING_FACTOR
        adj[u][v] = {"nm": length, "geometry": pts}
        adj[v][u] = {"nm": length, "geometry": list(reversed(pts))}
    return adj


ADJ = build_adjacency()


def verify_heuristic() -> Dict[str, Any]:
    """Consistency check: for every edge (u,v), cost(u,v) >= h(u) - h(v) holds iff
    edge distance >= great-circle(u,v). Verified exhaustively at startup."""
    violations = []
    checked = 0
    for u, nbrs in ADJ.items():
        for v, edge in nbrs.items():
            checked += 1
            gc = haversine_nm(coord(u), coord(v))
            if edge["nm"] + 1e-9 < gc:
                violations.append({"edge": f"{u}->{v}", "edge_nm": round(edge["nm"], 1), "great_circle_nm": round(gc, 1)})
    return {
        "admissible": not violations,
        "consistent": not violations,
        "directed_edges_checked": checked,
        "violations": violations,
    }


HEURISTIC_CHECK = verify_heuristic()


# =============================================================================
# THREAT STATE + OFFLINE FALLBACK INTELLIGENCE
# =============================================================================
OFFLINE_THREAT_BRIEFINGS: List[Dict[str, Any]] = [
    {
        "headline": "Red Sea corridor: anti-ship missile and drone threat to commercial transits",
        "summary": "Sustained attacks and threats against merchant vessels near the Bab-el-Mandeb have pushed major container and tanker operators to avoid the southern Red Sea. War-risk premiums for the corridor remain elevated and naval escort capacity is limited.",
        "choke_point_targeted": "BAB_EL_MANDEB",
        "threat_severity": 9,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Suspend Bab-el-Mandeb transits for container and tanker tonnage; divert Asia-Europe services via Cape of Good Hope; plan +10 to +14 days and additional bunkers.",
        "verified_citation": {"url": "https://gcaptain.com/", "publisher": "gCaptain", "published_timestamp": None},
    },
    {
        "headline": "Suez Canal transits depressed by spillover from Red Sea insecurity",
        "summary": "With the southern approach contested, Suez Canal volumes remain well below pre-crisis levels. Northbound and southbound bookings face schedule uncertainty and higher insurance costs even though the canal itself remains open.",
        "choke_point_targeted": "SUEZ",
        "threat_severity": 7,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Hold non-urgent Suez bookings; route high-value and tanker cargo via Cape of Good Hope; reconfirm war-risk cover before any Red Sea approach.",
        "verified_citation": {"url": "https://news.un.org/en/news/topic/economic-development", "publisher": "UN News", "published_timestamp": None},
    },
    {
        "headline": "Strait of Hormuz: tanker seizure and harassment risk in Gulf approaches",
        "summary": "Periodic vessel seizures and harassment of tankers in and near the Strait of Hormuz keep the Gulf on listed-area status. Roughly a fifth of seaborne crude passes this strait, so any escalation transmits rapidly into oil and LNG freight.",
        "choke_point_targeted": "HORMUZ",
        "threat_severity": 6,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Maintain Hormuz transits under listed-area procedures: daylight passage, naval liaison, strict AIS discipline; pre-position alternate liftings ex-Fujairah.",
        "verified_citation": {"url": "https://maritime-executive.com/", "publisher": "The Maritime Executive", "published_timestamp": None},
    },
    {
        "headline": "Singapore Strait: armed robbery incidents against underway vessels",
        "summary": "Regional reporting continues to record boardings and armed robbery against ships underway in the Singapore Strait, mostly at night and in the eastbound lane. Incidents are typically low-violence but recurrent.",
        "choke_point_targeted": "MALACCA",
        "threat_severity": 5,
        "event_type": "PIRACY",
        "tactical_directive": "Raise ship security level in the Singapore Strait; reinforce night anti-boarding watch and lighting; report all incidents to ReCAAP ISC and the nearest MRCC.",
        "verified_citation": {"url": "https://www.recaap.org/alerts", "publisher": "ReCAAP ISC", "published_timestamp": None},
    },
    {
        "headline": "Panama Canal: draft limits and slot management under watershed monitoring",
        "summary": "The Panama Canal Authority manages draft and daily transit slots against Gatun Lake water levels. Reduced precipitation periods can trigger additional measures, raising slot-auction costs for unbooked vessels.",
        "choke_point_targeted": "PANAMA",
        "threat_severity": 4,
        "event_type": "DROUGHT_WEATHER",
        "tactical_directive": "Book transit slots early, plan cargo for authority draft limits, and keep a Suez/Cape contingency for Asia-US East Coast loops.",
        "verified_citation": {"url": "https://pancanal.com/en/maritime-services/advisory-to-shipping/", "publisher": "Panama Canal Authority", "published_timestamp": None},
    },
    {
        "headline": "European hub ports: labour negotiations raise berth congestion risk",
        "summary": "Collective bargaining and intermittent industrial action at North European terminals can create berth windows slippage and yard congestion at Rotterdam and Antwerp.",
        "choke_point_targeted": "NONE",
        "threat_severity": 4,
        "event_type": "STRIKE_LABOR",
        "tactical_directive": "Build a 48-72h buffer into North European terminal ETAs; pre-book alternate berth windows at nearby hubs.",
        "verified_citation": {"url": "https://www.hellenicshippingnews.com/", "publisher": "Hellenic Shipping News", "published_timestamp": None},
    },
]


def _finalise_offline_events() -> List[Dict[str, Any]]:
    events = []
    for item in OFFLINE_THREAT_BRIEFINGS:
        e = dict(item)
        e["id"] = hashlib.sha1(e["headline"].encode()).hexdigest()[:10]
        e["source_mode"] = "BASELINE"
        e["analysis_engine"] = "BASELINE"
        e["rule_trace"] = []
        e["analyst_note"] = "Baseline briefing curated from the cited publisher's ongoing coverage."
        events.append(e)
    return events


STATE: Dict[str, Any] = {
    "mode": "BASELINE",
    "engine": "BASELINE",
    "model": None,
    "events": _finalise_offline_events(),
    "articles": [],
    "sources": [{"id": f["id"], "publisher": f["publisher"], "kind": f["kind"], "ok": None, "items": 0} for f in FEEDS],
    "traffic_sources": [],
    "market_sources": [],
    "traffic": {},       # node -> ai.analyse_traffic(...)
    "markets": {},       # code -> {spec..., "stats": ai.analyse_market(...)}
    "fusion": {},        # choke -> ai.fuse(...)
    "rag": None,         # ai.TfidfIndex over ingested articles
    "rag_docs": [],
    "scanned_at": None,
    "last_scan_ts": 0.0,
    "scan_count": 0,
    "articles_ingested": 0,
    "stories_found": 0,
    "sweep_s": None,
    "llm_note": None,
    "note": "Baseline intelligence dataset. Run an ingestion to pull live news, ship traffic and market data.",
}
SCAN_LOCK = asyncio.Lock()
LAST_ROUTE: Dict[str, Any] = {}


def live_severity(events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """Live risk per choke point. After an ingestion this is the fused posterior risk
    (news + traffic + rerouting + markets); before it, the max severity of baseline events."""
    if events is None and STATE.get("fusion"):
        return {c: float(STATE["fusion"][c]["risk"]) if c in STATE["fusion"] else 0.0 for c in CHOKE_POINTS}
    sev = {c: 0.0 for c in CHOKE_POINTS}
    for e in events if events is not None else STATE["events"]:
        c = e.get("choke_point_targeted")
        if c in sev:
            sev[c] = max(sev[c], float(e.get("threat_severity", 0)))
    return sev


def parse_override(text: str) -> Dict[str, float]:
    """'HORMUZ:10,SUEZ:8' -> {'HORMUZ': 10.0, 'SUEZ': 8.0} (choke points only, clamped 0..10)."""
    out: Dict[str, float] = {}
    for part in (text or "").split(","):
        if ":" not in part:
            continue
        name, val = part.split(":", 1)
        name = name.strip().upper()
        try:
            v = float(val)
        except ValueError:
            continue
        if name in CHOKE_POINTS:
            out[name] = max(0.0, min(10.0, v))
    return out


# =============================================================================
# UNIT 2: A* SEARCH  --  f(n) = g(n) + h(n)
# =============================================================================
def effective_risk(node: str, live: Dict[str, float]) -> float:
    """Risk level r in [0,10] = max(structural base hazard, live scraped severity)."""
    return max(NODES[node]["base_hazard"], live.get(node, 0.0))


def risk_penalty_days(r: float) -> float:
    """Exponential geopolitical risk penalty in equivalent days: P(r) = a(e^(b r) - 1)."""
    return RISK_ALPHA * (math.exp(RISK_BETA * r) - 1.0)


def edge_cost(u: str, v: str, live: Dict[str, float], risk_weight: float) -> Dict[str, float]:
    nm = ADJ[u][v]["nm"]
    transit = nm / NM_PER_DAY
    dwell = CANAL_DWELL_DAYS.get(v, 0.0)
    penalty = risk_penalty_days(effective_risk(v, live))
    return {
        "nm": nm,
        "transit_days": transit,
        "dwell_days": dwell,
        "risk_penalty_days": penalty,
        "cost": transit + dwell + risk_weight * penalty,
    }


def heuristic_days(n: str, goal: str) -> float:
    """h(n): great-circle distance to goal at service speed. Admissible and consistent
    because every edge is a polyline >= the great-circle chord, and all other cost terms
    (canal dwell, risk penalty) are non-negative."""
    return haversine_nm(coord(n), coord(goal)) / NM_PER_DAY


def a_star(start: str, goal: str, live: Dict[str, float], risk_weight: float) -> Optional[Dict[str, Any]]:
    counter = 0
    open_heap: List[Tuple[float, float, int, str]] = [(heuristic_days(start, goal), 0.0, counter, start)]
    g_score: Dict[str, float] = {start: 0.0}
    came_from: Dict[str, str] = {}
    closed: set = set()
    expansion_order: List[str] = []
    generated = 1
    max_open = 1

    while open_heap:
        _, g_cur, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return {
                "path": path,
                "objective_cost": g_cur,
                "nodes_expanded": len(expansion_order),
                "nodes_generated": generated,
                "max_open_set": max_open,
                "expansion_order": expansion_order,
            }
        closed.add(current)
        expansion_order.append(current)
        for nbr in ADJ[current]:
            if nbr in closed:
                continue
            tentative = g_cur + edge_cost(current, nbr, live, risk_weight)["cost"]
            if tentative < g_score.get(nbr, math.inf):
                g_score[nbr] = tentative
                came_from[nbr] = current
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic_days(nbr, goal), tentative, counter, nbr))
                generated += 1
        max_open = max(max_open, len(open_heap))
    return None


def unwrap_longitudes(points: List[Tuple[float, float]]) -> List[List[float]]:
    """Make longitudes continuous so trans-Pacific polylines do not wrap across the map."""
    out: List[List[float]] = []
    offset = 0.0
    prev: Optional[float] = None
    for lat, lng in points:
        if prev is not None:
            d = (lng + offset) - prev
            if d > 180:
                offset -= 360
            elif d < -180:
                offset += 360
        x = lng + offset
        out.append([round(lat, 4), round(x, 4)])
        prev = x
    return out


def route_metrics(label: str, search: Dict[str, Any], live: Dict[str, float]) -> Dict[str, Any]:
    path = search["path"]
    legs = []
    geometry_pts: List[Tuple[float, float]] = []
    total_nm = total_transit = total_dwell = total_penalty = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c = edge_cost(a, b, live, 1.0)
        total_nm += c["nm"]
        total_transit += c["transit_days"]
        total_dwell += c["dwell_days"]
        total_penalty += c["risk_penalty_days"]
        legs.append({
            "from": a, "to": b,
            "nm": round(c["nm"], 0),
            "transit_days": round(c["transit_days"], 2),
            "dwell_days": round(c["dwell_days"], 2),
            "risk_level": round(effective_risk(b, live), 1),
            "risk_penalty_days": round(c["risk_penalty_days"], 2),
        })
        seg = ADJ[a][b]["geometry"]
        geometry_pts.extend(seg if not geometry_pts else seg[1:])

    exposure_sum = sum(effective_risk(n, live) ** 2 for n in path)
    safety = 100.0 * math.exp(-exposure_sum / SAFETY_DECAY)
    threats = [
        {"node": n, "name": NODES[n]["name"], "severity": round(live.get(n, 0.0), 1)}
        for n in path if n in CHOKE_POINTS and live.get(n, 0.0) >= EXPOSURE_THRESHOLD
    ]
    return {
        "label": label,
        "path": path,
        "path_names": [NODES[n]["name"] for n in path],
        "eta_days": round(total_transit + total_dwell, 1),
        "transit_days": round(total_transit, 1),
        "canal_dwell_days": round(total_dwell, 1),
        "nautical_miles": int(round(total_nm, -1)),
        "safety_index": round(safety, 1),
        "risk_exposure_sum": round(exposure_sum, 1),
        "latent_risk_penalty_days": round(total_penalty, 1),
        "threat_exposure": threats,
        "choke_transits": [n for n in path if n in CHOKE_POINTS],
        "legs": legs,
        "geometry": unwrap_longitudes(geometry_pts),
        "search": {
            "objective_cost_days": round(search["objective_cost"], 2),
            "nodes_expanded": search["nodes_expanded"],
            "nodes_generated": search["nodes_generated"],
            "max_open_set": search["max_open_set"],
            "expansion_order": search["expansion_order"],
        },
    }


def compute_routes(start: str, end: str, live: Dict[str, float]) -> Dict[str, Any]:
    safe_search = a_star(start, end, live, risk_weight=1.0)
    direct_search = a_star(start, end, live, risk_weight=0.0)
    if not safe_search or not direct_search:
        return {"status": "error", "message": f"No navigable path between {start} and {end}."}
    safe = route_metrics("SAFE_A_STAR_DIVERSION", safe_search, live)
    direct = route_metrics("STANDARD_COMMERCIAL_DIRECT", direct_search, live)
    converged = safe["path"] == direct["path"]
    avoided = [t for t in direct["threat_exposure"] if t["node"] not in safe["path"]]
    d_days = round(safe["eta_days"] - direct["eta_days"], 1)
    d_safety = round(safe["safety_index"] - direct["safety_index"], 1)
    if converged:
        verdict = "Both options take the same path. No current threat on this lane is worth a detour."
    else:
        names = ", ".join(t["name"] for t in avoided) or "elevated-risk nodes"
        verdict = (f"Take the safer route. It adds {d_days} days but raises safety by {d_safety} points by avoiding {names}.")
    return {
        "status": "success",
        "start": start,
        "end": end,
        "safe_route": safe,
        "commercial_route": direct,
        "routes": [safe, direct],
        "comparison": {
            "converged": converged,
            "delta_eta_days": d_days,
            "delta_nautical_miles": safe["nautical_miles"] - direct["nautical_miles"],
            "delta_safety": d_safety,
            "hazards_avoided": avoided,
            "residual_exposure": safe["threat_exposure"],
            "verdict": verdict,
        },
    }


COST_MODEL = {
    "objective": "f(n) = g(n) + h(n)",
    "g": "g(n) = sum over edges of [ nm / (24 * v) + canal_dwell(v) + lambda * P(r_v) ]  (days)",
    "h": "h(n) = great_circle_nm(n, goal) / (24 * v)  (admissible, consistent lower bound)",
    "risk_penalty": f"P(r) = {RISK_ALPHA} * (e^({RISK_BETA} * r) - 1) days, r = max(base_hazard, live_severity) in [0,10]",
    "lambda_safe": 1.0,
    "lambda_commercial": 0.0,
    "safety_index": f"100 * exp(-sum(r_i^2) / {SAFETY_DECAY:.0f}) over all path nodes",
    "speed_knots": SHIP_SPEED_KNOTS,
    "design_note": "Risk is placed in the edge cost g(n), not in h(n): a risk term inside h(n) would "
                   "overestimate remaining cost, break admissibility and void A* optimality.",
}


# =============================================================================
# UNIT 5: K-MEANS CLUSTERING (pure Python, 2D, k-means++ init)
# =============================================================================
# code matches data_sources.MARKET_SERIES; vol/stress are the modelled fallback used before any
# market data has been ingested. exposure = share of seaborne supply routed through each choke point.
COMMODITIES = [
    {"name": "Brent crude",        "code": "BRENT", "vol": 72, "stress": 58, "exposure": {"HORMUZ": 0.9, "BAB_EL_MANDEB": 0.4, "SUEZ": 0.3, "MALACCA": 0.3}},
    {"name": "LNG / EU gas (TTF)", "code": "TTF",   "vol": 68, "stress": 55, "exposure": {"HORMUZ": 0.8, "BAB_EL_MANDEB": 0.4, "SUEZ": 0.3, "PANAMA": 0.2}},
    {"name": "US natural gas",     "code": "HH",    "vol": 55, "stress": 30, "exposure": {"PANAMA": 0.4}},
    {"name": "Tanker freight",     "code": "TANKER_FRT", "vol": 64, "stress": 60, "exposure": {"HORMUZ": 0.8, "BAB_EL_MANDEB": 0.5, "SUEZ": 0.4, "MALACCA": 0.3}},
    {"name": "Dry-bulk freight",   "code": "BULK_FRT",   "vol": 58, "stress": 45, "exposure": {"PANAMA": 0.4, "MALACCA": 0.4, "SUEZ": 0.2}},
    {"name": "Wheat",              "code": "WHEAT", "vol": 46, "stress": 38, "exposure": {"BAB_EL_MANDEB": 0.3, "SUEZ": 0.4}},
    {"name": "Soybeans",           "code": "SOY",   "vol": 33, "stress": 35, "exposure": {"PANAMA": 0.6}},
    {"name": "Corn",               "code": "CORN",  "vol": 35, "stress": 33, "exposure": {"PANAMA": 0.6}},
    {"name": "Copper",             "code": "CU",    "vol": 34, "stress": 30, "exposure": {"PANAMA": 0.3, "MALACCA": 0.2}},
    {"name": "Aluminium",          "code": "AL",    "vol": 31, "stress": 28, "exposure": {"SUEZ": 0.2, "HORMUZ": 0.2}},
    {"name": "Iron ore",           "code": "FE",    "vol": 30, "stress": 26, "exposure": {"MALACCA": 0.3}},
]


def commodity_vectors(live: Dict[str, float]) -> List[Dict[str, Any]]:
    """Feature vector [volatility, supply-chain stress], each 0..100.

    With market data (after an ingestion):
      volatility = 30-day realised volatility, annualised, scaled so 80% -> 100
      stress     = 100 * (0.6 * noisy-OR P(supply hit) + 0.4 * price shock)
                   noisy-OR over exposed choke points with P(c) = effective risk / 10,
                   price shock = clamp(3-month price z-score / 3, 0, 1)
    Without market data: modelled snapshot coupled to live choke-point severity."""
    p_choke = {c: effective_risk(c, live) / 10.0 for c in CHOKE_POINTS}
    markets = STATE.get("markets") or {}
    out = []
    for c in COMMODITIES:
        m = markets.get(c["code"], {}).get("stats")
        supply_p = ai.noisy_or(c["exposure"], p_choke)
        if m:
            vol = min(100.0, m["vol_30d_ann"] * 125.0)
            shock = ai.clamp(m["z_score_3m"] / 3.0, 0.0, 1.0)
            stress = 100.0 * (0.6 * supply_p + 0.4 * shock)
            source = "MARKET"
        else:
            vol, stress = float(c["vol"]), float(c["stress"])
            for choke, w in c["exposure"].items():
                excess = max(0.0, live.get(choke, 0.0) - 3.0)
                vol += w * excess * 2.2
                stress += w * excess * 3.0
            source = "MODELLED"
        out.append({"name": c["name"], "code": c["code"],
                    "volatility": round(min(100.0, vol), 1),
                    "stress": round(min(100.0, stress), 1),
                    "supply_hit_probability": round(supply_p, 3),
                    "market": {k: m[k] for k in ("last", "ret_1m", "vol_30d_ann", "z_score_3m")} if m else None,
                    "feature_source": source,
                    "exposure": c["exposure"]})
    return out


def _dist(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def kmeans(points: List[List[float]], k: int, eps: float = 1e-3, max_iter: int = 20, seed: int = 42) -> Dict[str, Any]:
    n = len(points)
    k = max(1, min(k, n))
    rng = random.Random(seed)

    # k-means++ initialisation (deterministic seed for reproducible grading)
    centroids = [list(points[rng.randrange(n)])]
    while len(centroids) < k:
        d2 = [min(_dist(p, c) ** 2 for c in centroids) for p in points]
        total = sum(d2)
        if total == 0:
            centroids.append(list(points[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(d2):
            acc += w
            if acc >= r:
                centroids.append(list(points[i]))
                break

    history = [[[round(x, 3) for x in c] for c in centroids]]
    labels = [0] * n
    iterations = 0
    converged = False
    final_shift = None
    for it in range(1, max_iter + 1):
        iterations = it
        # assignment step (Euclidean)
        labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in points]
        # update step
        new_centroids = []
        for j in range(k):
            members = [points[i] for i in range(n) if labels[i] == j]
            if members:
                new_centroids.append([sum(m[0] for m in members) / len(members),
                                      sum(m[1] for m in members) / len(members)])
            else:  # empty cluster: re-seed at the point farthest from its centroid
                far = max(range(n), key=lambda i: _dist(points[i], centroids[labels[i]]))
                new_centroids.append(list(points[far]))
        shift = max(_dist(centroids[j], new_centroids[j]) for j in range(k))
        centroids = new_centroids
        history.append([[round(x, 3) for x in c] for c in centroids])
        final_shift = shift
        if shift < eps:
            converged = True
            break

    labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in points]
    inertia = sum(_dist(points[i], centroids[labels[i]]) ** 2 for i in range(n))

    # silhouette coefficient
    sil_values = []
    for i in range(n):
        same = [points[j] for j in range(n) if labels[j] == labels[i] and j != i]
        if not same:
            sil_values.append(0.0)
            continue
        a = sum(_dist(points[i], q) for q in same) / len(same)
        b = math.inf
        for j in range(k):
            if j == labels[i]:
                continue
            other = [points[t] for t in range(n) if labels[t] == j]
            if other:
                b = min(b, sum(_dist(points[i], q) for q in other) / len(other))
        if b is math.inf:
            sil_values.append(0.0)
        else:
            sil_values.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    silhouette = sum(sil_values) / n if n else 0.0

    return {"centroids": centroids, "labels": labels, "iterations": iterations, "converged": converged,
            "final_shift": final_shift, "inertia": inertia, "silhouette": silhouette, "history": history}


def cluster_commodities(live: Dict[str, float], k: int = 2) -> Dict[str, Any]:
    k = 3 if k == 3 else 2
    vectors = commodity_vectors(live)
    points = [[v["volatility"], v["stress"]] for v in vectors]
    res = kmeans(points, k)

    # name clusters by centroid magnitude (higher = riskier)
    order = sorted(range(k), key=lambda j: -(res["centroids"][j][0] + res["centroids"][j][1]))
    names = ["HIGH_SYSTEMIC_RISK", "STABLE_NORMAL_FLOW"] if k == 2 else \
            ["HIGH_SYSTEMIC_RISK", "ELEVATED_WATCH", "STABLE_NORMAL_FLOW"]
    label_of = {cluster_idx: names[rank] for rank, cluster_idx in enumerate(order)}

    commodities = []
    for i, v in enumerate(vectors):
        j = res["labels"][i]
        commodities.append({**v, "cluster": label_of[j],
                            "distance_to_centroid": round(_dist(points[i], res["centroids"][j]), 2)})
    clusters = []
    for rank, j in enumerate(order):
        members = [c for c in commodities if c["cluster"] == label_of[j]]
        clusters.append({"label": label_of[j],
                         "centroid": [round(res["centroids"][j][0], 2), round(res["centroids"][j][1], 2)],
                         "size": len(members), "members": [m["name"] for m in members]})
    return {
        "status": "success",
        "algorithm": "K-Means (Lloyd's algorithm), k-means++ init, seed 42",
        "features": ["volatility_score_0_100", "supply_chain_stress_index_0_100"],
        "distance": "Euclidean",
        "k": k,
        "epsilon": 0.001,
        "max_iterations": 20,
        "iterations": res["iterations"],
        "converged": res["converged"],
        "final_centroid_shift": round(res["final_shift"], 6) if res["final_shift"] is not None else None,
        "inertia_wcss": round(res["inertia"], 2),
        "silhouette": round(res["silhouette"], 3),
        "centroid_history": res["history"],
        "clusters": clusters,
        "commodities": commodities,
        "high_risk_cluster": [c for c in commodities if c["cluster"] == "HIGH_SYSTEMIC_RISK"],
        "stable_cluster": [c for c in commodities if c["cluster"] == "STABLE_NORMAL_FLOW"],
        "data_note": ("Volatility = 30-day realised volatility from daily market prices; stress = noisy-OR supply exposure "
                      "to choke-point risk + 3-month price shock."
                      if any(v["feature_source"] == "MARKET" for v in vectors) else
                      "Indicator snapshot is modelled until market data is ingested: base values + coupling to choke-point risk."),
    }


# =============================================================================
# NEWS RELEVANCE (sensors themselves live in data_sources.py)
# =============================================================================
RELEVANCE_TERMS = {
    r"\bship(s|ping)?\b": 2, r"\bvessels?\b": 2, r"\btankers?\b": 3, r"\bcontainer": 2, r"\bports?\b": 1,
    r"\bcanal\b": 3, r"\bstrait\b": 3, r"\bred sea\b": 4, r"\bsuez\b": 4, r"\bhormuz\b": 4, r"\bpanama\b": 3,
    r"\bmalacca\b": 4, r"\bbab[\s-]?el[\s-]?mandeb\b": 4, r"\bhouthi": 4, r"\bpira(te|cy)": 4, r"\bfreight\b": 2,
    r"\bmaritime\b": 2, r"\bnav(y|al)\b": 2, r"\bcargo\b": 1, r"\bbunker": 1, r"\blng\b": 2, r"\bcrude\b": 2,
    r"\battack": 2, r"\bdrone": 2, r"\bmissile": 2, r"\bseiz": 2, r"\bdrought\b": 2, r"\bstrike\b": 1,
    r"\bsupply chain": 2, r"\btrade\b": 1,
}


def relevance_score(article: Dict[str, Any]) -> int:
    text = f"{article['title']} {article['full_summary']}".lower()
    return sum(w for pat, w in RELEVANCE_TERMS.items() if re.search(pat, text))


def rank_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for a in articles:
        key = re.sub(r"[^a-z0-9]", "", a["title"].lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        a["relevance"] = relevance_score(a)
        if a["relevance"] >= 2:
            unique.append(a)
    unique.sort(key=lambda a: (a["relevance"], a["published_timestamp"] or ""), reverse=True)
    return unique


# =============================================================================
# UNIT 6: FORWARD-CHAINING RULE-BASED EXPERT SYSTEM
# =============================================================================
KEYWORD_FACTS: Dict[str, List[str]] = {
    "LOC_SUEZ": [r"\bsuez\b", r"\bport said\b"],
    "LOC_BAB_EL_MANDEB": [r"bab[\s-]?el[\s-]?mandeb", r"\bred sea\b", r"\bgulf of aden\b", r"\bhouthis?\b", r"\byemen"],
    "LOC_HORMUZ": [r"\bhormuz\b", r"\bpersian gulf\b", r"\bgulf of oman\b", r"\biran(ian)?\b", r"\bfujairah\b"],
    "LOC_PANAMA": [r"\bpanama\b", r"\bgatun\b"],
    "LOC_MALACCA": [r"\bmalacca\b", r"\bsingapore strait\b", r"\bstrait of singapore\b"],
    "KW_KINETIC": [r"\bmissiles?\b", r"\bdrones?\b", r"\battack(ed|s)?\b", r"\bexplosions?\b", r"\bwarships?\b", r"\bmilitary\b", r"\bnaval\b"],
    "KW_SEIZURE": [r"\bseiz(e|ed|es|ure)\b", r"\bdetain(ed|s)?\b", r"\bboarded\b"],
    "KW_PIRACY": [r"\bpira(te|tes|cy)\b", r"\barmed robbery\b", r"\bhijack"],
    "KW_WEATHER": [r"\bdrought\b", r"\bwater levels?\b", r"\bdraft (limit|restriction)", r"\btyphoon\b",
                   r"\bcyclone\b", r"\bhurricane\b", r"\bstorm\b", r"\bprecipitation\b", r"\brainfall\b"],
    "KW_LABOR": [r"\bport strike\b", r"\blabou?r (strike|dispute|action)\b", r"\bwalkout", r"\bdockworkers?\b",
                 r"\bindustrial action\b", r"\bunions?\b", r"\bstrike action\b"],
    "KW_DISRUPTION": [r"\bclos(e|ed|ure|ing)\b", r"\bsuspend", r"\bhalt", r"\bblockade", r"\bdivert", r"\breroute",
                      r"\bdelays?\b", r"\bcongestion\b", r"\bdisrupt"],
    "KW_CASUALTY": [r"\bkilled\b", r"\bsunk\b", r"\bsank\b", r"\bcasualt", r"\bcrew (missing|injured)\b"],
}

RULES: List[Tuple[str, set, set]] = [
    ("R01 location in Suez => at choke point", {"LOC_SUEZ"}, {"AT_CHOKEPOINT"}),
    ("R02 location in Red Sea/Aden => at choke point", {"LOC_BAB_EL_MANDEB"}, {"AT_CHOKEPOINT"}),
    ("R03 location in Gulf/Hormuz => at choke point", {"LOC_HORMUZ"}, {"AT_CHOKEPOINT"}),
    ("R04 location in Panama => at choke point", {"LOC_PANAMA"}, {"AT_CHOKEPOINT"}),
    ("R05 location in Malacca/Singapore Str => at choke point", {"LOC_MALACCA"}, {"AT_CHOKEPOINT"}),
    ("R06 kinetic indicators => military conflict", {"KW_KINETIC"}, {"EVT_MILITARY_CONFLICT"}),
    ("R07 seizure + Hormuz => military conflict, high", {"KW_SEIZURE", "LOC_HORMUZ"}, {"EVT_MILITARY_CONFLICT", "SEV_HIGH"}),
    ("R08 piracy indicators => piracy, medium", {"KW_PIRACY"}, {"EVT_PIRACY", "SEV_MEDIUM"}),
    ("R09 weather indicators => drought/weather, medium", {"KW_WEATHER"}, {"EVT_DROUGHT_WEATHER", "SEV_MEDIUM"}),
    ("R10 labour indicators => strike/labour, medium", {"KW_LABOR"}, {"EVT_STRIKE_LABOR", "SEV_MEDIUM"}),
    ("R11 military + choke point => high", {"EVT_MILITARY_CONFLICT", "AT_CHOKEPOINT"}, {"SEV_HIGH"}),
    ("R12 high + disruption => critical", {"SEV_HIGH", "KW_DISRUPTION"}, {"SEV_CRITICAL"}),
    ("R13 military + casualties => critical", {"EVT_MILITARY_CONFLICT", "KW_CASUALTY"}, {"SEV_CRITICAL"}),
    ("R14 weather + Panama + disruption => high", {"EVT_DROUGHT_WEATHER", "LOC_PANAMA", "KW_DISRUPTION"}, {"SEV_HIGH"}),
    ("R15 piracy + choke point => high", {"EVT_PIRACY", "AT_CHOKEPOINT"}, {"SEV_HIGH"}),
    ("R16 disruption + choke point => medium", {"KW_DISRUPTION", "AT_CHOKEPOINT"}, {"SEV_MEDIUM"}),
]

DIRECTIVES: Dict[Tuple[str, str], str] = {
    ("MILITARY_CONFLICT", "BAB_EL_MANDEB"): "Suspend Bab-el-Mandeb transits; divert Asia-Europe tonnage via Cape of Good Hope; plan +10 to +14 days and extra bunkers.",
    ("MILITARY_CONFLICT", "SUEZ"): "Hold Suez bookings for high-value and tanker cargo; reroute via Cape of Good Hope; reconfirm war-risk cover.",
    ("MILITARY_CONFLICT", "HORMUZ"): "Restrict Hormuz transits to daylight windows with naval liaison; shift liftings to Fujairah/Yanbu where feasible; expect war-risk premium spike.",
    ("MILITARY_CONFLICT", "PANAMA"): "Hold Panama transits pending security advisory; evaluate Suez or Cape routing for Asia-US East Coast loops.",
    ("MILITARY_CONFLICT", "MALACCA"): "Re-route VLCC and bulk tonnage via Sunda/Lombok straits; anticipate +2 to +3 days.",
    ("PIRACY", "MALACCA"): "Raise ship security level in the Singapore Strait; reinforce night anti-boarding watch; report to ReCAAP ISC.",
    ("PIRACY", "BAB_EL_MANDEB"): "Transit the Gulf of Aden via the IRTC applying BMP5 measures; consider embarked security.",
    ("PIRACY", "NONE"): "Apply BMP5 hardening, maintain 24h anti-piracy watch, and report to the regional reporting centre.",
    ("DROUGHT_WEATHER", "PANAMA"): "Reduce draft to authority limit, secure slots early or via auction, or divert via Suez/Cape.",
    ("DROUGHT_WEATHER", "NONE"): "Apply weather routing; add schedule buffer and review heavy-weather cargo securing.",
    ("STRIKE_LABOR", "NONE"): "Build a 48-72h buffer into terminal ETAs; pre-book alternate berth windows at nearby hubs.",
    ("GENERAL_ADVISORY", "NONE"): "Monitor; no routing change required.",
}
EVENT_PRIORITY = ["MILITARY_CONFLICT", "PIRACY", "DROUGHT_WEATHER", "STRIKE_LABOR"]
SEVERITY_SCORE = {"SEV_CRITICAL": 9, "SEV_HIGH": 7, "SEV_MEDIUM": 5}


def directive_for(event: str, choke: str) -> str:
    return (DIRECTIVES.get((event, choke)) or DIRECTIVES.get((event, "NONE"))
            or f"Monitor {NODES.get(choke, {}).get('name', 'region')}; add schedule buffer and review risk cover.")


def rule_engine_assess(article: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{article['title']} {article['full_summary']}".lower()
    facts: set = set()
    loc_hits: Dict[str, int] = {}
    for fact, patterns in KEYWORD_FACTS.items():
        hits = sum(len(re.findall(p, text)) for p in patterns)
        if hits:
            facts.add(fact)
            if fact.startswith("LOC_"):
                loc_hits[fact[4:]] = hits

    # forward chaining to a fixed point
    fired: List[str] = []
    changed = True
    while changed:
        changed = False
        for name, conds, concl in RULES:
            if conds <= facts and not concl <= facts:
                facts |= concl
                fired.append(name)
                changed = True

    choke = max(loc_hits, key=loc_hits.get) if loc_hits else "NONE"
    event = next((e for e in EVENT_PRIORITY if f"EVT_{e}" in facts), "GENERAL_ADVISORY")
    severity = max((SEVERITY_SCORE[s] for s in SEVERITY_SCORE if s in facts), default=0)
    if severity == 0:
        severity = 3 if (choke != "NONE" or event != "GENERAL_ADVISORY") else 1
    return {
        "choke_point_targeted": choke,
        "threat_severity": severity,
        "event_type": event,
        "tactical_directive": directive_for(event, choke),
        "rule_trace": fired,
        "facts": sorted(facts),
    }


# =============================================================================
# INFERENCE: GEMINI SYNTHESIS (strict schema) WITH RULE-ENGINE FALLBACK
# =============================================================================
VALID_CHOKES = set(CHOKE_POINTS) | {"NONE"}
VALID_EVENTS = set(EVENT_PRIORITY) | {"GENERAL_ADVISORY"}


def story_article(story: Dict[str, Any]) -> Dict[str, Any]:
    """Collapses a TF-IDF story cluster into one analysable document: the representative
    article plus the headlines of the other outlets covering the same event."""
    rep = story["representative"]
    others = [m["title"] for m in story["members"] if m is not rep][:4]
    summary = rep.get("full_summary", "")
    if others:
        summary = (summary + " Also reported: " + " | ".join(others)).strip()
    seen, coverage = set(), []
    for m in [rep] + [m for m in story["members"] if m is not rep]:
        if m["publisher_source"] in seen:
            continue
        seen.add(m["publisher_source"])
        coverage.append({"publisher": m["publisher_source"], "url": m.get("canonical_url"), "title": m["title"]})
    return {**rep, "full_summary": summary[:1100], "corroboration": story["corroboration"],
            "coverage": coverage[:8], "key_terms": story["key_terms"], "first_seen": story["first_seen"],
            "story_size": len(story["members"])}


def build_event(article: Dict[str, Any], assessment: Dict[str, Any], engine: str, note: str = "") -> Dict[str, Any]:
    url = article.get("canonical_url")
    return {
        "id": hashlib.sha1((url or article["title"]).encode()).hexdigest()[:10],
        "headline": article["title"],
        "summary": article["full_summary"] or "No summary provided by publisher.",
        "choke_point_targeted": assessment["choke_point_targeted"],
        "threat_severity": int(assessment["threat_severity"]),
        "event_type": assessment["event_type"],
        "tactical_directive": assessment["tactical_directive"],
        "verified_citation": {"url": url, "publisher": article["publisher_source"],
                              "published_timestamp": article.get("published_timestamp")},
        "corroboration": article.get("corroboration", 1),
        "coverage": article.get("coverage", []),
        "story_size": article.get("story_size", 1),
        "key_terms": article.get("key_terms", []),
        "confidence": assessment.get("confidence"),
        "source_mode": "LIVE",
        "analysis_engine": engine,
        "rule_trace": assessment.get("rule_trace", []),
        "analyst_note": note,
    }


def extract_json_array(text: str) -> List[Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e > s:
        return json.loads(text[s:e + 1])
    raise ValueError("no JSON array in LLM output")


async def gemini_assess(articles: List[Dict[str, Any]]) -> Optional[Dict[int, Dict[str, Any]]]:
    """One Gemini call for up to MAX_STORIES_FOR_LLM stories. Returns {story index: event}."""
    payload = [{"index": i, "title": a["title"], "summary": a["full_summary"][:600],
                "publisher": a["publisher_source"], "outlets_reporting": a.get("corroboration", 1),
                "published": a.get("published_timestamp")}
               for i, a in enumerate(articles)]
    prompt = f"""You are a maritime geopolitical risk analyst for a naval logistics command.
Each item is a news STORY: articles from several outlets clustered by topic similarity.
Assess each story for threats to commercial shipping at strategic choke points.

Return ONLY a JSON array. One object per story index you consider (you may skip irrelevant ones):
{{
  "article_index": <int, index from input>,
  "choke_point_targeted": one of ["SUEZ","HORMUZ","BAB_EL_MANDEB","PANAMA","MALACCA","NONE"],
  "threat_severity": <int 1-10>,
  "event_type": one of ["MILITARY_CONFLICT","DROUGHT_WEATHER","PIRACY","STRIKE_LABOR","GENERAL_ADVISORY"],
  "confidence": <float 0-1, how certain the assessment is given the evidence>,
  "tactical_directive": "<concrete maritime command, max 220 chars, e.g. 'Divert VLCC tankers via Cape of Good Hope; anticipate +14 days transit'>",
  "analyst_note": "<one-sentence justification, max 200 chars>"
}}

Severity rubric: 9-10 active attacks/closure; 7-8 credible imminent threat or major disruption;
5-6 recurring incidents or significant restrictions; 3-4 watch-level; 1-2 no shipping impact.
Stories reported by more outlets are better corroborated. Use only the story content.
Do not invent URLs; citations are attached by the system.

STORIES:
{json.dumps(payload, ensure_ascii=False)}"""
    text = await LLM.generate(prompt, json_mode=True, temperature=0.1)
    if not text:
        return None
    try:
        rows = extract_json_array(text)
    except Exception as exc:
        LLM.last_error = f"schema parse failed: {exc}"
        return None

    events: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("article_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(articles) or idx in events:
            continue
        choke = str(row.get("choke_point_targeted", "NONE")).upper()
        event = str(row.get("event_type", "GENERAL_ADVISORY")).upper()
        try:
            sev = int(round(float(row.get("threat_severity", 1))))
        except (TypeError, ValueError):
            sev = 1
        try:
            conf = round(ai.clamp(float(row.get("confidence", 0.6)), 0.0, 1.0), 2)
        except (TypeError, ValueError):
            conf = None
        if choke not in VALID_CHOKES or event not in VALID_EVENTS:
            fallback = rule_engine_assess(articles[idx])  # schema violation -> expert system
            events[idx] = build_event(articles[idx], fallback, "RULE_ENGINE", "LLM output failed schema validation.")
            continue
        directive = str(row.get("tactical_directive") or "").strip()[:260] or directive_for(event, choke)
        assessment = {"choke_point_targeted": choke, "threat_severity": max(1, min(10, sev)),
                      "event_type": event, "tactical_directive": directive, "rule_trace": [], "confidence": conf}
        events[idx] = build_event(articles[idx], assessment, f"GEMINI:{LLM.active_model}",
                                  str(row.get("analyst_note") or "")[:220])
    return events


def rule_event(a: Dict[str, Any]) -> Dict[str, Any]:
    assessment = rule_engine_assess(a)
    if assessment["choke_point_targeted"] == "NONE" and assessment["event_type"] == "GENERAL_ADVISORY":
        assessment["threat_severity"] = 1
    return build_event(a, assessment, "RULE_ENGINE", "Forward-chaining expert system inference.")


async def analyse_stories(articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Gemini assesses the top stories; the expert system covers the rest (and everything when
    Gemini is unavailable)."""
    llm_events = await gemini_assess(articles[:MAX_STORIES_FOR_LLM])
    out = []
    for i, a in enumerate(articles):
        if llm_events and i in llm_events:
            out.append(llm_events[i])
        elif llm_events is not None and i < MAX_STORIES_FOR_LLM:
            continue  # Gemini judged this story irrelevant to shipping
        else:
            out.append(rule_event(a))
    return out, ("GEMINI" if llm_events else "RULE_ENGINE")


RECENT_NEWS_DAYS = 21


def is_recent(a: Dict[str, Any]) -> bool:
    ts = a.get("published_timestamp")
    if not ts:
        return True
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).days <= RECENT_NEWS_DAYS
    except ValueError:
        return True


async def run_scan() -> None:
    """Single ingestion round. Called ONLY from POST /api/intelligence/ingest.
    Sweeps every sensor concurrently, then runs the inference pipeline:
    relevance ranking -> TF-IDF story clustering -> Gemini / expert-system assessment ->
    traffic anomaly + forecast -> market features -> Naive-Bayes fusion per choke point."""
    async with SCAN_LOCK:
        if time.time() - STATE["last_scan_ts"] < SCAN_DEBOUNCE_S:
            return
        data = await asyncio.to_thread(sweep, SCAN_WALL_CLOCK_S)
        sources = data["news_status"]
        articles = data["news"]

        ranked = rank_articles([a for a in articles if is_recent(a)])[:250]
        stories = ai.cluster_stories(ranked)[:MAX_STORIES]
        responded = sum(1 for s in sources if s["ok"])

        # traffic + markets (keep the previous analysis for any source that failed this round)
        traffic = dict(STATE.get("traffic") or {})
        for node, rows in data["traffic"].items():
            res = ai.analyse_traffic(rows)
            if res:
                traffic[node] = res
        markets = dict(STATE.get("markets") or {})
        for code, spec in data["market"].items():
            stats = ai.analyse_market(spec["series"])
            if stats:
                markets[code] = {k: v for k, v in spec.items() if k != "series"} | {"stats": stats}

        if stories:
            story_docs = [story_article(s) for s in stories]
            events, engine = await analyse_stories(story_docs)
            events.sort(key=lambda e: (e["threat_severity"], e.get("corroboration", 1)), reverse=True)
            STATE.update(
                mode="LIVE", engine=engine, events=events,
                model=LLM.active_model if engine == "GEMINI" else None,
                llm_note=None if engine == "GEMINI" else LLM.unavailable_note(),
                note=(f"{len(ranked)} relevant articles from {responded} of {len(FEEDS)} news sources clustered into "
                      f"{len(stories)} stories; fused with {len(traffic)} traffic and {len(markets)} market series."),
            )
        else:
            STATE.update(
                mode="BASELINE", engine="BASELINE", events=_finalise_offline_events(), model=None, llm_note=None,
                note=("News sources could not be reached, so the baseline news dataset is shown."
                      if responded == 0 else
                      "Sources responded but carried no maritime-relevant articles; baseline news retained."),
            )

        STATE["traffic"] = traffic
        STATE["markets"] = markets
        STATE["fusion"] = {c: ai.fuse(c, NODES[c]["base_hazard"], STATE["events"], traffic, markets)
                           for c in CHOKE_POINTS} if (traffic or markets or stories) else {}
        STATE["rag_docs"] = [{k: a.get(k) for k in ("title", "full_summary", "canonical_url", "publisher_source",
                                                     "published_timestamp")} for a in ranked]
        STATE["rag"] = ai.TfidfIndex([ai.doc_text(a) for a in ranked]) if ranked else None
        STATE["sources"] = sources
        STATE["traffic_sources"] = data["traffic_status"]
        STATE["market_sources"] = data["market_status"]
        STATE["articles"] = [{k: a[k] for k in ("title", "canonical_url", "publisher_source",
                                                 "published_timestamp", "relevance")} for a in ranked[:60]]
        STATE["articles_ingested"] = len(articles)
        STATE["stories_found"] = len(stories)
        STATE["sweep_s"] = data["elapsed_s"]
        STATE["scanned_at"] = datetime.now(timezone.utc).isoformat()
        STATE["last_scan_ts"] = time.time()
        STATE["scan_count"] += 1


def intelligence_payload() -> Dict[str, Any]:
    return {
        "status": "success",
        "mode": STATE["mode"],
        "engine": STATE["engine"],
        "model": STATE.get("model"),
        "note": STATE.get("note"),
        "llm_note": STATE.get("llm_note"),
        "scanned_at": STATE["scanned_at"],
        "scan_count": STATE["scan_count"],
        "sources": STATE["sources"],
        "traffic_sources": STATE["traffic_sources"],
        "market_sources": STATE["market_sources"],
        "sources_responded": sum(1 for s in STATE["sources"] if s.get("ok")),
        "sources_total": len(FEEDS),
        "articles_ingested": STATE.get("articles_ingested", 0),
        "stories_found": STATE.get("stories_found", 0),
        "sweep_s": STATE.get("sweep_s"),
        "articles": STATE["articles"],
        "events": STATE["events"],
        "choke_severity": live_severity(),
        "llm": LLM.info(),
    }


# =============================================================================
# COPILOT: DETERMINISTIC ANALYST (always available) + GEMINI REFINEMENT
# =============================================================================
ALIASES = {
    "shanghai": "SHANGHAI", "ningbo": "SHANGHAI", "singapore strait": "MALACCA", "singapore": "SINGAPORE",
    "tokyo": "TOKYO", "yokohama": "TOKYO", "mumbai": "MUMBAI", "bombay": "MUMBAI", "nhava sheva": "MUMBAI",
    "dubai": "DUBAI", "jebel ali": "DUBAI", "rotterdam": "ROTTERDAM", "london": "LONDON",
    "felixstowe": "LONDON", "new york": "NEW_YORK", "nyc": "NEW_YORK", "los angeles": "LOS_ANGELES",
    "long beach": "LOS_ANGELES", "suez": "SUEZ", "hormuz": "HORMUZ", "bab el mandeb": "BAB_EL_MANDEB",
    "bab-el-mandeb": "BAB_EL_MANDEB", "red sea": "BAB_EL_MANDEB", "gulf of aden": "BAB_EL_MANDEB",
    "panama": "PANAMA", "malacca": "MALACCA", "cape of good hope": "CAPE_OF_GOOD_HOPE",
    "good hope": "CAPE_OF_GOOD_HOPE", "gibraltar": "GIBRALTAR",
}
REPRESENTATIVE_LANES = {
    "HORMUZ": [("DUBAI", "ROTTERDAM"), ("DUBAI", "SHANGHAI")],
    "SUEZ": [("SHANGHAI", "ROTTERDAM"), ("MUMBAI", "LONDON")],
    "BAB_EL_MANDEB": [("SHANGHAI", "ROTTERDAM"), ("DUBAI", "NEW_YORK")],
    "PANAMA": [("TOKYO", "NEW_YORK"), ("LOS_ANGELES", "NEW_YORK")],
    "MALACCA": [("SHANGHAI", "MUMBAI"), ("SHANGHAI", "ROTTERDAM")],
}


def find_mentions(text: str) -> List[str]:
    found: List[Tuple[int, str]] = []
    taken: List[Tuple[int, int]] = []
    for alias in sorted(ALIASES, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in taken):
                continue
            taken.append(span)
            found.append((m.start(), ALIASES[alias]))
    out: List[str] = []
    for _, node in sorted(found):
        if node not in out:
            out.append(node)
    return out


def fmt_path(path: List[str]) -> str:
    return " → ".join(NODES[n]["name"] for n in path)


GROUP_NAMES = {"HIGH_SYSTEMIC_RISK": "High risk", "ELEVATED_WATCH": "Watch", "STABLE_NORMAL_FLOW": "Stable"}
EVENT_NAMES = {"MILITARY_CONFLICT": "conflict", "PIRACY": "piracy", "DROUGHT_WEATHER": "drought / weather",
               "STRIKE_LABOR": "labour action", "GENERAL_ADVISORY": "advisory"}


def route_brief(start: str, end: str, live: Dict[str, float], title: str = "") -> str:
    res = compute_routes(start, end, live)
    if res["status"] != "success":
        return res["message"]
    s, d, c = res["safe_route"], res["commercial_route"], res["comparison"]
    lines = [f"### {title or 'Route comparison'}: {NODES[start]['name']} to {NODES[end]['name']}",
             f"- **Safest route:** {fmt_path(s['path'])}",
             f"  **{s['eta_days']} days**, {s['nautical_miles']:,} nm, safety **{s['safety_index']}/100**",
             f"- **Fastest route:** {fmt_path(d['path'])}",
             f"  **{d['eta_days']} days**, {d['nautical_miles']:,} nm, safety **{d['safety_index']}/100**",
             f"- **Recommendation:** {c['verdict']}"]
    if s["threat_exposure"]:
        lines.append("- **Still exposed to:** " +
                     ", ".join(f"{t['name']} (risk {t['severity']:.0f}/10)" for t in s["threat_exposure"]) +
                     ". There is no way around it from this port.")
    return "\n".join(lines)


def explain_route() -> str:
    start = LAST_ROUTE.get("start", "SHANGHAI")
    end = LAST_ROUTE.get("end", "ROTTERDAM")
    live = LAST_ROUTE.get("live") or live_severity()
    res = compute_routes(start, end, live)
    if res["status"] != "success":
        return res["message"]
    s, d, c = res["safe_route"], res["commercial_route"], res["comparison"]
    lines = [f"### Why this route: {NODES[start]['name']} to {NODES[end]['name']}"]
    if c["converged"]:
        lines += [f"Both options follow the same path: {fmt_path(s['path'])}.",
                  f"No hotspot on this lane is dangerous enough to justify a detour, so the fastest route "
                  f"(**{s['eta_days']} days**) is also the recommended one."]
        return "\n".join(lines)
    risky = [leg for leg in d["legs"] if leg["risk_level"] >= EXPOSURE_THRESHOLD]
    lines.append(f"The fastest route takes **{d['eta_days']} days** but passes through:")
    for leg in risky:
        lines.append(f"- **{NODES[leg['to']]['name']}**, currently at risk **{leg['risk_level']:.0f}/10**")
    lines += ["",
              f"The safest route goes {fmt_path(s['path'])} instead. It takes **{s['eta_days']} days** "
              f"(**{c['delta_eta_days']:+} days**) and raises the safety score from "
              f"**{d['safety_index']}** to **{s['safety_index']}** out of 100.",
              "",
              "The system weighs extra sailing time against danger. Risk grows sharply as a hotspot heats up, "
              "so a few extra days at sea is the better trade once a passage reaches the high-risk range."]
    return "\n".join(lines)


def scenario_closure(chokes: List[str], ports: List[str]) -> str:
    base_live = live_severity()
    scen_live = dict(base_live)
    for c in chokes:
        scen_live[c] = 10.0
    lanes = [(ports[0], ports[1])] if len(ports) >= 2 else REPRESENTATIVE_LANES.get(chokes[0], [])
    names = ", ".join(NODES[c]["name"] for c in chokes)
    lines = [f"### What if {names} closes?"]
    for a, b in lanes:
        base = compute_routes(a, b, base_live)
        scen = compute_routes(a, b, scen_live)
        if base["status"] != "success" or scen["status"] != "success":
            continue
        bs, ss = base["safe_route"], scen["safe_route"]
        lines.append(f"**{NODES[a]['name']} to {NODES[b]['name']}**")
        lines.append(f"- Today: {fmt_path(bs['path'])} ({bs['eta_days']} days, safety {bs['safety_index']})")
        lines.append(f"- After closure: {fmt_path(ss['path'])} ({ss['eta_days']} days, safety {ss['safety_index']})")
        trapped = [c for c in chokes if c in ss["path"]]
        if trapped:
            lines.append(f"- **No way around:** every route from this port passes "
                         f"{', '.join(NODES[c]['name'] for c in trapped)}. Options are to hold cargo, "
                         f"move it overland to another port, or sail with war-risk cover.")
        else:
            lines.append(f"- Extra time: **{round(ss['eta_days'] - bs['eta_days'], 1):+} days**")
    base_cl = {c["name"]: c for c in cluster_commodities(base_live)["commodities"]}
    scen_cl = cluster_commodities(scen_live)
    exposed = [c for c in scen_cl["commodities"] if any(ch in c["exposure"] for ch in chokes)]
    if exposed:
        lines.append("**Goods affected**")
        for c in sorted(exposed, key=lambda x: -x["stress"]):
            b = base_cl[c["name"]]
            moved = "" if b["cluster"] == c["cluster"] else f" (was {GROUP_NAMES[b['cluster']]})"
            lines.append(f"- {c['name']}: volatility {b['volatility']} to **{c['volatility']}**, "
                         f"supply stress {b['stress']} to **{c['stress']}**, now **{GROUP_NAMES[c['cluster']]}**{moved}")
    if "HORMUZ" in chokes:
        lines.append("- **For tankers:** stop new oil and LNG bookings loading inside the Gulf, shift loading to "
                     "Fujairah or Yanbu where possible, and expect insurance and freight costs to jump.")
    return "\n".join(lines)


def cluster_brief() -> str:
    cl = cluster_commodities(live_severity())
    lines = ["### Goods at risk right now"]
    for c in cl["clusters"]:
        lines.append(f"- **{GROUP_NAMES[c['label']]}:** {', '.join(c['members'])}")
    top = max(cl["commodities"], key=lambda x: x["volatility"] + x["stress"])
    lines.append(f"Most exposed: **{top['name']}** (volatility {top['volatility']}, supply stress {top['stress']}).")
    return "\n".join(lines)


def threat_brief(chokes: List[str]) -> str:
    events = STATE["events"]
    if chokes:
        events = [e for e in events if e["choke_point_targeted"] in chokes] or events
    lines = ["### Current situation"]
    for e in events[:5]:
        where = NODES[e["choke_point_targeted"]]["name"] if e["choke_point_targeted"] in NODES else "General"
        lines.append(f"- **{where}, risk {e['threat_severity']}/10 ({EVENT_NAMES.get(e['event_type'], 'advisory')}):** {e['headline']}")
        lines.append(f"  Action: {e['tactical_directive']}")
    sev = live_severity()
    lines.append("Risk by passage: " + ", ".join(f"{NODES[c]['name']} {max(sev[c], NODES[c]['base_hazard']):.0f}" for c in CHOKE_POINTS))
    return "\n".join(lines)


def current_fusion() -> Dict[str, Dict[str, Any]]:
    """Fused risk per choke point; before any ingestion it is computed from baseline news only."""
    if STATE.get("fusion"):
        return STATE["fusion"]
    return {c: ai.fuse(c, NODES[c]["base_hazard"], STATE["events"], {}, {}) for c in CHOKE_POINTS}


def signals_brief(chokes: List[str]) -> str:
    fusion = current_fusion()
    lines = ["### Why the risk levels are what they are"]
    for c in chokes or sorted(CHOKE_POINTS, key=lambda x: -fusion[x]["risk"]):
        f = fusion[c]
        lines.append(f"**{NODES[c]['name']}: risk {f['risk']:.1f}/10** (probability of disruption "
                     f"{f['posterior'] * 100:.0f}%, confidence {f['confidence']})")
        for e in f["evidence"]:
            arrow = "raises" if e["llr"] > 0.05 else "lowers" if e["llr"] < -0.05 else "neutral"
            lines.append(f"- {e['label']} ({arrow}, {e['llr']:+.2f}): {e['detail']}")
    return "\n".join(lines)


def traffic_brief(chokes: List[str]) -> str:
    traffic = STATE.get("traffic") or {}
    if not traffic:
        return "### Ship traffic\nNo traffic data yet. Press **Get latest news** to load AIS transit counts."
    nodes = chokes or [n for n in PORTWATCH_IDS if n in traffic]
    lines = ["### Ship traffic through the passages (daily transits, 7-day average)"]
    for n in nodes:
        t = traffic.get(n)
        if not t:
            continue
        f14 = t["forecast"][-1]
        lines.append(f"- **{NODES[n]['name']}:** {t['recent_7d_mean']:.0f}/day vs normal {t['baseline_median']:.0f} "
                     f"(**{t['pct_vs_baseline'] * 100:+.0f}%**{', unusual' if t['anomaly'] else ''}); trend {t['direction']}, "
                     f"about {f14['mean']:.0f}/day expected in 2 weeks (data to {t['as_of']})")
    if TRACKER.running:
        z = TRACKER.zone_counts(AIS_ZONES)
        live = [f"{NODES[k]['name']} {v['vessels']}" for k, v in z.items() if v["vessels"]]
        if live:
            lines.append("Live AIS positions right now: " + ", ".join(live) + " vessels in range.")
    return "\n".join(lines)


def market_brief() -> str:
    markets = STATE.get("markets") or {}
    if not markets:
        return "### Markets\nNo market data yet. Press **Get latest news** to load prices."
    rows = sorted(markets.values(), key=lambda m: -abs(m["stats"]["z_score_3m"]))
    lines = ["### Market signals (biggest moves vs. the last 3 months first)"]
    for m in rows[:8]:
        s = m["stats"]
        lines.append(f"- **{m['name']}:** {s['last']:,.2f} {m['unit']}, 1 month {s['ret_1m'] * 100:+.1f}%, "
                     f"{s['z_score_3m']:+.1f}σ vs 3-month average, volatility {s['vol_30d_ann'] * 100:.0f}%")
    return "\n".join(lines)


def search_articles(query: str, k: int = 5) -> List[Dict[str, Any]]:
    idx = STATE.get("rag")
    if not idx:
        return []
    out = []
    for i, score in idx.search(query, k):
        d = STATE["rag_docs"][i]
        out.append({"title": d["title"], "publisher": d["publisher_source"], "url": d["canonical_url"],
                    "published": d["published_timestamp"], "summary": (d["full_summary"] or "")[:400],
                    "similarity": round(score, 3)})
    return out


def related_reporting(query: str) -> str:
    hits = search_articles(query, 3)
    if not hits:
        return ""
    lines = ["", "**Related reporting**"]
    for h in hits:
        lines.append(f"- [{h['title']}]({h['url']}) ({h['publisher']})" if h["url"] else f"- {h['title']} ({h['publisher']})")
    return "\n".join(lines)


HELP_TEXT = """### Things you can ask
- `Why avoid Suez between Shanghai and Rotterdam?`
- `What happens to oil tankers if Hormuz closes?`
- `How much traffic is going through Hormuz?`
- `Why is the Red Sea rated so high?`
- `Which goods are most at risk?`
- `What are oil and freight markets doing?`"""


def rule_copilot(cmd: str) -> str:
    t = cmd.lower()
    mentions = find_mentions(t)
    ports = [m for m in mentions if NODES[m]["kind"] == "PORT"]
    chokes = [m for m in mentions if m in CHOKE_POINTS]
    if re.search(r"\b(clos|block|shut|seal|mined|blockade|halt)", t) and chokes:
        return scenario_closure(chokes, ports)
    if re.search(r"\b(why|explain|reason|justif|avoid)", t) and re.search(r"(rout|divert|cape|detour|path)", t):
        if len(ports) >= 2:
            LAST_ROUTE.update(start=ports[0], end=ports[1], live=live_severity())
        return explain_route()
    if len(ports) >= 2:
        return route_brief(ports[0], ports[1], live_severity())
    if re.search(r"(traffic|transit|ships? (are|going)|vessels?|\bais\b|how many ships|congest)", t):
        return traffic_brief(chokes)
    if re.search(r"(market|price|brent|oil price|freight rate|stock|futures|ttf)", t):
        return market_brief()
    if re.search(r"(why|evidence|how sure|confiden|signal|rated|score)", t) and (chokes or "risk" in t):
        return signals_brief(chokes)
    if re.search(r"(cluster|k-?means|commodit|volatil|goods|cargo|grain|wheat)", t):
        return cluster_brief()
    if re.search(r"(threat|intel|risk|status|situation|brief|happening|news)", t) or chokes:
        return threat_brief(chokes) + related_reporting(cmd)
    return HELP_TEXT + "\n\n" + threat_brief([])


# =============================================================================
# COPILOT AGENT: Gemini function calling over the platform's own models (+ RAG)
# =============================================================================
NODE_ENUM = list(NODES)
TOOL_DECLARATIONS = [
    {"name": "get_chokepoint_risk",
     "description": "Fused disruption risk for a choke point with the evidence behind it (news, ship-traffic "
                    "anomaly, Cape rerouting, market pricing), plus its latest news events.",
     "parameters": {"type": "object", "properties": {
         "chokepoint": {"type": "string", "enum": CHOKE_POINTS}}, "required": ["chokepoint"]}},
    {"name": "compare_routes",
     "description": "Safest vs fastest sea route between two locations under current risk, optionally with "
                    "choke points forced closed (what-if). Returns days, nautical miles, safety score, path.",
     "parameters": {"type": "object", "properties": {
         "origin": {"type": "string", "enum": NODE_ENUM},
         "destination": {"type": "string", "enum": NODE_ENUM},
         "closed_chokepoints": {"type": "array", "items": {"type": "string", "enum": CHOKE_POINTS}}},
         "required": ["origin", "destination"]}},
    {"name": "search_news",
     "description": "Search the ingested maritime news (TF-IDF retrieval). Returns titles, outlets, URLs, summaries.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "get_ship_traffic",
     "description": "Daily vessel transits through a passage from AIS data: recent level vs normal, anomaly score, "
                    "trend, 14-day forecast, and live AIS vessel counts when the live map is running.",
     "parameters": {"type": "object", "properties": {
         "chokepoint": {"type": "string", "enum": list(PORTWATCH_IDS)}}, "required": ["chokepoint"]}},
    {"name": "get_market_data",
     "description": "Latest market prices and statistics (1-month change, volatility, z-score vs 3 months) for "
                    "energy, grains, metals, freight indices and shipping equities.",
     "parameters": {"type": "object", "properties": {
         "codes": {"type": "array", "items": {"type": "string", "enum": [s["code"] for s in MARKET_SERIES]}}}}},
    {"name": "get_goods_at_risk",
     "description": "Commodities grouped by price volatility and supply-chain stress, with the probability that "
                    "each one's seaborne supply is hit.",
     "parameters": {"type": "object", "properties": {}}},
]


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "get_chokepoint_risk":
            c = str(args.get("chokepoint", "")).upper()
            if c not in CHOKE_POINTS:
                return {"error": f"unknown choke point {c}"}
            f = current_fusion()[c]
            events = [{k: e[k] for k in ("headline", "threat_severity", "event_type", "tactical_directive", "corroboration")}
                      | {"url": e["verified_citation"]["url"], "publisher": e["verified_citation"]["publisher"]}
                      for e in STATE["events"] if e["choke_point_targeted"] == c][:4]
            return {"name": NODES[c]["name"], "risk_0_10": f["risk"], "disruption_probability": f["posterior"],
                    "confidence": f["confidence"], "evidence": f["evidence"], "events": events,
                    "strategic_note": STRATEGIC_NOTES.get(c)}
        if name == "compare_routes":
            a, b = str(args.get("origin", "")).upper(), str(args.get("destination", "")).upper()
            if a not in NODES or b not in NODES or a == b:
                return {"error": "origin and destination must be two different known locations", "valid": NODE_ENUM}
            live = live_severity()
            closed = [c for c in (args.get("closed_chokepoints") or []) if c in CHOKE_POINTS]
            for c in closed:
                live[c] = 10.0
            res = compute_routes(a, b, live)
            if res["status"] != "success":
                return {"error": res["message"]}
            if not closed:
                LAST_ROUTE.clear()
                LAST_ROUTE.update(start=a, end=b, live=dict(live), override={})
            pick = lambda r: {"path": [NODES[n]["name"] for n in r["path"]], "eta_days": r["eta_days"],
                              "nautical_miles": r["nautical_miles"], "safety_0_100": r["safety_index"],
                              "hotspots_crossed": [t["name"] for t in r["threat_exposure"]]}
            return {"closed": closed, "safest": pick(res["safe_route"]), "fastest": pick(res["commercial_route"]),
                    "verdict": res["comparison"]["verdict"]}
        if name == "search_news":
            hits = search_articles(str(args.get("query", "")), int(args.get("k") or 5))
            return {"results": hits} if hits else {"results": [], "note": "no ingested news matches (run an ingestion first)"}
        if name == "get_ship_traffic":
            n = str(args.get("chokepoint", "")).upper()
            t = (STATE.get("traffic") or {}).get(n)
            if not t:
                return {"error": "no traffic data loaded for this passage yet"}
            out = {k: t[k] for k in ("as_of", "recent_7d_mean", "baseline_median", "pct_vs_baseline", "robust_z",
                                     "anomaly", "direction", "tanker_pct_vs_baseline", "container_pct_vs_baseline")}
            out["forecast_7d"] = t["forecast"][6]
            out["forecast_14d"] = t["forecast"][13]
            if TRACKER.running and n in AIS_ZONES:
                out["live_ais_now"] = TRACKER.zone_counts({n: AIS_ZONES[n]})[n]
            return out
        if name == "get_market_data":
            markets = STATE.get("markets") or {}
            codes = args.get("codes") or list(markets)
            return {"markets": [{"code": c, "name": markets[c]["name"], "unit": markets[c]["unit"],
                                 **{k: markets[c]["stats"][k] for k in ("last", "as_of", "ret_1d", "ret_1m", "ret_3m",
                                                                        "vol_30d_ann", "z_score_3m", "percentile_1y")}}
                                for c in codes if c in markets]} if markets else {"error": "no market data loaded yet"}
        if name == "get_goods_at_risk":
            cl = cluster_commodities(live_severity())
            return {"groups": [{"group": GROUP_NAMES[c["label"]], "members": c["members"]} for c in cl["clusters"]],
                    "items": [{k: c[k] for k in ("name", "volatility", "stress", "supply_hit_probability", "cluster")}
                              for c in cl["commodities"]]}
        return {"error": f"unknown tool {name}"}
    except Exception as exc:  # a tool failure must never break the agent loop
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


AGENT_SYSTEM = """You are the analyst inside GLOBAL CHAIN AI, a shipping risk monitor.
Use the tools to get facts before answering; call several tools in one turn when useful.
Answer in concise markdown (short heading, bullets, bold key figures, no tables, max ~250 words).
Only use numbers returned by tools. When you use news, cite it as [title](url).
Explain in everyday shipping terms; say "ship-traffic data", "market prices", "news reports" rather than
naming algorithms. If the tools cannot answer, say so plainly."""


async def agent_answer(question: str) -> Optional[Dict[str, Any]]:
    """ReAct-style loop: Gemini decides which tools to call, the server executes them, results go
    back to Gemini, until it writes the final answer (max AGENT_MAX_STEPS model calls)."""
    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": question}]}]
    trace: List[Dict[str, Any]] = []
    for step in range(AGENT_MAX_STEPS):
        last = step == AGENT_MAX_STEPS - 1
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": AGENT_SYSTEM}]},
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {"temperature": 0.2},
        }
        if last:
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        content = await LLM.call(body)
        if not content:
            return None
        content["role"] = "model"
        contents.append(content)  # echoed verbatim so any thought signatures are preserved
        calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not calls:
            text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought")).strip()
            return {"text": text, "trace": trace} if text else None
        responses = []
        for c in calls:
            args = c.get("args") or {}
            result = run_tool(c["name"], args)
            trace.append({"tool": c["name"], "args": args, "ok": "error" not in result})
            fr: Dict[str, Any] = {"name": c["name"], "response": {"result": result}}
            if c.get("id"):
                fr["id"] = c["id"]
            responses.append({"functionResponse": fr})
        contents.append({"role": "user", "parts": responses})
    return None


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


STRATEGIC_NOTES = {
    "SUEZ": "Links the Mediterranean to the Red Sea and is the shortest Asia-Europe sea route; in normal conditions it carries roughly 12% of global trade.",
    "BAB_EL_MANDEB": "Southern gateway to the Red Sea, about 30 km wide at its narrowest. Every Suez transit to or from Asia passes through it.",
    "HORMUZ": "The only sea exit from the Persian Gulf. Roughly a fifth of the world's oil consumption, plus major LNG volumes, passes through it.",
    "PANAMA": "Connects the Atlantic and Pacific. Lock capacity and permitted draft depend on Gatun Lake water levels, so droughts cut daily transits.",
    "MALACCA": "Shortest sea route between the Indian Ocean and the Pacific and one of the busiest shipping lanes in the world, including the Singapore Strait.",
}

# live-AIS counting zones (centre + radius) for the passages
AIS_ZONES = {
    "HORMUZ": {"lat": 26.4, "lng": 56.4, "radius_km": 120},
    "BAB_EL_MANDEB": {"lat": 12.6, "lng": 43.4, "radius_km": 110},
    "SUEZ": {"lat": 30.6, "lng": 32.4, "radius_km": 130},
    "PANAMA": {"lat": 9.1, "lng": -79.7, "radius_km": 70},
    "MALACCA": {"lat": 2.5, "lng": 101.5, "radius_km": 260},
    "CAPE_OF_GOOD_HOPE": {"lat": -34.8, "lng": 18.8, "radius_km": 300},
}


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    page = BASE_DIR / "index.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return JSONResponse({"status": "ok", "message": "Place index.html next to backend.py to serve the dashboard."})


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
            "fusion": {k: fusion[c][k] for k in ("posterior", "risk", "confidence", "signals_observed")},
            "traffic": {k: t[k] for k in ("recent_7d_mean", "baseline_median", "pct_vs_baseline", "anomaly",
                                          "direction", "as_of")} if t else None,
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


if __name__ == "__main__":
    print("=" * 66)
    print(" GLOBAL CHAIN AI - MARITIME RISK INTELLIGENCE BACKEND")
    print(f" LLM engine : {LLM.status} ({LLM.models[0] if LLM.models else LLM.last_error})")
    print(f" Sensors    : {len(FEEDS)} news feeds, {len(PORTWATCH_IDS)} AIS traffic series, {len(MARKET_SERIES)} market series")
    print(f" Live AIS   : " + ", ".join(f"{p['name']} [{p['state']}]" for p in TRACKER.providers.values()))
    print(f" Heuristic  : admissible={HEURISTIC_CHECK['admissible']} over {HEURISTIC_CHECK['directed_edges_checked']} edges")
    print(" Dashboard  : http://localhost:8000      API docs: http://localhost:8000/docs")
    print("=" * 66)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
