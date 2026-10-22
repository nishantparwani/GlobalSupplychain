"""
CORE (Owner: Nishant)
=====================
Shared core of the agent: configuration, the Gemini LLM engine, the maritime
knowledge base (graph nodes, sea-lane geometry, hazards, chokepoint profiles) and the threat state.
"""

import asyncio
import hashlib
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from Nishant.ais_live import AISTracker
from Yash.data_sources import FEEDS


# =============================================================================
# CONFIGURATION (.env is resolved at the project root, independent of the CWD)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent   # project root, one level above Nishant/
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
# Curated from public reporting as of 21 Oct 2026; shown until the first live ingestion.
BASELINE_AS_OF = "2026-10-21"
OFFLINE_THREAT_BRIEFINGS: List[Dict[str, Any]] = [
    {
        "headline": "Strait of Hormuz effectively closed to commercial shipping as attacks escalate",
        "summary": "Since the regional war began at the start of March 2026, AIS-visible transits through Hormuz have collapsed to single digits per day against a pre-crisis norm of roughly 85. Attacks on merchant ships continue and the IMO estimates about 6,000 seafarers are stranded in the Gulf.",
        "choke_point_targeted": "HORMUZ",
        "threat_severity": 10,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Suspend Gulf liftings and Hormuz transits; source crude and LNG from Atlantic Basin and West Africa; seek alternative export routes (Fujairah via pipeline, Yanbu) where available.",
        "verified_citation": {"url": "https://cyprus-mail.com/2026/10/21/6000-seafarers-trapped-as-attacks-deepen-strait-of-hormuz-shipping-crisis",
                              "publisher": "Cyprus Mail / IMO", "published_timestamp": "2026-10-21T00:00:00+00:00"},
    },
    {
        "headline": "Bab-el-Mandeb: container lines return but the Houthi threat has not gone away",
        "summary": "Box-ship transits via the Red Sea have recovered to more than half of pre-war levels, but the Houthis have strengthened their position along Yemen's Red Sea coast and war-risk premiums for the strait remain around 3% of hull value. Transits are naval-escorted.",
        "choke_point_targeted": "BAB_EL_MANDEB",
        "threat_severity": 7,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Transit only under naval coordination with hardened security; confirm war-risk cover before booking; keep Cape of Good Hope as fallback for tankers and high-value cargo.",
        "verified_citation": {"url": "https://www.seatrade-maritime.com/containers/full-suez-return-for-container-carriers-expected-by-q1-2027",
                              "publisher": "Seatrade Maritime", "published_timestamp": "2026-10-16T00:00:00+00:00"},
    },
    {
        "headline": "Suez Canal traffic at its highest since December 2023, full return expected by Q1 2027",
        "summary": "Suez transits rose again in September, the highest level since the Red Sea crisis began, although liner capacity through the canal (about 753,000 TEU in early October) is still roughly half of pre-crisis levels. Carriers expect a phased return over the next two quarters.",
        "choke_point_targeted": "SUEZ",
        "threat_severity": 5,
        "event_type": "MILITARY_CONFLICT",
        "tactical_directive": "Resume Suez bookings for container cargo with schedule buffers; keep tankers on the Cape route until Bab-el-Mandeb risk falls; recheck canal rebates and insurance each sailing.",
        "verified_citation": {"url": "https://mykn.kuehne-nagel.com/news/article/suez-recovery-gathers-pace-but-sector-split",
                              "publisher": "Kuehne+Nagel", "published_timestamp": "2026-10-10T00:00:00+00:00"},
    },
    {
        "headline": "Panama Canal caps daily transits at 32 after a dry rainy season",
        "summary": "Watershed rainfall from May to August came in 34% below average, so the Panama Canal Authority cut daily transits to 34 from 4 September and 32 from 15 September, and lowered the maximum draft to 47.5 ft. El Nino could extend water-saving measures into 2027.",
        "choke_point_targeted": "PANAMA",
        "threat_severity": 6,
        "event_type": "DROUGHT_WEATHER",
        "tactical_directive": "Book transit slots early or bid in slot auctions; load to the 47.5 ft draft limit; keep a Suez or US land-bridge contingency for Asia-US East Coast cargo.",
        "verified_citation": {"url": "https://www.bairdmaritime.com/amp/story/shipping/panama-canal-caps-daily-transits-as-el-ni%C3%B1o-threatens-water-levels",
                              "publisher": "Baird Maritime", "published_timestamp": "2026-08-28T00:00:00+00:00"},
    },
    {
        "headline": "Strait of Malacca flowing normally; sea robbery continues in the Singapore Strait",
        "summary": "Traffic through Malacca and Singapore is close to normal and is absorbing extra Asia-Europe volumes. Boardings and armed robbery against ships underway continue in the eastbound lane of the Singapore Strait, mostly at night and usually without violence.",
        "choke_point_targeted": "MALACCA",
        "threat_severity": 4,
        "event_type": "PIRACY",
        "tactical_directive": "Keep transits; raise ship security level in the Singapore Strait, reinforce night anti-boarding watch, and report incidents to ReCAAP ISC.",
        "verified_citation": {"url": "https://www.recaap.org/alerts", "publisher": "ReCAAP ISC", "published_timestamp": None},
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
        e["analyst_note"] = f"Baseline briefing curated from public reporting as of {BASELINE_AS_OF}."
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


STRATEGIC_NOTES = {
    "SUEZ": "Links the Mediterranean to the Red Sea and is the shortest Asia-Europe sea route; in normal conditions it carries about 12% of global trade.",
    "BAB_EL_MANDEB": "Southern gateway to the Red Sea, about 30 km wide at its narrowest. Every Suez transit to or from Asia passes through it.",
    "HORMUZ": "The only sea exit from the Persian Gulf. Before 2026 about 20 million barrels of oil a day (a fifth of world consumption) and a fifth of world LNG passed through it.",
    "PANAMA": "Connects the Atlantic and Pacific. Lock capacity and permitted draft depend on Gatun Lake water levels, so droughts cut daily transits.",
    "MALACCA": "Shortest sea route between the Indian Ocean and the Pacific and the world's largest oil chokepoint, including the Singapore Strait.",
}

# Reference facts per choke point (EIA World Oil Transit Chokepoints, 2023 data; ACP; SCA).
CHOKE_PROFILES = {
    "HORMUZ": {"oil_mbd": 20.9, "width_km": 33, "normal_transits_day": 85, "carries": "Crude, LNG, refined products",
               "share": "~20% of world oil consumption"},
    "MALACCA": {"oil_mbd": 23.7, "width_km": 2.7, "normal_transits_day": 230, "carries": "Crude, containers, LNG, dry bulk",
                "share": "~29% of seaborne oil"},
    "SUEZ": {"oil_mbd": 8.8, "width_km": 0.3, "normal_transits_day": 70, "carries": "Containers, crude and products, LNG",
             "share": "~12% of global trade"},
    "BAB_EL_MANDEB": {"oil_mbd": 8.6, "width_km": 29, "normal_transits_day": 70, "carries": "Containers, tankers to and from Suez",
                      "share": "All Suez-Asia traffic"},
    "PANAMA": {"oil_mbd": 2.1, "width_km": 0.05, "normal_transits_day": 36, "carries": "Containers, LPG, grain, vehicles",
               "share": "~5% of world seaborne trade"},
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
